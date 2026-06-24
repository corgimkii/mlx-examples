# Copyright © 2026 Apple Inc.

"""
Transformer layers for Wan 2.2 I2V-A14B DiT.

Same shape as wan2.1's `WanAttentionBlock` (RMSNorm + 3D RoPE self-attn,
text-only cross-attn, gated FFN, 6-way AdaLN modulation), but cross-attn
is text-only — Wan 2.2 I2V-A14B does not consume CLIP features, so the
`WanI2VCrossAttention` class from wan2.1 is dropped here.
"""

import math
from functools import partial
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from .rope import rope_apply


# Compiled to fuse x + y * gate into a single Metal kernel (hot path).
@partial(mx.compile, shapeless=True)
def _residual_gate(x, y, gate):
    return x + y * gate


class WanSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.o = nn.Linear(dim, dim)

        self.norm_q = nn.RMSNorm(dim, eps=eps)
        self.norm_k = nn.RMSNorm(dim, eps=eps)

    def _attend(self, x, grid_sizes):
        """Compute self-attention. Returns attn output [B, n, L, d]."""
        B, L, _ = x.shape
        n, d = self.num_heads, self.head_dim

        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = q.reshape(B, L, n, d)
        k = k.reshape(B, L, n, d)
        v = v.reshape(B, L, n, d)

        q = rope_apply(q, grid_sizes, self.head_dim)
        k = rope_apply(k, grid_sizes, self.head_dim)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)

    def __call__(self, x, grid_sizes):
        B, L, C = x.shape
        attn = self._attend(x, grid_sizes)
        return self.o(attn.transpose(0, 2, 1, 3).reshape(B, L, C))


class WanCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.o = nn.Linear(dim, dim)

        self.norm_q = nn.RMSNorm(dim, eps=eps)
        self.norm_k = nn.RMSNorm(dim, eps=eps)

    def _attend(self, x, context):
        """Compute text cross-attention. Returns (q, attn_out) both [B, n, L, d]."""
        B = x.shape[0]
        L1, L2 = x.shape[1], context.shape[1]
        n, d = self.num_heads, self.head_dim

        q = self.norm_q(self.q(x))
        kv = self.kv(context)
        k, v = mx.split(kv, 2, axis=-1)
        k = self.norm_k(k)

        q = q.reshape(B, L1, n, d).transpose(0, 2, 1, 3)
        k = k.reshape(B, L2, n, d).transpose(0, 2, 1, 3)
        v = v.reshape(B, L2, n, d).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5)

        return q, out

    def __call__(self, x, context):
        _, attn = self._attend(x, context)
        B, _, L1, _ = attn.shape
        x = attn.transpose(0, 2, 1, 3).reshape(B, L1, self.dim)
        return self.o(x)


class WanAttentionBlock(nn.Module):
    """
    Transformer block with self-attn, cross-attn, and FFN.

    Uses fused norm+modulate via mx.fast.layer_norm where the modulation
    scale/shift are passed as weight/bias. Requires sanitize to bake 1+
    into modulation scale positions.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        cross_attn_type: str = "t2v",  # accepted for API parity, ignored
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps

        if cross_attn_norm:
            self.norm3 = nn.LayerNorm(dim, eps=eps)
        else:
            self.norm3 = None

        self.self_attn = WanSelfAttention(dim, num_heads, eps)
        self.cross_attn = WanCrossAttention(dim, num_heads, eps)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approx="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # Modulation: [shift, scale, gate] x 2 for self-attn (indices 0-2) and FFN (indices 3-5)
        self.modulation = mx.zeros((1, 6, dim))

    def __call__(
        self,
        x: mx.array,
        e: mx.array,
        grid_sizes: list,
        context: mx.array,
    ) -> mx.array:
        e = self.modulation + e

        # Self-attention: fused LayerNorm where e[:,1]=scale (weight), e[:,0]=shift (bias), e[:,2]=gate
        y = self.self_attn(
            mx.fast.layer_norm(x, e[0, 1], e[0, 0], self.eps),
            grid_sizes,
        )
        x = _residual_gate(x, y, e[:, 2])

        # Cross-attention
        if self.norm3 is not None:
            x_normed = self.norm3(x)
        else:
            x_normed = x
        x = x + self.cross_attn(x_normed, context)

        # FFN: fused LayerNorm where e[:,4]=scale, e[:,3]=shift, e[:,5]=gate
        y = self.ffn(mx.fast.layer_norm(x, e[0, 4], e[0, 3], self.eps))
        x = _residual_gate(x, y, e[:, 5])

        return x


class Head(nn.Module):
    """Output head with fused norm+modulate and nn.Linear."""

    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: Tuple[int, int, int],
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        out_features = math.prod(patch_size) * out_dim
        self.linear = nn.Linear(dim, out_features)
        # Modulation: [shift, scale] for output head norm
        self.modulation = mx.zeros((1, 2, dim))

    def __call__(self, x: mx.array, e: mx.array) -> mx.array:
        e = self.modulation + e[:, None, :]
        x = mx.fast.layer_norm(x, e[0, 1], e[0, 0], self.eps)
        return self.linear(x)
