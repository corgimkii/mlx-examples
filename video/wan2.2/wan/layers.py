# Copyright © 2026 Apple Inc.

"""
Transformer layers for Wan2.2 DiT.

Norms, attention, blocks, and output head. Uses bidirectional (non-causal)
attention. Modulation (shift/scale/gate) is applied per-token to support
the TI2V mask-injection paradigm, where image-conditioned frames carry a
different timestep than noise frames.
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


# Per-token modulation requires broadcasting scale/shift over the token axis,
# which mx.fast.layer_norm's affine parameters do not support (they are 1-D).
# We pre-normalize with mx.fast.layer_norm (which fuses mean/var) and do the
# affine separately. The whole thing is shapelessly compiled so the mul/add
# fuses into a single Metal kernel alongside the layer norm. Scale is already
# stored with the "1 +" baked in (see WanModel.sanitize).
@partial(mx.compile, shapeless=True)
def _modulated_layer_norm(x, scale, shift, eps):
    """LayerNorm with per-token affine: out = norm(x) * scale + shift.

    x:     [B, L, D]
    scale: [B, S, D] (S=1 or L)
    shift: [B, S, D]
    """
    x = mx.fast.layer_norm(x, None, None, eps)
    return x * scale + shift


class WanAttentionBlock(nn.Module):
    """
    Transformer block with self-attn, cross-attn, and FFN.

    Modulation `e` is per-token, shape [B, S, 6, D] where S is 1 (broadcast
    timestep) or matches the patchified token count (TI2V per-token timestep).
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
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

        # Modulation: [shift, scale, gate] x 2 for self-attn (0-2) and FFN (3-5)
        self.modulation = mx.zeros((1, 6, dim))

    def __call__(
        self,
        x: mx.array,
        e: mx.array,
        grid_sizes: list,
        context: mx.array,
    ) -> mx.array:
        # e: [B, S, 6, D] + modulation [1, 6, D] -> [B, S, 6, D]
        e = self.modulation[:, None, :, :] + e

        # Self-attention with per-token shift/scale/gate
        y = self.self_attn(
            _modulated_layer_norm(x, e[:, :, 1], e[:, :, 0], self.eps),
            grid_sizes,
        )
        x = _residual_gate(x, y, e[:, :, 2])

        # Cross-attention (no modulation)
        if self.norm3 is not None:
            x_normed = self.norm3(x)
        else:
            x_normed = x
        x = x + self.cross_attn(x_normed, context)

        # FFN with per-token shift/scale/gate
        y = self.ffn(_modulated_layer_norm(x, e[:, :, 4], e[:, :, 3], self.eps))
        x = _residual_gate(x, y, e[:, :, 5])

        return x


class Head(nn.Module):
    """Output head with per-token modulation and nn.Linear."""

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
        # e: [B, S, D]; modulation [1, 2, D] -> [B, S, 2, D]
        e = self.modulation[:, None, :, :] + e[:, :, None, :]
        x = _modulated_layer_norm(x, e[:, :, 1], e[:, :, 0], self.eps)
        return self.linear(x)
