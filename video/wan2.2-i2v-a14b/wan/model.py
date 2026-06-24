# Copyright © 2026 Apple Inc.

"""
Wan 2.2 I2V-A14B bidirectional DiT (Diffusion Transformer).

This is one of the two experts in the A14B MoE — a 14B-parameter DiT used
for either the high-noise or low-noise half of the diffusion schedule. The
two experts share architecture and are selected by the timestep boundary
inside the pipeline; only one expert is in memory at a time.

Differences from Wan 2.1 I2V-14B (mlx-examples/video/wan2.1):
  * No CLIP image embedder. The reference Wan 2.2 i2v-A14B does not
    consume CLIP features — image conditioning is purely the VAE latent
    of the first frame (channel-concatenated with the latent input).
  * Cross-attention is text-only (`WanCrossAttention`, no image keys).
  * `in_dim=36` (= 16 VAE latent + 4 mask + 16 first-frame VAE latent).
"""

import math
import re
from functools import partial
from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
from einops import rearrange

from .layers import Head, WanAttentionBlock


@partial(mx.compile, shapeless=True)
def sinusoidal_embedding_1d(dim: int, position: mx.array) -> mx.array:
    assert dim % 2 == 0
    half = dim // 2
    dtype = position.dtype
    position = position.astype(mx.float32)
    sinusoid = (
        position[:, None]
        * mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)[None, :]
    )
    return mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=1).astype(dtype)


class WanModel(nn.Module):
    def __init__(
        self,
        model_type: str = "i2v",
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 36,
        dim: int = 5120,
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 40,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert model_type == "i2v"
        self.patch_size = patch_size
        self.dim = dim
        self.freq_dim = freq_dim

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size, bias=True
        )

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approx="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

        self.blocks = [
            WanAttentionBlock(
                dim,
                ffn_dim,
                num_heads,
                cross_attn_norm,
                eps,
                cross_attn_type="t2v",  # text-only — no CLIP path
            )
            for _ in range(num_layers)
        ]

        self.head = Head(dim, out_dim, patch_size, eps)

    def compute_time_embedding(self, t: mx.array) -> Tuple[mx.array, mx.array]:
        """Single-timestep embedding.

        t: shape [1] (the diffusion schedule timestep, broadcast across all
            tokens). Returns:
            t_emb: [1, dim] (pre-projection, used by head)
            e0:    [1, 6*dim] (projected, consumed by each block)
        """
        e = sinusoidal_embedding_1d(self.freq_dim, t)
        t_emb = self.time_embedding(e)
        e0 = self.time_projection(t_emb)
        return t_emb, e0

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        context: mx.array,
        first_frame: mx.array,
    ) -> mx.array:
        """
        Forward pass for I2V-A14B.

        Args:
            x:           Input latent [F, H, W, 16] (channels-last)
            t:           Timestep [1]
            context:     Text embedding [L, C_text]
            first_frame: Image conditioning [F, H, W, 20] (= 4 mask channels
                + 16 first-frame VAE latent). Channel-concatenated with `x`
                before the patch embedding so the conv sees in_dim=36.

        Returns:
            Output latent [F, H, W, 16]
        """
        # Channel-concat image conditioning before patchify
        x = mx.concatenate([x, first_frame], axis=-1)

        # Patchify: [F, H, W, C] -> [1, F, H, W, C] -> conv3d -> [1, Fp, Hp, Wp, dim]
        x = self.patch_embedding(x[None])
        _, Fp, Hp, Wp, _ = x.shape
        grid_sizes = [[Fp, Hp, Wp]]
        x = x.reshape(1, Fp * Hp * Wp, self.dim)

        # Embed text context: [L, C_text] -> [1, text_len, dim]
        context = self.text_embedding(context[None])

        # Time embedding
        t_emb, e = self.compute_time_embedding(t)
        e = e.reshape(1, 6, self.dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, e, grid_sizes, context)

        # Output head
        x = self.head(x, t_emb)

        # Unpatchify: [1, seq_len, patch_features] -> [F, H, W, C]
        pt, ph, pw = self.patch_size
        return rearrange(
            x[0],
            "(Fp Hp Wp) (pt ph pw c) -> (Fp pt) (Hp ph) (Wp pw) c",
            Fp=Fp,
            Hp=Hp,
            Wp=Wp,
            pt=pt,
            ph=ph,
            pw=pw,
        )

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Remap PyTorch checkpoint keys to MLX model format.

        The A14B i2v reference has no `img_emb.*` keys (CLIP not used).
        """
        remapped = {}
        for key, value in weights.items():
            new_key = key

            if "weight_scale" in new_key:
                continue

            if new_key.startswith("model."):
                new_key = new_key[6:]

            # PyTorch Conv3d [O,I,kT,kH,kW] -> MLX Conv3d [O,kT,kH,kW,I]
            if (
                "patch_embedding" in new_key
                and "weight" in new_key
                and len(value.shape) == 5
            ):
                value = mx.transpose(value, (0, 2, 3, 4, 1))

            new_key = new_key.replace("ffn.0.", "ffn.layers.0.")
            new_key = new_key.replace("ffn.2.", "ffn.layers.2.")
            new_key = new_key.replace("text_embedding.0.", "text_embedding.layers.0.")
            new_key = new_key.replace("text_embedding.2.", "text_embedding.layers.2.")
            new_key = new_key.replace("time_embedding.0.", "time_embedding.layers.0.")
            new_key = new_key.replace("time_embedding.2.", "time_embedding.layers.2.")
            new_key = new_key.replace("time_projection.1.", "time_projection.layers.1.")

            new_key = new_key.replace("head.head.", "head.linear.")

            remapped[new_key] = value

        remapped = WanModel._merge_qkv_weights(remapped)

        # Modulation vectors are [shift, scale, gate, ...]. The DiT block
        # applies x * (1 + scale) + shift, so we bake the "1 +" into the
        # stored scale weights here.
        for key in list(remapped.keys()):
            if key.endswith(".modulation"):
                v = remapped[key]
                if v.shape[1] == 6:  # block modulation [1, 6, dim]
                    remapped[key] = v + mx.array([0, 1, 0, 0, 1, 0])[:, None]
                elif v.shape[1] == 2:  # head modulation [1, 2, dim]
                    remapped[key] = v + mx.array([0, 1])[:, None]

        return remapped

    @staticmethod
    def _merge_qkv_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Merge separate q/k/v weights into qkv (self-attn) and kv (cross-attn)."""
        merged = {}
        consumed = set()

        for key in weights:
            m = re.match(r"(blocks\.\d+\.self_attn)\.(q)\.(weight|bias)$", key)
            if m:
                prefix, _, param = m.groups()
                q_key = f"{prefix}.q.{param}"
                k_key = f"{prefix}.k.{param}"
                v_key = f"{prefix}.v.{param}"
                if q_key in weights and k_key in weights and v_key in weights:
                    merged[f"{prefix}.qkv.{param}"] = mx.concatenate(
                        [weights[q_key], weights[k_key], weights[v_key]], axis=0
                    )
                    consumed.update([q_key, k_key, v_key])
                continue

            m = re.match(r"(blocks\.\d+\.cross_attn)\.(k)\.(weight|bias)$", key)
            if m:
                prefix, _, param = m.groups()
                k_key = f"{prefix}.k.{param}"
                v_key = f"{prefix}.v.{param}"
                if k_key in weights and v_key in weights:
                    merged[f"{prefix}.kv.{param}"] = mx.concatenate(
                        [weights[k_key], weights[v_key]], axis=0
                    )
                    consumed.update([k_key, v_key])
                continue

        for key, value in weights.items():
            if key not in consumed:
                merged[key] = value

        return merged
