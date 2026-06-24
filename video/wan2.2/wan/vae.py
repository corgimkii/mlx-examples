# Copyright © 2026 Apple Inc.

"""
Wan2.2 VAE encoder and decoder.

Compared to Wan 2.1:
  * z_dim is 48 (vs 16).
  * Spatial stride is 16 (vs 8) via an outer 2x2 patchify/unpatchify.
  * Encoder/Decoder stages wrap their residual blocks in a residual
    container (`Down_ResidualBlock` / `Up_ResidualBlock`) with a
    parameter-free `AvgDown3D` / `DupUp3D` shortcut.
  * `Resample` keeps channel count constant (Conv2d(dim, dim, 3, padding=1)).
  * Decoder hidden dim is 256 (vs encoder dim 160 — see WanVAE).

All tensors are channels-last (NTHWC) as required by MLX.
"""

import re
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

CACHE_T = 2


def _patchify(x: mx.array, patch_size: int) -> mx.array:
    """[B, T, H, W, C] -> [B, T, H/p, W/p, C*p*p].

    Match the PyTorch reference's `rearrange("b c f (h q) (w r) -> b (c r q) f h w")`:
    the flat patch axis runs (c outer, then p_w, then p_h innermost).
    """
    if patch_size == 1:
        return x
    B, T, H, W, C = x.shape
    p = patch_size
    # Reshape to expose p_h and p_w as separate axes.
    x = x.reshape(B, T, H // p, p, W // p, p, C)
    # Axes after reshape: [0=B, 1=T, 2=H', 3=p_h, 4=W', 5=p_w, 6=C].
    # Target layout for final reshape: (B, T, H', W', C, p_w, p_h).
    x = x.transpose(0, 1, 2, 4, 6, 5, 3)
    return x.reshape(B, T, H // p, W // p, C * p * p)


def _unpatchify(x: mx.array, patch_size: int) -> mx.array:
    """Inverse of `_patchify`."""
    if patch_size == 1:
        return x
    B, T, H, W, Cpp = x.shape
    p = patch_size
    C = Cpp // (p * p)
    # Reverse layout: flat axis was (C, p_w, p_h) outer→inner.
    x = x.reshape(B, T, H, W, C, p, p)
    # Axes: [0=B, 1=T, 2=H, 3=W, 4=C, 5=p_w, 6=p_h]
    # Target for reshape: (B, T, H, p_h, W, p_w, C).
    x = x.transpose(0, 1, 2, 6, 3, 5, 4)
    return x.reshape(B, T, H * p, W * p, C)


def _create_cache_entry(x: mx.array, existing_cache: Optional[mx.array]) -> mx.array:
    """Build temporal cache from the last CACHE_T frames of x."""
    t = x.shape[1]
    if t >= CACHE_T:
        return x[:, -CACHE_T:]
    cache_x = x[:, -t:]
    if existing_cache is not None:
        old_frames = existing_cache[:, -(CACHE_T - t) :]
        return mx.concatenate([old_frames, cache_x], axis=1)
    zeros = mx.zeros((x.shape[0], CACHE_T - t, *x.shape[2:]), dtype=x.dtype)
    return mx.concatenate([zeros, cache_x], axis=1)


def _normalize_tuple(value, n):
    if isinstance(value, int):
        return (value,) * n
    return tuple(value)


class CausalConv3d(nn.Module):
    """Conv3d with causal (time-left) padding and an optional `cache_x` of
    prior frames concatenated in time before the conv. Same semantics as
    Wan 2.1 mlx-examples implementation."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _normalize_tuple(kernel_size, 3)
        self.stride = _normalize_tuple(stride, 3)
        self.padding = _normalize_tuple(padding, 3)
        self._temporal_pad = self.padding[0] * 2
        self._spatial_pad_h = self.padding[1]
        self._spatial_pad_w = self.padding[2]

        scale = (
            1.0
            / (
                in_channels
                * self.kernel_size[0]
                * self.kernel_size[1]
                * self.kernel_size[2]
            )
            ** 0.5
        )
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, *self.kernel_size, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))

    def __call__(self, x, cache_x=None):
        temporal_pad = self._temporal_pad
        if cache_x is not None and self._temporal_pad > 0:
            x = mx.concatenate([cache_x, x], axis=1)
            temporal_pad = max(0, self._temporal_pad - cache_x.shape[1])

        if temporal_pad > 0:
            x = mx.pad(x, [(0, 0), (temporal_pad, 0), (0, 0), (0, 0), (0, 0)])

        if self._spatial_pad_h > 0 or self._spatial_pad_w > 0:
            x = mx.pad(
                x,
                [
                    (0, 0),
                    (0, 0),
                    (self._spatial_pad_h, self._spatial_pad_h),
                    (self._spatial_pad_w, self._spatial_pad_w),
                    (0, 0),
                ],
            )

        y = mx.conv3d(x, self.weight, stride=self.stride, padding=0)
        if "bias" in self:
            y = y + self.bias
        return y


class Resample(nn.Module):
    """Spatial up/downsample with optional causal temporal conv. Channel
    count is preserved (unlike Wan 2.1 which halves on upsample).

    Caches:
      * `upsample3d`: stores the input pre-time-conv. The first call sees
        an empty (placeholder) cache and skips temporal upsampling — this
        matches the PyTorch reference's "Rep" sentinel, which we model as
        cache=None with `first_call=True` returned alongside.
      * `downsample3d`: stores the last frame post-spatial-conv for use as
        the time_conv's left-side input on the next chunk.
    """

    def __init__(self, dim: int, mode: str):
        assert mode in ("upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode in ("upsample2d", "upsample3d"):
            self.upsample = nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest")
            self.conv = nn.Conv2d(
                dim, dim, kernel_size=3, stride=1, padding=1, bias=True
            )
            if mode == "upsample3d":
                self.time_conv = CausalConv3d(
                    dim, dim * 2, (3, 1, 1), padding=(1, 0, 0)
                )
        else:  # downsample2d / downsample3d
            self.conv = nn.Conv2d(
                dim, dim, kernel_size=3, stride=2, padding=0, bias=True
            )
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(
                    dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
                )

    def __call__(self, x, cache=None):
        b, t, h, w, c = x.shape
        new_cache = None

        if self.mode == "upsample3d":
            # Reference uses a "Rep" sentinel on chunk 0 to skip time upsampling.
            # We model the same three states by inspecting the cache:
            #   * None        — first call ever; skip time_conv, emit shape-0 sentinel
            #   * shape[1]==0 — second call ("Rep" path in ref); call time_conv with
            #                   no left context so CausalConv3d zero-pads internally
            #   * otherwise   — steady state; pass cache as the left context
            if cache is None:
                new_cache = mx.zeros((b, 0, h, w, c), dtype=x.dtype)
            elif cache.shape[1] == 0:
                cache_in = x
                x = self.time_conv(x, None)
                new_cache = _create_cache_entry(cache_in, None)
                x = x.reshape(b, t, h, w, 2, c)
                x = x.transpose(0, 1, 4, 2, 3, 5)
                x = x.reshape(b, t * 2, h, w, c)
            else:
                cache_in = x
                x = self.time_conv(x, cache)
                new_cache = _create_cache_entry(cache_in, cache)
                x = x.reshape(b, t, h, w, 2, c)
                x = x.transpose(0, 1, 4, 2, 3, 5)
                x = x.reshape(b, t * 2, h, w, c)

        t_out = x.shape[1]
        c_out = x.shape[4]
        x = x.reshape(b * t_out, x.shape[2], x.shape[3], c_out)

        if self.mode in ("upsample2d", "upsample3d"):
            x = self.upsample(x)
            x = self.conv(x)
        else:  # downsample2d / downsample3d
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
            x = self.conv(x)

        x = x.reshape(b, t_out, x.shape[1], x.shape[2], x.shape[3])

        if self.mode == "downsample3d":
            if cache is None:
                new_cache = x[:, -1:]
            else:
                x_with_cache = mx.concatenate([cache, x], axis=1)
                new_cache = x[:, -1:]
                x = self.time_conv(x_with_cache, None)

        return x, new_cache


class ResidualBlock(nn.Module):
    """RMSNorm + CausalConv3d x2 with skip; identical structure to Wan 2.1."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = nn.RMSNorm(in_dim, eps=1e-12)
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = nn.RMSNorm(out_dim, eps=1e-12)
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, padding=1)
        if in_dim != out_dim:
            self.shortcut = CausalConv3d(in_dim, out_dim, 1)
        else:
            self.shortcut = None

    def __call__(self, x, cache1, cache2):
        h = self.shortcut(x) if self.shortcut is not None else x

        residual = self.norm1(x)
        residual = nn.silu(residual)
        cache_input = residual
        residual = self.conv1(residual, cache1)
        new_cache1 = _create_cache_entry(cache_input, cache1)

        residual = self.norm2(residual)
        residual = nn.silu(residual)
        cache_input = residual
        residual = self.conv2(residual, cache2)
        new_cache2 = _create_cache_entry(cache_input, cache2)

        return h + residual, new_cache1, new_cache2


class AttentionBlock(nn.Module):
    """Frame-wise self-attention. Same structure as Wan 2.1."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.RMSNorm(dim, eps=1e-12)
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x):
        identity = x
        b, t, h, w, c = x.shape
        x = x.reshape(b * t, h, w, c)
        x = self.norm(x)
        qkv = self.to_qkv(x).reshape(b * t, h * w, 3, c)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.reshape(b * t, 1, h * w, c)
        k = k.reshape(b * t, 1, h * w, c)
        v = v.reshape(b * t, 1, h * w, c)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=c**-0.5)
        attn = attn.squeeze(1).reshape(b * t, h, w, c)
        out = self.proj(attn).reshape(b, t, h, w, c)
        return out + identity


class AvgDown3D(nn.Module):
    """Parameter-free spatio-temporal average-downsample shortcut."""

    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def __call__(self, x):
        # x: [B, T, H, W, C] (channels-last)
        B, T, H, W, C = x.shape
        pad_t = (self.factor_t - T % self.factor_t) % self.factor_t
        if pad_t > 0:
            x = mx.pad(x, [(0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)])
            T += pad_t
        x = x.reshape(
            B,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
            C,
        )
        # PyTorch `permute(0,1,3,5,7,2,4,6)` on (B,C,T',t_f,H',h_f,W',w_f)
        # produces a flat channel order (C, t_f, h_f, w_f) outer→inner. Match
        # that here: axes are [0=B,1=T',2=t_f,3=H',4=h_f,5=W',6=w_f,7=C], so
        # transpose to (B, T', H', W', C, t_f, h_f, w_f) puts C outermost.
        x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
        x = x.reshape(
            B,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
            self.out_channels,
            self.group_size,
        )
        return x.mean(axis=-1)


class DupUp3D(nn.Module):
    """Parameter-free spatio-temporal duplicate-upsample shortcut."""

    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def __call__(self, x, first_chunk: bool = False):
        # x: [B, T, H, W, C_in]
        B, T, H, W, _ = x.shape
        # repeat along channel — emulates PyTorch repeat_interleave(repeats, dim=1)
        x = mx.repeat(x, self.repeats, axis=-1)  # [B, T, H, W, C_in * repeats]
        x = x.reshape(
            B,
            T,
            H,
            W,
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
        )
        # Bring temporal/spatial factors next to their respective axes
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.reshape(
            B,
            T * self.factor_t,
            H * self.factor_s,
            W * self.factor_s,
            self.out_channels,
        )
        if first_chunk and self.factor_t > 1:
            x = x[:, self.factor_t - 1 :]
        return x


class DownStage(nn.Module):
    """Wan 2.2 encoder stage = AvgDown3D shortcut + residual blocks (+ optional Resample)."""

    def __init__(self, in_dim, out_dim, num_blocks, temperal_downsample, down_flag):
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        blocks = []
        d_in = in_dim
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(d_in, out_dim))
            d_in = out_dim
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            blocks.append(Resample(out_dim, mode=mode))
        self.downsamples = blocks

    def __call__(self, x, feat_cache, cache_idx):
        x_in = x
        for layer in self.downsamples:
            if isinstance(layer, ResidualBlock):
                x, c1, c2 = layer(x, feat_cache[cache_idx], feat_cache[cache_idx + 1])
                feat_cache[cache_idx] = c1
                feat_cache[cache_idx + 1] = c2
                cache_idx += 2
            elif isinstance(layer, Resample):
                x, c = layer(x, feat_cache[cache_idx])
                feat_cache[cache_idx] = c
                cache_idx += 1
        return x + self.avg_shortcut(x_in), cache_idx


class UpStage(nn.Module):
    """Wan 2.2 decoder stage = (optional DupUp3D shortcut) + residual blocks (+ optional Resample)."""

    def __init__(self, in_dim, out_dim, num_blocks, temperal_upsample, up_flag):
        super().__init__()
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        blocks = []
        d_in = in_dim
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(d_in, out_dim))
            d_in = out_dim
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            blocks.append(Resample(out_dim, mode=mode))
        self.upsamples = blocks

    def __call__(self, x, feat_cache, cache_idx, first_chunk):
        x_in = x
        for layer in self.upsamples:
            if isinstance(layer, ResidualBlock):
                x, c1, c2 = layer(x, feat_cache[cache_idx], feat_cache[cache_idx + 1])
                feat_cache[cache_idx] = c1
                feat_cache[cache_idx + 1] = c2
                cache_idx += 2
            elif isinstance(layer, Resample):
                x, c = layer(x, feat_cache[cache_idx])
                feat_cache[cache_idx] = c
                cache_idx += 1
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_in, first_chunk)
        return x, cache_idx


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 160,
        z_dim: int = 96,  # = z_dim*2 for mu+log_var
        dim_mult: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        temperal_downsample: Optional[List[bool]] = None,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if temperal_downsample is None:
            temperal_downsample = [False, True, True]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.temperal_downsample = temperal_downsample

        dims = [dim * u for u in [1] + dim_mult]
        # Input is patchified: 3 channels * 2*2 patch -> 12
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        self.downsamples = []
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            t_flag = temperal_downsample[i] if i < len(temperal_downsample) else False
            self.downsamples.append(
                DownStage(
                    in_dim=in_d,
                    out_dim=out_d,
                    num_blocks=num_res_blocks,
                    temperal_downsample=t_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )

        self.middle_res1 = ResidualBlock(dims[-1], dims[-1])
        self.middle_attn = AttentionBlock(dims[-1])
        self.middle_res2 = ResidualBlock(dims[-1], dims[-1])

        self.head_norm = nn.RMSNorm(dims[-1], eps=1e-12)
        self.head_conv = CausalConv3d(dims[-1], z_dim, 3, padding=1)

        # Cache slots: conv1 + each stage (2*num_blocks per block + 1 per Resample) + middle_res*2 + head_conv
        n = 1
        for i in range(len(dim_mult)):
            n += 2 * num_res_blocks
            if i != len(dim_mult) - 1:
                n += 1  # Resample
        n += 2 + 2 + 1
        self.num_cache_slots = n

    def __call__(self, x, feat_cache):
        cache_idx = 0
        cache_in = x
        x = self.conv1(x, feat_cache[cache_idx])
        feat_cache[cache_idx] = _create_cache_entry(cache_in, feat_cache[cache_idx])
        cache_idx += 1

        for stage in self.downsamples:
            x, cache_idx = stage(x, feat_cache, cache_idx)

        x, c1, c2 = self.middle_res1(
            x, feat_cache[cache_idx], feat_cache[cache_idx + 1]
        )
        feat_cache[cache_idx] = c1
        feat_cache[cache_idx + 1] = c2
        cache_idx += 2

        x = self.middle_attn(x)

        x, c1, c2 = self.middle_res2(
            x, feat_cache[cache_idx], feat_cache[cache_idx + 1]
        )
        feat_cache[cache_idx] = c1
        feat_cache[cache_idx + 1] = c2
        cache_idx += 2

        x = self.head_norm(x)
        x = nn.silu(x)
        cache_in = x
        x = self.head_conv(x, feat_cache[cache_idx])
        feat_cache[cache_idx] = _create_cache_entry(cache_in, feat_cache[cache_idx])
        cache_idx += 1

        return x, feat_cache


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        z_dim: int = 48,
        dim_mult: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        temperal_upsample: Optional[List[bool]] = None,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if temperal_upsample is None:
            temperal_upsample = [True, True, False]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.temperal_upsample = temperal_upsample

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        self.middle_res1 = ResidualBlock(dims[0], dims[0])
        self.middle_attn = AttentionBlock(dims[0])
        self.middle_res2 = ResidualBlock(dims[0], dims[0])

        self.upsamples = []
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            t_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            self.upsamples.append(
                UpStage(
                    in_dim=in_d,
                    out_dim=out_d,
                    num_blocks=num_res_blocks + 1,
                    temperal_upsample=t_flag,
                    up_flag=i != len(dim_mult) - 1,
                )
            )

        self.head_norm = nn.RMSNorm(dims[-1], eps=1e-12)
        # output channels = 3 * patch_size**2 = 12
        self.head_conv = CausalConv3d(dims[-1], 12, 3, padding=1)

        n = 1 + 2 + 2  # conv1 + middle_res1 + middle_res2
        for i in range(len(dim_mult)):
            n += 2 * (num_res_blocks + 1)
            if i != len(dim_mult) - 1:
                n += 1  # Resample
        n += 1  # head_conv
        self.num_cache_slots = n

    def __call__(self, x, feat_cache, first_chunk: bool = False):
        cache_idx = 0
        cache_in = x
        x = self.conv1(x, feat_cache[cache_idx])
        feat_cache[cache_idx] = _create_cache_entry(cache_in, feat_cache[cache_idx])
        cache_idx += 1

        x, c1, c2 = self.middle_res1(
            x, feat_cache[cache_idx], feat_cache[cache_idx + 1]
        )
        feat_cache[cache_idx] = c1
        feat_cache[cache_idx + 1] = c2
        cache_idx += 2

        x = self.middle_attn(x)

        x, c1, c2 = self.middle_res2(
            x, feat_cache[cache_idx], feat_cache[cache_idx + 1]
        )
        feat_cache[cache_idx] = c1
        feat_cache[cache_idx + 1] = c2
        cache_idx += 2

        for stage in self.upsamples:
            x, cache_idx = stage(x, feat_cache, cache_idx, first_chunk)

        x = self.head_norm(x)
        x = nn.silu(x)
        cache_in = x
        x = self.head_conv(x, feat_cache[cache_idx])
        feat_cache[cache_idx] = _create_cache_entry(cache_in, feat_cache[cache_idx])
        cache_idx += 1

        return x, feat_cache


# Latent-space mean/std for Wan2.2 VAE (z_dim=48). Lifted verbatim from
# Wan-Video/Wan2.2 reference (modules/vae2_2.py:Wan2_2_VAE).
_LATENT_MEAN = [
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
]
_LATENT_STD = [
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.0600,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
]


class WanVAE(nn.Module):
    """
    High-level Wan 2.2 VAE.

    encode:  [F, H, W, 3]  -> [F', H/16, W/16, 48]
    decode:  [F, H, W, 48] -> [F*4-3, H*16, W*16, 3]   (frame-by-frame loop)
    """

    PATCH_SIZE = 2

    def __init__(self):
        super().__init__()
        self.z_dim = 48
        self.encoder = Encoder3d(dim=160, z_dim=self.z_dim * 2)
        self.conv1 = CausalConv3d(self.z_dim * 2, self.z_dim * 2, 1)
        self.conv2 = CausalConv3d(self.z_dim, self.z_dim, 1)
        self.decoder = Decoder3d(dim=256, z_dim=self.z_dim)

        self.mean = mx.array(_LATENT_MEAN)
        self.std = mx.array(_LATENT_STD)

        # NOTE: Unlike Wan 2.1's VAE, we do *not* wrap the encoder/decoder in
        # mx.compile. The upsample3d Resample branches on the cache tensor's
        # time dimension (Rep sentinel = shape (B, 0, H, W, C) vs steady state
        # with shape (B, CACHE_T, H, W, C)), and mx.compile traces a single
        # shape only. The cache list is mutated in place by each chunk so the
        # branching is resolved by the eager call.

    def encode(self, x: mx.array, progress: bool = False) -> mx.array:
        """Encode a video tensor [F, H, W, 3] (channels-last, [-1, 1]).

        progress: if True, show a tqdm bar over input chunks. The chunked
            loop is silent otherwise and can take ~10s of seconds at high
            resolutions, so callers running interactively typically want
            this on.
        """
        x = x[None]  # [1, F, H, W, 3]
        x = _patchify(x, self.PATCH_SIZE)  # [1, F, H/2, W/2, 12]

        num_frames = x.shape[1]
        # First chunk is 1 frame (causal init), subsequent chunks are 4 frames
        num_chunks = 1 + (num_frames - 1 + 3) // 4 if num_frames > 0 else 0

        feat_cache = [None] * self.encoder.num_cache_slots
        outputs = []
        i = 0
        chunk_idx = 0
        chunk_iter = range(num_chunks)
        if progress:
            from tqdm import tqdm

            chunk_iter = tqdm(chunk_iter, desc="VAE encode", unit="chunk")
        for _ in chunk_iter:
            if chunk_idx == 0:
                chunk = x[:, i : i + 1]
                i += 1
            else:
                chunk = x[:, i : i + 4]
                i += 4
            out_chunk, feat_cache = self.encoder(chunk, feat_cache)
            mx.eval(out_chunk)
            outputs.append(out_chunk)
            chunk_idx += 1

        out = mx.concatenate(outputs, axis=1)
        out = self.conv1(out)
        mu = out[..., : self.z_dim]
        mu = (mu - self.mean) / self.std
        return mu[0]

    def decode(self, z: mx.array, progress: bool = False) -> mx.array:
        """Decode a latent [F, H, W, 48] back to a video tensor [-1, 1].

        progress: see `encode` — the frame-by-frame loop can take a minute
            or more at high resolutions, so verbose runs benefit from a
            tqdm bar here.
        """
        z = z[None]
        z = z * self.std + self.mean
        x = self.conv2(z)

        num_frames = x.shape[1]
        feat_cache = [None] * self.decoder.num_cache_slots
        outputs = []
        frame_iter = range(num_frames)
        if progress:
            from tqdm import tqdm

            frame_iter = tqdm(frame_iter, desc="VAE decode", unit="frame")
        for i in frame_iter:
            frame = x[:, i : i + 1]
            out_frame, feat_cache = self.decoder(
                frame, feat_cache, first_chunk=(i == 0)
            )
            mx.eval(out_frame)
            outputs.append(out_frame)

        out = mx.concatenate(outputs, axis=1)
        out = _unpatchify(out, self.PATCH_SIZE)
        out = mx.clip(out, -1.0, 1.0)
        return out[0]

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Remap PyTorch Wan 2.2 VAE keys to MLX layout.

        PyTorch key layout (selected examples):
          encoder.conv1.weight
          encoder.downsamples.0.downsamples.0.residual.{0,2,3,6}.{gamma,weight,bias}
          encoder.downsamples.0.downsamples.2.resample.1.weight       # Resample Conv2d
          encoder.downsamples.0.downsamples.2.time_conv.weight        # Resample temporal
          encoder.middle.{0,1,2}.* -> middle_res1 / middle_attn / middle_res2
          encoder.head.{0,2}.*    -> head_norm / head_conv
        Decoder mirrors with `upsamples` and an extra ResidualBlock per stage.
        `avg_shortcut` modules carry no parameters and are not present in the
        state dict.
        """
        remapped: Dict[str, mx.array] = {}
        for key, value in weights.items():
            new_key = key

            # Conv weight transposes — PyTorch (O,I,*) -> MLX (O,*,I)
            if "weight" in new_key:
                if value.ndim == 5:
                    value = mx.transpose(value, (0, 2, 3, 4, 1))
                elif value.ndim == 4:
                    value = mx.transpose(value, (0, 2, 3, 1))

            new_key = new_key.replace(".gamma", ".weight")

            # Encoder / Decoder middle and head blocks
            new_key = new_key.replace("encoder.middle.0.", "encoder.middle_res1.")
            new_key = new_key.replace("encoder.middle.1.", "encoder.middle_attn.")
            new_key = new_key.replace("encoder.middle.2.", "encoder.middle_res2.")
            new_key = new_key.replace("encoder.head.0.", "encoder.head_norm.")
            new_key = new_key.replace("encoder.head.2.", "encoder.head_conv.")
            new_key = new_key.replace("decoder.middle.0.", "decoder.middle_res1.")
            new_key = new_key.replace("decoder.middle.1.", "decoder.middle_attn.")
            new_key = new_key.replace("decoder.middle.2.", "decoder.middle_res2.")
            new_key = new_key.replace("decoder.head.0.", "decoder.head_norm.")
            new_key = new_key.replace("decoder.head.2.", "decoder.head_conv.")

            # Down/Up stage inner sequence
            #   encoder.downsamples.{s}.downsamples.{i}.X -> encoder.downsamples.{s}.downsamples.{i}.X (kept)
            #   decoder.upsamples.{s}.upsamples.{i}.X     -> decoder.upsamples.{s}.upsamples.{i}.X     (kept)
            # ResidualBlock decomposition (Sequential within `residual`)
            new_key = re.sub(r"\.residual\.0\.", ".norm1.", new_key)
            new_key = re.sub(r"\.residual\.2\.", ".conv1.", new_key)
            new_key = re.sub(r"\.residual\.3\.", ".norm2.", new_key)
            new_key = re.sub(r"\.residual\.6\.", ".conv2.", new_key)

            # Resample Conv2d sits inside `resample.1` (after Upsample or ZeroPad2d at index 0)
            new_key = re.sub(r"\.resample\.1\.", ".conv.", new_key)

            # Squeeze norm weight to 1-D for nn.RMSNorm
            if "norm" in new_key and "weight" in new_key:
                if value.ndim > 1:
                    value = mx.squeeze(value)

            # Squeeze 1x1 conv weights for nn.Linear (to_qkv / proj)
            if ("to_qkv" in new_key or "proj" in new_key) and "weight" in new_key:
                if value.ndim == 4 and value.shape[1] == 1 and value.shape[2] == 1:
                    value = value.reshape(value.shape[0], value.shape[3])

            remapped[new_key] = value
        return remapped
