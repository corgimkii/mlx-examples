# Copyright © 2026 Apple Inc.

"""
LoRA support for Wan 2.2 I2V-A14B.

Designed for inference-time fusion: LoRA weights are baked directly into
the base ``nn.Linear`` weight (``W += scale * B @ A``) before quantization.
This keeps the Metal kernel path identical to the un-LoRA'd model — there
is no runtime `B @ A` overhead per step — which matters at 40 layers × 50
denoising steps.

The reference HF format (e.g. ``lightx2v/Wan2.2-Lightning``) ships LoRA
adapters with q/k/v as three separate adapters per attention layer; our
model fuses q/k/v into a single ``self_attn.qkv`` (and ``cross_attn.kv``)
matrix, so ``sanitize_lora_weights`` joins them back together by indexing
the appropriate row range of the fused weight at fuse time.
"""

import re
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def fuse_lora(
    linear: nn.Linear,
    lora_down: mx.array,
    lora_up: mx.array,
    scale: float = 1.0,
    row_slice: Optional[Tuple[int, int]] = None,
) -> None:
    """In-place fuse `W += scale * (lora_up @ lora_down)` into a Linear.

    lora_down: shape [r, in_dim]
    lora_up:   shape [out_dim_chunk, r]
    row_slice: optional (start, end) on the output dimension. Used when
        the HF adapter is for q/k/v separately but our Linear is the
        fused qkv (or kv) — we only update the rows corresponding to
        this chunk.
    """
    weight = linear.weight  # [out, in]
    dtype = weight.dtype
    delta = (scale * (lora_up @ lora_down)).astype(dtype)

    if row_slice is None:
        linear.weight = weight + delta
    else:
        start, end = row_slice
        assert delta.shape == (end - start, weight.shape[1]), (
            f"LoRA delta shape {delta.shape} does not fit row slice "
            f"[{start}:{end}] of weight {weight.shape}"
        )
        new_weight = mx.concatenate(
            [weight[:start], weight[start:end] + delta, weight[end:]], axis=0
        )
        linear.weight = new_weight


def sanitize_lora_weights(
    hf_weights: Dict[str, mx.array],
) -> Dict[str, Dict]:
    """Group raw HF LoRA tensors by the target MLX Linear they should fuse into.

    Returns a dict keyed by MLX Linear path (e.g. ``blocks.0.self_attn.qkv``)
    whose value is a list of fuse jobs::

        {
            "down": [r, in_dim] tensor,
            "up":   [out_dim_chunk, r] tensor,
            "alpha": int (LoRA alpha, default = rank),
            "rank": int,
            "row_slice": (start, end) or None,
        }

    For fused qkv/kv targets the list has multiple entries (one per chunk).

    Drops:
      * ``__metadata__`` key if present.
      * ``.alpha`` scalars are read but not emitted as separate entries.
    """
    # First pass: collect raw adapters keyed by their HF path stem
    # (everything up to .lora_down / .lora_up / .alpha).
    by_stem: Dict[str, Dict[str, mx.array]] = {}
    for key, value in hf_weights.items():
        if key == "__metadata__":
            continue
        # Strip optional "diffusion_model." prefix.
        k = key
        if k.startswith("diffusion_model."):
            k = k[len("diffusion_model."):]

        m = re.match(r"^(.*)\.(lora_down|lora_up|alpha)(\.weight)?$", k)
        if m is None:
            # Some lightx2v variants (Distill 1022) ship extra ``diff`` /
            # ``diff_b`` / ``diff_m`` tensors that encode a raw bias /
            # modulation delta rather than a low-rank decomposition. We
            # skip them — the bulk of the adapter signal lives in the
            # rank-64 lora_down/up pairs, and the mix recipe ignores
            # these auxiliary deltas in the same way.
            if re.search(r"\.(diff|diff_b|diff_m)$", k):
                continue
            raise ValueError(f"Unrecognized LoRA key: {key}")
        stem, kind, _ = m.groups()
        slot = by_stem.setdefault(stem, {})
        if kind == "lora_down":
            slot["down"] = value
        elif kind == "lora_up":
            slot["up"] = value
        elif kind == "alpha":
            slot["alpha"] = int(value.item()) if hasattr(value, "item") else int(value)

    # Second pass: route each stem to the target MLX Linear (with optional
    # row_slice for fused qkv/kv).
    jobs: Dict[str, list] = {}
    for stem, slot in by_stem.items():
        if "down" not in slot or "up" not in slot:
            raise ValueError(f"LoRA stem {stem!r} missing down/up weights")
        down = slot["down"]
        up = slot["up"]
        rank = down.shape[0]
        alpha = slot.get("alpha", rank)

        target, row_slice = _map_to_target(stem, up.shape[0])
        jobs.setdefault(target, []).append(
            {
                "down": down,
                "up": up,
                "rank": rank,
                "alpha": alpha,
                "row_slice": row_slice,
            }
        )
    return jobs


def _map_to_target(stem: str, out_dim: int) -> Tuple[str, Optional[Tuple[int, int]]]:
    """Map an HF LoRA stem (e.g. ``blocks.0.self_attn.q``) to the MLX
    Linear path it fuses into, plus an optional row slice on the output
    dim for q/k/v split adapters."""

    # Self-attention q/k/v -> fused qkv; o stays as o.
    m = re.match(r"^(blocks\.\d+\.self_attn)\.([qkvo])$", stem)
    if m:
        prefix, sub = m.groups()
        if sub == "o":
            return f"{prefix}.o", None
        # qkv layout in our nn.Linear weight is [3*dim, dim], with rows
        # [0:dim]=q, [dim:2*dim]=k, [2*dim:3*dim]=v.
        i = {"q": 0, "k": 1, "v": 2}[sub]
        return f"{prefix}.qkv", (i * out_dim, (i + 1) * out_dim)

    # Cross-attention: q and o stay; k/v fuse into kv.
    m = re.match(r"^(blocks\.\d+\.cross_attn)\.([qkvo])$", stem)
    if m:
        prefix, sub = m.groups()
        if sub == "q" or sub == "o":
            return f"{prefix}.{sub}", None
        i = {"k": 0, "v": 1}[sub]
        return f"{prefix}.kv", (i * out_dim, (i + 1) * out_dim)

    # FFN: ffn.0 -> ffn.layers.0, ffn.2 -> ffn.layers.2.
    m = re.match(r"^(blocks\.\d+)\.ffn\.([02])$", stem)
    if m:
        prefix, idx = m.groups()
        return f"{prefix}.ffn.layers.{idx}", None

    raise ValueError(f"Cannot map LoRA stem to target: {stem!r}")


def apply_lora(
    model: nn.Module,
    hf_weights: Dict[str, mx.array],
    strength: float = 1.0,
) -> None:
    """Load HF LoRA weights and fuse them into a Wan DiT in place.

    Call this *before* `nn.quantize` so the LoRA delta gets quantized
    along with the base weight. ``strength`` multiplies the default
    ``alpha/rank`` scale and can be used to stack a second LoRA at a
    different weight (e.g. mix recipes that pair a strong high-noise
    adapter with a weaker low-noise one).
    """
    jobs = sanitize_lora_weights(hf_weights)

    for target_path, fuse_list in jobs.items():
        linear = _resolve_module(model, target_path)
        if not isinstance(linear, nn.Linear):
            raise ValueError(
                f"LoRA target {target_path} resolved to {type(linear).__name__}, "
                "expected nn.Linear"
            )
        for job in fuse_list:
            scale = strength * job["alpha"] / job["rank"]
            fuse_lora(
                linear,
                lora_down=job["down"],
                lora_up=job["up"],
                scale=scale,
                row_slice=job["row_slice"],
            )


def _resolve_module(model: nn.Module, path: str) -> nn.Module:
    """Walk ``path`` like ``blocks.0.self_attn.qkv`` to the nested module."""
    obj = model
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj
