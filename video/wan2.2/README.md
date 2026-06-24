Wan2.2 TI2V-5B
==============

Wan2.2 TI2V-5B (Text/Image-to-Video, 5B dense) implementation in MLX. A
single DiT covers both text-to-video and image-to-video — image conditioning
is injected at the latent level (no separate CLIP encoder). Weights are
pulled from the [Hugging Face Hub](https://huggingface.co/Wan-AI).

| Model | Task | HF Repo | RAM (unquantized) |
|-------|------|---------|-------------------|
| 5B | T2V / I2V | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | TBD (see [Memory](#memory) below) |

Compared to the Wan2.1 example next door, Wan2.2 TI2V-5B differs in:

* **Unified T2V / I2V** — one DiT, no CLIP. The conditioning frame is
  re-injected into the latent at every denoising step and its per-token
  timestep is held at zero.
* **VAE** — `z_dim=48` (vs 16) and an outer 2×2 spatial patchify, giving an
  effective stride of (4, 16, 16).
* **Native resolution** — 1280×704 / 704×1280, 121 frames @ 24 fps
  (≈5 seconds).

Installation
------------

```shell
pip install -r requirements.txt
```

Saving videos requires [ffmpeg](https://ffmpeg.org/) on your PATH.

Usage
-----

### Text-to-Video

```shell
python txt2video.py 'A cat playing piano' --output out.mp4
```

Adjust resolution, frame count, and sampling parameters:

```shell
python txt2video.py 'Ocean waves crashing on a rocky shore at sunset' \
    --size 704x1280 --frames 121 --steps 50 --guidance 5.0 --seed 42 \
    --output waves.mp4
```

For all options, `python txt2video.py --help`.

### Image-to-Video

```shell
python img2video.py 'Astronaut riding a horse' \
    --image ./inputs/astronaut-on-a-horse.png --output out_i2v.mp4
```

The default frame count is 121; the first frame is anchored to the input
image and the rest are denoised conditioned on it.

For all options, `python img2video.py --help`.

### Quantization

Pass `--quantize` (or `-q`) to the CLI for 8-bit (default) or `--quantize 4`
for 4-bit DiT weights:

```shell
python txt2video.py 'A cat playing piano' --quantize --output out_quantized.mp4
```

### Memory

The denoising step's peak memory is dominated by the DiT activations at
the configured (frames, size). The 5B DiT in bf16 is ~10 GB on its own;
add the T5 encoder (~10 GB bf16) during conditioning and the VAE during
decode. Use `--quantize` to cut DiT weight memory roughly in half (`-q 8`)
or quarter (`-q 4`).

To get additional memory savings at the expense of a bit of speed, pass
`--no-cache` to set `mx.set_cache_limit(0)`. See the
[documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html)
for details.

```shell
python txt2video.py 'A cat playing piano' --output out.mp4 --no-cache
```

### Custom DiT Weights

Use `--checkpoint` to load alternative DiT weights:

```shell
python txt2video.py 'A cat playing piano' \
    --checkpoint ./my_finetuned_wan22_ti2v_5b.safetensors --output out.mp4
```

# References

1. [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
