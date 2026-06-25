Wan 2.2 I2V-A14B
================

Wan 2.2 I2V-A14B (image-to-video, 27B MoE / 14B active) implementation in
MLX. Weights are pulled from the [Hugging Face
Hub](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B).

A14B is a Mixture-of-Time-Step-Experts: two 14B-parameter DiTs share the
same architecture and are selected by the diffusion timestep. The
`high_noise` expert runs while `t >= boundary` and the `low_noise` expert
takes over below — the schedule splits roughly at step 90% of the noise
range (`boundary=0.900`).

| Model | Task | HF Repo | Approach |
|-------|------|---------|----------|
| A14B | I2V | [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) | MoE: 2 × 14B DiT + Wan 2.1 VAE + umT5-XXL |

Compared to the Wan 2.1 I2V-14B example next door:

* **MoE** — two DiT checkpoints (`high_noise_model/`, `low_noise_model/`)
  selected by timestep boundary.
* **No CLIP** — image conditioning is purely the VAE latent of the first
  frame (channel-concatenated with the noise input). No separate vision
  encoder.
* **`in_dim=36`** (= 16 VAE latent + 4 temporal mask + 16 first-frame VAE
  latent), same as Wan 2.1 I2V-14B but without the CLIP token stream.

Installation
------------

```shell
pip install -r requirements.txt
```

Saving videos requires [ffmpeg](https://ffmpeg.org/) on your PATH.

Usage
-----

```shell
python img2video.py 'Astronaut riding a horse' \
    --image ./inputs/astronaut-on-a-horse.png --quantize --output out_i2v.mp4
```

For all options, `python img2video.py --help`.

### Quantization

A 14B expert is ~28 GB in bf16 and ~14 GB at 8-bit. The pipeline only ever
holds one expert resident (see [Memory](#memory) below), so `--quantize 8`
brings the live DiT weight to ~14 GB — comfortable on a 64 GB Mac. Use
`--quantize 4` for tighter budgets at the cost of some output quality:

```shell
python img2video.py 'Astronaut riding a horse' --image ./inputs/astronaut-on-a-horse.png \
    --quantize 8 --output out_i2v.mp4
```

Quantization is applied to each expert at the moment it is loaded, so the
un-quantized weights never sit in memory.

### Memory

Only one DiT expert is resident at a time. The denoising loop starts on
the high-noise expert; when the schedule crosses `boundary` (≈ step 5 out
of 50 at the default `boundary=0.900`) the pipeline frees that expert and
loads the low-noise expert in its place — see `_load_expert` /
`_flow_for` in [`wan/pipeline.py`](./wan/pipeline.py). Reading the
low-noise checkpoint from disk on the swap takes a few seconds, which is
negligible against the overall denoising wall-clock.

To get additional memory savings at the expense of a bit of speed, pass
`--no-cache` to set `mx.set_cache_limit(0)`. See the
[documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html)
for details.

### Custom DiT Weights

```shell
python img2video.py 'Astronaut riding a horse' --image ./inputs/astronaut-on-a-horse.png \
    --checkpoint-high ./my_high_noise.safetensors \
    --checkpoint-low  ./my_low_noise.safetensors \
    --output out.mp4
```

# References

1. [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
