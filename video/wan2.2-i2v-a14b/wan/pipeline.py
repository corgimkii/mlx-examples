# Copyright © 2026 Apple Inc.

"""
Wan 2.2 I2V-A14B image-to-video pipeline.

The A14B variant is a Mixture-of-Time-Step-Experts: two 14B-parameter DiTs
share architecture and are selected by the timestep boundary inside the
denoising loop. Only one expert is in memory at a time — `flow_high` runs
when ``t >= boundary``, `flow_low` runs below.

Image conditioning is supplied through `first_frame`, a 20-channel
tensor that channel-concats with the latent input before the DiT's patch
embedding: 4 channels of temporal mask (1 on the first frame, 0 elsewhere,
expanded by the VAE temporal stride) plus 16 channels of VAE latent (the
input image as frame 0, zeros for the rest). No CLIP encoder is involved.
"""

import logging
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

from .sampler import FlowUniPCMultistepScheduler
from .utils import configs, load_dit, load_t5, load_t5_tokenizer, load_vae


class WanPipeline:
    def __init__(
        self,
        name: str = "i2v-A14B",
        dtype: mx.Dtype = mx.bfloat16,
        checkpoint_high: Optional[str] = None,
        checkpoint_low: Optional[str] = None,
    ):
        self.dtype = dtype
        self.name = name
        self.vae_stride = (4, 8, 8)
        self.z_dim = 16
        self._null_context = None

        spec = configs[name]
        self.boundary = spec.boundary
        # Two DiT experts swapped by the diffusion timestep. They never
        # need to be resident in memory simultaneously, but this naive
        # implementation keeps both around — see README for memory notes.
        self.flow_high = load_dit(name, expert="high", checkpoint=checkpoint_high)
        self.flow_low = load_dit(name, expert="low", checkpoint=checkpoint_low)
        self.vae = load_vae(name)
        self.t5 = load_t5(name)
        self.t5_tokenizer = load_t5_tokenizer(name)
        self.sampler = FlowUniPCMultistepScheduler()

    def ensure_models_are_loaded(self):
        mx.eval(
            self.flow_high.parameters(),
            self.flow_low.parameters(),
            self.vae.parameters(),
            self.t5.parameters(),
        )

    def _encode_text(self, text: str) -> mx.array:
        """Encode text prompt with T5. Returns [512, 4096]."""
        tokens = self.t5_tokenizer(text)
        ids = tokens["input_ids"]
        mask = tokens["attention_mask"]
        embeddings = self.t5(ids, mask=mask)
        seq_len = int(mask.sum().item())
        context = embeddings[0, :seq_len, :]
        if seq_len < 512:
            padding = mx.zeros((512 - seq_len, context.shape[-1]))
            context = mx.concatenate([context, padding], axis=0)
        return context

    def _encode_null(self) -> mx.array:
        if self._null_context is None:
            self._null_context = self._encode_text("")
        return self._null_context

    def _prepare_image_conditioning(
        self, image_path: str, size: Tuple[int, int], frame_num: int
    ) -> mx.array:
        """Build the 20-channel `first_frame` conditioning tensor.

        Returns:
            y: [T', H', W', 20] = mask(4ch) + vae_latent(16ch), channels-last,
                matching the DiT's expected `in_dim=36 = 16 + 20` after
                channel-concat with the noise latent.
        """
        from PIL import Image

        W, H = size

        # Load image, resize (short side) + center crop to target resolution
        img = Image.open(image_path).convert("RGB")
        iw, ih = img.size
        scale = max(W / iw, H / ih)
        rw, rh = round(iw * scale), round(ih * scale)
        img = img.resize((rw, rh), Image.BICUBIC)
        left = (rw - W) // 2
        top = (rh - H) // 2
        img = img.crop((left, top, left + W, top + H))

        # Normalize to [-1, 1]
        img_arr = np.array(img).astype(np.float32) / 255.0
        img_arr = (img_arr - 0.5) / 0.5
        img_tensor = mx.array(img_arr)  # [H, W, 3]

        # Build video: first frame = image, rest = zeros -> [F, H, W, 3]
        zeros = mx.zeros((frame_num - 1, H, W, 3))
        video = mx.concatenate([img_tensor[None], zeros], axis=0)

        # VAE encode -> [T', H', W', 16]. The VAE decides the latent T dim
        # (typically ceil-like collapse of frame_num under the temporal
        # stride); trust its output shape instead of pre-computing T_latent
        # from `(frame_num - 1) // stride + 1`, which is only exact when
        # `(frame_num - 1) % stride == 0` and rounds differently otherwise.
        vae_latent = self.vae.encode(video)
        T_latent, H_latent, W_latent, _ = vae_latent.shape

        # Build temporal mask -> [T', H', W', 4]
        msk_first = mx.ones((1, H_latent, W_latent, 4))
        msk_rest = mx.zeros((T_latent - 1, H_latent, W_latent, 4))
        msk = mx.concatenate([msk_first, msk_rest], axis=0)

        # Concat: [T', H', W', 4+16] = [T', H', W', 20]
        y = mx.concatenate([msk, vae_latent], axis=-1)
        return y.astype(self.dtype)

    def _flow_for(self, t: mx.array):
        """Return the DiT expert selected by this timestep."""
        # The reference encodes the boundary as `0.900 * num_train_timesteps`;
        # our sampler timesteps are in the same `[0, num_train_timesteps)`
        # range so the same threshold applies directly.
        boundary_steps = self.boundary * self.sampler.num_train_timesteps
        return self.flow_high if float(t.item()) >= boundary_steps else self.flow_low

    def generate_latents(
        self,
        text: str,
        image_path: str,
        negative_prompt: str = "",
        size: Tuple[int, int] = (1280, 720),
        frame_num: int = 81,
        num_steps: int = 40,
        guidance: float = 3.5,
        shift: float = 5.0,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Generator yielding latents at each denoising step.

        First yield: conditioning tuple (for mx.eval by caller)
        Subsequent yields: latent at each denoising step
        """
        if seed is not None:
            mx.random.seed(seed)

        W, H = size

        # Text conditioning
        context = self._encode_text(text)
        context_null = (
            self._encode_text(negative_prompt)
            if negative_prompt
            else self._encode_null()
        )

        # Image conditioning (required for i2v). Do this first so `x_t`
        # inherits its latent T dim from what the VAE actually produced —
        # `(frame_num - 1) // stride + 1` disagrees with VAE output for
        # arbitrary `frame_num` (only exact when `(frame_num - 1) % stride == 0`).
        first_frame = self._prepare_image_conditioning(image_path, size, frame_num)
        T_latent, H_latent, W_latent, _ = first_frame.shape
        target_shape = (T_latent, H_latent, W_latent, self.z_dim)

        # Initial noise
        x_t = mx.random.normal(target_shape).astype(self.dtype)

        yield (x_t, context, context_null, first_frame)

        sampler = self.sampler
        sampler.set_timesteps(num_steps, shift=shift)

        for step_idx, t in enumerate(sampler.timesteps):
            t_val = t.reshape(1).astype(mx.float32)
            flow = self._flow_for(t)

            noise_cond = flow(x_t, t=t_val, context=context, first_frame=first_frame)
            if guidance > 1.0:
                noise_uncond = flow(
                    x_t, t=t_val, context=context_null, first_frame=first_frame
                )
                noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
            else:
                noise_pred = noise_cond

            x_t = sampler.step(noise_pred, t, x_t)
            mx.async_eval(x_t)
            yield x_t

            if verbose:
                logger.info(f"Step {step_idx}/{num_steps}")

    def decode(self, latents: mx.array, progress: bool = False) -> mx.array:
        return self.vae.decode(latents, progress=progress)
