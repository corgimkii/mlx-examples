# Copyright © 2026 Apple Inc.

"""
Wan2.2 unified text-to-video / image-to-video pipeline (TI2V-5B).

A single DiT model handles both modes. For I2V, the input image is encoded
into the VAE latent space and re-injected at every denoising step as a
masked overlay:

    latent = (1 - mask) * z_clean + mask * latent

where mask is 0 on the conditioning frame(s) and 1 on all noise frames.
The per-token timestep tensor is set to 0 on conditioning tokens so the DiT
treats them as already-denoised — no separate CLIP encoder or channel-concat
image input is needed.
"""

import logging
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

from .sampler import FlowUniPCMultistepScheduler
from .utils import load_dit, load_t5, load_t5_tokenizer, load_vae


class WanPipeline:
    def __init__(
        self,
        name: str = "ti2v-5B",
        dtype: mx.Dtype = mx.bfloat16,
        checkpoint: Optional[str] = None,
    ):
        self.dtype = dtype
        self.name = name
        self.vae_stride = (4, 16, 16)
        self.z_dim = 48
        self.patch_size = (1, 2, 2)
        self._null_context = None

        self.flow = load_dit(name, checkpoint=checkpoint)
        self.vae = load_vae(name)
        self.t5 = load_t5(name)
        self.t5_tokenizer = load_t5_tokenizer(name)
        self.sampler = FlowUniPCMultistepScheduler()

    def ensure_models_are_loaded(self):
        mx.eval(
            self.flow.parameters(),
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
    ) -> Tuple[mx.array, mx.array]:
        """Encode the input image into VAE latent space and build a temporal mask.

        Returns:
            z_clean: [T', H', W', z_dim] — VAE latent of the conditioning video
                (first frame = input image, rest = zeros)
            mask:    [T', H', W', 1]    — 0 on the conditioning frame, 1 elsewhere
        """
        from PIL import Image

        W, H = size

        img = Image.open(image_path).convert("RGB")
        iw, ih = img.size
        scale = max(W / iw, H / ih)
        rw, rh = round(iw * scale), round(ih * scale)
        img = img.resize((rw, rh), Image.BICUBIC)
        left = (rw - W) // 2
        top = (rh - H) // 2
        img = img.crop((left, top, left + W, top + H))

        img_arr = np.array(img).astype(np.float32) / 255.0
        img_arr = (img_arr - 0.5) / 0.5
        img_tensor = mx.array(img_arr)  # [H, W, 3]

        zeros = mx.zeros((frame_num - 1, H, W, 3))
        video = mx.concatenate([img_tensor[None], zeros], axis=0)

        z_clean = self.vae.encode(video).astype(self.dtype)  # [T', H', W', 48]

        T_lat, H_lat, W_lat, _ = z_clean.shape
        mask_first = mx.zeros((1, H_lat, W_lat, 1), dtype=self.dtype)
        mask_rest = mx.ones((T_lat - 1, H_lat, W_lat, 1), dtype=self.dtype)
        mask = mx.concatenate([mask_first, mask_rest], axis=0)

        return z_clean, mask

    def _per_token_timestep(
        self,
        t_scalar: mx.array,
        target_shape: Tuple[int, int, int, int],
        mask: Optional[mx.array],
    ) -> mx.array:
        """Build the per-token timestep tensor consumed by the DiT.

        For T2V (mask=None) every patchified token shares `t_scalar`.
        For I2V, conditioning tokens (mask==0) get timestep 0; noise tokens
        get `t_scalar`. The token count is Fp*Hp*Wp where (Fp,Hp,Wp) are
        the latent dims divided by `self.patch_size`.

        Returns: [1, Fp*Hp*Wp] float32
        """
        Tl, Hl, Wl, _ = target_shape
        pt, ph, pw = self.patch_size
        Fp, Hp, Wp = Tl // pt, Hl // ph, Wl // pw
        if mask is None:
            return mx.broadcast_to(t_scalar.reshape(1, 1), (1, Fp * Hp * Wp)).astype(
                mx.float32
            )
        # Average the spatial-strided mask across each patch to derive a
        # per-token gate (0 = conditioning, 1 = noise).
        m = mask[::pt, ::ph, ::pw, 0]  # [Fp, Hp, Wp]
        per_token = (m.reshape(1, Fp * Hp * Wp) * t_scalar.reshape(1, 1)).astype(
            mx.float32
        )
        return per_token

    def generate_latents(
        self,
        text: str,
        image_path: Optional[str] = None,
        negative_prompt: str = "",
        size: Tuple[int, int] = (1280, 704),
        frame_num: int = 121,
        num_steps: int = 50,
        guidance: float = 5.0,
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
        target_shape = (
            (frame_num - 1) // self.vae_stride[0] + 1,
            H // self.vae_stride[1],
            W // self.vae_stride[2],
            self.z_dim,
        )

        # Text conditioning
        context = self._encode_text(text)
        context_null = (
            self._encode_text(negative_prompt)
            if negative_prompt
            else self._encode_null()
        )

        # Image conditioning (I2V only)
        z_clean = None
        mask = None
        if image_path is not None:
            z_clean, mask = self._prepare_image_conditioning(
                image_path, size, frame_num
            )

        # Initial noise; for I2V the conditioning frame is re-injected at every step.
        x_t = mx.random.normal(target_shape).astype(self.dtype)
        if mask is not None:
            x_t = (1.0 - mask) * z_clean + mask * x_t

        yield (x_t, context, context_null, z_clean, mask)

        sampler = self.sampler
        sampler.set_timesteps(num_steps, shift=shift)

        flow = mx.compile(self.flow.__call__, inputs=[self.flow.state])

        for step_idx, t in enumerate(sampler.timesteps):
            t_scalar = t.reshape(1).astype(mx.float32)
            t_tokens = self._per_token_timestep(t_scalar, target_shape, mask)

            noise_cond = flow(x_t, t=t_tokens, context=context)
            if guidance > 1.0:
                noise_uncond = flow(x_t, t=t_tokens, context=context_null)
                noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
            else:
                noise_pred = noise_cond

            x_t = sampler.step(noise_pred, t, x_t)
            if mask is not None:
                x_t = (1.0 - mask) * z_clean + mask * x_t

            mx.async_eval(x_t)
            yield x_t

            if verbose:
                logger.info(f"Step {step_idx}/{num_steps}")

    def decode(self, latents: mx.array) -> mx.array:
        return self.vae.decode(latents)
