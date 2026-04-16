"""backbones/animatediff_refiner.py
====================================
AnimateDiff backbone refiner — wraps an AnimateDiff pipeline and uses
:class:`~adapters.animatediff_adapter.AnimateDiffAdapter` to inject scene
prior guides at generation time.

Refinement strategy: SDEdit (img2img)
    1. Encode coarse_rgb → latent z0
    2. Add noise up to timestep t_start (controlled by ``strength``)
    3. Denoise from t_start → 0 while injecting guides via hooks

This preserves coarse structure from the scene prior while allowing the
diffusion model to hallucinate fine-grained texture and lighting.

Usage example::

    from diffusers import AnimateDiffPipeline, ...
    pipe = AnimateDiffPipeline.from_pretrained(...)
    adapter = AnimateDiffAdapter(in_ch=64)
    refiner = AnimateDiffRefiner(pipe, adapter, n_steps=20, strength=0.5)

    scene_features = scene_guide_encoder(scene_out)   # (B, 64, H, W)
    guides = adapter.build_guides(scene_features)
    refined = refiner.refine(coarse_rgb, "a red ball and a blue cube", guides)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from adapters.animatediff_adapter import AnimateDiffAdapter
from backbones.interface import BackboneInterface


class AnimateDiffRefiner(BackboneInterface):
    """Wraps an AnimateDiff pipeline as a scene-prior-conditioned refiner.

    Args:
        pipe:     Loaded AnimateDiff (or Stable Diffusion) pipeline.  Must
                  expose ``vae``, ``unet``, ``scheduler``, and
                  ``encode_prompt`` / ``decode_latents`` methods (standard
                  diffusers API).
        adapter:  :class:`~adapters.animatediff_adapter.AnimateDiffAdapter`
                  instance bound to this refiner.
        n_steps:  Number of denoising steps.
        strength: SDEdit noise strength ∈ (0, 1].  ``strength=1.0`` ignores
                  coarse_rgb entirely; ``strength=0.5`` starts halfway through
                  denoising (recommended for structure preservation).
        device:   Torch device string (default ``"cuda"``).
    """

    def __init__(
        self,
        pipe,
        adapter: AnimateDiffAdapter,
        n_steps: int = 20,
        strength: float = 0.5,
        device: str = "cuda",
    ) -> None:
        self.pipe     = pipe
        self.adapter  = adapter
        self.n_steps  = n_steps
        self.strength = float(strength)
        self.device   = device

    # ------------------------------------------------------------------
    # BackboneInterface implementation
    # ------------------------------------------------------------------

    def refine(
        self,
        coarse_rgb: torch.Tensor,
        prompt: str,
        guides: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Refine a coarse composite image via SDEdit with scene prior guides.

        Args:
            coarse_rgb: (B, 3, H, W) float32 in [0, 1].
            prompt:     Text prompt (shared across batch).
            guides:     Per-block guide tensors from
                        :meth:`~adapters.animatediff_adapter.AnimateDiffAdapter.build_guides`.

        Returns:
            (B, 3, H, W) float32 refined image in [0, 1].
        """
        pipe    = self.pipe
        adapter = self.adapter
        device  = self.device
        dtype   = next(pipe.unet.parameters()).dtype

        # ---- 1. Store guides in adapter so hooks can retrieve them ----
        adapter.set_guides(guides)
        adapter.register_hooks(pipe.unet)

        try:
            refined = self._sdedit(coarse_rgb, prompt, dtype)
        finally:
            adapter.remove_hooks()
            adapter.clear_guides()

        return refined

    def get_adapter(self) -> AnimateDiffAdapter:
        """Return the adapter bound to this refiner."""
        return self.adapter

    # ------------------------------------------------------------------
    # SDEdit internals
    # ------------------------------------------------------------------

    def _sdedit(
        self,
        coarse_rgb: torch.Tensor,
        prompt: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """SDEdit: encode coarse_rgb, corrupt, denoise with guide hooks.

        Returns (B, 3, H, W) float32 in [0, 1].
        """
        pipe   = self.pipe
        vae    = pipe.vae
        sched  = pipe.scheduler
        device = self.device

        B = coarse_rgb.shape[0]

        # ---- Encode coarse image to latents ----
        # Normalise to [-1, 1] for VAE
        img_norm = coarse_rgb.to(device=device, dtype=dtype) * 2.0 - 1.0
        with torch.no_grad():
            latent_dist = vae.encode(img_norm)
            z0 = latent_dist.latent_dist.sample() * vae.config.scaling_factor

        # ---- Set up noise schedule ----
        sched.set_timesteps(self.n_steps, device=device)
        timesteps = sched.timesteps

        # Determine start timestep from strength
        t_start_idx = max(
            0, int(len(timesteps) * (1.0 - self.strength))
        )
        timesteps = timesteps[t_start_idx:]

        # ---- Add noise up to t_start ----
        noise = torch.randn_like(z0)
        if len(timesteps) > 0:
            t_noise = timesteps[0]
            z_t = sched.add_noise(z0, noise, t_noise.unsqueeze(0))
        else:
            z_t = z0

        # ---- Encode prompt ----
        with torch.no_grad():
            prompt_embeds, neg_prompt_embeds = self._encode_prompt(prompt, B)

        # Classifier-free guidance: duplicate latents
        do_cfg    = True
        cfg_scale = 7.5
        latent    = z_t

        # ---- Denoise ----
        for t in timesteps:
            t_batch = t.unsqueeze(0).expand(B)
            if do_cfg:
                latent_input = torch.cat([latent, latent])
                t_input      = torch.cat([t_batch, t_batch])
                emb_input    = torch.cat([neg_prompt_embeds, prompt_embeds])
            else:
                latent_input = latent
                t_input      = t_batch
                emb_input    = prompt_embeds

            with torch.no_grad():
                noise_pred = pipe.unet(
                    latent_input,
                    t_input,
                    encoder_hidden_states=emb_input,
                ).sample

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latent = sched.step(noise_pred, t, latent).prev_sample

        # ---- Decode latents ----
        with torch.no_grad():
            image = vae.decode(latent / vae.config.scaling_factor).sample

        # [-1, 1] → [0, 1]
        image = (image.clamp(-1.0, 1.0) + 1.0) * 0.5
        return image.float()

    def _encode_prompt(
        self, prompt: str, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompt and empty negative prompt.

        Returns:
            Tuple of (prompt_embeds, neg_prompt_embeds) each (B, seq, dim).
        """
        pipe   = self.pipe
        device = self.device

        # Use diffusers pipeline helper if available
        if hasattr(pipe, "encode_prompt"):
            prompt_embeds, neg_embeds, *_ = pipe.encode_prompt(
                prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt="",
            )
        else:
            # Minimal fallback: use tokenizer + text_encoder directly
            tokenizer    = pipe.tokenizer
            text_encoder = pipe.text_encoder

            def _encode(text: str) -> torch.Tensor:
                tokens = tokenizer(
                    text,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)
                return text_encoder(tokens)[0]

            prompt_embeds = _encode(prompt)
            neg_embeds    = _encode("")

        # Expand to batch_size
        if prompt_embeds.shape[0] == 1 and batch_size > 1:
            prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
            neg_embeds    = neg_embeds.expand(batch_size, -1, -1)

        return prompt_embeds, neg_embeds
