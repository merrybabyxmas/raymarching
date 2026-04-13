"""
Phase 62 — Rollout Runner
===========================

CFG-enabled composite rollout with hybrid volume update schedule.

Generates a video by running the full diffusion inference loop with
volume-guided injection. The volume is recomputed at scheduled steps
(hybrid: beginning, 1/3, 2/3 of timesteps).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import imageio.v2 as iio2
from PIL import Image

from training.phase62.evaluator import _encode_text


class Phase62RolloutRunner:
    """
    CFG-enabled composite rollout with hybrid volume update.

    Usage:
        runner = Phase62RolloutRunner()
        frames, overlay_frames = runner.generate_rollout(
            pipe, system, backbone_mgr, prompt, config, device,
            entity_masks=masks_np, toks_e0=..., toks_e1=...,
        )
    """

    def generate_rollout(
        self,
        pipe,
        system,         # Phase62System
        backbone_mgr,   # BackboneManager
        prompt: str,
        config,
        device: str,
        toks_e0: torch.Tensor,        # (n_tok,) int
        toks_e1: torch.Tensor,
        entity_masks: Optional[np.ndarray] = None,  # (T, 2, S) for overlay
        gt_frames: Optional[np.ndarray] = None,     # (T, H, W, 3) for MSE
    ) -> Dict:
        """
        Generate composite video with volume-guided diffusion.

        Args:
            pipe:         AnimateDiffPipeline
            system:       Phase62System (volume_pred + projector + assembler + injection_mgr)
            backbone_mgr: BackboneManager (has extractors with entity tokens)
            prompt:       text prompt
            config:       config namespace
            device:       'cuda' or 'cpu'
            toks_e0:      entity-0 token positions
            toks_e1:      entity-1 token positions
            entity_masks: optional GT masks for overlay visualization
            gt_frames:    optional GT frames for MSE computation

        Returns:
            dict with:
                frames:         list of (H, W, 3) uint8 numpy arrays
                overlay_frames: list of (H, W, 3) uint8 (if entity_masks provided)
                probe_mse:      float (if gt_frames provided)
                probe_score:    float (if gt_frames provided)
        """
        # Config extraction
        eval_cfg = config.eval if hasattr(config, 'eval') else config
        train_cfg = config.training if hasattr(config, 'training') else config
        n_steps = getattr(eval_cfg, 'n_steps', 20)
        guidance_scale = getattr(eval_cfg, 'guidance_scale', 7.5)
        eval_seed = getattr(eval_cfg, 'eval_seed', 42)
        n_frames = getattr(train_cfg, 'n_frames', 8)
        height = getattr(train_cfg, 'height', 256)
        width = getattr(train_cfg, 'width', 256)
        update_schedule = getattr(config, 'update_schedule', 'hybrid')

        backbone_mgr.set_entity_tokens(toks_e0, toks_e1)

        gen = torch.Generator(device=device).manual_seed(eval_seed)
        latent_shape = (1, 4, n_frames, height // 8, width // 8)
        eval_latents = torch.randn(
            latent_shape, generator=gen,
            device=device, dtype=torch.float16)
        eval_latents = eval_latents * pipe.scheduler.init_noise_sigma

        pipe.scheduler.set_timesteps(n_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        n_total = len(timesteps)

        enc_cond = _encode_text(pipe, prompt, device)
        neg_prompt = "blurry, deformed, extra limbs, watermark"
        enc_uncond = _encode_text(pipe, neg_prompt, device)

        # Hybrid schedule boundaries
        recompute_steps = set()
        if update_schedule == "fixed_once":
            recompute_steps = {0}
        elif update_schedule == "hybrid":
            recompute_steps = {0, n_total // 3, 2 * n_total // 3}
        elif update_schedule == "every_step":
            recompute_steps = set(range(n_total))

        current_guides: Dict[str, torch.Tensor] = {}
        predicted_visible_seq: List[np.ndarray] = []

        with torch.no_grad():
            for step_idx, step_t in enumerate(timesteps):
                backbone_mgr.reset_slot_store()

                # CFG: double batch
                lat2 = torch.cat([eval_latents] * 2, dim=0)  # (2, 4, T, H, W)
                lat2 = pipe.scheduler.scale_model_input(lat2, step_t)
                enc2 = torch.cat([enc_uncond, enc_cond], dim=0)  # (2, 77, 768)

                if step_idx in recompute_steps:
                    # Pass 1: extract features (no guide)
                    system.clear_guides()
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        _ = pipe.unet(
                            lat2, step_t,
                            encoder_hidden_states=enc2).sample

                    # Volume prediction (use cond batch only)
                    F_g_e = backbone_mgr.primary.last_Fg
                    F_0_e = backbone_mgr.primary.last_F0
                    F_1_e = backbone_mgr.primary.last_F1

                    if F_g_e is not None and F_0_e is not None and F_1_e is not None:
                        # Take cond half (second batch item)
                        B_e = F_g_e.shape[0]
                        half = B_e // 2
                        V_logits_e = system.predict_volume(
                            F_g_e[half:], F_0_e[half:], F_1_e[half:])
                        visible_class_e, _, _, current_guides = system.project_and_assemble(
                            V_logits_e, F_g_e[half:], F_0_e[half:], F_1_e[half:])
                        predicted_visible_seq = [
                            vc.astype(np.uint8)
                            for vc in visible_class_e.detach().cpu().numpy()
                        ]

                    # Reset for pass 2
                    backbone_mgr.reset_slot_store()

                # Pass 2 (or single pass): with guide
                system.set_guides(current_guides)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = pipe.unet(
                        lat2, step_t,
                        encoder_hidden_states=enc2).sample

                uncond_p, cond_p = pred.chunk(2, dim=0)
                noise_pred_cfg = uncond_p + guidance_scale * (cond_p - uncond_p)

                eval_latents = pipe.scheduler.step(
                    noise_pred_cfg.half(), step_t, eval_latents,
                    return_dict=False)[0]

        system.clear_guides()

        # Decode to frames
        latents_4d = eval_latents[0].permute(1, 0, 2, 3).half()  # (T, 4, H/8, W/8)
        scale_f = pipe.vae.config.scaling_factor
        comp_frames = []
        for fi in range(n_frames):
            z_in = (latents_4d[fi:fi + 1] / scale_f).half()
            with torch.no_grad():
                decoded = pipe.vae.decode(z_in).sample
            img = ((decoded.float() / 2 + 0.5).clamp(0, 1)[0]
                   .permute(1, 2, 0).cpu().numpy() * 255
                   ).astype(np.uint8)
            comp_frames.append(img)

        result: Dict = {"frames": comp_frames}

        # Overlay: predicted visible class map on generated frames.
        if predicted_visible_seq:
            overlay_frames = []
            T_ov = len(comp_frames)
            for fi in range(T_ov):
                frame = comp_frames[fi].copy()
                cls_map = predicted_visible_seq[min(fi, len(predicted_visible_seq) - 1)]
                cls_up = np.array(Image.fromarray(
                    cls_map.astype(np.uint8)).resize(
                    (width, height), Image.NEAREST), dtype=np.uint8)
                overlay = frame.astype(np.float32)
                overlay[:, :, 0] += (cls_up == 1).astype(np.float32) * 80.0
                overlay[:, :, 2] += (cls_up == 2).astype(np.float32) * 80.0
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                overlay_frames.append(overlay)
            result["overlay_frames"] = overlay_frames

        # Probe MSE
        if gt_frames is not None:
            gt_f = np.asarray(gt_frames[:n_frames], dtype=np.float32) / 255.0
            pred_f = np.asarray(comp_frames[:n_frames], dtype=np.float32) / 255.0
            T_cmp = min(len(gt_f), len(pred_f))
            if T_cmp > 0:
                mse = float(np.mean((pred_f[:T_cmp] - gt_f[:T_cmp]) ** 2))
                result["probe_mse"] = mse
                result["probe_score"] = 1.0 / (1.0 + mse)

        return result

    def save_rollout(
        self,
        result: Dict,
        output_dir: Path,
        prefix: str = "eval",
    ) -> Dict[str, Path]:
        """
        Save rollout frames as GIF and overlay PNG.

        Returns dict of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        frames = result.get("frames", [])
        if frames:
            gif_path = output_dir / f"{prefix}_composite.gif"
            iio2.mimwrite(str(gif_path), frames, fps=8, loop=0)
            paths["composite_gif"] = gif_path

        overlay_frames = result.get("overlay_frames", [])
        if overlay_frames:
            ov_png = output_dir / f"{prefix}_overlay.png"
            Image.fromarray(overlay_frames[0]).save(str(ov_png))
            paths["overlay_png"] = ov_png

            ov_gif = output_dir / f"{prefix}_overlay.gif"
            iio2.mimwrite(str(ov_gif), overlay_frames, fps=8, loop=0)
            paths["overlay_gif"] = ov_gif

        return paths
