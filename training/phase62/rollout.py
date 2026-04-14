"""
Phase 62 — Rollout Runner (Guide-Injected)
=============================================

Two-phase rollout:
  1. Feature extraction pass: single UNet forward (no CFG) to get F_g, F_0, F_1
  2. Volume → projection → guide assembly → set guides
  3. Generation pass: pipe() with bypass=True (no slot extraction for CFG safety)
     but guide hooks ACTIVE (inject learned entity features into UNet blocks)

This produces composite frames that reflect the learned volume structure,
not just vanilla pipeline output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as iio2
from PIL import Image


def _encode_text(pipe, text: str, device: str) -> torch.Tensor:
    tok = pipe.tokenizer(
        text, return_tensors="pt", padding="max_length",
        max_length=pipe.tokenizer.model_max_length, truncation=True,
    ).to(device)
    with torch.no_grad():
        enc = pipe.text_encoder(**tok).last_hidden_state.half()
    return enc


class Phase62RolloutRunner:
    """
    Generate guide-injected composite rollout.

    Phase 1: Extract features with a single UNet forward (entity tokens active,
             no CFG, backbone extractors fully enabled).
    Phase 2: Predict volume, project to 2D, assemble guide features.
    Phase 3: Generate video via pipe() with guide hooks injecting the learned
             entity features. Backbone extractors are bypassed during pipe()
             to prevent CFG-related pink artifacts.
    """

    def generate_rollout(
        self,
        pipe,
        system,
        backbone_mgr,
        prompt: str,
        config,
        device: str,
        toks_e0=None,
        toks_e1=None,
        entity_masks=None,
        gt_frames=None,
    ) -> dict:
        eval_cfg = config.eval if hasattr(config, 'eval') else config
        train_cfg = config.training if hasattr(config, 'training') else config
        n_steps = getattr(eval_cfg, 'n_steps', 20)
        guidance_scale = getattr(eval_cfg, 'guidance_scale', 7.5)
        eval_seed = getattr(eval_cfg, 'eval_seed', 42)
        n_frames = getattr(train_cfg, 'n_frames', 8)
        height = getattr(train_cfg, 'height', 256)
        width = getattr(train_cfg, 'width', 256)
        guide_family = getattr(config, 'guide_family', 'dual') or 'none'

        if toks_e0 is not None and toks_e1 is not None:
            backbone_mgr.set_entity_tokens(toks_e0, toks_e1)

        # ── Phase 1: Feature extraction (no CFG, full extractors) ────────
        # Single UNet forward to extract entity features for volume prediction.
        guides = {}
        vol_outputs = None

        if guide_family != 'none':
            backbone_mgr.set_bypass(False)
            backbone_mgr.reset_slot_store()
            system.clear_guides()

            enc_full = _encode_text(pipe, prompt, device)

            # Create dummy noisy latent for feature extraction
            gen_feat = torch.Generator(device=device).manual_seed(eval_seed)
            latent_shape = (1, 4, n_frames, height // 8, width // 8)
            dummy_latent = torch.randn(latent_shape, generator=gen_feat,
                                       device=device, dtype=torch.float16)
            t_mid = torch.tensor([150], device=device).long()

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = pipe.unet(dummy_latent, t_mid, encoder_hidden_states=enc_full).sample

            F_g = backbone_mgr.primary.last_Fg
            F_0 = backbone_mgr.primary.last_F0
            F_1 = backbone_mgr.primary.last_F1

            # ── Phase 2: Volume → projection → guide assembly ───────────
            if F_g is not None and F_0 is not None and F_1 is not None:
                vol_outputs = system.predict_volume(F_g, F_0, F_1)
                vol_outputs, guides = system.project_and_assemble(
                    vol_outputs, F_g, F_0, F_1)

        # ── Phase 3: Guide-injected generation ──────────────────────────
        # Set guides on injection manager (hooks will add them during pipe())
        if guides:
            system.set_guides(guides)
        else:
            system.clear_guides()

        # Bypass extractors during pipe() to prevent CFG pink artifacts.
        # Guide injection hooks are separate and remain active.
        backbone_mgr.set_bypass(True)

        gen = torch.Generator(device=device).manual_seed(eval_seed)
        with torch.no_grad():
            out = pipe(
                prompt=prompt,
                num_frames=n_frames,
                height=height,
                width=width,
                num_inference_steps=n_steps,
                guidance_scale=guidance_scale,
                generator=gen,
                output_type="np",
            )

        backbone_mgr.set_bypass(False)
        system.clear_guides()

        comp_frames = [(f * 255).astype(np.uint8) for f in out.frames[0]]
        result: dict = {"frames": comp_frames}

        # ── Overlay: PREDICTED visible map (not GT) ──────────────────────
        # Shows what the model learned to separate, not just ground truth.
        if vol_outputs is not None and vol_outputs.visible_class is not None:
            pred_overlay_frames = self._make_predicted_overlay(
                comp_frames, vol_outputs, height, width)
            result["pred_overlay_frames"] = pred_overlay_frames

        # Also save GT overlay for comparison
        if entity_masks is not None:
            gt_overlay_frames = self._make_gt_overlay(
                comp_frames, entity_masks, height, width)
            result["overlay_frames"] = gt_overlay_frames

        # Probe MSE
        if gt_frames is not None:
            gt_list = [np.array(Image.fromarray(gf).resize((width, height), Image.BILINEAR))
                       for gf in gt_frames[:n_frames]]
            gt = np.stack(gt_list).astype(np.float32) / 255.0
            pred = np.stack(comp_frames[:n_frames]).astype(np.float32) / 255.0
            T_comp = min(gt.shape[0], pred.shape[0])
            probe_mse = float(((gt[:T_comp] - pred[:T_comp]) ** 2).mean())
            result["probe_mse"] = probe_mse
            result["probe_score"] = 1.0 / (1.0 + probe_mse * 10)

        return result

    def _make_predicted_overlay(
        self, frames, vol_outputs, height, width,
    ):
        """Overlay PREDICTED visible class map on frames (red=e0, blue=e1)."""
        vc = vol_outputs.visible_class  # (B, H_vol, W_vol)
        overlay_frames = []
        for fi, frame in enumerate(frames):
            overlay = frame.astype(np.float32)
            fi_v = min(fi, vc.shape[0] - 1)
            vc_np = vc[fi_v].cpu().numpy()  # (H_vol, W_vol)
            # Upscale to frame resolution
            vc_up = np.array(Image.fromarray(vc_np.astype(np.uint8)).resize(
                (width, height), Image.NEAREST))
            overlay[:, :, 0] += (vc_up == 1).astype(np.float32) * 80.0  # red = e0
            overlay[:, :, 2] += (vc_up == 2).astype(np.float32) * 80.0  # blue = e1
            overlay_frames.append(overlay.clip(0, 255).astype(np.uint8))
        return overlay_frames

    def _make_gt_overlay(self, frames, entity_masks, height, width):
        """Overlay GT entity masks on frames (red=e0, blue=e1)."""
        overlay_frames = []
        for fi, frame in enumerate(frames):
            overlay = frame.astype(np.float32)
            S = entity_masks.shape[-1]
            H_mask = int(S ** 0.5)
            fi_m = min(fi, entity_masks.shape[0] - 1)
            m0 = entity_masks[fi_m, 0].reshape(H_mask, H_mask)
            m1 = entity_masks[fi_m, 1].reshape(H_mask, H_mask)
            m0_up = np.array(Image.fromarray(
                (m0 * 255).astype(np.uint8)).resize(
                (width, height), Image.NEAREST), dtype=np.float32) / 255.0
            m1_up = np.array(Image.fromarray(
                (m1 * 255).astype(np.uint8)).resize(
                (width, height), Image.NEAREST), dtype=np.float32) / 255.0
            overlay[:, :, 0] += m0_up * 80.0
            overlay[:, :, 2] += m1_up * 80.0
            overlay_frames.append(overlay.clip(0, 255).astype(np.uint8))
        return overlay_frames

    def save_rollout(
        self,
        result: dict,
        debug_dir,
        prefix: str = "eval",
    ) -> dict:
        """Save rollout frames as GIF + overlays (predicted + GT)."""
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        frames = result.get("frames", [])
        if frames:
            gif_path = debug_dir / f"{prefix}_composite.gif"
            iio2.mimwrite(str(gif_path), frames, fps=8, loop=0)
            paths["composite_gif"] = str(gif_path)

        # Predicted overlay (model's visible class map)
        pred_overlay = result.get("pred_overlay_frames", [])
        if pred_overlay:
            pred_gif = debug_dir / f"{prefix}_pred_overlay.gif"
            iio2.mimwrite(str(pred_gif), pred_overlay, fps=8, loop=0)
            paths["pred_overlay_gif"] = str(pred_gif)

            pred_png = debug_dir / f"{prefix}_pred_overlay.png"
            Image.fromarray(pred_overlay[0]).save(str(pred_png))
            paths["pred_overlay_png"] = str(pred_png)

        # GT overlay
        gt_overlay = result.get("overlay_frames", [])
        if gt_overlay:
            gt_gif = debug_dir / f"{prefix}_gt_overlay.gif"
            iio2.mimwrite(str(gt_gif), gt_overlay, fps=8, loop=0)
            paths["gt_overlay_gif"] = str(gt_gif)

            gt_png = debug_dir / f"{prefix}_gt_overlay.png"
            Image.fromarray(gt_overlay[0]).save(str(gt_png))
            paths["gt_overlay_png"] = str(gt_png)

        # Legacy compatibility: overlay.png
        overlay = result.get("overlay_frames", [])
        if overlay:
            overlay_png = debug_dir / f"{prefix}_overlay.png"
            Image.fromarray(overlay[0]).save(str(overlay_png))
            paths["overlay_png"] = str(overlay_png)

        return paths
