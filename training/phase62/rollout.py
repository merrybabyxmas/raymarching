"""
Phase 62 — Rollout Runner
===========================

Generate composite video using pipe() API directly.
This avoids the pink artifact caused by manual CFG loop + backbone extractors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import imageio.v2 as iio2
from PIL import Image


class Phase62RolloutRunner:
    """
    Generate composite rollout using pipe() high-level API.

    Previous manual CFG loop caused pink artifacts because backbone extractors
    corrupted the uncond batch's cross-attention with entity tokens.
    Using pipe() API lets diffusers handle CFG correctly.
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

        if toks_e0 is not None and toks_e1 is not None:
            backbone_mgr.set_entity_tokens(toks_e0, toks_e1)

        system.clear_guides()

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

        comp_frames = [(f * 255).astype(np.uint8) for f in out.frames[0]]

        result: dict = {"frames": comp_frames}

        # Overlay: projected visible class on frames (if entity masks provided)
        if entity_masks is not None:
            overlay_frames = []
            for fi, frame in enumerate(comp_frames):
                overlay = frame.astype(np.float32)
                S = entity_masks.shape[-1]
                H_mask = int(S ** 0.5)
                T_mask = entity_masks.shape[0]
                fi_m = min(fi, T_mask - 1)
                m0 = entity_masks[fi_m, 0].reshape(H_mask, H_mask)
                m1 = entity_masks[fi_m, 1].reshape(H_mask, H_mask)
                m0_up = np.array(Image.fromarray(
                    (m0 * 255).astype(np.uint8)).resize(
                    (width, height), Image.NEAREST), dtype=np.float32) / 255.0
                m1_up = np.array(Image.fromarray(
                    (m1 * 255).astype(np.uint8)).resize(
                    (width, height), Image.NEAREST), dtype=np.float32) / 255.0
                overlay[:, :, 0] += m0_up * 80.0  # red for e0
                overlay[:, :, 2] += m1_up * 80.0  # blue for e1
                overlay_frames.append(overlay.clip(0, 255).astype(np.uint8))
            result["overlay_frames"] = overlay_frames

        # Compute probe MSE if GT available
        if gt_frames is not None:
            gt = np.stack(gt_frames[:n_frames], axis=0).astype(np.float32) / 255.0
            pred = np.stack(comp_frames[:n_frames], axis=0).astype(np.float32) / 255.0
            T_comp = min(gt.shape[0], pred.shape[0])
            probe_mse = float(((gt[:T_comp] - pred[:T_comp]) ** 2).mean())
            probe_score = 1.0 / (1.0 + probe_mse * 10)
            result["probe_mse"] = probe_mse
            result["probe_score"] = probe_score

        return result
