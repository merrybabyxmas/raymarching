"""
Phase 62 — Evaluator
======================

Validation evaluation: 3-pass process for each sample.
  1. UNet forward (no guide) -> extract F_g, F_0, F_1
  2. Volume prediction -> VolumeOutputs
  3. Compute structural loss, accuracy, projected IoU
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from training.phase62.losses import loss_diffusion, loss_volume_ce, compute_volume_accuracy
from training.phase62.metrics import compute_projected_class_iou, compute_entity_accuracy
from data.phase62.volume_gt_builder import VolumeGTBuilder
from scripts.train_animatediff_vca import encode_frames_to_latents
from scripts.train_phase35 import get_entity_token_positions


def _encode_text(pipe, text: str, device: str) -> torch.Tensor:
    tok = pipe.tokenizer(
        text, return_tensors="pt", padding="max_length",
        max_length=pipe.tokenizer.model_max_length, truncation=True,
    ).to(device)
    with torch.no_grad():
        enc = pipe.text_encoder(**tok).last_hidden_state.half()
    return enc


def _get_entity_tokens_with_fallback(pipe, meta, device):
    toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)

    if not toks_e0 or not toks_e1:
        kw0 = str(meta.get("keyword0", "cat"))
        kw1 = str(meta.get("keyword1", "dog"))
        tok_ids = pipe.tokenizer.encode(full_prompt)
        kw0_ids = pipe.tokenizer.encode(kw0, add_special_tokens=False)
        kw1_ids = pipe.tokenizer.encode(kw1, add_special_tokens=False)
        if not toks_e0 and kw0_ids:
            for i in range(len(tok_ids) - len(kw0_ids) + 1):
                if tok_ids[i:i + len(kw0_ids)] == kw0_ids:
                    toks_e0 = list(range(i, i + len(kw0_ids)))
                    break
        if not toks_e1 and kw1_ids:
            for i in range(len(tok_ids) - len(kw1_ids) + 1):
                if tok_ids[i:i + len(kw1_ids)] == kw1_ids:
                    toks_e1 = list(range(i, i + len(kw1_ids)))
                    break
        if not toks_e0:
            toks_e0 = [1, 2]
        if not toks_e1:
            toks_e1 = [3, 4]

    toks_e0_t = torch.tensor(toks_e0, dtype=torch.long, device=device)
    toks_e1_t = torch.tensor(toks_e1, dtype=torch.long, device=device)
    return toks_e0_t, toks_e1_t, full_prompt


class Phase62Evaluator:

    def __init__(self):
        self._gt_builder: VolumeGTBuilder | None = None

    def _get_gt_builder(self, config) -> VolumeGTBuilder:
        if self._gt_builder is None:
            self._gt_builder = VolumeGTBuilder(
                depth_bins=config.depth_bins,
                spatial_h=config.spatial_h,
                spatial_w=config.spatial_w,
                render_resolution=getattr(config.data, "volume_gt_render_resolution", 128),
            )
        return self._gt_builder

    @torch.no_grad()
    def evaluate(
        self,
        pipe,
        system,
        backbone_mgr,
        dataset,
        val_idx: List[int],
        device: str,
        config,
    ) -> Dict:
        system.eval()
        backbone_mgr.eval()

        gt_builder = self._get_gt_builder(config)
        t_fixed = getattr(config, 'training', config).t_max // 2 \
            if hasattr(config, 'training') and hasattr(config.training, 't_max') \
            else 150

        diff_losses: List[float] = []
        vol_ce_losses: List[float] = []
        vol_accs_overall: List[float] = []
        vol_accs_entity: List[float] = []
        ious_e0: List[float] = []
        ious_e1: List[float] = []

        for vi in val_idx:
            try:
                sample = dataset[vi]

                visible_masks = None
                sample_dir = None
                if isinstance(sample, dict):
                    frames_np = sample["frames"]
                    depth_np = sample["depth"]
                    depth_orders = sample["depth_orders"]
                    meta = sample["meta"]
                    sample_dir = sample.get("sample_dir")
                    entity_masks = sample["entity_masks"]
                    visible_masks = sample.get("visible_masks")
                else:
                    if len(sample) >= 8:
                        frames_np, depth_np, depth_orders = sample[0], sample[1], sample[2]
                        meta, entity_masks = sample[3], sample[4]
                        visible_masks = sample[5]
                    else:
                        frames_np, depth_np, depth_orders, meta, entity_masks = sample[:5]

                toks_e0_t, toks_e1_t, full_prompt = _get_entity_tokens_with_fallback(
                    pipe, meta, device)
                backbone_mgr.set_entity_tokens(toks_e0_t, toks_e1_t)
                backbone_mgr.reset_slot_store()

                latents = encode_frames_to_latents(pipe, frames_np, device)
                noise = torch.randn_like(latents)
                t_tensor = torch.tensor([t_fixed], device=device).long()
                noisy = pipe.scheduler.add_noise(latents, noise, t_tensor)
                enc_full = _encode_text(pipe, full_prompt, device)

                T_frames = min(frames_np.shape[0], entity_masks.shape[0],
                               len(depth_orders))

                system.clear_guides()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    noise_pred = pipe.unet(
                        noisy, t_tensor, encoder_hidden_states=enc_full).sample

                noise_t = noise[:, :, :T_frames].float()
                noise_pred_t = noise_pred[:, :, :T_frames].float()
                l_diff = loss_diffusion(noise_pred_t, noise_t)
                diff_losses.append(float(l_diff.item()))

                F_g = backbone_mgr.primary.last_Fg
                F_0 = backbone_mgr.primary.last_F0
                F_1 = backbone_mgr.primary.last_F1
                if F_g is None or F_0 is None or F_1 is None:
                    continue

                # Volume prediction returns VolumeOutputs
                vol_outputs = system.predict_volume(F_g, F_0, F_1)

                depth_maps = depth_np[:T_frames]
                V_gt_np = gt_builder.build_batch(
                    depth_maps, entity_masks[:T_frames],
                    depth_orders[:T_frames],
                    visible_masks=(visible_masks[:T_frames] if visible_masks is not None else None),
                    meta=meta,
                    sample_dir=sample_dir,
                )
                V_gt = torch.from_numpy(V_gt_np).to(device).long()

                B_feat = vol_outputs.entity_probs.shape[0]
                if V_gt.shape[0] < B_feat:
                    n_rep = max(1, B_feat // V_gt.shape[0])
                    V_gt = V_gt.repeat(n_rep, 1, 1, 1)[:B_feat]

                # Volume CE (backward compat — uses entity_logits if available)
                if vol_outputs.entity_logits is not None:
                    V_logits_3ch = torch.cat([
                        torch.zeros(B_feat, 1, *vol_outputs.entity_logits.shape[2:],
                                    device=device),
                        vol_outputs.entity_logits,
                    ], dim=1)
                    l_vol = loss_volume_ce(V_logits_3ch, V_gt)
                    vol_ce_losses.append(float(l_vol.item()))

                    acc = compute_volume_accuracy(V_logits_3ch, V_gt)
                else:
                    # Factorized: compute accuracy from entity_probs
                    ep = vol_outputs.entity_probs
                    p0, p1 = ep[:, 0], ep[:, 1]
                    pred_class = torch.zeros_like(V_gt.long())
                    has_ent = (p0 > 0.5) | (p1 > 0.5)
                    pred_class = torch.where(has_ent & (p0 >= p1), torch.ones_like(pred_class), pred_class)
                    pred_class = torch.where(has_ent & (p1 > p0), torch.full_like(pred_class, 2), pred_class)
                    correct = (pred_class == V_gt.long())
                    acc = {
                        "overall_acc": correct.float().mean().item(),
                        "entity_acc": correct[V_gt > 0].float().mean().item() if (V_gt > 0).any() else 0.0,
                    }

                vol_accs_overall.append(acc["overall_acc"])
                vol_accs_entity.append(acc["entity_acc"])

                # Project for IoU
                vol_outputs = system.projector(vol_outputs)
                visible_class = vol_outputs.visible_class

                if entity_masks.shape[0] > 0:
                    src_masks = visible_masks if visible_masks is not None else entity_masks
                    n_eval = min(int(visible_class.shape[0]), int(src_masks.shape[0]))
                    if n_eval > 0:
                        vc_eval = visible_class[:n_eval]
                        m0_gt = torch.from_numpy(
                            src_masks[:n_eval, 0].astype(np.float32)).to(device)
                        m1_gt = torch.from_numpy(
                            src_masks[:n_eval, 1].astype(np.float32)).to(device)
                        iou_0 = compute_projected_class_iou(
                            vc_eval, m0_gt, entity_idx=1,
                            spatial_h=config.spatial_h, spatial_w=config.spatial_w)
                        iou_1 = compute_projected_class_iou(
                            vc_eval, m1_gt, entity_idx=2,
                            spatial_h=config.spatial_h, spatial_w=config.spatial_w)
                        ious_e0.append(iou_0)
                        ious_e1.append(iou_1)

            except Exception as e:
                print(f"  [val warn] idx={vi}: {e}", flush=True)
                continue

        def _avg(lst):
            return sum(lst) / max(len(lst), 1) if lst else 999.0

        val_diff = _avg(diff_losses)
        val_vol_ce = _avg(vol_ce_losses) if vol_ce_losses else 0.0
        val_acc_all = _avg(vol_accs_overall) if vol_accs_overall else 0.0
        val_acc_ent = _avg(vol_accs_entity) if vol_accs_entity else 0.0
        val_iou_e0 = _avg(ious_e0) if ious_e0 else 0.0
        val_iou_e1 = _avg(ious_e1) if ious_e1 else 0.0

        val_iou_min = min(val_iou_e0, val_iou_e1)

        val_score = (
            0.10 * (1.0 / (1.0 + val_diff))
            + 0.10 * (1.0 / (1.0 + val_vol_ce))
            + 0.20 * val_acc_ent
            + 0.15 * val_iou_e0
            + 0.15 * val_iou_e1
            + 0.30 * val_iou_min
        )

        return {
            "val_score": val_score,
            "val_diff_mse": val_diff,
            "val_vol_ce": val_vol_ce,
            "val_acc_overall": val_acc_all,
            "val_acc_entity": val_acc_ent,
            "val_iou_e0": val_iou_e0,
            "val_iou_e1": val_iou_e1,
            "val_iou_min": val_iou_min,
            "n_samples": len(diff_losses),
        }
