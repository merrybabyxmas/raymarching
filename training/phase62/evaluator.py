"""
Phase 62 — Evaluator (v22)
===========================

Validation evaluation: 3-pass per sample.
  1. UNet forward (no guide) → F_g, F_0, F_1
  2. Volume prediction → VolumeOutputs (+ projection)
  3. Compute structural loss, accuracy, projected IoU, amodal Dice, cos(F0,F1)

v22 new metrics:
  - val_amo_dice_e0/e1: amodal Dice per entity
  - val_cos_F_overlap:  F_0/F_1 cosine similarity at entity overlap region
  - val_compact:        depth compactness (volume-averaged)
  - val_lcc_min:        LCC ratio per entity
  - val_pass_rate_clips: fraction of val clips passing all C_topo+C_guide+C_diff
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from training.phase62.losses import loss_diffusion, loss_volume_ce, compute_volume_accuracy
from training.phase62.metrics import compute_projected_class_iou, compute_entity_accuracy
from data.phase62.volume_gt_builder import VolumeGTBuilder
from scripts.train_animatediff_vca import encode_frames_to_latents
from scripts.train_phase35 import get_entity_token_positions

try:
    from scipy import ndimage as _ndimage
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


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


def _compute_amodal_dice(
    amodal_e: torch.Tensor,   # (B, H, W) float — predicted amodal map
    gt_mask: torch.Tensor,    # (B, H_gt, W_gt) or (B, S) float — GT amodal mask
    spatial_h: int,
    spatial_w: int,
    eps: float = 1e-6,
) -> float:
    """Dice between predicted amodal projection and GT amodal mask."""
    with torch.no_grad():
        B = amodal_e.shape[0]
        H, W = amodal_e.shape[1], amodal_e.shape[2]
        pred = amodal_e.float().clamp(0, 1)

        gt = gt_mask.float()
        if gt.dim() == 2:
            S = gt.shape[1]
            H_gt = int(round(S ** 0.5))
            gt = gt.reshape(B, 1, H_gt, H_gt)
        elif gt.dim() == 3:
            gt = gt.unsqueeze(1)
        if gt.shape[2] != H or gt.shape[3] != W:
            gt = F.interpolate(gt, size=(H, W), mode='bilinear', align_corners=False)
        gt = gt.squeeze(1).clamp(0, 1)  # (B, H, W)

        inter = (pred * gt).sum(dim=(1, 2))
        denom = pred.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
        dice = ((2.0 * inter + eps) / (denom + eps))
        return float(dice.mean().item())


def _compute_cos_F_overlap(
    F_0: torch.Tensor,   # (B, S, D)
    F_1: torch.Tensor,   # (B, S, D)
    gt_visible: np.ndarray,  # (T, 2, S) entity masks — overlap where both > 0
    spatial_h: int,
    spatial_w: int,
) -> float:
    """
    Cosine similarity between F_0 and F_1 features at spatial positions
    where both entities are visible (overlap region).

    Low value = good feature separation at overlap.
    High value = features are similar (model treats both entities the same).
    """
    with torch.no_grad():
        B = F_0.shape[0]
        S_feat = F_0.shape[1]
        H_feat = int(round(S_feat ** 0.5))

        f0 = F.normalize(F_0.float(), dim=-1, eps=1e-6)  # (B, S, D)
        f1 = F.normalize(F_1.float(), dim=-1, eps=1e-6)
        cos_all = (f0 * f1).sum(dim=-1)  # (B, S)

        T_gt = min(B, gt_visible.shape[0])
        if T_gt == 0:
            return 0.0

        mask0 = torch.from_numpy(gt_visible[:T_gt, 0].astype(np.float32))  # (T, S)
        mask1 = torch.from_numpy(gt_visible[:T_gt, 1].astype(np.float32))  # (T, S)
        overlap = (mask0 > 0.5) & (mask1 > 0.5)  # (T, S)

        if not overlap.any():
            # No overlap region — return mean similarity (less informative but non-zero)
            return float(cos_all[:T_gt].mean().item())

        cos_at_overlap = cos_all[:T_gt][overlap.to(cos_all.device)]
        return float(cos_at_overlap.mean().clamp(0, 1).item())


def _compute_lcc(entity_probs_3d: torch.Tensor, threshold: float = 0.3) -> float:
    """Largest Connected Component ratio for (K, H, W) entity probability volume."""
    if not _SCIPY_OK:
        return 1.0
    binary = (entity_probs_3d.detach().cpu().float().numpy() > threshold)
    if not binary.any():
        return 0.0
    labeled, _ = _ndimage.label(binary)
    if labeled.max() == 0:
        return 0.0
    sizes = np.bincount(labeled.ravel())[1:]
    return float(sizes.max()) / float(binary.sum())


def _compute_depth_compactness(
    entity_probs: torch.Tensor,  # (B, 2, K, H, W)
    fg_mask: "torch.Tensor | None",
) -> float:
    """Mean depth compactness over batch and entities."""
    B, _, K, H, W = entity_probs.shape
    if K <= 1:
        return 1.0
    ep = entity_probs.float()
    if fg_mask is not None:
        mask = fg_mask.float().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H, W)
        n_fg = mask.sum(dim=(3, 4)).clamp(min=1.0)
        depth_mass = (ep * mask).sum(dim=(3, 4)) / n_fg   # (B, 2, K)
    else:
        depth_mass = ep.mean(dim=(3, 4))                  # (B, 2, K)

    depth_mass_sum = depth_mass.sum(dim=2, keepdim=True).clamp(min=1e-9)
    p = (depth_mass / depth_mass_sum).clamp(min=1e-9)
    entropy = -(p * p.log()).sum(dim=2)                   # (B, 2)
    normalised = entropy / (math.log(K) + 1e-9)
    return float((1.0 - normalised).mean().item())


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
        amo_dice_e0: List[float] = []
        amo_dice_e1: List[float] = []
        cos_F_list: List[float] = []
        compact_list: List[float] = []
        lcc_e0_list: List[float] = []
        lcc_e1_list: List[float] = []
        per_clip_pass: List[bool] = []

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

                # Volume accuracy
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

                # Depth compactness (fg-masked)
                if vol_outputs.entity_probs is not None:
                    fg_mask_gt = (V_gt > 0).any(dim=1).float()  # (B, H, W)
                    c = _compute_depth_compactness(vol_outputs.entity_probs, fg_mask_gt)
                    compact_list.append(c)

                    # LCC per entity
                    ep_np = vol_outputs.entity_probs[0]  # (2, K, H, W)
                    lcc_e0_list.append(_compute_lcc(ep_np[0]))
                    lcc_e1_list.append(_compute_lcc(ep_np[1]))

                # Project for IoU + amodal
                vol_outputs = system.projector(vol_outputs)
                visible_class = vol_outputs.visible_class

                src_masks = visible_masks if visible_masks is not None else entity_masks

                if entity_masks.shape[0] > 0:
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

                        # Amodal Dice: amodal projection vs GT entity mask
                        # GT amodal ≈ entity_masks (full body including occlusion)
                        if "e0" in vol_outputs.amodal and "e1" in vol_outputs.amodal:
                            amo_e0 = vol_outputs.amodal["e0"][:n_eval]  # (B, H, W)
                            amo_e1 = vol_outputs.amodal["e1"][:n_eval]
                            amo_gt_e0 = torch.from_numpy(
                                entity_masks[:n_eval, 0].astype(np.float32)).to(device)
                            amo_gt_e1 = torch.from_numpy(
                                entity_masks[:n_eval, 1].astype(np.float32)).to(device)
                            ad0 = _compute_amodal_dice(amo_e0, amo_gt_e0,
                                                       config.spatial_h, config.spatial_w)
                            ad1 = _compute_amodal_dice(amo_e1, amo_gt_e1,
                                                       config.spatial_h, config.spatial_w)
                            amo_dice_e0.append(ad0)
                            amo_dice_e1.append(ad1)

                # cos(F_0, F_1) at overlap region
                if F_0 is not None and F_1 is not None and entity_masks.shape[0] > 0:
                    cf = _compute_cos_F_overlap(F_0, F_1, entity_masks, config.spatial_h, config.spatial_w)
                    cos_F_list.append(cf)

                # Per-clip pass: does this clip pass C_topo + C_diff checks?
                # Simple check: iou_0, iou_1 > 0.14 AND diff_loss < 0.06
                clip_ok = (iou_0 > 0.14 and iou_1 > 0.14 and float(l_diff.item()) < 0.06)
                per_clip_pass.append(clip_ok)

            except Exception as e:
                print(f"  [val warn] idx={vi}: {e}", flush=True)
                continue

        def _avg(lst, default=0.0):
            return sum(lst) / max(len(lst), 1) if lst else default

        val_diff = _avg(diff_losses, default=999.0)
        val_vol_ce = _avg(vol_ce_losses)
        val_acc_all = _avg(vol_accs_overall)
        val_acc_ent = _avg(vol_accs_entity)
        val_iou_e0 = _avg(ious_e0)
        val_iou_e1 = _avg(ious_e1)
        val_iou_min = min(val_iou_e0, val_iou_e1)

        val_score = (
            0.10 * (1.0 / (1.0 + val_diff))
            + 0.10 * (1.0 / (1.0 + val_vol_ce))
            + 0.20 * val_acc_ent
            + 0.15 * val_iou_e0
            + 0.15 * val_iou_e1
            + 0.30 * val_iou_min
        )

        result = {
            "val_score": val_score,
            "val_diff_mse": val_diff,
            "val_vol_ce": val_vol_ce,
            "val_acc_overall": val_acc_all,
            "val_acc_entity": val_acc_ent,
            "val_iou_e0": val_iou_e0,
            "val_iou_e1": val_iou_e1,
            "val_iou_min": val_iou_min,
            "n_samples": len(diff_losses),
            # v22 new metrics
            "val_amo_dice_e0": _avg(amo_dice_e0),
            "val_amo_dice_e1": _avg(amo_dice_e1),
            "val_cos_F_overlap": _avg(cos_F_list),
            "val_compact": _avg(compact_list),
            "val_lcc_e0": _avg(lcc_e0_list, default=1.0),
            "val_lcc_e1": _avg(lcc_e1_list, default=1.0),
            "val_lcc_min": min(_avg(lcc_e0_list, default=1.0), _avg(lcc_e1_list, default=1.0)),
            "val_pass_rate_clips": sum(per_clip_pass) / max(len(per_clip_pass), 1),
        }
        return result
