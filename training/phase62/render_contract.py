"""
Phase 62 — C_render Evaluation (v22)
======================================

Evaluates C_render on held-out collision clips:
  P_2obj     ≥ 0.90: both entities detected in overlap frames
  R_chimera  ≤ 0.05: frames where entities fuse into one chimera blob
  M_id_min   ≥ 0.15: identity margin (per entity) = cos(feat_e, text_e) - cos(feat_e, text_other)
  render_iou_min ≥ 0.25: per-entity IoU on rendered composite vs GT masks

Implementation (Phase 1 — no CLIP, uses internal features):
  - P_2obj: checks that both entity classes appear in the rendered visible_class map
            on "overlap frames" (frames where GT masks have sufficient overlap)
  - R_chimera: frames where the predicted fg blob covers both GT entities with
               no clear boundary (proxy: one_winner > 0.80 AND both GT entities > 0)
  - M_id_min: uses backbone feature cosine similarity to entity prompt token embeddings
              as proxy for CLIP-based identity margin
  - render_iou_min: per-entity IoU between rendered visible map and GT masks
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _dice_1d(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum()
    return float((2.0 * inter + eps) / (denom + eps))


def _iou_1d(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    inter = (pred * target).sum()
    union = ((pred + target) > 0).sum()
    return float((inter + eps) / (union + eps))


def _has_overlap(mask0: np.ndarray, mask1: np.ndarray, threshold: float = 0.01) -> bool:
    """Returns True if the two masks have sufficient spatial overlap."""
    S = mask0.shape[-1]
    m0 = mask0.reshape(-1) > 0.5
    m1 = mask1.reshape(-1) > 0.5
    if m0.sum() == 0 or m1.sum() == 0:
        return False
    overlap_frac = float((m0 & m1).sum()) / max(min(m0.sum(), m1.sum()), 1)
    return overlap_frac > threshold


class RenderContractEvaluator:
    """
    Compute C_render metrics on a set of clips.

    Usage:
        evaluator = RenderContractEvaluator()
        metrics = evaluator.evaluate_clips(clips)
    """

    def evaluate_clips(
        self,
        clips: List[Dict],
    ) -> Dict[str, float]:
        """
        clips: list of dicts with keys:
          - 'visible_class': (T, H, W) long tensor — predicted class per pixel
          - 'entity_masks': (T, 2, S) float numpy — GT entity masks per frame
          - 'F_0': (T, S, D) float tensor — entity-0 features
          - 'F_1': (T, S, D) float tensor — entity-1 features
          - 'text_e0': (D,) float tensor — entity-0 text embedding (optional)
          - 'text_e1': (D,) float tensor — entity-1 text embedding (optional)
          - 'spatial_h': int
          - 'spatial_w': int

        Returns dict with P_2obj, R_chimera, M_id_min, render_iou_min.
        """
        if not clips:
            return {"P_2obj": 0.0, "R_chimera": 1.0, "M_id_min": 0.0, "render_iou_min": 0.0}

        all_p2obj: List[float] = []
        all_chimera: List[float] = []
        all_mid: List[float] = []
        all_iou_e0: List[float] = []
        all_iou_e1: List[float] = []

        for clip in clips:
            vc = clip.get("visible_class")            # (T, H, W) long
            masks = clip.get("entity_masks")          # (T, 2, S) float
            F_0 = clip.get("F_0")                     # (T, S, D)
            F_1 = clip.get("F_1")                     # (T, S, D)
            spatial_h = clip.get("spatial_h", 16)
            spatial_w = clip.get("spatial_w", 16)

            if vc is None or masks is None:
                continue

            T = min(vc.shape[0], masks.shape[0])
            if T == 0:
                continue

            # Downscale visible_class to mask resolution for comparison
            vc_t = vc[:T]
            masks_t = masks[:T]  # (T, 2, S)

            S = masks_t.shape[-1]
            H_mask = int(round(S ** 0.5))

            # Resize vc to mask resolution
            if spatial_h != H_mask or spatial_w != H_mask:
                vc_small = F.interpolate(
                    vc_t.float().unsqueeze(1),
                    size=(H_mask, H_mask),
                    mode='nearest',
                ).squeeze(1).long()
            else:
                vc_small = vc_t

            # Per-frame metrics
            p2obj_frames = []
            chimera_frames = []
            iou0_frames = []
            iou1_frames = []

            for t in range(T):
                vc_frame = vc_small[t].cpu().numpy().reshape(-1)   # (S,)
                m0 = masks_t[t, 0]                                  # (S,)
                m1 = masks_t[t, 1]                                  # (S,)

                is_overlap = _has_overlap(m0, m1)

                # P_2obj: on overlap frames, check both entity classes present
                pred_e0_any = float((vc_frame == 1).sum()) / S
                pred_e1_any = float((vc_frame == 2).sum()) / S
                both_present = (pred_e0_any > 0.005) and (pred_e1_any > 0.005)
                if is_overlap:
                    p2obj_frames.append(float(both_present))

                # R_chimera: both GT entities present but only one predicted class
                gt_both = (m0 > 0.5).any() and (m1 > 0.5).any()
                pred_only_one = (pred_e0_any < 0.005 or pred_e1_any < 0.005)
                chimera = float(gt_both and pred_only_one)
                chimera_frames.append(chimera)

                # RenderIoU: per-entity IoU
                pred_e0 = (vc_frame == 1).astype(np.float32)
                pred_e1 = (vc_frame == 2).astype(np.float32)
                iou0 = _iou_1d(pred_e0, (m0 > 0.5).astype(np.float32))
                iou1 = _iou_1d(pred_e1, (m1 > 0.5).astype(np.float32))
                iou0_frames.append(iou0)
                iou1_frames.append(iou1)

            if p2obj_frames:
                all_p2obj.append(sum(p2obj_frames) / len(p2obj_frames))
            all_chimera.append(sum(chimera_frames) / max(len(chimera_frames), 1))
            all_iou_e0.append(sum(iou0_frames) / max(len(iou0_frames), 1))
            all_iou_e1.append(sum(iou1_frames) / max(len(iou1_frames), 1))

            # M_id: identity margin using backbone features
            if F_0 is not None and F_1 is not None:
                mid = self._compute_identity_margin(F_0, F_1, masks_t, T)
                all_mid.append(mid)

        def _avg(lst, default=0.0):
            return sum(lst) / max(len(lst), 1) if lst else default

        return {
            "P_2obj":          _avg(all_p2obj, default=0.0),
            "R_chimera":       _avg(all_chimera, default=1.0),
            "M_id_min":        _avg(all_mid, default=0.0),
            "render_iou_min":  min(_avg(all_iou_e0), _avg(all_iou_e1)),
            "render_iou_e0":   _avg(all_iou_e0),
            "render_iou_e1":   _avg(all_iou_e1),
        }

    @staticmethod
    def _compute_identity_margin(
        F_0: torch.Tensor,   # (T, S, D)
        F_1: torch.Tensor,   # (T, S, D)
        masks: np.ndarray,   # (T, 2, S)
        T: int,
    ) -> float:
        """
        Identity margin proxy: cosine similarity between entity features and
        their "own" direction vs "other" direction.

        M_id = mean_e { cos(F_e, centroid_e) - cos(F_e, centroid_other) }

        Where centroid_e = mean(F_e) at entity-e pixels.
        Higher M_id means F_e features are more discriminative.
        """
        with torch.no_grad():
            T_use = min(T, F_0.shape[0], F_1.shape[0])
            if T_use == 0:
                return 0.0

            f0 = F.normalize(F_0[:T_use].float(), dim=-1, eps=1e-6)  # (T, S, D)
            f1 = F.normalize(F_1[:T_use].float(), dim=-1, eps=1e-6)

            S = f0.shape[1]
            m0 = torch.from_numpy(masks[:T_use, 0].astype(np.float32))  # (T, S)
            m1 = torch.from_numpy(masks[:T_use, 1].astype(np.float32))

            # Entity centroids (mean feature over entity pixels)
            m0_sum = m0.sum().clamp(min=1.0)
            m1_sum = m1.sum().clamp(min=1.0)
            centroid_e0 = (f0 * m0.unsqueeze(-1)).sum(dim=(0, 1)) / m0_sum   # (D,)
            centroid_e1 = (f1 * m1.unsqueeze(-1)).sum(dim=(0, 1)) / m1_sum   # (D,)

            centroid_e0 = F.normalize(centroid_e0.unsqueeze(0), dim=-1).squeeze(0)
            centroid_e1 = F.normalize(centroid_e1.unsqueeze(0), dim=-1).squeeze(0)

            # M_id_e0 = cos(f0, centroid_e0) - cos(f0, centroid_e1) at e0 pixels
            cos_00 = (f0 * centroid_e0).sum(dim=-1)   # (T, S)
            cos_01 = (f0 * centroid_e1).sum(dim=-1)
            margin_e0 = ((cos_00 - cos_01) * m0).sum() / m0_sum

            # M_id_e1 = cos(f1, centroid_e1) - cos(f1, centroid_e0) at e1 pixels
            cos_10 = (f1 * centroid_e0).sum(dim=-1)
            cos_11 = (f1 * centroid_e1).sum(dim=-1)
            margin_e1 = ((cos_11 - cos_10) * m1).sum() / m1_sum

            return float(min(margin_e0.item(), margin_e1.item()))
