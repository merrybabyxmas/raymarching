from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .losses import _masked_avg_pool
from .scene_outputs import SceneState


def _binary_iou(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5, eps: float = 1e-6) -> float:
    pb = (pred > thresh).float()
    tb = (target > thresh).float()
    inter = (pb * tb).sum(dim=(1, 2, 3))
    union = (pb + tb - pb * tb).sum(dim=(1, 2, 3))
    return float(((inter + eps) / (union + eps)).mean().item())


class Phase65Evaluator:
    def evaluate_scene(
        self,
        scene_state: SceneState,
        gt_visible: torch.Tensor,
        gt_amodal: torch.Tensor,
        prev_state: Optional[SceneState] = None,
    ) -> Dict[str, float]:
        m = scene_state.maps
        vis_iou_e0 = _binary_iou(m.visible_e0, gt_visible[:, 0:1])
        vis_iou_e1 = _binary_iou(m.visible_e1, gt_visible[:, 1:2])
        amo_iou_e0 = _binary_iou(m.amodal_e0, gt_amodal[:, 0:1])
        amo_iou_e1 = _binary_iou(m.amodal_e1, gt_amodal[:, 1:2])
        visible_survival_e0 = float((m.visible_e0.mean(dim=(1, 2, 3)) > 0.02).float().mean().item())
        visible_survival_e1 = float((m.visible_e1.mean(dim=(1, 2, 3)) > 0.02).float().mean().item())
        out = {
            "visible_iou_e0": vis_iou_e0,
            "visible_iou_e1": vis_iou_e1,
            "visible_iou_min": min(vis_iou_e0, vis_iou_e1),
            "amodal_iou_e0": amo_iou_e0,
            "amodal_iou_e1": amo_iou_e1,
            "amodal_iou_min": min(amo_iou_e0, amo_iou_e1),
            "visible_survival_min": min(visible_survival_e0, visible_survival_e1),
        }
        if prev_state is not None:
            z0 = _masked_avg_pool(scene_state.features.feat_e0, scene_state.maps.amodal_e0)
            z1 = _masked_avg_pool(scene_state.features.feat_e1, scene_state.maps.amodal_e1)
            z0_prev = _masked_avg_pool(prev_state.features.feat_e0, prev_state.maps.amodal_e0)
            z1_prev = _masked_avg_pool(prev_state.features.feat_e1, prev_state.maps.amodal_e1)
            drift = F.mse_loss(z0, z0_prev) + F.mse_loss(z1, z1_prev)
            out["temporal_identity_drift"] = float(drift.item())
        return out

    def score_for_checkpoint(self, metrics: Dict[str, float]) -> float:
        # Hard reject obvious single-entity collapse.
        if metrics.get("visible_survival_min", 0.0) < 0.10:
            return -1.0
        return (
            0.40 * metrics.get("visible_iou_min", 0.0)
            + 0.30 * metrics.get("amodal_iou_min", 0.0)
            + 0.20 * metrics.get("visible_survival_min", 0.0)
            - 0.10 * metrics.get("temporal_identity_drift", 0.0)
        )
