"""
Projected Visible-Only objective.

V_n(h,w) = sum_k T_k(h,w) * p_n(k,h,w)
L = sum_n (1 - Dice(V_n, V_n^gt))
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from training.phase62.objectives.base import VolumeObjective, VolumeOutputs


def _dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(-2, -1))
    denom = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


class ProjectedVisibleObjective(VolumeObjective):

    def forward(
        self,
        outputs: VolumeOutputs,
        V_gt: torch.Tensor,
        gt_visible: Optional[torch.Tensor] = None,
        gt_amodal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if gt_visible is None:
            raise ValueError("projected_visible_only requires gt_visible")

        visible_e0 = outputs.visible["e0"]  # (B, H, W)
        visible_e1 = outputs.visible["e1"]  # (B, H, W)

        L_vis_e0 = _dice_loss(visible_e0, gt_visible[:, 0])
        L_vis_e1 = _dice_loss(visible_e1, gt_visible[:, 1])

        total = L_vis_e0 + L_vis_e1
        return {
            "total": total,
            "L_vis_e0": L_vis_e0.detach(),
            "L_vis_e1": L_vis_e1.detach(),
        }
