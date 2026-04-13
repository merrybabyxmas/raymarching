"""
Projected Amodal-Only objective.

A_n(h,w) = 1 - prod_k(1 - p_n(k,h,w))
L = sum_n (1 - Dice(A_n, A_n^gt))
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from training.phase62.objectives.base import VolumeObjective, VolumeOutputs


def _dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(-2, -1))
    denom = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


class ProjectedAmodalObjective(VolumeObjective):

    def forward(
        self,
        outputs: VolumeOutputs,
        V_gt: torch.Tensor,
        gt_visible: Optional[torch.Tensor] = None,
        gt_amodal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if gt_amodal is None:
            raise ValueError("projected_amodal_only requires gt_amodal")

        amodal_e0 = outputs.amodal["e0"]  # (B, H, W)
        amodal_e1 = outputs.amodal["e1"]  # (B, H, W)

        L_amo_e0 = _dice_loss(amodal_e0, gt_amodal[:, 0])
        L_amo_e1 = _dice_loss(amodal_e1, gt_amodal[:, 1])

        total = L_amo_e0 + L_amo_e1
        return {
            "total": total,
            "L_amo_e0": L_amo_e0.detach(),
            "L_amo_e1": L_amo_e1.detach(),
        }
