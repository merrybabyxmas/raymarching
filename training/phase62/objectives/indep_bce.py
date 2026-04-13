"""
Independent BCE objective — current baseline.

L = sum_n BCE(z_n, Y_n) with entity_pos_weight upweighting and dynamic balancing.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from training.phase62.objectives.base import VolumeObjective, VolumeOutputs


class IndependentBCEObjective(VolumeObjective):

    def __init__(self, entity_pos_weight: float = 50.0, clamp_max: float = 20.0):
        super().__init__()
        self.entity_pos_weight = entity_pos_weight
        self.clamp_max = clamp_max

    def forward(
        self,
        outputs: VolumeOutputs,
        V_gt: torch.Tensor,
        gt_visible: Optional[torch.Tensor] = None,
        gt_amodal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = outputs.entity_logits  # (B, 2, K, H, W)
        target_e0 = (V_gt == 1).float()
        target_e1 = (V_gt == 2).float()

        def _entity_bce(logits_n, tgt_n):
            bce = F.binary_cross_entropy_with_logits(logits_n, tgt_n, reduction="none")
            bce = bce.clamp(max=self.clamp_max)
            pos = (tgt_n > 0.5)
            neg = ~pos
            n_pos = pos.float().sum().clamp(min=1.0)
            n_neg = neg.float().sum().clamp(min=1.0)
            l_pos = (bce * pos.float()).sum() / n_pos
            l_neg = (bce * neg.float()).sum() / n_neg
            return self.entity_pos_weight * l_pos + l_neg

        l_e0 = _entity_bce(logits[:, 0], target_e0)
        l_e1 = _entity_bce(logits[:, 1], target_e1)

        with torch.no_grad():
            ratio = (l_e0 / (l_e1 + 1e-6)).clamp(0.8, 1.25)
            w0 = ratio / (ratio + 1.0)
            w1 = 1.0 - w0

        total = w0 * l_e0 + w1 * l_e1
        return {"total": total, "l_e0": l_e0.detach(), "l_e1": l_e1.detach()}
