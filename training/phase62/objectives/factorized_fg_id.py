"""
Factorized Foreground + Identity objective.

p_fg(x) = sigmoid(z_fg(x))
q_n(x)  = softmax(z_id(x))_n,  n in {0, 1}
p_n(x)  = p_fg(x) * q_n(x)

L_fg = BCE(z_fg, Y_fg)         where Y_fg = 1[V_gt > 0]
L_id = CE(z_id, Y_id | Y_fg=1) where Y_id = 0 if V_gt=1, 1 if V_gt=2
L    = L_fg + lambda_id * L_id
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from training.phase62.objectives.base import VolumeObjective, VolumeOutputs


class FactorizedFgIdObjective(VolumeObjective):

    def __init__(
        self,
        lambda_id: float = 1.0,
        fg_pos_weight: float = 20.0,
    ):
        super().__init__()
        self.lambda_id = lambda_id
        self.fg_pos_weight = fg_pos_weight

    def forward(
        self,
        outputs: VolumeOutputs,
        V_gt: torch.Tensor,
        gt_visible: Optional[torch.Tensor] = None,
        gt_amodal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if outputs.fg_logit is None or outputs.id_logits is None:
            raise ValueError(
                "factorized_fg_id requires fg_logit and id_logits in VolumeOutputs"
            )

        fg_logit = outputs.fg_logit[:, 0]   # (B, K, H, W)
        id_logits = outputs.id_logits        # (B, 2, K, H, W)

        # Y_fg = 1[V_gt > 0]
        Y_fg = (V_gt > 0).float()

        # L_fg: BCE with pos_weight to combat sparse foreground (~1% of voxels)
        pos_weight = torch.tensor([self.fg_pos_weight], device=fg_logit.device)
        L_fg = F.binary_cross_entropy_with_logits(
            fg_logit, Y_fg,
            pos_weight=pos_weight,
            reduction="mean",
        )

        # L_id: CE only where Y_fg == 1
        fg_mask = (V_gt > 0)  # (B, K, H, W)
        if fg_mask.any():
            # Y_id: 0 where V_gt==1, 1 where V_gt==2
            Y_id = (V_gt - 1).clamp(min=0).long()  # 0 for e0, 1 for e1

            # id_logits: (B, 2, K, H, W) -> gather fg voxels
            # Reshape to (N, 2) where N = number of fg voxels
            id_logits_flat = id_logits.permute(0, 2, 3, 4, 1)  # (B, K, H, W, 2)
            id_at_fg = id_logits_flat[fg_mask]  # (N, 2)
            Y_id_at_fg = Y_id[fg_mask]          # (N,)

            L_id = F.cross_entropy(id_at_fg, Y_id_at_fg, reduction="mean")
        else:
            L_id = fg_logit.new_zeros(())

        total = L_fg + self.lambda_id * L_id
        return {
            "total": total,
            "L_fg": L_fg.detach(),
            "L_id": L_id.detach(),
        }
