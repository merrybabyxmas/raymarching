"""
Factorized Foreground + Identity objective with rendering-consistent terms.

p_fg(x) = sigmoid(z_fg(x))
q_n(x)  = softmax(z_id(x))_n,  n in {0, 1}
p_n(x)  = p_fg(x) * q_n(x)

L_fg  = BCE(z_fg, Y_fg)           where Y_fg = 1[V_gt > 0]
L_id  = CE(z_id, Y_id | Y_fg=1)  where Y_id = 0 if V_gt=1, 1 if V_gt=2
L_vis = Dice(visible_e_n, gt_visible_n)  rendered-space loss (Issue 1 fix)
L     = L_fg + lambda_id * L_id + lambda_vis * L_vis
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from training.phase62.objectives.base import VolumeObjective, VolumeOutputs


def _dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(-2, -1))
    denom = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


class FactorizedFgIdObjective(VolumeObjective):

    def __init__(
        self,
        lambda_id: float = 1.0,
        fg_pos_weight: float = 20.0,
        lambda_vis: float = 0.5,
    ):
        super().__init__()
        self.lambda_id = lambda_id
        self.fg_pos_weight = fg_pos_weight
        self.lambda_vis = lambda_vis

    def forward(
        self,
        outputs: VolumeOutputs,
        V_gt: torch.Tensor,
        gt_visible: Optional[torch.Tensor] = None,
        gt_amodal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fg_logit = outputs.fg_logit[:, 0]   # (B, K, H, W)
        id_logits = outputs.id_logits        # (B, 2, K, H, W)

        # L_fg: BCE with pos_weight for sparse foreground
        Y_fg = (V_gt > 0).float()
        pos_weight = torch.tensor([self.fg_pos_weight], device=fg_logit.device)
        L_fg = F.binary_cross_entropy_with_logits(
            fg_logit, Y_fg, pos_weight=pos_weight, reduction="mean")

        # L_id: CE only where foreground exists
        fg_mask = (V_gt > 0)
        if fg_mask.any():
            Y_id = (V_gt - 1).clamp(min=0).long()
            id_logits_flat = id_logits.permute(0, 2, 3, 4, 1)
            id_at_fg = id_logits_flat[fg_mask]
            Y_id_at_fg = Y_id[fg_mask]
            L_id = F.cross_entropy(id_at_fg, Y_id_at_fg, reduction="mean")
        else:
            L_id = fg_logit.new_zeros(())

        total = L_fg + self.lambda_id * L_id

        # L_vis: rendering-consistent Dice on projected visible output
        # Only active when visible projections are available (after projection step)
        L_vis = fg_logit.new_zeros(())
        if gt_visible is not None and "e0" in outputs.visible and "e1" in outputs.visible:
            vis_e0 = outputs.visible["e0"]
            vis_e1 = outputs.visible["e1"]
            B_vis = min(vis_e0.shape[0], gt_visible.shape[0])
            L_vis = _dice(vis_e0[:B_vis], gt_visible[:B_vis, 0]) + \
                    _dice(vis_e1[:B_vis], gt_visible[:B_vis, 1])
            total = total + self.lambda_vis * L_vis

        return {
            "total": total,
            "L_fg": L_fg.detach(),
            "L_id": L_id.detach(),
            "L_vis": L_vis.detach(),
        }
