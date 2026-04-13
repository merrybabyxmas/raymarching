"""
Center-Offset + Occupancy objective.

For each entity n, predicts per-voxel offset o_n(x) to entity center c_n.
L = sum_{x in Omega_n} ||x + o_n(x) - c_n||_1

Also includes a simple occupancy BCE for foreground detection.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from training.phase62.objectives.base import VolumeObjective, VolumeOutputs


class CenterOffsetObjective(VolumeObjective):

    def __init__(self, lambda_occ: float = 1.0, fg_pos_weight: float = 20.0):
        super().__init__()
        self.lambda_occ = lambda_occ
        self.fg_pos_weight = fg_pos_weight

    def forward(
        self,
        outputs: VolumeOutputs,
        V_gt: torch.Tensor,           # (B, K, H, W)
        gt_visible: Optional[torch.Tensor] = None,
        gt_amodal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fg_logit = outputs.fg_logit[:, 0]  # (B, K, H, W)
        # offset_e0, offset_e1 stored in outputs.amodal as special keys
        offset_e0 = outputs.amodal["offset_e0"]  # (B, 3, K, H, W)
        offset_e1 = outputs.amodal["offset_e1"]  # (B, 3, K, H, W)

        B, K, H, W = V_gt.shape
        device = V_gt.device

        # Occupancy BCE
        Y_fg = (V_gt > 0).float()
        pos_weight = torch.tensor([self.fg_pos_weight], device=device)
        L_occ = F.binary_cross_entropy_with_logits(
            fg_logit, Y_fg, pos_weight=pos_weight, reduction="mean")

        # Build coordinate grid: (K, H, W, 3) normalized to [0, 1]
        coords_k = torch.linspace(0, 1, K, device=device)
        coords_h = torch.linspace(0, 1, H, device=device)
        coords_w = torch.linspace(0, 1, W, device=device)
        grid_k, grid_h, grid_w = torch.meshgrid(coords_k, coords_h, coords_w, indexing="ij")
        coords = torch.stack([grid_k, grid_h, grid_w], dim=-1)  # (K, H, W, 3)
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, K, H, W, 3)

        def _center_loss(offset, entity_mask, coords):
            # offset: (B, 3, K, H, W), entity_mask: (B, K, H, W) bool
            if not entity_mask.any():
                return offset.new_zeros(())

            # Compute GT center per batch element
            coords_masked = coords[entity_mask]   # (N, 3)
            # We need per-batch centers; use scatter
            batch_idx = torch.where(entity_mask)[0]  # (N,)
            center = torch.zeros(B, 3, device=device)
            count = torch.zeros(B, device=device)
            center.scatter_add_(0, batch_idx.unsqueeze(1).expand(-1, 3), coords_masked)
            count.scatter_add_(0, batch_idx, torch.ones(batch_idx.shape[0], device=device))
            count = count.clamp(min=1.0)
            center = center / count.unsqueeze(1)  # (B, 3)

            # c_n broadcast to (B, 3, K, H, W)
            c_n = center[:, :, None, None, None].expand_as(offset)

            # x + o_n(x) - c_n, only at entity voxels
            offset_perm = offset.permute(0, 2, 3, 4, 1)  # (B, K, H, W, 3)
            predicted_center = coords + offset_perm       # (B, K, H, W, 3)
            c_target = center[:, None, None, None, :]     # (B, 1, 1, 1, 3)

            diff = (predicted_center - c_target).abs()     # (B, K, H, W, 3)
            # L1 only at entity voxels
            l1 = diff[entity_mask].sum() / entity_mask.float().sum().clamp(min=1.0)
            return l1

        mask_e0 = (V_gt == 1)
        mask_e1 = (V_gt == 2)
        L_offset_e0 = _center_loss(offset_e0, mask_e0, coords)
        L_offset_e1 = _center_loss(offset_e1, mask_e1, coords)

        total = self.lambda_occ * L_occ + L_offset_e0 + L_offset_e1
        return {
            "total": total,
            "L_occ": L_occ.detach(),
            "L_offset_e0": L_offset_e0.detach(),
            "L_offset_e1": L_offset_e1.detach(),
        }
