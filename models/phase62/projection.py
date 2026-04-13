"""
Phase 62 — First-Hit Projection
=================================

Scans the 3D volume front-to-back along the depth axis.
The first non-background class encountered wins the pixel.

No max pooling. No weighted average. No transparency.

Training: straight-through estimator (hard argmax forward, soft backward).
Inference: hard argmax.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstHitProjector(nn.Module):
    """
    First-hit projection from 3D volume to 2D class map.

    For each pixel (b, h, w), scans depth bins k=0..K-1:
      - Compute class = argmax(softmax(V_logits[:, :, k, h, w]))
      - First k where class != bg_class wins

    Returns:
        visible_class: (B, H, W) int — hard class index per pixel
        visible_probs: (B, C, H, W) — differentiable softmax probs at first-hit depth
    """

    def __init__(self, n_classes: int = 3, bg_class: int = 0):
        super().__init__()
        self.n_classes = n_classes
        self.bg_class = bg_class

    def forward(
        self,
        V_logits: torch.Tensor,  # (B, C, K, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            V_logits: (B, C, K, H, W) — per-voxel class logits

        Returns:
            visible_class: (B, H, W) int64 — visible class per pixel
            front_probs: (B, C, H, W) float — differentiable first-hit projected probs
            back_probs: (B, C, H, W) float — differentiable behind-front projected probs
        """
        B, C, K, H, W = V_logits.shape
        device = V_logits.device

        # Independent entity presences; no zero-sum competition at voxel level.
        entity_probs = torch.sigmoid(V_logits[:, 1:3].float())  # (B, 2, K, H, W)
        occ = 1.0 - (1.0 - entity_probs[:, 0]) * (1.0 - entity_probs[:, 1])  # (B, K, H, W)

        # Differentiable first-hit projection.
        trans_before = []
        running = torch.ones(B, H, W, device=device, dtype=entity_probs.dtype)
        for k in range(K):
            trans_before.append(running)
            running = running * (1.0 - occ[:, k])
        trans_before = torch.stack(trans_before, dim=1)  # (B, K, H, W)

        front_entity = (entity_probs * trans_before.unsqueeze(1)).sum(dim=2)  # (B, 2, H, W)
        front_bg = running.unsqueeze(1)
        front_probs = torch.cat([front_bg, front_entity], dim=1)
        front_probs = front_probs / front_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Hard class per voxel from independent entity presences.
        entity_max, entity_idx = entity_probs.max(dim=1)  # (B, K, H, W)
        class_per_voxel = torch.where(
            entity_max > 0.5,
            entity_idx + 1,
            torch.zeros_like(entity_idx),
        )

        # First-hit scan: for each (b, h, w), find first k where class != bg
        visible_class = torch.zeros(
            B, H, W, dtype=torch.long, device=device)  # (B, H, W)
        visible_depth = torch.full(
            (B, H, W), K - 1, dtype=torch.long, device=device)  # default: last bin

        for k in range(K):
            cls_k = class_per_voxel[:, k]  # (B, H, W)
            # Update where: visible_class still bg AND cls_k is non-bg
            update_mask = (visible_class == self.bg_class) & (cls_k != self.bg_class)
            visible_class = torch.where(update_mask, cls_k, visible_class)
            visible_depth = torch.where(
                update_mask,
                torch.full_like(visible_depth, k),
                visible_depth,
            )

        # Straight-through front probabilities: hard in forward, soft in backward.
        visible_hard = F.one_hot(visible_class, num_classes=C).permute(0, 3, 1, 2).float()
        front_probs_st = visible_hard - front_probs.detach() + front_probs

        # Behind-front soft projection. Anything that has a non-background object
        # somewhere in front contributes to the back stream.
        has_front_before = 1.0 - trans_before  # (B, K, H, W)
        back_entity = (entity_probs * has_front_before.unsqueeze(1)).sum(dim=2)  # (B, 2, H, W)
        back_bg = torch.zeros(B, 1, H, W, device=device, dtype=entity_probs.dtype)
        back_probs = torch.cat([back_bg, back_entity], dim=1)
        back_sum = back_probs.sum(dim=1, keepdim=True)
        back_probs = torch.where(
            back_sum > 1e-6,
            back_probs / back_sum.clamp(min=1e-6),
            back_probs,
        )

        return visible_class, front_probs_st, back_probs
