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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            V_logits: (B, C, K, H, W) — per-voxel class logits

        Returns:
            visible_class: (B, H, W) int64 — visible class per pixel
            visible_probs: (B, C, H, W) float — differentiable class probs at first-hit depth
        """
        B, C, K, H, W = V_logits.shape
        device = V_logits.device

        # Softmax over class dimension at each voxel
        probs = torch.softmax(V_logits.float(), dim=1)  # (B, C, K, H, W)

        # Hard class per voxel: argmax over C
        class_per_voxel = probs.argmax(dim=1)  # (B, K, H, W)

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

        # Differentiable path: gather softmax probs at first-hit depth
        # depth_idx: (B, H, W) -> expand to (B, C, 1, H, W) for gather on dim=2
        depth_idx = visible_depth.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H, W)
        depth_idx = depth_idx.expand(-1, C, -1, -1, -1)     # (B, C, 1, H, W)
        visible_probs = probs.gather(dim=2, index=depth_idx).squeeze(2)  # (B, C, H, W)

        # Straight-through: hard class in forward, soft probs carry gradients backward
        # The gradient flows through visible_probs (softmax at first-hit depth)
        # visible_class is used for discrete operations (embedding lookup etc.)

        return visible_class, visible_probs
