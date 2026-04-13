"""
Phase 62 — Losses
==================

Exactly 2 losses:
  1. L_diffusion: standard MSE between predicted and target noise
  2. L_volume_ce: cross-entropy on 3D volume class predictions

No transparency loss, no ownership loss, no depth ordering loss.
The volume CE directly supervises the 3D structure; the diffusion
loss ensures generation quality.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def loss_diffusion(
    noise_pred: torch.Tensor,  # (B, C, T, H, W) or (B, C, H, W)
    noise_gt: torch.Tensor,    # same shape
) -> torch.Tensor:
    """Standard MSE diffusion loss."""
    return F.mse_loss(noise_pred.float(), noise_gt.float())


def loss_volume_ce(
    V_logits: torch.Tensor,                       # (B, C, K, H, W) class logits
    V_gt: torch.Tensor,                            # (B, K, H, W) class indices
    class_weights: Optional[torch.Tensor] = None,  # (C,) optional
) -> torch.Tensor:
    """
    Cross-entropy on 3D volume predictions.

    V_logits: (B, N+1, K, H, W) — per-voxel class logits (N+1 = 3)
    V_gt: (B, K, H, W) — per-voxel class indices (0=bg, 1=entity0, 2=entity1)
    class_weights: optional (N+1,) tensor for class imbalance

    Reshapes to (B*K*H*W, C) vs (B*K*H*W,) for F.cross_entropy.
    """
    B, C, K, H, W = V_logits.shape

    # Reshape: permute to (B, K, H, W, C), flatten to (N, C)
    logits_flat = V_logits.permute(0, 2, 3, 4, 1).reshape(-1, C).float()  # (B*K*H*W, C)
    target_flat = V_gt.reshape(-1).long()  # (B*K*H*W,)

    if class_weights is not None:
        class_weights = class_weights.to(device=logits_flat.device, dtype=torch.float32)
        return F.cross_entropy(logits_flat, target_flat, weight=class_weights)

    return F.cross_entropy(logits_flat, target_flat)


def compute_volume_accuracy(
    V_logits: torch.Tensor,  # (B, C, K, H, W)
    V_gt: torch.Tensor,      # (B, K, H, W)
) -> dict:
    """
    Compute per-class and overall accuracy for volume predictions.
    Returns dict with overall_acc, bg_acc, entity_acc.
    """
    with torch.no_grad():
        pred_class = V_logits.argmax(dim=1)  # (B, K, H, W)
        correct = (pred_class == V_gt.long())

        overall_acc = correct.float().mean().item()

        # Per-class accuracy
        bg_mask = (V_gt == 0)
        entity_mask = (V_gt > 0)

        bg_acc = correct[bg_mask].float().mean().item() if bg_mask.any() else 1.0
        entity_acc = correct[entity_mask].float().mean().item() if entity_mask.any() else 0.0

    return {
        "overall_acc": overall_acc,
        "bg_acc": bg_acc,
        "entity_acc": entity_acc,
    }
