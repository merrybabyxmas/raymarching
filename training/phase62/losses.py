"""
Phase 62 — Losses
==================

Main losses used by Phase62.

The volume loss is no longer a class-softmax CE. Instead, entity-0 and
entity-1 are treated as independent voxel presences with BCE-with-logits.
This reduces zero-sum winner-take-all behavior during early topology learning.
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
    V_logits: torch.Tensor,                        # (B, C, K, H, W) logits, use channels 1:3
    V_gt: torch.Tensor,                            # (B, K, H, W) class indices
    class_weights: Optional[torch.Tensor] = None,
    voxel_weights: Optional[torch.Tensor] = None,
    entity_pos_weight: float = 50.0,
) -> torch.Tensor:
    """
    Independent BCE-with-logits on entity voxel presences.

    CRITICAL: Entity voxels are ~1% of total volume. Without strong
    positive weighting, the model learns "predict all bg" trivially.
    entity_pos_weight upweights positive (entity-present) voxels.
    """
    logits_e = V_logits[:, 1:3].float()  # (B, 2, K, H, W)
    target_e0 = (V_gt == 1).float()
    target_e1 = (V_gt == 2).float()
    targets = torch.stack([target_e0, target_e1], dim=1)  # (B,2,K,H,W)

    # Split into bg-region loss and entity-region loss, then combine.
    # This prevents entity signal from being drowned in the 99% bg voxels.
    entity_mask = (targets > 0.5)  # where entities should be present
    bg_mask = ~entity_mask

    bce_all = F.binary_cross_entropy_with_logits(
        logits_e, targets, reduction="none")

    # Entity-region loss: averaged only over entity voxels
    n_entity = entity_mask.float().sum().clamp(min=1.0)
    l_entity = (bce_all * entity_mask.float()).sum() / n_entity

    # BG-region loss: averaged only over bg voxels (much weaker weight)
    n_bg = bg_mask.float().sum().clamp(min=1.0)
    l_bg = (bce_all * bg_mask.float()).sum() / n_bg

    # Entity loss dominates — this is the key to breaking all-bg collapse
    return entity_pos_weight * l_entity + l_bg


def loss_projected_global(
    front_probs: torch.Tensor,      # (B, C, H, W)
    gt_visible: torch.Tensor,       # (B, 2, H, W)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Differentiable global projection loss on the front-hit 2D projection.

    Optimizes projected entity occupancy directly in image space so that
    both entities must retain visible area after projection.
    """
    pred = front_probs[:, 1:3].float()  # entity channels only
    gt = gt_visible.float()

    inter = (pred * gt).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + gt.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return (1.0 - iou).mean()


def loss_projected_balance(
    front_probs: torch.Tensor,      # (B, C, H, W)
    gt_visible: torch.Tensor,       # (B, 2, H, W)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Penalize asymmetric collapse: loss = 1 - min(IoU_e0, IoU_e1).

    If EITHER entity dies (IoU→0), loss→1 regardless of the other.
    Forces the model to keep BOTH entities alive simultaneously.
    """
    pred = front_probs[:, 1:3].float()  # (B, 2, H, W)
    gt = gt_visible.float()  # (B, 2, H, W)

    # Per-entity IoU
    inter = (pred * gt).sum(dim=(2, 3))  # (B, 2)
    union = pred.sum(dim=(2, 3)) + gt.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)  # (B, 2)

    # min-IoU: penalizes the weaker entity
    min_iou = iou.min(dim=1).values  # (B,)
    return (1.0 - min_iou).mean()


def compute_volume_accuracy(
    V_logits: torch.Tensor,  # (B, C, K, H, W)
    V_gt: torch.Tensor,      # (B, K, H, W)
) -> dict:
    """
    Compute per-class and overall accuracy for volume predictions.
    Returns dict with overall_acc, bg_acc, entity_acc.
    """
    with torch.no_grad():
        p_e0 = torch.sigmoid(V_logits[:, 1].float())
        p_e1 = torch.sigmoid(V_logits[:, 2].float())
        pred_class = torch.zeros_like(V_gt.long())
        has_entity = (p_e0 > 0.5) | (p_e1 > 0.5)
        pred_class = torch.where(has_entity & (p_e0 >= p_e1), torch.ones_like(pred_class), pred_class)
        pred_class = torch.where(has_entity & (p_e1 > p_e0), torch.full_like(pred_class, 2), pred_class)
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
