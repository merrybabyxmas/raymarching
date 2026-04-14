"""
Phase 62 — Mainline Losses
============================

Only production-validated losses live here.
Experimental losses are in losses_ablation.py.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def loss_diffusion(
    noise_pred: torch.Tensor,
    noise_gt: torch.Tensor,
) -> torch.Tensor:
    """Standard MSE diffusion loss."""
    return F.mse_loss(noise_pred.float(), noise_gt.float())


def loss_volume_ce(
    V_logits: torch.Tensor,                        # (B, C, K, H, W)
    V_gt: torch.Tensor,                            # (B, K, H, W)
    class_weights: Optional[torch.Tensor] = None,
    voxel_weights: Optional[torch.Tensor] = None,
    entity_pos_weight: float = 50.0,
) -> torch.Tensor:
    """
    Independent BCE-with-logits on entity voxel presences.

    Entity voxels are ~1% of total volume. Without strong positive
    weighting, the model learns "predict all bg" trivially.
    """
    logits_e = V_logits[:, 1:3].float()  # (B, 2, K, H, W)
    target_e0 = (V_gt == 1).float()
    target_e1 = (V_gt == 2).float()
    targets = torch.stack([target_e0, target_e1], dim=1)

    logits_e0 = logits_e[:, 0]
    logits_e1 = logits_e[:, 1]
    tgt_e0 = targets[:, 0]
    tgt_e1 = targets[:, 1]

    def _entity_loss(logits, tgt):
        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
        bce = bce.clamp(max=20.0)
        pos_mask = (tgt > 0.5)
        neg_mask = ~pos_mask
        n_pos = pos_mask.float().sum().clamp(min=1.0)
        n_neg = neg_mask.float().sum().clamp(min=1.0)
        l_pos = (bce * pos_mask.float()).sum() / n_pos
        l_neg = (bce * neg_mask.float()).sum() / n_neg
        return entity_pos_weight * l_pos + l_neg

    l_e0 = _entity_loss(logits_e0, tgt_e0)
    l_e1 = _entity_loss(logits_e1, tgt_e1)

    with torch.no_grad():
        ratio = (l_e0 / (l_e1 + 1e-6)).clamp(0.8, 1.25)
        w0 = ratio / (ratio + 1.0)
        w1 = 1.0 - w0
    return w0 * l_e0 + w1 * l_e1


def loss_projected_global(
    front_probs: torch.Tensor,      # (B, C, H, W)
    gt_visible: torch.Tensor,       # (B, 2, H, W)
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable global projection loss on front-hit 2D projection."""
    pred = front_probs[:, 1:3].float()
    gt = gt_visible.float()
    inter = (pred * gt).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + gt.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return (1.0 - iou).mean()


def loss_min_iou_balance(
    front_probs: torch.Tensor,      # (B, C, H, W)
    gt_visible: torch.Tensor,       # (B, 2, H, W)
    eps: float = 1e-6,
) -> torch.Tensor:
    """Quadratic min-IoU balance: ((1 - min_iou)^2).mean()"""
    pred = front_probs[:, 1:3].float()
    gt = gt_visible.float()
    inter = (pred * gt).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + gt.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    min_iou = iou.min(dim=1).values
    return ((1.0 - min_iou) ** 2).mean()


def loss_projected_balance(
    front_probs: torch.Tensor,
    gt_visible: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Legacy — kept for backward compat but now returns 0."""
    return front_probs.new_zeros(())


def loss_feature_separation(
    F_0: torch.Tensor,  # (B, S, D)
    F_1: torch.Tensor,  # (B, S, D)
) -> torch.Tensor:
    """
    Push F_0 and F_1 feature representations apart.

    Minimizes cosine similarity between per-pixel feature vectors.
    If F_0 and F_1 are already orthogonal (cos_sim=0), loss is 0.
    If identical (cos_sim=1), loss is 1.
    """
    f0 = F.normalize(F_0.float(), dim=-1, eps=1e-6)
    f1 = F.normalize(F_1.float(), dim=-1, eps=1e-6)
    cos_sim = (f0 * f1).sum(dim=-1)  # (B, S)
    return cos_sim.clamp(min=0.0).mean()


def loss_depth_compactness(
    entity_probs: torch.Tensor,  # (B, 2, K, H, W)
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Encourage entity_probs to be localised in a few depth bins (compact blob).

    Minimises the entropy of depth-wise activation mass per entity.
    Entropy=0 → all mass in one slice (perfect blob).
    Entropy=log(K) → uniform slab (worst case).

    Contract stage1 pass requires compact ≥ 0.20
    (1 - normalised_entropy ≥ 0.20, i.e. normalised_entropy ≤ 0.80).
    """
    B, _, K, H, W = entity_probs.shape
    # depth-wise mean activation mass per entity: (B, 2, K)
    depth_mass = entity_probs.float().mean(dim=(3, 4))
    # normalise to probability distribution
    depth_mass_sum = depth_mass.sum(dim=2, keepdim=True).clamp(min=eps)
    p = (depth_mass / depth_mass_sum).clamp(min=eps)
    # Shannon entropy, normalised by log(K)
    entropy = -(p * p.log()).sum(dim=2)          # (B, 2)
    normalised_entropy = entropy / (math.log(K) + eps)
    # Penalise high entropy (diffuse slab)
    return normalised_entropy.mean()


def loss_rendered_dice(
    visible_e0: torch.Tensor,  # (B, H, W)
    visible_e1: torch.Tensor,  # (B, H, W)
    gt_visible: torch.Tensor,  # (B, 2, H, W)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Rendering-consistent loss: Dice on the RENDERED 2D visible output.

    Unlike per-voxel BCE, this loss matches the actual rendering math
    (transmittance compositing), so gradients align with what we see.
    """
    def _dice(pred, target):
        inter = (pred * target).sum(dim=(-2, -1))
        denom = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()

    return _dice(visible_e0, gt_visible[:, 0]) + _dice(visible_e1, gt_visible[:, 1])


def compute_volume_accuracy(
    V_logits: torch.Tensor,  # (B, C, K, H, W)
    V_gt: torch.Tensor,      # (B, K, H, W)
) -> dict:
    """Per-class and overall accuracy for volume predictions."""
    with torch.no_grad():
        p_e0 = torch.sigmoid(V_logits[:, 1].float())
        p_e1 = torch.sigmoid(V_logits[:, 2].float())
        pred_class = torch.zeros_like(V_gt.long())
        has_entity = (p_e0 > 0.5) | (p_e1 > 0.5)
        pred_class = torch.where(has_entity & (p_e0 >= p_e1), torch.ones_like(pred_class), pred_class)
        pred_class = torch.where(has_entity & (p_e1 > p_e0), torch.full_like(pred_class, 2), pred_class)
        correct = (pred_class == V_gt.long())

        overall_acc = correct.float().mean().item()
        bg_mask = (V_gt == 0)
        entity_mask = (V_gt > 0)
        bg_acc = correct[bg_mask].float().mean().item() if bg_mask.any() else 1.0
        entity_acc = correct[entity_mask].float().mean().item() if entity_mask.any() else 0.0

    return {
        "overall_acc": overall_acc,
        "bg_acc": bg_acc,
        "entity_acc": entity_acc,
    }
