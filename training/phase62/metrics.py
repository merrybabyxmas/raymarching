"""
Phase 62 — Evaluation Metrics
===============================

Metrics for projected 2D class maps and 3D volume predictions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def compute_projected_class_iou(
    visible_class: torch.Tensor,    # (B, H, W) int64 — predicted visible class
    gt_visible_mask: torch.Tensor,  # (B, H_gt, W_gt) or (B, S) float — GT entity mask
    entity_idx: int,                # which entity class to compute IoU for (1 or 2)
    spatial_h: int = 16,
    spatial_w: int = 16,
) -> float:
    """
    IoU between projected visible class and GT visible mask for a specific entity.

    Args:
        visible_class: (B, H, W) — predicted class index per pixel
        gt_visible_mask: (B, S) or (B, H_gt, W_gt) — GT binary mask (float 0/1)
        entity_idx: class index to evaluate (1=entity0, 2=entity1)
        spatial_h, spatial_w: target spatial resolution

    Returns:
        IoU score (float), averaged over batch.
    """
    with torch.no_grad():
        B = visible_class.shape[0]
        H, W = visible_class.shape[1], visible_class.shape[2]

        # Predicted binary mask for this entity
        pred_mask = (visible_class == entity_idx).float()  # (B, H, W)

        # Process GT mask
        gt = gt_visible_mask.float()
        if gt.dim() == 2:
            # (B, S) -> (B, 1, H_gt, W_gt) -> resize to (B, 1, H, W)
            S = gt.shape[1]
            H_gt = int(round(S ** 0.5))
            gt = gt.reshape(B, 1, H_gt, H_gt)
        elif gt.dim() == 3:
            gt = gt.unsqueeze(1)  # (B, 1, H_gt, W_gt)

        if gt.shape[2] != H or gt.shape[3] != W:
            gt = F.interpolate(gt, size=(H, W), mode='bilinear', align_corners=False)
        gt = (gt.squeeze(1) > 0.5).float()  # (B, H, W)

        # IoU
        intersection = (pred_mask * gt).sum(dim=(1, 2))  # (B,)
        union = ((pred_mask + gt) > 0).float().sum(dim=(1, 2))  # (B,)

        iou_per_sample = intersection / (union + 1e-8)  # (B,)
        return float(iou_per_sample.mean().item())


def compute_entity_accuracy(
    V_logits: torch.Tensor,  # (B, C, K, H, W)
    V_gt: torch.Tensor,      # (B, K, H, W)
) -> float:
    """
    Accuracy on entity voxels only (excluding background).

    Evaluates how well the model predicts entity classes where entities
    actually exist, ignoring the dominant background class.

    Returns:
        Accuracy (float) on non-background voxels, or 0.0 if no entities.
    """
    with torch.no_grad():
        pred_class = V_logits.argmax(dim=1)  # (B, K, H, W)
        entity_mask = (V_gt > 0)

        if not entity_mask.any():
            return 0.0

        correct = (pred_class == V_gt.long())
        return float(correct[entity_mask].float().mean().item())


def compute_class_distribution(
    visible_class: torch.Tensor,  # (B, H, W) int64
    n_classes: int = 3,
) -> dict:
    """
    Compute class distribution in projected 2D map.

    Returns dict with fraction of pixels per class.
    Useful for monitoring volume collapse (all bg or all entity).
    """
    with torch.no_grad():
        total = visible_class.numel()
        dist = {}
        for c in range(n_classes):
            count = (visible_class == c).sum().item()
            dist[f"class_{c}_frac"] = count / max(total, 1)
        return dist
