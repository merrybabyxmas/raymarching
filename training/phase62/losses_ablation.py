"""
Phase 62 — Ablation-Only Losses
=================================

Experimental losses that are NOT part of the mainline training.
Only used when explicitly enabled via ablation config.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(-2, -1))
    denom = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def loss_amodal_dice(
    amodal_e0: torch.Tensor,
    amodal_e1: torch.Tensor,
    gt_amodal: torch.Tensor,
    spatial_h: int = 16,
    spatial_w: int = 16,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-entity amodal Dice: forces full-body presence in 3D volume."""
    B = amodal_e0.shape[0]
    H, W = amodal_e0.shape[1], amodal_e0.shape[2]
    if gt_amodal.dim() == 3 and gt_amodal.shape[-1] != H:
        S = gt_amodal.shape[-1]
        Hm = int(S ** 0.5)
        gt_2d = gt_amodal.float().reshape(B, 2, Hm, Hm)
        gt_2d = F.interpolate(gt_2d, size=(H, W), mode='nearest')
    elif gt_amodal.dim() == 4:
        gt_2d = gt_amodal.float()
    else:
        gt_2d = gt_amodal.float().reshape(B, 2, H, W)
    return _dice(amodal_e0, gt_2d[:, 0]) + _dice(amodal_e1, gt_2d[:, 1])


def loss_visible_dice(
    visible_e0: torch.Tensor,
    visible_e1: torch.Tensor,
    gt_visible: torch.Tensor,
    spatial_h: int = 16,
    spatial_w: int = 16,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-entity visible Dice: forces correct visible ownership."""
    B = visible_e0.shape[0]
    H, W = visible_e0.shape[1], visible_e0.shape[2]
    if gt_visible.dim() == 3 and gt_visible.shape[-1] != H:
        S = gt_visible.shape[-1]
        Hm = int(S ** 0.5)
        gt_2d = gt_visible.float().reshape(B, 2, Hm, Hm)
        gt_2d = F.interpolate(gt_2d, size=(H, W), mode='nearest')
    elif gt_visible.dim() == 4:
        gt_2d = gt_visible.float()
    else:
        gt_2d = gt_visible.float().reshape(B, 2, H, W)
    return _dice(visible_e0, gt_2d[:, 0]) + _dice(visible_e1, gt_2d[:, 1])


def loss_voxel_exclusive(
    entity_probs: torch.Tensor,    # (B, 2, K, H, W)
) -> torch.Tensor:
    """Penalize same-voxel co-activation: E[p0 * p1]."""
    return (entity_probs[:, 0] * entity_probs[:, 1]).mean()
