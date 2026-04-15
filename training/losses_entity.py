"""
Phase 63 — Entity-focused Loss Library
========================================

Seven loss functions aligned with the Phase 63 design objective: *both
entities must be individually identifiable throughout the sequence, even
through occlusion and collision*.

All functions operate on float32 tensors.  Masks are expected in [0, 1].

Functions
---------
loss_visible_dice                  Dice on first-hit visible masks
loss_amodal_dice                   Dice on amodal (full-body) masks
loss_identity_separation           Contrastive-triplet on pooled appearance
loss_temporal_slot_consistency     Appearance consistency across frames
loss_occlusion_consistency         Enforce V_i <= A_i (visible subset of amodal)
loss_isolation_consistency         L1 between isolated render and solo frame
loss_entity_survival               Hinge on visible mean — prevents collapse

compute_entity_metrics             Convenience — IoU / survival metrics
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Try to import kornia; fall back to a minimal in-house dice if unavailable
# ---------------------------------------------------------------------------
try:
    import kornia.losses as _kornia_losses                                     # noqa: F401
    _HAS_KORNIA = True
except Exception:                                                              # pragma: no cover
    _HAS_KORNIA = False


def _dice_loss_binary(
    pred: torch.Tensor,           # (B, H, W) or (B, 1, H, W) in [0, 1]
    target: torch.Tensor,         # same shape, values in {0, 1} or [0, 1]
    eps: float = 1.0,
) -> torch.Tensor:
    """Soft Dice for a single binary channel.  Reduces to a scalar.

    If kornia is available we still compute Dice directly — kornia's
    ``dice_loss`` assumes multi-class one-hot targets, which is a poor fit
    for our independent per-entity masks.  We keep the in-house implementation
    for consistency.
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    pred = pred.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)

    # Flatten per-sample.
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# 1 — Visible Dice
# ---------------------------------------------------------------------------
def loss_visible_dice(
    visible_e0: torch.Tensor,     # (B, H, W) predicted visible
    visible_e1: torch.Tensor,
    gt_visible_e0: torch.Tensor,  # (B, H, W) GT visible mask in {0,1}
    gt_visible_e1: torch.Tensor,
) -> torch.Tensor:
    """Dice loss between predicted visible and GT visible, averaged over
    entities."""
    return 0.5 * (
        _dice_loss_binary(visible_e0, gt_visible_e0)
        + _dice_loss_binary(visible_e1, gt_visible_e1)
    )


# ---------------------------------------------------------------------------
# 2 — Amodal Dice
# ---------------------------------------------------------------------------
def loss_amodal_dice(
    amodal_e0: torch.Tensor,      # (B, H, W) predicted amodal
    amodal_e1: torch.Tensor,
    gt_amodal_e0: torch.Tensor,
    gt_amodal_e1: torch.Tensor,
) -> torch.Tensor:
    """Dice loss on amodal (full-body) masks, averaged over entities."""
    return 0.5 * (
        _dice_loss_binary(amodal_e0, gt_amodal_e0)
        + _dice_loss_binary(amodal_e1, gt_amodal_e1)
    )


# ---------------------------------------------------------------------------
# 3 — Identity Separation (contrastive / triplet)
# ---------------------------------------------------------------------------
def loss_identity_separation(
    h_e0: torch.Tensor,           # (B, D) pooled entity-0 feature
    h_e1: torch.Tensor,           # (B, D) pooled entity-1 feature
    identity_e0: torch.Tensor,    # (B, D) reference identity embedding for e0
    identity_e1: torch.Tensor,    # (B, D) reference identity embedding for e1
    margin: float = 1.0,
) -> torch.Tensor:
    """
    L_id = ||h_e0 - id_e0||^2 + ||h_e1 - id_e1||^2
         + max(0, margin - ||h_e0 - h_e1||)

    Pulls each pooled entity feature towards its reference identity and
    pushes the two pooled features apart by at least ``margin``.
    """
    # Cast to float32 for numerical stability.
    h0, h1 = h_e0.float(), h_e1.float()
    id0, id1 = identity_e0.float(), identity_e1.float()

    pull_0 = F.mse_loss(h0, id0)
    pull_1 = F.mse_loss(h1, id1)
    # Euclidean distance between entities, averaged over batch.
    push_dist = (h0 - h1).norm(dim=-1).mean()
    push = F.relu(margin - push_dist)
    return pull_0 + pull_1 + push


def pool_entity_feature(
    appearance: torch.Tensor,     # (B, app_dim, H, W)
    density_2d: torch.Tensor,     # (B, H, W) (use amodal as weights — "where is the entity?")
    eps: float = 1e-6,
) -> torch.Tensor:
    """Spatially mass-weighted pool of an appearance feature map.

    Returns (B, app_dim).  Used to build the ``h_ei`` inputs of
    ``loss_identity_separation``.
    """
    w = density_2d.float().unsqueeze(1).clamp(min=0.0)                 # (B, 1, H, W)
    num = (appearance.float() * w).sum(dim=(2, 3))                     # (B, app_dim)
    den = w.sum(dim=(2, 3)).clamp(min=eps)                             # (B, 1)
    return num / den


# ---------------------------------------------------------------------------
# 4 — Temporal Slot Consistency (contrastive across frames)
# ---------------------------------------------------------------------------
def loss_temporal_slot_consistency(
    density_frames: List[torch.Tensor],       # list of (B, 2, K, H, W)
    appearance_frames: List[torch.Tensor],    # list of (B, 2, app_dim, H, W)
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Same entity across different frames should have similar pooled appearance;
    different entities (within a frame) should differ by at least ``margin``.

    Implemented as an InfoNCE-style pairwise contrast over frame pairs.

    density_frames[t]: (B, 2, K, H, W)         # per-frame independent densities
    appearance_frames[t]: (B, 2, app_dim, H, W) — per-frame, per-entity 2D app
    """
    if len(density_frames) < 2:
        # Need at least two frames to measure temporal consistency.
        return density_frames[0].new_zeros(())

    # Pool each frame / entity via amodal-style weights (sum over K).
    pooled: List[torch.Tensor] = []                                     # each (B, 2, app_dim)
    for dens, app in zip(density_frames, appearance_frames):
        # dens: (B, 2, K, H, W) -> weight (B, 2, H, W) via amodal-ish reduce.
        w = 1.0 - (1.0 - dens.float().clamp(0, 1)).prod(dim=2)          # (B, 2, H, W)
        w = w.unsqueeze(2)                                              # (B, 2, 1, H, W)
        num = (app.float() * w).sum(dim=(3, 4))                         # (B, 2, app_dim)
        den = w.sum(dim=(3, 4)).clamp(min=1e-6)                         # (B, 2, 1)
        pooled.append(num / den)

    pooled = torch.stack(pooled, dim=0)                                 # (T, B, 2, app_dim)
    T = pooled.shape[0]

    # Pull: same entity across frames close.
    # Average pairwise MSE over (t, t') with t<t'.
    pull = pooled.new_zeros(())
    n_pairs = 0
    for t in range(T):
        for tp in range(t + 1, T):
            pull = pull + F.mse_loss(pooled[t], pooled[tp])
            n_pairs += 1
    if n_pairs > 0:
        pull = pull / n_pairs

    # Push: different entities within same frame far (triplet margin).
    # dist_{e0, e1} averaged across frames and batch.
    diffs = (pooled[:, :, 0] - pooled[:, :, 1]).norm(dim=-1)            # (T, B)
    push = F.relu(margin - diffs).mean()

    return pull + push


# ---------------------------------------------------------------------------
# 5 — Occlusion Consistency  (V_i <= A_i)
# ---------------------------------------------------------------------------
def loss_occlusion_consistency(
    visible_e0: torch.Tensor,
    visible_e1: torch.Tensor,
    amodal_e0: torch.Tensor,
    amodal_e1: torch.Tensor,
) -> torch.Tensor:
    """Soft hinge: penalise any pixel where V_i > A_i.

    By construction of the renderer, V_i <= A_i should hold to floating-point
    precision; this loss acts as a safety net when ground-truth-supervised
    training pulls V and A from different signals.
    """
    v0 = visible_e0.float()
    v1 = visible_e1.float()
    a0 = amodal_e0.float()
    a1 = amodal_e1.float()
    excess_0 = F.relu(v0 - a0)
    excess_1 = F.relu(v1 - a1)
    return 0.5 * (excess_0.mean() + excess_1.mean())


# ---------------------------------------------------------------------------
# 6 — Isolation Consistency
# ---------------------------------------------------------------------------
def loss_isolation_consistency(
    isolated_render_e0: torch.Tensor,   # (B, C, H, W) render with only e0 present
    isolated_render_e1: torch.Tensor,
    solo_frame_e0: torch.Tensor,        # (B, C, H, W) GT frame with only e0
    solo_frame_e1: torch.Tensor,
) -> torch.Tensor:
    """L1 between rendered isolated-entity frames and their GT solo frames.

    Either argument may be ``None`` if the solo frame is unavailable for a
    given batch — the caller should filter before calling.
    """
    return 0.5 * (
        F.l1_loss(isolated_render_e0.float(), solo_frame_e0.float())
        + F.l1_loss(isolated_render_e1.float(), solo_frame_e1.float())
    )


# ---------------------------------------------------------------------------
# 7 — Entity Survival (anti-collapse hinge)
# ---------------------------------------------------------------------------
def loss_entity_survival(
    visible_e0: torch.Tensor,     # (B, H, W)
    visible_e1: torch.Tensor,
    min_survival: float = 0.02,
) -> torch.Tensor:
    """Hinge that penalises any batch element where the per-image mean visible
    area of an entity drops below ``min_survival`` (default 2% of the frame).

    This is the *direct* counter-measure to "single-entity collapse": the
    optimiser pays a price whenever one entity is effectively rendered empty.
    """
    mean_0 = visible_e0.float().mean(dim=(1, 2))     # (B,)
    mean_1 = visible_e1.float().mean(dim=(1, 2))     # (B,)
    hinge_0 = F.relu(min_survival - mean_0)          # positive iff e0 is dying
    hinge_1 = F.relu(min_survival - mean_1)
    return 0.5 * (hinge_0.mean() + hinge_1.mean())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def _iou(pred: torch.Tensor, gt: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """Mean IoU for a binarised pred against a binary GT.  Returns scalar."""
    p = (pred.float() > thresh).float()
    g = (gt.float() > 0.5).float()
    inter = (p * g).sum(dim=(-1, -2))
    union = (p + g - p * g).sum(dim=(-1, -2)).clamp(min=1e-6)
    return (inter / union).mean()


@torch.no_grad()
def compute_entity_metrics(
    visible_e0: torch.Tensor,
    visible_e1: torch.Tensor,
    amodal_e0: torch.Tensor,
    amodal_e1: torch.Tensor,
    gt_visible: torch.Tensor,          # (B, 2, H, W)
    gt_amodal: torch.Tensor,           # (B, 2, H, W)
) -> Dict[str, float]:
    """Convenience dict of per-entity and min-over-entities IoU metrics."""
    visible_iou_e0 = _iou(visible_e0, gt_visible[:, 0]).item()
    visible_iou_e1 = _iou(visible_e1, gt_visible[:, 1]).item()
    amodal_iou_e0 = _iou(amodal_e0, gt_amodal[:, 0]).item()
    amodal_iou_e1 = _iou(amodal_e1, gt_amodal[:, 1]).item()

    return {
        "visible_iou_e0": visible_iou_e0,
        "visible_iou_e1": visible_iou_e1,
        "visible_iou_min": min(visible_iou_e0, visible_iou_e1),
        "amodal_iou_e0": amodal_iou_e0,
        "amodal_iou_e1": amodal_iou_e1,
        "amodal_iou_min": min(amodal_iou_e0, amodal_iou_e1),
        "visible_mean_e0": visible_e0.float().mean().item(),
        "visible_mean_e1": visible_e1.float().mean().item(),
    }


__all__ = [
    "loss_visible_dice",
    "loss_amodal_dice",
    "loss_identity_separation",
    "loss_temporal_slot_consistency",
    "loss_occlusion_consistency",
    "loss_isolation_consistency",
    "loss_entity_survival",
    "compute_entity_metrics",
    "pool_entity_feature",
]
