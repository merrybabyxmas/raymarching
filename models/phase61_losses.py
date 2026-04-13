"""
Phase 61 — Loss Functions for Depth-Layered Volume Diffusion
==============================================================

4 losses operating on DepthVolumeHead + VolumeCompositor outputs:

  L_composite        : MSE(noise_pred, noise_gt) — scene-level diffusion quality
  L_alpha_volume     : BCE on per-bin alpha predictions vs depth-bin targets
  L_visible_ownership: BCE on summed rendering weights vs GT visible masks
  L_depth_expected   : expected-depth ordering margin loss in overlap regions
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def loss_composite(
    noise_pred: torch.Tensor,  # (1, 4, T, H, W)
    noise_gt:   torch.Tensor,  # (1, 4, T, H, W)
) -> torch.Tensor:
    """Standard MSE on final noise prediction."""
    return F.mse_loss(noise_pred.float(), noise_gt.float())


def loss_alpha_volume(
    alpha0_logits: torch.Tensor,  # (B, S, K) raw logits from DepthVolumeHead
    alpha1_logits: torch.Tensor,  # (B, S, K) raw logits
    tgt0_bins:     torch.Tensor,  # (B, S, K) target
    tgt1_bins:     torch.Tensor,  # (B, S, K) target
    valid0:        torch.Tensor = None,  # (B, S, K) optional mask for valid bins
    valid1:        torch.Tensor = None,
    eps:           float = 1e-6,
) -> torch.Tensor:
    """
    BCE with logits on per-bin alpha predictions vs depth-bin targets.

    Uses binary_cross_entropy_with_logits for numerical stability.
    Optional valid masks allow ignoring bins where GT is ambiguous.
    """
    l0 = alpha0_logits.float()
    l1 = alpha1_logits.float()
    t0 = tgt0_bins.float()
    t1 = tgt1_bins.float()

    if valid0 is not None and valid1 is not None:
        v0 = valid0.float()
        v1 = valid1.float()
        bce0 = (F.binary_cross_entropy_with_logits(l0, t0, reduction="none") * v0).sum() / v0.sum().clamp(min=1.0)
        bce1 = (F.binary_cross_entropy_with_logits(l1, t1, reduction="none") * v1).sum() / v1.sum().clamp(min=1.0)
    else:
        bce0 = F.binary_cross_entropy_with_logits(l0, t0, reduction="mean")
        bce1 = F.binary_cross_entropy_with_logits(l1, t1, reduction="mean")

    return 0.5 * (bce0 + bce1)


def loss_visible_ownership(
    w0_bins:          torch.Tensor,  # (B, S, K) rendering weights for entity 0
    w1_bins:          torch.Tensor,  # (B, S, K) rendering weights for entity 1
    visible_masks_BS: torch.Tensor,  # (B, 2, S) GT visible masks
    eps:              float = 1e-6,
) -> torch.Tensor:
    """
    Sum rendering weights across depth bins → visible weight per entity.
    BCE vs GT visible masks.

    visible_weight_e0 = w0_bins.sum(dim=2)  → (B, S)
    This should match the GT visible mask for entity 0.
    """
    v0 = visible_masks_BS[:, 0, :].float()  # (B, S)
    v1 = visible_masks_BS[:, 1, :].float()  # (B, S)

    vis_w0 = w0_bins.float().sum(dim=2).clamp(eps, 1.0 - eps)  # (B, S)
    vis_w1 = w1_bins.float().sum(dim=2).clamp(eps, 1.0 - eps)  # (B, S)

    bce0 = F.binary_cross_entropy(vis_w0, v0, reduction="mean")
    bce1 = F.binary_cross_entropy(vis_w1, v1, reduction="mean")

    return 0.5 * (bce0 + bce1)


def loss_depth_expected(
    alpha0_bins:     torch.Tensor,  # (B, S, K) predicted alpha per bin
    alpha1_bins:     torch.Tensor,  # (B, S, K) predicted alpha per bin
    depth_orders:    list,          # [(front_idx, back_idx), ...] per frame
    entity_masks_BS: torch.Tensor,  # (B, 2, S) GT entity masks
    margin:          float = 0.3,
) -> torch.Tensor:
    """
    Expected depth from alpha-weighted bin index.

    d_i = sum(alpha_i(z) * z) / sum(alpha_i(z))

    In overlap regions: front entity expected depth < back entity expected depth.
    Hinge loss: relu(d_front - d_back + margin).

    Per-frame depth ordering from depth_orders.
    """
    B, S, K = alpha0_bins.shape
    device = alpha0_bins.device

    # Apply sigmoid since inputs are now raw logits
    a0 = torch.sigmoid(alpha0_bins.float())  # (B, S, K)
    a1 = torch.sigmoid(alpha1_bins.float())  # (B, S, K)

    # Bin indices: [0, 1, ..., K-1]
    bin_idx = torch.arange(K, device=device, dtype=torch.float32)  # (K,)

    # Expected depth: sum(alpha * z) / sum(alpha)
    # a0: (B, S, K), bin_idx: (K,) → weighted sum over K
    eps = 1e-6
    d0 = (a0 * bin_idx).sum(dim=2) / (a0.sum(dim=2) + eps)  # (B, S)
    d1 = (a1 * bin_idx).sum(dim=2) / (a1.sum(dim=2) + eps)  # (B, S)

    m0 = entity_masks_BS[:, 0, :].float()  # (B, S)
    m1 = entity_masks_BS[:, 1, :].float()  # (B, S)
    overlap = (m0 > 0.5).float() * (m1 > 0.5).float()  # (B, S)

    if overlap.sum() < 1.0:
        return torch.tensor(0.0, device=device)

    total_violation = torch.tensor(0.0, device=device)
    n_total = torch.tensor(0.0, device=device)

    for b in range(B):
        ov_b = overlap[b]  # (S,)
        if ov_b.sum() < 1.0:
            continue

        fi = min(b, len(depth_orders) - 1) if depth_orders else 0
        if fi >= len(depth_orders):
            front = 0
        else:
            front = int(depth_orders[fi][0])

        if front == 0:
            # Entity 0 in front → d0 should be < d1
            viol = F.relu(d0[b] - d1[b] + margin)  # (S,)
        else:
            # Entity 1 in front → d1 should be < d0
            viol = F.relu(d1[b] - d0[b] + margin)  # (S,)

        total_violation = total_violation + (viol * ov_b).sum()
        n_total = n_total + ov_b.sum()

    return total_violation / n_total.clamp(min=1.0)
