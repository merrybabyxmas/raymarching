"""
Phase 60 — Loss Functions for Depth-Ordered Layered Video Diffusion
====================================================================

All losses operate on predictions from the Phase60Processor primary block
and/or the composited UNet output.

Loss overview:
  L_composite       : MSE(noise_composite, noise_gt) — scene-level diffusion quality
  L_alpha           : BCE + Dice on predicted alpha vs GT entity masks (occupancy)
  L_ownership       : BCE on predicted ownership vs GT visible masks
  L_depth_order     : hinge loss — front entity depth < back entity depth in overlap
  L_entity_visible  : weighted MSE on entity features in visible regions
  L_leak            : entity features should be near-zero outside own alpha region
  L_temporal        : consecutive-frame alpha/ownership smoothness
  L_identity_solo   : solo render denoising anchor for entity identity
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Composite noise loss
# =============================================================================

def loss_composite(
    noise_composite: torch.Tensor,  # (1, 4, T, H, W)
    noise_gt:        torch.Tensor,  # (1, 4, T, H, W)
) -> torch.Tensor:
    """MSE between composited noise prediction and GT noise."""
    return F.mse_loss(noise_composite.float(), noise_gt.float())


# =============================================================================
# Alpha occupancy loss (BCE + Dice)
# =============================================================================

def loss_alpha(
    alpha0: torch.Tensor,          # (B, S) predicted occupancy for entity 0
    alpha1: torch.Tensor,          # (B, S) predicted occupancy for entity 1
    entity_masks_BS: torch.Tensor, # (B, 2, S) GT entity masks at same spatial res
    eps:    float = 1e-6,
) -> torch.Tensor:
    """
    BCE + Dice loss on alpha predictions vs GT entity masks.

    The Dice term prevents class-imbalance collapse (entities are small).
    """
    m0 = entity_masks_BS[:, 0, :].float()  # (B, S) GT entity 0 mask
    m1 = entity_masks_BS[:, 1, :].float()  # (B, S) GT entity 1 mask

    # BCE — ensure same dtype
    a0 = alpha0.float().clamp(eps, 1.0 - eps)
    a1 = alpha1.float().clamp(eps, 1.0 - eps)
    bce0 = F.binary_cross_entropy(a0, m0, reduction='mean')
    bce1 = F.binary_cross_entropy(a1, m1, reduction='mean')
    bce = 0.5 * (bce0 + bce1)

    # Dice
    def _dice(pred, target):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1.0 - (2.0 * intersection + eps) / (union + eps)

    dice0 = _dice(alpha0, m0.float())
    dice1 = _dice(alpha1, m1.float())
    dice = 0.5 * (dice0 + dice1)

    return bce + dice


# =============================================================================
# Ownership loss (BCE vs GT visible masks)
# =============================================================================

def loss_ownership(
    own0: torch.Tensor,             # (B, S) predicted ownership entity 0
    own1: torch.Tensor,             # (B, S) predicted ownership entity 1
    visible_masks_BS: torch.Tensor, # (B, 2, S) GT visible masks
    eps:  float = 1e-6,
) -> torch.Tensor:
    """
    BCE on predicted ownership vs GT visible masks.

    Visible masks = what's actually visible after occlusion.
    This is the key signal for ownership learning.
    """
    v0 = visible_masks_BS[:, 0, :]  # (B, S)
    v1 = visible_masks_BS[:, 1, :]  # (B, S)

    bce0 = F.binary_cross_entropy(
        own0.float().clamp(eps, 1.0 - eps), v0.float(), reduction='mean')
    bce1 = F.binary_cross_entropy(
        own1.float().clamp(eps, 1.0 - eps), v1.float(), reduction='mean')

    return 0.5 * (bce0 + bce1)


# =============================================================================
# Depth ordering loss
# =============================================================================

def loss_depth_order(
    depth0: torch.Tensor,           # (B, S) depth logit entity 0
    depth1: torch.Tensor,           # (B, S) depth logit entity 1
    depth_orders: list,             # [(front_idx, back_idx), ...] per frame
    entity_masks_BS: torch.Tensor,  # (B, 2, S) GT entity masks
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Hinge loss: in overlap regions, front entity depth < back entity depth.

    depth0 < depth1 means entity0 is closer to camera (in front).
    If front=0: want depth0 < depth1 - margin → relu(depth0 - depth1 + margin)
    If front=1: want depth1 < depth0 - margin → relu(depth1 - depth0 + margin)

    Only applied in overlap regions (both entity masks > 0.5).
    """
    m0 = entity_masks_BS[:, 0, :]  # (B, S)
    m1 = entity_masks_BS[:, 1, :]  # (B, S)
    overlap = (m0 > 0.5).float() * (m1 > 0.5).float()  # (B, S)

    if overlap.sum() < 1.0:
        return torch.tensor(0.0, device=depth0.device)

    # Determine front entity (use first frame's depth order for the batch)
    # In practice, batch_size=1 with multiple frames flattened
    if len(depth_orders) > 0:
        front = int(depth_orders[0][0])
    else:
        front = 0

    if front == 0:
        # entity0 is front → depth0 should be smaller (closer)
        violation = F.relu(depth0 - depth1 + margin)  # (B, S)
    else:
        # entity1 is front → depth1 should be smaller
        violation = F.relu(depth1 - depth0 + margin)  # (B, S)

    # Only penalize in overlap
    masked_violation = violation * overlap
    n_overlap = overlap.sum().clamp(min=1.0)
    return masked_violation.sum() / n_overlap


# =============================================================================
# Entity visible region loss
# =============================================================================

def loss_entity_visible(
    F_0:  torch.Tensor,              # (B, S, D) entity 0 features
    F_1:  torch.Tensor,              # (B, S, D) entity 1 features
    F_g:  torch.Tensor,              # (B, S, D) global features
    own0: torch.Tensor,              # (B, S)
    own1: torch.Tensor,              # (B, S)
    visible_masks_BS: torch.Tensor,  # (B, 2, S) GT visible masks
    eps:  float = 1e-6,
) -> torch.Tensor:
    """
    Entity features should diverge from global features IN visible regions
    and stay close to global features OUTSIDE.

    L = -mean(||F_0 - F_g|| * v0) - mean(||F_1 - F_g|| * v1)
      + mean(||F_0 - F_g|| * (1-m0)) + mean(||F_1 - F_g|| * (1-m1))

    Simplified: just use cosine distance in visible regions as a separation
    encouragement, weighted by ownership confidence.
    """
    v0 = visible_masks_BS[:, 0, :].float()  # (B, S)
    v1 = visible_masks_BS[:, 1, :].float()

    # Feature divergence from global in visible regions
    delta0 = (F_0.float() - F_g.float()).pow(2).mean(dim=-1)  # (B, S)
    delta1 = (F_1.float() - F_g.float()).pow(2).mean(dim=-1)  # (B, S)

    # We WANT divergence in visible regions → maximize delta in visible
    # Use negative mean of delta in visible regions (to encourage divergence)
    n_v0 = v0.sum().clamp(min=1.0)
    n_v1 = v1.sum().clamp(min=1.0)

    vis_div = -((delta0 * v0).sum() / n_v0 + (delta1 * v1).sum() / n_v1) * 0.5

    # Clamp to prevent runaway divergence
    return vis_div.clamp(min=-1.0)


# =============================================================================
# Leak suppression loss
# =============================================================================

def loss_leak(
    F_0:    torch.Tensor,  # (B, S, D) entity 0 features
    F_1:    torch.Tensor,  # (B, S, D) entity 1 features
    F_g:    torch.Tensor,  # (B, S, D) global features
    alpha0: torch.Tensor,  # (B, S) predicted entity 0 occupancy
    alpha1: torch.Tensor,  # (B, S) predicted entity 1 occupancy
    la:     float = 0.3,
) -> torch.Tensor:
    """
    Entity features should be close to global features OUTSIDE own alpha region.

    Prevents entity identity from leaking into background or other entity regions.

    L = mean(||F_0 - F_g||^2 * (1 - alpha0)) + mean(||F_1 - F_g||^2 * (1 - alpha1))
    """
    outside0 = (1.0 - alpha0.detach()).clamp(0, 1).unsqueeze(-1)  # (B, S, 1)
    outside1 = (1.0 - alpha1.detach()).clamp(0, 1).unsqueeze(-1)  # (B, S, 1)

    diff0 = (F_0.float() - F_g.float().detach()).pow(2)  # (B, S, D)
    diff1 = (F_1.float() - F_g.float().detach()).pow(2)  # (B, S, D)

    leak0 = (diff0 * outside0).mean()
    leak1 = (diff1 * outside1).mean()

    return la * (leak0 + leak1)


# =============================================================================
# Temporal smoothness loss
# =============================================================================

def loss_temporal(
    alpha0_seq: torch.Tensor,  # (T, S) alpha for entity 0 across frames
    alpha1_seq: torch.Tensor,  # (T, S) alpha for entity 1 across frames
    own0_seq:   torch.Tensor,  # (T, S) ownership for entity 0 across frames
    own1_seq:   torch.Tensor,  # (T, S) ownership for entity 1 across frames
    la:         float = 0.1,
) -> torch.Tensor:
    """
    Consecutive-frame alpha/ownership smoothness.

    L = mean(|alpha_t - alpha_{t+1}|) + mean(|own_t - own_{t+1}|)
    """
    T = alpha0_seq.shape[0]
    if T < 2:
        return torch.tensor(0.0, device=alpha0_seq.device)

    l_alpha = (
        (alpha0_seq[1:] - alpha0_seq[:-1]).abs().mean()
        + (alpha1_seq[1:] - alpha1_seq[:-1]).abs().mean()
    ) * 0.5

    l_own = (
        (own0_seq[1:] - own0_seq[:-1]).abs().mean()
        + (own1_seq[1:] - own1_seq[:-1]).abs().mean()
    ) * 0.5

    return la * (l_alpha + l_own)


# =============================================================================
# Solo entity denoising anchor
# =============================================================================

def loss_identity_solo(
    pipe,
    noisy_solo:    torch.Tensor,  # (1, 4, T, H, W) noisy solo latent
    noise_solo:    torch.Tensor,  # (1, 4, T, H, W) GT noise for solo
    t:             torch.Tensor,  # timestep
    enc_entity:    torch.Tensor,  # (1, 77, 768) entity text embedding
) -> torch.Tensor:
    """
    Solo entity denoising anchor: the UNet (with entity tokens active)
    should correctly denoise a solo-rendered frame of the entity.

    This forces the shared backbone + entity branch to learn actual identity,
    grounded by a clean single-entity render.
    """
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        pred = pipe.unet(noisy_solo, t, encoder_hidden_states=enc_entity).sample
    return F.mse_loss(pred.float(), noise_solo.float())
