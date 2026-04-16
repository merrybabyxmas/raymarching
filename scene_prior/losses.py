"""scene_prior/losses.py
================================
Loss functions for the backbone-agnostic scene prior (blueprint §14).

All functions accept and return plain ``torch.Tensor`` scalars unless noted.
Every function is differentiable.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from scene_prior.scene_outputs import SceneOutputs


# ---------------------------------------------------------------------------
# Primitive: Dice loss
# ---------------------------------------------------------------------------

def dice_loss(
    pred:   torch.Tensor,   # (B, H, W) or (B, *)  values in [0, 1]
    target: torch.Tensor,   # same shape
    eps:    float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss.

    L_dice = 1 - (2 * |pred ∩ target|) / (|pred| + |target| + eps)

    Averaged over the batch.
    """
    pred   = pred.float().flatten(1)    # (B, N)
    target = target.float().flatten(1)  # (B, N)

    intersection = (pred * target).sum(dim=1)          # (B,)
    union        = pred.sum(dim=1) + target.sum(dim=1) # (B,)
    dice         = (2.0 * intersection + eps) / (union + eps)
    return (1.0 - dice).mean()


# ---------------------------------------------------------------------------
# Visible mask loss
# ---------------------------------------------------------------------------

def loss_visible(
    vis_e0:    torch.Tensor,   # (B, H, W)
    vis_e1:    torch.Tensor,   # (B, H, W)
    gt_vis_e0: torch.Tensor,   # (B, H, W)
    gt_vis_e1: torch.Tensor,   # (B, H, W)
) -> torch.Tensor:
    """Dice loss on visible masks for both entities."""
    return dice_loss(vis_e0, gt_vis_e0) + dice_loss(vis_e1, gt_vis_e1)


# ---------------------------------------------------------------------------
# Amodal mask loss
# ---------------------------------------------------------------------------

def loss_amodal(
    amo_e0:    torch.Tensor,   # (B, H, W)
    amo_e1:    torch.Tensor,   # (B, H, W)
    gt_amo_e0: torch.Tensor,   # (B, H, W)
    gt_amo_e1: torch.Tensor,   # (B, H, W)
) -> torch.Tensor:
    """Dice loss on amodal masks for both entities."""
    return dice_loss(amo_e0, gt_amo_e0) + dice_loss(amo_e1, gt_amo_e1)


# ---------------------------------------------------------------------------
# Occlusion consistency loss:  visible ≤ amodal
# ---------------------------------------------------------------------------

def loss_occlusion(
    vis_e0: torch.Tensor,    # (B, H, W)
    vis_e1: torch.Tensor,    # (B, H, W)
    amo_e0: torch.Tensor,    # (B, H, W)
    amo_e1: torch.Tensor,    # (B, H, W)
) -> torch.Tensor:
    """Penalise pixels where visible > amodal (physically impossible).

    L_occ = mean(relu(visible_ei - amodal_ei)) for each entity.
    """
    l0 = F.relu(vis_e0 - amo_e0).mean()
    l1 = F.relu(vis_e1 - amo_e1).mean()
    return l0 + l1


# ---------------------------------------------------------------------------
# Survival loss: both entities must appear somewhere
# ---------------------------------------------------------------------------

def loss_survival(
    vis_e0: torch.Tensor,   # (B, H, W)
    vis_e1: torch.Tensor,   # (B, H, W)
    tau:    float = 0.02,
) -> torch.Tensor:
    """Penalise an entity for having near-zero total visible mass.

    L_surv = relu(tau - mean_spatial(visible_ei)) per entity, mean over batch.
    This encourages both entities to remain "alive" (non-degenerate).
    """
    # Per-sample spatial mean
    mean_e0 = vis_e0.float().flatten(1).mean(dim=1)   # (B,)
    mean_e1 = vis_e1.float().flatten(1).mean(dim=1)   # (B,)
    l0 = F.relu(tau - mean_e0).mean()
    l1 = F.relu(tau - mean_e1).mean()
    return l0 + l1


# ---------------------------------------------------------------------------
# Separation loss: maximise |sep_map|
# ---------------------------------------------------------------------------

def loss_separation(
    sep_map: torch.Tensor,            # (B, H, W)
    gt_sep:  Optional[torch.Tensor] = None,   # (B, H, W), optional
) -> torch.Tensor:
    """Encourage the two entities to occupy distinct pixels.

    If ``gt_sep`` is provided: MSE between pred and gt separation maps.
    Otherwise: -mean(|sep_map|)  (unsupervised: maximise absolute separation).
    """
    if gt_sep is not None:
        return F.mse_loss(sep_map.float(), gt_sep.float())
    # Unsupervised: maximise absolute separation (negative = penalty)
    return -sep_map.float().abs().mean()


# ---------------------------------------------------------------------------
# Identity contrastive loss
# ---------------------------------------------------------------------------

def loss_identity_contrastive(
    h1:     torch.Tensor,   # (B, D)  entity 0 features across two views
    h2:     torch.Tensor,   # (B, D)  entity 0 features across two views
    z1:     torch.Tensor,   # (B, D)  entity 1 features across two views
    z2:     torch.Tensor,   # (B, D)
    margin: float = 0.5,
) -> torch.Tensor:
    """Identity contrastive loss.

    Same-entity pairs (h1, h2) and (z1, z2) should be close.
    Cross-entity pairs should be at least ``margin`` apart.

    L = MSE(h1, h2) + MSE(z1, z2)
      + relu(margin - ||h1 - z1||_2)
    """
    # Pull same-entity features together
    l_pull = F.mse_loss(h1.float(), h2.float()) + F.mse_loss(z1.float(), z2.float())

    # Push cross-entity features apart (hinge on mean distance)
    dist_cross = (h1.float() - z1.float()).norm(dim=-1)   # (B,)
    l_push = F.relu(margin - dist_cross).mean()

    return l_pull + l_push


# ---------------------------------------------------------------------------
# Reappearance loss
# ---------------------------------------------------------------------------

def loss_reappearance(
    hidden_hist:  torch.Tensor,   # (B, H, W) — hidden (occluded) map at t-1
    reappeared:   torch.Tensor,   # (B, H, W) — visible map at t
) -> torch.Tensor:
    """Encourage an entity that was hidden to reappear.

    If the entity was significantly hidden at t-1, penalise low visibility at t.

    L_reapp = mean(hidden_hist * relu(0.5 - reappeared))
    """
    return (hidden_hist.float() * F.relu(0.5 - reappeared.float())).mean()


# ---------------------------------------------------------------------------
# Color routing loss
# ---------------------------------------------------------------------------

def loss_color_routing(
    amo_e0:   torch.Tensor,   # (B, H, W)
    amo_e1:   torch.Tensor,   # (B, H, W)
    routing0: torch.Tensor,   # (B, 1, H, W) or (B, H, W) color similarity map
    routing1: torch.Tensor,   # (B, 1, H, W) or (B, H, W)
) -> torch.Tensor:
    """BCE between amodal masks and color routing maps.

    Forces entity fields to follow the color-based spatial prior.
    """
    def _squeeze(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4:
            return t.squeeze(1)   # (B, H, W)
        return t

    r0 = _squeeze(routing0).float().clamp(0.0, 1.0)
    r1 = _squeeze(routing1).float().clamp(0.0, 1.0)

    l0 = F.binary_cross_entropy(amo_e0.float().clamp(1e-6, 1.0 - 1e-6), r0)
    l1 = F.binary_cross_entropy(amo_e1.float().clamp(1e-6, 1.0 - 1e-6), r1)
    return l0 + l1


# ---------------------------------------------------------------------------
# Total scene loss
# ---------------------------------------------------------------------------

def total_scene_loss(
    scene_out: SceneOutputs,
    gt:        Dict[str, Any],
    lam:       Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """Compute all scene prior losses and return a dict with 'total'.

    Parameters
    ----------
    scene_out : SceneOutputs
        Renderer outputs.
    gt : dict
        Ground-truth tensors.  Expected keys (all (B, H, W)):
          - gt_vis_e0, gt_vis_e1   : visible GT masks
          - gt_amo_e0, gt_amo_e1   : amodal GT masks
          Optional keys:
          - routing0, routing1     : (B, 1, H, W) color maps
          - gt_sep                 : separation GT
          - hidden_hist_e0/e1      : previous-frame hidden maps for reappearance
          - reappeared_e0/e1       : visibility at current frame for reappearance
          - id_h1, id_h2, id_z1, id_z2 : identity features for contrastive loss
    lam : dict
        Lambda coefficients.  Supported keys:
          lambda_vis, lambda_amo, lambda_occ, lambda_surv, lambda_sep,
          lambda_color, lambda_id, lambda_reapp.
        Missing keys default to 0.

    Returns
    -------
    dict with keys: 'total', 'l_vis', 'l_amo', 'l_occ', 'l_surv',
                    'l_sep', 'l_color', 'l_id', 'l_reapp'.
    """
    losses: Dict[str, torch.Tensor] = {}

    def _lam(key: str) -> float:
        return float(lam.get(key, 0.0))

    # ---- Visible -----------------------------------------------------------
    l_vis = loss_visible(
        scene_out.visible_e0, scene_out.visible_e1,
        gt["gt_vis_e0"], gt["gt_vis_e1"],
    )
    losses["l_vis"] = l_vis

    # ---- Amodal ------------------------------------------------------------
    l_amo = loss_amodal(
        scene_out.amodal_e0, scene_out.amodal_e1,
        gt["gt_amo_e0"], gt["gt_amo_e1"],
    )
    losses["l_amo"] = l_amo

    # ---- Occlusion consistency ---------------------------------------------
    l_occ = loss_occlusion(
        scene_out.visible_e0, scene_out.visible_e1,
        scene_out.amodal_e0,  scene_out.amodal_e1,
    )
    losses["l_occ"] = l_occ

    # ---- Survival ----------------------------------------------------------
    l_surv = loss_survival(scene_out.visible_e0, scene_out.visible_e1)
    losses["l_surv"] = l_surv

    # ---- Separation --------------------------------------------------------
    gt_sep = gt.get("gt_sep", None)
    l_sep = loss_separation(scene_out.sep_map, gt_sep)
    losses["l_sep"] = l_sep

    # ---- Color routing (optional) -----------------------------------------
    routing0 = gt.get("routing0", None)
    routing1 = gt.get("routing1", None)
    if routing0 is not None and routing1 is not None and _lam("lambda_color") > 0.0:
        l_color = loss_color_routing(
            scene_out.amodal_e0, scene_out.amodal_e1, routing0, routing1,
        )
    else:
        l_color = torch.tensor(0.0, device=scene_out.visible_e0.device)
    losses["l_color"] = l_color

    # ---- Identity contrastive (optional) ----------------------------------
    if all(k in gt for k in ("id_h1", "id_h2", "id_z1", "id_z2")) and _lam("lambda_id") > 0.0:
        l_id = loss_identity_contrastive(
            gt["id_h1"], gt["id_h2"], gt["id_z1"], gt["id_z2"],
        )
    else:
        l_id = torch.tensor(0.0, device=scene_out.visible_e0.device)
    losses["l_id"] = l_id

    # ---- Reappearance (optional) ------------------------------------------
    l_reapp = torch.tensor(0.0, device=scene_out.visible_e0.device)
    if _lam("lambda_reapp") > 0.0:
        for eid in ("e0", "e1"):
            hk = f"hidden_hist_{eid}"
            rk = f"reappeared_{eid}"
            if hk in gt and rk in gt:
                vis_t = scene_out.visible_e0 if eid == "e0" else scene_out.visible_e1
                l_reapp = l_reapp + loss_reappearance(gt[hk], vis_t)
    losses["l_reapp"] = l_reapp

    # ---- Total -------------------------------------------------------------
    total = (
          _lam("lambda_vis")   * l_vis
        + _lam("lambda_amo")   * l_amo
        + _lam("lambda_occ")   * l_occ
        + _lam("lambda_surv")  * l_surv
        + _lam("lambda_sep")   * l_sep
        + _lam("lambda_color") * l_color
        + _lam("lambda_id")    * l_id
        + _lam("lambda_reapp") * l_reapp
    )
    losses["total"] = total

    return losses
