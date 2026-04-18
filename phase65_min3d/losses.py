from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .scene_outputs import SceneState


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def visible_loss(scene_state: SceneState, gt_visible: torch.Tensor) -> torch.Tensor:
    pred_e0 = scene_state.maps.visible_e0
    pred_e1 = scene_state.maps.visible_e1
    gt_e0 = gt_visible[:, 0:1]
    gt_e1 = gt_visible[:, 1:2]
    return dice_loss(pred_e0, gt_e0) + dice_loss(pred_e1, gt_e1)


def amodal_loss(scene_state: SceneState, gt_amodal: torch.Tensor) -> torch.Tensor:
    pred_e0 = scene_state.maps.amodal_e0
    pred_e1 = scene_state.maps.amodal_e1
    gt_e0 = gt_amodal[:, 0:1]
    gt_e1 = gt_amodal[:, 1:2]
    return dice_loss(pred_e0, gt_e0) + dice_loss(pred_e1, gt_e1)


def occlusion_consistency_loss(scene_state: SceneState) -> torch.Tensor:
    m = scene_state.maps
    return (
        F.relu(m.visible_e0 - m.amodal_e0).mean()
        + F.relu(m.visible_e1 - m.amodal_e1).mean()
    )


def _masked_avg_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if feat.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask, size=feat.shape[-2:], mode='bilinear', align_corners=False)
    denom = mask.sum(dim=(2, 3)).clamp(min=1e-6)
    pooled = (feat * mask).sum(dim=(2, 3)) / denom
    return pooled


def temporal_identity_loss(curr_state: SceneState, prev_state: Optional[SceneState]) -> torch.Tensor:
    if prev_state is None:
        return torch.tensor(0.0, device=curr_state.features.feat_e0.device)
    z0_curr = _masked_avg_pool(curr_state.features.feat_e0, curr_state.maps.amodal_e0)
    z1_curr = _masked_avg_pool(curr_state.features.feat_e1, curr_state.maps.amodal_e1)
    z0_prev = _masked_avg_pool(prev_state.features.feat_e0, prev_state.maps.amodal_e0).detach()
    z1_prev = _masked_avg_pool(prev_state.features.feat_e1, prev_state.maps.amodal_e1).detach()
    return F.mse_loss(z0_curr, z0_prev) + F.mse_loss(z1_curr, z1_prev)


def slot_separation_loss(scene_state: SceneState, margin: float = 0.25) -> torch.Tensor:
    """Encourage entity slots to represent different objects instead of becoming identical.

    Uses the explicit slot embeddings when available and falls back to amodal-pooled
    features otherwise.
    """
    if scene_state.slot_e0 is not None and scene_state.slot_e1 is not None:
        z0 = scene_state.slot_e0
        z1 = scene_state.slot_e1
    else:
        z0 = _masked_avg_pool(scene_state.features.feat_e0, scene_state.maps.amodal_e0)
        z1 = _masked_avg_pool(scene_state.features.feat_e1, scene_state.maps.amodal_e1)
    z0 = F.normalize(z0, dim=-1)
    z1 = F.normalize(z1, dim=-1)
    cos = (z0 * z1).sum(dim=-1)
    return F.relu(cos - margin).mean()


def cross_view_slot_consistency_loss(state_a: SceneState, state_b: SceneState, margin: float = 0.20) -> torch.Tensor:
    if state_a.slot_e0 is None or state_a.slot_e1 is None or state_b.slot_e0 is None or state_b.slot_e1 is None:
        return torch.tensor(0.0, device=state_a.features.feat_e0.device)
    a0 = F.normalize(state_a.slot_e0, dim=-1)
    a1 = F.normalize(state_a.slot_e1, dim=-1)
    b0 = F.normalize(state_b.slot_e0, dim=-1)
    b1 = F.normalize(state_b.slot_e1, dim=-1)
    pos = (2.0 - (a0 * b0).sum(dim=-1) - (a1 * b1).sum(dim=-1)).mean()
    neg01 = (a0 * b1).sum(dim=-1)
    neg10 = (a1 * b0).sum(dim=-1)
    neg = F.relu(neg01 - margin).mean() + F.relu(neg10 - margin).mean()
    return pos + neg


def depth_ordering_loss(scene_state: SceneState, gt_front_idx: Optional[torch.Tensor] = None, margin: float = 0.05) -> torch.Tensor:
    if gt_front_idx is None:
        return torch.tensor(0.0, device=scene_state.maps.depth_e0.device)
    d0 = scene_state.maps.depth_e0
    d1 = scene_state.maps.depth_e1
    overlap = (scene_state.maps.amodal_e0 > 0.1) & (scene_state.maps.amodal_e1 > 0.1)
    if not overlap.any():
        return torch.tensor(0.0, device=d0.device)
    front0 = (gt_front_idx == 0).view(-1, 1, 1, 1)
    front1 = (gt_front_idx == 1).view(-1, 1, 1, 1)
    l0 = F.relu(d0 - d1 + margin) * overlap * front0
    l1 = F.relu(d1 - d0 + margin) * overlap * front1
    return l0.mean() + l1.mean()


def reconstruction_loss(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_rgb, gt_rgb) + 0.1 * F.mse_loss(pred_rgb, gt_rgb)
