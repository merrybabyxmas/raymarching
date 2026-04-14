"""
Phase 62 — First-Hit Projection with Temperature Scaling
==========================================================

Fix for Issue 3: soft transmittance leaks ~10-20% wrong entity.
Temperature parameter sharpens entity_probs before compositing,
producing near-hard rendering while maintaining differentiability.

At temperature=1.0: standard sigmoid (soft, ~10% leak)
At temperature=0.1: near-binary (hard, <1% leak)
Schedule: anneal from 1.0 → 0.1 over training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.phase62.objectives.base import VolumeOutputs


class FirstHitProjector(nn.Module):

    def __init__(self, n_classes: int = 3, bg_class: int = 0, temperature: float = 1.0):
        super().__init__()
        self.n_classes = n_classes
        self.bg_class = bg_class
        self.temperature = temperature

    def set_temperature(self, t: float):
        self.temperature = max(t, 0.01)

    def forward(self, vol_outputs: VolumeOutputs) -> VolumeOutputs:
        entity_probs_raw = vol_outputs.entity_probs  # (B, 2, K, H, W)
        B, _, K, H, W = entity_probs_raw.shape
        C = self.n_classes
        device = entity_probs_raw.device

        # Temperature scaling: sharpen entity presence probabilities.
        # entity_probs come from sigmoid(logit) or sigmoid(fg)*softmax(id).
        # To sharpen without breaking gradients, re-scale via logit space:
        #   p_sharp = sigmoid(logit / tau) where logit = log(p/(1-p))
        # This makes p closer to 0 or 1 as tau → 0.
        if self.temperature != 1.0:
            eps = 1e-6
            p_clamped = entity_probs_raw.clamp(eps, 1.0 - eps)
            logits = torch.log(p_clamped / (1.0 - p_clamped))
            entity_probs = torch.sigmoid(logits / self.temperature)
        else:
            entity_probs = entity_probs_raw

        occ = 1.0 - (1.0 - entity_probs[:, 0]) * (1.0 - entity_probs[:, 1])

        # Transmittance: T_k = prod_{j<k}(1 - occ_j)
        trans_before = []
        running = torch.ones(B, H, W, device=device, dtype=entity_probs.dtype)
        for k in range(K):
            trans_before.append(running)
            running = running * (1.0 - occ[:, k])
        trans_before = torch.stack(trans_before, dim=1)

        front_entity = (entity_probs * trans_before.unsqueeze(1)).sum(dim=2)
        front_bg = running.unsqueeze(1)
        front_probs = torch.cat([front_bg, front_entity], dim=1)
        front_probs = front_probs / front_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Hard class from sharpened probs
        entity_max, entity_idx = entity_probs.max(dim=1)
        class_per_voxel = torch.where(
            entity_max > 0.5,
            entity_idx + 1,
            torch.zeros_like(entity_idx),
        )

        visible_class = torch.zeros(B, H, W, dtype=torch.long, device=device)
        for k in range(K):
            cls_k = class_per_voxel[:, k]
            update_mask = (visible_class == self.bg_class) & (cls_k != self.bg_class)
            visible_class = torch.where(update_mask, cls_k, visible_class)

        # Straight-through: hard forward, soft backward
        visible_hard = F.one_hot(visible_class, num_classes=C).permute(0, 3, 1, 2).float()
        front_probs_st = visible_hard - front_probs.detach() + front_probs

        # Back projection
        has_front_before = 1.0 - trans_before
        back_entity = (entity_probs * has_front_before.unsqueeze(1)).sum(dim=2)
        back_bg = torch.zeros(B, 1, H, W, device=device, dtype=entity_probs.dtype)
        back_probs = torch.cat([back_bg, back_entity], dim=1)
        back_sum = back_probs.sum(dim=1, keepdim=True)
        back_probs = torch.where(back_sum > 1e-6, back_probs / back_sum.clamp(min=1e-6), back_probs)

        # Amodal/visible projections use sharpened probs too
        amodal_e0 = 1.0 - (1.0 - entity_probs[:, 0]).prod(dim=1)
        amodal_e1 = 1.0 - (1.0 - entity_probs[:, 1]).prod(dim=1)
        visible_e0 = (trans_before * entity_probs[:, 0]).sum(dim=1)
        visible_e1 = (trans_before * entity_probs[:, 1]).sum(dim=1)

        vol_outputs.visible_class = visible_class
        vol_outputs.front_probs = front_probs_st
        vol_outputs.back_probs = back_probs
        vol_outputs.amodal.update({"e0": amodal_e0, "e1": amodal_e1})
        vol_outputs.visible.update({"e0": visible_e0, "e1": visible_e1})

        return vol_outputs
