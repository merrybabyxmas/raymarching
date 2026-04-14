"""
Factorized Foreground + Identity objective with rendering-consistent terms.

p_fg(x) = sigmoid(z_fg(x))
q_n(x)  = softmax(z_id(x))_n,  n in {0, 1}
p_n(x)  = p_fg(x) * q_n(x)

L_fg      = BCE(z_fg, Y_fg) + lambda_dice * Dice(p_fg, Y_fg)
            where Y_fg = 1[V_gt > 0]
            NOTE: Dice prevents the trivial BCE minimum at all-zero predictions.
            Dice(pred=0, Y_fg) = 1.0 → non-zero gradient even at collapse.
L_id      = CE(z_id, Y_id | Y_fg=1)  where Y_id = 0 if V_gt=1, 1 if V_gt=2
L_vis     = Dice(visible_e_n, gt_visible_n)  rendered-space loss (Issue 1 fix)
L_compact = H(depth_mass_e0) + H(depth_mass_e1)  depth entropy minimisation
L         = L_fg + lambda_id * L_id + lambda_vis * L_vis + lambda_compact * L_compact
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from training.phase62.objectives.base import VolumeObjective, VolumeOutputs
from training.phase62.losses import loss_depth_compactness


def _dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(-2, -1))
    denom = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


class FactorizedFgIdObjective(VolumeObjective):

    def __init__(
        self,
        lambda_id: float = 1.0,
        fg_pos_weight: float = 20.0,
        lambda_vis: float = 0.5,
        lambda_compact: float = 0.5,
        lambda_dice: float = 1.0,
        lambda_hinge: float = 1.0,
        hinge_margin: float = 1.0,
        hinge_density_thresh: float = 0.20,
        lambda_depth_vis: float = 0.0,
    ):
        super().__init__()
        self.lambda_id = lambda_id
        self.fg_pos_weight = fg_pos_weight
        self.lambda_vis = lambda_vis
        self.lambda_compact = lambda_compact
        self.lambda_dice = lambda_dice
        self.lambda_hinge = lambda_hinge
        self.hinge_margin = hinge_margin
        self.hinge_density_thresh = hinge_density_thresh
        self.lambda_depth_vis = lambda_depth_vis

    def forward(
        self,
        outputs: VolumeOutputs,
        V_gt: torch.Tensor,
        gt_visible: Optional[torch.Tensor] = None,
        gt_amodal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fg_logit = outputs.fg_logit[:, 0]   # (B, K, H, W)
        id_logits = outputs.id_logits        # (B, 2, K, H, W)

        # L_fg: BCE + Dice + Hinge for sparse foreground
        #
        # BCE alone has trivial min at all-zero (logit→-∞, loss→0).
        # Dice is better but also uses sigmoid: gradient→0 as logit→-∞.
        #
        # Hinge on fg voxels: relu(margin - logit[Y_fg=1])
        #   gradient = -1 for logit < margin, regardless of how negative logit is.
        #   No sigmoid saturation → non-zero gradient even at logit=-100.
        #   Breaks the "uniform slab → collapse" cycle by forcing fg voxels
        #   to maintain logit > margin before L_compact fires.
        Y_fg = (V_gt > 0).float()
        pos_weight = torch.tensor([self.fg_pos_weight], device=fg_logit.device)
        L_bce = F.binary_cross_entropy_with_logits(
            fg_logit, Y_fg, pos_weight=pos_weight, reduction="mean")

        # Dice component
        p_fg = torch.sigmoid(fg_logit)
        eps_d = 1e-6
        fg_inter = (p_fg * Y_fg).sum()
        fg_denom = p_fg.sum() + Y_fg.sum() + eps_d
        L_dice_fg = 1.0 - (2.0 * fg_inter + eps_d) / (fg_denom + eps_d)

        # Hinge component: ensures fg_logit > margin at DENSE GT fg voxels only.
        # "Dense" = depth bin has >= hinge_density_thresh * max_per_bin fg voxels.
        # Avoids applying hinge at depth bins with only 1-2 fg voxels (tails of 3D
        # object), which would force entity_probs to stay high at many depth bins
        # and prevent depth compactness (compact would stay near 0 permanently).
        #
        # With density gating: only the front/most-occupied bins trigger hinge,
        # allowing the tail bins to have entity_probs → 0, enabling concentration.
        Y_fg_density = Y_fg.sum(dim=(2, 3), keepdim=True)          # (B, K, 1, 1)
        max_density = Y_fg_density.amax(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1, 1, 1)
        dense_mask = (Y_fg_density / max_density) > self.hinge_density_thresh   # (B, K, 1, 1)
        hinge_active = Y_fg.bool() & dense_mask.expand_as(Y_fg)  # (B, K, H, W)
        if hinge_active.any():
            fg_logits_at_dense = fg_logit[hinge_active]
            L_hinge = F.relu(self.hinge_margin - fg_logits_at_dense).mean()
        else:
            L_hinge = fg_logit.new_zeros(())

        L_fg = L_bce + self.lambda_dice * L_dice_fg + self.lambda_hinge * L_hinge

        # L_id: CE only where foreground exists
        fg_mask = (V_gt > 0)
        if fg_mask.any():
            Y_id = (V_gt - 1).clamp(min=0).long()
            id_logits_flat = id_logits.permute(0, 2, 3, 4, 1)
            id_at_fg = id_logits_flat[fg_mask]
            Y_id_at_fg = Y_id[fg_mask]
            L_id = F.cross_entropy(id_at_fg, Y_id_at_fg, reduction="mean")
        else:
            L_id = fg_logit.new_zeros(())

        total = L_fg + self.lambda_id * L_id

        # L_vis: rendering-consistent Dice on projected visible output
        # Only active when visible projections are available (after projection step)
        L_vis = fg_logit.new_zeros(())
        if gt_visible is not None and "e0" in outputs.visible and "e1" in outputs.visible:
            vis_e0 = outputs.visible["e0"]
            vis_e1 = outputs.visible["e1"]
            B_vis = min(vis_e0.shape[0], gt_visible.shape[0])
            # Only apply rendered dice when fg has learned something (prevents
            # fighting foreground growth at init where visible ≈ 0.5 everywhere)
            fg_mass = vis_e0[:B_vis].sum() + vis_e1[:B_vis].sum()
            if fg_mass.item() > 1.0:
                L_vis = _dice(vis_e0[:B_vis], gt_visible[:B_vis, 0]) + \
                        _dice(vis_e1[:B_vis], gt_visible[:B_vis, 1])
                L_vis = L_vis.clamp(max=10.0)
                total = total + self.lambda_vis * L_vis

        # L_compact: encourage entity_probs to be localised in depth (compact blob)
        # Minimises entropy of depth-wise mass distribution per entity.
        # Guard: only activate when there's sufficient predicted entity mass.
        # Without guard, model can trivially collapse entity_probs→0 to avoid
        # both L_fg and L_compact (both go to 0/trivial with no predictions).
        L_compact = fg_logit.new_zeros(())
        if self.lambda_compact > 0 and outputs.entity_probs is not None:
            ep_mass = outputs.entity_probs.float().sum()
            # Only apply when model is actually predicting foreground.
            # Threshold: 2% of voxels per entity (lowered from 5% to catch early collapse).
            # Dice loss now prevents trivial all-zero escape so this guard is secondary.
            n_vox_per_entity = float(outputs.entity_probs[0, 0].numel())
            if ep_mass.item() > n_vox_per_entity * 0.02:
                # Use fg_spatial_mask to avoid bg leakage: averaging entity_probs over
                # ALL 256 spatial pixels (254 bg + 2 fg) spreads depth_mass uniformly
                # even when fg is perfectly concentrated, keeping compact near 0.
                # Masking to fg-only spatial locations gives accurate depth distribution.
                fg_spatial = (V_gt > 0).any(dim=1)  # (B, H, W)
                L_compact = loss_depth_compactness(outputs.entity_probs,
                                                   fg_spatial_mask=fg_spatial)
                total = total + self.lambda_compact * L_compact

        # L_depth_vis: directly supervise entity_probs to concentrate at the
        # front-most depth bin where each entity is present, for visible fg pixels.
        #
        # L_compact encourages concentration in aggregate (entropy minimisation)
        # but provides no per-pixel depth target.  L_depth_vis directly maximises
        # entity_probs[n, k_gt, h, w] at the correct depth bin k_gt — a much
        # stronger and more direct signal for depth localisation.
        #
        # Only activated when gt_visible and entity_probs are available.
        L_depth_vis = fg_logit.new_zeros(())
        if (self.lambda_depth_vis > 0
                and gt_visible is not None
                and outputs.entity_probs is not None):
            B, K_v, H_v, W_v = V_gt.shape
            ep = outputs.entity_probs  # (B, 2, K, H, W)
            K_ep = ep.shape[2]
            B_vis = min(gt_visible.shape[0], B)

            depth_losses = []
            for n, entity_class in enumerate([1, 2]):
                entity_present = (V_gt[:B_vis] == entity_class)  # (B_vis, K, H, W)
                has_entity = entity_present.any(dim=1)             # (B_vis, H, W)
                if not has_entity.any():
                    continue

                # Front-most depth bin (minimum k) where entity is present at each pixel.
                # (K - 1 - argmax_of_flipped) = argmin of original along K dim.
                entity_float = entity_present.float()              # (B_vis, K, H, W)
                front_depth = (K_v - 1) - entity_float.flip(1).argmax(dim=1)  # (B_vis, H, W)
                front_depth = front_depth.clamp(0, K_ep - 1)

                # Gather entity_probs at front depth bin for entity n
                depth_idx = front_depth.unsqueeze(1)               # (B_vis, 1, H, W)
                ep_at_front = ep[:B_vis, n].gather(1, depth_idx).squeeze(1)  # (B_vis, H, W)

                # Loss only where entity is visible (gt_visible) AND present in 3D
                vis_mask = (gt_visible[:B_vis, n] > 0.5) & has_entity  # (B_vis, H, W)
                if vis_mask.any():
                    ep_at_vis = ep_at_front[vis_mask].clamp(min=1e-8)
                    depth_losses.append(-torch.log(ep_at_vis).mean())

            if depth_losses:
                L_depth_vis = torch.stack(depth_losses).mean().clamp(max=10.0)
                total = total + self.lambda_depth_vis * L_depth_vis

        return {
            "total": total,
            "L_fg": L_fg.detach(),
            "L_bce": L_bce.detach(),
            "L_dice_fg": L_dice_fg.detach(),
            "L_hinge": L_hinge.detach(),
            "L_id": L_id.detach(),
            "L_vis": L_vis.detach(),
            "L_compact": L_compact.detach(),
            "L_depth_vis": L_depth_vis.detach(),
        }
