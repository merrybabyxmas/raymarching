"""
Factorized Foreground + Identity objective with rendering-consistent terms.

Architecture (v12 — decoupled fg/depth heads):
  fg_magnitude = sigmoid(fg_spatial_logit)  → (B, 1, H, W) spatial fg presence
  depth_attn = softmax_K(fg_logit_vol)      → (B, K, H, W) depth localization
  p_fg = fg_magnitude × depth_attn
  q_n = softmax(id_logits)_n
  p_n = p_fg × q_n

L_fg_spatial = BCE(fg_spatial_logit, Y_fg_any)
               where Y_fg_any = 1 if ANY entity at (h,w), 0 if pure background.
               Trains fg_magnitude to be 1 at fg pixels, 0 at bg.
               Clean spatial supervision — no coupling with depth.
L_depth_ce   = CE(fg_logit_vol at fg-any pixels, k_front_target)
               Directly trains depth_attn to concentrate at the front-most occupied
               K bin. CE is the cleanest depth supervision — no gradient coupling
               between fg_magnitude and depth_attn.
L_id         = CE(z_id, Y_id | Y_fg_full=1)  on ALL occupied bins
L_vis        = Dice(visible_e_n, gt_visible_n)  rendered-space loss
L_compact    = H(depth_mass_e0) + H(depth_mass_e1)  depth entropy (secondary)
L_depth_vis  = -log(entity_probs[n, k_front, h, w]) for visible fg pixels (direct depth)
L            = L_fg_spatial + lambda_depth_ce * L_depth_ce
               + lambda_id * L_id + lambda_vis * L_vis
               + lambda_compact * L_compact + lambda_depth_vis * L_depth_vis
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


def _front_surface_mask(V_gt: torch.Tensor) -> torch.Tensor:
    """
    Returns (B, K, H, W) float mask with 1.0 only at the FRONT-MOST occupied
    depth bin per (b, h, w) column, 0.0 elsewhere.

    "Front-most" = smallest K index where V_gt > 0 (closest to the camera).

    Motivation: entity_probs should concentrate at the visible surface, not
    spread across all occupied depth bins. Full-volume fg supervision makes
    fg_logit uniformly high across all occupied K → compact ≈ 0 always.
    Front-surface supervision → fg_logit peaks at ONE K per pixel → compact >> 0.
    """
    occupied = (V_gt > 0).float()  # (B, K, H, W)
    B, K, H, W = occupied.shape
    has_any = occupied.any(dim=1)               # (B, H, W)
    if not has_any.any():
        return occupied  # no entity → return zeros

    # Front-most bin = smallest k s.t. occupied[b,k,h,w]=1
    # argmax on a binary tensor returns the FIRST True = smallest k (nearest camera).
    # Do NOT use (K-1)-argmax(flip): that gives the BACK surface (largest k).
    front_k = occupied.argmax(dim=1)  # (B, H, W) — first True in K dim = front
    # One-hot at front_k only where entity exists
    front_k_idx = front_k.unsqueeze(1)  # (B, 1, H, W)
    y_front = torch.zeros_like(occupied)
    y_front.scatter_(1, front_k_idx, 1.0)
    # Zero out pixels where no entity exists (front_k is undefined there)
    y_front = y_front * has_any.float().unsqueeze(1)
    return y_front


class FactorizedFgIdObjective(VolumeObjective):

    def __init__(
        self,
        lambda_id: float = 1.0,
        fg_pos_weight: float = 10.0,
        lambda_vis: float = 0.5,
        lambda_compact: float = 0.5,
        lambda_depth_ce: float = 3.0,
        lambda_depth_vis: float = 0.0,
        lambda_balance: float = 0.0,   # v22: penalise vis_e0 ≠ vis_e1 imbalance
        detach_fg_from_entity_losses: bool = False,  # v34: stop L_vis/L_depth_vis/L_balance gradient to fg_magnitude
        lambda_overlay_preserve: float = 0.0,  # v35: rendered fg-coverage preservation loss (activates in stage3)
        # Legacy params (kept for config backwards-compat, not used in v12 objective):
        lambda_dice: float = 0.0,
        lambda_hinge: float = 0.0,
        hinge_margin: float = 1.0,
        hinge_density_thresh: float = 0.20,
    ):
        super().__init__()
        self.lambda_id = lambda_id
        self.fg_pos_weight = fg_pos_weight
        self.lambda_vis = lambda_vis
        self.lambda_compact = lambda_compact
        self.lambda_depth_ce = lambda_depth_ce
        self.lambda_depth_vis = lambda_depth_vis
        self.lambda_balance = lambda_balance
        self.detach_fg_from_entity_losses = detach_fg_from_entity_losses
        self.lambda_overlay_preserve = lambda_overlay_preserve

    def forward(
        self,
        outputs: VolumeOutputs,
        V_gt: torch.Tensor,
        gt_visible: Optional[torch.Tensor] = None,
        gt_amodal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fg_logit = outputs.fg_logit[:, 0]   # (B, K, H, W) — depth logit
        id_logits = outputs.id_logits        # (B, 2, K, H, W)

        # ── Y_fg targets ──────────────────────────────────────────────────────
        Y_fg_any = (V_gt > 0).any(dim=1).float()   # (B, H, W): fg presence at pixel
        occupied = (V_gt > 0).float()               # (B, K, H, W): all occupied bins
        k_front = occupied.argmax(dim=1)            # (B, H, W): front-most depth bin

        # ── L_fg_spatial: BCE on spatial fg/bg detection ──────────────────────
        # Trains fg_magnitude = sigmoid(fg_spatial_head(h_2d)) → 1 at fg, 0 at bg.
        # Clean 2D supervision, NO coupling with depth_attn (fg_logit unchanged).
        pos_weight = torch.tensor([self.fg_pos_weight], device=fg_logit.device)
        if outputs.fg_spatial_logit is not None:
            fg_spatial = outputs.fg_spatial_logit[:, 0]  # (B, H, W)
            L_fg_spatial = F.binary_cross_entropy_with_logits(
                fg_spatial, Y_fg_any, pos_weight=pos_weight, reduction="mean")
        else:
            # Fallback: use max over K as proxy spatial logit (backward compat)
            fg_spatial = fg_logit.max(dim=1).values  # (B, H, W)
            L_fg_spatial = F.binary_cross_entropy_with_logits(
                fg_spatial, Y_fg_any, pos_weight=pos_weight, reduction="mean")

        # ── L_depth_ce: CE on depth_attn = softmax_K(fg_logit) ───────────────
        # Directly trains depth_attn to concentrate at k_front for fg pixels.
        # CE gradient: pushes fg_logit[k_front] up, all k_others down — no coupling
        # with fg_magnitude since fg_spatial_head is separate.
        Y_fg_any_bool = Y_fg_any.bool()
        if Y_fg_any_bool.any():
            depth_logit_fg = fg_logit.permute(0, 2, 3, 1)   # (B, H, W, K)
            depth_logit_flat = depth_logit_fg[Y_fg_any_bool]  # (n_fg, K)
            k_front_flat = k_front[Y_fg_any_bool]              # (n_fg,)
            L_depth_ce = F.cross_entropy(depth_logit_flat, k_front_flat, reduction="mean")
            L_depth_ce = L_depth_ce.clamp(max=10.0)
        else:
            L_depth_ce = fg_logit.new_zeros(())

        L_fg = L_fg_spatial + self.lambda_depth_ce * L_depth_ce

        # L_id: CE over ALL occupied depth bins (not just front surface).
        # Identity discrimination (e0 vs e1) is useful wherever the entity
        # exists in 3D, not just at the front surface.
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

        # ── Detached-fg entity_probs for entity-specific losses ──────────────
        # v34: detach_fg_from_entity_losses=True stops L_vis/L_depth_vis/L_balance
        # gradients from reaching fg_magnitude (fg_spatial_head parameters).
        #
        # Root cause: with balanced entities (lambda_id=10, sharp q), L_vis Dice
        # denominator produces gradients that push p_fg DOWN at wrong-entity pixels.
        # With collapsed entity (lambda_id=3), entity 1 covers full fg → L_vis pushes
        # p_fg toward fg_union (overlay HIGH). The coupling between entity-specific
        # L_vis and fg_magnitude is what causes high-lambda_id → low-overlay.
        #
        # Fix: entity-specific losses (L_vis, L_depth_vis, L_balance) use
        # entity_probs recomputed with p_fg.detach() — gradient flows to q and
        # depth_attn only. fg_magnitude is trained solely by L_fg_spatial (direct
        # BCE toward GT fg union) → overlay preserved regardless of lambda_id.
        if (self.detach_fg_from_entity_losses
                and outputs.entity_probs is not None
                and outputs.id_logits is not None):
            # q = softmax(id_logits, dim=1): (B, 2, K, H, W)
            q_probs = torch.softmax(outputs.id_logits, dim=1)
            # p_fg = sum of entity_probs over entity axis (since sum_e q_e = 1 for 2-class)
            p_fg_sum = outputs.entity_probs.sum(dim=1, keepdim=True)  # (B, 1, K, H, W)
            ep_for_entity_losses = (p_fg_sum.detach() * q_probs).clamp(0.0, 1.0)
            # Alpha-composite over K axis for visible projections
            _vis_e0_det = 1.0 - (1.0 - ep_for_entity_losses[:, 0].clamp(0.0, 1.0 - 1e-7)).prod(dim=1)
            _vis_e1_det = 1.0 - (1.0 - ep_for_entity_losses[:, 1].clamp(0.0, 1.0 - 1e-7)).prod(dim=1)
        else:
            ep_for_entity_losses = outputs.entity_probs
            _vis_e0_det = outputs.visible.get("e0") if outputs.visible else None
            _vis_e1_det = outputs.visible.get("e1") if outputs.visible else None

        # L_vis: rendering-consistent Dice on projected visible output
        # Only active when visible projections are available (after projection step)
        L_vis = fg_logit.new_zeros(())
        if gt_visible is not None and _vis_e0_det is not None and _vis_e1_det is not None:
            vis_e0 = _vis_e0_det
            vis_e1 = _vis_e1_det
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
                and ep_for_entity_losses is not None):
            B, K_v, H_v, W_v = V_gt.shape
            ep = ep_for_entity_losses  # (B, 2, K, H, W) — fg detached if detach_fg_from_entity_losses
            K_ep = ep.shape[2]
            B_vis = min(gt_visible.shape[0], B)

            depth_losses = []
            for n, entity_class in enumerate([1, 2]):
                entity_present = (V_gt[:B_vis] == entity_class)  # (B_vis, K, H, W)
                has_entity = entity_present.any(dim=1)             # (B_vis, H, W)
                if not has_entity.any():
                    continue

                # Front-most depth bin = smallest k where entity is present (argmin).
                # argmax on binary tensor returns the FIRST True = smallest k = front.
                # Do NOT use (K-1)-argmax(flip): that gives back surface (largest k).
                entity_float = entity_present.float()              # (B_vis, K, H, W)
                front_depth = entity_float.argmax(dim=1)           # (B_vis, H, W)
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

        # L_balance: penalise asymmetric entity visibility.
        # WinnerRatio = max(vis_e0, vis_e1) / (vis_e0 + vis_e1) should be ≤ 0.45.
        # Direct fix: minimise squared difference of mean visible projections.
        # L_balance = (mean(vis_e0) - mean(vis_e1))^2
        # Only activates when both projections are present.
        L_balance = fg_logit.new_zeros(())
        if (self.lambda_balance > 0
                and _vis_e0_det is not None and _vis_e1_det is not None):
            vis_e0_b = _vis_e0_det
            vis_e1_b = _vis_e1_det
            B_b = min(vis_e0_b.shape[0], vis_e1_b.shape[0])
            L_balance = (vis_e0_b[:B_b].mean() - vis_e1_b[:B_b].mean()).pow(2)
            total = total + self.lambda_balance * L_balance

        # L_overlay_preserve: rendered fg-coverage preservation loss (v35).
        # Activates in stage3 (trainer sets lambda_overlay_preserve > 0 on stage3 entry).
        #
        # Root cause of stage3 overlay drop (v28-v34): UNet features drift as LoRA/adapters
        # train → vol/fg_spatial outputs change even with frozen params → entity_probs change
        # → visible_class changes → overlay drops.
        #
        # Fix: Dice(front_probs[:, 1:].sum(), GT_fg_any) directly supervises the rendered
        # fg probability toward GT fg union. With fg_spatial unfrozen in stage3, gradient
        # flows: L_overlay_preserve → front_probs (straight-through) → entity_probs → p_fg
        # → fg_magnitude (fg_spatial_head) → fg_spatial adapts to maintain fg coverage
        # regardless of UNet feature drift.
        #
        # Must only activate in stage3 (NOT stage1/2) to avoid reinforcing entity 0
        # dominance before entity balance forms. Trainer sets lambda_overlay_preserve=0
        # during stage1/2 and switches to lambda_overlay_preserve_s3 at stage3 entry.
        L_overlay_preserve = fg_logit.new_zeros(())
        if (self.lambda_overlay_preserve > 0
                and outputs.front_probs is not None
                and gt_visible is not None):
            B_op = min(outputs.front_probs.shape[0], gt_visible.shape[0])
            # GT fg union: any entity visible at each pixel
            gt_fg_any = (gt_visible[:B_op] > 0.5).any(dim=1).float()  # (B, H, W)
            # Rendered fg probability: entity 0 + entity 1 front probs (differentiable)
            pred_fg_prob = outputs.front_probs[:B_op, 1:].sum(dim=1)  # (B, H, W)
            L_overlay_preserve = _dice(pred_fg_prob, gt_fg_any).clamp(max=5.0)
            total = total + self.lambda_overlay_preserve * L_overlay_preserve

        return {
            "total": total,
            "L_fg": L_fg.detach(),
            "L_fg_spatial": L_fg_spatial.detach(),
            "L_depth_ce": L_depth_ce.detach(),
            "L_id": L_id.detach(),
            "L_vis": L_vis.detach(),
            "L_compact": L_compact.detach(),
            "L_depth_vis": L_depth_vis.detach(),
            "L_balance": L_balance.detach(),
            "L_overlay_preserve": L_overlay_preserve.detach(),
        }
