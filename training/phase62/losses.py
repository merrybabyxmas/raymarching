"""
Phase 62 — Mainline Losses
============================

Only production-validated losses live here.
Experimental losses are in losses_ablation.py.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def loss_diffusion(
    noise_pred: torch.Tensor,
    noise_gt: torch.Tensor,
) -> torch.Tensor:
    """Standard MSE diffusion loss."""
    return F.mse_loss(noise_pred.float(), noise_gt.float())


def loss_volume_ce(
    V_logits: torch.Tensor,                        # (B, C, K, H, W)
    V_gt: torch.Tensor,                            # (B, K, H, W)
    class_weights: Optional[torch.Tensor] = None,
    voxel_weights: Optional[torch.Tensor] = None,
    entity_pos_weight: float = 50.0,
) -> torch.Tensor:
    """
    Independent BCE-with-logits on entity voxel presences.

    Entity voxels are ~1% of total volume. Without strong positive
    weighting, the model learns "predict all bg" trivially.
    """
    logits_e = V_logits[:, 1:3].float()  # (B, 2, K, H, W)
    target_e0 = (V_gt == 1).float()
    target_e1 = (V_gt == 2).float()
    targets = torch.stack([target_e0, target_e1], dim=1)

    logits_e0 = logits_e[:, 0]
    logits_e1 = logits_e[:, 1]
    tgt_e0 = targets[:, 0]
    tgt_e1 = targets[:, 1]

    def _entity_loss(logits, tgt):
        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
        bce = bce.clamp(max=20.0)
        pos_mask = (tgt > 0.5)
        neg_mask = ~pos_mask
        n_pos = pos_mask.float().sum().clamp(min=1.0)
        n_neg = neg_mask.float().sum().clamp(min=1.0)
        l_pos = (bce * pos_mask.float()).sum() / n_pos
        l_neg = (bce * neg_mask.float()).sum() / n_neg
        return entity_pos_weight * l_pos + l_neg

    l_e0 = _entity_loss(logits_e0, tgt_e0)
    l_e1 = _entity_loss(logits_e1, tgt_e1)

    with torch.no_grad():
        ratio = (l_e0 / (l_e1 + 1e-6)).clamp(0.8, 1.25)
        w0 = ratio / (ratio + 1.0)
        w1 = 1.0 - w0
    return w0 * l_e0 + w1 * l_e1


def loss_projected_global(
    front_probs: torch.Tensor,      # (B, C, H, W)
    gt_visible: torch.Tensor,       # (B, 2, H, W)
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable global projection loss on front-hit 2D projection."""
    pred = front_probs[:, 1:3].float()
    gt = gt_visible.float()
    inter = (pred * gt).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + gt.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return (1.0 - iou).mean()


def loss_min_iou_balance(
    front_probs: torch.Tensor,      # (B, C, H, W)
    gt_visible: torch.Tensor,       # (B, 2, H, W)
    eps: float = 1e-6,
) -> torch.Tensor:
    """Quadratic min-IoU balance: ((1 - min_iou)^2).mean()"""
    pred = front_probs[:, 1:3].float()
    gt = gt_visible.float()
    inter = (pred * gt).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + gt.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    min_iou = iou.min(dim=1).values
    return ((1.0 - min_iou) ** 2).mean()


def loss_projected_balance(
    front_probs: torch.Tensor,
    gt_visible: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Legacy — kept for backward compat but now returns 0."""
    return front_probs.new_zeros(())


def loss_feature_separation(
    F_0: torch.Tensor,  # (B, S, D)
    F_1: torch.Tensor,  # (B, S, D)
) -> torch.Tensor:
    """
    Push F_0 and F_1 feature representations apart.

    Minimizes cosine similarity between per-pixel feature vectors.
    If F_0 and F_1 are already orthogonal (cos_sim=0), loss is 0.
    If identical (cos_sim=1), loss is 1.
    """
    f0 = F.normalize(F_0.float(), dim=-1, eps=1e-6)
    f1 = F.normalize(F_1.float(), dim=-1, eps=1e-6)
    cos_sim = (f0 * f1).sum(dim=-1)  # (B, S)
    return cos_sim.clamp(min=0.0).mean()


def loss_depth_compactness(
    entity_probs: torch.Tensor,  # (B, 2, K, H, W)
    eps: float = 1e-9,
    fg_spatial_mask: "Optional[torch.Tensor]" = None,  # (B, H, W) bool or float
) -> torch.Tensor:
    """
    Encourage entity_probs to be localised in a few depth bins (compact blob).

    Minimises the entropy of depth-wise activation mass per entity.
    Entropy=0 → all mass in one slice (perfect blob).
    Entropy=log(K) → uniform slab (worst case).

    fg_spatial_mask: if provided, depth_mass is averaged ONLY over fg spatial
    locations (where any entity exists in the GT). This prevents background
    leakage from ~0.01 entity_probs at 252 bg pixels dominating the depth
    distribution and killing compactness (bg leakage otherwise spreads depth_mass
    uniformly → entropy ≈ log(K) regardless of how well fg is concentrated).

    Contract stage1 pass requires compact ≥ 0.20
    (1 - normalised_entropy ≥ 0.20, i.e. normalised_entropy ≤ 0.80).
    """
    B, _, K, H, W = entity_probs.shape
    ep = entity_probs.float()

    if fg_spatial_mask is not None:
        # Compute depth mass only at fg spatial locations
        # fg_spatial_mask: (B, H, W) — 1 at locations where any entity is present
        mask = fg_spatial_mask.float().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H, W)
        n_fg = mask.sum(dim=(3, 4)).clamp(min=1.0)                # (B, 1, 1)
        depth_mass = (ep * mask).sum(dim=(3, 4)) / n_fg           # (B, 2, K)
    else:
        # Original: average over all H×W (including bg — avoid when possible)
        depth_mass = ep.mean(dim=(3, 4))

    # normalise to probability distribution
    depth_mass_sum = depth_mass.sum(dim=2, keepdim=True).clamp(min=eps)
    p = (depth_mass / depth_mass_sum).clamp(min=eps)
    # Shannon entropy, normalised by log(K)
    entropy = -(p * p.log()).sum(dim=2)          # (B, 2)
    normalised_entropy = entropy / (math.log(K) + eps)
    # Penalise high entropy (diffuse slab)
    return normalised_entropy.mean()


def loss_rendered_dice(
    visible_e0: torch.Tensor,  # (B, H, W)
    visible_e1: torch.Tensor,  # (B, H, W)
    gt_visible: torch.Tensor,  # (B, 2, H, W)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Rendering-consistent loss: Dice on the RENDERED 2D visible output.

    Unlike per-voxel BCE, this loss matches the actual rendering math
    (transmittance compositing), so gradients align with what we see.
    """
    def _dice(pred, target):
        inter = (pred * target).sum(dim=(-2, -1))
        denom = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()

    return _dice(visible_e0, gt_visible[:, 0]) + _dice(visible_e1, gt_visible[:, 1])


def loss_spatial_coherence(
    entity_probs: torch.Tensor,  # (B, 2, K, H, W)
) -> torch.Tensor:
    """
    Total-Variation regularization on the spatial (H, W) dimensions of entity_probs.

    Penalises rapid changes in entity assignment between neighbouring voxels.
    Encourages compact, connected entity regions → higher LCC.

    TV loss = mean |p_n[k,h+1,w] - p_n[k,h,w]| + mean |p_n[k,h,w+1] - p_n[k,h,w]|
    summed over both entities.

    Activation: set lambda_spatial_coherence > 0 in config.
    """
    ep = entity_probs.float()  # (B, 2, K, H, W)
    diff_h = (ep[:, :, :, 1:, :] - ep[:, :, :, :-1, :]).abs()
    diff_w = (ep[:, :, :, :, 1:] - ep[:, :, :, :, :-1]).abs()
    return diff_h.mean() + diff_w.mean()


def loss_fg_coverage_prior(
    entity_probs: torch.Tensor,   # (B, 2, K, H, W)
    min_fg_fraction: float = 0.05,
) -> torch.Tensor:
    """
    Foreground coverage prior: prevent all-background collapse.

    If the total foreground probability per sample falls below min_fg_fraction,
    penalise with a hinge loss to push it back up.

    fg_fraction = mean over (K, H, W) of max_over_entities entity_probs
    L_prior = relu(min_fg_fraction - fg_fraction)^2

    Without this, the model can reduce all structural losses by zeroing out
    entity_probs entirely (predicting 100% background).

    Priority 4 from analysis.md: "fg prior" to prevent entity collapse.
    """
    ep = entity_probs.float()
    # Max over entities → scalar fg probability per voxel
    fg_max = ep.max(dim=1).values  # (B, K, H, W)
    fg_fraction = fg_max.mean(dim=(1, 2, 3))  # (B,)
    # Hinge: penalise if fraction is below floor
    shortfall = (min_fg_fraction - fg_fraction).clamp(min=0.0)
    return (shortfall ** 2).mean()


def loss_permutation_consistency(
    entity_probs: torch.Tensor,   # (B, 2, K, H, W) for current frame
    entity_probs_prev: torch.Tensor,   # (B, 2, K, H, W) for previous frame
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Cross-frame entity permutation consistency.

    If entity0 in frame t corresponds to entity0 in frame t+1 (no flip),
    the spatial overlap of entity0 between frames should be high.
    If the model randomly flips entity labels between frames, this loss fires.

    For each entity n:
      overlap(n, n) = sum(p_n_t * p_n_{t-1}) / (sum(p_n_t) + sum(p_n_{t-1}) + eps)

    The loss compares:
      - Direct overlap: entity0_t with entity0_{t-1}  (should be HIGH)
      - Flipped overlap: entity0_t with entity1_{t-1}  (should be LOW)

    L_perm = relu(flip_overlap - direct_overlap + margin)
    where margin ensures direct_overlap >> flip_overlap.

    Priority 4 from analysis.md: "permutation consistency" for temporal label stability.
    """
    margin = 0.05
    ep = entity_probs.float()        # (B, 2, K, H, W)
    ep_prev = entity_probs_prev.float()  # (B, 2, K, H, W)

    def _overlap(a, b):
        """Dice-like overlap between two (B, K, H, W) volumes."""
        num = (a * b).sum(dim=(1, 2, 3))         # (B,)
        denom = a.sum(dim=(1, 2, 3)) + b.sum(dim=(1, 2, 3)) + eps
        return num / denom  # (B,) in [0, 1]

    # Direct: entity0_t vs entity0_{t-1}, entity1_t vs entity1_{t-1}
    direct_0 = _overlap(ep[:, 0], ep_prev[:, 0])
    direct_1 = _overlap(ep[:, 1], ep_prev[:, 1])
    direct = (direct_0 + direct_1) / 2.0

    # Flipped: entity0_t vs entity1_{t-1}, entity1_t vs entity0_{t-1}
    flip_0 = _overlap(ep[:, 0], ep_prev[:, 1])
    flip_1 = _overlap(ep[:, 1], ep_prev[:, 0])
    flip = (flip_0 + flip_1) / 2.0

    # Hinge: penalise when flip ≥ direct (i.e., labels are inconsistent)
    return torch.relu(flip - direct + margin).mean()


def loss_amodal_entity_coverage(
    vol_outputs,
    min_coverage: float = 0.03,
) -> torch.Tensor:
    """
    Amodal entity coverage prior: ensure BOTH entities have sufficient
    amodal (3D volumetric) presence.

    Differs from loss_fg_coverage_prior which only checks max over entities.
    This checks EACH entity independently, preventing one entity from
    "surviving" at the expense of the other going to zero.

    amodal_en = 1 - prod_k(1 - entity_probs[:, n, k]) (marginalized over depth)
    L = relu(min_coverage - mean(amodal_e0))^2 + relu(min_coverage - mean(amodal_e1))^2

    Used with four_stream guide to ensure back_e0 and back_e1 streams
    (occluded entity features) carry actual signal during training.
    """
    amo_e0 = vol_outputs.amodal.get("e0") if vol_outputs.amodal else None
    amo_e1 = vol_outputs.amodal.get("e1") if vol_outputs.amodal else None

    if amo_e0 is None or amo_e1 is None:
        # Fallback: compute from entity_probs directly
        if vol_outputs.entity_probs is None:
            return torch.tensor(0.0)
        ep = vol_outputs.entity_probs.float()   # (B, 2, K, H, W)
        amo_e0 = 1.0 - (1.0 - ep[:, 0]).prod(dim=1)   # (B, H, W)
        amo_e1 = 1.0 - (1.0 - ep[:, 1]).prod(dim=1)   # (B, H, W)

    cov_e0 = amo_e0.float().mean(dim=(-2, -1))  # (B,)
    cov_e1 = amo_e1.float().mean(dim=(-2, -1))  # (B,)

    shortfall_e0 = (min_coverage - cov_e0).clamp(min=0.0)
    shortfall_e1 = (min_coverage - cov_e1).clamp(min=0.0)
    return (shortfall_e0 ** 2 + shortfall_e1 ** 2).mean()


def loss_temporal_centroid_consistency(
    entity_probs_frames: "list[torch.Tensor]",
    margin: float = 0.05,
) -> torch.Tensor:
    """
    Centroid-based temporal entity slot consistency.

    Enforces that the spatial centroid of each entity in the 2D amodal
    field moves smoothly between frames, and that entity assignments
    are not flipped (entity0 stays entity0, entity1 stays entity1).

    For each consecutive frame pair (t, t+1):
      cost_same = || centroid_e0(t) - centroid_e0(t+1) ||^2
                + || centroid_e1(t) - centroid_e1(t+1) ||^2
      cost_swap = || centroid_e0(t) - centroid_e1(t+1) ||^2
                + || centroid_e1(t) - centroid_e0(t+1) ||^2

    L = relu(cost_swap - cost_same + margin)  per pair

    Unlike loss_permutation_consistency which uses volumetric Dice overlap,
    centroid-based loss is more robust during contact/collision frames where
    volumetric overlap is expected (both entities occupy similar spatial regions).

    entity_probs_frames: list of (B, 2, K, H, W) tensors, one per frame.
    """
    if len(entity_probs_frames) < 2:
        return torch.tensor(0.0, device=entity_probs_frames[0].device
                            if entity_probs_frames else torch.device("cpu"))

    device = entity_probs_frames[0].device
    total = torch.tensor(0.0, device=device)
    n_pairs = 0

    for t in range(len(entity_probs_frames) - 1):
        ep_t = entity_probs_frames[t].float()    # (B, 2, K, H, W)
        ep_t1 = entity_probs_frames[t + 1].float()

        # 2D amodal field: marginalize over depth
        amo_t  = 1.0 - (1.0 - ep_t ).prod(dim=2)   # (B, 2, H, W)
        amo_t1 = 1.0 - (1.0 - ep_t1).prod(dim=2)   # (B, 2, H, W)

        B, _, H, W = amo_t.shape
        grid_y = torch.arange(H, device=device, dtype=torch.float32)
        grid_x = torch.arange(W, device=device, dtype=torch.float32)

        def _centroid(amodal_2d):
            # amodal_2d: (B, 2, H, W)
            mass = amodal_2d.sum(dim=(-2, -1)).clamp(min=1e-6)  # (B, 2)
            cy = (amodal_2d * grid_y.view(1, 1, H, 1)).sum(dim=(-2, -1)) / mass
            cx = (amodal_2d * grid_x.view(1, 1, 1, W)).sum(dim=(-2, -1)) / mass
            return torch.stack([cy, cx], dim=-1)  # (B, 2, 2)

        c_t  = _centroid(amo_t)   # (B, 2, 2)  — [entity, (y, x)]
        c_t1 = _centroid(amo_t1)  # (B, 2, 2)

        # Same assignment: e0→e0, e1→e1 (want this to be CHEAP)
        cost_same = ((c_t - c_t1) ** 2).sum(dim=-1).sum(dim=-1)      # (B,)
        # Swapped assignment: e0→e1, e1→e0 (want this to be EXPENSIVE)
        cost_swap = ((c_t[:, [1, 0], :] - c_t1) ** 2).sum(dim=-1).sum(dim=-1)  # (B,)

        # Penalize when cost_same > cost_swap + margin
        # (i.e., same assignment is more expensive than swap → flip detected)
        L_pair = torch.relu(cost_same - cost_swap - margin).mean()
        total = total + L_pair
        n_pairs += 1

    return total / max(n_pairs, 1)


def compute_volume_accuracy(
    V_logits: torch.Tensor,  # (B, C, K, H, W)
    V_gt: torch.Tensor,      # (B, K, H, W)
) -> dict:
    """Per-class and overall accuracy for volume predictions."""
    with torch.no_grad():
        p_e0 = torch.sigmoid(V_logits[:, 1].float())
        p_e1 = torch.sigmoid(V_logits[:, 2].float())
        pred_class = torch.zeros_like(V_gt.long())
        has_entity = (p_e0 > 0.5) | (p_e1 > 0.5)
        pred_class = torch.where(has_entity & (p_e0 >= p_e1), torch.ones_like(pred_class), pred_class)
        pred_class = torch.where(has_entity & (p_e1 > p_e0), torch.full_like(pred_class, 2), pred_class)
        correct = (pred_class == V_gt.long())

        overall_acc = correct.float().mean().item()
        bg_mask = (V_gt == 0)
        entity_mask = (V_gt > 0)
        bg_acc = correct[bg_mask].float().mean().item() if bg_mask.any() else 1.0
        entity_acc = correct[entity_mask].float().mean().item() if entity_mask.any() else 0.0

    return {
        "overall_acc": overall_acc,
        "bg_acc": bg_acc,
        "entity_acc": entity_acc,
    }
