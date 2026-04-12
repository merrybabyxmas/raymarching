"""
Phase 53 — Explicit Decomposition Heads
=========================================

Phase 46-52 failure root causes (7 phases, all failed):
  1. OBH features uniform for identical entities (F0 ≈ F1 → overlap_proxy ≈ same everywhere)
  2. base_blend structurally inverted (higher in exclusive due to VCA alpha behavior)
  3. w0*w1 proxy assumption violated in actual data (exclusive w0*w1 > overlap w0*w1)
  4. l_occ_region_aware vs l_occ_struct competing gradients
  5. l_blend_direct_gt gradients flow through base_blend into occupancy (cross-term pollution)

Phase 53 core change — explicit decomposition:
──────────────────────────────────────────────────────────────────
A. DecompositionHeads: 4 separate heads sharing [F0, F1, Fg] input
     p0_head   : F0 → entity0 occupancy probability  (B, S)
     p1_head   : F1 → entity1 occupancy probability  (B, S)
     pov_head  : [h0, h1, hg, |h0-h1|, h0*h1] → overlap occupancy (B, S)
     pfront_head: same input → entity0-front probability  (B, S)
   Zero-init last layers → starts at 0.5 everywhere

B. Deterministic compositing (replaces weight_head as primary routing):
     w_e0_ex = relu(p0 - pov)          exclusive entity0 weight
     w_e1_ex = relu(p1 - pov)          exclusive entity1 weight
     w_e0_ov = pov * pfront            entity0 in overlap (front)
     w_e1_ov = pov * (1 - pfront)      entity1 in overlap (back)
     w0 = w_e0_ex + w_e0_ov
     w1 = w_e1_ex + w_e1_ov
     w_bg = clamp(1 - max(p0, p1), 0, 1)
     normalize(w0, w1, w_bg)

   blend_map = pov  (stored as last_blend_map for blend_sep metric)
   entity_presence = max(p0, p1).clamp(0.05, 0.95)  (actual blending gate)
   composed  = w0*F0 + w1*F1 + w_bg*Fg
   blended   = entity_presence * composed + (1 - entity_presence) * Fg

C. Losses:
     l_decomp_occ  : region-aware BCE on (p0, m0) + (p1, m1) + dice
     l_decomp_ov   : BCE(pov, m0*m1) — directly supervised overlap
     l_decomp_hier : relu(pov-p0) + relu(pov-p1) per pixel (hierarchy)
     l_decomp_front: BCE(pfront, front_target) in overlap region
     l_vis, l_wrong, l_sigma, l_depth, l_excl, l_slot_ref, l_slot_cont: kept

Why this breaks the Phase 46-52 deadlock:
  - pov_head sees [h0, h1, hg, |h0-h1|, h0*h1]: spatial position still
    discriminates overlap vs exclusive even if F0 ≈ F1 globally
  - l_decomp_ov directly trains pov → m0*m1 with simple BCE → no proxy
  - blend_sep = mean(pov in overlap) - mean(pov in exclusive) > 0 by construction
    once pov_head is trained (pov≈1 in overlap, ≈0 in exclusive)
  - occ_prod_sep = mean(p0*p1 in overlap) - mean(p0*p1 in exclusive) > 0
    once p0_head, p1_head learn exclusive vs overlap distinction
"""
from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.entity_slot_phase46 import (
    Phase46Processor,
    MultiBlockSlotManagerP46,
    inject_multi_block_entity_slot_p46,
    restore_multiblock_state_p46,
    val_score_phase46,
    l_occ_region_aware,     # reuse for l_decomp_occ
    # Constants
)
from models.entity_slot_phase45 import (
    l_occupancy,
    l_visible_weights_region_balanced,
    l_visible_iou_soft,
    collect_occupancy_stats,
    OccupancyHead,
    reroute_entity_weights_with_occupancy,
    BLOCK_INNER_DIMS,
    DEFAULT_INJECT_KEYS,
    CROSS_ATTN_DIM,
    PRIMARY_DIM,
)
from models.entity_slot_phase44 import (
    collect_blend_stats_detailed,
)
from models.entity_slot_phase40 import SlotLoRA


# =============================================================================
# DecompositionHeads
# =============================================================================

class DecompositionHeads(nn.Module):
    """
    4 explicit decomposition heads sharing projected [F0, F1, Fg] features.

    p0_head   : F0             → entity0 occupancy probability  (B, S)
    p1_head   : F1             → entity1 occupancy probability  (B, S)
    pov_head  : [h0,h1,hg,Δ,⊗] → overlap probability           (B, S)
    pfront_head: same input    → entity0-front probability       (B, S)

    Shared feature projectors reduce dimension before joint heads:
      proj_dim=32 × 5 components → combined_dim=160

    All last-layer weights/biases zero-initialized → sigmoid(0)=0.5 everywhere.
    """

    def __init__(self, in_dim: int, proj_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.proj_dim = proj_dim

        # Shared projectors: F0, F1, Fg → lower-dim representations
        self.proj0 = nn.Linear(in_dim, proj_dim)
        self.proj1 = nn.Linear(in_dim, proj_dim)
        self.projg = nn.Linear(in_dim, proj_dim)

        combined_dim = proj_dim * 5  # [h0, h1, hg, |h0-h1|, h0*h1]

        # Per-entity heads: only see own features (identity-preserving)
        self.p0_head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.p1_head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))

        # Joint heads: see comparison between both entities
        self.pov_head = nn.Sequential(
            nn.Linear(combined_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.pfront_head = nn.Sequential(
            nn.Linear(combined_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))

        # Zero-init all last layers → sigmoid(0) = 0.5 everywhere
        for head in (self.p0_head, self.p1_head, self.pov_head, self.pfront_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        F0: torch.Tensor,   # (B, S, D)
        F1: torch.Tensor,   # (B, S, D)
        Fg: torch.Tensor,   # (B, S, D)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (p0, p1, pov, pfront) — all (B, S) in [0, 1].

        p0, p1 : per-entity occupancy probabilities
        pov    : overlap probability (should be ≤ min(p0, p1))
        pfront : probability that entity0 is in front in overlap
        """
        F0f = F0.float()
        F1f = F1.float()
        Fgf = Fg.float()

        # Per-entity predictions from own features
        p0 = torch.sigmoid(self.p0_head(F0f).squeeze(-1))   # (B, S)
        p1 = torch.sigmoid(self.p1_head(F1f).squeeze(-1))   # (B, S)

        # Joint predictions from projected comparison features
        h0 = self.proj0(F0f)                                  # (B, S, proj_dim)
        h1 = self.proj1(F1f)
        hg = self.projg(Fgf)
        combined = torch.cat([h0, h1, hg, (h0 - h1).abs(), h0 * h1], dim=-1)  # (B, S, 5*proj_dim)

        pov    = torch.sigmoid(self.pov_head(combined).squeeze(-1))    # (B, S)
        pfront = torch.sigmoid(self.pfront_head(combined).squeeze(-1)) # (B, S)

        return p0, p1, pov, pfront


# =============================================================================
# Deterministic compositing
# =============================================================================

def decompose_entity_weights(
    p0:     torch.Tensor,   # (B, S) entity0 occupancy
    p1:     torch.Tensor,   # (B, S) entity1 occupancy
    pov:    torch.Tensor,   # (B, S) overlap occupancy
    pfront: torch.Tensor,   # (B, S) entity0-front probability
    eps:    float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Deterministic compositing from decomposition heads.

    Returns w0, w1, w_bg — all (B, S), non-negative, sum to 1.

    Algebra:
      exclusive0 contribution: entity0 is present but overlap is not
      exclusive1 contribution: entity1 is present but overlap is not
      overlap0 contribution:   overlap, entity0 in front
      overlap1 contribution:   overlap, entity1 in front

    The overlap mass is stabilized by the learned head and the product proxy:
      overlap_mass = max(pov, p0 * p1)

    This keeps the routing grounded in the structurally reliable occupancy-product
    signal while still letting the dedicated overlap head help when it is better.
    """
    overlap_mass = torch.maximum(pov, (p0 * p1).clamp(0.0, 1.0))
    union_mass = (p0 + p1 - overlap_mass).clamp(0.0, 1.0)

    # Exclusive contributions: entity present but overlap not
    w_e0_ex = F.relu(p0 - overlap_mass)        # entity0 exclusive weight
    w_e1_ex = F.relu(p1 - overlap_mass)        # entity1 exclusive weight

    # Overlap contributions: depth-ordered
    w_e0_ov = overlap_mass * pfront             # entity0 in overlap (front weight)
    w_e1_ov = overlap_mass * (1.0 - pfront)    # entity1 in overlap (back weight)

    w0   = w_e0_ex + w_e0_ov
    w1   = w_e1_ex + w_e1_ov
    w_bg = (1.0 - union_mass).clamp(0.0, 1.0)

    # Normalize so weights sum to 1
    total = (w0 + w1 + w_bg).clamp(min=eps)
    return w0 / total, w1 / total, w_bg / total


# =============================================================================
# Phase 53 Loss functions
# =============================================================================

def l_decomp_occ(
    p0:         torch.Tensor,   # (B, S) entity0 occupancy
    p1:         torch.Tensor,   # (B, S) entity1 occupancy
    masks_BNS:  torch.Tensor,   # (B, 2, S) GT mask
    la_ov:      float = 8.0,    # push BOTH high in overlap
    la_ex_pos:  float = 4.0,    # push CORRECT entity high in exclusive
    la_ex_neg:  float = 4.0,    # push WRONG entity low in exclusive
    la_bg:      float = 0.5,    # push BOTH low in background
    eps:        float = 1e-6,
) -> torch.Tensor:
    """
    Region-aware occupancy BCE for decomposition heads.

    Identical to l_occ_region_aware but with explicit parameter naming.
    Applied to (p0, p1) output from DecompositionHeads.
    """
    return l_occ_region_aware(
        p0, p1, masks_BNS,
        la_ov=la_ov, la_ex_pos=la_ex_pos, la_ex_neg=la_ex_neg, la_bg=la_bg, eps=eps)


def l_decomp_ov(
    pov:        torch.Tensor,   # (B, S) overlap probability
    masks_BNS:  torch.Tensor,   # (B, 2, S) GT mask
    la:         float = 5.0,
    eps:        float = 1e-6,
) -> torch.Tensor:
    """
    BCE supervision of overlap head using GT overlap mask.

    target = m0 * m1 (1 only where BOTH entities present)
    l_decomp_ov = -la * (m0*m1 * log(pov) + (1-m0*m1) * log(1-pov)) / N

    At pov=0.5 everywhere (init):
      l ≈ -la * log(0.5) ≈ 3.47 (for la=5)
    At pov→1 in overlap, pov→0 elsewhere:
      l → 0

    Why this works for identical entities:
      Even if F0 ≈ F1, spatial location of features differs.
      pov_head input includes |h0-h1| and h0*h1 → spatial position encodes overlap.
    """
    m0 = masks_BNS[:, 0, :].float().to(pov.device)
    m1 = masks_BNS[:, 1, :].float().to(pov.device)
    gt_ov = m0 * m1      # (B, S) GT overlap mask

    pov_c = pov.float().clamp(eps, 1.0 - eps)
    n = gt_ov.numel() + eps

    # Class-balanced BCE
    n_pos = gt_ov.sum() + eps
    n_neg = (1.0 - gt_ov).sum() + eps

    l_pos = -(gt_ov * torch.log(pov_c)).sum() / n_pos
    l_neg = -((1.0 - gt_ov) * torch.log(1.0 - pov_c)).sum() / n_neg

    # Soft Dice for overlap
    inter = (pov_c * gt_ov).sum()
    dice  = 1.0 - (2.0 * inter + eps) / (pov_c.sum() + gt_ov.sum() + eps)

    return la * (0.5 * l_pos + 0.25 * l_neg + 0.25 * dice)


def l_decomp_hier(
    p0:        torch.Tensor,   # (B, S)
    p1:        torch.Tensor,   # (B, S)
    pov:       torch.Tensor,   # (B, S)
    masks_BNS: torch.Tensor,   # (B, 2, S) GT mask — for region weighting
    la:        float = 2.0,
    eps:       float = 1e-6,
) -> torch.Tensor:
    """
    Hierarchy constraint: pov ≤ min(p0, p1).

    Physically: overlap probability cannot exceed individual entity probabilities.
    If pov > p0: entity0 is "more overlapping than it is present" — impossible.

    relu(pov - p0) + relu(pov - p1) per pixel, averaged over GT overlap region.

    Gradient: if pov > p0, gradient pushes pov DOWN and p0 UP simultaneously.
    """
    m0 = masks_BNS[:, 0, :].float().to(pov.device)
    m1 = masks_BNS[:, 1, :].float().to(pov.device)
    ov = m0 * m1
    n  = ov.sum() + eps

    # Violations: penalize pov > p0 and pov > p1
    viol_0 = F.relu(pov.float() - p0.float())   # (B, S) violation for entity0
    viol_1 = F.relu(pov.float() - p1.float())   # (B, S) violation for entity1

    # Weighted by overlap region (hierarchy matters most in overlap)
    l = la * ((viol_0 + viol_1) * ov).sum() / n

    # Also apply globally (unweighted) with lower weight
    n_all = pov.numel() + eps
    l_global = (la * 0.1) * (viol_0 + viol_1).sum() / n_all

    return l + l_global


def l_decomp_front(
    pfront:        torch.Tensor,   # (B, S) entity0-front probability
    depth_orders_B: list,          # list of depth order tuples
    masks_BNS:     torch.Tensor,   # (B, 2, S) GT mask
    la:            float = 3.0,
    eps:           float = 1e-6,
) -> torch.Tensor:
    """
    Supervise front probability using depth order GT.

    In overlap region (m0*m1=1):
      If entity0 is front (depth_order[0]=0): pfront → 1 (BCE target=0.9)
      If entity1 is front (depth_order[0]=1): pfront → 0 (BCE target=0.1)

    Only applied in overlap region. No-op when overlap is empty.
    """
    _FRONT_SOFT = 0.9
    _BACK_SOFT  = 0.1

    m0  = masks_BNS[:, 0, :].float().to(pfront.device)
    m1  = masks_BNS[:, 1, :].float().to(pfront.device)
    ov  = m0 * m1   # (B, S)
    n_ov = ov.sum() + eps

    if n_ov.item() < 1.0:
        return pfront.new_zeros(())

    pf = pfront.float().clamp(eps, 1.0 - eps)
    B  = min(pfront.shape[0], len(depth_orders_B))

    total_loss = pfront.new_zeros(())
    n_valid    = 0

    for b in range(B):
        do = depth_orders_B[b]
        front_entity = int(do[0]) if len(do) > 0 else 0
        # front_entity==0 → entity0 in front → pfront target = FRONT_SOFT
        # front_entity==1 → entity1 in front → pfront target = BACK_SOFT
        target = _FRONT_SOFT if front_entity == 0 else _BACK_SOFT

        ov_b = ov[b]
        n_b  = ov_b.sum().clamp(min=eps)
        if n_b.item() < 1.0:
            continue

        l_b = -(
            target         * torch.log(pf[b])
          + (1.0 - target) * torch.log(1.0 - pf[b])
        )
        total_loss = total_loss + (l_b * ov_b).sum() / n_b
        n_valid += 1

    if n_valid == 0:
        return pfront.new_zeros(())

    return la * total_loss / float(n_valid)


def l_decomp_presence(
    p0:        torch.Tensor,
    p1:        torch.Tensor,
    pov:       torch.Tensor,
    masks_BNS: torch.Tensor,
    la:        float = 2.0,
    eps:       float = 1e-6,
) -> torch.Tensor:
    """
    Supervise the union-presence gate used for final compositing.

    This targets the stable foreground/background split:
      presence = p0 + p1 - overlap
    where overlap is the stabilized overlap proxy.
    """
    m0 = masks_BNS[:, 0, :].float().to(p0.device)
    m1 = masks_BNS[:, 1, :].float().to(p0.device)
    gt_union = torch.maximum(m0, m1)

    overlap_mass = torch.maximum(pov.float(), (p0.float() * p1.float()).clamp(0.0, 1.0))
    pred_union = (p0.float() + p1.float() - overlap_mass).clamp(0.0, 1.0)
    pred_c = pred_union.clamp(eps, 1.0 - eps)

    n_pos = gt_union.sum() + eps
    n_neg = (1.0 - gt_union).sum() + eps
    l_pos = -(gt_union * torch.log(pred_c)).sum() / n_pos
    l_neg = -((1.0 - gt_union) * torch.log(1.0 - pred_c)).sum() / n_neg
    return la * (0.5 * l_pos + 0.5 * l_neg)


def l_pair_identity_preservation(
    F0_slot:     torch.Tensor,
    F1_slot:     torch.Tensor,
    F0_ref:      torch.Tensor,
    F1_ref:      torch.Tensor,
    mask_e0:     torch.Tensor,
    mask_e1:     torch.Tensor,
    margin:      float = 0.15,
    pair_target: float = 0.20,
    pair_weight: float = 0.50,
    eps:         float = 1e-6,
) -> torch.Tensor:
    """
    Strong anti-collapse objective for real object identity preservation.

    The teacher-gap part keeps slot0 close to the cat teacher and slot1 close to
    the dog teacher, while the direct pair penalty prevents both slots from
    collapsing onto the same object manifold.

    Weighted more heavily in overlap regions, where collapse is hardest to
    diagnose from routing alone.
    """
    F0_s = F.normalize(F0_slot.float(), dim=-1)
    F1_s = F.normalize(F1_slot.float(), dim=-1)
    F0_r = F.normalize(F0_ref.float().detach(), dim=-1)
    F1_r = F.normalize(F1_ref.float().detach(), dim=-1)

    m0 = mask_e0.float().unsqueeze(-1).to(F0_s.device)
    m1 = mask_e1.float().unsqueeze(-1).to(F0_s.device)
    fg = torch.maximum(mask_e0.float(), mask_e1.float()).unsqueeze(-1).to(F0_s.device)
    ov = (mask_e0.float() * mask_e1.float()).unsqueeze(-1).to(F0_s.device)

    n0 = m0.sum() + eps
    n1 = m1.sum() + eps
    n_fg = fg.sum() + eps

    # Teacher gap: slot0 should like its own teacher more than the opposite teacher
    sim_00 = (F0_s * F0_r).sum(-1, keepdim=True)
    sim_01 = (F0_s * F1_r).sum(-1, keepdim=True)
    sim_11 = (F1_s * F1_r).sum(-1, keepdim=True)
    sim_10 = (F1_s * F0_r).sum(-1, keepdim=True)

    l0 = (F.relu(sim_01 - sim_00 + margin) * m0).sum() / n0
    l1 = (F.relu(sim_10 - sim_11 + margin) * m1).sum() / n1
    teacher_gap = 0.5 * (l0 + l1)

    # Direct collapse penalty: slots should not become too similar to each other.
    slot_sim = (F0_s * F1_s).sum(-1, keepdim=True)
    pair_mask = (0.50 * fg + 1.50 * ov).clamp(min=0.0)
    pair_loss = (F.relu(slot_sim - pair_target) * pair_mask).sum() / (
        pair_mask.sum() + eps
    )

    # Slightly encourage the pair to remain balanced in foreground regions so
    # one object does not absorb all structure while the other becomes background.
    mean_0 = (sim_00 * m0).sum() / n0
    mean_1 = (sim_11 * m1).sum() / n1
    balance = F.relu((mean_0 - mean_1).abs() - 0.15)

    return teacher_gap + pair_weight * pair_loss + 0.25 * balance


def l_solo_masked_reconstruction(
    pred_x0:    torch.Tensor,
    target_x0:  torch.Tensor,
    mask_B1HW:  torch.Tensor,
    la:         float = 1.0,
    eps:        float = 1e-6,
) -> torch.Tensor:
    """
    Masked reconstruction loss for solo renders.

    Only the visible object region contributes, so background appearance does
    not dominate the teacher signal.
    """
    pred = pred_x0.float()
    target = target_x0.float()
    mask = mask_B1HW.float()

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)

    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
    if mask.shape[0] != pred.shape[0]:
        mask = mask.expand(pred.shape[0], -1, -1, -1)
    mask_mass = mask[:, :1].sum()
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)

    diff = (pred - target).pow(2)
    masked = diff * mask
    denom = mask_mass * max(pred.shape[1], 1)
    return la * masked.sum() / (denom + eps)


def l_output_entity_divergence(
    out_e0:    torch.Tensor,   # (B, S, D) entity0 output
    out_e1:    torch.Tensor,   # (B, S, D) entity1 output
    masks_BNS: torch.Tensor,   # (B, 2, S) GT mask
    la:        float = 1.0,
    eps:       float = 1e-6,
) -> torch.Tensor:
    """
    Push entity-specific outputs to diverge in exclusive regions.

    In e0-exclusive: out_e0 should differ from out_e1 (cat ≠ dog)
    In e1-exclusive: out_e1 should differ from out_e0

    This directly addresses the Phase54 bottleneck: even if F_0 ≈ F_1 (similar
    cross-attention features for identical entities), the entity-specific output
    LoRA projections should produce DIFFERENT outputs.

    Loss = cosine_similarity(out_e0, out_e1) in exclusive regions (minimize it).
    """
    m0 = masks_BNS[:, 0, :].float().to(out_e0.device)
    m1 = masks_BNS[:, 1, :].float().to(out_e0.device)
    e0_ex = m0 * (1.0 - m1)
    e1_ex = m1 * (1.0 - m0)

    o0n = F.normalize(out_e0.float(), dim=-1)
    o1n = F.normalize(out_e1.float(), dim=-1)
    sim = (o0n * o1n).sum(-1)    # (B, S) cosine similarity

    n_e0 = e0_ex.sum() + eps
    n_e1 = e1_ex.sum() + eps

    l_e0 = (sim * e0_ex).sum() / n_e0
    l_e1 = (sim * e1_ex).sum() / n_e1

    return la * 0.5 * (l_e0 + l_e1)


def l_route_entropy(
    w0:        torch.Tensor,
    w1:        torch.Tensor,
    w_bg:      torch.Tensor,
    masks_BNS: torch.Tensor,
    la:        float = 0.2,
    eps:       float = 1e-6,
) -> torch.Tensor:
    """
    Encourage peaky deterministic routing.

    This is a weak auxiliary term that reduces the soft, hazy routing pattern
    by pushing the 3-way compositing distribution toward low entropy.
    """
    m0 = masks_BNS[:, 0, :].float().to(w0.device)
    m1 = masks_BNS[:, 1, :].float().to(w0.device)
    fg = torch.maximum(m0, m1)
    bg = (1.0 - m0) * (1.0 - m1)
    probs = torch.stack([w0.float(), w1.float(), w_bg.float()], dim=-1).clamp(eps, 1.0 - eps)
    ent = -(probs * torch.log(probs)).sum(dim=-1)
    weight = (0.75 * fg + 0.25 * bg).clamp(min=0.0)
    return la * (ent * weight).sum() / (weight.sum() + eps)


# =============================================================================
# Phase53Processor
# =============================================================================

class Phase53Processor(Phase46Processor):
    """
    Phase 53: Explicit Decomposition Heads for entity compositing.

    Inherits full Phase46Processor infrastructure (adapters, LoRA, VCA,
    weight_head, occ_head pair, overlap_blend_head) — all loaded from
    Phase52 checkpoint and kept for Stage B fine-tuning.

    KEY ADDITION: DecompositionHeads (fresh, zero-initialized)
      p0_head, p1_head: per-entity occupancy (replaces occ_head_e0/e1 as primary)
      pov_head:         overlap probability   (replaces overlap_blend_head as blend metric)
      pfront_head:      front probability     (replaces VCA-based e0_front)

    ROUTING: Deterministic compositing via decompose_entity_weights().
    BLEND:   blend_map = pov (stored as last_blend_map for blend_sep metric)
    ACTUAL BLEND GATE: entity_presence = max(p0, p1) (preserves entity features
                       in exclusive regions, unlike pov which → 0 there)
    """

    def __init__(
        self,
        query_dim:            int,
        vca_layer             = None,
        entity_ctx:           Optional[torch.Tensor] = None,
        slot_blend_init:      float = 0.3,
        inner_dim:            Optional[int] = None,
        adapter_rank:         int   = 64,
        use_blend_head:       bool  = True,
        lora_rank:            int   = 4,
        cross_attention_dim:  int   = CROSS_ATTN_DIM,
        weight_head_hidden:   int   = 32,
        primary_dim:          int   = PRIMARY_DIM,
        proj_hidden:          int   = 256,
        obh_hidden:           int   = 32,
        occ_hidden:           int   = 64,
        decomp_proj_dim:      int   = 32,
        decomp_hidden:        int   = 64,
        use_decomp_heads:     bool  = True,
    ):
        super().__init__(
            query_dim           = query_dim,
            vca_layer           = vca_layer,
            entity_ctx          = entity_ctx,
            slot_blend_init     = slot_blend_init,
            inner_dim           = inner_dim,
            adapter_rank        = adapter_rank,
            use_blend_head      = use_blend_head,
            lora_rank           = lora_rank,
            cross_attention_dim = cross_attention_dim,
            weight_head_hidden  = weight_head_hidden,
            primary_dim         = primary_dim,
            proj_hidden         = proj_hidden,
            obh_hidden          = obh_hidden,
            occ_hidden          = occ_hidden,
        )
        eff_dim = inner_dim if inner_dim is not None else query_dim

        self.use_decomp_heads = use_decomp_heads
        if use_decomp_heads:
            self.decomp_heads = DecompositionHeads(
                in_dim   = eff_dim,
                proj_dim = decomp_proj_dim,
                hidden   = decomp_hidden,
            )
        else:
            self.decomp_heads = None

        # Entity-specific K/V LoRA: each entity gets its own text-to-key/value
        # transformation. "cat" and "dog" tokens produce DIFFERENT K/V through
        # lora_k_e0 vs lora_k_e1, making F_0 ≠ F_1 even for similar entities.
        self.lora_k_e0 = SlotLoRA(cross_attention_dim, eff_dim, rank=lora_rank)
        self.lora_k_e1 = SlotLoRA(cross_attention_dim, eff_dim, rank=lora_rank)
        self.lora_v_e0 = SlotLoRA(cross_attention_dim, eff_dim, rank=lora_rank)
        self.lora_v_e1 = SlotLoRA(cross_attention_dim, eff_dim, rank=lora_rank)

        # Entity-specific output LoRA: each entity gets its own output
        # projection so "cat" and "dog" can diverge even when F_0 ≈ F_1.
        self.lora_out_e0 = SlotLoRA(eff_dim, eff_dim, rank=lora_rank)
        self.lora_out_e1 = SlotLoRA(eff_dim, eff_dim, rank=lora_rank)

        # Phase 53 storage
        self.last_p0_for_loss:     Optional[torch.Tensor] = None
        self.last_p1_for_loss:     Optional[torch.Tensor] = None
        self.last_pov_for_loss:    Optional[torch.Tensor] = None
        self.last_pfront_for_loss: Optional[torch.Tensor] = None
        self.last_presence_for_loss: Optional[torch.Tensor] = None
        self.last_wbg: Optional[torch.Tensor] = None
        self.last_out_e0: Optional[torch.Tensor] = None
        self.last_out_e1: Optional[torch.Tensor] = None
        self._debug_shape_logged = False
        self._phase53_key: Optional[str] = None

    def reset_slot_store(self):
        super().reset_slot_store()
        self.last_p0_for_loss     = None
        self.last_p1_for_loss     = None
        self.last_pov_for_loss    = None
        self.last_pfront_for_loss = None
        self.last_presence_for_loss = None
        self.last_wbg = None
        self.last_out_e0 = None
        self.last_out_e1 = None

    def decomp_head_params(self) -> List[nn.Parameter]:
        if self.decomp_heads is None:
            return []
        return list(self.decomp_heads.parameters())

    def entity_lora_kv_params(self) -> List[nn.Parameter]:
        return (list(self.lora_k_e0.parameters()) + list(self.lora_k_e1.parameters())
              + list(self.lora_v_e0.parameters()) + list(self.lora_v_e1.parameters()))

    def entity_lora_out_params(self) -> List[nn.Parameter]:
        return list(self.lora_out_e0.parameters()) + list(self.lora_out_e1.parameters())

    # ------------------------------------------------------------------
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask        = None,
        temb                  = None,
        **kwargs,
    ):
        """
        Phase53 forward: decomposition heads as primary routing.

        Flow:
          1. Compute F0, F1, Fg (masked attention + slot adapters) — same as before
          2. Run VCA sigma (for Stage B l_sigma / l_depth) — same as before
          3. Run DecompositionHeads → p0, p1, pov, pfront
          4. Deterministic compositing → w0, w1, w_bg
          5. composed = w0*F0 + w1*F1 + w_bg*Fg
          6. entity_presence = max(p0, p1).clamp(0.05, 0.95)  [actual blend gate]
          7. blended = entity_presence * composed + (1-entity_presence) * Fg
          8. Store last_blend_map = pov  [for blend_sep metric]
        """
        B, S, D   = hidden_states.shape
        dtype     = hidden_states.dtype
        enc_hs    = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        if enc_hs.shape[0] == 1 and B > 1:
            # AnimateDiff keeps text conditioning batch=1 while the video hidden
            # states are frame-flattened to batch=B. Broadcast text to each frame.
            enc_hs = enc_hs.expand(B, -1, -1).contiguous()
        T_seq     = enc_hs.shape[1]

        inner_dim = attn.to_q.weight.shape[0]
        n_heads   = attn.heads
        head_dim  = inner_dim // n_heads
        scale     = head_dim ** -0.5

        def _mh(x, seq_len):
            return x.reshape(B, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)

        # ── K, V with LoRA (global — for F_g) ────────────────────────────
        enc_hs_f = enc_hs.float()
        k = attn.to_k(enc_hs) + self.lora_k(enc_hs_f).to(dtype=enc_hs.dtype)
        v = attn.to_v(enc_hs) + self.lora_v(enc_hs_f).to(dtype=enc_hs.dtype)
        k_mh = _mh(k, T_seq)
        v_mh = _mh(v, T_seq)

        # ── Entity-specific K, V (each entity sees text through its own lens) ──
        k_e0 = attn.to_k(enc_hs) + self.lora_k_e0(enc_hs_f).to(dtype=enc_hs.dtype)
        v_e0 = attn.to_v(enc_hs) + self.lora_v_e0(enc_hs_f).to(dtype=enc_hs.dtype)
        k_e1 = attn.to_k(enc_hs) + self.lora_k_e1(enc_hs_f).to(dtype=enc_hs.dtype)
        v_e1 = attn.to_v(enc_hs) + self.lora_v_e1(enc_hs_f).to(dtype=enc_hs.dtype)
        k_e0_mh = _mh(k_e0, T_seq)
        v_e0_mh = _mh(v_e0, T_seq)
        k_e1_mh = _mh(k_e1, T_seq)
        v_e1_mh = _mh(v_e1, T_seq)

        q    = attn.to_q(hidden_states)
        q_mh = _mh(q, S)

        # ── Global attention (shared K/V) ─────────────────────────────────
        scores_g = torch.matmul(q_mh, k_mh.transpose(-1, -2)) * scale
        w_g      = scores_g.softmax(dim=-1)
        F_g = (torch.matmul(w_g, v_mh).permute(0, 2, 1, 3).reshape(B, S, inner_dim))

        # ── Entity slot attention (entity-specific K/V) ───────────────────
        F_0_raw = self._masked_attn(
            q_mh, k_e0_mh, v_e0_mh, self.toks_e0, T_seq,
            scale, B, S, n_heads, head_dim, inner_dim, fallback=F_g)
        F_1_raw = self._masked_attn(
            q_mh, k_e1_mh, v_e1_mh, self.toks_e1, T_seq,
            scale, B, S, n_heads, head_dim, inner_dim, fallback=F_g)

        F_0 = self.slot0_adapter(F_0_raw.float()).to(dtype)
        F_1 = self.slot1_adapter(F_1_raw.float()).to(dtype)

        # ── VCA sigma (kept for l_sigma, l_depth in Stage B) ─────────────
        sigma = None
        if self.vca is not None and self.entity_ctx is not None:
            ctx = self.entity_ctx.expand(B, -1, -1).to(dtype)
            if self.training:
                _ = self.vca(hidden_states.float(), ctx.float())
            else:
                with torch.no_grad():
                    _ = self.vca(hidden_states.float(), ctx.float())
            sigma = getattr(self.vca, 'last_sigma', None)
            if sigma is not None:
                self.last_sigma = sigma.detach().float()
                sigma_raw = getattr(self.vca, 'last_sigma_raw', None) if self.training else sigma
                if self.training and sigma_raw is not None:
                    self.sigma_acc.append(sigma_raw.float())
                    sigma = sigma_raw

        # ── Store VCA sigma features for Stage B losses ───────────────────
        alpha_0 = alpha_1 = e0_front_vca = None
        if sigma is not None and sigma.shape[:2] == (B, S):
            sig = sigma.to(device=F_g.device, dtype=torch.float32)
            alpha_0     = sig[:, :, 0, :].max(dim=-1).values
            alpha_1     = sig[:, :, 1, :].max(dim=-1).values
            e0_front_vca = torch.sigmoid(5.0 * (sig[:, :, 0, 0] - sig[:, :, 1, 0]))

            # weight_head delta (for l_w_residual in Stage B)
            feat = torch.stack([
                alpha_0, alpha_1,
                sig[:, :, 0, 0],
                sig[:, :, 0, min(1, sig.shape[-1]-1)],
                sig[:, :, 1, 0],
                sig[:, :, 1, min(1, sig.shape[-1]-1)],
                alpha_0 * alpha_1, e0_front_vca,
            ], dim=-1).float()
            delta = self.weight_head(feat)
            self.last_w_delta = delta
        else:
            self.last_w_delta = None

        # ── Phase 53: Decomposition heads (PRIMARY ROUTING) ───────────────
        if (self.use_decomp_heads and self.decomp_heads is not None):
            p0, p1, pov, pfront = self.decomp_heads(
                F_0.float(), F_1.float(), F_g.float())

            overlap_mass = torch.maximum(pov, (p0 * p1).clamp(0.0, 1.0))
            union_mass = (p0 + p1 - overlap_mass).clamp(0.0, 1.0)

            # Store for loss computation
            self.last_p0_for_loss     = p0
            self.last_p1_for_loss     = p1
            self.last_pov_for_loss    = overlap_mass
            self.last_pfront_for_loss = pfront
            self.last_presence_for_loss = union_mass

            # Backward-compat: expose p0/p1 as legacy occ_head outputs
            self.last_o0_for_loss = p0
            self.last_o1_for_loss = p1
            self.last_o0          = p0.detach()
            self.last_o1          = p1.detach()

            # Deterministic compositing
            w0, w1, w_bg = decompose_entity_weights(p0, p1, overlap_mass, pfront)
            self.last_wbg = w_bg.detach()

            w0_f  = w0.unsqueeze(-1).to(dtype=F_g.dtype)
            w1_f  = w1.unsqueeze(-1).to(dtype=F_g.dtype)
            wbg_f = w_bg.unsqueeze(-1).to(dtype=F_g.dtype)
            composed = w0_f * F_0 + w1_f * F_1 + wbg_f * F_g

            # Actual blend gate: entity_presence = max(p0, p1)
            # Using this (not pov) preserves entity features in exclusive regions.
            entity_presence = union_mass.clamp(0.05, 0.98)
            blend_map_f = entity_presence.unsqueeze(-1).to(dtype=F_g.dtype)

            # Store stabilized overlap proxy for blend_sep / routing diagnostics.
            self.last_blend_map_for_loss = overlap_mass
            self.last_blend_map          = overlap_mass.detach()
            self.last_blend              = overlap_mass.detach()
            self.last_base_blend         = entity_presence.detach()  # kept for compat
            self.last_overlap_proxy      = overlap_mass.detach()

        elif sigma is not None and sigma.shape[:2] == (B, S):
            # Fallback: Phase46-style routing when decomp_heads disabled
            base_w0  = e0_front_vca * alpha_0 + (1.0 - e0_front_vca) * alpha_0 * (1.0 - alpha_1)
            base_w1  = (1.0 - e0_front_vca) * alpha_1 + e0_front_vca * alpha_1 * (1.0 - alpha_0)
            base_wbg = (1.0 - base_w0 - base_w1).clamp(min=0.0)

            base_logits = torch.log(
                torch.stack([base_wbg, base_w0, base_w1], dim=-1).clamp(min=1e-6))
            probs   = (base_logits + delta).softmax(dim=-1)
            w_bg, w0, w1 = probs.unbind(dim=-1)

            w0_f  = w0.unsqueeze(-1).to(dtype=F_g.dtype)
            w1_f  = w1.unsqueeze(-1).to(dtype=F_g.dtype)
            wbg_f = w_bg.unsqueeze(-1).to(dtype=F_g.dtype)
            composed = w0_f * F_0 + w1_f * F_1 + wbg_f * F_g

            from models.entity_slot_phase46 import compute_base_blend_v3
            o0_for_loss = self.occ_head_e0(F_0.float()) if self.occ_head_e0 else torch.full((B, S), 0.5, device=F_g.device)
            o1_for_loss = self.occ_head_e1(F_1.float()) if self.occ_head_e1 else torch.full((B, S), 0.5, device=F_g.device)
            self.last_o0_for_loss = o0_for_loss
            self.last_o1_for_loss = o1_for_loss
            self.last_o0          = o0_for_loss.detach()
            self.last_o1          = o1_for_loss.detach()
            self.last_wbg         = w_bg.detach()

            base_blend = compute_base_blend_v3(o0_for_loss, o1_for_loss)
            from models.entity_slot_phase45 import OccupancyHead
            feat_blend = torch.stack([o0_for_loss, o1_for_loss, o0_for_loss*o1_for_loss,
                                       e0_front_vca, w0, w1, w_bg, (o0_for_loss-o1_for_loss).abs()],
                                      dim=-1).float()
            delta_b    = self.overlap_blend_head(feat_blend)
            blend_logit = torch.logit(base_blend, eps=1e-6).unsqueeze(-1)
            blend_map   = torch.sigmoid(blend_logit + delta_b)
            self.last_blend_map_for_loss = blend_map.squeeze(-1)
            self.last_blend_map          = blend_map.squeeze(-1).detach()
            self.last_blend              = self.last_blend_map
            self.last_base_blend         = base_blend.detach()
            self.last_overlap_proxy      = (o0_for_loss * o1_for_loss).detach()
            blend_map_f = blend_map.to(dtype=F_g.dtype)

        else:
            # No sigma path: uniform routing fallback
            w0  = torch.ones(B, S, device=F_g.device) / 3
            w1  = torch.ones(B, S, device=F_g.device) / 3
            w_bg = torch.ones(B, S, device=F_g.device) / 3
            w0_f  = w0.unsqueeze(-1).to(dtype=F_g.dtype)
            w1_f  = w1.unsqueeze(-1).to(dtype=F_g.dtype)
            wbg_f = w_bg.unsqueeze(-1).to(dtype=F_g.dtype)
            blend_map_f = self.slot_blend.to(dtype=F_g.dtype)
            self.last_blend_map_for_loss = None
            self.last_blend_map          = None
            self.last_blend              = None
            self.last_wbg               = None
            self.last_o0_for_loss = None
            self.last_o1_for_loss = None
            self.last_o0 = None
            self.last_o1 = None
            self.last_presence_for_loss = None

        # ── Final blend (feature-level composition, same path as codex) ─────
        blended = blend_map_f * composed + (1.0 - blend_map_f) * F_g

        # ── Store outputs ─────────────────────────────────────────────────
        self.last_w0 = w0
        self.last_w1 = w1
        if alpha_0 is not None:
            self.last_alpha0 = alpha_0
            self.last_alpha1 = alpha_1
        self.last_F0      = F_0
        self.last_F1      = F_1
        self.last_Fg      = F_g
        self.last_blended = blended

        # Entity-specific output LoRA: compute diverged outputs for loss only.
        # These don't participate in the main forward path (stable backward compat).
        # The divergence loss trains lora_out_e0/e1 to differentiate entities.
        self.last_out_e0 = self.lora_out_e0(F_0.float())
        self.last_out_e1 = self.lora_out_e1(F_1.float())

        out = (attn.to_out[0](blended)
               + self.lora_out(blended.float()).to(dtype=blended.dtype))
        out = attn.to_out[1](out)
        return out.to(dtype)


# =============================================================================
# Multi-block injection for Phase 53
# =============================================================================

def inject_multi_block_entity_slot_p53(
    pipe,
    vca_layer,
    entity_ctx,
    inject_keys         = None,
    primary_idx:   int  = 1,
    slot_blend_init: float = 0.3,
    adapter_rank:  int  = 64,
    lora_rank:     int  = 4,
    use_blend_head: bool = True,
    weight_head_hidden: int = 32,
    proj_hidden:   int  = 256,
    obh_hidden:    int  = 32,
    occ_hidden:    int  = 64,
    decomp_proj_dim: int = 32,
    decomp_hidden:   int = 64,
) -> Tuple[List[Phase53Processor], dict]:
    """Injects Phase53Processor into UNet."""
    if inject_keys is None:
        inject_keys = DEFAULT_INJECT_KEYS

    unet       = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))
    new_procs  = dict(unet.attn_processors)
    procs: List[Phase53Processor] = []

    for i, key in enumerate(inject_keys):
        inner_dim  = BLOCK_INNER_DIMS.get(key, 640)
        is_primary = (i == primary_idx)
        proc = Phase53Processor(
            query_dim           = inner_dim,
            vca_layer           = vca_layer if is_primary else None,
            entity_ctx          = entity_ctx if is_primary else None,
            slot_blend_init     = slot_blend_init,
            inner_dim           = inner_dim,
            adapter_rank        = adapter_rank,
            use_blend_head      = (use_blend_head and is_primary),
            lora_rank           = lora_rank,
            cross_attention_dim = CROSS_ATTN_DIM,
            weight_head_hidden  = weight_head_hidden,
            primary_dim         = PRIMARY_DIM,
            proj_hidden         = proj_hidden,
            obh_hidden          = obh_hidden,
            occ_hidden          = occ_hidden,
            decomp_proj_dim     = decomp_proj_dim,
            decomp_hidden       = decomp_hidden,
            use_decomp_heads    = True,
        )
        proc._phase53_key = key
        new_procs[key] = proc
        procs.append(proc)

    unet.set_attn_processor(new_procs)
    return procs, orig_procs


class MultiBlockSlotManagerP53(MultiBlockSlotManagerP46):
    """Phase53Processor manager."""
    def __init__(self, procs: List[Phase53Processor], keys: List[str], primary_idx: int = 1):
        super().__init__(procs, keys, primary_idx)

    def decomp_head_params(self) -> List[nn.Parameter]:
        params = []
        for proc in self.procs:
            if hasattr(proc, 'decomp_head_params'):
                params += proc.decomp_head_params()
        return params

    def entity_lora_kv_params(self) -> List[nn.Parameter]:
        params = []
        for proc in self.procs:
            if hasattr(proc, 'entity_lora_kv_params'):
                params += proc.entity_lora_kv_params()
        return params

    def entity_lora_out_params(self) -> List[nn.Parameter]:
        params = []
        for proc in self.procs:
            if hasattr(proc, 'entity_lora_out_params'):
                params += proc.entity_lora_out_params()
        return params


def restore_multiblock_state_p53(manager, ckpt: dict, device: str = "cpu"):
    """
    Load Phase52/46 checkpoint into Phase53 manager.

    Loads: slot_blend_raw, slot0/1_adapter, blend_head, lora_k/v/out,
           weight_head, ref_proj_e0/e1, overlap_blend_head, occ_head_e0/e1
    Skips: decomp_heads (freshly initialized — Phase53 new heads)
    Handles: missing keys gracefully (forward-compat)
    """
    procs_state = ckpt.get("procs_state", [])
    for i, proc in enumerate(manager.procs):
        if i >= len(procs_state):
            break
        ps = procs_state[i]

        # slot_blend_raw
        if "slot_blend_raw" in ps and ps["slot_blend_raw"] is not None:
            proc.slot_blend_raw.data.copy_(ps["slot_blend_raw"].to(device))

        # Adapters
        for attr in ("slot0_adapter", "slot1_adapter", "blend_head",
                     "lora_k", "lora_v", "lora_out",
                     "weight_head", "ref_proj_e0", "ref_proj_e1",
                     "overlap_blend_head"):
            if attr in ps and ps[attr] and hasattr(proc, attr):
                try:
                    getattr(proc, attr).load_state_dict(ps[attr], strict=False)
                except Exception as e:
                    print(f"  [restore p53] {attr} load warn: {e}", flush=True)

        # OccupancyHead (old — kept for fallback, not primary in Phase53)
        for occ_name in ("occ_head_e0", "occ_head_e1"):
            if occ_name in ps and ps[occ_name] and hasattr(proc, occ_name):
                head = getattr(proc, occ_name, None)
                if head is not None:
                    try:
                        head.load_state_dict(ps[occ_name], strict=False)
                    except Exception as e:
                        print(f"  [restore p53] {occ_name} load warn: {e}", flush=True)

        # decomp_heads: load when present, otherwise keep fresh zero-init.
        if "decomp_heads" in ps and ps["decomp_heads"] and hasattr(proc, "decomp_heads"):
            try:
                proc.decomp_heads.load_state_dict(ps["decomp_heads"], strict=False)
            except Exception as e:
                print(f"  [restore p53] decomp_heads load warn: {e}", flush=True)

        # Entity-specific K/V LoRA: load when present, otherwise init from shared.
        for elora_name, src_name in [
            ("lora_k_e0", "lora_k"), ("lora_k_e1", "lora_k"),
            ("lora_v_e0", "lora_v"), ("lora_v_e1", "lora_v"),
            ("lora_out_e0", "lora_out"), ("lora_out_e1", "lora_out"),
        ]:
            if elora_name in ps and ps[elora_name] and hasattr(proc, elora_name):
                try:
                    getattr(proc, elora_name).load_state_dict(ps[elora_name], strict=False)
                except Exception as e:
                    print(f"  [restore p53] {elora_name} load warn: {e}", flush=True)
            elif hasattr(proc, elora_name) and hasattr(proc, src_name):
                src = getattr(proc, src_name)
                dst = getattr(proc, elora_name)
                try:
                    dst.lora_A.weight.data.copy_(src.lora_A.weight.data)
                    dst.lora_B.weight.data.copy_(src.lora_B.weight.data)
                except Exception as e:
                    print(f"  [restore p53] {elora_name} init warn: {e}", flush=True)

    # VCA
    if "vca_state_dict" in ckpt:
        vca = getattr(manager, 'vca_layer', None) or getattr(manager.primary, 'vca', None)
        if vca is not None:
            try:
                vca.load_state_dict(ckpt["vca_state_dict"], strict=False)
            except Exception as e:
                print(f"  [restore p53] VCA load warn: {e}", flush=True)
    print("[Phase 53] checkpoint 복원 완료 (decomp_heads: fresh init)", flush=True)
