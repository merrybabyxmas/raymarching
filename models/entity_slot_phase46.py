"""
Phase 46 — Contrastive Occupancy + Direct Blend Ordering
=========================================================

Phase 45 failure root causes (20 epochs, all runs):
  1. occ_prod_exclusive_mean ≈ 0.638 > occ_prod_overlap_mean ≈ 0.587 (INVERTED)
     → both o0/o1 stay high in exclusive regions
     → l_occupancy_structure's l_ex_gap = (1-|o0-o1|)·exclusive penalizes
       similarity but never enforces WHICH entity should be zero
  2. blend_sep < 0 always (exclusive blend > overlap blend)
     → base_blend_v2 gets inflated in exclusive because o0*o1 high
     → OBH delta can't overcome logit(0.67)≈0.69 offset to reach target 0.35
  3. l_blend_target_balanced provides region-normalized MSE signal but no
     ORDERING constraint → model finds escape by raising both regions equally

Phase 46 fixes:
──────────────────────────────────────────────────
A. l_occ_contrastive(o0, o1, masks, margin=0.50)
   In e0-exclusive: enforce o0 > o1 + margin (relu hinge)
   In e1-exclusive: enforce o1 > o0 + margin
   → Direct gradient signal pushing the WRONG entity to zero

B. l_blend_ordering(blend_map, masks, margin=0.10)
   Enforce mean_blend_in_overlap > mean_blend_in_exclusive + margin
   Enforce mean_blend_in_exclusive > mean_blend_in_bg + margin
   → Constraint acts on blend output directly, independent of occupancy accuracy

C. compute_base_blend_v3(o0, o1)
   soft_o = (2*(o-0.5)).clamp(0,1)  → 0 below 0.5, linear above
   overlap_proxy = soft_o0 * soft_o1  → near-zero unless BOTH > 0.5
   At collapse (o0=0.92, o1=0.69): v2 gives 0.66, v3 gives 0.49  (less inflation)
   At correct (o0=0.92, o1=0.05): v2 gives 0.28, v3 gives 0.28  (unchanged)
   At overlap (o0=0.9, o1=0.9):   v2 gives 0.89, v3 gives 0.69  (OBH delta covers gap)

D. val_score_phase46 — optional rollout (fair scoring when not computed every eval)

E. Stage B: occ_head + OBH at lr/5 (stop Stage B from destroying Stage A progress)
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.entity_slot_phase45 import (
    # Architecture
    OccupancyHead,
    Phase45Processor,
    MultiBlockSlotManagerP45,
    inject_multi_block_entity_slot_p45,
    restore_multiblock_state_p45,
    reroute_entity_weights_with_occupancy,
    # Losses (reuse)
    l_occupancy,
    l_occupancy_structure,
    l_blend_target_balanced,
    l_blend_rank,
    l_visible_weights_region_balanced,
    l_visible_iou_soft,
    # Stats / utilities
    collect_occupancy_stats,
    collect_blend_stats_detailed,
    # Constants
    BLOCK_INNER_DIMS,
    DEFAULT_INJECT_KEYS,
    CROSS_ATTN_DIM,
    PRIMARY_DIM,
)
from models.entity_slot_phase44 import (
    OverlapBlendHead,
    build_blend_targets,
    BLOCK_INNER_DIMS,
    DEFAULT_INJECT_KEYS,
    CROSS_ATTN_DIM,
    PRIMARY_DIM,
)
from models.entity_slot_phase42 import l_w_residual


# =============================================================================
# New loss: l_occ_contrastive
# =============================================================================

def l_occ_contrastive(
    o0_for_loss: torch.Tensor,   # (B, S) — no detach
    o1_for_loss: torch.Tensor,   # (B, S)
    masks_BNS:   torch.Tensor,   # (B, 2, S) GT mask
    margin:      float = 0.50,
    eps:         float = 1e-6,
) -> torch.Tensor:
    """
    Margin-based exclusive entity suppression.

    핵심 수정: l_occupancy_structure의 l_ex_gap는 "비슷하면 페널티"이지만
    "어느 쪽을 0으로 내려야 하는가"를 알려주지 않는다.
    l_occ_contrastive는 명확한 방향성을 강제한다:

      e0-exclusive에서: relu(o1 - o0 + margin) → o0가 o1보다 margin 이상 커야 함
      e1-exclusive에서: relu(o0 - o1 + margin) → o1가 o0보다 margin 이상 커야 함

    margin=0.5이면:
      o0=0.92, o1=0.69 (현재 collapse) → penalty = relu(0.69-0.92+0.5)=0.27 (active!)
      o0=0.92, o1=0.40 (near-fix)     → penalty = relu(0.40-0.92+0.5)=0.0  (satisfied)
      Goal: push o1 < o0 - 0.5 in e0-exclusive → o1 < ~0.42 when o0=0.92
    """
    m0 = masks_BNS[:, 0, :].float().to(o0_for_loss.device)
    m1 = masks_BNS[:, 1, :].float().to(o0_for_loss.device)

    # Per-entity exclusive regions
    e0_excl = m0 * (1.0 - m1)   # only entity 0 present
    e1_excl = m1 * (1.0 - m0)   # only entity 1 present

    n_e0 = e0_excl.sum() + eps
    n_e1 = e1_excl.sum() + eps

    o0 = o0_for_loss.float()
    o1 = o1_for_loss.float()

    # In e0-exclusive: o0 should dominate o1 by margin
    l_e0 = (F.relu(o1 - o0 + margin) * e0_excl).sum() / n_e0
    # In e1-exclusive: o1 should dominate o0 by margin
    l_e1 = (F.relu(o0 - o1 + margin) * e1_excl).sum() / n_e1

    return l_e0 + l_e1


# =============================================================================
# New loss: l_blend_ordering
# =============================================================================

def l_blend_ordering(
    blend_map: torch.Tensor,   # (B, S)
    masks_BNS: torch.Tensor,   # (B, 2, S)
    margin:    float = 0.10,
    eps:       float = 1e-6,
) -> torch.Tensor:
    """
    Direct ordering constraint on region-mean blend values.

    blend_sep = blend_overlap_mean - blend_exclusive_mean は常에 negative。
    l_blend_target_balanced (MSE to fixed targets)은 ordering을 보장하지 않는다:
    모델이 모든 region blend를 같이 올리면서 loss를 최소화할 수 있다.

    l_blend_ordering은 순서를 직접 강제한다:
      mean(blend_in_overlap) > mean(blend_in_exclusive) + margin
      mean(blend_in_exclusive) > mean(blend_in_bg) + margin

    gradient:
      d/d(blend)[overlap] ∝ +1/n_ov  (push overlap blend UP)
      d/d(blend)[exclusive] ∝ -1/n_ex (push exclusive blend DOWN)
    이것이 정확히 blend_sep > 0를 만드는 방향이다.
    """
    m0 = masks_BNS[:, 0, :].float().to(blend_map.device)
    m1 = masks_BNS[:, 1, :].float().to(blend_map.device)

    overlap   = m0 * m1
    exclusive = (m0 + m1).clamp(0.0, 1.0) - overlap
    bg        = 1.0 - (m0 + m1).clamp(0.0, 1.0)

    bm = blend_map.float()
    n_ov = overlap.sum() + eps
    n_ex = exclusive.sum() + eps
    n_bg = bg.sum() + eps

    mean_ov = (bm * overlap).sum()   / n_ov
    mean_ex = (bm * exclusive).sum() / n_ex
    mean_bg = (bm * bg).sum()        / n_bg

    # Hinge losses — zero when ordering satisfied, positive when violated
    l_ov_ex = F.relu(mean_ex - mean_ov + margin)    # penalize if ex ≥ ov - margin
    l_ex_bg = F.relu(mean_bg - mean_ex + margin)    # penalize if bg ≥ ex - margin

    return l_ov_ex + l_ex_bg


# =============================================================================
# compute_base_blend_v3  —  soft-thresholded product
# =============================================================================

def compute_base_blend_v3(
    o0: torch.Tensor,
    o1: torch.Tensor,
) -> torch.Tensor:
    """
    Phase46 base blend prior — soft-thresholded product.

    Phase45 실패 패턴:
      exclusive에서 o0≈0.92, o1≈0.69 → product=0.635 → base_blend≈0.66 (too high)
      OBH가 logit(0.66)≈0.66에서 logit(0.35)≈-0.62까지 내려야 → Δ=-1.28 (어려움)

    v3 수정:
      soft_o = (2*(o - 0.5)).clamp(0, 1)
        → below 0.5: soft_o = 0 (confident absence)
        → above 0.5: linear 0→1 (confident presence)
      overlap_proxy = soft_o0 * soft_o1
        → 0.5 미만이면 즉시 0 → false positive overlap 억제

    At exclusive collapse (o0=0.92, o1=0.69):
      soft_o0=0.84, soft_o1=0.38 → product=0.32
      base_blend = 0.05 + 0.25*0.92 + 0.65*0.32 = 0.488  (vs v2: 0.661 — 26% reduction!)

    At correct exclusive (o0=0.92, o1=0.08):
      soft_o0=0.84, soft_o1=0.0 → product=0
      base_blend = 0.05 + 0.25*0.92 = 0.28  (correct exclusive prior)

    At overlap (o0=0.9, o1=0.9):
      soft_o0=0.80, soft_o1=0.80 → product=0.64
      base_blend = 0.05 + 0.25*0.90 + 0.65*0.64 = 0.691
      (OBH delta of +0.8 logit units gets to 0.87 — achievable with hidden=32)
    """
    soft_o0 = (2.0 * (o0 - 0.5)).clamp(0.0, 1.0)
    soft_o1 = (2.0 * (o1 - 0.5)).clamp(0.0, 1.0)

    overlap_proxy = soft_o0 * soft_o1
    entity_proxy  = torch.maximum(o0, o1)

    return (0.05 + 0.25 * entity_proxy + 0.65 * overlap_proxy).clamp(0.05, 0.95)


# =============================================================================
# val_score_phase46 — optional rollout (fair scoring)
# =============================================================================

def val_score_phase46(
    tf_iou_e0:      float,
    tf_iou_e1:      float,
    tf_ord:         float,
    tf_wrong:       float,
    rollout_iou_e0: float = 0.0,
    rollout_iou_e1: float = 0.0,
    blend_sep:      float = 0.0,
    has_rollout:    bool  = True,
) -> float:
    """
    Phase46 val score.

    has_rollout=False: rollout 미계산 시 해당 항을 제외하고 나머지로 renormalize.
    Phase45 문제: rollout=0을 그대로 넣으면 non-rollout epoch의 score가 아무 이유 없이 낮음.

    has_rollout=True  (weights sum 1.0):
      0.15·iou_e0 + 0.15·iou_e1 + 0.10·ord + 0.10·(1-wrong)
      + 0.20·rollout_e0 + 0.20·rollout_e1 + 0.10·blend_score

    has_rollout=False (weights sum 1.0, rollout excluded):
      0.25·iou_e0 + 0.25·iou_e1 + 0.15·ord + 0.15·(1-wrong) + 0.20·blend_score
    """
    blend_score = max(0.0, min(1.0, (blend_sep + 0.15) / 0.30))

    if has_rollout:
        return (0.15 * tf_iou_e0
              + 0.15 * tf_iou_e1
              + 0.10 * tf_ord
              + 0.10 * (1.0 - tf_wrong)
              + 0.20 * rollout_iou_e0
              + 0.20 * rollout_iou_e1
              + 0.10 * blend_score)
    else:
        return (0.25 * tf_iou_e0
              + 0.25 * tf_iou_e1
              + 0.15 * tf_ord
              + 0.15 * (1.0 - tf_wrong)
              + 0.20 * blend_score)


# =============================================================================
# Phase46Processor — uses compute_base_blend_v3
# =============================================================================

class Phase46Processor(Phase45Processor):
    """
    Phase45Processor과 동일하되 compute_base_blend_v2 → v3.

    이것만으로도 Stage A 이후 exclusive base_blend가 낮아져
    OBH delta burden이 줄어든다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask        = None,
        temb                  = None,
        **kwargs,
    ):
        """Phase45.__call__ 기반, base_blend 계산만 v3으로 교체."""
        B, S, D = hidden_states.shape
        dtype   = hidden_states.dtype
        enc_hs  = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        T_seq   = enc_hs.shape[1]

        inner_dim = attn.to_q.weight.shape[0]
        n_heads   = attn.heads
        head_dim  = inner_dim // n_heads
        scale     = head_dim ** -0.5

        def _mh(x, seq_len):
            return x.reshape(B, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)

        enc_hs_f = enc_hs.float()
        k = (attn.to_k(enc_hs)
             + self.lora_k(enc_hs_f).to(dtype=enc_hs.dtype))
        v = (attn.to_v(enc_hs)
             + self.lora_v(enc_hs_f).to(dtype=enc_hs.dtype))
        k_mh = _mh(k, T_seq)
        v_mh = _mh(v, T_seq)

        q    = attn.to_q(hidden_states)
        q_mh = _mh(q, S)

        scores_g = torch.matmul(q_mh, k_mh.transpose(-1, -2)) * scale
        w_g      = scores_g.softmax(dim=-1)
        F_g = (torch.matmul(w_g, v_mh)
               .permute(0, 2, 1, 3)
               .reshape(B, S, inner_dim))

        F_0_raw = self._masked_attn(
            q_mh, k_mh, v_mh, self.toks_e0, T_seq,
            scale, B, S, n_heads, head_dim, inner_dim, fallback=F_g)
        F_1_raw = self._masked_attn(
            q_mh, k_mh, v_mh, self.toks_e1, T_seq,
            scale, B, S, n_heads, head_dim, inner_dim, fallback=F_g)

        F_0 = self.slot0_adapter(F_0_raw.float()).to(dtype)
        F_1 = self.slot1_adapter(F_1_raw.float()).to(dtype)

        # Occupancy heads (same as Phase45)
        if self.use_occ_head and self.occ_head_e0 is not None:
            o0_for_loss = self.occ_head_e0(F_0.float())
            o1_for_loss = self.occ_head_e1(F_1.float())
            self.last_o0_for_loss = o0_for_loss
            self.last_o1_for_loss = o1_for_loss
            self.last_o0          = o0_for_loss.detach()
            self.last_o1          = o1_for_loss.detach()
        else:
            o0_for_loss = torch.full((B, S), 0.5, device=F_0.device)
            o1_for_loss = torch.full((B, S), 0.5, device=F_1.device)
            self.last_o0_for_loss = None
            self.last_o1_for_loss = None
            self.last_o0 = None
            self.last_o1 = None

        # VCA sigma
        sigma = None
        if self.vca is not None and self.entity_ctx is not None:
            ctx   = self.entity_ctx.expand(B, -1, -1).to(dtype)
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

        if sigma is not None and sigma.shape[:2] == (B, S):
            sig = sigma.to(device=F_g.device, dtype=torch.float32)

            alpha_0  = sig[:, :, 0, :].max(dim=-1).values
            alpha_1  = sig[:, :, 1, :].max(dim=-1).values
            e0_front = torch.sigmoid(5.0 * (sig[:, :, 0, 0] - sig[:, :, 1, 0]))

            base_w0  = e0_front * alpha_0 + (1.0 - e0_front) * alpha_0 * (1.0 - alpha_1)
            base_w1  = (1.0 - e0_front) * alpha_1 + e0_front * alpha_1 * (1.0 - alpha_0)
            base_wbg = (1.0 - base_w0 - base_w1).clamp(min=0.0)

            feat = torch.stack([
                alpha_0, alpha_1,
                sig[:, :, 0, 0],
                sig[:, :, 0, min(1, sig.shape[-1]-1)],
                sig[:, :, 1, 0],
                sig[:, :, 1, min(1, sig.shape[-1]-1)],
                alpha_0 * alpha_1, e0_front,
            ], dim=-1).float()

            delta      = self.weight_head(feat)
            base_logits = torch.log(
                torch.stack([base_wbg, base_w0, base_w1], dim=-1).clamp(min=1e-6))
            probs       = (base_logits + delta).softmax(dim=-1)
            w_bg, w0, w1 = probs.unbind(dim=-1)
            self.last_w_delta = delta

            o0 = o0_for_loss
            o1 = o1_for_loss

            # Occupancy routing (same as Phase45)
            w0, w1, route_mix = reroute_entity_weights_with_occupancy(
                w0, w1, o0, o1,
                e0_front=e0_front,
                strength=self.occ_route_strength,
                overlap_strength=self.occ_overlap_route,
            )
            w_bg = (1.0 - w0 - w1).clamp(min=0.0)
            norm = (w0 + w1 + w_bg).clamp(min=1e-6)
            w0, w1, w_bg = w0 / norm, w1 / norm, w_bg / norm
            self.last_route_mix = route_mix.detach()

            w0_f  = w0.unsqueeze(-1).to(dtype=F_g.dtype)
            w1_f  = w1.unsqueeze(-1).to(dtype=F_g.dtype)
            wbg_f = w_bg.unsqueeze(-1).to(dtype=F_g.dtype)
            composed = w0_f * F_0 + w1_f * F_1 + wbg_f * F_g

            # ── PHASE 46 KEY CHANGE: use compute_base_blend_v3 ──────────
            overlap_proxy_occ = o0 * o1
            base_blend = compute_base_blend_v3(o0, o1)           # ← v3 here

            feat_blend = torch.stack([
                o0, o1,
                overlap_proxy_occ,
                e0_front,
                w0, w1, w_bg,
                (o0 - o1).abs(),
            ], dim=-1).float()                                    # (B, S, 8)

            delta_b    = self.overlap_blend_head(feat_blend)
            blend_logit = torch.logit(base_blend, eps=1e-6).unsqueeze(-1)
            blend_map   = torch.sigmoid(blend_logit + delta_b)   # (B, S, 1)

            self.last_blend_map_for_loss = blend_map.squeeze(-1)
            self.last_blend_map          = blend_map.squeeze(-1).detach()
            self.last_base_blend         = base_blend.detach()
            self.last_overlap_proxy      = overlap_proxy_occ.detach()
            self.last_blend              = self.last_blend_map

            blend_map_f = blend_map.to(dtype=F_g.dtype)

        else:
            composed  = (F_0 + F_1 + F_g) / 3.0
            w0 = torch.ones(B, S, device=F_g.device) / 3
            w1 = torch.ones(B, S, device=F_g.device) / 3
            w_bg = torch.ones(B, S, device=F_g.device) / 3
            blend_map_f = self.slot_blend.to(dtype=F_g.dtype)
            self.last_w_delta            = None
            self.last_blend_map_for_loss = None
            self.last_blend_map          = None
            self.last_blend              = None
            self.last_route_mix          = None

        blended = blend_map_f * composed + (1.0 - blend_map_f) * F_g

        self.last_w0      = w0
        self.last_w1      = w1
        if sigma is not None and sigma.shape[:2] == (B, S):
            self.last_alpha0 = alpha_0
            self.last_alpha1 = alpha_1
        self.last_F0      = F_0
        self.last_F1      = F_1
        self.last_Fg      = F_g
        self.last_blended = blended

        out = (attn.to_out[0](blended)
               + self.lora_out(blended.float()).to(dtype=blended.dtype))
        out = attn.to_out[1](out)
        return out.to(dtype)


# =============================================================================
# Multi-block injection for Phase 46
# =============================================================================

def inject_multi_block_entity_slot_p46(
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
):
    """Injects Phase46Processor into UNet (same signature as p45 variant)."""
    import copy as _copy
    if inject_keys is None:
        inject_keys = DEFAULT_INJECT_KEYS

    unet       = pipe.unet
    orig_procs = _copy.copy(dict(unet.attn_processors))
    new_procs  = dict(unet.attn_processors)
    procs: List[Phase46Processor] = []

    for i, key in enumerate(inject_keys):
        inner_dim  = BLOCK_INNER_DIMS.get(key, 640)
        is_primary = (i == primary_idx)
        proc = Phase46Processor(
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
            use_occ_head        = True,
        )
        new_procs[key] = proc
        procs.append(proc)

    unet.set_attn_processor(new_procs)
    return procs, orig_procs


class MultiBlockSlotManagerP46(MultiBlockSlotManagerP45):
    """Phase46Processor용 manager (same interface as P45)."""
    def __init__(self, procs: List[Phase46Processor], keys: List[str], primary_idx: int = 1):
        super().__init__(procs, keys, primary_idx)


# restore_multiblock_state_p46 == restore_multiblock_state_p45 (same checkpoint format)
restore_multiblock_state_p46 = restore_multiblock_state_p45
