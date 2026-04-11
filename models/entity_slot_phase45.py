"""
Phase 45 — Occupancy-Guided Blend Prior
========================================

Phase 44 실패 원인 분석:
  - VCA sigma는 spatial occupancy가 아니라 shared-attention 부산물 →
    exclusive region에서 alpha_0≈alpha_1≈0.7 (구분 없음) → base_blend=0.63
    overlap region에서 alpha_0≈alpha_1≈0.5 (공유 감소) → base_blend=0.33
    즉 base_blend(exclusive) > base_blend(overlap) — 부호 반전
  - blend_rank가 0.6대에 고착: 전체 blend를 낮춰 MSE를 줄이는 방식으로
    loss를 속임 (region-balanced loss로 해결)
  - w0*w1는 visibility weight product → occlusion overlap에서 front=0.85,
    back=0.05 → w0*w1≈0.04 (낮음). ambiguity proxy이지 overlap proxy가 아님.

Phase 45 핵심 변경:
──────────────────────────────────────────────────
A. OccupancyHead — F_0/F_1 (slot features)에서 직접 occupancy 예측
   o0, o1 = sigmoid(MLP(F_0)), sigmoid(MLP(F_1))
   GT mask로 BCE 감독 → l_occupancy

B. compute_base_blend_v2(o0, o1) — occupancy 기반 prior
   overlap_proxy = o0 * o1      ← 두 entity 모두 있을 때 ↑
   entity_proxy  = max(o0, o1)
   base_blend = 0.05 + 0.25*entity_proxy + 0.60*overlap_proxy

   학습 후 올바른 순서:
     overlap (o0≈1, o1≈1): ~0.90
     exclusive (o0≈1, o1≈0): ~0.30
     bg (o0≈0, o1≈0): ~0.05

C. OBH feat_blend 교체
   [o0, o1, o0*o1, e0_front, w0, w1, w_bg, |o0-o1|]
   (alpha/sigma → o0/o1 + visibility context)

D. l_blend_target_balanced — region-normalized MSE
   각 영역을 자기 픽셀 수로 normalize → overlap 희소성 보정
   L = w_ov*(MSE_ov/n_ov) + w_ex*(MSE_ex/n_ex) + w_bg*(MSE_bg/n_bg)

E. w0/w1는 Porter-Duff visibility compositing에만 사용 (변경 없음)
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.entity_slot_phase44 import (
    Phase44Processor,
    MultiBlockSlotManagerP44,
    OverlapBlendHead,
    build_blend_targets,
    l_blend_rank,
    collect_blend_stats_detailed,
    val_score_phase44 as val_score_phase45,
    BLOCK_INNER_DIMS,
    DEFAULT_INJECT_KEYS,
    CROSS_ATTN_DIM,
    PRIMARY_DIM,
    inject_multi_block_entity_slot_p44,
)
from models.entity_slot_phase43 import (
    MultiBlockSlotManagerP43,
    FeatureProjector,
    restore_multiblock_state_p43,
)
from models.entity_slot_phase42 import (
    l_w_residual,
    WeightHead,
)


# =============================================================================
# OccupancyHead
# =============================================================================

class OccupancyHead(nn.Module):
    """
    Slot features → per-pixel occupancy probability.

    입력: (B, S, in_dim)  — slot adapter 출력 F_0 or F_1
    출력: (B, S)           — sigmoid occupancy ∈ [0, 1]

    zero-init 보장:
      초기 delta=0 → sigmoid(0) = 0.5 everywhere
    GT mask로 BCE 감독 → 점차 exclusive/overlap/bg 구분
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """(B, S, D) → sigmoid → (B, S)"""
        return torch.sigmoid(self.net(feat).squeeze(-1))


def compute_base_blend_v2(
    o0: torch.Tensor,
    o1: torch.Tensor,
) -> torch.Tensor:
    """
    Occupancy-based blend prior (replaces Phase44's alpha-based version).

    o0, o1 = GT-supervised occupancy (not VCA sigma)

    base_blend = 0.05 + 0.25*max(o0,o1) + 0.60*(o0*o1)

    올바른 prior (학습 후):
      overlap (o0≈1, o1≈1): ~0.90
      exclusive (o0≈1, o1≈0): ~0.30
      background (o0≈0, o1≈0): ~0.05
    """
    overlap_proxy = o0 * o1
    entity_proxy  = torch.maximum(o0, o1)
    return (0.05 + 0.25 * entity_proxy + 0.60 * overlap_proxy).clamp(0.05, 0.95)


# =============================================================================
# Loss functions
# =============================================================================

def l_occupancy(
    o0_for_loss: torch.Tensor,   # (B, S) — no detach
    o1_for_loss: torch.Tensor,   # (B, S)
    masks_BNS:   torch.Tensor,   # (B, 2, S) GT mask
    eps:         float = 1e-6,
) -> torch.Tensor:
    """
    BCE loss: occupancy vs GT mask.

    l_occ = BCE(o0, mask0) + BCE(o1, mask1)
    """
    m0 = masks_BNS[:, 0, :].float().to(o0_for_loss.device)
    m1 = masks_BNS[:, 1, :].float().to(o0_for_loss.device)
    l0 = F.binary_cross_entropy(
        o0_for_loss.float().clamp(eps, 1 - eps), m0, reduction='mean')
    l1 = F.binary_cross_entropy(
        o1_for_loss.float().clamp(eps, 1 - eps), m1, reduction='mean')
    return l0 + l1


def l_blend_target_balanced(
    blend_map:         torch.Tensor,   # (B, S)
    entity_masks_BNS:  torch.Tensor,   # (B, 2, S)
    overlap_val:       float = 0.90,
    exclusive_val:     float = 0.35,
    bg_val:            float = 0.05,
    w_ov:              float = 1.0,    # overlap region weight
    w_ex:              float = 0.5,    # exclusive region weight
    w_bg:              float = 0.2,    # bg region weight
    eps:               float = 1e-6,
) -> torch.Tensor:
    """
    Region-balanced MSE (Phase44 l_blend_target 대체).

    각 영역을 자기 픽셀 수로 normalize → overlap 희소성 보정.
    L = w_ov * (MSE_ov/n_ov) + w_ex * (MSE_ex/n_ex) + w_bg * (MSE_bg/n_bg)

    Phase44 실패 원인 중 하나:
      global mean으로 나누면 overlap(5%) << bg(60%) → overlap gradient 희석
      region-normalize로 각 영역 equally important.
    """
    m0 = entity_masks_BNS[:, 0, :].float().to(blend_map.device)
    m1 = entity_masks_BNS[:, 1, :].float().to(blend_map.device)

    overlap   = m0 * m1
    exclusive = (m0 + m1).clamp(0.0, 1.0) - overlap
    bg        = 1.0 - (m0 + m1).clamp(0.0, 1.0)

    bm = blend_map.float()

    def _region_mse(mask: torch.Tensor, target_val: float) -> torch.Tensor:
        n   = mask.sum() + eps
        return ((bm - target_val).pow(2) * mask).sum() / n

    l_ov  = _region_mse(overlap,   overlap_val)
    l_ex  = _region_mse(exclusive, exclusive_val)
    l_bg  = _region_mse(bg,        bg_val)

    return w_ov * l_ov + w_ex * l_ex + w_bg * l_bg


# =============================================================================
# Phase45Processor
# =============================================================================

class Phase45Processor(Phase44Processor):
    """
    Phase44Processor + OccupancyHead pair (e0, e1).

    변경 사항:
      - occ_head_e0, occ_head_e1: F_0/F_1 → occupancy (GT mask 감독)
      - base_blend: compute_base_blend_v2(o0, o1) (alpha-based 제거)
      - feat_blend: [o0, o1, o0*o1, e0_front, w0, w1, w_bg, |o0-o1|]
      - last_o0_for_loss, last_o1_for_loss: grad path (l_occupancy용)
      - last_o0, last_o1: detached (diagnostics)

    w0/w1는 Porter-Duff compositing용으로 유지 (visibility ≠ occupancy).
    """

    def __init__(
        self,
        query_dim:           int,
        vca_layer            = None,
        entity_ctx:          Optional[torch.Tensor] = None,
        slot_blend_init:     float = 0.3,
        inner_dim:           Optional[int] = None,
        adapter_rank:        int   = 64,
        use_blend_head:      bool  = True,
        lora_rank:           int   = 4,
        cross_attention_dim: int   = CROSS_ATTN_DIM,
        weight_head_hidden:  int   = 32,
        primary_dim:         int   = PRIMARY_DIM,
        proj_hidden:         int   = 256,
        obh_hidden:          int   = 32,
        occ_hidden:          int   = 64,
        use_occ_head:        bool  = True,
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
        )
        # OccupancyHead pair — F_0/F_1 → o0/o1
        self.use_occ_head = use_occ_head
        eff_dim = inner_dim if inner_dim is not None else query_dim
        if use_occ_head:
            self.occ_head_e0 = OccupancyHead(in_dim=eff_dim, hidden=occ_hidden)
            self.occ_head_e1 = OccupancyHead(in_dim=eff_dim, hidden=occ_hidden)
        else:
            self.occ_head_e0 = None
            self.occ_head_e1 = None

        # Phase 45 추가 저장
        self.last_o0_for_loss: Optional[torch.Tensor] = None
        self.last_o1_for_loss: Optional[torch.Tensor] = None
        self.last_o0:          Optional[torch.Tensor] = None
        self.last_o1:          Optional[torch.Tensor] = None

    def reset_slot_store(self):
        super().reset_slot_store()
        self.last_o0_for_loss = None
        self.last_o1_for_loss = None
        self.last_o0          = None
        self.last_o1          = None

    def occupancy_head_params(self) -> List[torch.nn.Parameter]:
        params = []
        for head in (self.occ_head_e0, self.occ_head_e1):
            if head is not None:
                params += list(head.parameters())
        return params

    # ------------------------------------------------------------------
    def __call__(
        self,
        attn,
        hidden_states:          torch.Tensor,
        encoder_hidden_states:  Optional[torch.Tensor] = None,
        attention_mask          = None,
        temb                    = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Phase44 __call__ 기반, blend 계산을 OccupancyHead 기반으로 교체.
        """
        B, S, D = hidden_states.shape
        dtype   = hidden_states.dtype
        enc_hs  = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        T_seq   = enc_hs.shape[1]

        inner_dim = attn.to_q.weight.shape[0]
        n_heads   = attn.heads
        head_dim  = inner_dim // n_heads
        scale     = head_dim ** -0.5

        def _mh(x: torch.Tensor, seq_len: int) -> torch.Tensor:
            return x.reshape(B, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)

        # ── K, V with LoRA ──────────────────────────────────────────────
        enc_hs_f = enc_hs.float()
        k = (attn.to_k(enc_hs)
             + self.lora_k(enc_hs_f).to(dtype=enc_hs.dtype))
        v = (attn.to_v(enc_hs)
             + self.lora_v(enc_hs_f).to(dtype=enc_hs.dtype))
        k_mh = _mh(k, T_seq)
        v_mh = _mh(v, T_seq)

        # ── Q ───────────────────────────────────────────────────────────
        q    = attn.to_q(hidden_states)
        q_mh = _mh(q, S)

        # ── Global attention ─────────────────────────────────────────────
        scores_g = torch.matmul(q_mh, k_mh.transpose(-1, -2)) * scale
        w_g      = scores_g.softmax(dim=-1)
        F_g = (torch.matmul(w_g, v_mh)
               .permute(0, 2, 1, 3)
               .reshape(B, S, inner_dim))

        # ── Entity slot attention ────────────────────────────────────────
        F_0_raw = self._masked_attn(
            q_mh, k_mh, v_mh, self.toks_e0, T_seq,
            scale, B, S, n_heads, head_dim, inner_dim, fallback=F_g)
        F_1_raw = self._masked_attn(
            q_mh, k_mh, v_mh, self.toks_e1, T_seq,
            scale, B, S, n_heads, head_dim, inner_dim, fallback=F_g)

        # ── Slot adapters ────────────────────────────────────────────────
        F_0 = self.slot0_adapter(F_0_raw.float()).to(dtype)
        F_1 = self.slot1_adapter(F_1_raw.float()).to(dtype)

        # ── Phase 45: Occupancy from slot features (always, not from sigma) ──
        if self.use_occ_head and self.occ_head_e0 is not None:
            o0_for_loss = self.occ_head_e0(F_0.float())   # (B, S) — grad path
            o1_for_loss = self.occ_head_e1(F_1.float())   # (B, S)
            self.last_o0_for_loss = o0_for_loss
            self.last_o1_for_loss = o1_for_loss
            self.last_o0          = o0_for_loss.detach()
            self.last_o1          = o1_for_loss.detach()
        else:
            o0_for_loss = torch.full((B, S), 0.5, device=F_0.device)
            o1_for_loss = torch.full((B, S), 0.5, device=F_1.device)
            self.last_o0_for_loss = None
            self.last_o1_for_loss = None
            self.last_o0          = None
            self.last_o1          = None

        # ── VCA sigma ────────────────────────────────────────────────────
        sigma: Optional[torch.Tensor] = None
        if self.vca is not None and self.entity_ctx is not None:
            ctx   = self.entity_ctx.expand(B, -1, -1).to(dtype)
            vca_h = hidden_states.float()
            vca_c = ctx.float()
            if self.training:
                _ = self.vca(vca_h, vca_c)
            else:
                with torch.no_grad():
                    _ = self.vca(vca_h, vca_c)
            sigma = getattr(self.vca, 'last_sigma', None)
            if sigma is not None:
                self.last_sigma = sigma.detach().float()
                sigma_raw = (getattr(self.vca, 'last_sigma_raw', None)
                             if self.training else sigma)
                if self.training and sigma_raw is not None:
                    self.sigma_acc.append(sigma_raw.float())
                    sigma = sigma_raw

        # ── Porter-Duff base weights + WeightHead ────────────────────────
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
            probs      = (base_logits + delta).softmax(dim=-1)
            w_bg, w0, w1 = probs.unbind(dim=-1)

            self.last_w_delta = delta

            w0_f  = w0.unsqueeze(-1).to(dtype=F_g.dtype)
            w1_f  = w1.unsqueeze(-1).to(dtype=F_g.dtype)
            wbg_f = w_bg.unsqueeze(-1).to(dtype=F_g.dtype)
            composed = w0_f * F_0 + w1_f * F_1 + wbg_f * F_g

            # ── Phase 45: OccupancyHead 기반 base_blend ──────────────────
            o0 = o0_for_loss   # grad path intact
            o1 = o1_for_loss
            overlap_proxy_occ = o0 * o1
            base_blend = compute_base_blend_v2(o0, o1)  # occupancy 기반

            # Updated feat_blend: [o0, o1, o0*o1, e0_front, w0, w1, w_bg, |o0-o1|]
            feat_blend = torch.stack([
                o0, o1,
                overlap_proxy_occ,
                e0_front,
                w0, w1, w_bg,
                (o0 - o1).abs(),
            ], dim=-1).float()                                    # (B, S, 8)

            delta_b    = self.overlap_blend_head(feat_blend)      # (B, S, 1)
            blend_logit = torch.logit(base_blend, eps=1e-6).unsqueeze(-1)
            blend_map   = torch.sigmoid(blend_logit + delta_b)    # (B, S, 1)

            # Store for gradient-carrying loss (no detach)
            self.last_blend_map_for_loss = blend_map.squeeze(-1)
            # Store diagnostics (detached)
            self.last_blend_map     = blend_map.squeeze(-1).detach()
            self.last_base_blend    = base_blend.detach()
            self.last_overlap_proxy = overlap_proxy_occ.detach()
            self.last_blend         = self.last_blend_map

            blend_map_f = blend_map.to(dtype=F_g.dtype)

        else:
            composed  = (F_0 + F_1 + F_g) / 3.0
            w0        = torch.ones(B, S, device=F_g.device) / 3
            w1        = torch.ones(B, S, device=F_g.device) / 3
            w_bg      = torch.ones(B, S, device=F_g.device) / 3
            blend_map_f = self.slot_blend.to(dtype=F_g.dtype)
            self.last_w_delta            = None
            self.last_blend_map_for_loss = None
            self.last_blend_map          = None
            self.last_blend              = None

        # ── Blend ────────────────────────────────────────────────────────
        blended = blend_map_f * composed + (1.0 - blend_map_f) * F_g

        # ── Store ────────────────────────────────────────────────────────
        self.last_w0      = w0
        self.last_w1      = w1
        if sigma is not None and sigma.shape[:2] == (B, S):
            self.last_alpha0 = alpha_0
            self.last_alpha1 = alpha_1
        self.last_F0      = F_0
        self.last_F1      = F_1
        self.last_Fg      = F_g
        self.last_blended = blended

        # ── Output projection with LoRA ──────────────────────────────────
        out = (attn.to_out[0](blended)
               + self.lora_out(blended.float()).to(dtype=blended.dtype))
        out = attn.to_out[1](out)
        return out.to(dtype)


# =============================================================================
# Multi-block injection for Phase 45
# =============================================================================

def inject_multi_block_entity_slot_p45(
    pipe,
    vca_layer,
    entity_ctx:          torch.Tensor,
    inject_keys:         Optional[List[str]] = None,
    primary_idx:         int   = 1,
    slot_blend_init:     float = 0.3,
    adapter_rank:        int   = 64,
    lora_rank:           int   = 4,
    use_blend_head:      bool  = True,
    weight_head_hidden:  int   = 32,
    proj_hidden:         int   = 256,
    obh_hidden:          int   = 32,
    occ_hidden:          int   = 64,
) -> Tuple[List[Phase45Processor], Dict]:
    if inject_keys is None:
        inject_keys = DEFAULT_INJECT_KEYS

    unet       = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))
    new_procs  = dict(unet.attn_processors)
    procs: List[Phase45Processor] = []

    for i, key in enumerate(inject_keys):
        inner_dim  = BLOCK_INNER_DIMS.get(key, 640)
        block_vca  = vca_layer if i == primary_idx else None
        block_ctx  = entity_ctx if i == primary_idx else None
        is_primary = (i == primary_idx)
        proc = Phase45Processor(
            query_dim           = inner_dim,
            vca_layer           = block_vca,
            entity_ctx          = block_ctx,
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
            use_occ_head        = True,   # all blocks get occupancy head
        )
        new_procs[key] = proc
        procs.append(proc)

    unet.set_attn_processor(new_procs)
    return procs, orig_procs


# =============================================================================
# MultiBlockSlotManagerP45
# =============================================================================

class MultiBlockSlotManagerP45(MultiBlockSlotManagerP44):
    """Phase45Processor용 manager. occupancy_head params 접근 추가."""

    def __init__(self, procs: List[Phase45Processor], keys: List[str],
                 primary_idx: int = 1):
        super().__init__(procs, keys, primary_idx)

    def occupancy_head_params(self) -> List[torch.nn.Parameter]:
        params = []
        for p in self.procs:
            if hasattr(p, 'occupancy_head_params'):
                params += p.occupancy_head_params()
        return params


# =============================================================================
# Checkpoint restoration for Phase 45
# =============================================================================

def restore_multiblock_state_p45(
    manager,
    ckpt:   dict,
    device: str = "cpu",
) -> None:
    """
    Phase40/41/42/43/44 checkpoint에서 state 복원.
    occ_head_e0/e1는 phase45 신규 → ckpt에 없으면 zero-init 유지.
    """
    proc_states = ckpt.get("procs_state", None)
    if proc_states is None:
        raise RuntimeError("ckpt에 'procs_state' 없음. phase40+ checkpoint 필요.")

    if len(proc_states) != len(manager.procs):
        raise RuntimeError(
            f"procs_state 개수 불일치: ckpt={len(proc_states)}, "
            f"manager.procs={len(manager.procs)}")

    for i, (proc, state) in enumerate(zip(manager.procs, proc_states)):
        dev = proc.slot_blend_raw.device

        sbr = state["slot_blend_raw"]
        proc.slot_blend_raw.data.copy_(
            sbr.to(dev) if hasattr(sbr, 'to') else torch.tensor(sbr).to(dev))

        for mod_name in ("slot0_adapter", "slot1_adapter", "blend_head",
                         "lora_k", "lora_v", "lora_out"):
            if mod_name not in state:
                raise RuntimeError(
                    f"block[{i}] procs_state에 '{mod_name}' 없음.")
            sd = {k: v.to(dev) if hasattr(v, 'to') else v
                  for k, v in state[mod_name].items()}
            getattr(proc, mod_name).load_state_dict(sd, strict=False)

        # weight_head (Phase42+)
        if "weight_head" in state and hasattr(proc, 'weight_head'):
            sd = {k: v.to(dev) for k, v in state["weight_head"].items()}
            proc.weight_head.load_state_dict(sd, strict=False)
            print(f"  [restore] block[{i}] weight_head loaded", flush=True)

        # ref_proj (Phase43+)
        for proj_name in ("ref_proj_e0", "ref_proj_e1"):
            if proj_name in state and hasattr(proc, proj_name):
                sd = {k: v.to(dev) for k, v in state[proj_name].items()}
                getattr(proc, proj_name).load_state_dict(sd, strict=False)

        # overlap_blend_head (Phase44+)
        if "overlap_blend_head" in state and hasattr(proc, 'overlap_blend_head'):
            sd = {k: v.to(dev) for k, v in state["overlap_blend_head"].items()}
            proc.overlap_blend_head.load_state_dict(sd, strict=False)
            print(f"  [restore] block[{i}] overlap_blend_head loaded", flush=True)

        # occ_head (Phase45+) — zero-init if not in ckpt
        for occ_name in ("occ_head_e0", "occ_head_e1"):
            if occ_name in state and hasattr(proc, occ_name):
                mod = getattr(proc, occ_name)
                if mod is not None:
                    sd = {k: v.to(dev) for k, v in state[occ_name].items()}
                    mod.load_state_dict(sd, strict=False)
                    print(f"  [restore] block[{i}] {occ_name} loaded", flush=True)

        print(f"  [restore] block[{i}] OK  "
              f"(blend={float(proc.slot_blend.item()):.4f})", flush=True)
