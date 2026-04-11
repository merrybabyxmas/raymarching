"""
Phase 44 — Overlap-Aware Blend Prior + GT Blend Supervision
============================================================

Phase 43 진단 결과:
  - slot_ref 켜짐 (0.49~0.77) → IoU 0.088→0.096 미미한 상승
  - blend_sep 계속 -0.118 → compositing branch가 overlap에서 더 켜지는 압력 없음
  - slot_cont ≈ 0 → entity 간 구분 신호 약함

Phase 44 핵심 변경 (blend 직접 감독):
─────────────────────────────────────────
A. OverlapBlendHead: overlap prior + residual delta
   base_blend = 0.05 + 0.25*entity_proxy + 0.60*overlap_proxy (구조적 prior)
   delta = MLP(8 features) — zero-init (초기엔 base_blend 그대로)
   blend_map = sigmoid(logit(base_blend) + delta)

B. build_blend_targets: GT overlap/exclusive/bg로 직접 감독
   overlap=0.90, exclusive=0.35, bg=0.05

C. l_blend_target: weighted MSE (overlap region 가중치 ↑)
   l_blend_rank: overlap > exclusive > bg margin loss

D. last_blend_map 명시 저장 (last_blend_map: (B,S) 확정)

E. val_score_phase44: blend_score 포함
   0.15*iou_e0 + 0.15*iou_e1 + 0.10*ord + 0.10*(1-wrong)
   + 0.20*rollout_iou_e0 + 0.20*rollout_iou_e1 + 0.10*blend_score

학습 전략 (2-stage):
  Stage A (5 epochs): blend_head 병목 단독 검증 — adapters/lora/vca 동결
  Stage B (15 epochs): 전체 미세조정
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.entity_slot_phase43 import (
    Phase43Processor,
    MultiBlockSlotManagerP43,
    FeatureProjector,
    inject_multi_block_entity_slot_p43,
    restore_multiblock_state_p43,
    BLOCK_INNER_DIMS,
    CROSS_ATTN_DIM,
    DEFAULT_INJECT_KEYS,
    PRIMARY_DIM,
    l_slot_ref,
    l_slot_contrast,
    l_visible_weights_soft,
    l_blend_overlap,
    collect_blend_stats,
)
from models.entity_slot_phase42 import (
    l_w_residual,
    WeightHead,
)
import copy


# =============================================================================
# OverlapBlendHead
# =============================================================================

class OverlapBlendHead(nn.Module):
    """
    Overlap-aware blend prior + residual correction.

    입력 features (B, S, 8):
      alpha_0, alpha_1,
      overlap_proxy = alpha_0 * alpha_1,
      e0_front,
      sig[:,:,0,0], sig[:,:,0,1],
      sig[:,:,1,0], sig[:,:,1,1]

    base_blend = 0.05 + 0.25*max(alpha_0,alpha_1) + 0.60*(alpha_0*alpha_1)
    delta = MLP(feat) — (B,S,1), zero-init
    blend_map = sigmoid(logit(base_blend) + delta)

    zero-init 보장:
      초기 delta=0 → blend_map = base_blend (구조적 prior 그대로)
    """

    def __init__(self, in_features: int = 8, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, S, 8) → delta: (B, S, 1)"""
        return self.net(feat)


def compute_base_blend(
    alpha_0: torch.Tensor,
    alpha_1: torch.Tensor,
) -> torch.Tensor:
    """
    Overlap-aware base blend prior.

    base_blend = 0.05 + 0.25*entity_proxy + 0.60*overlap_proxy
    where:
      entity_proxy  = max(alpha_0, alpha_1)  — 어떤 entity라도 있으면 높음
      overlap_proxy = alpha_0 * alpha_1      — 두 entity 겹침에서 특히 높음

    초기 신호:
      bg:      ~0.05 (둘 다 0)
      single entity α≈0.5: ~0.05 + 0.25*0.5 = 0.175
      overlap α0=α1≈0.9:   ~0.05 + 0.25*0.9 + 0.60*0.81 = 0.791
    """
    entity_proxy  = torch.maximum(alpha_0, alpha_1)
    overlap_proxy = alpha_0 * alpha_1
    return (0.05 + 0.25 * entity_proxy + 0.60 * overlap_proxy).clamp(0.05, 0.95)


# =============================================================================
# Phase44Processor
# =============================================================================

class Phase44Processor(Phase43Processor):
    """
    Phase43Processor + OverlapBlendHead (blend_head 대체).

    blend_head (Phase40의 일반적 head) → OverlapBlendHead (overlap prior + delta).

    추가 저장 attributes:
      last_blend_map:     (B, S) — 최종 blend map (sigmoid 출력)
      last_base_blend:    (B, S) — overlap prior 기반 base blend
      last_overlap_proxy: (B, S) — alpha_0 * alpha_1
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
        )
        # Phase 44 추가: OverlapBlendHead
        self.overlap_blend_head = OverlapBlendHead(in_features=8, hidden=obh_hidden)

        # Phase 44 추가 저장
        self.last_blend_map_for_loss: Optional[torch.Tensor] = None  # grad path intact
        self.last_blend_map:      Optional[torch.Tensor] = None
        self.last_base_blend:     Optional[torch.Tensor] = None
        self.last_overlap_proxy:  Optional[torch.Tensor] = None

    def reset_slot_store(self):
        super().reset_slot_store()
        self.last_blend_map_for_loss = None
        self.last_blend_map     = None
        self.last_base_blend    = None
        self.last_overlap_proxy = None

    def overlap_blend_head_params(self) -> List[torch.nn.Parameter]:
        return list(self.overlap_blend_head.parameters())

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
        Phase42 __call__과 동일하나 blend_map 계산을 OverlapBlendHead로 교체.
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

            z_bins = sig.shape[-1]
            s00 = sig[:, :, 0, 0]
            s01 = sig[:, :, 0, min(1, z_bins-1)]
            s10 = sig[:, :, 1, 0]
            s11 = sig[:, :, 1, min(1, z_bins-1)]

            feat = torch.stack([
                alpha_0, alpha_1, s00, s01, s10, s11,
                alpha_0 * alpha_1, e0_front,
            ], dim=-1).float()

            delta  = self.weight_head(feat)
            base_logits = torch.log(
                torch.stack([base_wbg, base_w0, base_w1], dim=-1).clamp(min=1e-6))
            probs  = (base_logits + delta).softmax(dim=-1)
            w_bg, w0, w1 = probs.unbind(dim=-1)

            self.last_w_delta = delta

            w0_f  = w0.unsqueeze(-1).to(dtype=F_g.dtype)
            w1_f  = w1.unsqueeze(-1).to(dtype=F_g.dtype)
            wbg_f = w_bg.unsqueeze(-1).to(dtype=F_g.dtype)
            composed = w0_f * F_0 + w1_f * F_1 + wbg_f * F_g

            # ── Phase 44: OverlapBlendHead ────────────────────────────────
            overlap_proxy = alpha_0 * alpha_1                   # (B, S)
            base_blend    = compute_base_blend(alpha_0, alpha_1) # (B, S)

            feat_blend = torch.stack([
                alpha_0, alpha_1,
                overlap_proxy,
                e0_front,
                s00, s01, s10, s11,
            ], dim=-1).float()                                   # (B, S, 8)
            delta_b   = self.overlap_blend_head(feat_blend)      # (B, S, 1)
            blend_logit = torch.logit(base_blend, eps=1e-6).unsqueeze(-1)
            blend_map   = torch.sigmoid(blend_logit + delta_b)   # (B, S, 1)

            # Store for gradient-carrying loss (no detach — used by l_blend_target/rank)
            self.last_blend_map_for_loss = blend_map.squeeze(-1)       # (B, S) — grad path intact
            # Store diagnostics (detached — used for logging/val)
            self.last_blend_map     = blend_map.squeeze(-1).detach()   # (B, S)
            self.last_base_blend    = base_blend.detach()              # (B, S)
            self.last_overlap_proxy = overlap_proxy.detach()           # (B, S)
            # backward compatibility
            self.last_blend         = self.last_blend_map

            blend_map_f = blend_map.to(dtype=F_g.dtype)

        else:
            composed  = (F_0 + F_1 + F_g) / 3.0
            w0        = torch.ones(B, S, device=F_g.device) / 3
            w1        = torch.ones(B, S, device=F_g.device) / 3
            w_bg      = torch.ones(B, S, device=F_g.device) / 3
            blend_map_f = self.slot_blend.to(dtype=F_g.dtype)
            alpha_0   = w0
            alpha_1   = w1
            self.last_w_delta            = None
            self.last_blend_map_for_loss = None
            self.last_blend_map          = None
            self.last_blend              = None

        # ── Blend ────────────────────────────────────────────────────────
        blended = blend_map_f * composed + (1.0 - blend_map_f) * F_g

        # ── Store ────────────────────────────────────────────────────────
        self.last_w0 = w0
        self.last_w1 = w1
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
# Multi-block injection for Phase 44
# =============================================================================

def inject_multi_block_entity_slot_p44(
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
) -> Tuple[List[Phase44Processor], Dict]:
    if inject_keys is None:
        inject_keys = DEFAULT_INJECT_KEYS

    unet       = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))
    new_procs  = dict(unet.attn_processors)
    procs: List[Phase44Processor] = []

    for i, key in enumerate(inject_keys):
        inner_dim  = BLOCK_INNER_DIMS.get(key, 640)
        block_vca  = vca_layer if i == primary_idx else None
        block_ctx  = entity_ctx if i == primary_idx else None
        proc = Phase44Processor(
            query_dim           = inner_dim,
            vca_layer           = block_vca,
            entity_ctx          = block_ctx,
            slot_blend_init     = slot_blend_init,
            inner_dim           = inner_dim,
            adapter_rank        = adapter_rank,
            use_blend_head      = (use_blend_head and i == primary_idx),
            lora_rank           = lora_rank,
            cross_attention_dim = CROSS_ATTN_DIM,
            weight_head_hidden  = weight_head_hidden,
            primary_dim         = PRIMARY_DIM,
            proj_hidden         = proj_hidden,
            obh_hidden          = obh_hidden,
        )
        new_procs[key] = proc
        procs.append(proc)

    unet.set_attn_processor(new_procs)
    return procs, orig_procs


# =============================================================================
# MultiBlockSlotManagerP44
# =============================================================================

class MultiBlockSlotManagerP44(MultiBlockSlotManagerP43):
    """Phase44Processor용 manager. overlap_blend_head params 접근 추가."""

    def __init__(self, procs: List[Phase44Processor], keys: List[str],
                 primary_idx: int = 1):
        super().__init__(procs, keys, primary_idx)

    def overlap_blend_head_params(self) -> List[torch.nn.Parameter]:
        params = []
        for p in self.procs:
            if hasattr(p, 'overlap_blend_head_params'):
                params += p.overlap_blend_head_params()
        return params


# =============================================================================
# Checkpoint restoration for Phase 44
# =============================================================================

def restore_multiblock_state_p44(
    manager,
    ckpt:   dict,
    device: str = "cpu",
) -> None:
    """
    Phase40/41/42/43 checkpoint에서 state 복원.
    overlap_blend_head는 phase44 신규 → ckpt에 없으면 zero-init 유지.
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

        # overlap_blend_head (Phase44+) — zero-init if not in ckpt
        if "overlap_blend_head" in state and hasattr(proc, 'overlap_blend_head'):
            sd = {k: v.to(dev) for k, v in state["overlap_blend_head"].items()}
            proc.overlap_blend_head.load_state_dict(sd, strict=False)
            print(f"  [restore] block[{i}] overlap_blend_head loaded", flush=True)

        print(f"  [restore] block[{i}] OK  "
              f"(blend={float(proc.slot_blend.item()):.4f})", flush=True)


# =============================================================================
# GT blend supervision
# =============================================================================

def build_blend_targets(
    entity_masks_BNS: torch.Tensor,
    overlap_val:      float = 0.90,
    exclusive_val:    float = 0.35,
    bg_val:           float = 0.05,
) -> torch.Tensor:
    """
    GT mask에서 blend target 생성.

    overlap:   두 entity 겹침 → overlap_val (default 0.90)
    exclusive: 한 entity만    → exclusive_val (default 0.35)
    background: 어느 entity도 없음 → bg_val (default 0.05)

    Returns: blend_target (B, S) float [0, 1]
    """
    m0 = entity_masks_BNS[:, 0, :].float()
    m1 = entity_masks_BNS[:, 1, :].float()

    overlap   = m0 * m1
    exclusive = ((m0 + m1).clamp(0.0, 1.0) - overlap).clamp(0.0, 1.0)
    bg        = 1.0 - (m0 + m1).clamp(0.0, 1.0)

    return (overlap_val * overlap
            + exclusive_val * exclusive
            + bg_val * bg).clamp(0.0, 1.0)


def l_blend_target(
    blend_map:         torch.Tensor,   # (B, S)
    entity_masks_BNS:  torch.Tensor,   # (B, 2, S)
    overlap_val:        float = 0.90,
    exclusive_val:      float = 0.35,
    bg_val:             float = 0.05,
    overlap_weight:     float = 3.0,   # overlap 영역 가중치 강화
    eps:                float = 1e-6,
) -> torch.Tensor:
    """
    Weighted MSE vs GT blend target.

    overlap region에 overlap_weight를 곱해 강한 감독.
    """
    m0 = entity_masks_BNS[:, 0, :].float().to(blend_map.device)
    m1 = entity_masks_BNS[:, 1, :].float().to(blend_map.device)

    overlap   = m0 * m1
    exclusive = ((m0 + m1).clamp(0.0, 1.0) - overlap).clamp(0.0, 1.0)
    bg        = 1.0 - (m0 + m1).clamp(0.0, 1.0)

    target = (overlap_val * overlap
              + exclusive_val * exclusive
              + bg_val * bg).clamp(0.0, 1.0)

    # Per-pixel loss weight: overlap region upweighted
    weight = 1.0 + (overlap_weight - 1.0) * overlap

    diff = (blend_map.float() - target).pow(2) * weight
    return diff.mean()


def l_blend_rank(
    blend_map:        torch.Tensor,   # (B, S)
    entity_masks_BNS: torch.Tensor,   # (B, 2, S)
    margin:           float = 0.10,
    eps:              float = 1e-6,
) -> torch.Tensor:
    """
    Ranking loss: mean(overlap) > mean(exclusive) > mean(bg).

    L_rank = relu(margin - (blend_ov - blend_ex))
           + relu(margin - (blend_ex - blend_bg))
    """
    m0 = entity_masks_BNS[:, 0, :].float().to(blend_map.device)
    m1 = entity_masks_BNS[:, 1, :].float().to(blend_map.device)

    overlap   = m0 * m1
    exclusive = ((m0 + m1).clamp(0.0, 1.0) - overlap).clamp(0.0, 1.0)
    bg        = 1.0 - (m0 + m1).clamp(0.0, 1.0)

    n_ov  = overlap.sum()   + eps
    n_ex  = exclusive.sum() + eps
    n_bg  = bg.sum()        + eps

    bm = blend_map.float()
    blend_ov = (bm * overlap).sum()   / n_ov
    blend_ex = (bm * exclusive).sum() / n_ex
    blend_bg = (bm * bg).sum()        / n_bg

    return (F.relu(margin - (blend_ov - blend_ex))
            + F.relu(margin - (blend_ex - blend_bg)))


def collect_blend_stats_detailed(
    blend_map:        torch.Tensor,   # (B, S)
    entity_masks_BNS: torch.Tensor,   # (B, 2, S)
    eps:              float = 1e-6,
) -> dict:
    """
    overlap / exclusive / bg별 blend 통계.

    blend_sep     = overlap_mean - exclusive_mean
    blend_gap_bg  = exclusive_mean - bg_mean
    """
    m0 = entity_masks_BNS[:, 0, :].float().to(blend_map.device)
    m1 = entity_masks_BNS[:, 1, :].float().to(blend_map.device)

    overlap   = m0 * m1
    exclusive = ((m0 + m1).clamp(0.0, 1.0) - overlap).clamp(0.0, 1.0)
    bg        = 1.0 - (m0 + m1).clamp(0.0, 1.0)

    bm   = blend_map.float()
    n_ov = float(overlap.sum().item()) + eps
    n_ex = float(exclusive.sum().item()) + eps
    n_bg = float(bg.sum().item()) + eps

    b_ov = float((bm * overlap).sum().item())   / n_ov
    b_ex = float((bm * exclusive).sum().item()) / n_ex
    b_bg = float((bm * bg).sum().item())        / n_bg

    return {
        "blend_mean":           float(bm.mean().item()),
        "blend_overlap_mean":   b_ov,
        "blend_exclusive_mean": b_ex,
        "blend_bg_mean":        b_bg,
        "blend_sep":            b_ov - b_ex,
        "blend_gap_bg":         b_ex - b_bg,
    }


# =============================================================================
# Phase 44 validation score
# =============================================================================

def val_score_phase44(
    tf_iou_e0:        float,
    tf_iou_e1:        float,
    tf_ord:           float,
    tf_wrong:         float,
    rollout_iou_e0:   float = 0.0,
    rollout_iou_e1:   float = 0.0,
    blend_sep:        float = 0.0,
) -> float:
    """
    Phase 44 validation score (blend_score 포함).

    weights:
      tf_iou_e0        0.15
      tf_iou_e1        0.15
      tf_ord           0.10
      (1-tf_wrong)     0.10
      rollout_iou_e0   0.20
      rollout_iou_e1   0.20
      blend_score      0.10

    blend_score = clamp((blend_sep + 0.15) / 0.30, 0, 1)
      blend_sep = -0.15 → score=0
      blend_sep =  0.00 → score=0.5
      blend_sep =  0.15 → score=1.0
    """
    blend_score = max(0.0, min(1.0, (blend_sep + 0.15) / 0.30))
    return (0.15 * tf_iou_e0
          + 0.15 * tf_iou_e1
          + 0.10 * tf_ord
          + 0.10 * (1.0 - tf_wrong)
          + 0.20 * rollout_iou_e0
          + 0.20 * rollout_iou_e1
          + 0.10 * blend_score)
