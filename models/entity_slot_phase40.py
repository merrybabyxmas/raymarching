"""
Phase 40 — Entity-Layer Consistency Distillation
=================================================

Phase 39의 weight supervision + appearance identity supervision 추가.

핵심 아이디어: "weight가 맞는 것"이 아니라 "appearance가 분리된 것"을 직접 학습.

Phase 39 → Phase 40 핵심 변경
-------------------------------
1. 3-block injection (up_blocks.1/2/3) — appearance 용량 확보
2. SlotLoRA (rank=4~8) on K, V, out — appearance channel 변환 능력
3. L_solo_feat_visible: F_slot → F_ref alignment in visible entity regions
4. L_id_contrast: cosine margin, entity0 slot ↔ entity0 ref > entity0 slot ↔ entity1 ref
5. L_bg_feat: background region에서 Fg_comp → Fg_ref 일치
6. Per-entity visible_iou_e0 / visible_iou_e1 (기존 combined iou 대체)
7. id_feature_margin metric: appearance separation 측정
8. Rollout eval metric (multi-step denoising quality)

Phase 39 핵심 보존
------------------
- sigma_raw (not detached) 사용 for Porter-Duff — Phase 39에서 발견한 핵심 버그픽스
- blend_head inputs NOT detached
- l_sigma_spatial 유지
- val_slot_score 대신 val_score_phase40 사용

Architecture per block
-----------------------
  slot0_adapter, slot1_adapter   (residual adapter, rank=64)
  blend_head                      (per-pixel blend map)
  lora_k, lora_v, lora_out        (LoRA, rank=4 or 8)

  shared: vca_layer (Porter-Duff depth ordering)

LoRA dimensions (SD1.5 AnimateDiff)
-------------------------------------
  up_blocks.1: inner_dim=1280, cross_attention_dim=768
  up_blocks.2: inner_dim=640  (Phase 39 block)
  up_blocks.3: inner_dim=320
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.entity_slot import (
    SlotAdapter,
    BlendHead,
    EntitySlotAttnProcessor,
    build_visible_targets,
    l_visible_weights,
    l_wrong_slot_suppression,
    l_sigma_spatial,
    l_entity_exclusive,
    l_overlap_ordering,
    compute_overlap_score,
)


# SD1.5 AnimateDiff 기준 block별 inner_dim
BLOCK_INNER_DIMS: Dict[str, int] = {
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor": 1280,
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor": 640,
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor": 320,
}
CROSS_ATTN_DIM = 768   # SD1.5 text encoder output dim
DEFAULT_INJECT_KEYS = list(BLOCK_INNER_DIMS.keys())


# =============================================================================
# SlotLoRA: LoRA adapter for K, V, out projections
# =============================================================================

class SlotLoRA(nn.Module):
    """
    Low-rank adaptation for a single linear projection.

    Δout = lora_B(lora_A(x)) × (scale / rank)

    zero-init (lora_B.weight = 0) → identity at start.
    LoRA가 없으면 adapter만으론 attention K/V/out channel을 직접 못 바꿔
    entity appearance를 분리하기 어려움.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scale  = 1.0 / rank

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)   # zero-init → identity at start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scale


# =============================================================================
# Phase40Processor: Phase39 + LoRA
# =============================================================================

class Phase40Processor(EntitySlotAttnProcessor):
    """
    Phase 39 EntitySlotAttnProcessor + SlotLoRA on K, V, out.

    LoRA가 추가되면 attention이 entity-specific appearance feature를
    직접 다른 방향으로 project할 수 있어 slot adapter만으론 부족했던
    appearance separation 용량이 확보됨.

    Parameters
    ----------
    query_dim          : D (= inner_dim of this attention block)
    inner_dim          : attention inner dim (= query_dim for SD1.5 attn2)
    cross_attention_dim: K/V input dim (768 for SD1.5 text encoder)
    lora_rank          : LoRA rank (default 4; use 8 for larger blocks)
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
    ):
        super().__init__(
            query_dim       = query_dim,
            vca_layer       = vca_layer,
            entity_ctx      = entity_ctx,
            slot_blend_init = slot_blend_init,
            inner_dim       = inner_dim,
            adapter_rank    = adapter_rank,
            use_blend_head  = use_blend_head,
        )
        self.lora_rank = lora_rank
        d = self._inner_dim

        # LoRA on K, V (cross-attention: in=cross_attention_dim, out=inner_dim)
        self.lora_k = SlotLoRA(cross_attention_dim, d, rank=lora_rank)
        self.lora_v = SlotLoRA(cross_attention_dim, d, rank=lora_rank)
        # LoRA on out projection (in=inner_dim, out=inner_dim for SD1.5)
        self.lora_out = SlotLoRA(d, d, rank=lora_rank)

        # extra store for solo distillation
        self.last_F0_ref: Optional[torch.Tensor] = None   # (B, S, D)
        self.last_F1_ref: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def reset_slot_store(self):
        super().reset_slot_store()
        self.last_F0_ref = None
        self.last_F1_ref = None

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

        # ── Q ──────────────────────────────────────────────────────────
        q    = attn.to_q(hidden_states)
        q_mh = _mh(q, S)

        # ── Global attention (F_global) ──────────────────────────────────
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

        # ── VCA sigma (Phase 39 fix: use sigma_raw when training) ────────
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
                    # use sigma_raw for Porter-Duff (Phase 39 critical gradient fix)
                    sigma = sigma_raw

        # ── Porter-Duff compositing ──────────────────────────────────────
        if sigma is not None and sigma.shape[:2] == (B, S):
            sig = sigma.to(device=F_g.device, dtype=torch.float32)

            alpha_0  = sig[:, :, 0, :].max(dim=-1).values
            alpha_1  = sig[:, :, 1, :].max(dim=-1).values
            e0_front = torch.sigmoid(5.0 * (sig[:, :, 0, 0] - sig[:, :, 1, 0]))

            w0   = e0_front * alpha_0 + (1.0 - e0_front) * alpha_0 * (1.0 - alpha_1)
            w1   = (1.0 - e0_front) * alpha_1 + e0_front * alpha_1 * (1.0 - alpha_0)
            w_bg = (1.0 - w0 - w1).clamp(min=0.0)

            w0_f  = w0.unsqueeze(-1).to(dtype=F_g.dtype)
            w1_f  = w1.unsqueeze(-1).to(dtype=F_g.dtype)
            wbg_f = w_bg.unsqueeze(-1).to(dtype=F_g.dtype)

            composed = w0_f * F_0 + w1_f * F_1 + wbg_f * F_g

            # blend_head: no detach (Phase 39 fix, stage2 gradient path)
            if self.use_blend_head:
                blend_map = self.blend_head(alpha_0, alpha_1, e0_front)
                blend_map = blend_map.to(dtype=F_g.dtype)
            else:
                blend_map = self.slot_blend.to(dtype=F_g.dtype)
        else:
            composed  = (F_0 + F_1 + F_g) / 3.0
            w0        = torch.ones(B, S, device=F_g.device) / 3
            w1        = torch.ones(B, S, device=F_g.device) / 3
            blend_map = self.slot_blend.to(dtype=F_g.dtype)
            alpha_0   = w0
            alpha_1   = w1

        # ── Blend ────────────────────────────────────────────────────────
        blended = blend_map * composed + (1.0 - blend_map) * F_g

        # ── Store ────────────────────────────────────────────────────────
        self.last_w0 = w0
        self.last_w1 = w1
        self.last_blend = (blend_map.squeeze(-1)
                           if isinstance(blend_map, torch.Tensor) and blend_map.dim() == 3
                           else blend_map)
        if sigma is not None and sigma.shape[:2] == (B, S):
            self.last_alpha0 = alpha_0
            self.last_alpha1 = alpha_1
        if self.training:
            self.last_F0 = F_0
            self.last_F1 = F_1
            self.last_Fg = F_g

        # ── Output projection with LoRA ──────────────────────────────────
        out = (attn.to_out[0](blended)
               + self.lora_out(blended.float()).to(dtype=blended.dtype))
        out = attn.to_out[1](out)
        return out.to(dtype)


# =============================================================================
# Multi-block injection
# =============================================================================

import copy


def inject_multi_block_entity_slot(
    pipe,
    vca_layer,
    entity_ctx:       torch.Tensor,
    inject_keys:      Optional[List[str]] = None,
    slot_blend_init:  float = 0.3,
    adapter_rank:     int   = 64,
    lora_rank:        int   = 4,
    use_blend_head:   bool  = True,
) -> Tuple[List[Phase40Processor], Dict]:
    """
    3개 attention block에 Phase40Processor 주입.

    Returns
    -------
    procs     : List[Phase40Processor] — 순서: inject_keys 순서
    orig_procs: 원본 processor dict (복원용)
    """
    if inject_keys is None:
        inject_keys = DEFAULT_INJECT_KEYS

    unet       = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))
    new_procs  = dict(unet.attn_processors)
    procs      = []

    for key in inject_keys:
        inner_dim = BLOCK_INNER_DIMS.get(key, 640)
        proc = Phase40Processor(
            query_dim           = inner_dim,
            vca_layer           = vca_layer,
            entity_ctx          = entity_ctx,
            slot_blend_init     = slot_blend_init,
            inner_dim           = inner_dim,
            adapter_rank        = adapter_rank,
            use_blend_head      = use_blend_head,
            lora_rank           = lora_rank,
        )
        new_procs[key] = proc
        procs.append(proc)

    unet.set_attn_processor(new_procs)
    return procs, orig_procs


class MultiBlockSlotManager:
    """
    여러 Phase40Processor를 묶어서 한꺼번에 관리.

    primary_idx = 1 (up_blocks.2, Phase39와 동일한 block) — w0/w1 읽기에 사용.
    """

    def __init__(self, procs: List[Phase40Processor], keys: List[str],
                 primary_idx: int = 1):
        self.procs       = procs
        self.keys        = keys
        self.primary_idx = min(primary_idx, len(procs) - 1)

    @property
    def primary(self) -> Phase40Processor:
        return self.procs[self.primary_idx]

    def train(self):
        for p in self.procs: p.train()

    def eval(self):
        for p in self.procs: p.eval()

    def set_entity_ctx(self, ctx: torch.Tensor):
        for p in self.procs: p.set_entity_ctx(ctx)

    def set_entity_tokens(self, toks_e0: List[int], toks_e1: List[int]):
        for p in self.procs: p.set_entity_tokens(toks_e0, toks_e1)

    def reset_slot_store(self):
        for p in self.procs: p.reset_slot_store()

    @property
    def last_w0(self): return self.primary.last_w0

    @property
    def last_w1(self): return self.primary.last_w1

    @property
    def last_alpha0(self): return self.primary.last_alpha0

    @property
    def last_alpha1(self): return self.primary.last_alpha1

    @property
    def sigma_acc(self):
        # merge sigma_acc from all blocks (primary only to avoid redundancy)
        return self.primary.sigma_acc

    def all_F0(self) -> List[Optional[torch.Tensor]]:
        return [p.last_F0 for p in self.procs]

    def all_F1(self) -> List[Optional[torch.Tensor]]:
        return [p.last_F1 for p in self.procs]

    def all_Fg(self) -> List[Optional[torch.Tensor]]:
        return [p.last_Fg for p in self.procs]

    def all_params(self) -> List[torch.nn.Parameter]:
        """모든 학습 가능한 파라미터 반환."""
        params = []
        for p in self.procs:
            params += list(p.parameters())
        return params

    def adapter_params(self) -> List[torch.nn.Parameter]:
        params = []
        for p in self.procs:
            params += (list(p.slot0_adapter.parameters())
                       + list(p.slot1_adapter.parameters()))
        return params

    def lora_params(self) -> List[torch.nn.Parameter]:
        params = []
        for p in self.procs:
            params += (list(p.lora_k.parameters())
                       + list(p.lora_v.parameters())
                       + list(p.lora_out.parameters()))
        return params

    def blend_params(self) -> List[torch.nn.Parameter]:
        params = []
        for p in self.procs:
            params += list(p.blend_head.parameters())
            params += [p.slot_blend_raw]
        return params


# =============================================================================
# Visible mask computation (from full masks + depth ordering)
# =============================================================================

def compute_visible_masks(
    entity_masks_TNS,            # (T, 2, S) Tensor or ndarray, float or bool
    depth_orders_T:   list,
) -> torch.Tensor:
    """
    GT visible mask computation.

    front entity: fully visible
    back entity: visible only where front entity mask = 0

    Returns (T, 2, S) float32 Tensor.
    Accepts both torch.Tensor and np.ndarray inputs.
    """
    import numpy as np
    if isinstance(entity_masks_TNS, np.ndarray):
        entity_masks_TNS = torch.from_numpy(entity_masks_TNS.astype(np.float32))
    T = entity_masks_TNS.shape[0]
    visible = torch.zeros_like(entity_masks_TNS, dtype=torch.float32)

    for t in range(min(T, len(depth_orders_T))):
        front = int(depth_orders_T[t][0])
        back  = int(depth_orders_T[t][1])
        m_front = entity_masks_TNS[t, front].float()
        m_back  = entity_masks_TNS[t, back].float()
        visible[t, front] = m_front
        visible[t, back]  = m_back * (1.0 - m_front)

    return visible


# =============================================================================
# Phase 40 loss functions
# =============================================================================

def l_solo_feat_visible(
    F_slot:          torch.Tensor,   # (B, S, D) — entity slot features (composite)
    F_ref:           torch.Tensor,   # (B, S, D) — solo/entity-prompted reference
    visible_mask_BS: torch.Tensor,   # (B, S) float — 1 where entity is visible
    eps:             float = 1e-6,
) -> torch.Tensor:
    """
    L_solo_feat: visible entity 영역에서 slot features → reference features 맞추기.

    F_ref는 stop-gradient (detach) — slot features가 reference를 따라가야 함.

    Phase 40의 핵심 loss: "cat은 cat으로, dog는 dog로 남아라"를 강제.
    """
    m = visible_mask_BS.float().unsqueeze(-1).to(F_slot.device)   # (B, S, 1)
    n = m.sum() * F_slot.shape[-1] + eps

    diff = (F_slot.float() - F_ref.float().detach()).pow(2)
    return (diff * m).sum() / n


def l_id_contrast(
    F0_slot:   torch.Tensor,   # (B, S, D)
    F1_slot:   torch.Tensor,
    F0_ref:    torch.Tensor,   # (B, S, D) — entity0 solo reference
    F1_ref:    torch.Tensor,   # (B, S, D) — entity1 solo reference
    mask_e0:   torch.Tensor,   # (B, S) float
    mask_e1:   torch.Tensor,
    margin:    float = 0.1,
    eps:       float = 1e-6,
) -> torch.Tensor:
    """
    L_id_contrast: cosine margin loss for entity appearance separation.

    entity0 영역: cos(F0_slot, F0_ref) > cos(F0_slot, F1_ref) + margin
    entity1 영역: cos(F1_slot, F1_ref) > cos(F1_slot, F0_ref) + margin

    F_ref는 stop-gradient.
    """
    F0_s = F.normalize(F0_slot.float(), dim=-1)
    F1_s = F.normalize(F1_slot.float(), dim=-1)
    F0_r = F.normalize(F0_ref.float().detach(), dim=-1)
    F1_r = F.normalize(F1_ref.float().detach(), dim=-1)

    m0 = mask_e0.float().unsqueeze(-1).to(F0_s.device)
    m1 = mask_e1.float().unsqueeze(-1).to(F1_s.device)

    # entity0 slot: closer to F0_ref than F1_ref
    sim_00 = (F0_s * F0_r).sum(-1, keepdim=True)  # (B, S, 1)
    sim_01 = (F0_s * F1_r).sum(-1, keepdim=True)
    loss_0 = F.relu(sim_01 - sim_00 + margin)
    n0     = m0.sum() + eps
    loss_0 = (loss_0 * m0).sum() / n0

    # entity1 slot: closer to F1_ref than F0_ref
    sim_11 = (F1_s * F1_r).sum(-1, keepdim=True)
    sim_10 = (F1_s * F0_r).sum(-1, keepdim=True)
    loss_1 = F.relu(sim_10 - sim_11 + margin)
    n1     = m1.sum() + eps
    loss_1 = (loss_1 * m1).sum() / n1

    return (loss_0 + loss_1) * 0.5


def l_bg_feat(
    Fg_slot:    torch.Tensor,   # (B, S, D)
    Fg_ref:     torch.Tensor,   # (B, S, D)
    bg_mask_BS: torch.Tensor,   # (B, S) float — background region
    eps:        float = 1e-6,
) -> torch.Tensor:
    """
    L_bg_feat: background 영역에서 global feature → reference 유지.

    entity가 없는 background에서 feature가 갑자기 바뀌면 안 됨.
    """
    m = bg_mask_BS.float().unsqueeze(-1).to(Fg_slot.device)
    n = m.sum() * Fg_slot.shape[-1] + eps
    diff = (Fg_slot.float() - Fg_ref.float().detach()).pow(2)
    return (diff * m).sum() / n


# =============================================================================
# Phase 40 metrics
# =============================================================================

def compute_visible_iou_e0(
    w0:               torch.Tensor,
    entity_masks_BNS: torch.Tensor,
    depth_orders_B:   list,
    eps:              float = 1e-6,
) -> float:
    """entity0 전용 visible IoU."""
    m0 = entity_masks_BNS[:, 0, :].float().to(w0.device)
    m1 = entity_masks_BNS[:, 1, :].float().to(w0.device)
    excl_0  = m0 * (1.0 - m1)
    overlap = m0 * m1

    w0_target = excl_0.clone()
    for b in range(min(w0.shape[0], len(depth_orders_B))):
        if int(depth_orders_B[b][0]) == 0:
            w0_target[b] = w0_target[b] + overlap[b]

    inter = (w0.float() * w0_target).sum()
    union = (w0.float() + w0_target - w0.float() * w0_target).sum()
    return float((inter / (union + eps)).item())


def compute_visible_iou_e1(
    w1:               torch.Tensor,
    entity_masks_BNS: torch.Tensor,
    depth_orders_B:   list,
    eps:              float = 1e-6,
) -> float:
    """entity1 전용 visible IoU."""
    m0 = entity_masks_BNS[:, 0, :].float().to(w1.device)
    m1 = entity_masks_BNS[:, 1, :].float().to(w1.device)
    excl_1  = m1 * (1.0 - m0)
    overlap = m0 * m1

    w1_target = excl_1.clone()
    for b in range(min(w1.shape[0], len(depth_orders_B))):
        if int(depth_orders_B[b][0]) == 1:
            w1_target[b] = w1_target[b] + overlap[b]

    inter = (w1.float() * w1_target).sum()
    union = (w1.float() + w1_target - w1.float() * w1_target).sum()
    return float((inter / (union + eps)).item())


def compute_id_feature_margin(
    F0_slot:   torch.Tensor,   # (B, S, D)
    F1_slot:   torch.Tensor,
    F0_ref:    torch.Tensor,
    F1_ref:    torch.Tensor,
    mask_e0:   torch.Tensor,   # (B, S)
    mask_e1:   torch.Tensor,
    eps:       float = 1e-6,
) -> float:
    """
    Identity feature margin: appearance separation 측정.

    margin = 0.5 × [ avg_e0(cos(F0,F0r) - cos(F0,F1r))
                    + avg_e1(cos(F1,F1r) - cos(F1,F0r)) ]

    높을수록 entity0과 entity1 appearance가 잘 분리됨.
    """
    F0_s = F.normalize(F0_slot.float(), dim=-1)
    F1_s = F.normalize(F1_slot.float(), dim=-1)
    F0_r = F.normalize(F0_ref.float(), dim=-1)
    F1_r = F.normalize(F1_ref.float(), dim=-1)

    m0 = mask_e0.float().to(F0_s.device)
    m1 = mask_e1.float().to(F1_s.device)

    sim_00 = (F0_s * F0_r).sum(-1)   # (B, S)
    sim_01 = (F0_s * F1_r).sum(-1)
    margin_0 = ((sim_00 - sim_01) * m0).sum() / (m0.sum() + eps)

    sim_11 = (F1_s * F1_r).sum(-1)
    sim_10 = (F1_s * F0_r).sum(-1)
    margin_1 = ((sim_11 - sim_10) * m1).sum() / (m1.sum() + eps)

    return float(((margin_0 + margin_1) * 0.5).item())


def val_score_phase40(
    iou_e0:      float,
    iou_e1:      float,
    ordering_acc: float,
    wrong_leak:  float,
    id_margin:   float,
    rollout_id:  float = 0.0,
) -> float:
    """
    Phase 40 복합 validation score.

    가중치:
      visible_iou_e0   0.20 — entity0 weight alignment
      visible_iou_e1   0.20 — entity1 weight alignment
      ordering_acc     0.20 — front/back 순서 정확도
      wrong_slot_leak  0.15 — exclusive 영역 wrong entity
      id_feature_margin 0.15 — appearance separation (Phase 40 신규)
      rollout_id_score  0.10 — free generation identity quality (Phase 40 신규)
    """
    return (0.20 * iou_e0
          + 0.20 * iou_e1
          + 0.20 * ordering_acc
          + 0.15 * (1.0 - wrong_leak)
          + 0.15 * id_margin
          + 0.10 * rollout_id)


# =============================================================================
# Reference feature extractor (solo or entity-prompted)
# =============================================================================

def extract_entity_ref_features(
    pipe,
    manager:     MultiBlockSlotManager,
    latents:     torch.Tensor,             # (1, C, T, H, W)
    t_tensor:    torch.Tensor,             # (1,) timestep
    entity_ctx:  torch.Tensor,             # (1, 2, 768)
    toks_e0:     List[int],
    toks_e1:     List[int],
    entity_idx:  int,                       # 0 or 1
    full_prompt: str,
    device:      str,
) -> Optional[torch.Tensor]:
    """
    Entity-prompted reference feature extraction.

    entity0 ref: run UNet with entity0-only text prompt → extract F_0
    entity1 ref: run UNet with entity1-only text prompt → extract F_1

    solo render가 없을 때의 근사: composite latent에 entity-only 텍스트를 주면
    "entity_i만 존재한다면 어떻게 보일까"에 근사한 feature를 얻을 수 있음.

    NOTE: frozen UNet + proc.eval() 모드로 실행. gradient 없음.
    """
    proc = manager.primary

    # build entity-only prompt
    # entity_ctx: (1, 2, 768) → entity_i text embedding
    proc.eval()
    for blk in manager.procs:
        blk.eval()

    manager.reset_slot_store()
    manager.set_entity_ctx(entity_ctx.float())
    manager.set_entity_tokens(toks_e0, toks_e1)

    # tokenize entity-only prompt
    meta_key = 'prompt_entity0' if entity_idx == 0 else 'prompt_entity1'

    try:
        with torch.no_grad():
            # use entity-specific text (entity1 slots all go to padding for entity0 run)
            # simplification: mask entity_i+1 tokens by setting their positions to empty
            enc_ctx_solo = entity_ctx.clone()
            other_idx = 1 - entity_idx
            enc_ctx_solo[:, other_idx, :] = 0.0   # zero out other entity embedding

            proc.set_entity_ctx(enc_ctx_solo.float())

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = pipe.unet(
                    latents, t_tensor,
                    encoder_hidden_states=enc_ctx_solo.to(device).half().expand(
                        latents.shape[0], -1, -1),
                ).sample

        if entity_idx == 0:
            return proc.last_F0
        else:
            return proc.last_F1

    except Exception:
        return None
