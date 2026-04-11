"""
Phase 38/39 — Entity-Slot Attention with Porter-Duff Compositing
=================================================================

근본 문제
---------
Phase 35~37 (volumetric z_pe): 단 하나의 attention stream에서 z_pe로 front/back을
분리하려 했지만, K/V는 여전히 두 entity token을 모두 포함 → z_pe가 아무리 커져도
front bin이 여전히 back entity 토큰에 주목할 수 있음. 공유 appearance stream이
chimera의 근본 원인.

Phase 38 핵심 아이디어: Entity Slot Attention
-------------------------------------------
각 entity 전용 attention stream (slot) 을 만들어 완전히 분리된 K/V 공간에서 처리한다.

    F_global = Attn(Q, K[all_tokens], V[all_tokens])   — 원본 품질 보존용
    F_0      = Attn(Q, K[e0_toks],   V[e0_toks])       — entity 0 전용
    F_1      = Attn(Q, K[e1_toks],   V[e1_toks])       — entity 1 전용

Phase 39 개선 사항
-------------------
1. slot0_adapter / slot1_adapter: F_0, F_1에 학습 가능한 residual adapter 추가
   → L_exclusive가 구조적으로 강해짐 (frozen q/k/v만으론 부족)
2. blend_head: 전체 동일한 scalar blend가 아닌 per-pixel blend map
   → collision 영역만 slot을 강하게 쓰고 나머지는 global 유지
3. build_visible_targets / l_visible_weights: GT mask + depth order로 w0/w1 직접 supervision
   → "앞뒤 대충 맞추기"가 아닌 실제 visible region 학습
4. l_wrong_slot_suppression: exclusive 영역에서 wrong entity weight 패널티
5. entity_score 계열은 debug-only — checkpoint selection은 GT-mask 기반 val_slot_score

Porter-Duff 합성 (VCA sigma 기반 depth 순서)
--------------------------------------------
    alpha_0 = sigmoid(vca_sigma[:,:,0,:].max(z))   — entity 0 점유율
    alpha_1 = sigmoid(vca_sigma[:,:,1,:].max(z))   — entity 1 점유율
    e0_front = sigmoid(5 * (sigma_0_z0 - sigma_1_z0))  — front 판정 (soft)

    w0   = e0_front * alpha_0 + (1-e0_front) * alpha_0 * (1-alpha_1)
    w1   = (1-e0_front) * alpha_1 + e0_front * alpha_1 * (1-alpha_0)
    w_bg = (1-w0-w1).clamp(0)

    composed = w0*F_0 + w1*F_1 + w_bg*F_global
    output   = blend_map*composed + (1-blend_map)*F_global

blend_head: per-pixel blend map (alpha0, alpha1, alpha0*alpha1, e0_front) → (B,S,1)
slot_blend_raw: 하위 호환용 scalar (Phase38 checkpoint 로드 시 사용)

손실 함수
---------
Phase 39:
  L = λ_vis   × L_visible_weights     ← GT visible target 직접 supervision (핵심)
    + λ_wrong  × L_wrong_slot_suppression ← exclusive 영역 wrong entity 패널티
    + λ_ov     × L_overlap_ordering    ← 겹침 영역 front > back (보조)
    + λ_depth  × L_depth               ← VCA sigma depth ordering
    + λ_diff   × L_diff                ← 생성 품질 (2-stage: 처음엔 0, 후반 ramp-up)

메트릭 (validation — GT mask 기반, RGB threshold 아님)
------
  visible_iou:      predicted w0/w1 vs GT visible targets의 soft IoU
  ordering_acc:     overlap 영역에서 front entity weight > back entity weight 비율
  wrong_slot_leak:  exclusive 영역에서 wrong entity weight 평균
  dra:              depth rank accuracy (Phase 31과 동일)
  val_slot_score = 0.4*visible_iou + 0.3*ordering_acc + 0.2*(1-wrong_slot_leak) + 0.1*dra

디버그 전용 (checkpoint selection에 사용하지 말 것):
  entity_survival_rate, chimera_rate, entity_score — RGB threshold 기반이라
  배경색도 entity로 잡으며 entity 소멸을 chimera 감소로 오인할 수 있음.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# SlotAdapter: small residual adapter for entity slot outputs
# =============================================================================

class SlotAdapter(nn.Module):
    """
    Small residual adapter: LayerNorm → Linear(dim, r) → SiLU → Linear(r, dim)

    zero-initialized 출력으로 시작 (F_i에 아무 영향도 주지 않다가 점진적으로 학습).
    Phase 38의 frozen q/k/v만으로는 L_exclusive가 구조적으로 약하므로 도입.
    """

    def __init__(self, dim: int, r: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, r, bias=True)
        self.act  = nn.SiLU()
        self.up   = nn.Linear(r, dim, bias=True)
        # zero init: 학습 초기엔 identity (F_i + 0 = F_i)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, dim) — float32 recommended
        return x + self.up(self.act(self.down(self.norm(x))))


# =============================================================================
# BlendHead: per-pixel blend map (replaces scalar slot_blend_raw)
# =============================================================================

class BlendHead(nn.Module):
    """
    Per-pixel blend map from (alpha0, alpha1, alpha0*alpha1, e0_front) → [0,1].

    collision 영역 (alpha0*alpha1 ↑) 에서 blend ↑, 단독 영역에선 blend↓ 할 수 있게
    spatial control을 줌. scalar slot_blend_raw와 달리 위치별로 다른 blend.

    init_bias: sigmoid(init_bias) ≈ slot_blend_init (default 0.3 → bias ≈ -0.847)
    """

    def __init__(self, hidden: int = 16, init_bias: float = -0.847):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # zero-init linear weights, set bias for slot_blend_init ≈ 0.3
        nn.init.zeros_(self.net[2].weight)
        nn.init.constant_(self.net[2].bias, init_bias)

    def forward(
        self,
        alpha0:   torch.Tensor,   # (B, S)
        alpha1:   torch.Tensor,   # (B, S)
        e0_front: torch.Tensor,   # (B, S)
    ) -> torch.Tensor:
        """Returns per-pixel blend map (B, S, 1) ∈ [0, 1]."""
        feats = torch.stack([
            alpha0,
            alpha1,
            alpha0 * alpha1,
            e0_front,
        ], dim=-1)  # (B, S, 4)
        return torch.sigmoid(self.net(feats))  # (B, S, 1)


# =============================================================================
# EntitySlotAttnProcessor
# =============================================================================

class EntitySlotAttnProcessor(nn.Module):
    """
    attn2 (text cross-attention) 를 entity-slot 방식으로 교체.

    Parameters
    ----------
    query_dim      : D (= INJECT_QUERY_DIM, up_blocks.2에서는 640)
    inner_dim      : attention inner dim (기본값 = query_dim; 640)
    vca_layer      : VCALayer — sigma 계산용 (depth ordering)
    entity_ctx     : (1, N=2, 768) entity CLIP 임베딩
    slot_blend_init: slot blend ratio 초기값 (0=순수 global, 1=순수 slot)
    adapter_rank   : SlotAdapter bottleneck dim (default 64)
    use_blend_head : True → per-pixel BlendHead, False → scalar slot_blend_raw
    """

    def __init__(
        self,
        query_dim: int,
        vca_layer=None,
        entity_ctx: Optional[torch.Tensor] = None,
        slot_blend_init: float = 0.3,
        inner_dim: Optional[int] = None,
        adapter_rank: int = 64,
        use_blend_head: bool = True,
    ):
        super().__init__()
        self.query_dim = query_dim
        self._inner_dim = inner_dim if inner_dim is not None else query_dim
        self.vca = vca_layer
        self.entity_ctx = entity_ctx
        self.use_blend_head = use_blend_head

        # ── scalar blend (Phase 38 backward compat + fallback) ─────────
        raw_init = math.log(slot_blend_init / (1.0 - slot_blend_init + 1e-8))
        self.slot_blend_raw = nn.Parameter(torch.tensor(float(raw_init)))

        # ── Phase 39: per-pixel blend head ─────────────────────────────
        self.blend_head = BlendHead(hidden=16, init_bias=float(raw_init))

        # ── Phase 39: slot adapters ──────────────────────────────────────
        self.slot0_adapter = SlotAdapter(self._inner_dim, r=adapter_rank)
        self.slot1_adapter = SlotAdapter(self._inner_dim, r=adapter_rank)

        # entity token index lists — set before each forward
        self.toks_e0: List[int] = []
        self.toks_e1: List[int] = []

        # 훈련 루프에서 loss 계산을 위해 보관
        self.last_F0:     Optional[torch.Tensor] = None   # (B, S, inner_dim)
        self.last_F1:     Optional[torch.Tensor] = None
        self.last_Fg:     Optional[torch.Tensor] = None
        self.last_w0:     Optional[torch.Tensor] = None   # (B, S)
        self.last_w1:     Optional[torch.Tensor] = None
        self.last_blend:  Optional[torch.Tensor] = None   # (B, S) or scalar
        self.last_sigma:  Optional[torch.Tensor] = None
        self.last_alpha0: Optional[torch.Tensor] = None   # (B, S) Porter-Duff alpha
        self.last_alpha1: Optional[torch.Tensor] = None
        self.sigma_acc:   list = []

    # ------------------------------------------------------------------
    def reset_slot_store(self):
        self.last_F0     = None
        self.last_F1     = None
        self.last_Fg     = None
        self.last_w0     = None
        self.last_w1     = None
        self.last_blend  = None
        self.last_sigma  = None
        self.last_alpha0 = None
        self.last_alpha1 = None
        self.sigma_acc   = []

    def set_entity_ctx(self, ctx: torch.Tensor):
        self.entity_ctx = ctx

    def set_entity_tokens(self, toks_e0: List[int], toks_e1: List[int]):
        self.toks_e0 = toks_e0
        self.toks_e1 = toks_e1

    # ------------------------------------------------------------------
    @property
    def slot_blend(self) -> torch.Tensor:
        """Scalar blend (backward compat / fallback)."""
        return torch.sigmoid(self.slot_blend_raw)

    # ------------------------------------------------------------------
    def _masked_attn(
        self,
        q_mh:      torch.Tensor,    # (B, H, S, Dh)
        k_mh:      torch.Tensor,    # (B, H, T, Dh)
        v_mh:      torch.Tensor,    # (B, H, T, Dh)
        tok_idx:   List[int],
        T_seq:     int,
        scale:     float,
        B:         int,
        S:         int,
        n_heads:   int,
        head_dim:  int,
        inner_dim: int,
        fallback:  torch.Tensor,
    ) -> torch.Tensor:
        """token subset에 대해서만 softmax attention 계산."""
        valid = [t for t in tok_idx if 0 <= t < T_seq]
        if not valid:
            return fallback
        idx   = torch.tensor(valid, device=q_mh.device, dtype=torch.long)
        k_sub = k_mh[:, :, idx, :]  # (B, H, n_tok, Dh)
        v_sub = v_mh[:, :, idx, :]
        scores = torch.matmul(q_mh, k_sub.transpose(-1, -2)) * scale  # (B,H,S,n_tok)
        w = scores.softmax(dim=-1)
        out = torch.matmul(w, v_sub)   # (B, H, S, Dh)
        return out.permute(0, 2, 1, 3).reshape(B, S, inner_dim)

    # ------------------------------------------------------------------
    def __call__(
        self,
        attn,
        hidden_states:           torch.Tensor,                   # (B, S, D)
        encoder_hidden_states:   Optional[torch.Tensor] = None,
        attention_mask           = None,
        temb                     = None,
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

        # ── K, V 계산 (shared) ──────────────────────────────────────────
        k    = attn.to_k(enc_hs)
        v    = attn.to_v(enc_hs)
        k_mh = _mh(k, T_seq)
        v_mh = _mh(v, T_seq)

        # ── Q 계산 ───────────────────────────────────────────────────────
        q    = attn.to_q(hidden_states)
        q_mh = _mh(q, S)

        # ── Global attention (F_global) ──────────────────────────────────
        scores_g = torch.matmul(q_mh, k_mh.transpose(-1, -2)) * scale
        w_g      = scores_g.softmax(dim=-1)
        F_g = (torch.matmul(w_g, v_mh)
               .permute(0, 2, 1, 3)
               .reshape(B, S, inner_dim))

        # ── Entity slot attention (F_0, F_1) ────────────────────────────
        F_0_raw = self._masked_attn(q_mh, k_mh, v_mh, self.toks_e0, T_seq,
                                    scale, B, S, n_heads, head_dim, inner_dim,
                                    fallback=F_g)
        F_1_raw = self._masked_attn(q_mh, k_mh, v_mh, self.toks_e1, T_seq,
                                    scale, B, S, n_heads, head_dim, inner_dim,
                                    fallback=F_g)

        # ── Phase 39: apply slot adapters ───────────────────────────────
        # adapters are float32; cast input and output back to dtype
        F_0 = self.slot0_adapter(F_0_raw.float()).to(dtype)
        F_1 = self.slot1_adapter(F_1_raw.float()).to(dtype)

        # ── VCA sigma 계산 → Porter-Duff 가중치 ─────────────────────────
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
                    # Use sigma_raw for Porter-Duff when training:
                    # last_sigma is detached → alpha_0/1/w0/w1 have NO gradient
                    # sigma_raw keeps grad → l_vis / l_sigma can train VCA
                    sigma = sigma_raw

        # ── Porter-Duff compositing weights ─────────────────────────────
        if sigma is not None and sigma.shape[:2] == (B, S):
            sig = sigma.to(device=F_g.device, dtype=torch.float32)

            alpha_0  = sig[:, :, 0, :].max(dim=-1).values   # (B, S)
            alpha_1  = sig[:, :, 1, :].max(dim=-1).values
            e0_front = torch.sigmoid(5.0 * (sig[:, :, 0, 0] - sig[:, :, 1, 0]))

            w0    = e0_front * alpha_0 + (1.0 - e0_front) * alpha_0 * (1.0 - alpha_1)
            w1    = (1.0 - e0_front) * alpha_1 + e0_front * alpha_1 * (1.0 - alpha_0)
            w_bg  = (1.0 - w0 - w1).clamp(min=0.0)

            w0_f  = w0.unsqueeze(-1).to(dtype=F_g.dtype)
            w1_f  = w1.unsqueeze(-1).to(dtype=F_g.dtype)
            wbg_f = w_bg.unsqueeze(-1).to(dtype=F_g.dtype)

            composed = w0_f * F_0 + w1_f * F_1 + wbg_f * F_g

            # ── Phase 39: per-pixel blend map ───────────────────────────
            # NOTE: no detach on inputs — blend_head needs gradient in stage2
            if self.use_blend_head:
                blend_map = self.blend_head(
                    alpha_0, alpha_1, e0_front
                )   # (B, S, 1) float32
                blend_map = blend_map.to(dtype=F_g.dtype)
            else:
                # fallback: scalar (Phase 38 compat)
                blend_map = self.slot_blend.to(dtype=F_g.dtype)

        else:
            # sigma 없으면 단순 평균
            composed = (F_0 + F_1 + F_g) / 3.0
            w0 = torch.ones(B, S, device=F_g.device) / 3
            w1 = torch.ones(B, S, device=F_g.device) / 3
            blend_map = self.slot_blend.to(dtype=F_g.dtype)

        # ── Blend: blend_map*composed + (1-blend_map)*F_global ──────────
        blended = blend_map * composed + (1.0 - blend_map) * F_g

        # ── Store for loss/metric computation ────────────────────────────
        # w0/w1/alpha0/alpha1은 training/eval 모두 저장 (validation teacher-forced eval에서 필요)
        # F0/F1/Fg는 training에서만 저장 (메모리 절약)
        self.last_w0 = w0
        self.last_w1 = w1
        if sigma is not None and sigma.shape[:2] == (B, S):
            self.last_alpha0 = alpha_0
            self.last_alpha1 = alpha_1
        self.last_blend = (blend_map.squeeze(-1)
                           if isinstance(blend_map, torch.Tensor) and blend_map.dim() == 3
                           else blend_map)
        if self.training:
            self.last_F0 = F_0
            self.last_F1 = F_1
            self.last_Fg = F_g

        # ── Output projection ────────────────────────────────────────────
        out = attn.to_out[0](blended)
        out = attn.to_out[1](out)
        return out.to(dtype)


# =============================================================================
# Helper: inject / restore
# =============================================================================

import copy


def inject_entity_slot(
    pipe,
    vca_layer,
    entity_ctx:       torch.Tensor,
    inject_key:       str,
    slot_blend_init:  float = 0.3,
    adapter_rank:     int   = 64,
    use_blend_head:   bool  = True,
):
    """
    EntitySlotAttnProcessor 를 지정 key에 주입.
    Returns (proc, orig_procs)
    """
    unet      = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))

    INJECT_INNER_DIM = 640   # up_blocks.2 inner_dim (AnimateDiff SD1.5)
    proc = EntitySlotAttnProcessor(
        query_dim      = INJECT_INNER_DIM,
        vca_layer      = vca_layer,
        entity_ctx     = entity_ctx,
        slot_blend_init = slot_blend_init,
        inner_dim      = INJECT_INNER_DIM,
        adapter_rank   = adapter_rank,
        use_blend_head = use_blend_head,
    )
    new_procs = dict(unet.attn_processors)
    new_procs[inject_key] = proc
    unet.set_attn_processor(new_procs)
    return proc, orig_procs


# =============================================================================
# Loss functions
# =============================================================================

def l_entity_exclusive(
    F_0:              torch.Tensor,          # (B, S, D)
    F_1:              torch.Tensor,
    F_global:         torch.Tensor,
    entity_masks_BNS: torch.Tensor,          # (B, 2, S) float
    eps:              float = 1e-6,
) -> torch.Tensor:
    """
    L_exclusive: entity i slot이 entity i 의 exclusive 픽셀에서 F_global을 재현.

    Target은 stop-gradient (F_global.detach()).
    """
    target = F_global.detach().float()
    F_0f   = F_0.float()
    F_1f   = F_1.float()
    m0     = entity_masks_BNS[:, 0, :]
    m1     = entity_masks_BNS[:, 1, :]

    excl_for_e0 = (m0 * (1.0 - m1)).unsqueeze(-1)
    excl_for_e1 = (m1 * (1.0 - m0)).unsqueeze(-1)

    D      = target.shape[-1]
    losses = []
    n0     = excl_for_e0.sum() + eps
    n1     = excl_for_e1.sum() + eps
    if n0 > 1:
        losses.append(((F_0f - target).pow(2) * excl_for_e0).sum() / (n0 * D))
    if n1 > 1:
        losses.append(((F_1f - target).pow(2) * excl_for_e1).sum() / (n1 * D))

    if not losses:
        return F_0.sum() * 0.0
    return torch.stack(losses).mean()


def l_overlap_ordering(
    w0:               torch.Tensor,          # (B, S)
    w1:               torch.Tensor,
    entity_masks_BNS: torch.Tensor,          # (B, 2, S) float
    depth_orders_B:   list,
    eps:              float = 1e-6,
) -> torch.Tensor:
    """
    L_overlap_ordering: 겹침 영역에서 front entity의 w > back entity의 w.
    Loss = mean over overlap pixels: relu(w_back - w_front + margin)^2
    """
    MARGIN = 0.1
    m0      = entity_masks_BNS[:, 0, :].float()
    m1      = entity_masks_BNS[:, 1, :].float()
    overlap = m0 * m1

    losses = []
    for b in range(min(w0.shape[0], len(depth_orders_B))):
        front = int(depth_orders_B[b][0])
        back  = int(depth_orders_B[b][1])

        w_front = w0[b] if front == 0 else w1[b]
        w_back  = w1[b] if front == 0 else w0[b]

        ov    = overlap[b]
        n_ov  = ov.sum() + eps
        if n_ov < 1:
            continue

        violation = F.relu(w_back - w_front + MARGIN)
        losses.append((violation.pow(2) * ov).sum() / n_ov)

    if not losses:
        return w0.sum() * 0.0
    return torch.stack(losses).mean()


# =============================================================================
# Phase 39: GT-mask based losses (핵심 추가)
# =============================================================================

def build_visible_targets(
    entity_masks_BNS: torch.Tensor,   # (B, 2, S) float
    depth_orders_B:   list,           # [(front_idx, back_idx)] len >= B
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GT visible weight targets for w0 and w1.

    Rules
    -----
    exclusive e0 (m0=1, m1=0): w0_target=1, w1_target=0
    exclusive e1 (m0=0, m1=1): w0_target=0, w1_target=1
    overlap    (m0=1, m1=1):   front entity target=1, back entity target=0
    background (m0=0, m1=0):   w0_target=0, w1_target=0

    Returns
    -------
    w0_target, w1_target: (B, S) float, values in {0, 1}
    """
    B  = entity_masks_BNS.shape[0]
    m0 = entity_masks_BNS[:, 0, :].float()
    m1 = entity_masks_BNS[:, 1, :].float()

    overlap = m0 * m1
    excl_0  = m0 * (1.0 - m1)
    excl_1  = m1 * (1.0 - m0)

    w0_target = excl_0.clone()   # e0 exclusive: w0=1
    w1_target = excl_1.clone()   # e1 exclusive: w1=1

    for b in range(min(B, len(depth_orders_B))):
        front = int(depth_orders_B[b][0])
        ov_b  = overlap[b]
        if front == 0:
            w0_target[b] = w0_target[b] + ov_b   # e0 front: w0=1 in overlap
        else:
            w1_target[b] = w1_target[b] + ov_b   # e1 front: w1=1 in overlap

    return w0_target, w1_target


def l_visible_weights(
    w0:               torch.Tensor,          # (B, S) predicted weight entity 0
    w1:               torch.Tensor,          # (B, S) predicted weight entity 1
    entity_masks_BNS: torch.Tensor,          # (B, 2, S) float
    depth_orders_B:   list,
    eps:              float = 1e-6,
) -> torch.Tensor:
    """
    L_visible_weights: GT visible target으로 w0, w1 직접 supervision.

    exclusive region: 자기 entity w_i→1, 다른 entity w_j→0
    overlap region:   front entity w_front→1, back entity w_back→0
    background:       w0,w1→0 (mask 없으면 loss 없음)

    Phase 39에서 가장 핵심적인 loss.
    """
    w0_target, w1_target = build_visible_targets(entity_masks_BNS, depth_orders_B)
    w0_target = w0_target.to(device=w0.device)
    w1_target = w1_target.to(device=w1.device)

    # entity mask: any entity present → non-background
    m_any = (entity_masks_BNS[:, 0, :] + entity_masks_BNS[:, 1, :]).clamp(max=1.0)
    n     = m_any.sum() + eps

    l0 = ((w0.float() - w0_target).pow(2) * m_any).sum() / n
    l1 = ((w1.float() - w1_target).pow(2) * m_any).sum() / n
    return (l0 + l1) * 0.5


def l_wrong_slot_suppression(
    w0:               torch.Tensor,          # (B, S)
    w1:               torch.Tensor,
    entity_masks_BNS: torch.Tensor,          # (B, 2, S) float
    eps:              float = 1e-6,
) -> torch.Tensor:
    """
    L_wrong_slot: exclusive 영역에서 wrong entity weight 패널티.

    entity0 exclusive에서 w1이 커지면 패널티.
    entity1 exclusive에서 w0이 커지면 패널티.

    L_exclusive의 보완 — appearance가 아닌 weight 관점에서 직접 제약.
    """
    m0     = entity_masks_BNS[:, 0, :].float()
    m1     = entity_masks_BNS[:, 1, :].float()
    excl_0 = m0 * (1.0 - m1)   # e0 exclusive: w1 should be 0
    excl_1 = m1 * (1.0 - m0)   # e1 exclusive: w0 should be 0

    n0 = excl_0.sum() + eps
    n1 = excl_1.sum() + eps

    l_wrong_e0 = (w1.float().pow(2) * excl_0).sum() / n0
    l_wrong_e1 = (w0.float().pow(2) * excl_1).sum() / n1
    return (l_wrong_e0 + l_wrong_e1) * 0.5


def l_sigma_spatial(
    alpha0:           torch.Tensor,          # (B, S) Porter-Duff alpha entity 0
    alpha1:           torch.Tensor,          # (B, S) Porter-Duff alpha entity 1
    entity_masks_BNS: torch.Tensor,          # (B, 2, S) float
) -> torch.Tensor:
    """
    L_sigma_spatial: VCA sigma를 공간적으로 entity mask와 일치시킴.

    핵심 동기
    ----------
    VCA는 Phase 31~38에서 depth ordering (front/back) 만 학습했음 → DRA=0.975 달성.
    하지만 alpha_0/alpha_1 (sigma의 max over z_bins)는 공간적으로 uniform →
    Porter-Duff w0/w1이 entity mask와 무관하게 됨 → visible_iou가 0.082에서 고착.

    Fix: alpha_0 → entity_mask_0, alpha_1 → entity_mask_1 직접 MSE supervision.

    주의: depth loss (l_depth)와 함께 쓰면 spatial + ordering 동시 학습.
    """
    m0 = entity_masks_BNS[:, 0, :].float().to(alpha0.device)
    m1 = entity_masks_BNS[:, 1, :].float().to(alpha1.device)
    l0 = F.mse_loss(alpha0.float(), m0)
    l1 = F.mse_loss(alpha1.float(), m1)
    return (l0 + l1) * 0.5


# =============================================================================
# Validation metrics (GT mask 기반 — RGB threshold 아님)
# =============================================================================

def compute_visible_iou(
    w0:               torch.Tensor,          # (B, S)
    w1:               torch.Tensor,
    entity_masks_BNS: torch.Tensor,          # (B, 2, S) float
    depth_orders_B:   list,
    eps:              float = 1e-6,
) -> float:
    """
    Soft IoU between predicted visible weights and GT visible targets.
    높을수록 좋음 (0~1).
    """
    w0_target, w1_target = build_visible_targets(entity_masks_BNS, depth_orders_B)
    w0t = w0_target.to(w0.device).float()
    w1t = w1_target.to(w1.device).float()

    pred = (w0.float() + w1.float()).clamp(0.0, 1.0)
    gt   = (w0t + w1t).clamp(0.0, 1.0)

    inter = (pred * gt).sum()
    union = (pred + gt - pred * gt).sum()
    return float((inter / (union + eps)).item())


def compute_ordering_accuracy(
    w0:               torch.Tensor,
    w1:               torch.Tensor,
    entity_masks_BNS: torch.Tensor,
    depth_orders_B:   list,
    eps:              float = 1e-6,
) -> float:
    """
    overlap 영역에서 front entity weight > back entity weight 비율.
    높을수록 좋음 (0~1).
    """
    m0      = entity_masks_BNS[:, 0, :].float()
    m1      = entity_masks_BNS[:, 1, :].float()
    overlap = m0 * m1   # (B, S)

    total_correct  = 0.0
    total_overlap  = 0.0

    for b in range(min(w0.shape[0], len(depth_orders_B))):
        front   = int(depth_orders_B[b][0])
        w_front = w0[b].float() if front == 0 else w1[b].float()
        w_back  = w1[b].float() if front == 0 else w0[b].float()
        ov      = overlap[b]

        total_overlap  += float(ov.sum().item())
        total_correct  += float(((w_front > w_back) * ov).sum().item())

    return float(total_correct / (total_overlap + eps))


def compute_wrong_slot_leak(
    w0:               torch.Tensor,
    w1:               torch.Tensor,
    entity_masks_BNS: torch.Tensor,
    eps:              float = 1e-6,
) -> float:
    """
    exclusive 영역에서 wrong entity weight 평균.
    낮을수록 좋음 (0~1).
    """
    m0     = entity_masks_BNS[:, 0, :].float()
    m1     = entity_masks_BNS[:, 1, :].float()
    excl_0 = m0 * (1.0 - m1)
    excl_1 = m1 * (1.0 - m0)

    n0   = float(excl_0.sum().item()) + eps
    n1   = float(excl_1.sum().item()) + eps
    leak_e0 = float((w1.float() * excl_0).sum().item()) / n0
    leak_e1 = float((w0.float() * excl_1).sum().item()) / n1
    return (leak_e0 + leak_e1) * 0.5


def val_slot_score(
    visible_iou:       float,
    ordering_acc:      float,
    wrong_slot_leak:   float,
    dra:               float,
) -> float:
    """
    GT-mask 기반 validation score.
    best checkpoint selection 기준.
    val_score = 0.4*visible_iou + 0.3*ordering_acc + 0.2*(1-wrong_slot_leak) + 0.1*dra
    """
    return (0.4 * visible_iou
          + 0.3 * ordering_acc
          + 0.2 * (1.0 - wrong_slot_leak)
          + 0.1 * dra)


# =============================================================================
# Overlap score (for train/val split)
# =============================================================================

def compute_overlap_score(entity_masks: np.ndarray) -> float:
    """
    단일 시퀀스의 overlap score 계산.

    Parameters
    ----------
    entity_masks : (T, 2, S) float array
    Returns
    -------
    float ∈ [0, 1] — 높을수록 두 entity가 많이 겹치는 시퀀스
    """
    m0      = (entity_masks[:, 0, :] > 0.5).astype(np.float32)  # (T, S)
    m1      = (entity_masks[:, 1, :] > 0.5).astype(np.float32)
    overlap = (m0 * m1).sum(axis=-1)                              # (T,)
    union   = ((m0 + m1) > 0).astype(np.float32).sum(axis=-1)    # (T,)
    per_frame = overlap / (union + 1e-6)
    return float(per_frame.mean())


# =============================================================================
# Metrics — DEBUG ONLY (RGB threshold 기반; checkpoint selection에 쓰지 말 것)
# =============================================================================

def entity_survival_rate(
    frames_rgb: np.ndarray,
    min_pixels: int = 100,
    red_thresh: int = 100,
    blue_thresh: int = 100,
) -> float:
    """
    [DEBUG ONLY] RGB threshold 기반 entity survival.
    배경색도 entity로 잡으며 checkpoint selection에 부적합.
    """
    survive = 0
    T = len(frames_rgb)
    for frame in frames_rgb:
        r = frame[:, :, 0].astype(np.int32)
        g = frame[:, :, 1].astype(np.int32)
        b = frame[:, :, 2].astype(np.int32)
        is_red  = (r > red_thresh)   & (g < 140) & (b < 140)
        is_blue = (b > blue_thresh)  & (r < 140) & (g < 140)
        if is_red.sum() >= min_pixels and is_blue.sum() >= min_pixels:
            survive += 1
    return float(survive) / float(max(T, 1))


def chimera_rate(
    frames_rgb: np.ndarray,
    eps:        float = 1e-6,
) -> float:
    """
    [DEBUG ONLY] RGB threshold 기반 chimera score.
    배경색도 entity로 잡으며 entity 소멸을 chimera 감소로 오인할 수 있음.
    """
    scores = []
    for frame in frames_rgb:
        r = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)
        b = frame[:, :, 2].astype(np.float32)
        is_red  = (r > 80) & (g < 120) & (b < 120)
        is_blue = (b > 80) & (r < 120) & (g < 120)
        both   = float((is_red & is_blue).sum())
        either = float((is_red | is_blue).sum())
        scores.append(both / (either + eps))
    return float(np.mean(scores))


def entity_score(
    frames_rgb: np.ndarray,
    min_pixels: int = 100,
) -> Tuple[float, float, float]:
    """
    [DEBUG ONLY] entity_score = survival × (1 - chimera).
    RGB threshold 기반이므로 정량 비교에 사용하지 말 것.
    Returns (entity_score, survival_rate, chimera_rate)
    """
    sr = entity_survival_rate(frames_rgb, min_pixels=min_pixels)
    cr = chimera_rate(frames_rgb)
    es = sr * (1.0 - cr)
    return es, sr, cr
