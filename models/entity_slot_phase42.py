"""
Phase 42 — Multi-Block Restore + Residual Weight Head
=======================================================

Phase 41 사후 분석:
  alpha0_entity=0.94 (이미 좋음)인데도 iou_e0/e1이 0.09에서 60 epoch 동안 고착.
  → alpha→w 변환 (Porter-Duff heuristic)이 병목.

Phase 42 핵심 수정 (3가지):
─────────────────────────────
A. Checkpoint 완전 복원
   phase40/41은 primary block만 복원하고 나머지 2개 block을 랜덤 초기화로 방치.
   restore_multiblock_state()로 모든 block state 복원. 복원 실패 시 즉시 중단.

B. alpha→w 변환: heuristic → residual-corrected WeightHead
   Porter-Duff 결과를 base로 두고 WeightHead가 delta logit만 보정.
   zero-init → 학습 초기엔 Porter-Duff와 동일, 점진적으로 보정 학습.

C. val_score에서 id_margin 제거
   val_score_phase42 = 0.3*iou_e0 + 0.3*iou_e1 + 0.2*ord + 0.2*(1-wrong)
   id_margin, rollout_id는 로깅만.

추가:
D. l_slot_ref + l_slot_contrast: 모든 block에 appearance supervision
   primary block F0/F1을 reference로 삼아 나머지 block이 일관된 entity feature 생성.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.entity_slot_phase40 import (
    Phase40Processor,
    MultiBlockSlotManager,
    BLOCK_INNER_DIMS,
    CROSS_ATTN_DIM,
    DEFAULT_INJECT_KEYS,
    compute_visible_iou_e0,
    compute_visible_iou_e1,
)
import copy


# =============================================================================
# WeightHead: residual correction for Porter-Duff weights
# =============================================================================

class WeightHead(nn.Module):
    """
    Porter-Duff base weight에 delta logit을 더해 w_bg/w0/w1을 보정.

    Architecture
    ------------
    Linear(8, 32) → SiLU → Linear(32, 3)   [zero-init last layer]

    입력 features (B, S, 8):
      alpha0, alpha1,
      sig[:,:,0,0], sig[:,:,0,1],   # entity 0 z-bin 0/1
      sig[:,:,1,0], sig[:,:,1,1],   # entity 1 z-bin 0/1
      alpha0 * alpha1,
      e0_front

    zero-init 보장:
      초기 delta=0 → base_logits + 0 → 정확히 Porter-Duff와 동일 w 출력.
    """

    def __init__(self, in_features: int = 8, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        # zero-init last layer → delta=0 at start
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        feat : (B, S, 8) float32

        Returns
        -------
        delta : (B, S, 3) float32  — logit correction [w_bg, w0, w1]
        """
        return self.net(feat)


# =============================================================================
# Phase42Processor: Phase40 + WeightHead
# =============================================================================

class Phase42Processor(Phase40Processor):
    """
    Phase40Processor + WeightHead residual correction on w0/w1.

    주요 변경점:
    - WeightHead가 Porter-Duff base weights를 보정
    - last_w_delta 저장 (l_w_res regularizer용)
    - 나머지는 Phase40Processor와 동일
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
        )
        self.weight_head = WeightHead(in_features=8, hidden=weight_head_hidden)
        self.last_w_delta: Optional[torch.Tensor] = None   # (B, S, 3) for l_w_res

    def reset_slot_store(self):
        super().reset_slot_store()
        self.last_w_delta = None

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
        Phase40 로직과 동일하나, Porter-Duff 계산 후 WeightHead residual 보정 추가.
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
                    sigma = sigma_raw

        # ── Porter-Duff base weights ──────────────────────────────────────
        if sigma is not None and sigma.shape[:2] == (B, S):
            sig = sigma.to(device=F_g.device, dtype=torch.float32)

            alpha_0  = sig[:, :, 0, :].max(dim=-1).values  # (B, S)
            alpha_1  = sig[:, :, 1, :].max(dim=-1).values
            e0_front = torch.sigmoid(5.0 * (sig[:, :, 0, 0] - sig[:, :, 1, 0]))

            base_w0  = e0_front * alpha_0 + (1.0 - e0_front) * alpha_0 * (1.0 - alpha_1)
            base_w1  = (1.0 - e0_front) * alpha_1 + e0_front * alpha_1 * (1.0 - alpha_0)
            base_wbg = (1.0 - base_w0 - base_w1).clamp(min=0.0)

            # ── WeightHead residual correction ───────────────────────────
            # z-bins: sig shape (B, S, N_entities, Z_bins)
            z_bins = sig.shape[-1]
            s00 = sig[:, :, 0, 0]
            s01 = sig[:, :, 0, min(1, z_bins-1)]
            s10 = sig[:, :, 1, 0]
            s11 = sig[:, :, 1, min(1, z_bins-1)]

            feat = torch.stack([
                alpha_0, alpha_1,
                s00, s01,
                s10, s11,
                alpha_0 * alpha_1,
                e0_front,
            ], dim=-1).float()   # (B, S, 8)

            delta  = self.weight_head(feat)   # (B, S, 3)

            base_logits = torch.log(
                torch.stack([base_wbg, base_w0, base_w1], dim=-1).clamp(min=1e-6))
            logits = base_logits + delta
            probs  = logits.softmax(dim=-1)   # (B, S, 3)
            w_bg_c, w0_c, w1_c = probs.unbind(dim=-1)

            # Store delta for l_w_res
            self.last_w_delta = delta

            w0   = w0_c
            w1   = w1_c
            w_bg = w_bg_c

            w0_f  = w0.unsqueeze(-1).to(dtype=F_g.dtype)
            w1_f  = w1.unsqueeze(-1).to(dtype=F_g.dtype)
            wbg_f = w_bg.unsqueeze(-1).to(dtype=F_g.dtype)

            composed = w0_f * F_0 + w1_f * F_1 + wbg_f * F_g

            if self.use_blend_head:
                blend_map = self.blend_head(alpha_0, alpha_1, e0_front)
                blend_map = blend_map.to(dtype=F_g.dtype)
            else:
                blend_map = self.slot_blend.to(dtype=F_g.dtype)
        else:
            composed  = (F_0 + F_1 + F_g) / 3.0
            w0        = torch.ones(B, S, device=F_g.device) / 3
            w1        = torch.ones(B, S, device=F_g.device) / 3
            w_bg      = torch.ones(B, S, device=F_g.device) / 3
            blend_map = self.slot_blend.to(dtype=F_g.dtype)
            alpha_0   = w0
            alpha_1   = w1
            self.last_w_delta = None

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
# Multi-block injection for Phase 42
# =============================================================================

def inject_multi_block_entity_slot_p42(
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
) -> Tuple[List[Phase42Processor], Dict]:
    """
    Phase42Processor를 inject_keys 위치에 주입.
    VCA는 primary block(primary_idx)에만 주입.
    """
    if inject_keys is None:
        inject_keys = DEFAULT_INJECT_KEYS

    unet       = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))
    new_procs  = dict(unet.attn_processors)
    procs: List[Phase42Processor] = []

    for i, key in enumerate(inject_keys):
        inner_dim  = BLOCK_INNER_DIMS.get(key, 640)
        block_vca  = vca_layer if i == primary_idx else None
        block_ctx  = entity_ctx if i == primary_idx else None
        proc = Phase42Processor(
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
        )
        new_procs[key] = proc
        procs.append(proc)

    unet.set_attn_processor(new_procs)
    return procs, orig_procs


# =============================================================================
# Checkpoint restoration — ALL blocks
# =============================================================================

def restore_multiblock_state(
    manager,          # MultiBlockSlotManagerP42 or MultiBlockSlotManager
    ckpt:   dict,
    device: str = "cpu",
    strict: bool = True,
) -> None:
    """
    Phase40/41 checkpoint에서 모든 block state 복원.

    Phase41 버그 수정:
      phase41는 primary block만 복원하고 나머지 2개 block을 랜덤 초기화로 방치함.
      이 함수는 procs_state의 모든 block을 복원하며, 실패 시 즉시 RuntimeError 발생.

    Notes
    -----
    weight_head는 Phase42 신규 모듈이므로 ckpt에 없음 → zero-init 유지 (복원 불필요).
    """
    proc_states = ckpt.get("procs_state", None)
    if proc_states is None:
        raise RuntimeError(
            "ckpt에 'procs_state' 없음. phase40+ checkpoint이 필요합니다.")

    if len(proc_states) != len(manager.procs):
        raise RuntimeError(
            f"procs_state 개수 불일치: ckpt={len(proc_states)}, "
            f"manager.procs={len(manager.procs)}")

    for i, (proc, state) in enumerate(zip(manager.procs, proc_states)):
        dev = proc.slot_blend_raw.device

        # slot_blend_raw
        sbr = state["slot_blend_raw"]
        if hasattr(sbr, 'to'):
            sbr = sbr.to(dev)
        proc.slot_blend_raw.data.copy_(sbr)

        # sub-modules
        for mod_name in ("slot0_adapter", "slot1_adapter", "blend_head",
                         "lora_k", "lora_v", "lora_out"):
            if mod_name not in state:
                raise RuntimeError(
                    f"block[{i}] procs_state에 '{mod_name}' 없음. "
                    f"Available keys: {list(state.keys())}")
            mod = getattr(proc, mod_name)
            sd  = state[mod_name]
            # move state_dict tensors to device
            sd_dev = {k: v.to(dev) if hasattr(v, 'to') else v
                      for k, v in sd.items()}
            mod.load_state_dict(sd_dev, strict=strict)

        print(f"  [restore] block[{i}] restored OK "
              f"(blend={float(proc.slot_blend.item()):.4f})", flush=True)


# =============================================================================
# MultiBlockSlotManagerP42
# =============================================================================

class MultiBlockSlotManagerP42(MultiBlockSlotManager):
    """
    Phase42Processor용 manager. weight_head 파라미터 접근 추가.
    """

    def __init__(self, procs: List[Phase42Processor], keys: List[str],
                 primary_idx: int = 1):
        super().__init__(procs, keys, primary_idx)

    def weight_head_params(self) -> List[torch.nn.Parameter]:
        params = []
        for p in self.procs:
            if hasattr(p, 'weight_head'):
                params += list(p.weight_head.parameters())
        return params

    @property
    def last_w_delta(self):
        return self.primary.last_w_delta


# =============================================================================
# Phase 42 loss functions
# =============================================================================

def l_w_residual(
    w_delta: torch.Tensor,    # (B, S, 3) from WeightHead
    weight:  float = 1.0,
) -> torch.Tensor:
    """
    L_w_res: WeightHead delta regularizer.
    초반에 Porter-Duff에서 너무 멀리 떠나지 않도록 제약.
    L = mean(delta^2)
    """
    return (w_delta.float().pow(2)).mean() * weight


def l_slot_ref(
    F_slot:          torch.Tensor,   # (B, S, D) — block i slot feature
    F_ref:           torch.Tensor,   # (B, S, D) — primary block feature (detach)
    visible_mask_BS: torch.Tensor,   # (B, S) float — 1 where entity is visible
    eps:             float = 1e-6,
) -> torch.Tensor:
    """
    L_slot_ref: visible entity region에서 block_i의 F_slot이 primary block의 F_ref를 따라가게.

    Cross-block consistency: secondary blocks가 primary와 같은 entity features 생성.
    F_ref는 stop-gradient (primary block features.detach()).
    """
    if visible_mask_BS.sum() < 1:
        return F_slot.sum() * 0.0

    m = visible_mask_BS.float().unsqueeze(-1).to(F_slot.device)
    n = m.sum() * F_slot.shape[-1] + eps

    diff = (F_slot.float() - F_ref.float().detach()).pow(2)
    return (diff * m).sum() / n


def l_slot_contrast(
    F0_slot:  torch.Tensor,   # (B, S, D)
    F1_slot:  torch.Tensor,
    F0_ref:   torch.Tensor,   # (B, S, D) — primary F0 (detach)
    F1_ref:   torch.Tensor,   # (B, S, D) — primary F1 (detach)
    mask_e0:  torch.Tensor,   # (B, S)
    mask_e1:  torch.Tensor,
    margin:   float = 0.1,
    eps:      float = 1e-6,
) -> torch.Tensor:
    """
    L_slot_contrast: cosine margin loss across all blocks.

    entity0 region: cos(F0_slot, F0_ref) > cos(F0_slot, F1_ref) + margin
    entity1 region: cos(F1_slot, F1_ref) > cos(F1_slot, F0_ref) + margin

    F_ref는 stop-gradient (primary block features.detach()).
    """
    F0_s = F.normalize(F0_slot.float(), dim=-1)
    F1_s = F.normalize(F1_slot.float(), dim=-1)
    F0_r = F.normalize(F0_ref.float().detach(), dim=-1)
    F1_r = F.normalize(F1_ref.float().detach(), dim=-1)

    m0 = mask_e0.float().unsqueeze(-1).to(F0_s.device)
    m1 = mask_e1.float().unsqueeze(-1).to(F1_s.device)

    n0 = m0.sum() + eps
    n1 = m1.sum() + eps

    sim_00 = (F0_s * F0_r).sum(-1, keepdim=True)
    sim_01 = (F0_s * F1_r).sum(-1, keepdim=True)
    l0     = (F.relu(sim_01 - sim_00 + margin) * m0).sum() / n0

    sim_11 = (F1_s * F1_r).sum(-1, keepdim=True)
    sim_10 = (F1_s * F0_r).sum(-1, keepdim=True)
    l1     = (F.relu(sim_10 - sim_11 + margin) * m1).sum() / n1

    return (l0 + l1) * 0.5


# =============================================================================
# Phase 42 validation score
# =============================================================================

def val_score_phase42(
    iou_e0:      float,
    iou_e1:      float,
    ordering_acc: float,
    wrong_leak:  float,
) -> float:
    """
    Phase 42 validation score.
    id_margin 제거 — proxy가 아닌 실제 IoU/ordering 기반.

    weights:
      iou_e0        0.30
      iou_e1        0.30
      ordering_acc  0.20
      (1-wrong)     0.20
    """
    return (0.30 * iou_e0
          + 0.30 * iou_e1
          + 0.20 * ordering_acc
          + 0.20 * (1.0 - wrong_leak))
