"""
Phase 43 — Projected Slot Consistency + Rollout-Centric Selection
==================================================================

Phase 42 대비 핵심 변경:

1. FeatureProjector (640 → 1280/640/320)
   Phase42의 slot_ref=0.0000 근본 원인: dim mismatch skip.
   각 Phase43Processor에 ref_proj_e0/e1 내장 → secondary block도 ref loss 받음.

2. Entity-specific text forward as true reference
   entity0-only prompt → enc_hs_e0 → UNet forward (no_grad) → F_g per block
   → K/V가 entity0 text 기반으로 바뀌어 F_g = near-solo entity0 appearance
   이게 진짜 reference. primary F0를 projected한 것보다 훨씬 의미 있음.

3. Soft visible targets (build_visible_targets_soft)
   overlap: front=0.85, back=0.05  (hard binary 1/0 대신)
   W가 continuous compositing weight인데 target이 binary이면 optimization mismatch.

4. blend_map 진단 (collect_blend_stats + l_blend_overlap)
   overlap region에서 blend_map이 non-overlap보다 높아야 함.

5. val_score_phase43: rollout_iou 포함 (15% × 2)
   teacher-forced만 보는 phase42 선택 기준에서 벗어남.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.entity_slot_phase42 import (
    Phase42Processor,
    MultiBlockSlotManagerP42,
    BLOCK_INNER_DIMS,
    CROSS_ATTN_DIM,
    DEFAULT_INJECT_KEYS,
    WeightHead,
)
from models.entity_slot_phase42 import (
    l_slot_ref,
    l_slot_contrast,
)
from models.entity_slot import (
    build_visible_targets,
)
import copy


# Primary block dim (up_blocks.2)
PRIMARY_DIM = 640


# =============================================================================
# FeatureProjector: ref_proj 640 → block_dim
# =============================================================================

class FeatureProjector(nn.Module):
    """
    Primary block feature를 secondary block dim으로 project.

    in_dim = PRIMARY_DIM = 640 (항상)
    out_dim = secondary block inner_dim (1280 또는 320)

    zero-init last layer:
      초기에 projected ref = 0 → l_slot_ref gradient = 0
      학습이 시작되면서 projector가 점진적으로 유의미한 ref를 생성.

    in_dim == out_dim이면 residual (identity at start).
    in_dim != out_dim이면 pure projection.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.residual = (in_dim == out_dim)

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_dim) → (..., out_dim)
        residual if in_dim == out_dim (identity at start due to zero-init).
        """
        out = self.net(x.float())
        if self.residual:
            return (x.float() + out).to(x.dtype)
        return out.to(x.dtype)


# =============================================================================
# Phase43Processor: Phase42 + FeatureProjectors
# =============================================================================

class Phase43Processor(Phase42Processor):
    """
    Phase42Processor + per-entity FeatureProjectors.

    ref_proj_e0: PRIMARY_DIM → this block's inner_dim
    ref_proj_e1: PRIMARY_DIM → this block's inner_dim

    초기: projectors output 0 → projected ref = 0 → no gradient from proj-ref path.
    학습 진행: projectors가 primary ref을 secondary dim으로 매핑하는 법 학습.

    entity-specific text forward로 얻은 F_g refs (native dim)는 projector 없이
    직접 l_slot_ref/contrast에 쓸 수 있음 (같은 dim이므로).
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
        )
        blk_dim = self._inner_dim
        self.ref_proj_e0 = FeatureProjector(primary_dim, blk_dim, hidden=proj_hidden)
        self.ref_proj_e1 = FeatureProjector(primary_dim, blk_dim, hidden=proj_hidden)

    def projector_params(self) -> List[torch.nn.Parameter]:
        return (list(self.ref_proj_e0.parameters())
                + list(self.ref_proj_e1.parameters()))


# =============================================================================
# Multi-block injection for Phase 43
# =============================================================================

def inject_multi_block_entity_slot_p43(
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
) -> Tuple[List[Phase43Processor], Dict]:
    if inject_keys is None:
        inject_keys = DEFAULT_INJECT_KEYS

    unet       = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))
    new_procs  = dict(unet.attn_processors)
    procs: List[Phase43Processor] = []

    for i, key in enumerate(inject_keys):
        inner_dim  = BLOCK_INNER_DIMS.get(key, 640)
        block_vca  = vca_layer if i == primary_idx else None
        block_ctx  = entity_ctx if i == primary_idx else None
        proc = Phase43Processor(
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
        )
        new_procs[key] = proc
        procs.append(proc)

    unet.set_attn_processor(new_procs)
    return procs, orig_procs


# =============================================================================
# MultiBlockSlotManagerP43
# =============================================================================

class MultiBlockSlotManagerP43(MultiBlockSlotManagerP42):
    """Phase43Processor용 manager. projector params 접근 추가."""

    def __init__(self, procs: List[Phase43Processor], keys: List[str],
                 primary_idx: int = 1):
        super().__init__(procs, keys, primary_idx)

    def projector_params(self) -> List[torch.nn.Parameter]:
        params = []
        for p in self.procs:
            if hasattr(p, 'projector_params'):
                params += p.projector_params()
        return params


# =============================================================================
# Checkpoint restoration for Phase 43
# =============================================================================

def restore_multiblock_state_p43(
    manager,
    ckpt:   dict,
    device: str = "cpu",
) -> None:
    """
    Phase40/41/42 checkpoint에서 모든 block state 복원.
    weight_head와 ref_proj는 Phase43 신규이므로 ckpt에 없으면 zero-init 유지.
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

        # slot_blend_raw
        sbr = state["slot_blend_raw"]
        proc.slot_blend_raw.data.copy_(
            sbr.to(dev) if hasattr(sbr, 'to') else torch.tensor(sbr).to(dev))

        # Phase40/41/42 공통 sub-modules
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
            sd = {k: v.to(dev) if hasattr(v, 'to') else v
                  for k, v in state["weight_head"].items()}
            proc.weight_head.load_state_dict(sd, strict=False)
            print(f"  [restore] block[{i}] weight_head loaded", flush=True)

        # ref_proj (Phase43+) — zero-init if not in ckpt
        if "ref_proj_e0" in state and hasattr(proc, 'ref_proj_e0'):
            sd = {k: v.to(dev) if hasattr(v, 'to') else v
                  for k, v in state["ref_proj_e0"].items()}
            proc.ref_proj_e0.load_state_dict(sd, strict=False)
        if "ref_proj_e1" in state and hasattr(proc, 'ref_proj_e1'):
            sd = {k: v.to(dev) if hasattr(v, 'to') else v
                  for k, v in state["ref_proj_e1"].items()}
            proc.ref_proj_e1.load_state_dict(sd, strict=False)

        print(f"  [restore] block[{i}] OK  "
              f"(blend={float(proc.slot_blend.item()):.4f})", flush=True)


# =============================================================================
# Soft visible targets
# =============================================================================

def build_visible_targets_soft(
    entity_masks_BNS: torch.Tensor,
    depth_orders_B:   list,
    front_val:        float = 0.85,
    back_val:         float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft visible targets: overlap region에 soft front/back assignment.

    Phase42까지의 hard targets (front=1, back=0)는 continuous w와
    optimization mismatch가 큼. 이 함수는 이걸 완화.

    exclusive / background: 기존과 동일 (1.0 / 0.0).
    overlap front: front_val (default 0.85)
    overlap back:  back_val  (default 0.05)
    """
    B  = entity_masks_BNS.shape[0]
    m0 = entity_masks_BNS[:, 0, :].float()
    m1 = entity_masks_BNS[:, 1, :].float()

    overlap = m0 * m1
    excl_0  = m0 * (1.0 - m1)
    excl_1  = m1 * (1.0 - m0)

    w0_target = excl_0.clone()
    w1_target = excl_1.clone()

    for b in range(min(B, len(depth_orders_B))):
        front = int(depth_orders_B[b][0])
        ov_b  = overlap[b]
        if front == 0:
            w0_target[b] = w0_target[b] + front_val * ov_b
            w1_target[b] = w1_target[b] + back_val  * ov_b
        else:
            w1_target[b] = w1_target[b] + front_val * ov_b
            w0_target[b] = w0_target[b] + back_val  * ov_b

    return w0_target, w1_target


def l_visible_weights_soft(
    w0:               torch.Tensor,
    w1:               torch.Tensor,
    entity_masks_BNS: torch.Tensor,
    depth_orders_B:   list,
    front_val:        float = 0.85,
    back_val:         float = 0.05,
    bg_weight:        float = 0.2,
    eps:              float = 1e-6,
) -> torch.Tensor:
    """
    L_vis with soft overlap targets.
    background suppression은 phase41/42와 동일하게 유지.
    """
    w0_target, w1_target = build_visible_targets_soft(
        entity_masks_BNS, depth_orders_B, front_val, back_val)
    w0_target = w0_target.to(device=w0.device)
    w1_target = w1_target.to(device=w1.device)

    m_any = (entity_masks_BNS[:, 0, :] + entity_masks_BNS[:, 1, :]).clamp(max=1.0)
    n_ent = m_any.sum() + eps

    l0 = ((w0.float() - w0_target).pow(2) * m_any).sum() / n_ent
    l1 = ((w1.float() - w1_target).pow(2) * m_any).sum() / n_ent

    bg    = 1.0 - m_any
    n_bg  = bg.sum() + eps
    l_bg0 = (w0.float().pow(2) * bg).sum() / n_bg
    l_bg1 = (w1.float().pow(2) * bg).sum() / n_bg

    return (l0 + l1) * 0.5 + bg_weight * (l_bg0 + l_bg1) * 0.5


# =============================================================================
# Blend map supervision
# =============================================================================

def l_blend_overlap(
    blend_map_BS:     torch.Tensor,   # (B, S) float
    entity_masks_BNS: torch.Tensor,   # (B, 2, S) float
    margin:           float = 0.05,
    eps:              float = 1e-6,
) -> torch.Tensor:
    """
    L_blend_overlap: overlap 영역에서 blend_map이 non-overlap보다 높게.

    compositing branch가 실제로 collision 영역에서 더 쓰이는지 확인.
    loss = relu(margin - (blend_overlap_mean - blend_nonoverlap_mean))
    """
    m0      = entity_masks_BNS[:, 0, :].float().to(blend_map_BS.device)
    m1      = entity_masks_BNS[:, 1, :].float().to(blend_map_BS.device)
    overlap = m0 * m1
    nonoverlap = (m0 + m1).clamp(0.0, 1.0) * (1.0 - overlap)

    n_ov  = overlap.sum() + eps
    n_non = nonoverlap.sum() + eps

    if n_ov < 1 or n_non < 1:
        return blend_map_BS.sum() * 0.0

    blend = blend_map_BS.float()
    blend_ov  = (blend * overlap).sum()  / n_ov
    blend_non = (blend * nonoverlap).sum() / n_non

    return F.relu(margin - (blend_ov - blend_non))


def collect_blend_stats(
    blend_map_BS:     torch.Tensor,   # (B, S) float
    entity_masks_BNS: torch.Tensor,   # (B, 2, S) float
    eps:              float = 1e-6,
) -> dict:
    """
    매 eval마다 저장할 blend map 통계.

    성공 신호: blend_separation = blend_overlap_mean - blend_nonoverlap_mean > 0.05
    """
    m0      = entity_masks_BNS[:, 0, :].float().to(blend_map_BS.device)
    m1      = entity_masks_BNS[:, 1, :].float().to(blend_map_BS.device)
    overlap    = m0 * m1
    nonoverlap = (m0 + m1).clamp(0.0, 1.0) * (1.0 - overlap)

    blend   = blend_map_BS.float()
    n_ov    = overlap.sum().item()
    n_non   = nonoverlap.sum().item()

    b_ov  = float((blend * overlap).sum().item()    / (n_ov  + eps))
    b_non = float((blend * nonoverlap).sum().item() / (n_non + eps))

    return {
        "blend_mean":           float(blend.mean().item()),
        "blend_std":            float(blend.std().item()),
        "blend_overlap_mean":   b_ov,
        "blend_nonoverlap_mean": b_non,
        "blend_separation":     b_ov - b_non,
    }


# =============================================================================
# Phase 43 validation score
# =============================================================================

def val_score_phase43(
    tf_iou_e0:       float,
    tf_iou_e1:       float,
    tf_ord:          float,
    tf_wrong:        float,
    rollout_iou_e0:  float = 0.0,
    rollout_iou_e1:  float = 0.0,
) -> float:
    """
    Phase 43 validation score: rollout을 best selection 주축으로.

    weights:
      tf_iou_e0        0.20  teacher-forced IoU
      tf_iou_e1        0.20
      tf_ord           0.15  depth ordering accuracy
      (1-tf_wrong)     0.15
      rollout_iou_e0   0.15  free generation IoU
      rollout_iou_e1   0.15
    """
    return (0.20 * tf_iou_e0
          + 0.20 * tf_iou_e1
          + 0.15 * tf_ord
          + 0.15 * (1.0 - tf_wrong)
          + 0.15 * rollout_iou_e0
          + 0.15 * rollout_iou_e1)
