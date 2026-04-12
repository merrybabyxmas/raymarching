"""
Phase 46 — Contrastive Occupancy + Direct Blend Ordering
=========================================================

Phase 45 실패 원인 (20 epochs 전체, 모든 run):
  1. occ_prod_exclusive > occ_prod_overlap (INVERTED 20 epochs 내내)
     → l_occupancy_structure의 l_ex_gap는 방향성 없음
  2. blend_sep < 0 항상 (exclusive blend > overlap blend)
     → base_blend_v2가 exclusive에서 inflation (o0=0.92, o1=0.69 → 0.66)
  3. l_blend_target_balanced은 ordering 강제 불가
     → 모든 region blend를 같이 올려서 MSE 최소화 가능

Phase 46 변경:
  A. l_occ_contrastive (margin=0.5) — directional exclusive suppression
     e0-exclusive에서 relu(o1 - o0 + margin) → o0 must dominate o1 by margin
  B. l_blend_ordering (margin=0.1) — direct hinge on region-mean blend
     mean_blend_ov > mean_blend_ex + margin (직접 blend_sep > 0 강제)
  C. compute_base_blend_v3 — soft-thresholded product (Phase46Processor에 내장)
     exclusive collapse시 base_blend 26% 감소 (0.66 → 0.49)
  D. val_score_phase46 — has_rollout=True/False로 공정한 점수
  E. Stage B: occ_head + OBH lr/5 유지 (이미 phase45 collisionfocus에 적용)
"""
import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import imageio.v2 as iio2
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.entity_slot import (
    l_visible_weights,
    l_wrong_slot_suppression,
    l_sigma_spatial,
    l_entity_exclusive,
    l_overlap_ordering,
    compute_ordering_accuracy,
    compute_wrong_slot_leak,
    entity_score as compute_entity_score_debug,
)
from models.entity_slot_phase40 import (
    compute_visible_iou_e0,
    compute_visible_iou_e1,
    DEFAULT_INJECT_KEYS,
    BLOCK_INNER_DIMS,
)
from models.entity_slot_phase42 import l_w_residual, l_slot_ref, l_slot_contrast
from models.entity_slot_phase43 import (
    PRIMARY_DIM,
)
from models.entity_slot_phase44 import (
    collect_blend_stats_detailed,
    l_blend_rank,
)
from models.entity_slot_phase45 import (
    l_occupancy,
    l_occupancy_structure,
    l_blend_target_balanced,
    l_visible_weights_region_balanced,
    l_visible_iou_soft,
    collect_occupancy_stats,
)
from models.entity_slot_phase46 import (
    Phase46Processor,
    MultiBlockSlotManagerP46,
    inject_multi_block_entity_slot_p46,
    restore_multiblock_state_p46,
    l_occ_contrastive,
    l_blend_ordering,
    val_score_phase46,
)
from scripts.run_animatediff import load_pipeline
from scripts.train_animatediff_vca import encode_frames_to_latents
from scripts.train_phase31 import (
    INJECT_KEY, INJECT_QUERY_DIM,
    DEPTH_PE_INIT_SCALE,
    ObjaverseDatasetWithMasks,
    get_color_entity_context,
    l_zorder_direct,
)
from scripts.train_phase35 import (
    DEFAULT_Z_BINS,
    get_entity_token_positions,
)
from scripts.train_phase39 import (
    compute_dataset_overlap_scores,
    split_train_val,
    make_sampling_weights,
)
from scripts.generate_solo_renders import ObjaverseDatasetPhase40
from scripts.train_phase43 import (
    _get_solo_entity_token_positions,
    extract_entity_prompt_ref,
    _get_block_feat,
    _resize_feat,
    ROLLOUT_T_START,
    ROLLOUT_N_STEPS,
)

# soft target params (overlap front/back)
_SOFT_FRONT_VAL = 0.90
_SOFT_BACK_VAL  = 0.05

# ─── Phase 46 하이퍼파라미터 ──────────────────────────────────────────────────
DEFAULT_STAGE_A_EPOCHS   = 5
DEFAULT_STAGE_B_EPOCHS   = 15
DEFAULT_STEPS_PER_EPOCH  = 20

# Stage A (blend + occupancy + new Phase46 terms)
DEFAULT_LA_OCC            = 2.0
DEFAULT_LA_OCC_STRUCT     = 1.5
DEFAULT_LA_OCC_CONTRAST   = 3.0   # NEW: directional exclusive suppression
DEFAULT_LA_BLEND_TARGET   = 2.0
DEFAULT_LA_BLEND_RANK     = 1.5
DEFAULT_LA_BLEND_ORDER    = 2.0   # NEW: direct ordering constraint
DEFAULT_LA_VIS            = 0.5
DEFAULT_LA_VIS_IOU        = 0.0
DEFAULT_LA_WRONG          = 0.2
DEFAULT_LA_EXCL           = 0.3
DEFAULT_LA_SLOT_REF       = 0.8
DEFAULT_LA_SLOT_CONT      = 0.2
DEFAULT_LA_W_RES          = 0.01

# Stage B (full)
DEFAULT_LB_OCC            = 1.0
DEFAULT_LB_OCC_STRUCT     = 0.8
DEFAULT_LB_OCC_CONTRAST   = 1.5   # NEW
DEFAULT_LB_BLEND_ORDER    = 1.0   # NEW
DEFAULT_LB_VIS            = 2.0
DEFAULT_LB_VIS_IOU        = 0.0
DEFAULT_LB_WRONG          = 1.0
DEFAULT_LB_SIGMA          = 1.5
DEFAULT_LB_DEPTH          = 2.0
DEFAULT_LB_OV             = 0.5
DEFAULT_LB_EXCL           = 0.5
DEFAULT_LB_SLOT_REF       = 1.5
DEFAULT_LB_SLOT_CONT      = 0.3
DEFAULT_LB_BLEND_TARGET   = 1.5
DEFAULT_LB_BLEND_RANK     = 1.0
DEFAULT_LB_W_RES          = 0.01

# LR
DEFAULT_LR_VCA           = 2e-5
DEFAULT_LR_ADAPTER       = 1e-4
DEFAULT_LR_LORA          = 5e-5
DEFAULT_LR_BLEND         = 3e-4
DEFAULT_LR_WEIGHT_HEAD   = 1e-4
DEFAULT_LR_PROJECTOR     = 5e-5
DEFAULT_LR_OBH           = 5e-3
DEFAULT_LR_OCC           = 1e-3

DEFAULT_ADAPTER_RANK     = 64
DEFAULT_LORA_RANK        = 4
DEFAULT_SLOT_BLEND       = 0.3
VAL_FRAC                 = 0.2
DEFAULT_COLLISION_AUG_PROB = 0.2
DEFAULT_COLLISION_AUG_PROB_STAGE_A = 0.75
DEFAULT_COLLISION_AUG_PROB_STAGE_B = 0.40
DEFAULT_LOW_OVERLAP_FORCE_PROB = 0.85
DEFAULT_LOW_OVERLAP_THR = 0.002
DEFAULT_COLLISION_OV_MIN   = 0.003
DEFAULT_COLLISION_OV_MAX   = 0.020
DEFAULT_COLLISION_OV_MIN_STAGE_A = 0.005
DEFAULT_COLLISION_OV_MAX_STAGE_A = 0.030
DEFAULT_COLLISION_FRONT_SWAP_PROB_STAGE_A = 0.50
DEFAULT_COLLISION_FRONT_SWAP_PROB_STAGE_B = 0.25
DEFAULT_COLLISION_MIN_ACCEPT_OVERLAP = 0.003

DEFAULT_TRAIN_OVERLAP_BOOST = 4.0
DEFAULT_TRAIN_OVERLAP_CLIP  = 6.0
DEFAULT_MASK_DILATE_RADIUS  = 1

DEFAULT_VAL_OVERLAP_BOOST = 5.0
DEFAULT_VAL_OVERLAP_CLIP  = 8.0
DEFAULT_VAL_COLLISION_THR = 0.003
DEFAULT_VAL_COLLISION_MIX = 0.65


def _load_soft_masks_from_seq_dir(
    seq_dir:  Path,
    n_frames: int,
    hw:       int,
) -> Optional[np.ndarray]:
    mask_dir = seq_dir / "mask"
    if not mask_dir.exists():
        return None
    out = []
    for fi in range(n_frames):
        p0 = mask_dir / f"{fi:04d}_entity0.png"
        p1 = mask_dir / f"{fi:04d}_entity1.png"
        if not p0.exists() or not p1.exists():
            return None
        m0 = np.array(Image.open(p0).convert("L").resize((hw, hw), Image.BILINEAR),
                      dtype=np.float32) / 255.0
        m1 = np.array(Image.open(p1).convert("L").resize((hw, hw), Image.BILINEAR),
                      dtype=np.float32) / 255.0
        out.append(np.stack([m0.reshape(-1), m1.reshape(-1)], axis=0))
    return np.stack(out, axis=0).astype(np.float32)


def _get_soft_masks_for_sample(
    dataset,
    sample_idx:  int,
    hard_masks:  np.ndarray,
) -> np.ndarray:
    if not hasattr(dataset, "samples") or sample_idx >= len(dataset.samples):
        return hard_masks
    seq_dir = dataset.samples[sample_idx].get("dir", None)
    if seq_dir is None:
        return hard_masks
    hw = int(hard_masks.shape[-1] ** 0.5)
    soft = _load_soft_masks_from_seq_dir(Path(seq_dir), hard_masks.shape[0], hw)
    if soft is None or soft.shape != hard_masks.shape:
        return hard_masks
    return soft


def _shift_with_fill(arr: np.ndarray, dx: int, dy: int, fill_value=0) -> np.ndarray:
    out = np.full_like(arr, fill_value)
    h, w = arr.shape[:2]
    src_x0 = max(0, -dx); src_x1 = min(w, w - dx)
    dst_x0 = max(0,  dx); dst_x1 = min(w, w + dx)
    src_y0 = max(0, -dy); src_y1 = min(h, h - dy)
    dst_y0 = max(0,  dy); dst_y1 = min(h, h + dy)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out
    out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def _try_collision_augment(
    dataset,
    sample_idx:        int,
    sample,
    overlap_min:       float = 0.08,
    overlap_max:       float = 0.25,
    front_swap_prob:   float = 0.0,
    max_shift:         int   = 96,
    max_tries:         int   = 24,
    min_accept_overlap: float = DEFAULT_COLLISION_MIN_ACCEPT_OVERLAP,
) -> Optional[tuple]:
    if len(sample) < 8:
        return None
    if not hasattr(dataset, "samples") or sample_idx >= len(dataset.samples):
        return None

    frames_np, _, depth_orders, _meta, entity_masks, _visible, solo_e0, solo_e1 = sample
    seq_dir = dataset.samples[sample_idx].get("dir", None)
    if seq_dir is None:
        return None
    seq_dir = Path(seq_dir)
    mask_dir = seq_dir / "mask"
    if not mask_dir.exists():
        return None

    T = min(frames_np.shape[0], len(depth_orders), entity_masks.shape[0],
            solo_e0.shape[0], solo_e1.shape[0])
    if T <= 0:
        return None

    raw0 = []
    raw1 = []
    for fi in range(T):
        p0 = mask_dir / f"{fi:04d}_entity0.png"
        p1 = mask_dir / f"{fi:04d}_entity1.png"
        if not p0.exists() or not p1.exists():
            return None
        m0 = (np.array(Image.open(p0).convert("L"), dtype=np.uint8) > 128).astype(np.uint8)
        m1 = (np.array(Image.open(p1).convert("L"), dtype=np.uint8) > 128).astype(np.uint8)
        raw0.append(m0)
        raw1.append(m1)

    rng = np.random.default_rng()
    best = None
    best_err = 1e9
    hw = int(entity_masks.shape[-1] ** 0.5)

    def _centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return None
        return float(xs.mean()), float(ys.mean())

    cands: List[Tuple[int, int]] = []
    dx_list = []
    dy_list = []
    for fi in range(T):
        c0 = _centroid(raw0[fi])
        c1 = _centroid(raw1[fi])
        if c0 is None or c1 is None:
            continue
        dx_list.append(int(round(c0[0] - c1[0])))
        dy_list.append(int(round(c0[1] - c1[1])))
    if dx_list:
        dx0 = int(np.clip(int(round(float(np.mean(dx_list)))), -max_shift, max_shift))
        dy0 = int(np.clip(int(round(float(np.mean(dy_list)))), -max_shift, max_shift))
        cands.append((dx0, dy0))
        cands.append((int(np.clip(dx0 // 2, -max_shift, max_shift)),
                      int(np.clip(dy0 // 2, -max_shift, max_shift))))
        cands.append((0, 0))
        for ddx, ddy in [(-16, 0), (16, 0), (0, -16), (0, 16), (-24, -24), (24, 24)]:
            cands.append((int(np.clip(dx0 + ddx, -max_shift, max_shift)),
                          int(np.clip(dy0 + ddy, -max_shift, max_shift))))

    while len(cands) < max_tries:
        dx = int(rng.integers(-max_shift, max_shift + 1))
        dy = int(rng.integers(-max_shift, max_shift + 1))
        cands.append((dx, dy))

    for dx, dy in cands[:max_tries]:
        ov_vals = []
        for fi in range(T):
            m0s = raw0[fi]
            m1s = _shift_with_fill(raw1[fi], dx=dx, dy=dy, fill_value=0)
            m0_ds = np.array(Image.fromarray((m0s * 255).astype(np.uint8)).resize(
                (hw, hw), Image.NEAREST), dtype=np.uint8) > 128
            m1_ds = np.array(Image.fromarray((m1s * 255).astype(np.uint8)).resize(
                (hw, hw), Image.NEAREST), dtype=np.uint8) > 128
            ov_vals.append(float((m0_ds & m1_ds).mean()))
        ov_mean = float(np.mean(ov_vals))
        err = min(abs(ov_mean - overlap_min), abs(ov_mean - overlap_max))
        if err < best_err:
            best_err = err
            best = (dx, dy, ov_mean)
        if overlap_min <= ov_mean <= overlap_max:
            best = (dx, dy, ov_mean)
            break

    if best is None:
        return None
    dx, dy, ov_mean = best
    if ov_mean < min_accept_overlap:
        return None

    frames_aug = []
    masks16 = []
    depth_orders_aug = []
    white_bg = np.array([255, 255, 255], dtype=np.uint8)

    for fi in range(T):
        m0 = raw0[fi]
        m1 = _shift_with_fill(raw1[fi], dx=dx, dy=dy, fill_value=0)
        e0 = solo_e0[fi]
        e1 = _shift_with_fill(solo_e1[fi], dx=dx, dy=dy, fill_value=255)

        front = int(depth_orders[fi][0]) if fi < len(depth_orders) else 0
        if front_swap_prob > 0.0 and float(rng.random()) < front_swap_prob:
            front = 1 - front
        frame = np.full_like(e0, white_bg)
        if front == 0:
            frame = np.where(m1[..., None] > 0, e1, frame)
            frame = np.where(m0[..., None] > 0, e0, frame)
            depth_orders_aug.append([0, 1])
        else:
            frame = np.where(m0[..., None] > 0, e0, frame)
            frame = np.where(m1[..., None] > 0, e1, frame)
            depth_orders_aug.append([1, 0])

        m0_ds = np.array(Image.fromarray((m0 * 255).astype(np.uint8)).resize(
            (hw, hw), Image.NEAREST), dtype=np.uint8) > 128
        m1_ds = np.array(Image.fromarray((m1 * 255).astype(np.uint8)).resize(
            (hw, hw), Image.NEAREST), dtype=np.uint8) > 128

        frames_aug.append(frame.astype(np.uint8))
        masks16.append(np.stack([m0_ds.reshape(-1).astype(np.float32),
                                 m1_ds.reshape(-1).astype(np.float32)], axis=0))

    return np.stack(frames_aug, axis=0), depth_orders_aug, np.stack(masks16, axis=0)


def _frame_overlap_ratios_from_masks(
    masks_t2s: torch.Tensor,
) -> torch.Tensor:
    m0 = masks_t2s[:, 0, :].float()
    m1 = masks_t2s[:, 1, :].float()
    return (m0 * m1).mean(dim=1)


def _make_overlap_frame_weights(
    overlap_ratios_t: torch.Tensor,
    boost:            float,
    clip_ratio:       float,
    eps:              float = 1e-6,
) -> torch.Tensor:
    if overlap_ratios_t.numel() == 0:
        return overlap_ratios_t.new_zeros((0,))
    if boost <= 0.0:
        return torch.ones_like(overlap_ratios_t)
    base = overlap_ratios_t.mean().clamp(min=eps)
    rel = (overlap_ratios_t / base).clamp(min=0.0, max=clip_ratio)
    return 1.0 + boost * rel


def _weighted_np_mean(values: List[float], weights: Optional[np.ndarray] = None) -> float:
    if not values:
        return 0.0
    v = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(np.mean(v))
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != v.shape[0]:
        return float(np.mean(v))
    w_sum = float(w.sum())
    if w_sum <= 1e-12:
        return float(np.mean(v))
    return float(np.sum(v * w) / w_sum)


def _dilate_entity_masks(
    masks_t2s: torch.Tensor,
    radius:    int,
) -> torch.Tensor:
    if radius <= 0 or masks_t2s.numel() == 0:
        return masks_t2s
    s = int(masks_t2s.shape[-1])
    h = int(s ** 0.5)
    if h * h != s:
        return masks_t2s
    m = masks_t2s.view(masks_t2s.shape[0], 2, h, h).float()
    k = 2 * int(radius) + 1
    m = F.max_pool2d(m, kernel_size=k, stride=1, padding=radius)
    return m.view(masks_t2s.shape[0], 2, s).clamp(0.0, 1.0)


# =============================================================================
# Rollout evaluation (Phase 46)
# =============================================================================

@torch.no_grad()
def evaluate_rollout_p46(
    pipe,
    manager:    MultiBlockSlotManagerP46,
    dataset,
    val_idx:    list,
    device:     str,
    t_start:    int = ROLLOUT_T_START,
    n_steps:    int = ROLLOUT_N_STEPS,
    overlap_boost: float = DEFAULT_VAL_OVERLAP_BOOST,
    overlap_clip:  float = DEFAULT_VAL_OVERLAP_CLIP,
    collision_thr: float = DEFAULT_VAL_COLLISION_THR,
) -> dict:
    manager.eval()
    rollout_iou_e0s = []
    rollout_iou_e1s = []
    overlap_fracs   = []

    for vi in val_idx[:min(len(val_idx), 8)]:
        try:
            sample = dataset[vi]
            frames_np, _, depth_orders, meta, entity_masks = (
                sample[:5] if len(sample) < 8 else
                (sample[0], sample[1], sample[2], sample[3], sample[4]))

            entity_ctx = get_color_entity_context(pipe, meta, device)
            toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)
            manager.set_entity_ctx(entity_ctx.float())
            manager.set_entity_tokens(toks_e0, toks_e1)
            manager.reset_slot_store()

            latents  = encode_frames_to_latents(pipe, frames_np, device)
            noise    = torch.randn_like(latents)
            t_tensor = torch.tensor([t_start], device=device).long()
            noisy    = pipe.scheduler.add_noise(latents, noise, t_tensor)

            tok = pipe.tokenizer(
                full_prompt, return_tensors="pt", padding="max_length",
                max_length=pipe.tokenizer.model_max_length, truncation=True,
            ).to(device)
            enc_hs = pipe.text_encoder(**tok).last_hidden_state.half()

            x = noisy.clone()
            scheduler_state = copy.deepcopy(pipe.scheduler)
            scheduler_state.set_timesteps(50, device=device)
            ts_full   = scheduler_state.timesteps
            start_idx = int((ts_full - t_start).abs().argmin().item())
            rollout_ts = ts_full[start_idx: start_idx + n_steps]

            for step_t in rollout_ts:
                manager.reset_slot_store()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    noise_pred = pipe.unet(
                        x, step_t.unsqueeze(0).to(device),
                        encoder_hidden_states=enc_hs).sample
                x = scheduler_state.step(noise_pred, step_t, x).prev_sample

            if manager.last_w0 is None:
                continue

            T_frames = min(manager.last_w0.shape[0], entity_masks.shape[0])
            masks_t  = torch.from_numpy(entity_masks[:T_frames].astype(np.float32)).to(device)

            for fi in range(T_frames):
                w0_f = manager.last_w0[fi:fi+1].float()
                w1_f = manager.last_w1[fi:fi+1].float()
                m_f  = masks_t[fi:fi+1]
                do   = [depth_orders[fi]] if fi < len(depth_orders) else [(0, 1)]
                rollout_iou_e0s.append(compute_visible_iou_e0(w0_f, m_f, do))
                rollout_iou_e1s.append(compute_visible_iou_e1(w1_f, m_f, do))
                overlap_fracs.append(float((m_f[:, 0, :] * m_f[:, 1, :]).mean().item()))

        except Exception as e:
            print(f"  [rollout warn] val {vi}: {e}", flush=True)
            continue

    manager.train()
    if not rollout_iou_e0s:
        return {
            "rollout_iou_e0": 0.0,
            "rollout_iou_e1": 0.0,
            "rollout_collision_iou_e0": 0.0,
            "rollout_collision_iou_e1": 0.0,
            "rollout_collision_frac": 0.0,
        }

    ov = np.asarray(overlap_fracs, dtype=np.float64)
    ov_mean = max(float(ov.mean()), 1e-8)
    rel = np.clip(ov / ov_mean, 0.0, overlap_clip)
    w = 1.0 + float(overlap_boost) * rel

    col_mask = (ov >= float(collision_thr)) if ov.size > 0 else np.zeros((0,), dtype=bool)
    if ov.size > 0 and not np.any(col_mask):
        top_k = max(1, int(np.ceil(0.1 * ov.size)))
        idx_top = np.argsort(ov)[-top_k:]
        col_mask = np.zeros_like(ov, dtype=bool)
        col_mask[idx_top] = True

    if ov.size > 0 and np.any(col_mask):
        col_w = w[col_mask]
        roll_col_e0 = _weighted_np_mean(np.asarray(rollout_iou_e0s)[col_mask].tolist(), col_w)
        roll_col_e1 = _weighted_np_mean(np.asarray(rollout_iou_e1s)[col_mask].tolist(), col_w)
        col_frac = float(np.mean(col_mask.astype(np.float64)))
    else:
        roll_col_e0, roll_col_e1, col_frac = 0.0, 0.0, 0.0

    return {
        "rollout_iou_e0": _weighted_np_mean(rollout_iou_e0s, w),
        "rollout_iou_e1": _weighted_np_mean(rollout_iou_e1s, w),
        "rollout_collision_iou_e0": roll_col_e0,
        "rollout_collision_iou_e1": roll_col_e1,
        "rollout_collision_frac": col_frac,
    }


# =============================================================================
# Teacher-forced validation (Phase 46)
# =============================================================================

@torch.no_grad()
def evaluate_val_set_phase46(
    pipe,
    manager:    MultiBlockSlotManagerP46,
    vca_layer:  VCALayer,
    dataset,
    val_idx:    list,
    device:     str,
    t_fixed:    int = 200,
    overlap_boost: float = DEFAULT_VAL_OVERLAP_BOOST,
    overlap_clip:  float = DEFAULT_VAL_OVERLAP_CLIP,
    collision_thr: float = DEFAULT_VAL_COLLISION_THR,
) -> dict:
    manager.eval()
    iou_e0s     = []
    iou_e1s     = []
    ord_accs    = []
    wrong_leaks = []
    overlap_fracs = []
    blend_stats_list = []
    occ_stats_list   = []

    for vi in val_idx:
        try:
            sample = dataset[vi]
            frames_np, _, depth_orders, meta, entity_masks = (
                sample[:5] if len(sample) < 8 else
                (sample[0], sample[1], sample[2], sample[3], sample[4]))

            entity_ctx = get_color_entity_context(pipe, meta, device)
            toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)
            manager.set_entity_ctx(entity_ctx.float())
            manager.set_entity_tokens(toks_e0, toks_e1)
            manager.reset_slot_store()

            latents  = encode_frames_to_latents(pipe, frames_np, device)
            noise    = torch.randn_like(latents)
            t_tensor = torch.tensor([t_fixed], device=device).long()
            noisy    = pipe.scheduler.add_noise(latents, noise, t_tensor)

            tok = pipe.tokenizer(
                full_prompt, return_tensors="pt", padding="max_length",
                max_length=pipe.tokenizer.model_max_length, truncation=True,
            ).to(device)
            enc_hs = pipe.text_encoder(**tok).last_hidden_state.half()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = pipe.unet(noisy, t_tensor, encoder_hidden_states=enc_hs).sample

            if manager.last_w0 is None:
                continue

            T_frames = min(manager.last_w0.shape[0], entity_masks.shape[0])
            masks_t  = torch.from_numpy(entity_masks[:T_frames].astype(np.float32)).to(device)

            for fi in range(T_frames):
                w0_f = manager.last_w0[fi:fi+1].float()
                w1_f = manager.last_w1[fi:fi+1].float()
                m_f  = masks_t[fi:fi+1]
                do   = [depth_orders[fi]] if fi < len(depth_orders) else [(0, 1)]

                iou_e0s.append(compute_visible_iou_e0(w0_f, m_f, do))
                iou_e1s.append(compute_visible_iou_e1(w1_f, m_f, do))
                ord_accs.append(compute_ordering_accuracy(w0_f, w1_f, m_f, do))
                wrong_leaks.append(compute_wrong_slot_leak(w0_f, w1_f, m_f))
                overlap_fracs.append(float((m_f[:, 0, :] * m_f[:, 1, :]).mean().item()))

                bm_raw = getattr(manager.primary, 'last_blend_map', None)
                if (bm_raw is not None and
                        isinstance(bm_raw, torch.Tensor) and
                        bm_raw.dim() >= 2 and fi < bm_raw.shape[0]):
                    bm_f   = bm_raw[fi:fi+1].float()
                    b_stat = collect_blend_stats_detailed(bm_f, m_f)
                    blend_stats_list.append(b_stat)

                o0_raw = getattr(manager.primary, 'last_o0', None)
                o1_raw = getattr(manager.primary, 'last_o1', None)
                if (o0_raw is not None and o1_raw is not None and
                        isinstance(o0_raw, torch.Tensor) and
                        isinstance(o1_raw, torch.Tensor) and
                        o0_raw.dim() >= 2 and o1_raw.dim() >= 2 and
                        fi < o0_raw.shape[0] and fi < o1_raw.shape[0]):
                    o_stat = collect_occupancy_stats(
                        o0_raw[fi:fi+1].float(),
                        o1_raw[fi:fi+1].float(),
                        m_f)
                    occ_stats_list.append(o_stat)

        except Exception as e:
            print(f"  [val warn] {vi}: {e}", flush=True)
            continue

    manager.train()

    if not iou_e0s:
        return {"tf_iou_e0": 0.0, "tf_iou_e1": 0.0, "tf_ord": 0.0,
                "tf_wrong": 1.0, "val_score": 0.0,
                "collision_iou_e0": 0.0, "collision_iou_e1": 0.0,
                "collision_ord": 0.0, "collision_wrong": 1.0,
                "collision_frac": 0.0,
                "blend_sep": 0.0, "blend_overlap_mean": 0.0,
                "blend_exclusive_mean": 0.0, "blend_bg_mean": 0.0,
                "blend_gap_bg": 0.0,
                "occ_any_overlap_mean": 0.0, "occ_any_exclusive_mean": 0.0,
                "occ_any_bg_mean": 0.0, "occ_any_sep": 0.0,
                "occ_prod_overlap_mean": 0.0, "occ_prod_exclusive_mean": 0.0,
                "occ_prod_bg_mean": 0.0, "occ_prod_sep": 0.0}

    ov = np.asarray(overlap_fracs, dtype=np.float64)
    ov_mean = max(float(ov.mean()), 1e-8)
    rel = np.clip(ov / ov_mean, 0.0, float(overlap_clip))
    frame_w = 1.0 + float(overlap_boost) * rel

    tf_iou_e0 = _weighted_np_mean(iou_e0s, frame_w)
    tf_iou_e1 = _weighted_np_mean(iou_e1s, frame_w)
    tf_ord    = _weighted_np_mean(ord_accs, frame_w)
    tf_wrong  = _weighted_np_mean(wrong_leaks, frame_w)

    col_mask = (ov >= float(collision_thr))
    if not np.any(col_mask):
        top_k = max(1, int(np.ceil(0.1 * ov.shape[0])))
        idx_top = np.argsort(ov)[-top_k:]
        col_mask = np.zeros_like(ov, dtype=bool)
        col_mask[idx_top] = True

    col_w = frame_w[col_mask] if np.any(col_mask) else None
    iou_e0_np = np.asarray(iou_e0s, dtype=np.float64)
    iou_e1_np = np.asarray(iou_e1s, dtype=np.float64)
    ord_np = np.asarray(ord_accs, dtype=np.float64)
    wrong_np = np.asarray(wrong_leaks, dtype=np.float64)
    c_iou_e0 = _weighted_np_mean(iou_e0_np[col_mask].tolist(), col_w) if np.any(col_mask) else 0.0
    c_iou_e1 = _weighted_np_mean(iou_e1_np[col_mask].tolist(), col_w) if np.any(col_mask) else 0.0
    c_ord = _weighted_np_mean(ord_np[col_mask].tolist(), col_w) if np.any(col_mask) else 0.0
    c_wrong = _weighted_np_mean(wrong_np[col_mask].tolist(), col_w) if np.any(col_mask) else 1.0
    c_frac = float(np.mean(col_mask.astype(np.float64)))

    def _agg(key):
        if not blend_stats_list:
            return 0.0
        vals = [s[key] for s in blend_stats_list]
        if len(vals) == len(iou_e0s):
            return _weighted_np_mean(vals, frame_w)
        return _weighted_np_mean(vals, None)

    blend_sep    = _agg("blend_sep")
    blend_ov     = _agg("blend_overlap_mean")
    blend_ex     = _agg("blend_exclusive_mean")
    blend_bg     = _agg("blend_bg_mean")
    blend_gap_bg = _agg("blend_gap_bg")

    def _agg_occ(key):
        if not occ_stats_list:
            return 0.0
        vals = [s[key] for s in occ_stats_list]
        if len(vals) == len(iou_e0s):
            return _weighted_np_mean(vals, frame_w)
        return _weighted_np_mean(vals, None)

    occ_any_ov   = _agg_occ("occ_any_overlap_mean")
    occ_any_ex   = _agg_occ("occ_any_exclusive_mean")
    occ_any_bg   = _agg_occ("occ_any_bg_mean")
    occ_any_sep  = _agg_occ("occ_any_sep")
    occ_prod_ov  = _agg_occ("occ_prod_overlap_mean")
    occ_prod_ex  = _agg_occ("occ_prod_exclusive_mean")
    occ_prod_bg  = _agg_occ("occ_prod_bg_mean")
    occ_prod_sep = _agg_occ("occ_prod_sep")

    # val_score computed without rollout here; rollout applied later in main loop
    vs = val_score_phase46(tf_iou_e0, tf_iou_e1, tf_ord, tf_wrong,
                           blend_sep=blend_sep, has_rollout=False)
    return {
        "tf_iou_e0":            tf_iou_e0,
        "tf_iou_e1":            tf_iou_e1,
        "tf_ord":               tf_ord,
        "tf_wrong":             tf_wrong,
        "collision_iou_e0":     c_iou_e0,
        "collision_iou_e1":     c_iou_e1,
        "collision_ord":        c_ord,
        "collision_wrong":      c_wrong,
        "collision_frac":       c_frac,
        "blend_sep":            blend_sep,
        "blend_overlap_mean":   blend_ov,
        "blend_exclusive_mean": blend_ex,
        "blend_bg_mean":        blend_bg,
        "blend_gap_bg":         blend_gap_bg,
        "occ_any_overlap_mean": occ_any_ov,
        "occ_any_exclusive_mean": occ_any_ex,
        "occ_any_bg_mean":      occ_any_bg,
        "occ_any_sep":          occ_any_sep,
        "occ_prod_overlap_mean": occ_prod_ov,
        "occ_prod_exclusive_mean": occ_prod_ex,
        "occ_prod_bg_mean":     occ_prod_bg,
        "occ_prod_sep":         occ_prod_sep,
        "val_score":            vs,
    }


# =============================================================================
# Training loop
# =============================================================================

def _set_requires_grad_stage_a(manager, vca_layer, procs, slot_bootstrap: bool = True):
    """Stage A: occupancy/blend head + (optional) slot bootstrap."""
    for p in vca_layer.parameters():   p.requires_grad_(False)
    for proc in procs:
        for p in proc.parameters():    p.requires_grad_(False)
    for proc in procs:
        if hasattr(proc, 'occ_head_e0') and proc.occ_head_e0 is not None:
            for p in proc.occ_head_e0.parameters(): p.requires_grad_(True)
        if hasattr(proc, 'occ_head_e1') and proc.occ_head_e1 is not None:
            for p in proc.occ_head_e1.parameters(): p.requires_grad_(True)
        for p in proc.overlap_blend_head.parameters(): p.requires_grad_(True)
        for p in proc.weight_head.parameters():        p.requires_grad_(True)
        proc.slot_blend_raw.requires_grad_(True)
        if slot_bootstrap:
            for p in proc.slot0_adapter.parameters(): p.requires_grad_(True)
            for p in proc.slot1_adapter.parameters(): p.requires_grad_(True)
            for p in proc.ref_proj_e0.parameters(): p.requires_grad_(True)
            for p in proc.ref_proj_e1.parameters(): p.requires_grad_(True)
    if slot_bootstrap:
        print(
            "  [stage A] occ_heads + overlap_blend_head + weight_head + "
            "slot_blend_raw + slot_adapters + ref_projectors",
            flush=True,
        )
    else:
        print("  [stage A] occ_heads + overlap_blend_head + weight_head + slot_blend_raw", flush=True)


def _set_requires_grad_stage_b(manager, vca_layer, procs):
    """Stage B: 전체 joint finetuning."""
    for p in vca_layer.parameters():   p.requires_grad_(True)
    for proc in procs:
        for p in proc.parameters():    p.requires_grad_(True)
    print("  [stage B] 전체 파라미터 학습 (joint finetuning)", flush=True)


def train_phase46(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    debug_dir = Path(args.debug_dir)
    save_dir  = Path(args.save_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("[Phase 46] 파이프라인 로드 중...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    print(f"[Phase 46] checkpoint 로드: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location=device)
    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM, context_dim=768,
        n_heads=8, n_entities=2, z_bins=DEFAULT_Z_BINS, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=DEPTH_PE_INIT_SCALE,
    ).to(device)
    vca_layer.load_state_dict(ckpt["vca_state_dict"])
    print("  VCA loaded", flush=True)

    print("[Phase 46] 데이터셋 로드 중...", flush=True)
    try:
        dataset = ObjaverseDatasetPhase40(args.data_root, n_frames=args.n_frames)
        print(f"  ObjaverseDatasetPhase40: {len(dataset)} samples", flush=True)
    except Exception as e:
        print(f"  Phase40 dataset 실패({e}), fallback", flush=True)
        dataset = ObjaverseDatasetWithMasks(args.data_root, n_frames=args.n_frames)

    if len(dataset) == 0:
        raise RuntimeError(f"데이터셋 비어있음: {args.data_root}")

    overlap_scores = compute_dataset_overlap_scores(dataset)
    train_idx, val_idx = split_train_val(overlap_scores, val_frac=args.val_frac)
    sample_weights = make_sampling_weights(train_idx, overlap_scores)
    print(f"  train={len(train_idx)}  val={len(val_idx)}", flush=True)

    inject_keys = args.inject_keys.split(",") if args.inject_keys else DEFAULT_INJECT_KEYS
    init_ctx    = get_color_entity_context(pipe, dataset[train_idx[0]][3], device)
    procs, orig_procs = inject_multi_block_entity_slot_p46(
        pipe, vca_layer, init_ctx,
        inject_keys     = inject_keys,
        slot_blend_init = args.slot_blend,
        adapter_rank    = args.adapter_rank,
        lora_rank       = args.lora_rank,
        use_blend_head  = True,
        obh_hidden      = 32,
        proj_hidden     = 256,
        occ_hidden      = 64,
    )
    for p in procs:
        p.to(device)

    manager = MultiBlockSlotManagerP46(procs, inject_keys, primary_idx=1)

    print("[Phase 46] checkpoint 복원...", flush=True)
    restore_multiblock_state_p46(manager, ckpt, device=device)

    for p in pipe.unet.parameters():         p.requires_grad_(False)
    for p in pipe.text_encoder.parameters(): p.requires_grad_(False)
    for p in pipe.vae.parameters():          p.requires_grad_(False)

    # ── Epoch 0 no-op validation ─────────────────────────────────────────────
    print("\n[Phase 46] Epoch 0 no-op validation...", flush=True)
    vca_layer.eval(); manager.eval()
    val_m0 = evaluate_val_set_phase46(
        pipe, manager, vca_layer, dataset, val_idx, device,
        t_fixed=args.t_max // 2,
        overlap_boost=args.val_overlap_boost,
        overlap_clip=args.val_overlap_clip,
        collision_thr=args.val_collision_thr)
    print(f"  [epoch0-noop] iou_e0={val_m0['tf_iou_e0']:.4f}  "
          f"iou_e1={val_m0['tf_iou_e1']:.4f}  "
          f"c_iou_e0={val_m0['collision_iou_e0']:.4f}  "
          f"c_iou_e1={val_m0['collision_iou_e1']:.4f}  "
          f"blend_sep={val_m0['blend_sep']:.4f}  "
          f"blend_ov={val_m0['blend_overlap_mean']:.4f}  "
          f"blend_ex={val_m0['blend_exclusive_mean']:.4f}  "
          f"blend_bg={val_m0['blend_bg_mean']:.4f}  "
          f"val_score={val_m0['val_score']:.4f}", flush=True)
    print(f"  [epoch0-occ] any_ov={val_m0['occ_any_overlap_mean']:.4f}  "
          f"any_ex={val_m0['occ_any_exclusive_mean']:.4f}  "
          f"any_bg={val_m0['occ_any_bg_mean']:.4f}  "
          f"prod_ov={val_m0['occ_prod_overlap_mean']:.4f}  "
          f"prod_ex={val_m0['occ_prod_exclusive_mean']:.4f}",
          flush=True)
    vca_layer.train(); manager.train()

    history        = []
    best_val_score = -1.0
    best_epoch     = -1
    total_epochs   = args.stage_a_epochs + args.stage_b_epochs

    for epoch in range(total_epochs):
        is_stage_a = (epoch < args.stage_a_epochs)

        if epoch == 0:
            print(f"\n[Phase 46] Stage A 시작 ({args.stage_a_epochs} epochs): "
                  f"OccupancyHead + contrastive + blend ordering", flush=True)
            _set_requires_grad_stage_a(
                manager, vca_layer, procs,
                slot_bootstrap=args.stage_a_slot_bootstrap)
            stage_a_groups = [
                {"params": manager.occupancy_head_params(),
                 "lr": args.lr_occ,       "name": "occ_heads"},
                {"params": manager.overlap_blend_head_params(),
                 "lr": args.lr_obh,       "name": "obh"},
                {"params": manager.weight_head_params(),
                 "lr": args.lr_weight_head, "name": "weight_head"},
                {"params": [p.slot_blend_raw for p in procs],
                 "lr": args.lr_blend,     "name": "blend_raw"},
            ]
            if args.stage_a_slot_bootstrap:
                stage_a_groups += [
                    {"params": manager.adapter_params(),
                     "lr": args.lr_adapter * 0.5, "name": "adapters"},
                    {"params": manager.projector_params(),
                     "lr": args.lr_projector * 0.5, "name": "projectors"},
                ]
            optimizer = optim.AdamW(stage_a_groups, weight_decay=1e-4)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.stage_a_epochs, eta_min=1e-6)

        elif epoch == args.stage_a_epochs:
            print(f"\n[Phase 46] Stage B 시작 ({args.stage_b_epochs} epochs): "
                  f"전체 미세조정 (occ+OBH at lr/5)", flush=True)
            _set_requires_grad_stage_b(manager, vca_layer, procs)
            optimizer = optim.AdamW([
                {"params": list(vca_layer.parameters()),
                 "lr": args.lr_vca * 0.1,  "name": "vca"},
                {"params": manager.adapter_params(),
                 "lr": args.lr_adapter,     "name": "adapters"},
                {"params": manager.lora_params(),
                 "lr": args.lr_lora,        "name": "lora"},
                {"params": manager.blend_params(),
                 "lr": args.lr_blend,       "name": "blend"},
                {"params": manager.weight_head_params(),
                 "lr": args.lr_weight_head * 0.1, "name": "weight_head"},
                {"params": manager.projector_params(),
                 "lr": args.lr_projector,   "name": "projectors"},
                # Phase 46: occ_head + OBH at lr/5 to preserve Stage A progress
                {"params": manager.overlap_blend_head_params(),
                 "lr": args.lr_obh * 0.2,  "name": "obh"},
                {"params": manager.occupancy_head_params(),
                 "lr": args.lr_occ * 0.2,  "name": "occ_heads"},
            ], weight_decay=1e-4)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.stage_b_epochs, eta_min=args.lr_vca * 0.05)

        vca_layer.train(); manager.train()

        if is_stage_a:
            loss_keys = ["total", "occ", "occ_struct", "occ_cont", "blend_target",
                         "blend_rank", "blend_order",
                         "vis", "vis_iou", "wrong", "excl", "slot_ref", "slot_cont", "w_res"]
        else:
            loss_keys = ["total", "occ", "occ_struct", "occ_cont", "vis", "vis_iou",
                         "wrong", "sigma", "depth", "ov",
                         "excl", "slot_ref", "slot_cont", "blend_target",
                         "blend_rank", "blend_order", "w_res"]
        epoch_losses = {k: [] for k in loss_keys}

        chosen     = np.random.choice(len(train_idx), size=args.steps_per_epoch,
                                      replace=True, p=sample_weights)
        step_indices = [train_idx[ci] for ci in chosen]
        n_collision_attempt = 0
        n_collision_aug = 0
        n_low_overlap_samples = 0

        for batch_idx, data_idx in enumerate(step_indices):
            sample = dataset[data_idx]
            if len(sample) >= 8:
                frames_np, _, depth_orders, meta, entity_masks, _, solo_e0, solo_e1 = sample
            else:
                frames_np, _, depth_orders, meta, entity_masks = sample[:5]

            used_collision_aug = False
            sample_overlap_score = float(overlap_scores[data_idx]) if data_idx < len(overlap_scores) else 0.0
            collision_prob_stage = args.collision_aug_prob_stage_a if is_stage_a else args.collision_aug_prob_stage_b
            collision_prob = max(args.collision_aug_prob, collision_prob_stage)
            if sample_overlap_score < args.low_overlap_thr:
                n_low_overlap_samples += 1
                collision_prob = max(collision_prob, args.low_overlap_force_prob)

            ov_min = args.collision_ov_min_stage_a if is_stage_a else args.collision_ov_min
            ov_max = args.collision_ov_max_stage_a if is_stage_a else args.collision_ov_max
            front_swap_prob = (
                args.collision_front_swap_prob_stage_a
                if is_stage_a else args.collision_front_swap_prob_stage_b
            )

            if (collision_prob > 0.0 and
                    len(sample) >= 8 and
                    np.random.rand() < collision_prob):
                n_collision_attempt += 1
                aug = _try_collision_augment(
                    dataset, data_idx, sample,
                    overlap_min=ov_min,
                    overlap_max=ov_max,
                    front_swap_prob=front_swap_prob,
                    min_accept_overlap=args.collision_min_accept_overlap)
                if aug is not None:
                    frames_np, depth_orders, entity_masks = aug
                    used_collision_aug = True
                    n_collision_aug += 1

            if args.use_soft_masks and not used_collision_aug:
                train_masks = _get_soft_masks_for_sample(dataset, data_idx, entity_masks)
            else:
                train_masks = entity_masks

            entity_ctx = get_color_entity_context(pipe, meta, device)
            toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)

            with torch.no_grad():
                latents = encode_frames_to_latents(pipe, frames_np, device)
            noise = torch.randn_like(latents)
            t     = torch.randint(0, args.t_max, (1,), device=device).long()
            noisy = pipe.scheduler.add_noise(latents, noise, t)

            T_frames       = min(frames_np.shape[0], entity_masks.shape[0])
            masks_t_hard   = torch.from_numpy(entity_masks[:T_frames].astype(np.float32)).to(device)
            masks_t        = torch.from_numpy(train_masks[:T_frames].astype(np.float32)).to(device)
            depth_orders_t = depth_orders[:T_frames]
            if args.mask_dilate_radius > 0:
                masks_t = _dilate_entity_masks(masks_t, args.mask_dilate_radius)
                masks_overlap_t = _dilate_entity_masks(masks_t_hard, args.mask_dilate_radius)
            else:
                masks_overlap_t = masks_t_hard

            frame_overlap_t = _frame_overlap_ratios_from_masks(masks_overlap_t)
            frame_w_t = _make_overlap_frame_weights(
                frame_overlap_t, args.train_overlap_boost, args.train_overlap_clip)

            # ── Entity-only ref forwards ──────────────────────────────────
            F0_refs: Dict[int, Optional[torch.Tensor]] = {}
            F1_refs: Dict[int, Optional[torch.Tensor]] = {}
            need_refs = (not is_stage_a) or args.stage_a_slot_bootstrap
            if need_refs:
                prompt_e0 = meta.get("prompt_entity0", full_prompt)
                prompt_e1 = meta.get("prompt_entity1", full_prompt)
                manager.set_entity_ctx(entity_ctx.float())
                manager.set_entity_tokens(toks_e0, toks_e1)
                try:
                    F0_refs = extract_entity_prompt_ref(
                        pipe, manager, noisy, t, prompt_e0, device, entity_idx=0)
                except Exception as e:
                    print(f"  [warn] e0-ref: {e}", flush=True)
                try:
                    F1_refs = extract_entity_prompt_ref(
                        pipe, manager, noisy, t, prompt_e1, device, entity_idx=1)
                except Exception as e:
                    print(f"  [warn] e1-ref: {e}", flush=True)

            # ── Full forward (with grad) ──────────────────────────────────
            manager.set_entity_ctx(entity_ctx.float())
            manager.set_entity_tokens(toks_e0, toks_e1)
            manager.reset_slot_store()

            tok = pipe.tokenizer(
                full_prompt, return_tensors="pt", padding="max_length",
                max_length=pipe.tokenizer.model_max_length, truncation=True,
            ).to(device)
            with torch.no_grad():
                enc_hs = pipe.text_encoder(**tok).last_hidden_state.half()

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = pipe.unet(noisy, t, encoder_hidden_states=enc_hs).sample

            BF    = manager.last_w0.shape[0] if manager.last_w0 is not None else 0
            T_use = min(BF, T_frames)

            # ── L_occupancy + L_occ_contrastive (primary block only) ─────
            l_occ = torch.tensor(0.0, device=device)
            l_occ_struct = torch.tensor(0.0, device=device)
            l_occ_cont   = torch.tensor(0.0, device=device)
            primary = manager.primary
            o0_fl = getattr(primary, 'last_o0_for_loss', None)
            o1_fl = getattr(primary, 'last_o1_for_loss', None)
            masks_occ = None
            if o0_fl is not None and isinstance(o0_fl, torch.Tensor):
                S_blk  = o0_fl.shape[1]
                S_mask = masks_t.shape[-1]
                if S_blk != S_mask:
                    H_b = int(S_blk**0.5); H_m = int(S_mask**0.5)
                    m4d = masks_t.view(masks_t.shape[0], 2, H_m, H_m)
                    m4d = torch.nn.functional.interpolate(
                        m4d.float(), size=(H_b, H_b), mode='nearest')
                    masks_occ = m4d.view(masks_t.shape[0], 2, S_blk)
                else:
                    masks_occ = masks_t

                n_occ_frames = min(o0_fl.shape[0], T_frames)
                occ_w_sum = torch.tensor(0.0, device=device)
                for fi in range(n_occ_frames):
                    fw = frame_w_t[fi]
                    o0_fi = o0_fl[fi:fi+1].float()
                    o1_fi = o1_fl[fi:fi+1].float()
                    m_occ_fi = masks_occ[fi:fi+1]
                    l_occ = l_occ + l_occupancy(o0_fi, o1_fi, m_occ_fi) * fw
                    l_occ_struct = l_occ_struct + l_occupancy_structure(
                        o0_fi, o1_fi, m_occ_fi) * fw
                    # NEW Phase 46: contrastive directional suppression
                    l_occ_cont = l_occ_cont + l_occ_contrastive(
                        o0_fi, o1_fi, m_occ_fi,
                        margin=args.occ_contrastive_margin) * fw
                    occ_w_sum = occ_w_sum + fw
                occ_w_sum = occ_w_sum.clamp(min=1e-6)
                l_occ       = l_occ       / occ_w_sum
                l_occ_struct = l_occ_struct / occ_w_sum
                l_occ_cont   = l_occ_cont   / occ_w_sum

            # ── L_blend_ordering (primary block, NEW Phase 46) ────────────
            l_blend_ord = torch.tensor(0.0, device=device)
            bm_for_loss = getattr(primary, 'last_blend_map_for_loss', None)
            if (bm_for_loss is not None and
                    isinstance(bm_for_loss, torch.Tensor) and
                    bm_for_loss.dim() >= 2 and T_use > 0 and masks_occ is not None):
                n_blend_occ_frames = min(bm_for_loss.shape[0], T_use, masks_occ.shape[0])
                blend_ord_w_sum = torch.tensor(0.0, device=device)
                for fi in range(n_blend_occ_frames):
                    fw = frame_w_t[fi]
                    bm_f     = bm_for_loss[fi:fi+1].float()
                    m_occ_fi = masks_occ[fi:fi+1]
                    l_blend_ord = l_blend_ord + l_blend_ordering(
                        bm_f, m_occ_fi,
                        margin=args.blend_order_margin) * fw
                    blend_ord_w_sum = blend_ord_w_sum + fw
                l_blend_ord = l_blend_ord / blend_ord_w_sum.clamp(min=1e-6)

            # ── L_vis ─────────────────────────────────────────────────────
            l_vis = torch.tensor(0.0, device=device)
            l_vis_iou = torch.tensor(0.0, device=device)
            if manager.last_w0 is not None:
                vis_w_sum = torch.tensor(0.0, device=device)
                for fi in range(T_use):
                    fw = frame_w_t[fi]
                    w0_f = manager.last_w0[fi:fi+1].float()
                    w1_f = manager.last_w1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    do   = [depth_orders_t[fi]] if fi < len(depth_orders_t) else [(0,1)]
                    l_vis = l_vis + l_visible_weights_region_balanced(
                        w0_f, w1_f, m_f, do,
                        front_val=_SOFT_FRONT_VAL, back_val=_SOFT_BACK_VAL) * fw
                    l_vis_iou = l_vis_iou + l_visible_iou_soft(
                        w0_f, w1_f, m_f, do,
                        front_val=_SOFT_FRONT_VAL, back_val=_SOFT_BACK_VAL) * fw
                    vis_w_sum = vis_w_sum + fw
                vis_w_sum = vis_w_sum.clamp(min=1e-6)
                l_vis = l_vis / vis_w_sum
                l_vis_iou = l_vis_iou / vis_w_sum

            # ── L_wrong ───────────────────────────────────────────────────
            l_wrong = torch.tensor(0.0, device=device)
            if manager.last_w0 is not None:
                wrong_w_sum = torch.tensor(0.0, device=device)
                for fi in range(T_use):
                    fw = frame_w_t[fi]
                    l_wrong = l_wrong + l_wrong_slot_suppression(
                        manager.last_w0[fi:fi+1].float(),
                        manager.last_w1[fi:fi+1].float(),
                        masks_t_hard[fi:fi+1]) * fw
                    wrong_w_sum = wrong_w_sum + fw
                l_wrong = l_wrong / wrong_w_sum.clamp(min=1e-6)

            # ── L_blend_target_balanced + L_blend_rank (primary) ─────────
            l_bt = torch.tensor(0.0, device=device)
            l_br = torch.tensor(0.0, device=device)
            if (bm_for_loss is not None and
                    isinstance(bm_for_loss, torch.Tensor) and
                    bm_for_loss.dim() >= 2 and T_use > 0):
                n_blend_frames = min(bm_for_loss.shape[0], T_use)
                blend_w_sum = torch.tensor(0.0, device=device)
                for fi in range(n_blend_frames):
                    fw = frame_w_t[fi]
                    bm_f = bm_for_loss[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    l_bt = l_bt + l_blend_target_balanced(bm_f, m_f) * fw
                    l_br = l_br + l_blend_rank(bm_f, m_f) * fw
                    blend_w_sum = blend_w_sum + fw
                blend_w_sum = blend_w_sum.clamp(min=1e-6)
                l_bt = l_bt / blend_w_sum
                l_br = l_br / blend_w_sum

            # ── L_w_res ───────────────────────────────────────────────────
            l_w_res = torch.tensor(0.0, device=device)
            if manager.last_w_delta is not None:
                l_w_res = l_w_residual(manager.last_w_delta)

            # ── Slot identity losses (cross-block) ───────────────────────
            l_excl = torch.tensor(0.0, device=device)
            l_slot_ref_all = torch.tensor(0.0, device=device)
            l_slot_cont_all = torch.tensor(0.0, device=device)
            excl_w_sum = torch.tensor(0.0, device=device)
            slot_ref_w_sum = torch.tensor(0.0, device=device)
            slot_cont_w_sum = torch.tensor(0.0, device=device)
            for blk_idx, _proc in enumerate(manager.procs):
                blk_F0 = _get_block_feat(manager, blk_idx, 0)
                blk_F1 = _get_block_feat(manager, blk_idx, 1)
                blk_Fg = _get_block_feat(manager, blk_idx, 'g')
                if blk_F0 is None:
                    continue

                S_blk = blk_F0.shape[1]
                S_mask = masks_t.shape[-1]
                if S_blk != S_mask:
                    H_b = int(S_blk**0.5)
                    H_m = int(S_mask**0.5)
                    m4d = masks_t.view(masks_t.shape[0], 2, H_m, H_m)
                    m4d = torch.nn.functional.interpolate(
                        m4d.float(), size=(H_b, H_b), mode='nearest')
                    masks_blk = m4d.view(masks_t.shape[0], 2, S_blk)
                else:
                    masks_blk = masks_t

                n_blk_frames = min(blk_F0.shape[0], T_frames)
                for fi in range(n_blk_frames):
                    fw = frame_w_t[fi]
                    l_excl = l_excl + l_entity_exclusive(
                        blk_F0[fi:fi+1].float(), blk_F1[fi:fi+1].float(),
                        blk_Fg[fi:fi+1].float(), masks_blk[fi:fi+1]) * fw
                    excl_w_sum = excl_w_sum + fw

                    if F0_refs and F1_refs:
                        if S_blk != masks_t_hard.shape[-1]:
                            H_bh = int(S_blk**0.5)
                            H_mh = int(masks_t_hard.shape[-1]**0.5)
                            mh4d = masks_t_hard.view(masks_t_hard.shape[0], 2, H_mh, H_mh)
                            mh4d = torch.nn.functional.interpolate(
                                mh4d.float(), size=(H_bh, H_bh), mode='nearest')
                            masks_blk_hard = mh4d.view(masks_t_hard.shape[0], 2, S_blk)
                        else:
                            masks_blk_hard = masks_t_hard
                        vis_e0 = masks_blk_hard[fi, 0, :].unsqueeze(0)
                        vis_e1 = masks_blk_hard[fi, 1, :].unsqueeze(0)
                        F0_ref = F0_refs.get(blk_idx, None)
                        F1_ref = F1_refs.get(blk_idx, None)
                        if F0_ref is not None and F1_ref is not None:
                            F0_rs = F0_ref[fi:fi+1]
                            F1_rs = F1_ref[fi:fi+1]
                            if F0_rs.shape[1] != S_blk:
                                F0_rs = _resize_feat(F0_rs, S_blk)
                                F1_rs = _resize_feat(F1_rs, S_blk)
                            l_slot_ref_all = l_slot_ref_all + l_slot_ref(
                                blk_F0[fi:fi+1].float(), F0_rs.detach().float(), vis_e0) * fw
                            l_slot_ref_all = l_slot_ref_all + l_slot_ref(
                                blk_F1[fi:fi+1].float(), F1_rs.detach().float(), vis_e1) * fw
                            l_slot_cont_all = l_slot_cont_all + l_slot_contrast(
                                blk_F0[fi:fi+1].float(), blk_F1[fi:fi+1].float(),
                                F0_rs.detach().float(), F1_rs.detach().float(),
                                vis_e0, vis_e1) * fw
                            slot_ref_w_sum = slot_ref_w_sum + 2.0 * fw
                            slot_cont_w_sum = slot_cont_w_sum + fw

            l_excl = l_excl / excl_w_sum.clamp(min=1e-6)
            if slot_ref_w_sum.item() > 0.0:
                l_slot_ref_all = l_slot_ref_all / slot_ref_w_sum
            if slot_cont_w_sum.item() > 0.0:
                l_slot_cont_all = l_slot_cont_all / slot_cont_w_sum

            if is_stage_a:
                loss = (args.la_occ          * l_occ
                      + args.la_occ_struct   * l_occ_struct
                      + args.la_occ_contrast * l_occ_cont      # NEW Phase 46
                      + args.la_blend_target * l_bt
                      + args.la_blend_rank   * l_br
                      + args.la_blend_order  * l_blend_ord      # NEW Phase 46
                      + args.la_vis          * l_vis
                      + args.la_vis_iou      * l_vis_iou
                      + args.la_wrong        * l_wrong
                      + args.la_excl         * l_excl
                      + args.la_slot_ref     * l_slot_ref_all
                      + args.la_slot_cont    * l_slot_cont_all
                      + args.la_w_res        * l_w_res)

                for k, v in [("total", loss), ("occ", l_occ), ("occ_struct", l_occ_struct),
                              ("occ_cont", l_occ_cont),
                              ("blend_target", l_bt), ("blend_rank", l_br),
                              ("blend_order", l_blend_ord),
                              ("vis", l_vis), ("vis_iou", l_vis_iou),
                              ("wrong", l_wrong), ("excl", l_excl),
                              ("slot_ref", l_slot_ref_all), ("slot_cont", l_slot_cont_all),
                              ("w_res", l_w_res)]:
                    epoch_losses[k].append(float(v.item())
                                           if isinstance(v, torch.Tensor) else float(v))
            else:
                # Stage B: full loss + Phase 46 terms
                l_sigma = torch.tensor(0.0, device=device)
                if manager.last_alpha0 is not None:
                    sigma_w_sum = torch.tensor(0.0, device=device)
                    for fi in range(T_use):
                        fw = frame_w_t[fi]
                        l_sigma = l_sigma + l_sigma_spatial(
                            manager.last_alpha0[fi:fi+1].float(),
                            manager.last_alpha1[fi:fi+1].float(),
                            masks_t[fi:fi+1]) * fw
                        sigma_w_sum = sigma_w_sum + fw
                    l_sigma = l_sigma / sigma_w_sum.clamp(min=1e-6)

                sigma_acc = list(manager.sigma_acc)
                depth_masks_np = (masks_t_hard[:T_frames].detach().cpu().numpy() > 0.5)
                l_depth   = l_zorder_direct(sigma_acc, depth_orders_t, depth_masks_np)

                l_ov = torch.tensor(0.0, device=device)
                if manager.last_w0 is not None:
                    ov_w_sum = torch.tensor(0.0, device=device)
                    for fi in range(T_use):
                        fw = frame_w_t[fi]
                        do = [depth_orders_t[fi]] if fi < len(depth_orders_t) else [(0,1)]
                        l_ov = l_ov + l_overlap_ordering(
                            manager.last_w0[fi:fi+1].float(),
                            manager.last_w1[fi:fi+1].float(),
                            masks_t[fi:fi+1], do) * fw
                        ov_w_sum = ov_w_sum + fw
                    l_ov = l_ov / ov_w_sum.clamp(min=1e-6)

                loss = (args.lb_occ          * l_occ
                      + args.lb_occ_struct   * l_occ_struct
                      + args.lb_occ_contrast * l_occ_cont       # NEW Phase 46
                      + args.lb_vis          * l_vis
                      + args.lb_vis_iou      * l_vis_iou
                      + args.lb_wrong        * l_wrong
                      + args.lb_sigma        * l_sigma
                      + args.lb_depth        * l_depth
                      + args.lb_ov           * l_ov
                      + args.lb_excl         * l_excl
                      + args.lb_slot_ref     * l_slot_ref_all
                      + args.lb_slot_cont    * l_slot_cont_all
                      + args.lb_blend_target * l_bt
                      + args.lb_blend_rank   * l_br
                      + args.lb_blend_order  * l_blend_ord       # NEW Phase 46
                      + args.lb_w_res        * l_w_res)

                for k, v in [("total", loss), ("occ", l_occ), ("occ_struct", l_occ_struct),
                              ("occ_cont", l_occ_cont),
                              ("vis", l_vis), ("vis_iou", l_vis_iou),
                              ("wrong", l_wrong), ("sigma", l_sigma), ("depth", l_depth),
                              ("ov", l_ov), ("excl", l_excl), ("slot_ref", l_slot_ref_all),
                              ("slot_cont", l_slot_cont_all), ("blend_target", l_bt),
                              ("blend_rank", l_br), ("blend_order", l_blend_ord),
                              ("w_res", l_w_res)]:
                    epoch_losses[k].append(float(v.item()) if isinstance(v, torch.Tensor) else float(v))

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss ep={epoch} step={batch_idx} → skip", flush=True)
                continue

            loss.backward()

            if is_stage_a:
                torch.nn.utils.clip_grad_norm_(manager.occupancy_head_params(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.overlap_blend_head_params(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.weight_head_params(),        max_norm=1.0)
                if args.stage_a_slot_bootstrap:
                    torch.nn.utils.clip_grad_norm_(manager.adapter_params(),        max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(manager.projector_params(),      max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(list(vca_layer.parameters()),        max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.adapter_params(),            max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.lora_params(),               max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(manager.blend_params(),              max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(manager.weight_head_params(),        max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.projector_params(),          max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.overlap_blend_head_params(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.occupancy_head_params(),     max_norm=1.0)

            optimizer.step()

        lr_scheduler.step()

        avg       = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
        stage_lbl = "A" if is_stage_a else "B"
        print(f"[Phase 46][Stage {stage_lbl}] epoch {epoch:03d}/{total_epochs-1}  "
              f"loss={avg['total']:.4f}  occ={avg['occ']:.4f}  "
              f"occ_struct={avg['occ_struct']:.4f}  occ_cont={avg['occ_cont']:.4f}  "
              f"blend_target={avg['blend_target']:.4f}  blend_rank={avg['blend_rank']:.4f}  "
              f"blend_order={avg['blend_order']:.4f}  "
              f"vis={avg['vis']:.4f}  vis_iou={avg['vis_iou']:.4f}  "
              f"wrong={avg['wrong']:.4f}  "
              f"excl={avg['excl']:.4f}  slot_ref={avg['slot_ref']:.4f}  "
              f"slot_cont={avg['slot_cont']:.4f}  w_res={avg['w_res']:.4f}  "
              f"aug={n_collision_aug}/{n_collision_attempt}  "
              f"low_ov={n_low_overlap_samples}/{len(step_indices)}",
              flush=True)

        # ── Validation ────────────────────────────────────────────────────
        should_eval = (
            (is_stage_a and epoch % args.eval_every_stage_a == 0) or
            (not is_stage_a and epoch % args.eval_every == 0) or
            (epoch == total_epochs - 1)
        )
        # Rollout: every rollout_every epochs + last epoch
        should_rollout = (
            (epoch % args.rollout_every == 0) or
            (epoch == total_epochs - 1)
        ) and should_eval

        if should_eval:
            vca_layer.eval(); manager.eval()

            val_m = evaluate_val_set_phase46(
                pipe, manager, vca_layer, dataset, val_idx, device,
                t_fixed=args.t_max // 2,
                overlap_boost=args.val_overlap_boost,
                overlap_clip=args.val_overlap_clip,
                collision_thr=args.val_collision_thr)

            # Phase 46: only compute rollout when should_rollout
            if should_rollout:
                rollout_m = evaluate_rollout_p46(
                    pipe, manager, dataset, val_idx, device,
                    t_start=ROLLOUT_T_START, n_steps=ROLLOUT_N_STEPS,
                    overlap_boost=args.val_overlap_boost,
                    overlap_clip=args.val_overlap_clip,
                    collision_thr=args.val_collision_thr)
                print(f"  [rollout] iou_e0={rollout_m['rollout_iou_e0']:.4f}  "
                      f"iou_e1={rollout_m['rollout_iou_e1']:.4f}", flush=True)
                has_rollout = True
            else:
                rollout_m = {
                    "rollout_iou_e0": 0.0, "rollout_iou_e1": 0.0,
                    "rollout_collision_iou_e0": 0.0, "rollout_collision_iou_e1": 0.0,
                    "rollout_collision_frac": 0.0,
                }
                has_rollout = False

            # Phase 46: fair val_score using has_rollout flag
            vs_global = val_score_phase46(
                val_m["tf_iou_e0"], val_m["tf_iou_e1"],
                val_m["tf_ord"], val_m["tf_wrong"],
                rollout_m["rollout_iou_e0"], rollout_m["rollout_iou_e1"],
                val_m["blend_sep"],
                has_rollout=has_rollout)
            vs_collision = val_score_phase46(
                val_m["collision_iou_e0"], val_m["collision_iou_e1"],
                val_m["collision_ord"], val_m["collision_wrong"],
                rollout_m["rollout_collision_iou_e0"], rollout_m["rollout_collision_iou_e1"],
                val_m["blend_sep"],
                has_rollout=has_rollout)
            mix = float(np.clip(args.val_collision_mix, 0.0, 1.0))
            vs = (1.0 - mix) * vs_global + mix * vs_collision
            val_m["val_score"] = vs
            val_m["val_score_global"] = vs_global
            val_m["val_score_collision"] = vs_collision
            val_m["has_rollout"] = has_rollout

            print(f"  [val] tf_iou_e0={val_m['tf_iou_e0']:.4f}  "
                  f"tf_iou_e1={val_m['tf_iou_e1']:.4f}  "
                  f"tf_ord={val_m['tf_ord']:.4f}  "
                  f"tf_wrong={val_m['tf_wrong']:.4f}", flush=True)
            print(f"  [collision] frac={val_m['collision_frac']:.3f}  "
                  f"iou_e0={val_m['collision_iou_e0']:.4f}  "
                  f"iou_e1={val_m['collision_iou_e1']:.4f}  "
                  f"ord={val_m['collision_ord']:.4f}  "
                  f"wrong={val_m['collision_wrong']:.4f}  "
                  f"roll_e0={rollout_m['rollout_collision_iou_e0']:.4f}  "
                  f"roll_e1={rollout_m['rollout_collision_iou_e1']:.4f}"
                  f"  [rollout={'yes' if has_rollout else 'no'}]", flush=True)
            print(f"  [blend] sep={val_m['blend_sep']:.4f}  "
                  f"ov={val_m['blend_overlap_mean']:.4f}  "
                  f"ex={val_m['blend_exclusive_mean']:.4f}  "
                  f"bg={val_m['blend_bg_mean']:.4f}  "
                  f"val_score={vs:.4f} (global={vs_global:.4f}, col={vs_collision:.4f})",
                  flush=True)
            print(f"  [occ] any_ov={val_m['occ_any_overlap_mean']:.4f}  "
                  f"any_ex={val_m['occ_any_exclusive_mean']:.4f}  "
                  f"any_bg={val_m['occ_any_bg_mean']:.4f}  "
                  f"prod_ov={val_m['occ_prod_overlap_mean']:.4f}  "
                  f"prod_ex={val_m['occ_prod_exclusive_mean']:.4f}  "
                  f"prod_sep={val_m['occ_prod_sep']:.4f}",
                  flush=True)

            # GIF
            try:
                probe = dataset[val_idx[0]]
                probe_ctx = get_color_entity_context(pipe, probe[3], device)
                probe_e0, probe_e1, probe_prompt = get_entity_token_positions(pipe, probe[3])
                manager.set_entity_ctx(probe_ctx.float())
                manager.set_entity_tokens(probe_e0, probe_e1)
                manager.reset_slot_store()
                gen = torch.Generator(device=device).manual_seed(args.eval_seed)
                out = pipe(prompt=probe_prompt, num_frames=args.n_frames,
                           num_inference_steps=args.n_steps,
                           height=args.height, width=args.width,
                           generator=gen, output_type="np")
                frames_eval = (out.frames[0] * 255).astype(np.uint8)
                gif_path = debug_dir / f"eval_epoch{epoch:03d}.gif"
                iio2.mimwrite(str(gif_path), frames_eval, fps=8, loop=0)
                es, sr, cr = compute_entity_score_debug(frames_eval)
                print(f"  [debug] entity_score={es:.4f} survival={sr:.4f} chimera={cr:.4f}",
                      flush=True)
            except Exception as e:
                print(f"  [warn] GIF 실패: {e}", flush=True)

            history.append({
                "epoch": epoch, "stage": stage_lbl,
                "val_score": vs,
                "has_rollout": has_rollout,
                "collision_aug_count": int(n_collision_aug),
                "collision_aug_attempts": int(n_collision_attempt),
                "low_overlap_step_count": int(n_low_overlap_samples),
                **val_m, **rollout_m, **avg,
            })

            ckpt_data = {
                "epoch":          epoch,
                "stage":          stage_lbl,
                "vca_state_dict": vca_layer.state_dict(),
                "val_score":      vs,
                "tf_iou_e0":      val_m["tf_iou_e0"],
                "tf_iou_e1":      val_m["tf_iou_e1"],
                "blend_sep":      val_m["blend_sep"],
                "inject_keys":    inject_keys,
                "adapter_rank":   args.adapter_rank,
                "lora_rank":      args.lora_rank,
                "procs_state": [
                    {
                        "slot_blend_raw":     p.slot_blend_raw.detach().cpu(),
                        "slot0_adapter":      p.slot0_adapter.state_dict(),
                        "slot1_adapter":      p.slot1_adapter.state_dict(),
                        "blend_head":         p.blend_head.state_dict(),
                        "lora_k":             p.lora_k.state_dict(),
                        "lora_v":             p.lora_v.state_dict(),
                        "lora_out":           p.lora_out.state_dict(),
                        "weight_head":        p.weight_head.state_dict(),
                        "ref_proj_e0":        p.ref_proj_e0.state_dict(),
                        "ref_proj_e1":        p.ref_proj_e1.state_dict(),
                        "overlap_blend_head": p.overlap_blend_head.state_dict(),
                        "occ_head_e0": (p.occ_head_e0.state_dict()
                                        if p.occ_head_e0 is not None else {}),
                        "occ_head_e1": (p.occ_head_e1.state_dict()
                                        if p.occ_head_e1 is not None else {}),
                    }
                    for p in procs
                ],
            }
            torch.save(ckpt_data, str(save_dir / "latest.pt"))
            if vs > best_val_score:
                best_val_score = vs; best_epoch = epoch
                torch.save(ckpt_data, str(save_dir / "best.pt"))
                print(f"  ★ best epoch={best_epoch} val_score={vs:.4f} "
                      f"blend_sep={val_m['blend_sep']:.4f}  "
                      f"occ_prod_sep={val_m['occ_prod_sep']:.4f}  "
                      f"→ {save_dir}/best.pt", flush=True)

            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            vca_layer.train(); manager.train()

    print(f"\n[Phase 46] 완료. best epoch={best_epoch} val_score={best_val_score:.4f}",
          flush=True)
    print(f"  성공 기준: blend_sep ≥ 0, blend_ov > blend_ex, occ_prod_sep > 0", flush=True)


# =============================================================================
# argparse
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Phase 46: Contrastive Occupancy + Direct Blend Ordering")

    p.add_argument("--ckpt",       type=str,
                   default="checkpoints/phase45_collisionfocus_full_v2/best.pt")
    p.add_argument("--data-root",  type=str, default="toy/data_objaverse")
    p.add_argument("--save-dir",   type=str, default="checkpoints/phase46")
    p.add_argument("--debug-dir",  type=str, default="outputs/phase46_debug")

    p.add_argument("--stage-a-epochs",     type=int,   default=DEFAULT_STAGE_A_EPOCHS)
    p.add_argument("--stage-b-epochs",     type=int,   default=DEFAULT_STAGE_B_EPOCHS)
    p.add_argument("--steps-per-epoch",    type=int,   default=DEFAULT_STEPS_PER_EPOCH)
    p.add_argument("--n-frames",           type=int,   default=8)
    p.add_argument("--n-steps",            type=int,   default=20)
    p.add_argument("--t-max",              type=int,   default=300)
    p.add_argument("--height",             type=int,   default=256)
    p.add_argument("--width",              type=int,   default=256)
    p.add_argument("--adapter-rank",       type=int,   default=DEFAULT_ADAPTER_RANK)
    p.add_argument("--lora-rank",          type=int,   default=DEFAULT_LORA_RANK)
    p.add_argument("--slot-blend",         type=float, default=DEFAULT_SLOT_BLEND)
    p.add_argument("--inject-keys",        type=str,   default=None)

    p.add_argument("--lr-vca",         type=float, default=DEFAULT_LR_VCA)
    p.add_argument("--lr-adapter",     type=float, default=DEFAULT_LR_ADAPTER)
    p.add_argument("--lr-lora",        type=float, default=DEFAULT_LR_LORA)
    p.add_argument("--lr-blend",       type=float, default=DEFAULT_LR_BLEND)
    p.add_argument("--lr-weight-head", type=float, default=DEFAULT_LR_WEIGHT_HEAD)
    p.add_argument("--lr-projector",   type=float, default=DEFAULT_LR_PROJECTOR)
    p.add_argument("--lr-obh",         type=float, default=DEFAULT_LR_OBH)
    p.add_argument("--lr-occ",         type=float, default=DEFAULT_LR_OCC)

    # Stage A losses
    p.add_argument("--la-occ",          type=float, default=DEFAULT_LA_OCC)
    p.add_argument("--la-occ-struct",   type=float, default=DEFAULT_LA_OCC_STRUCT)
    p.add_argument("--la-occ-contrast", type=float, default=DEFAULT_LA_OCC_CONTRAST)
    p.add_argument("--la-blend-target", type=float, default=DEFAULT_LA_BLEND_TARGET)
    p.add_argument("--la-blend-rank",   type=float, default=DEFAULT_LA_BLEND_RANK)
    p.add_argument("--la-blend-order",  type=float, default=DEFAULT_LA_BLEND_ORDER)
    p.add_argument("--la-vis",          type=float, default=DEFAULT_LA_VIS)
    p.add_argument("--la-vis-iou",      type=float, default=DEFAULT_LA_VIS_IOU)
    p.add_argument("--la-wrong",        type=float, default=DEFAULT_LA_WRONG)
    p.add_argument("--la-excl",         type=float, default=DEFAULT_LA_EXCL)
    p.add_argument("--la-slot-ref",     type=float, default=DEFAULT_LA_SLOT_REF)
    p.add_argument("--la-slot-cont",    type=float, default=DEFAULT_LA_SLOT_CONT)
    p.add_argument("--la-w-res",        type=float, default=DEFAULT_LA_W_RES)

    # Stage B losses
    p.add_argument("--lb-occ",          type=float, default=DEFAULT_LB_OCC)
    p.add_argument("--lb-occ-struct",   type=float, default=DEFAULT_LB_OCC_STRUCT)
    p.add_argument("--lb-occ-contrast", type=float, default=DEFAULT_LB_OCC_CONTRAST)
    p.add_argument("--lb-blend-order",  type=float, default=DEFAULT_LB_BLEND_ORDER)
    p.add_argument("--lb-vis",          type=float, default=DEFAULT_LB_VIS)
    p.add_argument("--lb-vis-iou",      type=float, default=DEFAULT_LB_VIS_IOU)
    p.add_argument("--lb-wrong",        type=float, default=DEFAULT_LB_WRONG)
    p.add_argument("--lb-sigma",        type=float, default=DEFAULT_LB_SIGMA)
    p.add_argument("--lb-depth",        type=float, default=DEFAULT_LB_DEPTH)
    p.add_argument("--lb-ov",           type=float, default=DEFAULT_LB_OV)
    p.add_argument("--lb-excl",         type=float, default=DEFAULT_LB_EXCL)
    p.add_argument("--lb-slot-ref",     type=float, default=DEFAULT_LB_SLOT_REF)
    p.add_argument("--lb-slot-cont",    type=float, default=DEFAULT_LB_SLOT_CONT)
    p.add_argument("--lb-blend-target", type=float, default=DEFAULT_LB_BLEND_TARGET)
    p.add_argument("--lb-blend-rank",   type=float, default=DEFAULT_LB_BLEND_RANK)
    p.add_argument("--lb-w-res",        type=float, default=DEFAULT_LB_W_RES)

    # Phase 46-specific
    p.add_argument("--occ-contrastive-margin", type=float, default=0.50)
    p.add_argument("--blend-order-margin",     type=float, default=0.10)
    p.add_argument("--rollout-every",          type=int,   default=1,
                   help="Compute rollout every N eval epochs (1=every eval, 2=every other, etc.)")

    p.add_argument("--val-frac",          type=float, default=VAL_FRAC)
    p.add_argument("--eval-every",        type=int,   default=5)
    p.add_argument("--eval-every-stage-a", type=int,  default=1)
    p.add_argument("--eval-seed",         type=int,   default=42)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--stage-a-slot-bootstrap", dest="stage_a_slot_bootstrap",
                   action="store_true")
    p.add_argument("--no-stage-a-slot-bootstrap", dest="stage_a_slot_bootstrap",
                   action="store_false")
    p.set_defaults(stage_a_slot_bootstrap=False)
    p.add_argument("--use-soft-masks", dest="use_soft_masks", action="store_true")
    p.add_argument("--no-use-soft-masks", dest="use_soft_masks", action="store_false")
    p.set_defaults(use_soft_masks=False)
    p.add_argument("--collision-aug-prob",         type=float, default=DEFAULT_COLLISION_AUG_PROB)
    p.add_argument("--collision-aug-prob-stage-a", type=float,
                   default=DEFAULT_COLLISION_AUG_PROB_STAGE_A)
    p.add_argument("--collision-aug-prob-stage-b", type=float,
                   default=DEFAULT_COLLISION_AUG_PROB_STAGE_B)
    p.add_argument("--low-overlap-force-prob", type=float, default=DEFAULT_LOW_OVERLAP_FORCE_PROB)
    p.add_argument("--low-overlap-thr",        type=float, default=DEFAULT_LOW_OVERLAP_THR)
    p.add_argument("--collision-ov-min",         type=float, default=DEFAULT_COLLISION_OV_MIN)
    p.add_argument("--collision-ov-max",         type=float, default=DEFAULT_COLLISION_OV_MAX)
    p.add_argument("--collision-ov-min-stage-a", type=float, default=DEFAULT_COLLISION_OV_MIN_STAGE_A)
    p.add_argument("--collision-ov-max-stage-a", type=float, default=DEFAULT_COLLISION_OV_MAX_STAGE_A)
    p.add_argument("--collision-min-accept-overlap", type=float,
                   default=DEFAULT_COLLISION_MIN_ACCEPT_OVERLAP)
    p.add_argument("--mask-dilate-radius",              type=int,   default=DEFAULT_MASK_DILATE_RADIUS)
    p.add_argument("--collision-front-swap-prob-stage-a", type=float,
                   default=DEFAULT_COLLISION_FRONT_SWAP_PROB_STAGE_A)
    p.add_argument("--collision-front-swap-prob-stage-b", type=float,
                   default=DEFAULT_COLLISION_FRONT_SWAP_PROB_STAGE_B)
    p.add_argument("--train-overlap-boost", type=float, default=DEFAULT_TRAIN_OVERLAP_BOOST)
    p.add_argument("--train-overlap-clip",  type=float, default=DEFAULT_TRAIN_OVERLAP_CLIP)
    p.add_argument("--val-overlap-boost",   type=float, default=DEFAULT_VAL_OVERLAP_BOOST)
    p.add_argument("--val-overlap-clip",    type=float, default=DEFAULT_VAL_OVERLAP_CLIP)
    p.add_argument("--val-collision-thr",   type=float, default=DEFAULT_VAL_COLLISION_THR)
    p.add_argument("--val-collision-mix",   type=float, default=DEFAULT_VAL_COLLISION_MIX)

    args = p.parse_args()
    train_phase46(args)


if __name__ == "__main__":
    main()
