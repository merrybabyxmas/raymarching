"""
Phase 53 — Explicit Decomposition Heads Training
==================================================

Phase 46-52 실패 원인 요약:
  - OBH + occ_head 방식 7번 실패: blend_sep/occ_prod_sep 항상 음수
  - 근본 원인: identical entities (twin knights)에서 F0 ≈ F1,
    w0*w1 proxy가 실제 데이터에서 성립하지 않음 (exclusive에서 더 큼)

Phase 53 변경:
  - DecompositionHeads: p0, p1, pov, pfront 별도 head
  - 결정론적 compositing: relu(p0-pov), pov*pfront 등
  - blend_map = pov → blend_sep > 0 by construction (pov 학습 후)
  - l_decomp_ov: BCE(pov, m0*m1) 직접 감독
  - l_decomp_hier: pov ≤ min(p0, p1) 계층 제약
  - l_decomp_front: pfront vs depth_order CE

베이스 checkpoint: checkpoints/phase52/best.pt
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
from models.entity_slot_phase43 import PRIMARY_DIM
from models.entity_slot_phase44 import collect_blend_stats_detailed, l_blend_rank
from models.entity_slot_phase45 import (
    l_occupancy,
    l_occupancy_structure,
    l_blend_target_balanced,
    l_visible_weights_region_balanced,
    l_visible_iou_soft,
    collect_occupancy_stats,
)
from models.entity_slot_phase46 import (
    val_score_phase46,
)
from models.entity_slot_phase53 import (
    Phase53Processor,
    MultiBlockSlotManagerP53,
    inject_multi_block_entity_slot_p53,
    restore_multiblock_state_p53,
    l_decomp_occ,
    l_decomp_ov,
    l_decomp_hier,
    l_decomp_front,
    l_decomp_presence,
    l_pair_identity_preservation,
    l_solo_masked_reconstruction,
    l_route_entropy,
    l_output_entity_divergence,
    decompose_entity_weights,
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
from scripts.prompt_identity import make_identity_prompts
from scripts.prompt_identity import make_color_prompts
from scripts.train_phase35 import (
    DEFAULT_Z_BINS,
    get_entity_token_positions,
)
from scripts.train_phase39 import (
    compute_dataset_overlap_scores,
    split_train_val,
    make_sampling_weights,
)
from scripts.train_phase46 import (
    _make_overlap_frame_weights,
    _frame_overlap_ratios_from_masks,
    _dilate_entity_masks,
    _try_collision_augment,
    _get_soft_masks_for_sample,
    _load_soft_masks_from_seq_dir,
)
from scripts.generate_solo_renders import ObjaverseDatasetPhase40
from scripts.generate_solo_renders import compute_visible_masks_np
from scripts.train_phase43 import (
    _get_solo_entity_token_positions,
    extract_entity_prompt_ref,
    _get_block_feat,
    _resize_feat,
    ROLLOUT_T_START,
    ROLLOUT_N_STEPS,
)


def _weighted_np_mean(values: List[float], weights: Optional[np.ndarray] = None) -> float:
    """Local weighted mean helper to keep Phase53 self-contained."""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(arr.mean())
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != arr.shape[0]:
        raise ValueError(f"weights length mismatch: {w.shape[0]} vs {arr.shape[0]}")
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return float(arr.mean())
    return float(np.sum(arr * w) / w_sum)


def _downsample_entity_masks_to_latent(
    masks_t: torch.Tensor,
    latent_hw: Tuple[int, int],
) -> torch.Tensor:
    """Resize (T, 2, S) masks to latent spatial resolution (T, 2, h, w)."""
    T, _, S = masks_t.shape
    H = int(S ** 0.5)
    m4d = masks_t.view(T, 2, H, H)
    m4d = torch.nn.functional.interpolate(
        m4d.float(), size=latent_hw, mode="nearest")
    return m4d


def _predict_original_sample(
    pipe,
    noisy: torch.Tensor,
    t_tensor: torch.Tensor,
    noise_pred: torch.Tensor,
) -> torch.Tensor:
    """Recover x0 from a DDIM-style scheduler output, with a fallback formula."""
    try:
        out = pipe.scheduler.step(noise_pred, t_tensor, noisy, return_dict=True)
        pred = getattr(out, "pred_original_sample", None)
        if pred is not None:
            return pred
    except Exception:
        pass

    # Fallback for epsilon-prediction DDIM.
    if hasattr(pipe.scheduler, "alphas_cumprod"):
        t_idx = int(t_tensor.item())
        alpha_t = pipe.scheduler.alphas_cumprod[t_idx].to(noisy.device, noisy.dtype)
        while alpha_t.dim() < noisy.dim():
            alpha_t = alpha_t.view(*([1] * noisy.dim()))
        return (noisy - torch.sqrt(1.0 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

    return noisy - noise_pred


def _wrap_attn_processors_debug(pipe, prefix: str = "[Phase53 debug]"):
    """Wrap all UNet attention processors to print one-time shape traces."""
    import os

    if not os.environ.get("PHASE53_DEBUG_SHAPES"):
        return

    wrapped = {}

    class _Wrapper:
        def __init__(self, key, proc):
            self.key = key
            self.proc = proc
            self.seen = False

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, temb=None, **kwargs):
            if not self.seen:
                hs_shape = tuple(hidden_states.shape) if hasattr(hidden_states, "shape") else None
                enc_shape = tuple(encoder_hidden_states.shape) if (
                    encoder_hidden_states is not None and hasattr(encoder_hidden_states, "shape")
                ) else None
                print(f"{prefix} key={self.key} proc={type(self.proc).__name__} "
                      f"hidden={hs_shape} enc={enc_shape}", flush=True)
                self.seen = True
            out = self.proc(
                attn, hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                **kwargs,
            )
            if (hasattr(out, "shape") and hasattr(hidden_states, "shape")
                    and out.shape != hidden_states.shape
                    and out.numel() == hidden_states.numel()):
                out = out.reshape(hidden_states.shape)
            return out

    for key, proc in pipe.unet.attn_processors.items():
        wrapped[key] = _Wrapper(key, proc)
    pipe.unet.set_attn_processor(wrapped)

# soft target params (overlap front/back)
_SOFT_FRONT_VAL = 0.90
_SOFT_BACK_VAL  = 0.05

# ─── Phase 53 하이퍼파라미터 ──────────────────────────────────────────────────
DEFAULT_STAGE_A_EPOCHS   = 6
DEFAULT_STAGE_B_EPOCHS   = 14
DEFAULT_STEPS_PER_EPOCH  = 20

# Stage A (decomp heads focus)
DEFAULT_LA_DECOMP_OCC   = 5.0    # region-aware occ for p0, p1
DEFAULT_LA_DECOMP_OV    = 5.0    # BCE(pov, m0*m1)
DEFAULT_LA_DECOMP_HIER  = 2.0    # hierarchy: pov ≤ min(p0,p1)
DEFAULT_LA_DECOMP_FRONT = 3.0    # pfront vs depth_order CE
DEFAULT_LA_DECOMP_PRES = 2.0     # union presence gate BCE
DEFAULT_LA_VIS          = 0.5    # visible weight region-balanced
DEFAULT_LA_VIS_IOU      = 0.0
DEFAULT_LA_WRONG        = 0.2
DEFAULT_LA_EXCL         = 0.3
DEFAULT_LA_SLOT_REF     = 0.8
DEFAULT_LA_SLOT_CONT    = 0.2
DEFAULT_LA_PAIR_ID      = 0.8
DEFAULT_LA_W_RES        = 0.01
DEFAULT_LA_ROUTE_ENT    = 0.05
DEFAULT_LA_OUT_DIVERGE  = 0.5     # entity output divergence loss

# Stage B (full finetuning)
DEFAULT_LB_DECOMP_OCC   = 3.0
DEFAULT_LB_DECOMP_OV    = 3.0
DEFAULT_LB_DECOMP_HIER  = 1.5
DEFAULT_LB_DECOMP_FRONT = 2.0
DEFAULT_LB_DECOMP_PRES  = 1.5
DEFAULT_LB_VIS          = 2.0
DEFAULT_LB_VIS_IOU      = 0.0
DEFAULT_LB_WRONG        = 1.0
DEFAULT_LB_SIGMA        = 1.5
DEFAULT_LB_DEPTH        = 2.0
DEFAULT_LB_OV           = 0.5
DEFAULT_LB_EXCL         = 0.5
DEFAULT_LB_SLOT_REF     = 1.5
DEFAULT_LB_SLOT_CONT    = 0.3
DEFAULT_LB_PAIR_ID      = 1.5
DEFAULT_LB_W_RES        = 0.01
DEFAULT_LB_ROUTE_ENT    = 0.10
DEFAULT_LB_OUT_DIVERGE  = 0.5
DEFAULT_LA_NOISE        = 0.05
DEFAULT_LB_NOISE        = 0.10
DEFAULT_LA_SOLO_NOISE   = 0.10
DEFAULT_LB_SOLO_NOISE   = 0.20
DEFAULT_LA_SOLO_MASK_RECON = 0.00
DEFAULT_LB_SOLO_MASK_RECON = 0.00
DEFAULT_PAIR_MARGIN     = 0.15
DEFAULT_PAIR_TARGET     = 0.20

# LR
DEFAULT_LR_VCA          = 2e-5
DEFAULT_LR_ADAPTER      = 1e-4
DEFAULT_LR_LORA         = 5e-5
DEFAULT_LR_BLEND        = 3e-4
DEFAULT_LR_WEIGHT_HEAD  = 1e-4
DEFAULT_LR_PROJECTOR    = 5e-5
DEFAULT_LR_DECOMP       = 1e-3    # NEW: decomp heads lr (similar to occ lr)

DEFAULT_ADAPTER_RANK    = 64
DEFAULT_LORA_RANK       = 4
DEFAULT_SLOT_BLEND      = 0.3
VAL_FRAC                = 0.2
DEFAULT_COLLISION_AUG_PROB = 0.2
DEFAULT_COLLISION_AUG_PROB_STAGE_A = 0.80   # Higher for Phase53
DEFAULT_COLLISION_AUG_PROB_STAGE_B = 0.45
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


# =============================================================================
# Stage A/B parameter control
# =============================================================================

def _set_requires_grad_stage_a(manager, vca_layer, procs, slot_bootstrap: bool = False):
    """Stage A: decomp_heads + weight_head + slot_blend_raw (+ optional adapters)."""
    for p in vca_layer.parameters():   p.requires_grad_(False)
    for proc in procs:
        for p in proc.parameters():    p.requires_grad_(False)
    for proc in procs:
        # Decomp heads (Phase53 new)
        if hasattr(proc, 'decomp_heads') and proc.decomp_heads is not None:
            for p in proc.decomp_heads.parameters(): p.requires_grad_(True)
        # Weight head (for l_w_res)
        for p in proc.weight_head.parameters():       p.requires_grad_(True)
        # Entity-specific K/V/Out LoRA
        for elora in ('lora_k_e0', 'lora_k_e1', 'lora_v_e0', 'lora_v_e1',
                       'lora_out_e0', 'lora_out_e1'):
            if hasattr(proc, elora):
                for p in getattr(proc, elora).parameters(): p.requires_grad_(True)
        # Slot blend
        proc.slot_blend_raw.requires_grad_(True)
        if slot_bootstrap:
            for p in proc.slot0_adapter.parameters(): p.requires_grad_(True)
            for p in proc.slot1_adapter.parameters(): p.requires_grad_(True)
            for p in proc.ref_proj_e0.parameters():   p.requires_grad_(True)
            for p in proc.ref_proj_e1.parameters():   p.requires_grad_(True)
    lbl = "decomp_heads + weight_head + slot_blend_raw"
    if slot_bootstrap:
        lbl += " + adapters + projectors"
    print(f"  [stage A] {lbl}", flush=True)


def _set_requires_grad_stage_b(manager, vca_layer, procs):
    """Stage B: full joint finetuning."""
    for p in vca_layer.parameters():   p.requires_grad_(True)
    for proc in procs:
        for p in proc.parameters():    p.requires_grad_(True)
    print("  [stage B] 전체 파라미터 학습 (joint finetuning)", flush=True)


# =============================================================================
# Rollout evaluation
# =============================================================================

@torch.no_grad()
def evaluate_rollout_p53(
    pipe,
    manager:    MultiBlockSlotManagerP53,
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
            if enc_hs.shape[0] == 1 and latents.shape[2] > 1:
                pass  # AnimateDiff handles frame broadcasting internally

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
        "rollout_iou_e0":             _weighted_np_mean(rollout_iou_e0s, w),
        "rollout_iou_e1":             _weighted_np_mean(rollout_iou_e1s, w),
        "rollout_collision_iou_e0":   roll_col_e0,
        "rollout_collision_iou_e1":   roll_col_e1,
        "rollout_collision_frac":     col_frac,
    }


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def evaluate_val_set_phase53(
    pipe,
    manager:    MultiBlockSlotManagerP53,
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
            if enc_hs.shape[0] == 1 and latents.shape[2] > 1:
                pass  # AnimateDiff handles frame broadcasting internally

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

                # blend_sep: stored as pov in last_blend_map
                bm_raw = getattr(manager.primary, 'last_blend_map', None)
                if (bm_raw is not None and
                        isinstance(bm_raw, torch.Tensor) and
                        bm_raw.dim() >= 2 and fi < bm_raw.shape[0]):
                    bm_f   = bm_raw[fi:fi+1].float()
                    b_stat = collect_blend_stats_detailed(bm_f, m_f)
                    blend_stats_list.append(b_stat)

                # occ stats: p0, p1 stored as last_o0, last_o1
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
    ord_np    = np.asarray(ord_accs,  dtype=np.float64)
    wrong_np  = np.asarray(wrong_leaks, dtype=np.float64)

    c_iou_e0 = _weighted_np_mean(iou_e0_np[col_mask].tolist(), col_w) if np.any(col_mask) else 0.0
    c_iou_e1 = _weighted_np_mean(iou_e1_np[col_mask].tolist(), col_w) if np.any(col_mask) else 0.0
    c_ord     = _weighted_np_mean(ord_np[col_mask].tolist(),    col_w) if np.any(col_mask) else 0.0
    c_wrong   = _weighted_np_mean(wrong_np[col_mask].tolist(),  col_w) if np.any(col_mask) else 1.0
    c_frac    = float(np.mean(col_mask.astype(np.float64)))

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

def train_phase53(args):
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

    print("[Phase 53] 파이프라인 로드 중...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    print(f"[Phase 53] checkpoint 로드: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_has_decomp_heads = any(
        bool(ps.get("decomp_heads"))
        for ps in ckpt.get("procs_state", [])
        if isinstance(ps, dict)
    )
    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM, context_dim=768,
        n_heads=8, n_entities=2, z_bins=DEFAULT_Z_BINS, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=DEPTH_PE_INIT_SCALE,
    ).to(device)
    vca_layer.load_state_dict(ckpt["vca_state_dict"])
    print("  VCA loaded", flush=True)

    print("[Phase 53] 데이터셋 로드 중...", flush=True)
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
    procs, orig_procs = inject_multi_block_entity_slot_p53(
        pipe, vca_layer, init_ctx,
        inject_keys     = inject_keys,
        slot_blend_init = args.slot_blend,
        adapter_rank    = args.adapter_rank,
        lora_rank       = args.lora_rank,
        use_blend_head  = True,
        decomp_proj_dim = args.decomp_proj_dim,
        decomp_hidden   = args.decomp_hidden,
    )
    for p in procs:
        p.to(device)

    manager = MultiBlockSlotManagerP53(procs, inject_keys, primary_idx=1)
    _wrap_attn_processors_debug(pipe)

    print("[Phase 53] checkpoint 복원...", flush=True)
    restore_multiblock_state_p53(manager, ckpt, device=device)

    # Fresh Phase52/old ckpt starts with zero-init decomp heads.
    # Phase53 ckpt resume should keep the learned decomp heads that were loaded.
    if args.reset_decomp_heads and not ckpt_has_decomp_heads:
        print("[Phase 53] decomp_heads last layer zero-init 확인...", flush=True)
        for i, proc in enumerate(procs):
            if hasattr(proc, 'decomp_heads') and proc.decomp_heads is not None:
                for head_name in ('p0_head', 'p1_head', 'pov_head', 'pfront_head'):
                    head = getattr(proc.decomp_heads, head_name, None)
                    if head is not None:
                        torch.nn.init.zeros_(head[-1].weight)
                        torch.nn.init.zeros_(head[-1].bias)
                print(f"  block[{i}] decomp_heads: zero-init → all probs = 0.5", flush=True)
    elif ckpt_has_decomp_heads:
        print("[Phase 53] decomp_heads already present in ckpt → keep loaded weights", flush=True)

    for p in pipe.unet.parameters():         p.requires_grad_(False)
    for p in pipe.text_encoder.parameters(): p.requires_grad_(False)
    for p in pipe.vae.parameters():          p.requires_grad_(False)

    # ── Epoch 0 no-op validation ─────────────────────────────────────────
    print("\n[Phase 53] Epoch 0 no-op validation...", flush=True)
    vca_layer.eval(); manager.eval()
    val_m0 = evaluate_val_set_phase53(
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
          f"prod_sep={val_m0['occ_prod_sep']:.4f}  "
          f"val_score={val_m0['val_score']:.4f}", flush=True)
    vca_layer.train(); manager.train()

    history        = []
    best_val_score = -1.0
    best_epoch     = -1
    total_epochs   = args.stage_a_epochs + args.stage_b_epochs

    for epoch in range(total_epochs):
        is_stage_a = (epoch < args.stage_a_epochs)

        if epoch == 0 and args.stage_a_epochs > 0:
            print(f"\n[Phase 53] Stage A 시작 ({args.stage_a_epochs} epochs): "
                  f"DecompositionHeads 학습", flush=True)
            _set_requires_grad_stage_a(
                manager, vca_layer, procs,
                slot_bootstrap=args.stage_a_slot_bootstrap)
            stage_a_groups = [
                {"params": manager.decomp_head_params(),
                 "lr": args.lr_decomp, "name": "decomp_heads"},
                {"params": manager.entity_lora_kv_params(),
                 "lr": args.lr_lora, "name": "entity_lora_kv"},
                {"params": manager.entity_lora_out_params(),
                 "lr": args.lr_lora, "name": "entity_lora_out"},
                {"params": manager.weight_head_params(),
                 "lr": args.lr_weight_head, "name": "weight_head"},
                {"params": [p.slot_blend_raw for p in procs],
                 "lr": args.lr_blend, "name": "blend_raw"},
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
                optimizer, T_max=max(1, args.stage_a_epochs), eta_min=1e-6)

        elif epoch == 0 and args.stage_a_epochs == 0:
            print(f"\n[Phase 53] Stage B 시작 ({args.stage_b_epochs} epochs): "
                  f"전체 미세조정 (decomp_heads at lr/5)", flush=True)
            _set_requires_grad_stage_b(manager, vca_layer, procs)
            optimizer = optim.AdamW([
                {"params": list(vca_layer.parameters()),
                 "lr": args.lr_vca * 0.1,   "name": "vca"},
                {"params": manager.adapter_params(),
                 "lr": args.lr_adapter,       "name": "adapters"},
                {"params": manager.lora_params(),
                 "lr": args.lr_lora,          "name": "lora"},
                {"params": manager.blend_params(),
                 "lr": args.lr_blend,         "name": "blend"},
                {"params": manager.weight_head_params(),
                 "lr": args.lr_weight_head * 0.1, "name": "weight_head"},
                {"params": manager.projector_params(),
                 "lr": args.lr_projector,     "name": "projectors"},
                # decomp_heads at lr/5 (preserve Stage A progress)
                {"params": manager.decomp_head_params(),
                 "lr": args.lr_decomp * 0.2, "name": "decomp_heads"},
                {"params": manager.entity_lora_kv_params(),
                 "lr": args.lr_lora * 0.5, "name": "entity_lora_kv"},
                {"params": manager.entity_lora_out_params(),
                 "lr": args.lr_lora * 0.5, "name": "entity_lora_out"},
            ], weight_decay=1e-4)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.stage_b_epochs), eta_min=args.lr_vca * 0.05)

        elif epoch == args.stage_a_epochs:
            print(f"\n[Phase 53] Stage B 시작 ({args.stage_b_epochs} epochs): "
                  f"전체 미세조정 (decomp_heads at lr/5)", flush=True)
            _set_requires_grad_stage_b(manager, vca_layer, procs)
            optimizer = optim.AdamW([
                {"params": list(vca_layer.parameters()),
                 "lr": args.lr_vca * 0.1,   "name": "vca"},
                {"params": manager.adapter_params(),
                 "lr": args.lr_adapter,       "name": "adapters"},
                {"params": manager.lora_params(),
                 "lr": args.lr_lora,          "name": "lora"},
                {"params": manager.blend_params(),
                 "lr": args.lr_blend,         "name": "blend"},
                {"params": manager.weight_head_params(),
                 "lr": args.lr_weight_head * 0.1, "name": "weight_head"},
                {"params": manager.projector_params(),
                 "lr": args.lr_projector,     "name": "projectors"},
                # decomp_heads at lr/5 (preserve Stage A progress)
                {"params": manager.decomp_head_params(),
                 "lr": args.lr_decomp * 0.2, "name": "decomp_heads"},
                {"params": manager.entity_lora_kv_params(),
                 "lr": args.lr_lora * 0.5, "name": "entity_lora_kv"},
                {"params": manager.entity_lora_out_params(),
                 "lr": args.lr_lora * 0.5, "name": "entity_lora_out"},
            ], weight_decay=1e-4)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.stage_b_epochs), eta_min=args.lr_vca * 0.05)

        vca_layer.train(); manager.train()

        if is_stage_a:
            loss_keys = ["total",
                         "decomp_occ", "decomp_ov", "decomp_hier", "decomp_front", "decomp_pres",
                         "vis", "vis_iou", "wrong", "excl", "slot_ref", "slot_cont", "pair_id", "noise", "solo_noise", "solo_mask_recon", "w_res",
                         "route_ent", "out_div",
                         "collision_frac"]
        else:
            loss_keys = ["total",
                         "decomp_occ", "decomp_ov", "decomp_hier", "decomp_front", "decomp_pres",
                         "vis", "vis_iou", "wrong", "sigma", "depth", "ov",
                         "excl", "slot_ref", "slot_cont", "pair_id", "noise", "solo_noise", "solo_mask_recon", "w_res",
                         "route_ent", "out_div",
                         "collision_frac"]
        epoch_losses = {k: [] for k in loss_keys}

        chosen     = np.random.choice(len(train_idx), size=args.steps_per_epoch,
                                      replace=True, p=sample_weights)
        step_indices = [train_idx[ci] for ci in chosen]
        n_collision_attempt = 0
        n_collision_aug     = 0
        n_low_overlap_samples = 0
        collision_fracs_step  = []

        for batch_idx, data_idx in enumerate(step_indices):
            sample = dataset[data_idx]
            if len(sample) >= 8:
                frames_np, _, depth_orders, meta, entity_masks, visible_masks_sample, solo_e0, solo_e1 = sample
            else:
                frames_np, _, depth_orders, meta, entity_masks = sample[:5]
                visible_masks_sample = None
                solo_e0 = solo_e1 = None

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

            visible_masks_np = None
            if len(sample) >= 8:
                visible_masks_np = compute_visible_masks_np(
                    entity_masks.astype(np.float32), depth_orders)
                if visible_masks_sample is not None and not used_collision_aug:
                    # Keep the dataset-provided visible masks when no collision
                    # augmentation changed the sample; they are usually identical.
                    visible_masks_np = visible_masks_sample.astype(np.float32)

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

            T_frames     = min(frames_np.shape[0], entity_masks.shape[0])
            masks_t_hard = torch.from_numpy(entity_masks[:T_frames].astype(np.float32)).to(device)
            masks_t      = torch.from_numpy(train_masks[:T_frames].astype(np.float32)).to(device)
            depth_orders_t = depth_orders[:T_frames]

            if args.mask_dilate_radius > 0:
                masks_t = _dilate_entity_masks(masks_t, args.mask_dilate_radius)
                masks_overlap_t = _dilate_entity_masks(masks_t_hard, args.mask_dilate_radius)
            else:
                masks_overlap_t = masks_t_hard

            frame_overlap_t = _frame_overlap_ratios_from_masks(masks_overlap_t)
            frame_w_t = _make_overlap_frame_weights(
                frame_overlap_t, args.train_overlap_boost, args.train_overlap_clip)

            # Track collision fraction for logging
            step_col_frac = float(frame_overlap_t.mean().item())
            collision_fracs_step.append(step_col_frac)

            # ── Entity ref forwards ──────────────────────────────────────
            F0_refs: Dict[int, Optional[torch.Tensor]] = {}
            F1_refs: Dict[int, Optional[torch.Tensor]] = {}
            F0_refs_solo: Dict[int, Optional[torch.Tensor]] = {}
            F1_refs_solo: Dict[int, Optional[torch.Tensor]] = {}
            need_refs = (
                args.stage_a_slot_bootstrap
                or args.la_slot_ref > 0.0
                or args.lb_slot_ref > 0.0
                or args.la_slot_cont > 0.0
                or args.lb_slot_cont > 0.0
                or args.la_pair_id > 0.0
                or args.lb_pair_id > 0.0
                or args.la_noise > 0.0
                or args.lb_noise > 0.0
                or args.la_solo_noise > 0.0
                or args.lb_solo_noise > 0.0
            )
            if need_refs:
                prompt_e0, prompt_e1, full_prompt, _, _ = make_identity_prompts(meta)
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

                if len(sample) >= 8 and solo_e0 is not None and solo_e1 is not None:
                    # Use actual solo renders as semantic identity teachers.
                    # This is the strongest available "cat is cat, dog is dog"
                    # supervision path in the current dataset.
                    try:
                        with torch.no_grad():
                            solo0_lat = encode_frames_to_latents(pipe, solo_e0, device)
                            solo1_lat = encode_frames_to_latents(pipe, solo_e1, device)
                            solo0_noisy = pipe.scheduler.add_noise(solo0_lat, noise, t)
                            solo1_noisy = pipe.scheduler.add_noise(solo1_lat, noise, t)

                        F0_refs_solo = extract_entity_prompt_ref(
                            pipe, manager, solo0_noisy, t, prompt_e0, device, entity_idx=0)
                        F1_refs_solo = extract_entity_prompt_ref(
                            pipe, manager, solo1_noisy, t, prompt_e1, device, entity_idx=1)
                    except Exception as e:
                        print(f"  [warn] solo-ref: {e}", flush=True)

                    if args.semantic_teacher_mix > 0.0:
                        try:
                            color_e0, color_e1, _, _, _ = make_color_prompts(meta)
                            F0_refs_color = extract_entity_prompt_ref(
                                pipe, manager, solo0_noisy, t, color_e0, device, entity_idx=0)
                            F1_refs_color = extract_entity_prompt_ref(
                                pipe, manager, solo1_noisy, t, color_e1, device, entity_idx=1)
                            mix = float(np.clip(args.semantic_teacher_mix, 0.0, 1.0))
                            for blk_idx in set(F0_refs_solo.keys()) | set(F1_refs_solo.keys()):
                                f0_id = F0_refs_solo.get(blk_idx, None)
                                f1_id = F1_refs_solo.get(blk_idx, None)
                                f0_cl = F0_refs_color.get(blk_idx, None)
                                f1_cl = F1_refs_color.get(blk_idx, None)
                                if f0_id is not None and f0_cl is not None:
                                    F0_refs_solo[blk_idx] = ((1.0 - mix) * f0_id + mix * f0_cl)
                                if f1_id is not None and f1_cl is not None:
                                    F1_refs_solo[blk_idx] = ((1.0 - mix) * f1_id + mix * f1_cl)
                            print(f"  [semantic teacher] color mix={mix:.2f}", flush=True)
                        except Exception as e:
                            print(f"  [warn] color-teacher mix: {e}", flush=True)

                    manager.set_entity_ctx(entity_ctx.float())
                    manager.set_entity_tokens(toks_e0, toks_e1)
                    manager.reset_slot_store()

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
                # NOTE: do NOT expand enc_hs here. AnimateDiff UNet handles
                # batch/frame broadcasting internally. Expanding breaks motion modules.

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = pipe.unet(noisy, t, encoder_hidden_states=enc_hs).sample
            l_noise = F.mse_loss(noise_pred.float(), noise.float())
            l_solo_noise = torch.tensor(0.0, device=device)
            l_solo_mask_recon = torch.tensor(0.0, device=device)

            if (
                len(sample) >= 8 and solo_e0 is not None and solo_e1 is not None
                and (
                    args.la_solo_noise > 0.0 or args.lb_solo_noise > 0.0
                    or args.la_solo_mask_recon > 0.0 or args.lb_solo_mask_recon > 0.0
                )
            ):
                try:
                    with torch.no_grad():
                        solo0_lat = encode_frames_to_latents(pipe, solo_e0, device)
                        solo1_lat = encode_frames_to_latents(pipe, solo_e1, device)
                    solo0_noise = torch.randn_like(solo0_lat)
                    solo1_noise = torch.randn_like(solo1_lat)
                    solo0_noisy = pipe.scheduler.add_noise(solo0_lat, solo0_noise, t)
                    solo1_noisy = pipe.scheduler.add_noise(solo1_lat, solo1_noise, t)

                    tok0 = pipe.tokenizer(
                        prompt_e0, return_tensors="pt", padding="max_length",
                        max_length=pipe.tokenizer.model_max_length, truncation=True,
                    ).to(device)
                    tok1 = pipe.tokenizer(
                        prompt_e1, return_tensors="pt", padding="max_length",
                        max_length=pipe.tokenizer.model_max_length, truncation=True,
                    ).to(device)
                    with torch.no_grad():
                        enc0 = pipe.text_encoder(**tok0).last_hidden_state.half()
                        enc1 = pipe.text_encoder(**tok1).last_hidden_state.half()

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred0 = pipe.unet(solo0_noisy, t, encoder_hidden_states=enc0).sample
                        pred1 = pipe.unet(solo1_noisy, t, encoder_hidden_states=enc1).sample
                    l_solo_noise = 0.5 * (
                        F.mse_loss(pred0.float(), solo0_noise.float())
                        + F.mse_loss(pred1.float(), solo1_noise.float())
                    )

                    if visible_masks_np is not None and (
                        args.la_solo_mask_recon > 0.0 or args.lb_solo_mask_recon > 0.0
                    ):
                        vis_t = torch.from_numpy(
                            visible_masks_np[:T_frames].astype(np.float32)
                        ).to(device)
                        vis_lat = _downsample_entity_masks_to_latent(
                            vis_t, solo0_lat.shape[-2:])
                        pred0_x0 = _predict_original_sample(pipe, solo0_noisy, t, pred0)
                        pred1_x0 = _predict_original_sample(pipe, solo1_noisy, t, pred1)

                        recon_sum = torch.tensor(0.0, device=device)
                        recon_w_sum = torch.tensor(0.0, device=device)
                        n_recon_frames = min(
                            solo0_lat.shape[2], solo1_lat.shape[2], vis_lat.shape[0], T_frames)
                        for fi in range(n_recon_frames):
                            fw = frame_w_t[fi]
                            m0 = vis_lat[fi, 0:1].unsqueeze(0)
                            m1 = vis_lat[fi, 1:2].unsqueeze(0)
                            if m0.sum() > 0:
                                recon_sum = recon_sum + l_solo_masked_reconstruction(
                                    pred0_x0[:, :, fi], solo0_lat[:, :, fi], m0) * fw
                                recon_w_sum = recon_w_sum + fw
                            if m1.sum() > 0:
                                recon_sum = recon_sum + l_solo_masked_reconstruction(
                                    pred1_x0[:, :, fi], solo1_lat[:, :, fi], m1) * fw
                                recon_w_sum = recon_w_sum + fw
                        if recon_w_sum.item() > 0.0:
                            l_solo_mask_recon = recon_sum / recon_w_sum
                except Exception as e:
                    print(f"  [warn] solo-noise: {e}", flush=True)

            BF    = manager.last_w0.shape[0] if manager.last_w0 is not None else 0
            T_use = min(BF, T_frames)

            primary = manager.primary

            # ── Decomp head targets — align mask resolution ───────────────
            p0_fl  = getattr(primary, 'last_p0_for_loss',     None)
            p1_fl  = getattr(primary, 'last_p1_for_loss',     None)
            pov_fl = getattr(primary, 'last_pov_for_loss',    None)
            pfront_fl = getattr(primary, 'last_pfront_for_loss', None)

            masks_occ = None
            if p0_fl is not None and isinstance(p0_fl, torch.Tensor):
                S_blk  = p0_fl.shape[1]
                S_mask = masks_t.shape[-1]
                if S_blk != S_mask:
                    H_b = int(S_blk**0.5); H_m = int(S_mask**0.5)
                    m4d = masks_t.view(masks_t.shape[0], 2, H_m, H_m)
                    m4d = torch.nn.functional.interpolate(
                        m4d.float(), size=(H_b, H_b), mode='nearest')
                    masks_occ = m4d.view(masks_t.shape[0], 2, S_blk)
                else:
                    masks_occ = masks_t

            # ── L_decomp_occ (p0, p1 vs m0, m1) ─────────────────────────
            l_d_occ  = torch.tensor(0.0, device=device)
            l_d_ov   = torch.tensor(0.0, device=device)
            l_d_hier = torch.tensor(0.0, device=device)
            l_d_front = torch.tensor(0.0, device=device)
            l_d_pres = torch.tensor(0.0, device=device)
            l_route_ent = torch.tensor(0.0, device=device)

            if (p0_fl is not None and p1_fl is not None and
                    pov_fl is not None and masks_occ is not None):
                n_occ_frames = min(p0_fl.shape[0], T_frames)
                occ_w_sum = torch.tensor(0.0, device=device)
                for fi in range(n_occ_frames):
                    fw = frame_w_t[fi]
                    p0_fi  = p0_fl[fi:fi+1].float()
                    p1_fi  = p1_fl[fi:fi+1].float()
                    pov_fi = pov_fl[fi:fi+1].float()
                    m_fi   = masks_occ[fi:fi+1]
                    do_fi  = [depth_orders_t[fi]] if fi < len(depth_orders_t) else [(0, 1)]

                    l_d_occ  = l_d_occ  + l_decomp_occ(p0_fi, p1_fi, m_fi) * fw
                    l_d_ov   = l_d_ov   + l_decomp_ov(pov_fi, m_fi) * fw
                    l_d_hier = l_d_hier + l_decomp_hier(p0_fi, p1_fi, pov_fi, m_fi) * fw

                    if pfront_fl is not None:
                        pfront_fi = pfront_fl[fi:fi+1].float()
                        l_d_front = l_d_front + l_decomp_front(pfront_fi, do_fi, m_fi) * fw

                    presence_fi = getattr(primary, "last_presence_for_loss", None)
                    if presence_fi is not None:
                        l_d_pres = l_d_pres + l_decomp_presence(
                            p0_fi, p1_fi, pov_fi, m_fi) * fw

                    occ_w_sum = occ_w_sum + fw

                occ_w_sum_c = occ_w_sum.clamp(min=1e-6)
                l_d_occ   = l_d_occ   / occ_w_sum_c
                l_d_ov    = l_d_ov    / occ_w_sum_c
                l_d_hier  = l_d_hier  / occ_w_sum_c
                l_d_front = l_d_front / occ_w_sum_c
                l_d_pres  = l_d_pres  / occ_w_sum_c
                if getattr(primary, "last_wbg", None) is not None:
                    route_w_sum = torch.tensor(0.0, device=device)
                    for fi in range(n_occ_frames):
                        fw = frame_w_t[fi]
                        l_route_ent = l_route_ent + l_route_entropy(
                            manager.last_w0[fi:fi+1].float(),
                            manager.last_w1[fi:fi+1].float(),
                            primary.last_wbg[fi:fi+1].float(),
                            masks_occ[fi:fi+1]) * fw
                        route_w_sum = route_w_sum + fw
                    l_route_ent = l_route_ent / route_w_sum.clamp(min=1e-6)

            # ── L_output_diverge (entity-specific output LoRA divergence) ──
            l_out_div = torch.tensor(0.0, device=device)
            out_e0_raw = getattr(primary, 'last_out_e0', None)
            out_e1_raw = getattr(primary, 'last_out_e1', None)
            if (out_e0_raw is not None and out_e1_raw is not None and
                    isinstance(out_e0_raw, torch.Tensor) and masks_occ is not None):
                n_div_frames = min(out_e0_raw.shape[0], T_frames, masks_occ.shape[0])
                div_w_sum = torch.tensor(0.0, device=device)
                for fi in range(n_div_frames):
                    fw = frame_w_t[fi]
                    l_out_div = l_out_div + l_output_entity_divergence(
                        out_e0_raw[fi:fi+1].float(),
                        out_e1_raw[fi:fi+1].float(),
                        masks_occ[fi:fi+1]) * fw
                    div_w_sum = div_w_sum + fw
                if div_w_sum.item() > 0:
                    l_out_div = l_out_div / div_w_sum.clamp(min=1e-6)

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
                    do   = [depth_orders_t[fi]] if fi < len(depth_orders_t) else [(0, 1)]
                    l_vis = l_vis + l_visible_weights_region_balanced(
                        w0_f, w1_f, m_f, do,
                        front_val=_SOFT_FRONT_VAL, back_val=_SOFT_BACK_VAL) * fw
                    l_vis_iou = l_vis_iou + l_visible_iou_soft(
                        w0_f, w1_f, m_f, do,
                        front_val=_SOFT_FRONT_VAL, back_val=_SOFT_BACK_VAL) * fw
                    vis_w_sum = vis_w_sum + fw
                vis_w_sum = vis_w_sum.clamp(min=1e-6)
                l_vis     = l_vis     / vis_w_sum
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

            # ── L_w_res ───────────────────────────────────────────────────
            l_w_res = torch.tensor(0.0, device=device)
            if manager.last_w_delta is not None:
                l_w_res = l_w_residual(manager.last_w_delta)

            # ── Cross-block slot identity losses ─────────────────────────
            l_excl = torch.tensor(0.0, device=device)
            l_slot_ref_all  = torch.tensor(0.0, device=device)
            l_slot_cont_all = torch.tensor(0.0, device=device)
            l_pair_id_all   = torch.tensor(0.0, device=device)
            excl_w_sum      = torch.tensor(0.0, device=device)
            slot_ref_w_sum  = torch.tensor(0.0, device=device)
            slot_cont_w_sum = torch.tensor(0.0, device=device)
            pair_id_w_sum   = torch.tensor(0.0, device=device)

            for blk_idx, _proc in enumerate(manager.procs):
                blk_F0 = _get_block_feat(manager, blk_idx, 0)
                blk_F1 = _get_block_feat(manager, blk_idx, 1)
                blk_Fg = _get_block_feat(manager, blk_idx, 'g')
                if blk_F0 is None:
                    continue

                S_blk  = blk_F0.shape[1]
                S_mask = masks_t.shape[-1]
                if S_blk != S_mask:
                    H_b = int(S_blk**0.5); H_m = int(S_mask**0.5)
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
                            H_bh = int(S_blk**0.5); H_mh = int(masks_t_hard.shape[-1]**0.5)
                            mh4d = masks_t_hard.view(masks_t_hard.shape[0], 2, H_mh, H_mh)
                            mh4d = torch.nn.functional.interpolate(
                                mh4d.float(), size=(H_bh, H_bh), mode='nearest')
                            masks_blk_hard = mh4d.view(masks_t_hard.shape[0], 2, S_blk)
                        else:
                            masks_blk_hard = masks_t_hard
                        vis_e0 = masks_blk_hard[fi, 0, :].unsqueeze(0)
                        vis_e1 = masks_blk_hard[fi, 1, :].unsqueeze(0)
                        F0_ref = F0_refs_solo.get(blk_idx, None) if F0_refs_solo else None
                        F1_ref = F1_refs_solo.get(blk_idx, None) if F1_refs_solo else None
                        if F0_ref is None or F1_ref is None:
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
                            l_pair_id_all = l_pair_id_all + l_pair_identity_preservation(
                                blk_F0[fi:fi+1].float(), blk_F1[fi:fi+1].float(),
                                F0_rs.detach().float(), F1_rs.detach().float(),
                                vis_e0, vis_e1,
                                margin=args.pair_margin,
                                pair_target=args.pair_target,
                                pair_weight=args.pair_weight) * fw
                            slot_ref_w_sum  = slot_ref_w_sum  + 2.0 * fw
                            slot_cont_w_sum = slot_cont_w_sum + fw
                            pair_id_w_sum   = pair_id_w_sum + fw

            l_excl = l_excl / excl_w_sum.clamp(min=1e-6)
            if slot_ref_w_sum.item() > 0.0:
                l_slot_ref_all  = l_slot_ref_all  / slot_ref_w_sum
            if slot_cont_w_sum.item() > 0.0:
                l_slot_cont_all = l_slot_cont_all / slot_cont_w_sum
            if pair_id_w_sum.item() > 0.0:
                l_pair_id_all = l_pair_id_all / pair_id_w_sum

            if is_stage_a:
                loss = (args.la_decomp_occ   * l_d_occ
                      + args.la_decomp_ov    * l_d_ov
                      + args.la_decomp_hier  * l_d_hier
                      + args.la_decomp_front * l_d_front
                      + args.la_decomp_pres  * l_d_pres
                      + args.la_vis          * l_vis
                      + args.la_vis_iou      * l_vis_iou
                      + args.la_wrong        * l_wrong
                      + args.la_excl         * l_excl
                      + args.la_slot_ref     * l_slot_ref_all
                      + args.la_slot_cont    * l_slot_cont_all
                      + args.la_pair_id      * l_pair_id_all
                      + args.la_noise        * l_noise
                      + args.la_solo_noise   * l_solo_noise
                      + args.la_solo_mask_recon * l_solo_mask_recon
                      + args.la_w_res        * l_w_res
                      + args.la_route_ent    * l_route_ent
                      + args.la_out_diverge  * l_out_div)

                for k, v in [("total", loss),
                              ("decomp_occ", l_d_occ), ("decomp_ov", l_d_ov),
                              ("decomp_hier", l_d_hier), ("decomp_front", l_d_front),
                              ("decomp_pres", l_d_pres),
                              ("vis", l_vis), ("vis_iou", l_vis_iou),
                              ("wrong", l_wrong), ("excl", l_excl),
                              ("slot_ref", l_slot_ref_all), ("slot_cont", l_slot_cont_all),
                              ("pair_id", l_pair_id_all),
                              ("noise", l_noise),
                              ("solo_noise", l_solo_noise),
                              ("solo_mask_recon", l_solo_mask_recon),
                              ("w_res", l_w_res), ("route_ent", l_route_ent), ("out_div", l_out_div),
                              ("collision_frac", step_col_frac)]:
                    epoch_losses[k].append(float(v.item())
                                           if isinstance(v, torch.Tensor) else float(v))
            else:
                # Stage B: full loss
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

                sigma_acc    = list(manager.sigma_acc)
                depth_masks_np = (masks_t_hard[:T_frames].detach().cpu().numpy() > 0.5)
                l_depth = l_zorder_direct(sigma_acc, depth_orders_t, depth_masks_np)

                l_ov = torch.tensor(0.0, device=device)
                if manager.last_w0 is not None:
                    ov_w_sum = torch.tensor(0.0, device=device)
                    for fi in range(T_use):
                        fw = frame_w_t[fi]
                        do = [depth_orders_t[fi]] if fi < len(depth_orders_t) else [(0, 1)]
                        l_ov = l_ov + l_overlap_ordering(
                            manager.last_w0[fi:fi+1].float(),
                            manager.last_w1[fi:fi+1].float(),
                            masks_t[fi:fi+1], do) * fw
                        ov_w_sum = ov_w_sum + fw
                    l_ov = l_ov / ov_w_sum.clamp(min=1e-6)

                # Union-presence supervision: encourage crisp foreground/background gating.
                l_pres = torch.tensor(0.0, device=device)
                pres_w_sum = torch.tensor(0.0, device=device)
                if getattr(primary, "last_presence_for_loss", None) is not None:
                    for fi in range(T_use):
                        fw = frame_w_t[fi]
                        l_pres = l_pres + l_decomp_presence(
                            primary.last_o0_for_loss[fi:fi+1].float(),
                            primary.last_o1_for_loss[fi:fi+1].float(),
                            primary.last_pov_for_loss[fi:fi+1].float(),
                            masks_t[fi:fi+1]) * fw
                        pres_w_sum = pres_w_sum + fw
                    l_pres = l_pres / pres_w_sum.clamp(min=1e-6)

                if getattr(primary, "last_wbg", None) is not None:
                    route_w_sum = torch.tensor(0.0, device=device)
                    for fi in range(T_use):
                        fw = frame_w_t[fi]
                        l_route_ent = l_route_ent + l_route_entropy(
                            manager.last_w0[fi:fi+1].float(),
                            manager.last_w1[fi:fi+1].float(),
                            primary.last_wbg[fi:fi+1].float(),
                            masks_t_hard[fi:fi+1]) * fw
                        route_w_sum = route_w_sum + fw
                    l_route_ent = l_route_ent / route_w_sum.clamp(min=1e-6)

                loss = (args.lb_decomp_occ   * l_d_occ
                      + args.lb_decomp_ov    * l_d_ov
                      + args.lb_decomp_hier  * l_d_hier
                      + args.lb_decomp_front * l_d_front
                      + args.lb_decomp_pres  * l_pres
                      + args.lb_vis          * l_vis
                      + args.lb_vis_iou      * l_vis_iou
                      + args.lb_wrong        * l_wrong
                      + args.lb_sigma        * l_sigma
                      + args.lb_depth        * l_depth
                      + args.lb_ov           * l_ov
                      + args.lb_excl         * l_excl
                      + args.lb_slot_ref     * l_slot_ref_all
                      + args.lb_slot_cont    * l_slot_cont_all
                      + args.lb_pair_id      * l_pair_id_all
                      + args.lb_noise        * l_noise
                      + args.lb_solo_noise   * l_solo_noise
                      + args.lb_solo_mask_recon * l_solo_mask_recon
                      + args.lb_w_res        * l_w_res
                      + args.lb_route_ent    * l_route_ent
                      + args.lb_out_diverge  * l_out_div)

                for k, v in [("total", loss),
                              ("decomp_occ", l_d_occ), ("decomp_ov", l_d_ov),
                              ("decomp_hier", l_d_hier), ("decomp_front", l_d_front),
                              ("decomp_pres", l_pres),
                              ("vis", l_vis), ("vis_iou", l_vis_iou),
                              ("wrong", l_wrong), ("sigma", l_sigma), ("depth", l_depth),
                              ("ov", l_ov), ("excl", l_excl),
                              ("slot_ref", l_slot_ref_all), ("slot_cont", l_slot_cont_all),
                              ("pair_id", l_pair_id_all),
                              ("noise", l_noise),
                              ("solo_noise", l_solo_noise),
                              ("solo_mask_recon", l_solo_mask_recon),
                              ("w_res", l_w_res), ("route_ent", l_route_ent), ("out_div", l_out_div),
                              ("collision_frac", step_col_frac)]:
                    epoch_losses[k].append(float(v.item())
                                           if isinstance(v, torch.Tensor) else float(v))

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss ep={epoch} step={batch_idx} → skip", flush=True)
                continue

            loss.backward()

            if is_stage_a:
                torch.nn.utils.clip_grad_norm_(manager.decomp_head_params(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.entity_lora_kv_params(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(manager.entity_lora_out_params(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(manager.weight_head_params(), max_norm=1.0)
                if args.stage_a_slot_bootstrap:
                    torch.nn.utils.clip_grad_norm_(manager.adapter_params(),    max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(manager.projector_params(),  max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(list(vca_layer.parameters()),    max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.adapter_params(),        max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.lora_params(),           max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(manager.entity_lora_kv_params(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(manager.entity_lora_out_params(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(manager.blend_params(),          max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(manager.weight_head_params(),    max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.projector_params(),      max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.decomp_head_params(),    max_norm=1.0)

            optimizer.step()

        lr_scheduler.step()

        avg         = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
        stage_lbl   = "A" if is_stage_a else "B"
        col_frac_ep = avg.get("collision_frac", 0.0)

        print(f"[Phase 53][Stage {stage_lbl}] epoch {epoch:03d}/{total_epochs-1}  "
              f"loss={avg['total']:.4f}  "
              f"d_occ={avg['decomp_occ']:.4f}  d_ov={avg['decomp_ov']:.4f}  "
              f"d_hier={avg['decomp_hier']:.4f}  d_front={avg['decomp_front']:.4f}  "
              f"d_pres={avg['decomp_pres']:.4f}  "
              f"vis={avg['vis']:.4f}  wrong={avg['wrong']:.4f}  "
              f"excl={avg['excl']:.4f}  slot_ref={avg['slot_ref']:.4f}  "
              f"pair_id={avg.get('pair_id', 0.0):.4f}  noise={avg.get('noise', 0.0):.4f}  "
              f"solo_noise={avg.get('solo_noise', 0.0):.4f}  "
              f"solo_mask_recon={avg.get('solo_mask_recon', 0.0):.4f}  "
              f"aug={n_collision_aug}/{n_collision_attempt}  "
              f"col_frac={col_frac_ep:.4f}  "
              f"low_ov={n_low_overlap_samples}/{len(step_indices)}",
              flush=True)

        # ── Validation ────────────────────────────────────────────────────
        should_eval = (
            (is_stage_a and epoch % args.eval_every_stage_a == 0) or
            (not is_stage_a and epoch % args.eval_every == 0) or
            (epoch == total_epochs - 1)
        )
        should_rollout = (
            args.rollout_every > 0 and
            ((epoch % args.rollout_every == 0) or (epoch == total_epochs - 1))
        ) and should_eval

        if should_eval:
            vca_layer.eval(); manager.eval()

            val_m = evaluate_val_set_phase53(
                pipe, manager, vca_layer, dataset, val_idx, device,
                t_fixed=args.t_max // 2,
                overlap_boost=args.val_overlap_boost,
                overlap_clip=args.val_overlap_clip,
                collision_thr=args.val_collision_thr)

            if should_rollout:
                rollout_m = evaluate_rollout_p53(
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

            vs_global    = val_score_phase46(
                val_m["tf_iou_e0"], val_m["tf_iou_e1"],
                val_m["tf_ord"], val_m["tf_wrong"],
                rollout_m["rollout_iou_e0"], rollout_m["rollout_iou_e1"],
                val_m["blend_sep"], has_rollout=has_rollout)
            vs_collision = val_score_phase46(
                val_m["collision_iou_e0"], val_m["collision_iou_e1"],
                val_m["collision_ord"], val_m["collision_wrong"],
                rollout_m["rollout_collision_iou_e0"], rollout_m["rollout_collision_iou_e1"],
                val_m["blend_sep"], has_rollout=has_rollout)
            mix = float(np.clip(args.val_collision_mix, 0.0, 1.0))
            vs  = (1.0 - mix) * vs_global + mix * vs_collision
            val_m["val_score"]           = vs
            val_m["val_score_global"]    = vs_global
            val_m["val_score_collision"] = vs_collision
            val_m["has_rollout"]         = has_rollout

            print(f"  [val] tf_iou_e0={val_m['tf_iou_e0']:.4f}  "
                  f"tf_iou_e1={val_m['tf_iou_e1']:.4f}  "
                  f"tf_ord={val_m['tf_ord']:.4f}  "
                  f"tf_wrong={val_m['tf_wrong']:.4f}", flush=True)
            print(f"  [collision] frac={val_m['collision_frac']:.3f}  "
                  f"iou_e0={val_m['collision_iou_e0']:.4f}  "
                  f"iou_e1={val_m['collision_iou_e1']:.4f}", flush=True)
            print(f"  [blend/occ] sep={val_m['blend_sep']:.4f}  "
                  f"ov={val_m['blend_overlap_mean']:.4f}  "
                  f"ex={val_m['blend_exclusive_mean']:.4f}  "
                  f"prod_ov={val_m['occ_prod_overlap_mean']:.4f}  "
                  f"prod_ex={val_m['occ_prod_exclusive_mean']:.4f}  "
                  f"prod_sep={val_m['occ_prod_sep']:.4f}  "
                  f"val_score={vs:.4f}", flush=True)

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
                "collision_aug_count":    int(n_collision_aug),
                "collision_aug_attempts": int(n_collision_attempt),
                "low_overlap_step_count": int(n_low_overlap_samples),
                **val_m, **rollout_m, **avg,
            })

            # Checkpoint
            ckpt_data = {
                "epoch":          epoch,
                "stage":          stage_lbl,
                "vca_state_dict": vca_layer.state_dict(),
                "val_score":      vs,
                "tf_iou_e0":      val_m["tf_iou_e0"],
                "tf_iou_e1":      val_m["tf_iou_e1"],
                "blend_sep":      val_m["blend_sep"],
                "occ_prod_sep":   val_m["occ_prod_sep"],
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
                        # Phase 53 new:
                        "decomp_heads": (p.decomp_heads.state_dict()
                                         if p.decomp_heads is not None else {}),
                        # Entity-specific K/V/Out LoRA:
                        "lora_k_e0": p.lora_k_e0.state_dict(),
                        "lora_k_e1": p.lora_k_e1.state_dict(),
                        "lora_v_e0": p.lora_v_e0.state_dict(),
                        "lora_v_e1": p.lora_v_e1.state_dict(),
                        "lora_out_e0": p.lora_out_e0.state_dict(),
                        "lora_out_e1": p.lora_out_e1.state_dict(),
                    }
                    for p in procs
                ],
            }
            torch.save(ckpt_data, str(save_dir / "latest.pt"))
            if vs > best_val_score:
                best_val_score = vs; best_epoch = epoch
                torch.save(ckpt_data, str(save_dir / "best.pt"))
                print(f"  ★ best epoch={best_epoch} val_score={vs:.4f}  "
                      f"blend_sep={val_m['blend_sep']:.4f}  "
                      f"occ_prod_sep={val_m['occ_prod_sep']:.4f}  "
                      f"→ {save_dir}/best.pt", flush=True)

            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            vca_layer.train(); manager.train()

    print(f"\n[Phase 53] 완료. best epoch={best_epoch} val_score={best_val_score:.4f}",
          flush=True)
    print(f"  성공 기준: blend_sep ≥ 0.01, occ_prod_sep > 0, collision_iou 개선",
          flush=True)


# =============================================================================
# argparse
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Phase 53: Explicit Decomposition Heads")

    p.add_argument("--ckpt",       type=str,
                   default="checkpoints/phase52/best.pt")
    p.add_argument("--data-root",  type=str, default="toy/data_objaverse")
    p.add_argument("--save-dir",   type=str, default="checkpoints/phase53")
    p.add_argument("--debug-dir",  type=str, default="outputs/phase53_debug")

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

    p.add_argument("--decomp-proj-dim",    type=int,   default=32)
    p.add_argument("--decomp-hidden",      type=int,   default=64)

    p.add_argument("--lr-vca",         type=float, default=DEFAULT_LR_VCA)
    p.add_argument("--lr-adapter",     type=float, default=DEFAULT_LR_ADAPTER)
    p.add_argument("--lr-lora",        type=float, default=DEFAULT_LR_LORA)
    p.add_argument("--lr-blend",       type=float, default=DEFAULT_LR_BLEND)
    p.add_argument("--lr-weight-head", type=float, default=DEFAULT_LR_WEIGHT_HEAD)
    p.add_argument("--lr-projector",   type=float, default=DEFAULT_LR_PROJECTOR)
    p.add_argument("--lr-decomp",      type=float, default=DEFAULT_LR_DECOMP)

    # Stage A losses
    p.add_argument("--la-decomp-occ",   type=float, default=DEFAULT_LA_DECOMP_OCC)
    p.add_argument("--la-decomp-ov",    type=float, default=DEFAULT_LA_DECOMP_OV)
    p.add_argument("--la-decomp-hier",  type=float, default=DEFAULT_LA_DECOMP_HIER)
    p.add_argument("--la-decomp-front", type=float, default=DEFAULT_LA_DECOMP_FRONT)
    p.add_argument("--la-decomp-pres",   type=float, default=DEFAULT_LA_DECOMP_PRES)
    p.add_argument("--la-vis",          type=float, default=DEFAULT_LA_VIS)
    p.add_argument("--la-vis-iou",      type=float, default=DEFAULT_LA_VIS_IOU)
    p.add_argument("--la-wrong",        type=float, default=DEFAULT_LA_WRONG)
    p.add_argument("--la-excl",         type=float, default=DEFAULT_LA_EXCL)
    p.add_argument("--la-slot-ref",     type=float, default=DEFAULT_LA_SLOT_REF)
    p.add_argument("--la-slot-cont",    type=float, default=DEFAULT_LA_SLOT_CONT)
    p.add_argument("--la-pair-id",      type=float, default=DEFAULT_LA_PAIR_ID)
    p.add_argument("--la-noise",        type=float, default=DEFAULT_LA_NOISE)
    p.add_argument("--la-solo-noise",   type=float, default=DEFAULT_LA_SOLO_NOISE)
    p.add_argument("--la-solo-mask-recon", type=float, default=DEFAULT_LA_SOLO_MASK_RECON)
    p.add_argument("--la-w-res",        type=float, default=DEFAULT_LA_W_RES)
    p.add_argument("--la-route-ent",    type=float, default=DEFAULT_LA_ROUTE_ENT)
    p.add_argument("--la-out-diverge",  type=float, default=DEFAULT_LA_OUT_DIVERGE)

    # Stage B losses
    p.add_argument("--lb-decomp-occ",   type=float, default=DEFAULT_LB_DECOMP_OCC)
    p.add_argument("--lb-decomp-ov",    type=float, default=DEFAULT_LB_DECOMP_OV)
    p.add_argument("--lb-decomp-hier",  type=float, default=DEFAULT_LB_DECOMP_HIER)
    p.add_argument("--lb-decomp-front", type=float, default=DEFAULT_LB_DECOMP_FRONT)
    p.add_argument("--lb-decomp-pres",  type=float, default=DEFAULT_LB_DECOMP_PRES)
    p.add_argument("--lb-vis",          type=float, default=DEFAULT_LB_VIS)
    p.add_argument("--lb-vis-iou",      type=float, default=DEFAULT_LB_VIS_IOU)
    p.add_argument("--lb-wrong",        type=float, default=DEFAULT_LB_WRONG)
    p.add_argument("--lb-sigma",        type=float, default=DEFAULT_LB_SIGMA)
    p.add_argument("--lb-depth",        type=float, default=DEFAULT_LB_DEPTH)
    p.add_argument("--lb-ov",           type=float, default=DEFAULT_LB_OV)
    p.add_argument("--lb-excl",         type=float, default=DEFAULT_LB_EXCL)
    p.add_argument("--lb-slot-ref",     type=float, default=DEFAULT_LB_SLOT_REF)
    p.add_argument("--lb-slot-cont",    type=float, default=DEFAULT_LB_SLOT_CONT)
    p.add_argument("--lb-pair-id",      type=float, default=DEFAULT_LB_PAIR_ID)
    p.add_argument("--lb-noise",        type=float, default=DEFAULT_LB_NOISE)
    p.add_argument("--lb-solo-noise",   type=float, default=DEFAULT_LB_SOLO_NOISE)
    p.add_argument("--lb-solo-mask-recon", type=float, default=DEFAULT_LB_SOLO_MASK_RECON)
    p.add_argument("--lb-w-res",        type=float, default=DEFAULT_LB_W_RES)
    p.add_argument("--lb-route-ent",    type=float, default=DEFAULT_LB_ROUTE_ENT)
    p.add_argument("--lb-out-diverge",  type=float, default=DEFAULT_LB_OUT_DIVERGE)
    p.add_argument("--pair-margin",     type=float, default=DEFAULT_PAIR_MARGIN)
    p.add_argument("--pair-target",     type=float, default=DEFAULT_PAIR_TARGET)
    p.add_argument("--pair-weight",     type=float, default=0.5)
    p.add_argument("--semantic-teacher-mix", type=float, default=0.0)

    # Misc
    p.add_argument("--reset-decomp-heads", dest="reset_decomp_heads",
                   action="store_true", help="Zero-init decomp_heads after restore")
    p.add_argument("--no-reset-decomp-heads", dest="reset_decomp_heads",
                   action="store_false")
    p.set_defaults(reset_decomp_heads=True)   # default: always reset (fresh start)

    p.add_argument("--val-frac",          type=float, default=VAL_FRAC)
    p.add_argument("--eval-every",        type=int,   default=5)
    p.add_argument("--eval-every-stage-a", type=int,  default=1)
    p.add_argument("--eval-seed",         type=int,   default=42)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--rollout-every",     type=int,   default=1)
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
    p.add_argument("--collision-ov-min-stage-a", type=float,
                   default=DEFAULT_COLLISION_OV_MIN_STAGE_A)
    p.add_argument("--collision-ov-max-stage-a", type=float,
                   default=DEFAULT_COLLISION_OV_MAX_STAGE_A)
    p.add_argument("--collision-min-accept-overlap", type=float,
                   default=DEFAULT_COLLISION_MIN_ACCEPT_OVERLAP)
    p.add_argument("--mask-dilate-radius",  type=int,   default=DEFAULT_MASK_DILATE_RADIUS)
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
    train_phase53(args)


if __name__ == "__main__":
    main()
