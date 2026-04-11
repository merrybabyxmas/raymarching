"""
Phase 44 — Overlap-Aware Blend Prior + GT Blend Supervision
=============================================================

2-Stage 학습:
  Stage A (5 epochs):  blend 병목 단독 검증
    학습: overlap_blend_head + weight_head + slot_blend_raw
    동결: adapters, lora, projectors, vca
    loss: 3.0*L_blend_target + 1.0*L_blend_rank + 0.5*L_vis + 0.2*L_wrong + 0.01*L_w_res

  Stage B (15 epochs): 전체 미세조정
    학습: 전부 (vca는 lr ×0.1)
    loss: full

성공 기준 (5 epoch A 안에):
  blend_sep: -0.118 → -0.03 이상
  blend_exclusive_mean > blend_bg_mean
  rollout_iou 유지 또는 상승
  best가 epoch 0이 아니게 됨
"""
import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
import imageio.v2 as iio2

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
    l_visible_weights_soft,
    PRIMARY_DIM,
)
from models.entity_slot_phase44 import (
    Phase44Processor,
    MultiBlockSlotManagerP44,
    inject_multi_block_entity_slot_p44,
    restore_multiblock_state_p44,
    l_blend_target,
    l_blend_rank,
    collect_blend_stats_detailed,
    val_score_phase44,
    build_blend_targets,
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

# soft target params (fallback if not exported from phase43)
_SOFT_FRONT_VAL = 0.85
_SOFT_BACK_VAL  = 0.05

# ─── Phase 44 하이퍼파라미터 ──────────────────────────────────────────────────
DEFAULT_STAGE_A_EPOCHS   = 5
DEFAULT_STAGE_B_EPOCHS   = 15
DEFAULT_STEPS_PER_EPOCH  = 20

# Stage A (blend-only)
DEFAULT_LA_BLEND_TARGET  = 3.0
DEFAULT_LA_BLEND_RANK    = 1.0
DEFAULT_LA_VIS           = 0.5
DEFAULT_LA_WRONG         = 0.2
DEFAULT_LA_W_RES         = 0.01

# Stage B (full)
DEFAULT_LB_VIS           = 2.0
DEFAULT_LB_WRONG         = 1.0
DEFAULT_LB_SIGMA         = 1.5
DEFAULT_LB_DEPTH         = 2.0
DEFAULT_LB_OV            = 0.5
DEFAULT_LB_EXCL          = 0.5
DEFAULT_LB_SLOT_REF      = 1.5
DEFAULT_LB_SLOT_CONT     = 0.3
DEFAULT_LB_BLEND_TARGET  = 2.0
DEFAULT_LB_BLEND_RANK    = 0.5
DEFAULT_LB_W_RES         = 0.01

# LR
DEFAULT_LR_VCA          = 2e-5
DEFAULT_LR_ADAPTER      = 1e-4
DEFAULT_LR_LORA         = 5e-5
DEFAULT_LR_BLEND        = 3e-4
DEFAULT_LR_WEIGHT_HEAD  = 1e-4
DEFAULT_LR_PROJECTOR    = 5e-5
DEFAULT_LR_OBH          = 2e-4   # OverlapBlendHead — 빠르게 수렴시킬 것

DEFAULT_ADAPTER_RANK    = 64
DEFAULT_LORA_RANK       = 4
DEFAULT_SLOT_BLEND      = 0.3
VAL_FRAC                = 0.2


# =============================================================================
# Rollout evaluation (버그 수정 포함)
# =============================================================================

@torch.no_grad()
def evaluate_rollout_p44(
    pipe,
    manager:    MultiBlockSlotManagerP44,
    dataset,
    val_idx:    list,
    device:     str,
    t_start:    int = ROLLOUT_T_START,
    n_steps:    int = ROLLOUT_N_STEPS,
) -> dict:
    manager.eval()
    rollout_iou_e0s = []
    rollout_iou_e1s = []

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

        except Exception as e:
            print(f"  [rollout warn] val {vi}: {e}", flush=True)
            continue

    manager.train()
    if not rollout_iou_e0s:
        return {"rollout_iou_e0": 0.0, "rollout_iou_e1": 0.0}
    return {
        "rollout_iou_e0": float(np.mean(rollout_iou_e0s)),
        "rollout_iou_e1": float(np.mean(rollout_iou_e1s)),
    }


# =============================================================================
# Teacher-forced validation (Phase 44)
# =============================================================================

@torch.no_grad()
def evaluate_val_set_phase44(
    pipe,
    manager:    MultiBlockSlotManagerP44,
    vca_layer:  VCALayer,
    dataset,
    val_idx:    list,
    device:     str,
    t_fixed:    int = 200,
) -> dict:
    manager.eval()
    iou_e0s     = []
    iou_e1s     = []
    ord_accs    = []
    wrong_leaks = []
    blend_stats_list = []

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

                # blend stats (last_blend_map is (B, S) in Phase44)
                bm_raw = getattr(manager.primary, 'last_blend_map', None)
                if (bm_raw is not None and
                        isinstance(bm_raw, torch.Tensor) and
                        bm_raw.dim() >= 2 and fi < bm_raw.shape[0]):
                    bm_f   = bm_raw[fi:fi+1].float()
                    b_stat = collect_blend_stats_detailed(bm_f, m_f)
                    blend_stats_list.append(b_stat)

        except Exception as e:
            print(f"  [val warn] {vi}: {e}", flush=True)
            continue

    manager.train()

    if not iou_e0s:
        return {"tf_iou_e0": 0.0, "tf_iou_e1": 0.0, "tf_ord": 0.0,
                "tf_wrong": 1.0, "val_score": 0.0,
                "blend_sep": 0.0, "blend_overlap_mean": 0.0,
                "blend_exclusive_mean": 0.0, "blend_bg_mean": 0.0,
                "blend_gap_bg": 0.0}

    tf_iou_e0 = float(np.mean(iou_e0s))
    tf_iou_e1 = float(np.mean(iou_e1s))
    tf_ord    = float(np.mean(ord_accs))
    tf_wrong  = float(np.mean(wrong_leaks))

    def _agg(key): return float(np.mean([s[key] for s in blend_stats_list])) if blend_stats_list else 0.0
    blend_sep    = _agg("blend_sep")
    blend_ov     = _agg("blend_overlap_mean")
    blend_ex     = _agg("blend_exclusive_mean")
    blend_bg     = _agg("blend_bg_mean")
    blend_gap_bg = _agg("blend_gap_bg")

    vs = val_score_phase44(tf_iou_e0, tf_iou_e1, tf_ord, tf_wrong,
                           blend_sep=blend_sep)
    return {
        "tf_iou_e0":            tf_iou_e0,
        "tf_iou_e1":            tf_iou_e1,
        "tf_ord":               tf_ord,
        "tf_wrong":             tf_wrong,
        "blend_sep":            blend_sep,
        "blend_overlap_mean":   blend_ov,
        "blend_exclusive_mean": blend_ex,
        "blend_bg_mean":        blend_bg,
        "blend_gap_bg":         blend_gap_bg,
        "val_score":            vs,
    }


# =============================================================================
# Training loop
# =============================================================================

def _set_requires_grad_stage_a(manager, vca_layer, procs):
    """Stage A: blend_head 계열 + weight_head만 학습. 나머지 동결."""
    # 전체 동결
    for p in vca_layer.parameters():   p.requires_grad_(False)
    for proc in procs:
        for p in proc.parameters():    p.requires_grad_(False)
    # 풀기: overlap_blend_head + weight_head + slot_blend_raw
    for proc in procs:
        for p in proc.overlap_blend_head.parameters(): p.requires_grad_(True)
        for p in proc.weight_head.parameters():        p.requires_grad_(True)
        proc.slot_blend_raw.requires_grad_(True)
    print("  [stage A] 학습 파라미터: overlap_blend_head + weight_head + slot_blend_raw", flush=True)


def _set_requires_grad_stage_b(manager, vca_layer, procs):
    """Stage B: 전체 학습."""
    for p in vca_layer.parameters():   p.requires_grad_(True)
    for proc in procs:
        for p in proc.parameters():    p.requires_grad_(True)
    print("  [stage B] 전체 파라미터 학습", flush=True)


def train_phase44(args):
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

    # ── 파이프라인 로드 ──────────────────────────────────────────────────────
    print("[Phase 44] 파이프라인 로드 중...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    # ── Checkpoint 로드 ──────────────────────────────────────────────────────
    print(f"[Phase 44] checkpoint 로드: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location=device)
    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM, context_dim=768,
        n_heads=8, n_entities=2, z_bins=DEFAULT_Z_BINS, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=DEPTH_PE_INIT_SCALE,
    ).to(device)
    vca_layer.load_state_dict(ckpt["vca_state_dict"])
    print("  VCA loaded", flush=True)

    # ── 데이터셋 ─────────────────────────────────────────────────────────────
    print("[Phase 44] 데이터셋 로드 중...", flush=True)
    try:
        dataset = ObjaverseDatasetPhase40(args.data_root, n_frames=args.n_frames)
        print(f"  ObjaverseDatasetPhase40: {len(dataset)} samples", flush=True)
    except Exception as e:
        print(f"  Phase40 dataset 실패({e}), fallback", flush=True)
        dataset = ObjaverseDatasetWithMasks(args.data_root, n_frames=args.n_frames)

    if len(dataset) == 0:
        raise RuntimeError(f"데이터셋 비어있음: {args.data_root}")

    # ── Train/Val 분리 ───────────────────────────────────────────────────────
    overlap_scores = compute_dataset_overlap_scores(dataset)
    train_idx, val_idx = split_train_val(overlap_scores, val_frac=args.val_frac)
    sample_weights = make_sampling_weights(train_idx, overlap_scores)
    print(f"  train={len(train_idx)}  val={len(val_idx)}", flush=True)

    # ── Phase44Processor 주입 ─────────────────────────────────────────────────
    inject_keys = args.inject_keys.split(",") if args.inject_keys else DEFAULT_INJECT_KEYS
    init_ctx    = get_color_entity_context(pipe, dataset[train_idx[0]][3], device)
    procs, orig_procs = inject_multi_block_entity_slot_p44(
        pipe, vca_layer, init_ctx,
        inject_keys     = inject_keys,
        slot_blend_init = args.slot_blend,
        adapter_rank    = args.adapter_rank,
        lora_rank       = args.lora_rank,
        use_blend_head  = True,
        obh_hidden      = 32,
        proj_hidden     = 256,
    )
    for p in procs:
        p.to(device)

    manager = MultiBlockSlotManagerP44(procs, inject_keys, primary_idx=1)

    # ── 모든 block state 복원 ────────────────────────────────────────────────
    print("[Phase 44] checkpoint 복원...", flush=True)
    restore_multiblock_state_p44(manager, ckpt, device=device)

    # ── UNet/텍스트인코더/VAE 동결 ────────────────────────────────────────────
    for p in pipe.unet.parameters():         p.requires_grad_(False)
    for p in pipe.text_encoder.parameters(): p.requires_grad_(False)
    for p in pipe.vae.parameters():          p.requires_grad_(False)

    # ── Epoch 0 no-op validation ─────────────────────────────────────────────
    print("\n[Phase 44] Epoch 0 no-op validation...", flush=True)
    vca_layer.eval(); manager.eval()
    val_m0 = evaluate_val_set_phase44(
        pipe, manager, vca_layer, dataset, val_idx, device,
        t_fixed=args.t_max // 2)
    print(f"  [epoch0-noop] iou_e0={val_m0['tf_iou_e0']:.4f}  "
          f"iou_e1={val_m0['tf_iou_e1']:.4f}  "
          f"blend_sep={val_m0['blend_sep']:.4f}  "
          f"blend_ov={val_m0['blend_overlap_mean']:.4f}  "
          f"blend_ex={val_m0['blend_exclusive_mean']:.4f}  "
          f"blend_bg={val_m0['blend_bg_mean']:.4f}  "
          f"val_score={val_m0['val_score']:.4f}", flush=True)
    vca_layer.train(); manager.train()

    history        = []
    best_val_score = -1.0
    best_epoch     = -1
    total_epochs   = args.stage_a_epochs + args.stage_b_epochs

    for epoch in range(total_epochs):
        is_stage_a = (epoch < args.stage_a_epochs)

        # ── Stage 전환 ────────────────────────────────────────────────────
        if epoch == 0:
            print(f"\n[Phase 44] Stage A 시작 ({args.stage_a_epochs} epochs): "
                  f"blend head 병목 단독 검증", flush=True)
            _set_requires_grad_stage_a(manager, vca_layer, procs)
            # Stage A optimizer
            optimizer = optim.AdamW([
                {"params": manager.overlap_blend_head_params(),
                 "lr": args.lr_obh,    "name": "obh"},
                {"params": manager.weight_head_params(),
                 "lr": args.lr_weight_head, "name": "weight_head"},
                {"params": [p.slot_blend_raw for p in procs],
                 "lr": args.lr_blend,  "name": "blend_raw"},
            ], weight_decay=1e-4)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.stage_a_epochs, eta_min=1e-6)

        elif epoch == args.stage_a_epochs:
            print(f"\n[Phase 44] Stage B 시작 ({args.stage_b_epochs} epochs): "
                  f"전체 미세조정", flush=True)
            _set_requires_grad_stage_b(manager, vca_layer, procs)
            # Stage B optimizer
            optimizer = optim.AdamW([
                {"params": list(vca_layer.parameters()),
                 "lr": args.lr_vca * 0.1, "name": "vca"},
                {"params": manager.adapter_params(),
                 "lr": args.lr_adapter,    "name": "adapters"},
                {"params": manager.lora_params(),
                 "lr": args.lr_lora,       "name": "lora"},
                {"params": manager.blend_params(),
                 "lr": args.lr_blend,      "name": "blend"},
                {"params": manager.weight_head_params(),
                 "lr": args.lr_weight_head,"name": "weight_head"},
                {"params": manager.projector_params(),
                 "lr": args.lr_projector,  "name": "projectors"},
                {"params": manager.overlap_blend_head_params(),
                 "lr": args.lr_obh,        "name": "obh"},
            ], weight_decay=1e-4)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.stage_b_epochs, eta_min=args.lr_vca * 0.05)

        vca_layer.train(); manager.train()

        loss_keys = (["total", "blend_target", "blend_rank", "vis", "wrong", "w_res"]
                     if is_stage_a else
                     ["total", "vis", "wrong", "sigma", "depth", "ov",
                      "excl", "slot_ref", "slot_cont", "blend_target",
                      "blend_rank", "w_res"])
        epoch_losses = {k: [] for k in loss_keys}

        chosen     = np.random.choice(len(train_idx), size=args.steps_per_epoch,
                                      replace=True, p=sample_weights)
        step_indices = [train_idx[ci] for ci in chosen]

        for batch_idx, data_idx in enumerate(step_indices):
            sample = dataset[data_idx]
            frames_np, _, depth_orders, meta, entity_masks = (
                sample[:5] if len(sample) < 8 else
                (sample[0], sample[1], sample[2], sample[3], sample[4]))

            entity_ctx = get_color_entity_context(pipe, meta, device)
            toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)

            with torch.no_grad():
                latents = encode_frames_to_latents(pipe, frames_np, device)
            noise = torch.randn_like(latents)
            t     = torch.randint(0, args.t_max, (1,), device=device).long()
            noisy = pipe.scheduler.add_noise(latents, noise, t)

            T_frames       = min(frames_np.shape[0], entity_masks.shape[0])
            masks_t        = torch.from_numpy(entity_masks[:T_frames].astype(np.float32)).to(device)
            depth_orders_t = depth_orders[:T_frames]

            # ── Entity-only ref forwards (Stage B only — Stage A는 생략) ─
            F0_refs: Dict[int, Optional[torch.Tensor]] = {}
            F1_refs: Dict[int, Optional[torch.Tensor]] = {}
            if not is_stage_a:
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

            # ── PRIMARY LOSSES ────────────────────────────────────────────

            # L_vis
            l_vis = torch.tensor(0.0, device=device)
            if manager.last_w0 is not None:
                for fi in range(T_use):
                    w0_f = manager.last_w0[fi:fi+1].float()
                    w1_f = manager.last_w1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    do   = [depth_orders_t[fi]] if fi < len(depth_orders_t) else [(0,1)]
                    l_vis = l_vis + l_visible_weights_soft(
                        w0_f, w1_f, m_f, do,
                        front_val=_SOFT_FRONT_VAL, back_val=_SOFT_BACK_VAL)
                l_vis = l_vis / max(T_use, 1)

            # L_wrong
            l_wrong = torch.tensor(0.0, device=device)
            if manager.last_w0 is not None:
                for fi in range(T_use):
                    l_wrong = l_wrong + l_wrong_slot_suppression(
                        manager.last_w0[fi:fi+1].float(),
                        manager.last_w1[fi:fi+1].float(),
                        masks_t[fi:fi+1])
                l_wrong = l_wrong / max(T_use, 1)

            # L_blend_target + L_blend_rank (primary, all frames)
            # Use last_blend_map_for_loss (non-detached) so grad flows to overlap_blend_head
            l_bt  = torch.tensor(0.0, device=device)
            l_br  = torch.tensor(0.0, device=device)
            primary = manager.primary
            bm_for_loss = getattr(primary, 'last_blend_map_for_loss', None)
            if (bm_for_loss is not None and
                    isinstance(bm_for_loss, torch.Tensor) and
                    bm_for_loss.dim() >= 2 and T_use > 0):
                for fi in range(min(bm_for_loss.shape[0], T_use)):
                    bm_f = bm_for_loss[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    l_bt = l_bt + l_blend_target(bm_f, m_f)
                    l_br = l_br + l_blend_rank(bm_f, m_f)
                l_bt = l_bt / max(T_use, 1)
                l_br = l_br / max(T_use, 1)

            # L_w_res
            l_w_res = torch.tensor(0.0, device=device)
            if manager.last_w_delta is not None:
                l_w_res = l_w_residual(manager.last_w_delta)

            if is_stage_a:
                # Stage A: blend-focused loss
                loss = (args.la_blend_target * l_bt
                      + args.la_blend_rank    * l_br
                      + args.la_vis           * l_vis
                      + args.la_wrong         * l_wrong
                      + args.la_w_res         * l_w_res)

                for k, v in [("total", loss), ("blend_target", l_bt),
                              ("blend_rank", l_br), ("vis", l_vis),
                              ("wrong", l_wrong), ("w_res", l_w_res)]:
                    epoch_losses[k].append(float(v.item())
                                           if isinstance(v, torch.Tensor) else float(v))
            else:
                # Stage B: full loss
                l_sigma = torch.tensor(0.0, device=device)
                if manager.last_alpha0 is not None:
                    for fi in range(T_use):
                        l_sigma = l_sigma + l_sigma_spatial(
                            manager.last_alpha0[fi:fi+1].float(),
                            manager.last_alpha1[fi:fi+1].float(),
                            masks_t[fi:fi+1])
                    l_sigma = l_sigma / max(T_use, 1)

                sigma_acc = list(manager.sigma_acc)
                l_depth   = l_zorder_direct(sigma_acc, depth_orders_t, entity_masks[:T_frames])

                l_ov = torch.tensor(0.0, device=device)
                if manager.last_w0 is not None:
                    for fi in range(T_use):
                        do = [depth_orders_t[fi]] if fi < len(depth_orders_t) else [(0,1)]
                        l_ov = l_ov + l_overlap_ordering(
                            manager.last_w0[fi:fi+1].float(),
                            manager.last_w1[fi:fi+1].float(),
                            masks_t[fi:fi+1], do)
                    l_ov = l_ov / max(T_use, 1)

                l_excl = l_slot_ref_all = l_slot_cont_all = torch.tensor(0.0, device=device)
                n_blk = 0
                for blk_idx, proc in enumerate(manager.procs):
                    blk_F0 = _get_block_feat(manager, blk_idx, 0)
                    blk_F1 = _get_block_feat(manager, blk_idx, 1)
                    blk_Fg = _get_block_feat(manager, blk_idx, 'g')
                    if blk_F0 is None: continue

                    S_blk = blk_F0.shape[1]; S_mask = masks_t.shape[-1]
                    if S_blk != S_mask:
                        H_b = int(S_blk**0.5); H_m = int(S_mask**0.5)
                        m4d = masks_t.view(masks_t.shape[0], 2, H_m, H_m)
                        m4d = torch.nn.functional.interpolate(m4d.float(), size=(H_b, H_b), mode='nearest')
                        masks_blk = m4d.view(masks_t.shape[0], 2, S_blk)
                    else:
                        masks_blk = masks_t

                    for fi in range(min(blk_F0.shape[0], T_frames)):
                        l_excl = l_excl + l_entity_exclusive(
                            blk_F0[fi:fi+1].float(), blk_F1[fi:fi+1].float(),
                            blk_Fg[fi:fi+1].float(), masks_blk[fi:fi+1])

                        vis_e0 = masks_blk[fi, 0, :].unsqueeze(0)
                        vis_e1 = masks_blk[fi, 1, :].unsqueeze(0)
                        F0_ref = F0_refs.get(blk_idx, None)
                        F1_ref = F1_refs.get(blk_idx, None)
                        if F0_ref is not None and F1_ref is not None:
                            F0_rs = F0_ref[fi:fi+1]
                            F1_rs = F1_ref[fi:fi+1]
                            if F0_rs.shape[1] != S_blk:
                                F0_rs = _resize_feat(F0_rs, S_blk)
                                F1_rs = _resize_feat(F1_rs, S_blk)
                            l_slot_ref_all = l_slot_ref_all + l_slot_ref(
                                blk_F0[fi:fi+1].float(), F0_rs.detach().float(), vis_e0)
                            l_slot_ref_all = l_slot_ref_all + l_slot_ref(
                                blk_F1[fi:fi+1].float(), F1_rs.detach().float(), vis_e1)
                            l_slot_cont_all = l_slot_cont_all + l_slot_contrast(
                                blk_F0[fi:fi+1].float(), blk_F1[fi:fi+1].float(),
                                F0_rs.detach().float(), F1_rs.detach().float(),
                                vis_e0, vis_e1)
                    n_blk += 1

                denom = max(T_use * max(n_blk, 1), 1)
                l_excl        = l_excl        / denom
                l_slot_ref_all = l_slot_ref_all / denom
                l_slot_cont_all = l_slot_cont_all / denom

                loss = (args.lb_vis          * l_vis
                      + args.lb_wrong        * l_wrong
                      + args.lb_sigma        * l_sigma
                      + args.lb_depth        * l_depth
                      + args.lb_ov           * l_ov
                      + args.lb_excl         * l_excl
                      + args.lb_slot_ref     * l_slot_ref_all
                      + args.lb_slot_cont    * l_slot_cont_all
                      + args.lb_blend_target * l_bt
                      + args.lb_blend_rank   * l_br
                      + args.lb_w_res        * l_w_res)

                for k, v in [("total", loss), ("vis", l_vis), ("wrong", l_wrong),
                              ("sigma", l_sigma), ("depth", l_depth), ("ov", l_ov),
                              ("excl", l_excl), ("slot_ref", l_slot_ref_all),
                              ("slot_cont", l_slot_cont_all), ("blend_target", l_bt),
                              ("blend_rank", l_br), ("w_res", l_w_res)]:
                    epoch_losses[k].append(float(v.item()) if isinstance(v, torch.Tensor) else float(v))

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss ep={epoch} step={batch_idx} → skip", flush=True)
                continue

            loss.backward()

            if is_stage_a:
                torch.nn.utils.clip_grad_norm_(manager.overlap_blend_head_params(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.weight_head_params(),        max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(list(vca_layer.parameters()),        max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.adapter_params(),            max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.lora_params(),               max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(manager.blend_params(),              max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(manager.weight_head_params(),        max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.projector_params(),          max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manager.overlap_blend_head_params(), max_norm=1.0)

            optimizer.step()

        lr_scheduler.step()

        avg      = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
        stage_lbl = "A" if is_stage_a else "B"
        print(f"[Phase 44][Stage {stage_lbl}] epoch {epoch:03d}/{total_epochs-1}  "
              f"loss={avg['total']:.4f}  blend_target={avg['blend_target']:.4f}  "
              f"blend_rank={avg['blend_rank']:.4f}  vis={avg['vis']:.4f}  "
              f"wrong={avg['wrong']:.4f}  w_res={avg['w_res']:.4f}", flush=True)

        # ── Validation ────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == total_epochs - 1:
            vca_layer.eval(); manager.eval()

            val_m = evaluate_val_set_phase44(
                pipe, manager, vca_layer, dataset, val_idx, device,
                t_fixed=args.t_max // 2)

            rollout_m = {"rollout_iou_e0": 0.0, "rollout_iou_e1": 0.0}
            if epoch % (args.eval_every * 2) == 0 or epoch == total_epochs - 1:
                rollout_m = evaluate_rollout_p44(
                    pipe, manager, dataset, val_idx, device,
                    t_start=ROLLOUT_T_START, n_steps=ROLLOUT_N_STEPS)
                print(f"  [rollout] iou_e0={rollout_m['rollout_iou_e0']:.4f}  "
                      f"iou_e1={rollout_m['rollout_iou_e1']:.4f}", flush=True)

            vs = val_score_phase44(
                val_m["tf_iou_e0"], val_m["tf_iou_e1"],
                val_m["tf_ord"], val_m["tf_wrong"],
                rollout_m["rollout_iou_e0"], rollout_m["rollout_iou_e1"],
                val_m["blend_sep"])
            val_m["val_score"] = vs

            print(f"  [val] tf_iou_e0={val_m['tf_iou_e0']:.4f}  "
                  f"tf_iou_e1={val_m['tf_iou_e1']:.4f}  "
                  f"tf_ord={val_m['tf_ord']:.4f}  "
                  f"tf_wrong={val_m['tf_wrong']:.4f}", flush=True)
            print(f"  [blend] sep={val_m['blend_sep']:.4f}  "
                  f"ov={val_m['blend_overlap_mean']:.4f}  "
                  f"ex={val_m['blend_exclusive_mean']:.4f}  "
                  f"bg={val_m['blend_bg_mean']:.4f}  "
                  f"gap_bg={val_m['blend_gap_bg']:.4f}  "
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
                "val_score": vs, **val_m, **rollout_m, **avg,
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
                    }
                    for p in procs
                ],
            }
            torch.save(ckpt_data, str(save_dir / "latest.pt"))
            if vs > best_val_score:
                best_val_score = vs; best_epoch = epoch
                torch.save(ckpt_data, str(save_dir / "best.pt"))
                print(f"  ★ best epoch={best_epoch} val_score={vs:.4f} "
                      f"blend_sep={val_m['blend_sep']:.4f}  → {save_dir}/best.pt", flush=True)

            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            vca_layer.train(); manager.train()

    print(f"\n[Phase 44] 완료. best epoch={best_epoch} val_score={best_val_score:.4f}",
          flush=True)
    print(f"  성공 기준: blend_sep > -0.03, rollout_iou 유지, best ≠ epoch0", flush=True)


# =============================================================================
# argparse
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Phase 44: Overlap-Aware Blend Prior")

    p.add_argument("--ckpt",         type=str, default="checkpoints/phase43/best.pt")
    p.add_argument("--data-root",    type=str, default="toy/data_objaverse")
    p.add_argument("--save-dir",     type=str, default="checkpoints/phase44")
    p.add_argument("--debug-dir",    type=str, default="outputs/phase44_debug")

    p.add_argument("--stage-a-epochs",   type=int,   default=DEFAULT_STAGE_A_EPOCHS)
    p.add_argument("--stage-b-epochs",   type=int,   default=DEFAULT_STAGE_B_EPOCHS)
    p.add_argument("--steps-per-epoch",  type=int,   default=DEFAULT_STEPS_PER_EPOCH)
    p.add_argument("--n-frames",         type=int,   default=8)
    p.add_argument("--n-steps",          type=int,   default=20)
    p.add_argument("--t-max",            type=int,   default=300)
    p.add_argument("--height",           type=int,   default=256)
    p.add_argument("--width",            type=int,   default=256)
    p.add_argument("--adapter-rank",     type=int,   default=DEFAULT_ADAPTER_RANK)
    p.add_argument("--lora-rank",        type=int,   default=DEFAULT_LORA_RANK)
    p.add_argument("--slot-blend",       type=float, default=DEFAULT_SLOT_BLEND)
    p.add_argument("--inject-keys",      type=str,   default=None)

    # LR
    p.add_argument("--lr-vca",          type=float, default=DEFAULT_LR_VCA)
    p.add_argument("--lr-adapter",      type=float, default=DEFAULT_LR_ADAPTER)
    p.add_argument("--lr-lora",         type=float, default=DEFAULT_LR_LORA)
    p.add_argument("--lr-blend",        type=float, default=DEFAULT_LR_BLEND)
    p.add_argument("--lr-weight-head",  type=float, default=DEFAULT_LR_WEIGHT_HEAD)
    p.add_argument("--lr-projector",    type=float, default=DEFAULT_LR_PROJECTOR)
    p.add_argument("--lr-obh",          type=float, default=DEFAULT_LR_OBH)

    # Stage A lambdas
    p.add_argument("--la-blend-target", type=float, default=DEFAULT_LA_BLEND_TARGET)
    p.add_argument("--la-blend-rank",   type=float, default=DEFAULT_LA_BLEND_RANK)
    p.add_argument("--la-vis",          type=float, default=DEFAULT_LA_VIS)
    p.add_argument("--la-wrong",        type=float, default=DEFAULT_LA_WRONG)
    p.add_argument("--la-w-res",        type=float, default=DEFAULT_LA_W_RES)

    # Stage B lambdas
    p.add_argument("--lb-vis",          type=float, default=DEFAULT_LB_VIS)
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

    p.add_argument("--val-frac",   type=float, default=VAL_FRAC)
    p.add_argument("--eval-every", type=int,   default=5)
    p.add_argument("--eval-seed",  type=int,   default=42)
    p.add_argument("--seed",       type=int,   default=42)

    args = p.parse_args()
    train_phase44(args)


if __name__ == "__main__":
    main()
