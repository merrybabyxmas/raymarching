"""
Phase 42 — Multi-Block Restore + Residual Weight Head
======================================================

Phase 41과 비교한 핵심 변경:
  A. restore_multiblock_state() — 모든 block state 완전 복원 (primary-only 버그 수정)
  B. WeightHead — Porter-Duff → residual-corrected learned weights
  C. val_score_phase42 — id_margin 제거 (iou_e0/e1/ord/wrong만)
  D. All-block appearance: l_slot_ref + l_slot_contrast

학습 설정:
  - checkpoints/phase40_v2/best.pt에서 시작
  - epoch 0 no-op validation으로 phase40_v2 수치 재현 확인
  - 20 epochs (병목 판별용 최소 실험)
  - primary block: l_vis, l_wrong, l_sigma, l_depth, l_ov, l_w_res
  - all blocks: l_excl, l_slot_ref, l_slot_contrast
  - eval every 5 epochs

성공 기준 (20 epoch 내):
  iou_e0 > 0.14
  iou_e1 > 0.14
  ordering_acc > 0.50
  wrong_slot_leak < 0.22
"""
import argparse
import copy
import json
import sys
from pathlib import Path

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
    compute_overlap_score,
    entity_score as compute_entity_score_debug,
)
from models.entity_slot_phase40 import (
    compute_visible_iou_e0,
    compute_visible_iou_e1,
    DEFAULT_INJECT_KEYS,
    BLOCK_INNER_DIMS,
    val_score_phase40,
)
from models.entity_slot_phase42 import (
    Phase42Processor,
    MultiBlockSlotManagerP42,
    inject_multi_block_entity_slot_p42,
    restore_multiblock_state,
    l_w_residual,
    l_slot_ref,
    l_slot_contrast,
    val_score_phase42,
)
from scripts.run_animatediff import load_pipeline
from scripts.train_animatediff_vca import encode_frames_to_latents
from scripts.train_phase31 import (
    INJECT_KEY, INJECT_QUERY_DIM,
    DEPTH_PE_INIT_SCALE,
    ObjaverseDatasetWithMasks,
    make_color_prompts, get_color_entity_context,
    l_zorder_direct,
    measure_depth_rank_accuracy,
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
from scripts.generate_solo_renders import (
    ObjaverseDatasetPhase40,
    compute_visible_masks_np,
)


# ─── Phase 42 하이퍼파라미터 ──────────────────────────────────────────────────
DEFAULT_TOTAL_EPOCHS     = 20   # 병목 판별용 최소 실험 (phase41처럼 60 epoch 낭비 X)
DEFAULT_LAMBDA_VIS       = 2.0
DEFAULT_LAMBDA_WRONG     = 1.0
DEFAULT_LAMBDA_SIGMA     = 2.0   # phase41처럼 5.0으로 올리지 않음
DEFAULT_LAMBDA_DEPTH     = 2.0
DEFAULT_LAMBDA_OV        = 0.5
DEFAULT_LAMBDA_EXCL      = 0.5
DEFAULT_LAMBDA_SLOT_REF  = 2.0
DEFAULT_LAMBDA_SLOT_CONT = 0.5
DEFAULT_LAMBDA_W_RES     = 0.01  # small regularizer (초반 Porter-Duff에서 안 멀리 가도록)

DEFAULT_LR_VCA         = 2e-5
DEFAULT_LR_ADAPTER     = 1e-4
DEFAULT_LR_LORA        = 5e-5
DEFAULT_LR_BLEND       = 3e-4
DEFAULT_LR_WEIGHT_HEAD = 1e-4

DEFAULT_ADAPTER_RANK     = 64
DEFAULT_LORA_RANK        = 4
DEFAULT_SLOT_BLEND       = 0.3
DEFAULT_STEPS_PER_EPOCH  = 20

ROLLOUT_T_START  = 250
ROLLOUT_N_STEPS  = 5
VAL_FRAC         = 0.2


# =============================================================================
# Rollout evaluation
# =============================================================================

@torch.no_grad()
def evaluate_rollout(
    pipe,
    manager:    MultiBlockSlotManagerP42,
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
            if len(sample) >= 8:
                frames_np, _, depth_orders, meta, entity_masks, _, _, _ = sample
            else:
                frames_np, _, depth_orders, meta, entity_masks = sample

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
            ts = scheduler_state.timesteps

            for step_t in (ts[:n_steps] if ts is not None and len(ts) >= n_steps
                           else [t_tensor[0]]):
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
# Teacher-forced validation
# =============================================================================

@torch.no_grad()
def evaluate_val_set_phase42(
    pipe,
    manager:    MultiBlockSlotManagerP42,
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
    # id_margin은 로깅 전용 — val_score에서 제외
    id_margins  = []

    for vi in val_idx:
        try:
            sample = dataset[vi]
            if len(sample) >= 8:
                frames_np, _, depth_orders, meta, entity_masks, _, _, _ = sample
            else:
                frames_np, _, depth_orders, meta, entity_masks = sample

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

            proc = manager.primary
            for fi in range(T_frames):
                w0_f = manager.last_w0[fi:fi+1].float()
                w1_f = manager.last_w1[fi:fi+1].float()
                m_f  = masks_t[fi:fi+1]
                do   = [depth_orders[fi]] if fi < len(depth_orders) else [(0, 1)]

                iou_e0s.append(compute_visible_iou_e0(w0_f, m_f, do))
                iou_e1s.append(compute_visible_iou_e1(w1_f, m_f, do))
                ord_accs.append(compute_ordering_accuracy(w0_f, w1_f, m_f, do))
                wrong_leaks.append(compute_wrong_slot_leak(w0_f, w1_f, m_f))

                # id_margin: F0 vs F1 cosine divergence (logging only)
                if proc.last_F0 is not None and proc.last_F1 is not None:
                    F0_n = torch.nn.functional.normalize(
                        proc.last_F0[fi:fi+1].float(), dim=-1)
                    F1_n = torch.nn.functional.normalize(
                        proc.last_F1[fi:fi+1].float(), dim=-1)
                    id_margins.append(
                        1.0 - (F0_n * F1_n).sum(-1).mean().item())

        except Exception as e:
            print(f"  [val warn] {vi}: {e}", flush=True)
            continue

    manager.train()

    if not iou_e0s:
        return {"iou_e0": 0.0, "iou_e1": 0.0, "ordering_acc": 0.0,
                "wrong_slot_leak": 1.0, "id_margin": 0.0, "dra": 0.0,
                "val_score": 0.0}

    iou_e0_m = float(np.mean(iou_e0s))
    iou_e1_m = float(np.mean(iou_e1s))
    ord_m    = float(np.mean(ord_accs))
    wl_m     = float(np.mean(wrong_leaks))
    id_m     = float(np.mean(id_margins)) if id_margins else 0.0

    try:
        class _Sub:
            def __init__(self, ds, idx): self._ds, self._idx = ds, idx
            def __len__(self): return len(self._idx)
            def __getitem__(self, i): return self._ds[self._idx[i]]
        sub = _Sub(dataset, val_idx)
        dra, _, _ = measure_depth_rank_accuracy(
            pipe, vca_layer, sub, device, n_samples=min(len(val_idx), 10))
    except Exception:
        dra = 0.0

    vs = val_score_phase42(iou_e0_m, iou_e1_m, ord_m, wl_m)
    return {
        "iou_e0":          iou_e0_m,
        "iou_e1":          iou_e1_m,
        "ordering_acc":    ord_m,
        "wrong_slot_leak": wl_m,
        "id_margin":       id_m,     # logging only
        "dra":             dra,
        "val_score":       vs,
    }


# =============================================================================
# Training loop
# =============================================================================

def train_phase42(args):
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
    print("[Phase 42] 파이프라인 로드 중...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    # ── Checkpoint 로드 ──────────────────────────────────────────────────────
    print(f"[Phase 42] checkpoint 로드: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location=device)
    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM, context_dim=768,
        n_heads=8, n_entities=2, z_bins=DEFAULT_Z_BINS, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=DEPTH_PE_INIT_SCALE,
    ).to(device)
    vca_layer.load_state_dict(ckpt["vca_state_dict"])
    print(f"  VCA loaded", flush=True)

    # ── 데이터셋 ─────────────────────────────────────────────────────────────
    print("[Phase 42] 데이터셋 로드 중...", flush=True)
    try:
        dataset = ObjaverseDatasetPhase40(args.data_root, n_frames=args.n_frames)
        print(f"  ObjaverseDatasetPhase40: {len(dataset)} samples", flush=True)
    except Exception as e:
        print(f"  Phase40 dataset 실패({e}), Phase39 dataset 사용", flush=True)
        dataset = ObjaverseDatasetWithMasks(args.data_root, n_frames=args.n_frames)

    if len(dataset) == 0:
        raise RuntimeError(f"데이터셋 비어있음: {args.data_root}")

    # ── Train/Val 분리 ───────────────────────────────────────────────────────
    print("[Phase 42] overlap score 계산 중...", flush=True)
    overlap_scores = compute_dataset_overlap_scores(dataset)
    train_idx, val_idx = split_train_val(overlap_scores, val_frac=args.val_frac)
    sample_weights = make_sampling_weights(train_idx, overlap_scores)
    print(f"  train={len(train_idx)}  val={len(val_idx)}", flush=True)

    # ── Phase42Processor 주입 ─────────────────────────────────────────────────
    init_sample = dataset[train_idx[0]]
    init_meta   = init_sample[3]
    init_ctx    = get_color_entity_context(pipe, init_meta, device)

    inject_keys = args.inject_keys.split(",") if args.inject_keys else DEFAULT_INJECT_KEYS
    procs, orig_procs = inject_multi_block_entity_slot_p42(
        pipe, vca_layer, init_ctx,
        inject_keys        = inject_keys,
        slot_blend_init    = args.slot_blend,
        adapter_rank       = args.adapter_rank,
        lora_rank          = args.lora_rank,
        use_blend_head     = True,
        weight_head_hidden = 32,
    )
    for p in procs:
        p.to(device)

    manager = MultiBlockSlotManagerP42(procs, inject_keys, primary_idx=1)

    # ── 수정 A: 모든 block state 완전 복원 ──────────────────────────────────
    print("[Phase 42] 모든 block state 복원 중...", flush=True)
    restore_multiblock_state(manager, ckpt, device=device, strict=False)
    print(f"  inject_keys={inject_keys}", flush=True)
    print(f"  adapter_rank={args.adapter_rank}  lora_rank={args.lora_rank}", flush=True)

    # ── 파라미터 동결 / 학습 대상 ────────────────────────────────────────────
    for p in pipe.unet.parameters():         p.requires_grad_(False)
    for p in pipe.text_encoder.parameters(): p.requires_grad_(False)
    for p in pipe.vae.parameters():          p.requires_grad_(False)
    for p in vca_layer.parameters():         p.requires_grad_(True)
    for proc in procs:
        for p in proc.parameters():          p.requires_grad_(True)

    # ── Optimizer ──────────────────────────────────────────────────────────
    param_groups = [
        {"params": list(vca_layer.parameters()),   "lr": args.lr_vca,         "name": "vca"},
        {"params": manager.adapter_params(),        "lr": args.lr_adapter,     "name": "adapters"},
        {"params": manager.lora_params(),           "lr": args.lr_lora,        "name": "lora"},
        {"params": manager.blend_params(),          "lr": args.lr_blend,       "name": "blend"},
        {"params": manager.weight_head_params(),    "lr": args.lr_weight_head, "name": "weight_head"},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_epochs, eta_min=args.lr_vca * 0.05)

    history        = []
    best_val_score = -1.0
    best_epoch     = -1

    print(f"\n[Phase 42] 훈련 시작: {args.total_epochs} epochs", flush=True)
    print(f"  λ_vis={args.lambda_vis}  λ_wrong={args.lambda_wrong}  "
          f"λ_sigma={args.lambda_sigma}  λ_depth={args.lambda_depth}", flush=True)
    print(f"  λ_slot_ref={args.lambda_slot_ref}  λ_slot_cont={args.lambda_slot_cont}  "
          f"λ_w_res={args.lambda_w_res}", flush=True)

    # ── Epoch 0 no-op validation — phase40_v2 수치 재현 확인 ──────────────────
    print("\n[Phase 42] Epoch 0 no-op validation (phase40_v2 재현 확인)...", flush=True)
    vca_layer.eval()
    manager.eval()
    val_m0 = evaluate_val_set_phase42(
        pipe, manager, vca_layer, dataset, val_idx, device,
        t_fixed=args.t_max // 2)
    print(f"  [epoch0-noop] iou_e0={val_m0['iou_e0']:.4f}  "
          f"iou_e1={val_m0['iou_e1']:.4f}  "
          f"ord={val_m0['ordering_acc']:.4f}  "
          f"wrong={val_m0['wrong_slot_leak']:.4f}  "
          f"id_margin={val_m0['id_margin']:.4f}  "
          f"val_score(p42)={val_m0['val_score']:.4f}", flush=True)
    print(f"  [참고] phase40_v2 best: iou_e0=0.0832  iou_e1=0.1012  "
          f"ord=0.4164  wrong=0.4332", flush=True)
    vca_layer.train()
    manager.train()

    for epoch in range(args.total_epochs):
        vca_layer.train()
        manager.train()

        epoch_losses = {k: [] for k in
                        ["total", "vis", "wrong", "sigma", "depth",
                         "ov", "excl", "slot_ref", "slot_cont", "w_res"]}

        chosen = np.random.choice(
            len(train_idx), size=args.steps_per_epoch,
            replace=True, p=sample_weights)
        step_indices = [train_idx[ci] for ci in chosen]

        for batch_idx, data_idx in enumerate(step_indices):
            sample = dataset[data_idx]
            if len(sample) >= 8:
                frames_np, _, depth_orders, meta, entity_masks, _, _, _ = sample
            else:
                frames_np, _, depth_orders, meta, entity_masks = sample

            entity_ctx = get_color_entity_context(pipe, meta, device)
            toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)

            manager.set_entity_ctx(entity_ctx.float())
            manager.set_entity_tokens(toks_e0, toks_e1)
            manager.reset_slot_store()

            with torch.no_grad():
                latents = encode_frames_to_latents(pipe, frames_np, device)
            noise = torch.randn_like(latents)
            t     = torch.randint(0, args.t_max, (1,), device=device).long()
            noisy = pipe.scheduler.add_noise(latents, noise, t)

            tok = pipe.tokenizer(
                full_prompt, return_tensors="pt", padding="max_length",
                max_length=pipe.tokenizer.model_max_length, truncation=True,
            ).to(device)
            with torch.no_grad():
                enc_hs = pipe.text_encoder(**tok).last_hidden_state.half()

            T_frames       = min(frames_np.shape[0], entity_masks.shape[0])
            masks_t        = torch.from_numpy(entity_masks[:T_frames].astype(np.float32)).to(device)
            depth_orders_t = depth_orders[:T_frames]

            # ── UNet forward ──────────────────────────────────────────────
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = pipe.unet(noisy, t, encoder_hidden_states=enc_hs).sample

            BF    = manager.last_w0.shape[0] if manager.last_w0 is not None else 0
            T_use = min(BF, T_frames)

            # ── PRIMARY BLOCK LOSSES ──────────────────────────────────────

            # L_visible_weights
            l_vis = torch.tensor(0.0, device=device)
            if manager.last_w0 is not None:
                for fi in range(T_use):
                    w0_f = manager.last_w0[fi:fi+1].float()
                    w1_f = manager.last_w1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    do   = [depth_orders_t[fi]] if fi < len(depth_orders_t) else [(0, 1)]
                    l_vis = l_vis + l_visible_weights(w0_f, w1_f, m_f, do)
                l_vis = l_vis / max(T_use, 1)

            # L_wrong_slot
            l_wrong = torch.tensor(0.0, device=device)
            if manager.last_w0 is not None:
                for fi in range(T_use):
                    w0_f = manager.last_w0[fi:fi+1].float()
                    w1_f = manager.last_w1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    l_wrong = l_wrong + l_wrong_slot_suppression(w0_f, w1_f, m_f)
                l_wrong = l_wrong / max(T_use, 1)

            # L_sigma_spatial (class-balanced)
            l_sigma = torch.tensor(0.0, device=device)
            if manager.last_alpha0 is not None:
                for fi in range(T_use):
                    a0_f = manager.last_alpha0[fi:fi+1].float()
                    a1_f = manager.last_alpha1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    l_sigma = l_sigma + l_sigma_spatial(a0_f, a1_f, m_f)
                l_sigma = l_sigma / max(T_use, 1)

                if epoch == 0 and batch_idx == 0:
                    a0_all = manager.last_alpha0.float()
                    m0_all = masks_t[:T_use, 0, :].float()
                    alpha_ent = (a0_all[:T_use][m0_all.bool()].mean().item()
                                 if m0_all.any() else 0.0)
                    alpha_bg  = a0_all[:T_use][~m0_all.bool()].mean().item()
                    n_ent     = int(m0_all.sum().item())
                    print(f"  [debug alpha] ep0 step0: alpha0_entity={alpha_ent:.4f} "
                          f"alpha0_bg={alpha_bg:.4f} n_entity={n_ent}", flush=True)

            # L_depth
            sigma_acc = list(manager.sigma_acc)
            l_depth   = l_zorder_direct(sigma_acc, depth_orders_t, entity_masks[:T_frames])

            # L_overlap_ordering
            l_ov = torch.tensor(0.0, device=device)
            if manager.last_w0 is not None:
                for fi in range(T_use):
                    w0_f = manager.last_w0[fi:fi+1].float()
                    w1_f = manager.last_w1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    do   = [depth_orders_t[fi]] if fi < len(depth_orders_t) else [(0, 1)]
                    l_ov = l_ov + l_overlap_ordering(w0_f, w1_f, m_f, do)
                l_ov = l_ov / max(T_use, 1)

            # L_w_residual (WeightHead regularizer, primary only)
            l_w_res = torch.tensor(0.0, device=device)
            if manager.last_w_delta is not None:
                l_w_res = l_w_residual(manager.last_w_delta)

            # ── ALL BLOCK LOSSES ──────────────────────────────────────────

            # Primary block F0/F1 as reference (detached)
            primary = manager.primary
            F0_primary = (primary.last_F0.detach()
                          if primary.last_F0 is not None else None)
            F1_primary = (primary.last_F1.detach()
                          if primary.last_F1 is not None else None)

            l_excl     = torch.tensor(0.0, device=device)
            l_slot_ref_all  = torch.tensor(0.0, device=device)
            l_slot_cont_all = torch.tensor(0.0, device=device)

            n_blk_terms = 0

            for blk_idx, (proc, blk_F0, blk_F1, blk_Fg) in enumerate(zip(
                    manager.procs,
                    manager.all_F0(), manager.all_F1(), manager.all_Fg())):

                if blk_F0 is None or blk_F1 is None or blk_Fg is None:
                    continue

                S_blk  = blk_F0.shape[1]
                S_mask = masks_t.shape[-1]

                # resize masks to block spatial size if needed
                if S_blk != S_mask:
                    H_blk = int(S_blk ** 0.5)
                    H_msk = int(S_mask ** 0.5)
                    m_4d  = masks_t.view(masks_t.shape[0], 2, H_msk, H_msk)
                    m_4d  = torch.nn.functional.interpolate(
                        m_4d.float(), size=(H_blk, H_blk), mode='nearest')
                    masks_blk = m_4d.view(masks_t.shape[0], 2, S_blk)
                else:
                    masks_blk = masks_t

                for fi in range(min(blk_F0.shape[0], T_frames)):
                    # L_exclusive (all blocks)
                    l_excl = l_excl + l_entity_exclusive(
                        blk_F0[fi:fi+1].float(),
                        blk_F1[fi:fi+1].float(),
                        blk_Fg[fi:fi+1].float(),
                        masks_blk[fi:fi+1])

                    # L_slot_ref + L_slot_contrast: secondary blocks → primary consistency
                    if blk_idx != manager.primary_idx and F0_primary is not None:
                        # spatial resize if needed (handles different H×W per block)
                        F0_ref_s = (F0_primary[fi:fi+1]
                                    if F0_primary.shape[1] == S_blk
                                    else _resize_feat(F0_primary[fi:fi+1], S_blk))
                        F1_ref_s = (F1_primary[fi:fi+1]
                                    if F1_primary.shape[1] == S_blk
                                    else _resize_feat(F1_primary[fi:fi+1], S_blk))

                        vis_e0 = masks_blk[fi, 0, :]
                        vis_e1 = masks_blk[fi, 1, :]

                        # L_slot_ref: MSE only when feature dims match
                        if blk_F0.shape[-1] == F0_ref_s.shape[-1]:
                            l_slot_ref_all = l_slot_ref_all + \
                                l_slot_ref(blk_F0[fi:fi+1].float(),
                                           F0_ref_s, vis_e0.unsqueeze(0))
                            l_slot_ref_all = l_slot_ref_all + \
                                l_slot_ref(blk_F1[fi:fi+1].float(),
                                           F1_ref_s, vis_e1.unsqueeze(0))

                        # L_slot_contrast: normalize → dim-agnostic
                        # project ref to block dim if needed via average pooling
                        D_blk = blk_F0.shape[-1]
                        D_ref = F0_ref_s.shape[-1]
                        if D_blk != D_ref:
                            # align ref dim: chunk-average primary features
                            ratio = D_ref // D_blk
                            if ratio >= 1 and D_ref % D_blk == 0:
                                F0_ref_c = F0_ref_s.view(
                                    F0_ref_s.shape[0], S_blk, D_blk, ratio
                                ).mean(dim=-1)
                                F1_ref_c = F1_ref_s.view(
                                    F1_ref_s.shape[0], S_blk, D_blk, ratio
                                ).mean(dim=-1)
                            else:
                                # can't align easily — skip contrast for this block
                                continue
                        else:
                            F0_ref_c = F0_ref_s
                            F1_ref_c = F1_ref_s

                        l_slot_cont_all = l_slot_cont_all + l_slot_contrast(
                            blk_F0[fi:fi+1].float(), blk_F1[fi:fi+1].float(),
                            F0_ref_c, F1_ref_c,
                            masks_blk[fi:fi+1, 0, :], masks_blk[fi:fi+1, 1, :])

                n_blk_terms += 1

            if T_use > 0 and n_blk_terms > 0:
                l_excl         = l_excl         / max(T_use * n_blk_terms, 1)
                l_slot_ref_all = l_slot_ref_all / max(T_use * (n_blk_terms - 1), 1) \
                    if n_blk_terms > 1 else l_slot_ref_all * 0
                l_slot_cont_all = l_slot_cont_all / max(T_use * (n_blk_terms - 1), 1) \
                    if n_blk_terms > 1 else l_slot_cont_all * 0

            # ── Total loss ────────────────────────────────────────────────
            loss = (args.lambda_vis        * l_vis
                  + args.lambda_wrong      * l_wrong
                  + args.lambda_sigma      * l_sigma
                  + args.lambda_depth      * l_depth
                  + args.lambda_ov         * l_ov
                  + args.lambda_excl       * l_excl
                  + args.lambda_slot_ref   * l_slot_ref_all
                  + args.lambda_slot_cont  * l_slot_cont_all
                  + args.lambda_w_res      * l_w_res)

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss ep={epoch} step={batch_idx} → skip",
                      flush=True)
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(vca_layer.parameters()),    max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(manager.adapter_params(),        max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(manager.lora_params(),           max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(manager.blend_params(),          max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(manager.weight_head_params(),    max_norm=1.0)

            optimizer.step()

            def _f(v): return float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            epoch_losses["total"].append(_f(loss))
            epoch_losses["vis"].append(_f(l_vis))
            epoch_losses["wrong"].append(_f(l_wrong))
            epoch_losses["sigma"].append(_f(l_sigma))
            epoch_losses["depth"].append(_f(l_depth))
            epoch_losses["ov"].append(_f(l_ov))
            epoch_losses["excl"].append(_f(l_excl))
            epoch_losses["slot_ref"].append(_f(l_slot_ref_all))
            epoch_losses["slot_cont"].append(_f(l_slot_cont_all))
            epoch_losses["w_res"].append(_f(l_w_res))

        scheduler.step()

        avg = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
        blend_val = float(primary.slot_blend.item())

        print(f"[Phase 42] epoch {epoch:03d}/{args.total_epochs-1}  "
              f"loss={avg['total']:.4f}  vis={avg['vis']:.4f}  "
              f"wrong={avg['wrong']:.4f}  sigma={avg['sigma']:.4f}  "
              f"depth={avg['depth']:.4f}  "
              f"slot_ref={avg['slot_ref']:.4f}  w_res={avg['w_res']:.4f}  "
              f"blend={blend_val:.4f}", flush=True)

        # ── Validation ────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.total_epochs - 1:
            vca_layer.eval()
            manager.eval()

            val_m = evaluate_val_set_phase42(
                pipe, manager, vca_layer, dataset, val_idx, device,
                t_fixed=args.t_max // 2)
            vs = val_m["val_score"]

            print(f"  [val] iou_e0={val_m['iou_e0']:.4f}  iou_e1={val_m['iou_e1']:.4f}  "
                  f"ord={val_m['ordering_acc']:.4f}  wrong={val_m['wrong_slot_leak']:.4f}  "
                  f"id_margin(log)={val_m['id_margin']:.4f}  "
                  f"val_score_p42={vs:.4f}", flush=True)

            rollout_m = {"rollout_iou_e0": 0.0, "rollout_iou_e1": 0.0}
            if epoch % (args.eval_every * 2) == 0 or epoch == args.total_epochs - 1:
                rollout_m = evaluate_rollout(
                    pipe, manager, dataset, val_idx, device,
                    t_start=ROLLOUT_T_START, n_steps=ROLLOUT_N_STEPS)
                print(f"  [rollout] iou_e0={rollout_m['rollout_iou_e0']:.4f}  "
                      f"iou_e1={rollout_m['rollout_iou_e1']:.4f}", flush=True)

            # GIF
            try:
                probe_sample = dataset[val_idx[0]]
                probe_meta   = probe_sample[3]
                probe_ctx    = get_color_entity_context(pipe, probe_meta, device)
                probe_e0, probe_e1, probe_prompt = get_entity_token_positions(
                    pipe, probe_meta)
                manager.set_entity_ctx(probe_ctx.float())
                manager.set_entity_tokens(probe_e0, probe_e1)
                manager.reset_slot_store()
                gen = torch.Generator(device=device).manual_seed(args.eval_seed)
                out = pipe(
                    prompt=probe_prompt, num_frames=args.n_frames,
                    num_inference_steps=args.n_steps,
                    height=args.height, width=args.width,
                    generator=gen, output_type="np",
                )
                frames_eval = (out.frames[0] * 255).astype(np.uint8)
                gif_path = debug_dir / f"eval_epoch{epoch:03d}.gif"
                iio2.mimwrite(str(gif_path), frames_eval, fps=8, loop=0)
                es_dbg, sr_dbg, cr_dbg = compute_entity_score_debug(frames_eval)
                print(f"  [debug] entity_score={es_dbg:.4f} "
                      f"survival={sr_dbg:.4f} chimera={cr_dbg:.4f}", flush=True)
            except Exception as e:
                print(f"  [warn] GIF 생성 실패: {e}", flush=True)
                frames_eval = None

            history.append({
                "epoch": epoch,
                "val_score": vs,
                **val_m, **rollout_m, **avg,
                "slot_blend": blend_val,
            })

            # ── Checkpoint 저장 ───────────────────────────────────────────
            ckpt_data = {
                "epoch":          epoch,
                "vca_state_dict": vca_layer.state_dict(),
                "val_score":      vs,
                "iou_e0":         val_m["iou_e0"],
                "iou_e1":         val_m["iou_e1"],
                "ordering_acc":   val_m["ordering_acc"],
                "wrong_slot_leak": val_m["wrong_slot_leak"],
                "inject_keys":    inject_keys,
                "adapter_rank":   args.adapter_rank,
                "lora_rank":      args.lora_rank,
                # 모든 block state 저장 (phase42 표준)
                "procs_state": [
                    {
                        "slot_blend_raw": p.slot_blend_raw.detach().cpu(),
                        "slot0_adapter":  p.slot0_adapter.state_dict(),
                        "slot1_adapter":  p.slot1_adapter.state_dict(),
                        "blend_head":     p.blend_head.state_dict(),
                        "lora_k":         p.lora_k.state_dict(),
                        "lora_v":         p.lora_v.state_dict(),
                        "lora_out":       p.lora_out.state_dict(),
                        "weight_head":    p.weight_head.state_dict(),
                    }
                    for p in procs
                ],
            }
            torch.save(ckpt_data, str(save_dir / "latest.pt"))

            if vs > best_val_score:
                best_val_score = vs
                best_epoch     = epoch
                torch.save(ckpt_data, str(save_dir / "best.pt"))
                print(f"  ★ best epoch={best_epoch}  val_score={vs:.4f}  "
                      f"(iou_e0={val_m['iou_e0']:.4f} "
                      f"iou_e1={val_m['iou_e1']:.4f} "
                      f"ord={val_m['ordering_acc']:.4f} "
                      f"wrong={val_m['wrong_slot_leak']:.4f})  "
                      f"→ {save_dir}/best.pt", flush=True)

            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            vca_layer.train()
            manager.train()

    print(f"\n[Phase 42] 훈련 완료. best epoch={best_epoch} "
          f"val_score={best_val_score:.4f}", flush=True)

    if best_epoch == 0:
        print("[Phase 42] WARNING: best epoch=0 — 학습이 val_score를 개선하지 못함.",
              flush=True)
        print("  → weight_head가 아닌 dataset/visible-target이 병목일 가능성.",
              flush=True)


# =============================================================================
# Helper
# =============================================================================

def _resize_feat(feat: torch.Tensor, target_S: int) -> torch.Tensor:
    """
    (B, S, D) feature를 target spatial size로 interpolate.
    S_src ≠ target_S 일 때만 호출됨.
    """
    B, S, D = feat.shape
    H_src = int(S ** 0.5)
    H_tgt = int(target_S ** 0.5)
    if H_src * H_src != S or H_tgt * H_tgt != target_S:
        return feat[:, :target_S, :] if S >= target_S else feat.expand(-1, target_S, -1)
    x = feat.permute(0, 2, 1).reshape(B, D, H_src, H_src)
    x = torch.nn.functional.interpolate(x, size=(H_tgt, H_tgt), mode='bilinear',
                                         align_corners=False)
    return x.reshape(B, D, target_S).permute(0, 2, 1).contiguous()


# =============================================================================
# argparse
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Phase 42: Multi-Block Restore + Residual Weight Head")

    # paths
    p.add_argument("--ckpt",         type=str, default="checkpoints/phase40_v2/best.pt")
    p.add_argument("--data-root",    type=str, default="toy/data_objaverse")
    p.add_argument("--save-dir",     type=str, default="checkpoints/phase42")
    p.add_argument("--debug-dir",    type=str, default="outputs/phase42_debug")

    # training
    p.add_argument("--total-epochs",    type=int, default=DEFAULT_TOTAL_EPOCHS)
    p.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    p.add_argument("--n-frames",        type=int, default=8)
    p.add_argument("--n-steps",         type=int, default=20)
    p.add_argument("--t-max",           type=int, default=300)
    p.add_argument("--height",          type=int, default=256)
    p.add_argument("--width",           type=int, default=256)

    # architecture
    p.add_argument("--adapter-rank",  type=int,   default=DEFAULT_ADAPTER_RANK)
    p.add_argument("--lora-rank",     type=int,   default=DEFAULT_LORA_RANK)
    p.add_argument("--slot-blend",    type=float, default=DEFAULT_SLOT_BLEND)
    p.add_argument("--inject-keys",   type=str,   default=None)

    # LR
    p.add_argument("--lr-vca",         type=float, default=DEFAULT_LR_VCA)
    p.add_argument("--lr-adapter",     type=float, default=DEFAULT_LR_ADAPTER)
    p.add_argument("--lr-lora",        type=float, default=DEFAULT_LR_LORA)
    p.add_argument("--lr-blend",       type=float, default=DEFAULT_LR_BLEND)
    p.add_argument("--lr-weight-head", type=float, default=DEFAULT_LR_WEIGHT_HEAD)

    # lambdas
    p.add_argument("--lambda-vis",       type=float, default=DEFAULT_LAMBDA_VIS)
    p.add_argument("--lambda-wrong",     type=float, default=DEFAULT_LAMBDA_WRONG)
    p.add_argument("--lambda-sigma",     type=float, default=DEFAULT_LAMBDA_SIGMA)
    p.add_argument("--lambda-depth",     type=float, default=DEFAULT_LAMBDA_DEPTH)
    p.add_argument("--lambda-ov",        type=float, default=DEFAULT_LAMBDA_OV)
    p.add_argument("--lambda-excl",      type=float, default=DEFAULT_LAMBDA_EXCL)
    p.add_argument("--lambda-slot-ref",  type=float, default=DEFAULT_LAMBDA_SLOT_REF)
    p.add_argument("--lambda-slot-cont", type=float, default=DEFAULT_LAMBDA_SLOT_CONT)
    p.add_argument("--lambda-w-res",     type=float, default=DEFAULT_LAMBDA_W_RES)

    # eval
    p.add_argument("--val-frac",   type=float, default=VAL_FRAC)
    p.add_argument("--eval-every", type=int,   default=5)
    p.add_argument("--eval-seed",  type=int,   default=42)
    p.add_argument("--seed",       type=int,   default=42)

    args = p.parse_args()
    train_phase42(args)


if __name__ == "__main__":
    main()
