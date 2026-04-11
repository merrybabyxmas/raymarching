"""
Phase 41 — Spatial Localization Fix (Class-Balanced Sigma Supervision)
=======================================================================

Phase 40 분석 결과:
  iou_e0/e1 stuck at 0.09-0.12 despite 80 epochs of l_sigma_spatial supervision.

Root cause:
  l_sigma_spatial 이전 버전 = F.mse_loss(alpha0, mask0) — uniform average over
  all 256 spatial positions. Entity coverage ≈ 4% (10/256 pixels).
  → 96% background gradient "alpha→0" 이 4% entity gradient를 압도.
  → model이 alpha≈0 everywhere 학습 (MSE-optimal) → IoU 고착.

Phase 41 fixes (models/entity_slot.py 수정):
  1. l_sigma_spatial: class-balanced MSE
     entity pixel weight=1, bg pixel weight = n_entity/n_bg
     → entity/bg total gradient 균형 → model이 alpha→0 shortcut 불가
  2. l_visible_weights: + background suppression (bg_weight=0.2)
     기존 코드는 entity 픽셀만 감독 → background w0/w1 무제약 상승 허용
     Phase 41: bg 픽셀에서 w0,w1→0 penalty 추가

Phase 41 simplification:
  - Single stage (60 epochs) — S2/S3 solo distillation 제거
    (Solo distillation은 id_margin 악화시키고 IoU 개선 못함 → Phase 41 이후 재검토)
  - Higher lambda_sigma = 5.0 (from 2.0)
  - Same architecture: 3-block injection, SlotLoRA rank-4

성공 기준
---------
  visible_iou_e0 > 0.30
  visible_iou_e1 > 0.30
  ordering_acc   > 0.65
  wrong_slot_leak < 0.15
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
    Phase40Processor,
    MultiBlockSlotManager,
    inject_multi_block_entity_slot,
    compute_visible_masks,
    compute_visible_iou_e0,
    compute_visible_iou_e1,
    val_score_phase40,
    DEFAULT_INJECT_KEYS,
    BLOCK_INNER_DIMS,
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


# ─── Phase 41 하이퍼파라미터 ──────────────────────────────────────────────────
DEFAULT_TOTAL_EPOCHS = 60       # single stage (no S2/S3)
DEFAULT_LAMBDA_VIS    = 2.0
DEFAULT_LAMBDA_WRONG  = 1.0
DEFAULT_LAMBDA_SIGMA  = 5.0     # ↑ from 2.0 — class-balanced loss allows higher weight
DEFAULT_LAMBDA_DEPTH  = 2.0
DEFAULT_LAMBDA_OV     = 1.0
DEFAULT_LAMBDA_EXCL   = 0.5

DEFAULT_LR_VCA      = 2e-5
DEFAULT_LR_ADAPTER  = 1e-4
DEFAULT_LR_LORA     = 5e-5
DEFAULT_LR_BLEND    = 3e-4

DEFAULT_ADAPTER_RANK     = 64
DEFAULT_LORA_RANK        = 4
DEFAULT_SLOT_BLEND       = 0.3
DEFAULT_STEPS_PER_EPOCH  = 20

ROLLOUT_T_START  = 250
ROLLOUT_N_STEPS  = 5
VAL_FRAC         = 0.2
MIN_VAL_SAMPLES  = 4


# =============================================================================
# Rollout evaluation (Phase 40 code, unchanged)
# =============================================================================

@torch.no_grad()
def evaluate_rollout(
    pipe,
    manager:    MultiBlockSlotManager,
    dataset,
    val_idx:    list,
    device:     str,
    t_start:    int = ROLLOUT_T_START,
    n_steps:    int = ROLLOUT_N_STEPS,
) -> dict:
    manager.eval()
    rollout_iou_e0s = []
    rollout_iou_e1s = []
    rollout_id_scores = []

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
            t_indices = (ts <= t_start).nonzero(as_tuple=True)[0]
            if len(t_indices) == 0:
                continue
            t_start_idx = t_indices[0].item()
            t_end_idx   = min(t_start_idx + n_steps, len(ts))

            for step_t in ts[t_start_idx:t_end_idx]:
                step_t_batch = torch.tensor([step_t], device=device).long()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    noise_pred = pipe.unet(x, step_t_batch, encoder_hidden_states=enc_hs).sample
                x = scheduler_state.step(noise_pred, step_t, x).prev_sample

            manager.reset_slot_store()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = pipe.unet(x, ts[t_end_idx - 1:t_end_idx], encoder_hidden_states=enc_hs).sample

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

            proc = manager.primary
            if proc.last_F0 is not None and proc.last_F1 is not None:
                T_use = min(proc.last_F0.shape[0], T_frames)
                for fi in range(T_use):
                    F0_f = proc.last_F0[fi:fi+1].float()
                    F1_f = proc.last_F1[fi:fi+1].float()
                    F0_n = torch.nn.functional.normalize(F0_f, dim=-1)
                    F1_n = torch.nn.functional.normalize(F1_f, dim=-1)
                    cross_sim = (F0_n * F1_n).sum(-1).mean().item()
                    rollout_id_scores.append(1.0 - cross_sim)

        except Exception as e:
            print(f"  [rollout warn] val {vi}: {e}", flush=True)
            continue

    manager.train()
    if not rollout_iou_e0s:
        return {"rollout_iou_e0": 0.0, "rollout_iou_e1": 0.0, "rollout_id_score": 0.0}
    return {
        "rollout_iou_e0":   float(np.mean(rollout_iou_e0s)),
        "rollout_iou_e1":   float(np.mean(rollout_iou_e1s)),
        "rollout_id_score": float(np.mean(rollout_id_scores)) if rollout_id_scores else 0.0,
    }


# =============================================================================
# Teacher-forced validation
# =============================================================================

@torch.no_grad()
def evaluate_val_set_phase41(
    pipe,
    manager:    MultiBlockSlotManager,
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

                if proc.last_F0 is not None and proc.last_F1 is not None:
                    F0_f = proc.last_F0[fi:fi+1].float()
                    F1_f = proc.last_F1[fi:fi+1].float()
                    F0_n = torch.nn.functional.normalize(F0_f, dim=-1)
                    F1_n = torch.nn.functional.normalize(F1_f, dim=-1)
                    cross_sim = (F0_n * F1_n).sum(-1).mean().item()
                    id_margins.append(1.0 - cross_sim)

        except Exception as e:
            print(f"  [val warn] {vi}: {e}", flush=True)
            continue

    manager.train()

    if not iou_e0s:
        return {"iou_e0": 0.0, "iou_e1": 0.0, "ordering_acc": 0.0,
                "wrong_slot_leak": 1.0, "id_margin": 0.0, "dra": 0.0, "val_score": 0.0}

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

    vs = val_score_phase40(iou_e0_m, iou_e1_m, ord_m, wl_m, id_m)
    return {
        "iou_e0":          iou_e0_m,
        "iou_e1":          iou_e1_m,
        "ordering_acc":    ord_m,
        "wrong_slot_leak": wl_m,
        "id_margin":       id_m,
        "dra":             dra,
        "val_score":       vs,
    }


# =============================================================================
# Training loop
# =============================================================================

def train_phase41(args):
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
    print("[Phase 41] 파이프라인 로드 중...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    # ── VCA 로드 ─────────────────────────────────────────────────────────────
    print(f"[Phase 41] checkpoint 로드: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location=device)
    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM, context_dim=768,
        n_heads=8, n_entities=2, z_bins=DEFAULT_Z_BINS, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=DEPTH_PE_INIT_SCALE,
    ).to(device)
    vca_layer.load_state_dict(ckpt["vca_state_dict"])
    print(f"  VCA loaded", flush=True)

    # ── 데이터셋 ─────────────────────────────────────────────────────────────
    print("[Phase 41] 데이터셋 로드 중...", flush=True)
    try:
        dataset = ObjaverseDatasetPhase40(args.data_root, n_frames=args.n_frames)
        print(f"  ObjaverseDatasetPhase40: {len(dataset)} samples", flush=True)
    except Exception as e:
        print(f"  Phase40 dataset 실패({e}), Phase39 dataset 사용", flush=True)
        dataset = ObjaverseDatasetWithMasks(args.data_root, n_frames=args.n_frames)

    if len(dataset) == 0:
        raise RuntimeError(f"데이터셋 비어있음: {args.data_root}")

    # ── Train/Val 분리 ───────────────────────────────────────────────────────
    print("[Phase 41] overlap score 계산 중...", flush=True)
    overlap_scores = compute_dataset_overlap_scores(dataset)
    train_idx, val_idx = split_train_val(overlap_scores, val_frac=args.val_frac)
    sample_weights = make_sampling_weights(train_idx, overlap_scores)
    print(f"  train={len(train_idx)}  val={len(val_idx)}", flush=True)

    # ── Multi-block injection ─────────────────────────────────────────────────
    init_sample = dataset[train_idx[0]]
    init_meta   = init_sample[3]
    init_ctx    = get_color_entity_context(pipe, init_meta, device)

    inject_keys = args.inject_keys.split(",") if args.inject_keys else DEFAULT_INJECT_KEYS
    procs, orig_procs = inject_multi_block_entity_slot(
        pipe, vca_layer, init_ctx,
        inject_keys     = inject_keys,
        slot_blend_init  = args.slot_blend,
        adapter_rank    = args.adapter_rank,
        lora_rank       = args.lora_rank,
        use_blend_head  = True,
    )
    manager = MultiBlockSlotManager(procs, inject_keys, primary_idx=1)
    for p in procs:
        p.to(device)

    # ── Phase 40 상태 복원 ────────────────────────────────────────────────────
    primary = manager.primary
    if "slot_blend_raw" in ckpt:
        primary.slot_blend_raw.data.copy_(ckpt["slot_blend_raw"].to(device))
        print(f"  slot_blend_raw 복원: {float(primary.slot_blend.item()):.4f}", flush=True)
    for key_name in ("slot0_adapter", "slot1_adapter", "blend_head"):
        if key_name in ckpt:
            try:
                getattr(primary, key_name).load_state_dict(ckpt[key_name])
                print(f"  {key_name} 복원", flush=True)
            except Exception as e:
                print(f"  [warn] {key_name} 복원 실패: {e}", flush=True)

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
        {"params": list(vca_layer.parameters()),   "lr": args.lr_vca,     "name": "vca"},
        {"params": manager.adapter_params(),        "lr": args.lr_adapter, "name": "adapters"},
        {"params": manager.lora_params(),           "lr": args.lr_lora,    "name": "lora"},
        {"params": manager.blend_params(),          "lr": args.lr_blend,   "name": "blend"},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_epochs, eta_min=args.lr_vca * 0.05)

    # ── 학습 기록 ────────────────────────────────────────────────────────────
    history        = []
    best_val_score = -1.0
    best_epoch     = -1

    print(f"\n[Phase 41] 훈련 시작: {args.total_epochs} epochs (single stage)", flush=True)
    print(f"  λ_vis={args.lambda_vis}  λ_wrong={args.lambda_wrong}  "
          f"λ_sigma={args.lambda_sigma}  λ_depth={args.lambda_depth}", flush=True)
    print(f"  Key fix: class-balanced l_sigma_spatial + l_vis background suppression",
          flush=True)

    for epoch in range(args.total_epochs):
        vca_layer.train()
        manager.train()

        epoch_losses = {k: [] for k in
                        ["total", "vis", "wrong", "sigma", "depth", "ov", "excl"]}

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

            # ── L_visible_weights (Phase 41: + bg suppression) ───────────
            l_vis = torch.tensor(0.0, device=device)
            if manager.last_w0 is not None:
                for fi in range(T_use):
                    w0_f = manager.last_w0[fi:fi+1].float()
                    w1_f = manager.last_w1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    do   = [depth_orders_t[fi]] if fi < len(depth_orders_t) else [(0, 1)]
                    l_vis = l_vis + l_visible_weights(w0_f, w1_f, m_f, do)
                l_vis = l_vis / max(T_use, 1)

            # ── L_wrong_slot ──────────────────────────────────────────────
            l_wrong = torch.tensor(0.0, device=device)
            if manager.last_w0 is not None:
                for fi in range(T_use):
                    w0_f = manager.last_w0[fi:fi+1].float()
                    w1_f = manager.last_w1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    l_wrong = l_wrong + l_wrong_slot_suppression(w0_f, w1_f, m_f)
                l_wrong = l_wrong / max(T_use, 1)

            # ── L_sigma_spatial (Phase 41: class-balanced) ───────────────
            l_sigma = torch.tensor(0.0, device=device)
            if manager.last_alpha0 is not None:
                for fi in range(T_use):
                    a0_f = manager.last_alpha0[fi:fi+1].float()
                    a1_f = manager.last_alpha1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    l_sigma = l_sigma + l_sigma_spatial(a0_f, a1_f, m_f)
                l_sigma = l_sigma / max(T_use, 1)

                # Debug: log alpha0 values at first epoch to verify fix
                if epoch == 0 and batch_idx == 0:
                    a0_all = manager.last_alpha0.float()
                    m0_all = masks_t[:T_use, 0, :].float()
                    alpha_ent = a0_all[:T_use][m0_all.bool()].mean().item() if m0_all.any() else 0.0
                    alpha_bg  = a0_all[:T_use][~m0_all.bool()].mean().item()
                    n_ent     = int(m0_all.sum().item())
                    print(f"  [debug alpha] ep0 step0: alpha0_entity={alpha_ent:.4f} "
                          f"alpha0_bg={alpha_bg:.4f} n_entity={n_ent}", flush=True)

            # ── L_depth ───────────────────────────────────────────────────
            sigma_acc = list(manager.sigma_acc)
            l_depth   = l_zorder_direct(sigma_acc, depth_orders_t, entity_masks[:T_frames])

            # ── L_overlap_ordering ────────────────────────────────────────
            l_ov = torch.tensor(0.0, device=device)
            if manager.last_w0 is not None:
                for fi in range(T_use):
                    w0_f = manager.last_w0[fi:fi+1].float()
                    w1_f = manager.last_w1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    do   = [depth_orders_t[fi]] if fi < len(depth_orders_t) else [(0, 1)]
                    l_ov = l_ov + l_overlap_ordering(w0_f, w1_f, m_f, do)
                l_ov = l_ov / max(T_use, 1)

            # ── L_entity_exclusive ────────────────────────────────────────
            l_excl = torch.tensor(0.0, device=device)
            for blk_F0, blk_F1, blk_Fg in zip(
                    manager.all_F0(), manager.all_F1(), manager.all_Fg()):
                if blk_F0 is not None and blk_F1 is not None and blk_Fg is not None:
                    S_blk  = blk_F0.shape[1]
                    S_mask = masks_t.shape[-1]
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
                        l_excl = l_excl + l_entity_exclusive(
                            blk_F0[fi:fi+1].float(),
                            blk_F1[fi:fi+1].float(),
                            blk_Fg[fi:fi+1].float(),
                            masks_blk[fi:fi+1])
            if T_use > 0 and len(manager.procs) > 0:
                l_excl = l_excl / max(T_use * len(manager.procs), 1)

            # ── Total loss ────────────────────────────────────────────────
            loss = (args.lambda_vis   * l_vis
                  + args.lambda_wrong * l_wrong
                  + args.lambda_sigma * l_sigma
                  + args.lambda_depth * l_depth
                  + args.lambda_ov    * l_ov
                  + args.lambda_excl  * l_excl)

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss ep={epoch} step={batch_idx} → skip",
                      flush=True)
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(vca_layer.parameters()), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(manager.adapter_params(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(manager.lora_params(),    max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(manager.blend_params(),   max_norm=5.0)

            optimizer.step()

            def _f(v): return float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            epoch_losses["total"].append(_f(loss))
            epoch_losses["vis"].append(_f(l_vis))
            epoch_losses["wrong"].append(_f(l_wrong))
            epoch_losses["sigma"].append(_f(l_sigma))
            epoch_losses["depth"].append(_f(l_depth))
            epoch_losses["ov"].append(_f(l_ov))
            epoch_losses["excl"].append(_f(l_excl))

        scheduler.step()

        avg = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
        blend_val = float(primary.slot_blend.item())

        print(f"[Phase 41] epoch {epoch:03d}/{args.total_epochs-1}  "
              f"loss={avg['total']:.4f}  vis={avg['vis']:.4f}  "
              f"wrong={avg['wrong']:.4f}  sigma={avg['sigma']:.4f}  "
              f"depth={avg['depth']:.4f}  blend={blend_val:.4f}", flush=True)

        # ── Validation ────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.total_epochs - 1:
            vca_layer.eval()
            manager.eval()

            val_m = evaluate_val_set_phase41(
                pipe, manager, vca_layer, dataset, val_idx, device,
                t_fixed=args.t_max // 2)
            vs = val_m["val_score"]

            print(f"  [val] iou_e0={val_m['iou_e0']:.4f}  iou_e1={val_m['iou_e1']:.4f}  "
                  f"ord={val_m['ordering_acc']:.4f}  wrong={val_m['wrong_slot_leak']:.4f}  "
                  f"id_margin={val_m['id_margin']:.4f}  dra={val_m['dra']:.4f}  "
                  f"val_score={vs:.4f}", flush=True)

            # Rollout eval every 2 × eval_every
            rollout_m = {"rollout_iou_e0": 0.0, "rollout_iou_e1": 0.0,
                         "rollout_id_score": 0.0}
            if epoch % (args.eval_every * 2) == 0 or epoch == args.total_epochs - 1:
                rollout_m = evaluate_rollout(
                    pipe, manager, dataset, val_idx, device,
                    t_start=ROLLOUT_T_START, n_steps=ROLLOUT_N_STEPS)
                print(f"  [rollout] iou_e0={rollout_m['rollout_iou_e0']:.4f}  "
                      f"iou_e1={rollout_m['rollout_iou_e1']:.4f}  "
                      f"id_score={rollout_m['rollout_id_score']:.4f}", flush=True)

            # full val_score with rollout
            vs_full = val_score_phase40(
                val_m["iou_e0"], val_m["iou_e1"], val_m["ordering_acc"],
                val_m["wrong_slot_leak"], val_m["id_margin"],
                rollout_m["rollout_id_score"])

            # GIF
            try:
                probe_sample = dataset[val_idx[0]]
                probe_meta   = probe_sample[3]
                probe_ctx    = get_color_entity_context(pipe, probe_meta, device)
                probe_e0, probe_e1, probe_prompt = get_entity_token_positions(pipe, probe_meta)
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
                "val_score": vs_full,
                **val_m, **rollout_m, **avg,
                "slot_blend": blend_val,
            })

            ckpt_data = {
                "epoch":          epoch,
                "vca_state_dict": vca_layer.state_dict(),
                "slot_blend_raw": primary.slot_blend_raw.detach().cpu(),
                "slot_blend":     blend_val,
                "slot0_adapter":  primary.slot0_adapter.state_dict(),
                "slot1_adapter":  primary.slot1_adapter.state_dict(),
                "blend_head":     primary.blend_head.state_dict(),
                "lora_k":         primary.lora_k.state_dict(),
                "lora_v":         primary.lora_v.state_dict(),
                "lora_out":       primary.lora_out.state_dict(),
                "procs_state": [
                    {k: v.detach().cpu() if hasattr(v, 'cpu') else v
                     for k, v in {
                         "slot_blend_raw": p.slot_blend_raw,
                         "slot0_adapter":  p.slot0_adapter.state_dict(),
                         "slot1_adapter":  p.slot1_adapter.state_dict(),
                         "blend_head":     p.blend_head.state_dict(),
                         "lora_k":         p.lora_k.state_dict(),
                         "lora_v":         p.lora_v.state_dict(),
                         "lora_out":       p.lora_out.state_dict(),
                     }.items()}
                    for p in procs
                ],
                "val_score":       vs_full,
                "iou_e0":          val_m["iou_e0"],
                "iou_e1":          val_m["iou_e1"],
                "ordering_acc":    val_m["ordering_acc"],
                "wrong_slot_leak": val_m["wrong_slot_leak"],
                "id_margin":       val_m["id_margin"],
                "rollout_id_score": rollout_m["rollout_id_score"],
                "inject_keys":     inject_keys,
                "adapter_rank":    args.adapter_rank,
                "lora_rank":       args.lora_rank,
            }
            torch.save(ckpt_data, str(save_dir / "latest.pt"))

            if vs_full > best_val_score:
                best_val_score = vs_full
                best_epoch     = epoch
                torch.save(ckpt_data, str(save_dir / "best.pt"))
                print(f"  ★ best epoch={best_epoch}  val_score={vs_full:.4f}  "
                      f"(iou_e0={val_m['iou_e0']:.4f} "
                      f"iou_e1={val_m['iou_e1']:.4f} "
                      f"ord={val_m['ordering_acc']:.4f} "
                      f"wrong={val_m['wrong_slot_leak']:.4f} "
                      f"id={val_m['id_margin']:.4f})  "
                      f"→ {save_dir}/best.pt", flush=True)
                if frames_eval is not None:
                    iio2.mimwrite(str(debug_dir / "best.gif"), frames_eval, fps=8, loop=0)

            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            vca_layer.train()
            manager.train()

    print(f"\n[Phase 41] 훈련 완료. best epoch={best_epoch} val_score={best_val_score:.4f}",
          flush=True)


# =============================================================================
# argparse
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Phase 41: Spatial Localization Fix (Class-Balanced Sigma)")

    # paths
    p.add_argument("--ckpt",         type=str, default="checkpoints/phase40_v2/best.pt")
    p.add_argument("--data-root",    type=str, default="toy/data_objaverse")
    p.add_argument("--save-dir",     type=str, default="checkpoints/phase41")
    p.add_argument("--debug-dir",    type=str, default="outputs/phase41_debug")

    # training
    p.add_argument("--total-epochs", type=int, default=DEFAULT_TOTAL_EPOCHS)
    p.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    p.add_argument("--n-frames",     type=int, default=8)
    p.add_argument("--n-steps",      type=int, default=20)
    p.add_argument("--t-max",        type=int, default=300)
    p.add_argument("--height",       type=int, default=256)
    p.add_argument("--width",        type=int, default=256)

    # architecture
    p.add_argument("--adapter-rank", type=int,   default=DEFAULT_ADAPTER_RANK)
    p.add_argument("--lora-rank",    type=int,   default=DEFAULT_LORA_RANK)
    p.add_argument("--slot-blend",   type=float, default=DEFAULT_SLOT_BLEND)
    p.add_argument("--inject-keys",  type=str,   default=None)

    # LR
    p.add_argument("--lr-vca",       type=float, default=DEFAULT_LR_VCA)
    p.add_argument("--lr-adapter",   type=float, default=DEFAULT_LR_ADAPTER)
    p.add_argument("--lr-lora",      type=float, default=DEFAULT_LR_LORA)
    p.add_argument("--lr-blend",     type=float, default=DEFAULT_LR_BLEND)

    # lambdas
    p.add_argument("--lambda-vis",   type=float, default=DEFAULT_LAMBDA_VIS)
    p.add_argument("--lambda-wrong", type=float, default=DEFAULT_LAMBDA_WRONG)
    p.add_argument("--lambda-sigma", type=float, default=DEFAULT_LAMBDA_SIGMA)
    p.add_argument("--lambda-depth", type=float, default=DEFAULT_LAMBDA_DEPTH)
    p.add_argument("--lambda-ov",    type=float, default=DEFAULT_LAMBDA_OV)
    p.add_argument("--lambda-excl",  type=float, default=DEFAULT_LAMBDA_EXCL)

    # eval
    p.add_argument("--val-frac",     type=float, default=VAL_FRAC)
    p.add_argument("--eval-every",   type=int,   default=5)
    p.add_argument("--eval-seed",    type=int,   default=42)
    p.add_argument("--seed",         type=int,   default=42)

    args = p.parse_args()
    train_phase41(args)


if __name__ == "__main__":
    main()
