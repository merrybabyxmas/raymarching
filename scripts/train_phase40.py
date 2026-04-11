"""
Phase 40 — Entity-Layer Consistency Distillation Training
==========================================================

Phase 39의 weight supervision을 유지하면서 appearance identity supervision 추가.

핵심 변경
----------
1. Multi-block injection (up_blocks.1/2/3) — appearance 용량 확보
2. SlotLoRA on K, V, out — appearance channel 변환
3. L_solo_feat_visible: F_slot → solo reference alignment
4. L_id_contrast: cosine margin loss
5. 3-stage 학습:
   Stage 1 (0 → s1): weight sup only (L_vis, L_wrong, L_sigma, L_depth)
   Stage 2 (s1 → s1+s2): + appearance distillation (L_solo, L_id_contrast)
   Stage 3 (s1+s2 → total): + L_diff + rollout consistency

평가
----
- TF eval: teacher-forced, fast (per-epoch)
- Rollout eval: multi-step denoising, realistic (eval_every × 2)
- Best checkpoint: val_score_phase40 (rollout-aware)

성공 기준
---------
  visible_iou_e0 > 0.30
  visible_iou_e1 > 0.30
  ordering_acc   > 0.65
  wrong_slot_leak < 0.15
  rollout_id_score > Phase39 baseline
  GIF에서 두 엔티티가 collision frame에서 섞이지 않음
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
    compute_visible_iou,
    compute_ordering_accuracy,
    compute_wrong_slot_leak,
    val_slot_score,
    compute_overlap_score,
    entity_score as compute_entity_score_debug,
)
from models.entity_slot_phase40 import (
    Phase40Processor,
    MultiBlockSlotManager,
    inject_multi_block_entity_slot,
    compute_visible_masks,
    l_solo_feat_visible,
    l_id_contrast,
    l_bg_feat,
    compute_visible_iou_e0,
    compute_visible_iou_e1,
    compute_id_feature_margin,
    val_score_phase40,
    DEFAULT_INJECT_KEYS,
    BLOCK_INNER_DIMS,
)
from models.losses import l_diff as loss_diff
from scripts.run_animatediff import load_pipeline
from scripts.train_animatediff_vca import encode_frames_to_latents
from scripts.train_phase31 import (
    INJECT_KEY, INJECT_QUERY_DIM,
    DEPTH_PE_INIT_SCALE, VCA_ALPHA,
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
    get_lambda_diff,
)
from scripts.generate_solo_renders import (
    ObjaverseDatasetPhase40,
    compute_visible_masks_np,
    make_pseudo_solo_from_composite,
    generate_solo_data_for_dataset,
)


# ─── Phase 40 하이퍼파라미터 ──────────────────────────────────────────────────
# Stage 분할 (epoch 기준)
DEFAULT_S1_EPOCHS  = 20    # stage1: weight supervision only
DEFAULT_S2_EPOCHS  = 40    # stage2: + solo feat distillation
DEFAULT_S3_EPOCHS  = 20    # stage3: + L_diff + rollout
DEFAULT_TOTAL_EPOCHS = DEFAULT_S1_EPOCHS + DEFAULT_S2_EPOCHS + DEFAULT_S3_EPOCHS

# Loss 가중치
DEFAULT_LAMBDA_VIS    = 2.0
DEFAULT_LAMBDA_WRONG  = 1.0
DEFAULT_LAMBDA_SIGMA  = 2.0
DEFAULT_LAMBDA_DEPTH  = 2.0
DEFAULT_LAMBDA_OV     = 1.0
DEFAULT_LAMBDA_EXCL   = 0.5
DEFAULT_LAMBDA_SOLO   = 3.0    # L_solo_feat_visible (Phase 40 핵심)
DEFAULT_LAMBDA_ID     = 1.0    # L_id_contrast
DEFAULT_LAMBDA_BG     = 0.5    # L_bg_feat
DEFAULT_LAMBDA_DIFF_S3 = 0.3   # stage3 l_diff 최대값

# LR
DEFAULT_LR_VCA      = 2e-5
DEFAULT_LR_ADAPTER  = 1e-4
DEFAULT_LR_LORA     = 5e-5
DEFAULT_LR_BLEND    = 3e-4
DEFAULT_LR_SLOT     = 3e-4

# Architecture
DEFAULT_ADAPTER_RANK = 64
DEFAULT_LORA_RANK    = 4
DEFAULT_SLOT_BLEND   = 0.3
DEFAULT_STEPS_PER_EPOCH = 20

# Rollout eval
ROLLOUT_T_START  = 250
ROLLOUT_N_STEPS  = 5

VAL_FRAC         = 0.2
MIN_VAL_SAMPLES  = 4


# =============================================================================
# Solo reference feature extraction
# =============================================================================

@torch.no_grad()
def extract_solo_ref_features(
    pipe,
    manager:      MultiBlockSlotManager,
    noisy_latents: torch.Tensor,        # (1, C, T, H, W)
    t_tensor:     torch.Tensor,
    entity_ctx:   torch.Tensor,         # (1, 2, 768)
    toks_e0:      list,
    toks_e1:      list,
    device:       str,
):
    """
    Entity-prompted reference feature extraction.

    entity0 ref: composite latent + entity0 text 임베딩만 → F_0 추출
    entity1 ref: composite latent + entity1 text 임베딩만 → F_1 추출

    solo render가 없을 때의 근사.
    gradient 없음 (frozen reference).
    """
    manager.eval()

    F0_ref_list = []
    F1_ref_list = []

    for ent_idx in range(2):
        manager.reset_slot_store()
        manager.set_entity_tokens(toks_e0, toks_e1)

        # zero out the other entity's embedding
        ctx_solo = entity_ctx.clone()
        ctx_solo[:, 1 - ent_idx, :] = 0.0
        manager.set_entity_ctx(ctx_solo.float())

        # build text embedding: use entity-i prompt only
        # enc_hs from ctx_solo (simplified: expand to match expected shape)
        # In practice this is a rough approximation of entity-only forward
        B = noisy_latents.shape[0]
        enc_solo = ctx_solo.expand(B, -1, -1).to(device).half()

        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = pipe.unet(
                    noisy_latents, t_tensor,
                    encoder_hidden_states=enc_solo,
                ).sample
        except Exception:
            pass

        proc = manager.primary
        if ent_idx == 0:
            F0_ref_list.append(proc.last_F0)
        else:
            F1_ref_list.append(proc.last_F1)

    manager.train()
    F0_ref = F0_ref_list[0] if F0_ref_list else None
    F1_ref = F1_ref_list[0] if F1_ref_list else None
    return F0_ref, F1_ref


# =============================================================================
# Rollout evaluation
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
    """
    Short denoising rollout eval (Phase 40 신규).

    GT latent에 noise 추가 → scheduler.step() × n_steps → decode
    → visible IoU / identity feature 측정.

    Teacher-forced eval보다 실제 GIF 품질과 훨씬 더 가까움.
    """
    manager.eval()

    rollout_iou_e0s = []
    rollout_iou_e1s = []
    rollout_id_scores = []

    for vi in val_idx[:min(len(val_idx), 8)]:   # 빠른 측정용: 최대 8 samples
        try:
            sample = dataset[vi]
            # Phase 40 dataset returns 8 items; fallback to 5-item
            if len(sample) >= 8:
                frames_np, _, depth_orders, meta, entity_masks, visible_masks, _, _ = sample
            else:
                frames_np, _, depth_orders, meta, entity_masks = sample
                visible_masks = compute_visible_masks_np(
                    entity_masks.astype(np.float32), depth_orders)

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

            # rollout: n_steps of denoising
            x = noisy.clone()
            scheduler_state = copy.deepcopy(pipe.scheduler)
            ts = scheduler_state.timesteps

            # find timesteps near t_start
            t_indices = (ts <= t_start).nonzero(as_tuple=True)[0]
            if len(t_indices) == 0:
                continue
            t_start_idx = t_indices[0].item()
            t_end_idx   = min(t_start_idx + n_steps, len(ts))

            for step_t in ts[t_start_idx:t_end_idx]:
                step_t_batch = torch.tensor([step_t], device=device).long()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    noise_pred = pipe.unet(
                        x, step_t_batch, encoder_hidden_states=enc_hs).sample
                x = scheduler_state.step(
                    noise_pred, step_t, x).prev_sample

            # final step: collect w0/w1 from last forward
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

            # identity score: F0 vs F1 cosine similarity in each entity region
            proc = manager.primary
            if proc.last_F0 is not None and proc.last_F1 is not None:
                T_use = min(proc.last_F0.shape[0], T_frames)
                for fi in range(T_use):
                    F0_f  = proc.last_F0[fi:fi+1].float()
                    F1_f  = proc.last_F1[fi:fi+1].float()
                    m_f   = masks_t[fi:fi+1]
                    m0 = m_f[:, 0, :].float()
                    m1 = m_f[:, 1, :].float()
                    # cross-entity cosine similarity (lower = better separation)
                    F0_n = torch.nn.functional.normalize(F0_f, dim=-1)
                    F1_n = torch.nn.functional.normalize(F1_f, dim=-1)
                    cross_sim = (F0_n * F1_n).sum(-1).mean().item()
                    rollout_id_scores.append(1.0 - cross_sim)   # higher = better

        except Exception as e:
            print(f"  [rollout warn] val {vi}: {e}", flush=True)
            continue

    manager.train()

    if not rollout_iou_e0s:
        return {"rollout_iou_e0": 0.0, "rollout_iou_e1": 0.0,
                "rollout_id_score": 0.0}

    return {
        "rollout_iou_e0":   float(np.mean(rollout_iou_e0s)),
        "rollout_iou_e1":   float(np.mean(rollout_iou_e1s)),
        "rollout_id_score": float(np.mean(rollout_id_scores)) if rollout_id_scores else 0.0,
    }


# =============================================================================
# Teacher-forced validation (Phase 39 방식 + Phase 40 per-entity metrics)
# =============================================================================

@torch.no_grad()
def evaluate_val_set_phase40(
    pipe,
    manager:    MultiBlockSlotManager,
    vca_layer:  VCALayer,
    dataset,
    val_idx:    list,
    device:     str,
    t_fixed:    int = 200,
) -> dict:
    """
    Teacher-forced validation with Phase 40 per-entity metrics.
    """
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
                frames_np, _, depth_orders, meta, entity_masks, visible_masks, _, _ = sample
            else:
                frames_np, _, depth_orders, meta, entity_masks = sample
                visible_masks = compute_visible_masks_np(
                    entity_masks.astype(np.float32), depth_orders)

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

                # id_margin (if F0/F1 available from same-step solo ref)
                if proc.last_F0 is not None and proc.last_F1 is not None:
                    F0_f  = proc.last_F0[fi:fi+1].float()
                    F1_f  = proc.last_F1[fi:fi+1].float()
                    m0 = m_f[:, 0, :].float()
                    m1 = m_f[:, 1, :].float()
                    # proxy: F0 vs F1 cross-cosine (lower cross-sim = better margin)
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

    # DRA
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
        "iou_e0":        iou_e0_m,
        "iou_e1":        iou_e1_m,
        "ordering_acc":  ord_m,
        "wrong_slot_leak": wl_m,
        "id_margin":     id_m,
        "dra":           dra,
        "val_score":     vs,
    }


# =============================================================================
# Training loop
# =============================================================================

def train_phase40(args):
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

    total_epochs = args.s1_epochs + args.s2_epochs + args.s3_epochs
    s1_end  = args.s1_epochs
    s2_end  = args.s1_epochs + args.s2_epochs

    def get_stage(epoch):
        if epoch < s1_end:  return 1
        if epoch < s2_end:  return 2
        return 3

    def get_lambda_diff_cur(epoch):
        if epoch < s2_end:
            return 0.0
        progress = (epoch - s2_end) / max(total_epochs - s2_end - 1, 1)
        return args.lambda_diff_s3 * min(progress, 1.0)

    # ── 파이프라인 로드 ──────────────────────────────────────────────────────
    print("[Phase 40] 파이프라인 로드 중...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    # ── VCA 로드 ─────────────────────────────────────────────────────────────
    print(f"[Phase 40] checkpoint 로드: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location=device)
    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM, context_dim=768,
        n_heads=8, n_entities=2, z_bins=DEFAULT_Z_BINS, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=DEPTH_PE_INIT_SCALE,
    ).to(device)
    vca_layer.load_state_dict(ckpt["vca_state_dict"])
    print(f"  VCA loaded", flush=True)

    # ── 데이터셋 ─────────────────────────────────────────────────────────────
    print("[Phase 40] 데이터셋 로드 중...", flush=True)
    # Try Phase40 dataset first; fall back to base
    try:
        dataset = ObjaverseDatasetPhase40(args.data_root, n_frames=args.n_frames)
        n_solo  = dataset.count_solo_renders()
        print(f"  ObjaverseDatasetPhase40: {len(dataset)} samples, solo={n_solo}", flush=True)
        if n_solo == 0:
            print("  [INFO] solo renders 없음 → pseudo-solo 자동 생성 중...", flush=True)
            generate_solo_data_for_dataset(args.data_root, method="pseudo", verbose=True)
            n_solo = dataset.count_solo_renders()
            print(f"  pseudo-solo 생성 완료: {n_solo}/{len(dataset)}", flush=True)
    except Exception as e:
        print(f"  ObjaverseDatasetPhase40 실패({e}), Phase39 dataset 사용", flush=True)
        dataset = ObjaverseDatasetWithMasks(args.data_root, n_frames=args.n_frames)

    if len(dataset) == 0:
        raise RuntimeError(f"데이터셋 비어있음: {args.data_root}")

    # ── Train/Val 분리 (overlap-heavy) ───────────────────────────────────────
    print("[Phase 40] overlap score 계산 중...", flush=True)
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
        inject_keys    = inject_keys,
        slot_blend_init = args.slot_blend,
        adapter_rank   = args.adapter_rank,
        lora_rank      = args.lora_rank,
        use_blend_head = True,
    )
    manager = MultiBlockSlotManager(procs, inject_keys, primary_idx=1)
    for p in procs:
        p.to(device)

    # ── Phase 39 상태 복원 (slot_blend_raw + adapter + blend_head) ────────────
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
    for p in pipe.unet.parameters():    p.requires_grad_(False)
    for p in pipe.text_encoder.parameters(): p.requires_grad_(False)
    for p in pipe.vae.parameters():     p.requires_grad_(False)
    for p in vca_layer.parameters():    p.requires_grad_(True)
    for proc in procs:
        for p in proc.parameters():     p.requires_grad_(True)

    # ── Optimizer ──────────────────────────────────────────────────────────
    param_groups = [
        {"params": list(vca_layer.parameters()),
         "lr": args.lr_vca,    "name": "vca"},
        {"params": manager.adapter_params(),
         "lr": args.lr_adapter, "name": "adapters"},
        {"params": manager.lora_params(),
         "lr": args.lr_lora,   "name": "lora"},
        {"params": manager.blend_params(),
         "lr": args.lr_blend,  "name": "blend"},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=args.lr_vca * 0.05)

    # ── 학습 기록 ────────────────────────────────────────────────────────────
    history        = []
    best_val_score = -1.0
    best_epoch     = -1

    print(f"\n[Phase 40] 훈련 시작: {total_epochs} epochs "
          f"(S1={s1_end}, S2={s2_end}, S3={total_epochs})", flush=True)
    print(f"  λ_vis={args.lambda_vis}  λ_wrong={args.lambda_wrong}  "
          f"λ_sigma={args.lambda_sigma}  λ_depth={args.lambda_depth}", flush=True)
    print(f"  λ_solo={args.lambda_solo}  λ_id={args.lambda_id}  "
          f"λ_diff_s3={args.lambda_diff_s3}", flush=True)

    for epoch in range(total_epochs):
        vca_layer.train()
        manager.train()

        stage          = get_stage(epoch)
        lambda_diff_cur = get_lambda_diff_cur(epoch)
        use_solo        = (stage >= 2)

        epoch_losses = {k: [] for k in
                        ["total", "vis", "wrong", "sigma", "depth",
                         "ov", "excl", "solo", "id", "bg", "diff"]}

        chosen = np.random.choice(
            len(train_idx), size=args.steps_per_epoch,
            replace=True, p=sample_weights)
        step_indices = [train_idx[ci] for ci in chosen]

        for batch_idx, data_idx in enumerate(step_indices):
            sample = dataset[data_idx]
            # unpack (flexible for both Phase39/40 datasets)
            if len(sample) >= 8:
                frames_np, _, depth_orders, meta, entity_masks, visible_masks, solo_e0, solo_e1 = sample
            else:
                frames_np, _, depth_orders, meta, entity_masks = sample
                visible_masks = compute_visible_masks_np(
                    entity_masks.astype(np.float32), depth_orders)
                solo_e0 = solo_e1 = None

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
            vis_masks_t    = torch.from_numpy(visible_masks[:T_frames].astype(np.float32)).to(device)
            depth_orders_t = depth_orders[:T_frames]

            # ── UNet primary forward ──────────────────────────────────────
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = pipe.unet(noisy, t, encoder_hidden_states=enc_hs).sample

            BF    = manager.last_w0.shape[0] if manager.last_w0 is not None else 0
            T_use = min(BF, T_frames)

            # ── L_visible_weights ─────────────────────────────────────────
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

            # ── L_sigma_spatial ───────────────────────────────────────────
            l_sigma = torch.tensor(0.0, device=device)
            if manager.last_alpha0 is not None:
                for fi in range(T_use):
                    a0_f = manager.last_alpha0[fi:fi+1].float()
                    a1_f = manager.last_alpha1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    l_sigma = l_sigma + l_sigma_spatial(a0_f, a1_f, m_f)
                l_sigma = l_sigma / max(T_use, 1)

            # ── L_depth ───────────────────────────────────────────────────
            sigma_acc = list(manager.sigma_acc)
            l_depth   = l_zorder_direct(sigma_acc, depth_orders_t, entity_masks[:T_frames])

            # ── L_overlap_ordering (보조) ──────────────────────────────────
            l_ov = torch.tensor(0.0, device=device)
            if manager.last_w0 is not None:
                for fi in range(T_use):
                    w0_f = manager.last_w0[fi:fi+1].float()
                    w1_f = manager.last_w1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    do   = [depth_orders_t[fi]] if fi < len(depth_orders_t) else [(0, 1)]
                    l_ov = l_ov + l_overlap_ordering(w0_f, w1_f, m_f, do)
                l_ov = l_ov / max(T_use, 1)

            # ── L_entity_exclusive (adapter gradient) ─────────────────────
            l_excl = torch.tensor(0.0, device=device)
            proc_pri = manager.primary
            for blk_F0, blk_F1, blk_Fg in zip(
                    manager.all_F0(), manager.all_F1(), manager.all_Fg()):
                if blk_F0 is not None and blk_F1 is not None and blk_Fg is not None:
                    for fi in range(min(blk_F0.shape[0], T_frames)):
                        l_excl = l_excl + l_entity_exclusive(
                            blk_F0[fi:fi+1].float(),
                            blk_F1[fi:fi+1].float(),
                            blk_Fg[fi:fi+1].float(),
                            masks_t[fi:fi+1])
            if T_use > 0 and len(manager.procs) > 0:
                l_excl = l_excl / max(T_use * len(manager.procs), 1)

            # ── Stage 2+: Solo feature distillation ───────────────────────
            l_solo = torch.tensor(0.0, device=device)
            l_id   = torch.tensor(0.0, device=device)
            l_bg   = torch.tensor(0.0, device=device)

            if use_solo and T_use > 0:
                # Get solo reference features
                # Method: entity-prompted forward (composite latent + entity-only text)
                F0_ref, F1_ref = extract_solo_ref_features(
                    pipe, manager, noisy, t,
                    entity_ctx, toks_e0, toks_e1, device,
                )

                # Restore training state after ref extraction
                manager.train()
                manager.set_entity_ctx(entity_ctx.float())
                manager.set_entity_tokens(toks_e0, toks_e1)
                # Note: proc F0/F1 from primary forward are in manager.primary.last_F0/F1

                if (F0_ref is not None and F1_ref is not None
                        and proc_pri.last_F0 is not None and proc_pri.last_F1 is not None):

                    for fi in range(min(proc_pri.last_F0.shape[0], T_use)):
                        F0_comp_f  = proc_pri.last_F0[fi:fi+1].float()
                        F1_comp_f  = proc_pri.last_F1[fi:fi+1].float()
                        Fg_comp_f  = proc_pri.last_Fg[fi:fi+1].float() if proc_pri.last_Fg is not None else None
                        F0_ref_f   = F0_ref[fi:fi+1].float() if F0_ref is not None else None
                        F1_ref_f   = F1_ref[fi:fi+1].float() if F1_ref is not None else None
                        vis_m_f    = vis_masks_t[fi:fi+1]
                        m_f        = masks_t[fi:fi+1]

                        # L_solo_feat_visible
                        if F0_ref_f is not None:
                            l_solo = l_solo + l_solo_feat_visible(
                                F0_comp_f, F0_ref_f, vis_m_f[:, 0, :])
                        if F1_ref_f is not None:
                            l_solo = l_solo + l_solo_feat_visible(
                                F1_comp_f, F1_ref_f, vis_m_f[:, 1, :])

                        # L_id_contrast
                        if F0_ref_f is not None and F1_ref_f is not None:
                            l_id = l_id + l_id_contrast(
                                F0_comp_f, F1_comp_f, F0_ref_f, F1_ref_f,
                                m_f[:, 0, :].float(), m_f[:, 1, :].float())

                        # L_bg_feat
                        if Fg_comp_f is not None and F0_ref_f is not None:
                            bg_mask = (1.0 - (m_f[:, 0, :] + m_f[:, 1, :]).clamp(max=1.0))
                            l_bg = l_bg + l_bg_feat(Fg_comp_f, F0_ref_f, bg_mask.float())

                    n_div = max(min(proc_pri.last_F0.shape[0], T_use), 1)
                    l_solo = l_solo / n_div
                    l_id   = l_id   / n_div
                    l_bg   = l_bg   / n_div

            # ── L_diff ────────────────────────────────────────────────────
            l_diff_val = loss_diff(noise_pred.float(), noise.float())

            # ── Total loss ────────────────────────────────────────────────
            loss = (args.lambda_vis   * l_vis
                  + args.lambda_wrong * l_wrong
                  + args.lambda_sigma * l_sigma
                  + args.lambda_depth * l_depth
                  + args.lambda_ov    * l_ov
                  + args.lambda_excl  * l_excl
                  + args.lambda_solo  * l_solo
                  + args.lambda_id    * l_id
                  + args.lambda_bg    * l_bg
                  + lambda_diff_cur   * l_diff_val)

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss ep={epoch} step={batch_idx} → skip",
                      flush=True)
                continue

            loss.backward()

            # gradient clipping (per group)
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
            epoch_losses["solo"].append(_f(l_solo))
            epoch_losses["id"].append(_f(l_id))
            epoch_losses["bg"].append(_f(l_bg))
            epoch_losses["diff"].append(_f(l_diff_val))

        scheduler.step()

        avg       = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
        blend_val = float(primary.slot_blend.item())
        stage_str = f"S{stage}"

        print(f"[Phase 40] epoch {epoch:03d}/{total_epochs-1} [{stage_str}]  "
              f"loss={avg['total']:.4f}  vis={avg['vis']:.4f}  "
              f"wrong={avg['wrong']:.4f}  sigma={avg['sigma']:.4f}  "
              f"solo={avg['solo']:.4f}  id={avg['id']:.4f}  "
              f"λ_diff={lambda_diff_cur:.3f}  blend={blend_val:.4f}", flush=True)

        # ── Validation ────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == total_epochs - 1:
            vca_layer.eval()
            manager.eval()

            # TF eval
            val_m = evaluate_val_set_phase40(
                pipe, manager, vca_layer, dataset, val_idx, device,
                t_fixed=args.t_max // 2)
            vs = val_m["val_score"]

            print(f"  [val] iou_e0={val_m['iou_e0']:.4f}  iou_e1={val_m['iou_e1']:.4f}  "
                  f"ord={val_m['ordering_acc']:.4f}  wrong={val_m['wrong_slot_leak']:.4f}  "
                  f"id_margin={val_m['id_margin']:.4f}  dra={val_m['dra']:.4f}  "
                  f"val_score={vs:.4f}", flush=True)

            # Rollout eval (every 2 × eval_every)
            rollout_m = {"rollout_iou_e0": 0.0, "rollout_iou_e1": 0.0,
                         "rollout_id_score": 0.0}
            if epoch % (args.eval_every * 2) == 0 or epoch == total_epochs - 1:
                rollout_m = evaluate_rollout(
                    pipe, manager, dataset, val_idx, device,
                    t_start=ROLLOUT_T_START, n_steps=ROLLOUT_N_STEPS)
                print(f"  [rollout] iou_e0={rollout_m['rollout_iou_e0']:.4f}  "
                      f"iou_e1={rollout_m['rollout_iou_e1']:.4f}  "
                      f"id_score={rollout_m['rollout_id_score']:.4f}", flush=True)

            # update val_score with rollout_id
            vs_full = val_score_phase40(
                val_m["iou_e0"], val_m["iou_e1"], val_m["ordering_acc"],
                val_m["wrong_slot_leak"], val_m["id_margin"],
                rollout_m["rollout_id_score"])

            # qualitative GIF
            frames_eval = None
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

            history.append({
                "epoch": epoch, "stage": stage,
                "val_score": vs_full,
                **val_m, **rollout_m, **avg,
                "slot_blend": blend_val, "lambda_diff": lambda_diff_cur,
            })

            # checkpoint
            ckpt_data = {
                "epoch":         epoch,
                "stage":         stage,
                "vca_state_dict": vca_layer.state_dict(),
                # primary block
                "slot_blend_raw": primary.slot_blend_raw.detach().cpu(),
                "slot_blend":    blend_val,
                "slot0_adapter": primary.slot0_adapter.state_dict(),
                "slot1_adapter": primary.slot1_adapter.state_dict(),
                "blend_head":    primary.blend_head.state_dict(),
                "lora_k":        primary.lora_k.state_dict(),
                "lora_v":        primary.lora_v.state_dict(),
                "lora_out":      primary.lora_out.state_dict(),
                # all blocks
                "procs_state": [
                    {k: v.detach().cpu() if hasattr(v, 'cpu') else v
                     for k, v in {
                         "slot_blend_raw": p.slot_blend_raw,
                         "slot0_adapter": p.slot0_adapter.state_dict(),
                         "slot1_adapter": p.slot1_adapter.state_dict(),
                         "blend_head":    p.blend_head.state_dict(),
                         "lora_k":        p.lora_k.state_dict(),
                         "lora_v":        p.lora_v.state_dict(),
                         "lora_out":      p.lora_out.state_dict(),
                     }.items()}
                    for p in procs
                ],
                # metrics
                "val_score":     vs_full,
                "iou_e0":        val_m["iou_e0"],
                "iou_e1":        val_m["iou_e1"],
                "ordering_acc":  val_m["ordering_acc"],
                "wrong_slot_leak": val_m["wrong_slot_leak"],
                "id_margin":     val_m["id_margin"],
                "rollout_id_score": rollout_m["rollout_id_score"],
                # hyperparams
                "inject_keys":   inject_keys,
                "adapter_rank":  args.adapter_rank,
                "lora_rank":     args.lora_rank,
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

            # history json
            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    print(f"\n[Phase 40] 훈련 완료. best epoch={best_epoch} val_score={best_val_score:.4f}",
          flush=True)


# =============================================================================
# argparse
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Phase 40: Entity-Layer Consistency Distillation")

    # paths
    p.add_argument("--ckpt",         type=str, default="checkpoints/phase39/best.pt")
    p.add_argument("--data-root",    type=str, default="toy/data_objaverse")
    p.add_argument("--save-dir",     type=str, default="checkpoints/phase40")
    p.add_argument("--debug-dir",    type=str, default="outputs/phase40_debug")

    # training
    p.add_argument("--s1-epochs",    type=int, default=DEFAULT_S1_EPOCHS)
    p.add_argument("--s2-epochs",    type=int, default=DEFAULT_S2_EPOCHS)
    p.add_argument("--s3-epochs",    type=int, default=DEFAULT_S3_EPOCHS)
    p.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    p.add_argument("--n-frames",     type=int, default=8)
    p.add_argument("--n-steps",      type=int, default=20)
    p.add_argument("--t-max",        type=int, default=300)
    p.add_argument("--height",       type=int, default=256)
    p.add_argument("--width",        type=int, default=256)

    # architecture
    p.add_argument("--adapter-rank", type=int, default=DEFAULT_ADAPTER_RANK)
    p.add_argument("--lora-rank",    type=int, default=DEFAULT_LORA_RANK)
    p.add_argument("--slot-blend",   type=float, default=DEFAULT_SLOT_BLEND)
    p.add_argument("--inject-keys",  type=str, default=None,
                   help="comma-separated inject keys; default=3-block")

    # LR
    p.add_argument("--lr-vca",       type=float, default=DEFAULT_LR_VCA)
    p.add_argument("--lr-adapter",   type=float, default=DEFAULT_LR_ADAPTER)
    p.add_argument("--lr-lora",      type=float, default=DEFAULT_LR_LORA)
    p.add_argument("--lr-blend",     type=float, default=DEFAULT_LR_BLEND)

    # lambdas
    p.add_argument("--lambda-vis",      type=float, default=DEFAULT_LAMBDA_VIS)
    p.add_argument("--lambda-wrong",    type=float, default=DEFAULT_LAMBDA_WRONG)
    p.add_argument("--lambda-sigma",    type=float, default=DEFAULT_LAMBDA_SIGMA)
    p.add_argument("--lambda-depth",    type=float, default=DEFAULT_LAMBDA_DEPTH)
    p.add_argument("--lambda-ov",       type=float, default=DEFAULT_LAMBDA_OV)
    p.add_argument("--lambda-excl",     type=float, default=DEFAULT_LAMBDA_EXCL)
    p.add_argument("--lambda-solo",     type=float, default=DEFAULT_LAMBDA_SOLO)
    p.add_argument("--lambda-id",       type=float, default=DEFAULT_LAMBDA_ID)
    p.add_argument("--lambda-bg",       type=float, default=DEFAULT_LAMBDA_BG)
    p.add_argument("--lambda-diff-s3",  type=float, default=DEFAULT_LAMBDA_DIFF_S3)

    # eval
    p.add_argument("--val-frac",        type=float, default=VAL_FRAC)
    p.add_argument("--eval-every",      type=int,   default=5)
    p.add_argument("--eval-seed",       type=int,   default=42)
    p.add_argument("--seed",            type=int,   default=42)

    args = p.parse_args()
    train_phase40(args)


if __name__ == "__main__":
    main()
