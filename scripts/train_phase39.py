"""
Phase 39 — Entity-Slot Attention v2: GT Visible-Weight Supervision
===================================================================

Phase 38 문제점 (4가지 핵심 결함)
-----------------------------------
1. RGB threshold 기반 entity_score/chimera_rate로 best checkpoint 선택
   → 배경색 오판 + entity 소멸을 chimera 감소로 오인

2. slot_blend_raw가 global scalar
   → collision 영역/단독 영역 구분 없이 동일한 blend 강도

3. F_0/F_1이 frozen q/k/v에서만 나와 L_exclusive 구조적으로 약함
   → slot에 학습 가능한 파라미터가 없어 gradient가 흐를 여지 없음

4. l_diff를 처음부터 세게 걸면 slot_blend→0으로 수렴
   → "원본과 달라지지 마" 압력이 slot을 꺼버림

Phase 39 수정 사항
------------------
[필수-1] RGB metric 제거 → GT mask 기반 held-out validation
[필수-2] GT visible-weight loss (L_visible_weights) 추가 — 가장 핵심
[필수-3] slot0_adapter / slot1_adapter + per-pixel blend_head 추가
[필수-4] overlap-heavy held-out validation으로 best checkpoint 선택 교체

추가 개선
---------
- l_wrong_slot_suppression: exclusive 영역 wrong entity weight 직접 패널티
- 2-stage 학습: stage1(λ_diff≈0, blend 자유 학습) → stage2(λ_diff ramp-up)
- 학습 샘플 weighted sampling: overlap 높은 시퀀스 oversampling
- val set: overlap 상위 20% 시퀀스 → 더 어려운 평가

아키텍처 (Phase 39)
--------------------
  F_global = Attn(Q, K[all], V[all])          — 원본 품질 보존
  F_0      = slot0_adapter(Attn(Q, K[e0], V[e0]))  — entity 0 전용 + adapter
  F_1      = slot1_adapter(Attn(Q, K[e1], V[e1]))  — entity 1 전용 + adapter

  Porter-Duff compositing (VCA sigma 기반):
    composed = w0*F_0 + w1*F_1 + w_bg*F_global
    output   = blend_map * composed + (1-blend_map) * F_global
    blend_map = blend_head(alpha0, alpha1, alpha0*alpha1, e0_front)  ← per-pixel

손실 함수 (Phase 39)
---------------------
  L = λ_vis   × L_visible_weights      ← 핵심: GT visible target 직접 supervision
    + λ_wrong  × L_wrong_slot           ← exclusive 영역 wrong entity weight 패널티
    + λ_depth  × L_depth                ← VCA sigma depth ordering
    + λ_ov     × L_overlap_ordering     ← 겹침 영역 front > back (보조)
    + λ_diff   × L_diff                 ← 생성 품질 (2-stage ramp-up)

학습 파라미터
-------------
  - vca_layer: VCA sigma 계산
  - proc.slot0_adapter, proc.slot1_adapter: entity slot residual adapters (신규)
  - proc.blend_head: per-pixel blend map MLP (신규)
  - proc.slot_blend_raw: scalar blend (backward compat, 보조)

검증 지표 (GT mask 기반)
------------------------
  visible_iou:     predicted w0/w1 vs GT visible targets soft IoU
  ordering_acc:    overlap 영역에서 front entity w > back entity w 비율
  wrong_slot_leak: exclusive 영역 wrong entity weight 평균
  dra:             depth rank accuracy
  val_slot_score = 0.4*visible_iou + 0.3*ordering_acc + 0.2*(1-wrong_slot_leak) + 0.1*dra

best.pt 선택 기준: val_slot_score (entity_score/chimera_rate 아님)
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
    EntitySlotAttnProcessor,
    inject_entity_slot,
    # Phase 39 losses
    l_entity_exclusive,
    l_overlap_ordering,
    l_visible_weights,
    l_wrong_slot_suppression,
    l_sigma_spatial,
    # Phase 39 metrics (GT mask 기반)
    compute_visible_iou,
    compute_ordering_accuracy,
    compute_wrong_slot_leak,
    val_slot_score,
    # overlap score for train/val split
    compute_overlap_score,
    # debug-only (qualitative GIF용)
    entity_score as compute_entity_score_debug,
)
from models.losses import l_diff as loss_diff
from scripts.run_animatediff import load_pipeline
from scripts.train_animatediff_vca import (
    compute_sigma_stats_train, encode_frames_to_latents,
)
from scripts.train_phase31 import (
    DEFAULT_LR,
    INJECT_KEY, INJECT_QUERY_DIM,
    DEPTH_PE_INIT_SCALE, VCA_ALPHA,
    ObjaverseDatasetWithMasks,
    make_color_prompts, get_color_entity_context,
    l_zorder_direct,
    restore_procs,
    measure_depth_rank_accuracy,
)
from scripts.train_phase35 import (
    DEFAULT_Z_BINS,
    get_entity_token_positions,
)


# ─── Phase 39 하이퍼파라미터 ──────────────────────────────────────────────────
DEFAULT_LAMBDA_VIS    = 3.0    # L_visible_weights (핵심 loss)
DEFAULT_LAMBDA_WRONG  = 1.5    # L_wrong_slot_suppression
DEFAULT_LAMBDA_DEPTH  = 3.0    # L_depth
DEFAULT_LAMBDA_OV     = 1.0    # L_overlap_ordering (보조)
DEFAULT_LAMBDA_SIGMA  = 5.0    # L_sigma_spatial (VCA alpha → entity mask alignment)
DEFAULT_LAMBDA_EXCL   = 0.5    # L_entity_exclusive (slot adapter gradient source)
DEFAULT_LAMBDA_DIFF_S1 = 0.0   # stage1: l_diff 꺼둠 (slot 자유 학습)
DEFAULT_LAMBDA_DIFF_S2 = 0.3   # stage2: 점진 ramp-up 최대값
DEFAULT_SLOT_BLEND    = 0.3    # initial slot blend
DEFAULT_ADAPTER_RANK  = 64     # SlotAdapter bottleneck dim
DEFAULT_EPOCHS        = 100
DEFAULT_STAGE1_FRAC   = 0.6    # 전체 epoch의 60%를 stage1로
DEFAULT_LR_SLOT       = 3e-4
DEFAULT_LR_VCA        = 3e-5
DEFAULT_LR_ADAPTER    = 1e-4   # slot adapter LR (adapter가 빠르게 학습해야)
DEFAULT_LR_BLEND      = 3e-4   # blend_head LR
DEFAULT_STEPS_PER_EPOCH = 20
VAL_FRAC              = 0.2    # overlap 상위 20% → validation set
MIN_VAL_SAMPLES       = 4      # val set 최소 크기


# =============================================================================
# Dataset split: overlap-heavy samples → val
# =============================================================================

def compute_dataset_overlap_scores(dataset) -> np.ndarray:
    """
    전체 dataset에 대해 overlap score 사전 계산.
    overlap이 높은 시퀀스 = 두 entity가 많이 겹치는 어려운 케이스.
    """
    scores = []
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            entity_masks = sample[4]   # (T, 2, S) float ndarray
            scores.append(compute_overlap_score(entity_masks))
        except Exception:
            scores.append(0.0)
    return np.array(scores)


def split_train_val(overlap_scores: np.ndarray, val_frac: float = VAL_FRAC,
                    min_val: int = MIN_VAL_SAMPLES):
    """
    overlap score 상위 val_frac → val set, 나머지 → train set.
    val은 overlap 높은 순 정렬.
    """
    N        = len(overlap_scores)
    n_val    = max(min_val, int(np.ceil(N * val_frac)))
    n_val    = min(n_val, N - 1)   # 최소 1개는 train에
    sorted_i = np.argsort(overlap_scores)[::-1]
    val_idx  = sorted_i[:n_val].tolist()
    train_idx = sorted_i[n_val:].tolist()
    return train_idx, val_idx


def make_sampling_weights(train_idx: list, overlap_scores: np.ndarray,
                          base_weight: float = 1.0) -> np.ndarray:
    """
    Hard-overlap oversampling: overlap score에 비례한 sampling weight.
    min weight = base_weight (uniform baseline).
    """
    scores = overlap_scores[train_idx]
    # normalize to [base_weight, base_weight + 1]
    s_min = scores.min()
    s_max = scores.max()
    if s_max > s_min:
        weights = base_weight + (scores - s_min) / (s_max - s_min)
    else:
        weights = np.ones(len(train_idx)) * base_weight
    return weights / weights.sum()


# =============================================================================
# Lambda ramp-up: 2-stage training
# =============================================================================

def get_lambda_diff(epoch: int, total_epochs: int,
                    stage1_frac: float,
                    lambda_diff_s1: float,
                    lambda_diff_s2: float) -> float:
    """
    Stage 1 (0 ~ stage1_frac*total_epochs): lambda_diff ≈ 0
    Stage 2 (stage1_frac*total_epochs ~ total_epochs): linear ramp-up to lambda_diff_s2

    Stage 1에서 slot이 visible/depth/ordering loss만으로 자유롭게 학습 후,
    Stage 2에서 품질 보존 압력을 점진적으로 가함.
    """
    stage1_end = int(stage1_frac * total_epochs)
    if epoch < stage1_end:
        return lambda_diff_s1
    else:
        # 마지막 epoch (total_epochs - 1)에서 정확히 lambda_diff_s2에 도달
        denom = max(total_epochs - stage1_end - 1, 1)
        progress = (epoch - stage1_end) / denom
        return lambda_diff_s1 + (lambda_diff_s2 - lambda_diff_s1) * min(progress, 1.0)


# =============================================================================
# Teacher-forced validation (GT mask 기반)
# =============================================================================

@torch.no_grad()
def evaluate_val_set(
    pipe,
    proc:       EntitySlotAttnProcessor,
    vca_layer:  VCALayer,
    dataset,
    val_idx:    list,
    device:     str,
    t_fixed:    int = 200,
) -> dict:
    """
    held-out val set에 대해 teacher-forced evaluation.

    자유 생성 대신 GT latents에 noise를 추가해 UNet forward 후
    proc.last_w0, proc.last_w1로 GT visible target과 비교.

    Returns dict with keys: visible_iou, ordering_acc, wrong_slot_leak, dra, val_score
    """
    vis_ious      = []
    ord_accs      = []
    wrong_leaks   = []

    for vi in val_idx:
        try:
            sample = dataset[vi]
            frames_np, _, depth_orders, meta, entity_masks = sample

            entity_ctx          = get_color_entity_context(pipe, meta, device)
            toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)

            proc.set_entity_ctx(entity_ctx.float())
            proc.set_entity_tokens(toks_e0, toks_e1)
            proc.reset_slot_store()

            latents = encode_frames_to_latents(pipe, frames_np, device)
            noise   = torch.randn_like(latents)
            t_tensor = torch.tensor([t_fixed], device=device).long()
            noisy   = pipe.scheduler.add_noise(latents, noise, t_tensor)

            tok = pipe.tokenizer(
                full_prompt, return_tensors="pt", padding="max_length",
                max_length=pipe.tokenizer.model_max_length, truncation=True,
            ).to(device)
            enc_hs = pipe.text_encoder(**tok).last_hidden_state.half()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = pipe.unet(noisy, t_tensor, encoder_hidden_states=enc_hs).sample

            if proc.last_w0 is None or proc.last_w1 is None:
                continue

            T_frames = min(proc.last_w0.shape[0], entity_masks.shape[0])
            masks_t  = torch.from_numpy(
                entity_masks[:T_frames].astype(np.float32)).to(device)

            for fi in range(T_frames):
                w0_f  = proc.last_w0[fi:fi+1].float()
                w1_f  = proc.last_w1[fi:fi+1].float()
                m_f   = masks_t[fi:fi+1]
                do    = [depth_orders[fi]] if fi < len(depth_orders) else [(0, 1)]

                vis_ious.append(compute_visible_iou(w0_f, w1_f, m_f, do))
                ord_accs.append(compute_ordering_accuracy(w0_f, w1_f, m_f, do))
                wrong_leaks.append(compute_wrong_slot_leak(w0_f, w1_f, m_f))

        except Exception as e:
            print(f"  [warn] val sample {vi} 실패: {e}", flush=True)
            continue

    if not vis_ious:
        return {"visible_iou": 0.0, "ordering_acc": 0.0,
                "wrong_slot_leak": 1.0, "dra": 0.0, "val_score": 0.0}

    vi_mean  = float(np.mean(vis_ious))
    oa_mean  = float(np.mean(ord_accs))
    wl_mean  = float(np.mean(wrong_leaks))

    # DRA (기존 measure_depth_rank_accuracy 활용 — val samples만)
    try:
        from scripts.train_phase31 import measure_depth_rank_accuracy
        # val_idx를 ObjaverseDatasetWithMasks 슬라이스처럼 임시로 구성
        class _SubDataset:
            def __init__(self, ds, idx):
                self._ds, self._idx = ds, idx
            def __len__(self): return len(self._idx)
            def __getitem__(self, i): return self._ds[self._idx[i]]
        sub_ds = _SubDataset(dataset, val_idx)
        dra_val, _, _ = measure_depth_rank_accuracy(
            pipe, vca_layer, sub_ds, device,
            n_samples=min(len(val_idx), 10))
    except Exception:
        dra_val = 0.0

    vs = val_slot_score(vi_mean, oa_mean, wl_mean, dra_val)

    return {
        "visible_iou":    vi_mean,
        "ordering_acc":   oa_mean,
        "wrong_slot_leak": wl_mean,
        "dra":            dra_val,
        "val_score":      vs,
    }


# =============================================================================
# 훈련 루프
# =============================================================================

def train_phase39(args):
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
    print("[Phase 39] 파이프라인 로드 중...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    # ── VCA 로드 (Phase 31 또는 Phase 38 체크포인트) ─────────────────────────
    print(f"[Phase 39] VCA 체크포인트 로드: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location=device)
    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM, context_dim=768,
        n_heads=8, n_entities=2, z_bins=DEFAULT_Z_BINS, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=DEPTH_PE_INIT_SCALE,
    ).to(device)
    vca_layer.load_state_dict(ckpt["vca_state_dict"])
    p_prev_gamma = float(ckpt.get("gamma_trained", VCA_ALPHA))
    print(f"  prev gamma_trained = {p_prev_gamma:.4f}", flush=True)

    # ── 데이터셋 ─────────────────────────────────────────────────────────────
    dataset = ObjaverseDatasetWithMasks(args.data_root, n_frames=args.n_frames)
    print(f"[Phase 39] 데이터셋: {len(dataset)} 시퀀스", flush=True)
    if len(dataset) == 0:
        raise RuntimeError(f"데이터셋 비어있음: {args.data_root}")

    # ── Overlap score 계산 → train/val 분리 ─────────────────────────────────
    print("[Phase 39] overlap score 계산 중...", flush=True)
    overlap_scores = compute_dataset_overlap_scores(dataset)
    train_idx, val_idx = split_train_val(
        overlap_scores, val_frac=args.val_frac, min_val=args.min_val_samples)
    print(f"  train={len(train_idx)}  val={len(val_idx)}  "
          f"(val_frac={args.val_frac}, val overlap ≥ "
          f"{float(overlap_scores[val_idx].min()):.4f})", flush=True)

    # sampling weights for hard-overlap oversampling
    sample_weights = make_sampling_weights(train_idx, overlap_scores)

    # ── EntitySlotAttnProcessor 주입 ─────────────────────────────────────────
    # 첫 번째 train sample로 entity_ctx 초기화 (나중에 step마다 교체)
    init_sample = dataset[train_idx[0]]
    init_meta   = init_sample[3]
    init_ctx    = get_color_entity_context(pipe, init_meta, device)

    proc, orig_procs = inject_entity_slot(
        pipe, vca_layer, init_ctx,
        inject_key    = INJECT_KEY,
        slot_blend_init = args.slot_blend,
        adapter_rank  = args.adapter_rank,
        use_blend_head = True,
    )
    proc = proc.to(device)

    # Phase 38 체크포인트에서 slot_blend_raw 복원 (있으면)
    if "slot_blend_raw" in ckpt:
        proc.slot_blend_raw.data.copy_(ckpt["slot_blend_raw"].to(device))
        print(f"  Phase 38 slot_blend_raw 복원: "
              f"blend={float(proc.slot_blend.item()):.4f}", flush=True)

    print(f"  slot_blend_init={args.slot_blend:.3f}  "
          f"adapter_rank={args.adapter_rank}  use_blend_head=True", flush=True)

    # ── 파라미터 동결 / 학습 대상 설정 ──────────────────────────────────────
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    for p in vca_layer.parameters():
        p.requires_grad_(True)
    proc.slot_blend_raw.requires_grad_(True)
    for p in proc.slot0_adapter.parameters():
        p.requires_grad_(True)
    for p in proc.slot1_adapter.parameters():
        p.requires_grad_(True)
    for p in proc.blend_head.parameters():
        p.requires_grad_(True)

    # ── 옵티마이저 ──────────────────────────────────────────────────────────
    param_groups = [
        {"params": list(vca_layer.parameters()),
         "lr": args.lr_vca,     "name": "vca"},
        {"params": [proc.slot_blend_raw],
         "lr": args.lr_slot,    "name": "slot_blend"},
        {"params": list(proc.slot0_adapter.parameters())
                 + list(proc.slot1_adapter.parameters()),
         "lr": args.lr_adapter, "name": "adapters"},
        {"params": list(proc.blend_head.parameters()),
         "lr": args.lr_blend,   "name": "blend_head"},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_vca * 0.05)

    # ── 메트릭 기록 ──────────────────────────────────────────────────────────
    history      = []
    best_val_score = -1.0
    best_epoch   = -1

    print(f"\n[Phase 39] 훈련 시작: {args.epochs} epochs", flush=True)
    print(f"  λ_vis={args.lambda_vis}  λ_wrong={args.lambda_wrong}  "
          f"λ_depth={args.lambda_depth}  λ_ov={args.lambda_ov}  "
          f"λ_sigma={args.lambda_sigma}  λ_excl={args.lambda_excl}", flush=True)
    print(f"  λ_diff: stage1={args.lambda_diff_s1} → stage2={args.lambda_diff_s2} "
          f"(stage1 frac={args.stage1_frac})", flush=True)
    print(f"  lr_vca={args.lr_vca:.2e}  lr_slot={args.lr_slot:.2e}  "
          f"lr_adapter={args.lr_adapter:.2e}  lr_blend={args.lr_blend:.2e}", flush=True)
    print(f"  stage1 ends at epoch {int(args.stage1_frac*args.epochs)}", flush=True)

    for epoch in range(args.epochs):
        vca_layer.train()
        proc.train()

        # lambda_diff 계산 (2-stage ramp-up)
        lambda_diff_cur = get_lambda_diff(
            epoch, args.epochs,
            args.stage1_frac,
            args.lambda_diff_s1,
            args.lambda_diff_s2,
        )

        epoch_losses = {
            "total": [], "vis": [], "wrong": [],
            "depth": [], "ov": [], "diff": [],
            "sigma": [], "excl": [],
        }

        # hard-overlap oversampling
        chosen = np.random.choice(
            len(train_idx),
            size=args.steps_per_epoch,
            replace=True,
            p=sample_weights,
        )
        step_indices = [train_idx[ci] for ci in chosen]

        for batch_idx, data_idx in enumerate(step_indices):
            sample = dataset[data_idx]
            frames_np, _, depth_orders, meta, entity_masks = sample

            entity_ctx = get_color_entity_context(pipe, meta, device)
            toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)

            proc.set_entity_ctx(entity_ctx.float())
            proc.set_entity_tokens(toks_e0, toks_e1)
            proc.reset_slot_store()

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
            masks_t        = torch.from_numpy(
                entity_masks[:T_frames].astype(np.float32)).to(device)
            depth_orders_t = depth_orders[:T_frames]

            # ── UNet forward ──────────────────────────────────────────────
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = pipe.unet(noisy, t,
                                       encoder_hidden_states=enc_hs).sample

            # ── L_visible_weights (핵심 loss) ──────────────────────────
            l_vis = torch.tensor(0.0, device=device)
            if proc.last_w0 is not None and proc.last_w1 is not None:
                BF    = proc.last_w0.shape[0]
                T_use = min(BF, T_frames)
                for fi in range(T_use):
                    w0_f = proc.last_w0[fi:fi+1].float()
                    w1_f = proc.last_w1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    do   = [depth_orders_t[fi]] if fi < len(depth_orders_t) else [(0, 1)]
                    l_vis = l_vis + l_visible_weights(w0_f, w1_f, m_f, do)
                l_vis = l_vis / max(T_use, 1)

            # ── L_wrong_slot_suppression ───────────────────────────────
            l_wrong = torch.tensor(0.0, device=device)
            if proc.last_w0 is not None and proc.last_w1 is not None:
                BF    = proc.last_w0.shape[0]
                T_use = min(BF, T_frames)
                for fi in range(T_use):
                    w0_f = proc.last_w0[fi:fi+1].float()
                    w1_f = proc.last_w1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    l_wrong = l_wrong + l_wrong_slot_suppression(w0_f, w1_f, m_f)
                l_wrong = l_wrong / max(T_use, 1)

            # ── L_depth ───────────────────────────────────────────────────
            sigma_acc = list(proc.sigma_acc)
            l_depth   = l_zorder_direct(sigma_acc, depth_orders_t,
                                        entity_masks[:T_frames])

            # ── L_overlap_ordering (보조) ─────────────────────────────────
            l_ov = torch.tensor(0.0, device=device)
            if proc.last_w0 is not None and proc.last_w1 is not None:
                BF    = proc.last_w0.shape[0]
                T_use = min(BF, T_frames)
                for fi in range(T_use):
                    w0_f = proc.last_w0[fi:fi+1].float()
                    w1_f = proc.last_w1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    do   = [depth_orders_t[fi]] if fi < len(depth_orders_t) else [(0, 1)]
                    l_ov = l_ov + l_overlap_ordering(w0_f, w1_f, m_f, do)
                l_ov = l_ov / max(T_use, 1)

            # ── L_sigma_spatial (VCA alpha → entity mask alignment) ────────
            l_sigma = torch.tensor(0.0, device=device)
            if proc.last_alpha0 is not None and proc.last_alpha1 is not None:
                BF    = proc.last_alpha0.shape[0]
                T_use = min(BF, T_frames)
                for fi in range(T_use):
                    a0_f = proc.last_alpha0[fi:fi+1].float()
                    a1_f = proc.last_alpha1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    l_sigma = l_sigma + l_sigma_spatial(a0_f, a1_f, m_f)
                l_sigma = l_sigma / max(T_use, 1)

            # ── L_entity_exclusive (slot adapter gradient source) ──────────
            l_excl = torch.tensor(0.0, device=device)
            if (proc.last_F0 is not None and proc.last_F1 is not None
                    and proc.last_Fg is not None):
                BF    = proc.last_F0.shape[0]
                T_use = min(BF, T_frames)
                for fi in range(T_use):
                    F0_f = proc.last_F0[fi:fi+1].float()
                    F1_f = proc.last_F1[fi:fi+1].float()
                    Fg_f = proc.last_Fg[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]
                    l_excl = l_excl + l_entity_exclusive(F0_f, F1_f, Fg_f, m_f)
                l_excl = l_excl / max(T_use, 1)

            # ── L_diff (2-stage ramp-up) ──────────────────────────────────
            l_diff_val = loss_diff(noise_pred.float(), noise.float())

            # ── Total loss ────────────────────────────────────────────────
            loss = (args.lambda_vis   * l_vis
                  + args.lambda_wrong * l_wrong
                  + args.lambda_depth * l_depth
                  + args.lambda_ov    * l_ov
                  + args.lambda_sigma * l_sigma
                  + args.lambda_excl  * l_excl
                  + lambda_diff_cur   * l_diff_val)

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss ep={epoch} step={batch_idx} "
                      f"vis={l_vis.item():.4f} wrong={l_wrong.item():.4f} "
                      f"depth={l_depth.item():.4f} sigma={l_sigma.item():.4f} → skip",
                      flush=True)
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(vca_layer.parameters()), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([proc.slot_blend_raw], max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(
                list(proc.slot0_adapter.parameters())
                + list(proc.slot1_adapter.parameters())
                + list(proc.blend_head.parameters()),
                max_norm=1.0,
            )

            optimizer.step()

            epoch_losses["total"].append(loss.item())
            epoch_losses["vis"].append(
                float(l_vis.item()) if isinstance(l_vis, torch.Tensor) else 0.0)
            epoch_losses["wrong"].append(
                float(l_wrong.item()) if isinstance(l_wrong, torch.Tensor) else 0.0)
            epoch_losses["depth"].append(float(l_depth.item()))
            epoch_losses["ov"].append(
                float(l_ov.item()) if isinstance(l_ov, torch.Tensor) else 0.0)
            epoch_losses["sigma"].append(
                float(l_sigma.item()) if isinstance(l_sigma, torch.Tensor) else 0.0)
            epoch_losses["excl"].append(
                float(l_excl.item()) if isinstance(l_excl, torch.Tensor) else 0.0)
            epoch_losses["diff"].append(float(l_diff_val.item()))

        scheduler.step()

        avg        = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
        blend_val  = float(proc.slot_blend.item())
        stage_str  = "S1" if epoch < int(args.stage1_frac * args.epochs) else "S2"

        print(f"[Phase 39] epoch {epoch:03d}/{args.epochs-1} [{stage_str}]  "
              f"loss={avg['total']:.4f}  vis={avg['vis']:.4f}  "
              f"wrong={avg['wrong']:.4f}  depth={avg['depth']:.4f}  "
              f"sigma={avg['sigma']:.4f}  excl={avg['excl']:.4f}  "
              f"λ_diff={lambda_diff_cur:.3f}  blend={blend_val:.4f}", flush=True)

        # ── Validation (GT mask 기반, held-out overlap-heavy split) ──────
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            vca_layer.eval()
            proc.eval()

            val_metrics = evaluate_val_set(
                pipe, proc, vca_layer, dataset, val_idx, device,
                t_fixed=args.t_max // 2,
            )
            vs = val_metrics["val_score"]

            print(f"  [val] visible_iou={val_metrics['visible_iou']:.4f}  "
                  f"ordering_acc={val_metrics['ordering_acc']:.4f}  "
                  f"wrong_leak={val_metrics['wrong_slot_leak']:.4f}  "
                  f"dra={val_metrics['dra']:.4f}  "
                  f"val_score={vs:.4f}", flush=True)

            # qualitative GIF (debug-only, entity_score는 저장만 함)
            frames_eval = None
            try:
                probe_sample = dataset[val_idx[0]]
                probe_meta   = probe_sample[3]
                probe_ctx    = get_color_entity_context(pipe, probe_meta, device)
                probe_te0, probe_te1, probe_prompt = get_entity_token_positions(
                    pipe, probe_meta)
                proc.set_entity_ctx(probe_ctx.float())
                proc.set_entity_tokens(probe_te0, probe_te1)
                proc.reset_slot_store()
                gen = torch.Generator(device=device).manual_seed(args.eval_seed)
                out = pipe(
                    prompt=probe_prompt,
                    num_frames=args.n_frames,
                    num_inference_steps=args.n_steps,
                    height=args.height, width=args.width,
                    generator=gen, output_type="np",
                )
                frames_eval = (out.frames[0] * 255).astype(np.uint8)
                gif_path = debug_dir / f"eval_epoch{epoch:03d}.gif"
                iio2.mimwrite(str(gif_path), frames_eval, fps=8, loop=0)
                # debug-only entity_score 기록
                es_dbg, sr_dbg, cr_dbg = compute_entity_score_debug(frames_eval)
                print(f"  [debug-only] entity_score={es_dbg:.4f} "
                      f"survival={sr_dbg:.4f} chimera={cr_dbg:.4f} "
                      f"(NOT used for best selection)", flush=True)
            except Exception as e:
                print(f"  [warn] qualitative GIF 실패: {e}", flush=True)

            history.append({
                "epoch": epoch,
                "val_score":      vs,
                "visible_iou":    val_metrics["visible_iou"],
                "ordering_acc":   val_metrics["ordering_acc"],
                "wrong_slot_leak": val_metrics["wrong_slot_leak"],
                "dra":            val_metrics["dra"],
                "slot_blend":     blend_val,
                "lambda_diff":    lambda_diff_cur,
                **avg,
            })

            ckpt_data = {
                "epoch":          epoch,
                "vca_state_dict": vca_layer.state_dict(),
                "slot_blend_raw": proc.slot_blend_raw.detach().cpu(),
                "slot_blend":     blend_val,
                "slot0_adapter":  proc.slot0_adapter.state_dict(),
                "slot1_adapter":  proc.slot1_adapter.state_dict(),
                "blend_head":     proc.blend_head.state_dict(),
                # val metrics
                "val_score":      vs,
                "visible_iou":    val_metrics["visible_iou"],
                "ordering_acc":   val_metrics["ordering_acc"],
                "wrong_slot_leak": val_metrics["wrong_slot_leak"],
                "dra":            val_metrics["dra"],
                # hyperparams
                "lambda_vis":     args.lambda_vis,
                "lambda_wrong":   args.lambda_wrong,
                "lambda_depth":   args.lambda_depth,
                "lambda_ov":      args.lambda_ov,
                "lambda_diff_s2": args.lambda_diff_s2,
                "adapter_rank":   args.adapter_rank,
            }
            torch.save(ckpt_data, str(save_dir / "latest.pt"))

            # best.pt: val_slot_score 기준 (entity_score 아님)
            if vs > best_val_score:
                best_val_score = vs
                best_epoch     = epoch
                torch.save(ckpt_data, str(save_dir / "best.pt"))
                print(f"  ★ best epoch={best_epoch}  val_score={vs:.4f}  "
                      f"(visible_iou={val_metrics['visible_iou']:.4f} "
                      f"ord_acc={val_metrics['ordering_acc']:.4f} "
                      f"wrong_leak={val_metrics['wrong_slot_leak']:.4f})  "
                      f"→ {save_dir}/best.pt", flush=True)
                if frames_eval is not None:
                    iio2.mimwrite(str(debug_dir / "best.gif"), frames_eval,
                                  fps=8, loop=0)

            vca_layer.train()
            proc.train()

        # ── 주기적 시각화 ─────────────────────────────────────────────────
        if epoch % args.vis_every == 0 or epoch == args.epochs - 1:
            try:
                _visualize_slot_weights(proc, debug_dir, epoch)
            except Exception as e:
                print(f"  [warn] slot 시각화 실패: {e}", flush=True)

    print(f"\n[Phase 39] 훈련 완료. best epoch={best_epoch} "
          f"val_score={best_val_score:.4f}", flush=True)
    with open(debug_dir / "phase39_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  history → {debug_dir}/phase39_history.json", flush=True)
    return best_val_score


# =============================================================================
# 시각화 유틸
# =============================================================================

def _visualize_slot_weights(
    proc:      EntitySlotAttnProcessor,
    debug_dir: Path,
    epoch:     int,
):
    """sigma heatmap + blend map 시각화."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigma = proc.last_sigma
    if sigma is None:
        return

    sigma_np = sigma[0].cpu().numpy()   # (S, N, Z)
    S = sigma_np.shape[0]
    HW = int(S ** 0.5)
    if HW * HW != S:
        return

    blend_np = None
    if proc.last_blend is not None:
        lb = proc.last_blend
        if isinstance(lb, torch.Tensor) and lb.dim() >= 2:
            blend_np = lb[0].detach().float().cpu().numpy().reshape(HW, HW)

    n_cols = 5 if blend_np is not None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    blend_val = float(proc.slot_blend.item())
    fig.suptitle(f"Epoch {epoch}  scalar_blend={blend_val:.3f}")

    for ni, (ei, zi, title) in enumerate([
        (0, 0, "sigma_e0_z0"), (0, 1, "sigma_e0_z1"),
        (1, 0, "sigma_e1_z0"), (1, 1, "sigma_e1_z1"),
    ]):
        ax = axes[ni]
        ax.imshow(sigma_np[:, ei, zi].reshape(HW, HW),
                  vmin=0, vmax=1, cmap="hot", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    if blend_np is not None:
        axes[4].imshow(blend_np, vmin=0, vmax=1,
                       cmap="RdYlGn", interpolation="nearest")
        axes[4].set_title("blend_map")
        axes[4].axis("off")

    plt.tight_layout()
    out_path = debug_dir / f"slot_sigma_epoch{epoch:03d}.png"
    plt.savefig(str(out_path), dpi=80)
    plt.close()


# =============================================================================
# CLI
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Phase 39: Entity-Slot Attention v2 + GT Visible-Weight Supervision")
    # paths
    p.add_argument("--ckpt",              type=str,   default="checkpoints/phase38/best.pt")
    p.add_argument("--data-root",         type=str,   default="toy/data_objaverse")
    p.add_argument("--save-dir",          type=str,   default="checkpoints/phase39")
    p.add_argument("--debug-dir",         type=str,   default="debug/train_phase39")
    # training
    p.add_argument("--epochs",            type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--steps-per-epoch",   type=int,   default=DEFAULT_STEPS_PER_EPOCH)
    p.add_argument("--n-frames",          type=int,   default=8)
    p.add_argument("--n-steps",           type=int,   default=20)
    p.add_argument("--t-max",             type=int,   default=300)
    p.add_argument("--height",            type=int,   default=256)
    p.add_argument("--width",             type=int,   default=256)
    # learning rates
    p.add_argument("--lr-vca",            type=float, default=DEFAULT_LR_VCA)
    p.add_argument("--lr-slot",           type=float, default=DEFAULT_LR_SLOT)
    p.add_argument("--lr-adapter",        type=float, default=DEFAULT_LR_ADAPTER)
    p.add_argument("--lr-blend",          type=float, default=DEFAULT_LR_BLEND)
    # lambdas
    p.add_argument("--lambda-vis",        type=float, default=DEFAULT_LAMBDA_VIS)
    p.add_argument("--lambda-wrong",      type=float, default=DEFAULT_LAMBDA_WRONG)
    p.add_argument("--lambda-depth",      type=float, default=DEFAULT_LAMBDA_DEPTH)
    p.add_argument("--lambda-ov",         type=float, default=DEFAULT_LAMBDA_OV)
    p.add_argument("--lambda-sigma",      type=float, default=DEFAULT_LAMBDA_SIGMA)
    p.add_argument("--lambda-excl",       type=float, default=DEFAULT_LAMBDA_EXCL)
    p.add_argument("--lambda-diff-s1",    type=float, default=DEFAULT_LAMBDA_DIFF_S1)
    p.add_argument("--lambda-diff-s2",    type=float, default=DEFAULT_LAMBDA_DIFF_S2)
    p.add_argument("--stage1-frac",       type=float, default=DEFAULT_STAGE1_FRAC)
    # architecture
    p.add_argument("--slot-blend",        type=float, default=DEFAULT_SLOT_BLEND)
    p.add_argument("--adapter-rank",      type=int,   default=DEFAULT_ADAPTER_RANK)
    # dataset split
    p.add_argument("--val-frac",          type=float, default=VAL_FRAC)
    p.add_argument("--min-val-samples",   type=int,   default=MIN_VAL_SAMPLES)
    # eval
    p.add_argument("--eval-every",        type=int,   default=5)
    p.add_argument("--vis-every",         type=int,   default=10)
    p.add_argument("--eval-seed",         type=int,   default=42)
    p.add_argument("--seed",              type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_phase39(args)
