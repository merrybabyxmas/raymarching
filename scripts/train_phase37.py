"""
Phase 37 — Volumetric Text Cross-Attention v3 (z_pe 수렴 수정)
=================================================================

Phase 36 실패 원인 및 수정
--------------------------
Phase 36 에서 z_pe 는 150 epoch 훈련 후에도 ||z_pe|| ≈ 12.8 (목표 25) 에 머물렀다.
원인 분석:
  steps_per_epoch=20  ×  z_pe_lr=3e-5  ×  max_norm=1.0  →  z_pe 업데이트 = 3e-5/step
  l_pe_norm gradient norm ≈ 36  →  clip 후 실제 전달 = 1/36 의 방향만 남음
  3000 steps × 3e-5 = 0.09  (실측 +0.39 = 다른 loss 기여 포함)
  목표 norm 25 도달에 필요: ~400,000 steps (2000 epochs) — 불가능

수정:
  1. z_pe LR = 1e-3  (33× 증가)  →  동일 3000 steps에서 충분히 성장
  2. z_pe max_norm = 100.0  →  gradient 희석 제거
  이 두 변경만으로 l_pe_norm gradient(≈36)가 완전히 전달 →
     per-step update ≈ 1e-3 × 36 ≈ 0.036
     18 epochs × 20 steps × 0.036 = 12.6  (12.4 → 25 도달) ✓

  3. l_pe_sep → l_pe_antisep: cos → -1 을 적극 유도 (단순 공동방향 패널티가 아님)
     l_pe_antisep = (1 + cos(z_pe[0], z_pe[1]))^2  ∈ [0,4]
     cos=-1 → 0  (antiparallel, ideal)
     cos= 0 → 1  (orthogonal)
     cos=+1 → 4  (co-alignment, worst)

  4. 진단 추가: key-alignment score
     attn key 방향과 z_pe 정렬 여부를 직접 측정
     align_e0 = cos(z_pe[0], K_e0 - K_e1)   # z_pe[0] → E0 방향?
     align_e1 = cos(z_pe[1], K_e1 - K_e0)   # z_pe[1] → E1 방향?

손실 함수 (Phase 37)
--------------------
  L = λ_vol    × l_vol_attn      ← volumetric attention GT supervision  (5.0)
    + λ_depth   × l_zorder_direct ← VCA sigma depth ordering              (3.0)
    + λ_diff    × l_diff          ← 생성 품질 보존                         (0.5)
    + λ_pe_anti × l_pe_antisep    ← z-bin antiparallel 유도               (2.0)
    + λ_pe_norm × l_pe_norm       ← z-bin norm ≥ 25 강제                  (0.5)

학습 파라미터
-----------
  - z_pe  (Z, D): z_pe_lr=1e-3, z_pe_max_norm=100  (Phase 37 핵심 변경)
  - vca_layer:    base LR (3e-5, Phase 31 체크포인트에서 초기화)
  - gamma:        frozen at 1.0 (순수 volumetric)
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.vca_volumetric import (
    VolumetricTextCrossAttentionProcessor,
    find_entity_token_positions,
    l_vol_attn_loss,
    l_pe_norm,
)
from models.losses import l_diff as loss_diff
from scripts.run_animatediff import load_pipeline
from scripts.train_animatediff_vca import (
    compute_sigma_stats_train, encode_frames_to_latents,
)
from scripts.train_objaverse_vca import (
    ObjaverseTrainDataset, get_entity_context_from_meta,
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
    measure_generation_diff,
    AdditiveVCAInferProcessor,
    debug_generation,
)
from scripts.train_phase35 import (
    inject_vca_p35,
    get_entity_token_positions,
    _visualize_vol_attn,
    DEFAULT_Z_BINS,
)

# ─── Phase 37 하이퍼파라미터 ──────────────────────────────────────────────────
DEFAULT_LAMBDA_VOL     = 5.0
DEFAULT_LAMBDA_DEPTH   = 3.0
DEFAULT_LAMBDA_DIFF    = 0.5
DEFAULT_LAMBDA_PE_ANTI = 2.0   # antiparallel loss (stronger than sep in p36)
DEFAULT_LAMBDA_PE_NORM = 0.5
DEFAULT_GAMMA_INIT     = 1.0   # frozen: pure volumetric
DEFAULT_Z_PE_STD       = 0.3   # same init as Phase 36
DEFAULT_Z_PE_LR        = 1e-3  # KEY FIX: 33× higher than VCA LR
DEFAULT_Z_PE_MAX_NORM  = 100.0 # KEY FIX: allow full gradient to reach z_pe
DEFAULT_Z_BINS         = 2
DEFAULT_EPOCHS         = 150
PE_TARGET_NORM         = 25.0  # sqrt(640) ≈ 25


# =============================================================================
# Anti-alignment loss (Phase 37 추가)
# =============================================================================

def l_pe_antisep(z_pe: torch.Tensor) -> torch.Tensor:
    """
    z_pe[0]과 z_pe[1] 을 antiparallel (cos → -1) 로 유도.

    l_pe_antisep = (1 + cos(z_pe[0], z_pe[1]))^2  ∈ [0, 4]
      cos = -1  → loss = 0   (antiparallel, ideal)
      cos =  0  → loss = 1   (orthogonal)
      cos = +1  → loss = 4   (co-alignment, worst)

    Phase 36 의 l_pe_sep (co-alignment 패널티만) 과 달리
    anti-alignment 를 적극 유도한다.
    """
    v0 = z_pe[0].float()
    v1 = z_pe[1].float()
    cos = torch.dot(v0, v1) / (v0.norm() * v1.norm() + 1e-8)
    return (1.0 + cos).pow(2)


# =============================================================================
# Key-alignment diagnostic
# =============================================================================

@torch.no_grad()
def measure_key_alignment(
    z_pe: torch.Tensor,
    attn_module,
    enc_hs: torch.Tensor,
    toks_e0: list,
    toks_e1: list,
) -> tuple:
    """
    z_pe 가 entity key 방향으로 정렬되어 있는지 측정.

    Returns (align_e0, align_e1) — 각각 cos ∈ [-1, 1]
      align_e0 > 0: z_pe[0] 이 K_E0 - K_E1 방향으로 정렬 (correct)
      align_e1 > 0: z_pe[1] 이 K_E1 - K_E0 방향으로 정렬 (correct)
    """
    if not toks_e0 or not toks_e1:
        return 0.0, 0.0

    T_seq = enc_hs.shape[1]
    k_all = attn_module.to_k(enc_hs.float())  # (1, T, inner_dim)

    tok_e0 = [t for t in toks_e0 if t < T_seq]
    tok_e1 = [t for t in toks_e1 if t < T_seq]
    if not tok_e0 or not tok_e1:
        return 0.0, 0.0

    k_e0 = k_all[0, tok_e0, :].mean(0)  # (inner_dim,)
    k_e1 = k_all[0, tok_e1, :].mean(0)

    diff_01 = k_e0 - k_e1   # direction favoring E0
    diff_10 = k_e1 - k_e0   # direction favoring E1

    def cos_sim(a, b):
        return float(F.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)).item())

    # z_pe[0] should prefer E0 → align with diff_01
    # z_pe[1] should prefer E1 → align with diff_10
    z0 = z_pe[0].float()
    z1 = z_pe[1].float()

    # Project z_pe to the same space as k (in case inner_dim != query_dim)
    # They're in the same D-space (query_dim = inner_dim here), so direct comparison
    if z0.shape[0] != k_e0.shape[0]:
        return 0.0, 0.0

    align_e0 = cos_sim(z0, diff_01)
    align_e1 = cos_sim(z1, diff_10)
    return align_e0, align_e1


# =============================================================================
# 훈련 루프
# =============================================================================

def train_phase37(args):
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
    print("[Phase 37] 파이프라인 로드 중...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    # ── VCA 로드 (Phase 31 체크포인트) ───────────────────────────────────────
    print(f"[Phase 37] VCA 체크포인트 로드: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location=device)
    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM, context_dim=768,
        n_heads=8, n_entities=2, z_bins=DEFAULT_Z_BINS, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=DEPTH_PE_INIT_SCALE,
    ).to(device)
    vca_layer.load_state_dict(ckpt["vca_state_dict"])
    p31_gamma = float(ckpt.get("gamma_trained", VCA_ALPHA))
    print(f"  Phase 31 gamma_trained = {p31_gamma:.4f}", flush=True)

    # ── 데이터셋 ─────────────────────────────────────────────────────────────
    dataset = ObjaverseDatasetWithMasks(args.data_root, n_frames=args.n_frames)
    print(f"[Phase 37] 데이터셋: {len(dataset)} 시퀀스", flush=True)
    if len(dataset) == 0:
        raise RuntimeError(f"데이터셋 비어있음: {args.data_root}")

    # ── Probe (고정 검증 샘플) ────────────────────────────────────────────────
    probe_idx = 0
    probe_sample = dataset[probe_idx]
    probe_frames_np, _, probe_depth_orders, probe_meta, probe_entity_masks = probe_sample
    probe_entity_ctx = get_color_entity_context(pipe, probe_meta, device)
    probe_toks_e0, probe_toks_e1, probe_full_prompt = get_entity_token_positions(
        pipe, probe_meta)

    probe_latents = encode_frames_to_latents(pipe, probe_frames_np, device)
    probe_tok = pipe.tokenizer(
        probe_full_prompt, return_tensors="pt", padding="max_length",
        max_length=pipe.tokenizer.model_max_length, truncation=True,
    ).to(device)
    with torch.no_grad():
        probe_enc_hs = pipe.text_encoder(**probe_tok).last_hidden_state.half()

    # ── VolumetricProcessor 주입 ─────────────────────────────────────────────
    proc, orig_procs = inject_vca_p35(
        pipe, vca_layer, probe_entity_ctx,
        gamma_init=DEFAULT_GAMMA_INIT,   # 1.0 — pure volumetric
        z_bins=DEFAULT_Z_BINS,
    )

    # z_pe 초기화 (Phase 35/36 과 동일: std=0.3)
    torch.nn.init.normal_(proc.z_pe, std=args.z_pe_std)
    print(f"  z_pe init: std={args.z_pe_std:.2f}  "
          f"|z_pe|={float(proc.z_pe.norm().item()):.4f}", flush=True)

    # ── gamma 고정 (frozen) ──────────────────────────────────────────────────
    proc.gamma.requires_grad_(False)
    print(f"  gamma = {float(proc.gamma.item()):.4f}  (frozen)", flush=True)

    # ── 옵티마이저: z_pe 는 별도 고 LR ─────────────────────────────────────
    # KEY FIX: z_pe_lr=1e-3 >> vca_lr=3e-5
    param_groups = [
        {"params": list(vca_layer.parameters()), "lr": args.lr,       "name": "vca"},
        {"params": [proc.z_pe],                  "lr": args.z_pe_lr,  "name": "z_pe"},
        # gamma 제외 — frozen
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    # ── 나머지 파라미터 동결 후 훈련 파라미터 requires_grad 복원 ────────────
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    for p in vca_layer.parameters():
        p.requires_grad_(True)
    proc.z_pe.requires_grad_(True)
    # proc.gamma stays False

    # ── 메트릭 기록 ──────────────────────────────────────────────────────────
    history = []
    best_dra   = -1.0
    best_gen   = 0.0
    best_epoch = -1

    print(f"\n[Phase 37] 훈련 시작: {args.epochs} epochs", flush=True)
    print(f"  λ_vol={args.lambda_vol}  λ_depth={args.lambda_depth}  "
          f"λ_diff={args.lambda_diff}", flush=True)
    print(f"  λ_pe_anti={args.lambda_pe_anti}  λ_pe_norm={args.lambda_pe_norm}  "
          f"pe_target_norm={args.pe_target_norm:.1f}", flush=True)
    print(f"  vca_lr={args.lr:.2e}  z_pe_lr={args.z_pe_lr:.2e}  "
          f"z_pe_max_norm={args.z_pe_max_norm:.1f}", flush=True)

    for epoch in range(args.epochs):
        vca_layer.train()
        proc.train()

        epoch_losses = {"total": [], "vol_attn": [], "depth": [], "diff": [],
                        "pe_anti": [], "pe_norm": []}

        indices = np.random.permutation(len(dataset)).tolist()

        for batch_idx, data_idx in enumerate(indices[:args.steps_per_epoch]):
            sample = dataset[data_idx]
            frames_np, _, depth_orders, meta, entity_masks = sample

            entity_ctx = get_color_entity_context(pipe, meta, device)
            proc.set_entity_ctx(entity_ctx.float())
            proc.reset_sigma_acc()

            toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)

            with torch.no_grad():
                latents = encode_frames_to_latents(pipe, frames_np, device)
            noise   = torch.randn_like(latents)
            t       = torch.randint(0, args.t_max, (1,), device=device).long()
            noisy   = pipe.scheduler.add_noise(latents, noise, t)

            tok = pipe.tokenizer(
                full_prompt, return_tensors="pt", padding="max_length",
                max_length=pipe.tokenizer.model_max_length, truncation=True,
            ).to(device)
            with torch.no_grad():
                enc_hs = pipe.text_encoder(**tok).last_hidden_state.half()

            T_frames   = min(frames_np.shape[0], entity_masks.shape[0])
            masks_t    = torch.from_numpy(
                entity_masks[:T_frames].astype(np.float32)).to(device)
            depth_orders_t = depth_orders[:T_frames]

            # ── UNet forward ──────────────────────────────────────────────
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = pipe.unet(noisy, t,
                                       encoder_hidden_states=enc_hs).sample

            # ── l_vol_attn ────────────────────────────────────────────────
            attn_w = proc.last_attn_weights
            l_vol  = torch.tensor(0.0, device=device)
            if attn_w is not None and len(toks_e0) > 0 and len(toks_e1) > 0:
                BF    = attn_w.shape[0]
                T_use = min(BF, T_frames)
                l_vol = l_vol_attn_loss(
                    attn_w[:T_use].float(),
                    toks_e0, toks_e1,
                    masks_t[:T_use],
                    depth_orders_t[:T_use],
                    z_bins=DEFAULT_Z_BINS,
                )

            # ── l_zorder_direct ───────────────────────────────────────────
            sigma_acc = list(proc.sigma_acc)
            l_depth   = l_zorder_direct(sigma_acc, depth_orders_t,
                                         entity_masks[:T_frames])

            # ── l_diff ────────────────────────────────────────────────────
            l_diff_val = loss_diff(noise_pred.float(), noise.float())

            # ── l_pe_antisep, l_pe_norm ───────────────────────────────────
            l_anti = l_pe_antisep(proc.z_pe)
            l_norm = l_pe_norm(proc.z_pe, target_norm=args.pe_target_norm)

            # ── total loss ────────────────────────────────────────────────
            loss = (args.lambda_vol     * l_vol
                  + args.lambda_depth   * l_depth
                  + args.lambda_diff    * l_diff_val
                  + args.lambda_pe_anti * l_anti
                  + args.lambda_pe_norm * l_norm)

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss ep={epoch} step={batch_idx} "
                      f"vol={l_vol.item():.4f} anti={l_anti.item():.4f} "
                      f"norm={l_norm.item():.4f} → skip", flush=True)
                continue

            loss.backward()

            # KEY FIX: 별도 clip — z_pe 는 높은 max_norm 으로 gradient 보존
            torch.nn.utils.clip_grad_norm_(list(vca_layer.parameters()), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([proc.z_pe], max_norm=args.z_pe_max_norm)

            optimizer.step()

            epoch_losses["total"].append(loss.item())
            epoch_losses["vol_attn"].append(
                l_vol.item() if isinstance(l_vol, torch.Tensor) else 0.0)
            epoch_losses["depth"].append(l_depth.item())
            epoch_losses["diff"].append(l_diff_val.item())
            epoch_losses["pe_anti"].append(l_anti.item())
            epoch_losses["pe_norm"].append(l_norm.item())

        scheduler.step()

        avg = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
        z_pe_norm_val = float(proc.z_pe.norm().item())
        with torch.no_grad():
            v0 = proc.z_pe[0].float()
            v1 = proc.z_pe[1].float()
            cos_bins = float(
                (torch.dot(v0, v1) / (v0.norm() * v1.norm() + 1e-8)).item())

        print(f"[Phase 37] epoch {epoch:03d}/{args.epochs-1}  "
              f"loss={avg['total']:.4f}  vol={avg['vol_attn']:.4f}  "
              f"anti={avg['pe_anti']:.4f}  norm={avg['pe_norm']:.4f}  "
              f"|z_pe|={z_pe_norm_val:.4f}  cos={cos_bins:.3f}", flush=True)

        # ── 검증 ─────────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            vca_layer.eval()
            proc.eval()

            # key-alignment diagnostic
            # INJECT_KEY ends with ".processor"; strip it to get the Attention module path
            try:
                attn_mod_key = INJECT_KEY.replace(".processor", "")
                attn_mod = pipe.unet.get_submodule(attn_mod_key)
                align_e0, align_e1 = measure_key_alignment(
                    proc.z_pe, attn_mod,
                    probe_enc_hs, probe_toks_e0, probe_toks_e1)
                print(f"  key-align: align_e0={align_e0:.3f}  "
                      f"align_e1={align_e1:.3f}  "
                      f"(>0 means correct direction)", flush=True)
            except Exception as e:
                print(f"  [warn] key-align 실패: {e}", flush=True)
                align_e0, align_e1 = 0.0, 0.0

            try:
                proc.set_entity_ctx(probe_entity_ctx.float())
                dra, n_correct, n_total = measure_depth_rank_accuracy(
                    pipe, vca_layer, dataset, device,
                    n_samples=min(20, len(dataset)),
                )
                print(f"  DRA = {dra:.4f} ({n_correct}/{n_total})", flush=True)
            except Exception as e:
                print(f"  [warn] DRA 실패: {e}", flush=True)
                dra = 0.0

            try:
                proc.set_entity_ctx(probe_entity_ctx.float())
                gen_diff = measure_generation_diff(
                    pipe, vca_layer, orig_procs,
                    dict(pipe.unet.attn_processors),
                    probe_meta, probe_entity_ctx, device,
                )
            except Exception as e:
                print(f"  [warn] gen_diff 실패: {e}", flush=True)
                gen_diff = 0.0

            history.append({
                "epoch": epoch,
                "dra": dra,
                "gen_diff": gen_diff,
                "z_pe_norm": z_pe_norm_val,
                "cos_bins": cos_bins,
                "align_e0": align_e0,
                "align_e1": align_e1,
                **avg,
            })

            ckpt_data = {
                "epoch": epoch,
                "vca_state_dict": vca_layer.state_dict(),
                "z_pe": proc.z_pe.detach().cpu(),
                "gamma_trained": float(proc.gamma.item()),
                "dra": dra,
                "gen_diff": gen_diff,
                "lambda_pe_anti": args.lambda_pe_anti,
                "lambda_pe_norm": args.lambda_pe_norm,
                "pe_target_norm": args.pe_target_norm,
                "z_pe_lr": args.z_pe_lr,
                "z_pe_max_norm": args.z_pe_max_norm,
            }
            torch.save(ckpt_data, str(save_dir / "latest.pt"))

            save_best = False
            if dra > 0.0:
                if dra > best_dra:
                    best_dra = dra
                    save_best = True
            else:
                if best_epoch < 0 or gen_diff >= best_gen:
                    best_gen  = gen_diff
                    save_best = True

            if save_best:
                best_epoch = epoch
                torch.save(ckpt_data, str(save_dir / "best.pt"))
                print(f"  ★ best epoch={best_epoch} DRA={dra:.4f} "
                      f"gen_diff={gen_diff:.1f}px → {save_dir}/best.pt",
                      flush=True)

            vca_layer.train()
            proc.train()

        # ── 시각화 ───────────────────────────────────────────────────────
        if epoch % args.vis_every == 0 or epoch == args.epochs - 1:
            try:
                proc.set_entity_ctx(probe_entity_ctx.float())
                debug_generation(
                    pipe, vca_layer, orig_procs,
                    dict(pipe.unet.attn_processors),
                    probe_frames_np, probe_meta, probe_entity_ctx,
                    debug_dir, epoch,
                )
            except Exception as e:
                print(f"  [warn] debug_generation 실패: {e}", flush=True)

            _visualize_vol_attn(proc, probe_full_prompt, vca_layer,
                                debug_dir, epoch)

    print(f"\n[Phase 37] 훈련 완료. best epoch={best_epoch} DRA={best_dra:.4f}",
          flush=True)
    with open(debug_dir / "phase37_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  history → {debug_dir}/phase37_history.json", flush=True)
    return best_dra


# =============================================================================
# CLI
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Phase 37: Volumetric Text CA v3 (z_pe fix)")
    p.add_argument("--ckpt",             type=str,   default="checkpoints/phase31/best.pt")
    p.add_argument("--data-root",        type=str,   default="toy/data_objaverse")
    p.add_argument("--save-dir",         type=str,   default="checkpoints/phase37")
    p.add_argument("--debug-dir",        type=str,   default="debug/train_phase37")
    p.add_argument("--epochs",           type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--steps-per-epoch",  type=int,   default=20)
    p.add_argument("--n-frames",         type=int,   default=8)
    p.add_argument("--t-max",            type=int,   default=300)
    p.add_argument("--lr",               type=float, default=DEFAULT_LR)
    p.add_argument("--z-pe-lr",          type=float, default=DEFAULT_Z_PE_LR)
    p.add_argument("--z-pe-max-norm",    type=float, default=DEFAULT_Z_PE_MAX_NORM)
    p.add_argument("--lambda-vol",       type=float, default=DEFAULT_LAMBDA_VOL)
    p.add_argument("--lambda-depth",     type=float, default=DEFAULT_LAMBDA_DEPTH)
    p.add_argument("--lambda-diff",      type=float, default=DEFAULT_LAMBDA_DIFF)
    p.add_argument("--lambda-pe-anti",   type=float, default=DEFAULT_LAMBDA_PE_ANTI)
    p.add_argument("--lambda-pe-norm",   type=float, default=DEFAULT_LAMBDA_PE_NORM)
    p.add_argument("--pe-target-norm",   type=float, default=PE_TARGET_NORM)
    p.add_argument("--z-pe-std",         type=float, default=DEFAULT_Z_PE_STD)
    p.add_argument("--eval-every",       type=int,   default=5)
    p.add_argument("--vis-every",        type=int,   default=10)
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_phase37(args)
