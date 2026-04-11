"""
Phase 36 — Volumetric Text Cross-Attention v2 (강화 버전)
==========================================================

Phase 35 실패 원인 및 수정
--------------------------
1. gamma=0.5  → standard attention 이 출력의 절반을 희석  → gamma=1.0 고정 (frozen)
2. z_pe norm ≈ 0.76 (dim=640)  → attention score 에 미미한 영향
   → z_pe init std=0.3 (15×), LR=full base LR
3. depth bin 이 같은 방향으로 collapse 가능
   → l_pe_sep: cos(z_pe[0], z_pe[1])^2 페널티 추가
4. z_pe 크기가 target 이하로 줄어들 수 있음
   → l_pe_norm: ||z_pe[z]|| ≥ sqrt(D) ≈ 25 강제

손실 함수 (Phase 36)
--------------------
  L = λ_vol    × l_vol_attn      ← volumetric attention GT supervision  (5.0)
    + λ_depth   × l_zorder_direct ← VCA sigma depth ordering              (3.0)
    + λ_diff    × l_diff          ← 생성 품질 보존                         (0.5)
    + λ_pe_sep  × l_pe_sep        ← z-bin cosine 분리 패널티              (1.0)
    + λ_pe_norm × l_pe_norm       ← z-bin norm collapse 방지              (0.5)

학습 파라미터
-----------
  - z_pe  (Z, D): full base LR, init std=0.3
  - vca_layer:    base LR (Phase 31 체크포인트에서 초기화)
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
    l_pe_sep,
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

# ─── Phase 36 하이퍼파라미터 ──────────────────────────────────────────────────
DEFAULT_LAMBDA_VOL    = 5.0     # 2.5× Phase 35
DEFAULT_LAMBDA_DEPTH  = 3.0
DEFAULT_LAMBDA_DIFF   = 0.5
DEFAULT_LAMBDA_PE_SEP  = 1.0   # NEW: depth bin 코사인 분리
DEFAULT_LAMBDA_PE_NORM = 0.5   # NEW: depth bin norm collapse 방지
DEFAULT_GAMMA_INIT    = 1.0    # 고정: 순수 volumetric
DEFAULT_Z_PE_STD      = 0.3    # 15× larger than Phase 35
DEFAULT_Z_BINS        = 2
DEFAULT_EPOCHS        = 150
PE_TARGET_NORM        = 25.0   # sqrt(640) ≈ 25 — dim 당 단위 norm


# =============================================================================
# 훈련 루프
# =============================================================================

def train_phase36(args):
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
    print("[Phase 36] 파이프라인 로드 중...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    # ── VCA 로드 (Phase 31 체크포인트) ───────────────────────────────────────
    print(f"[Phase 36] VCA 체크포인트 로드: {args.ckpt}", flush=True)
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
    print(f"[Phase 36] 데이터셋: {len(dataset)} 시퀀스", flush=True)
    if len(dataset) == 0:
        raise RuntimeError(f"데이터셋 비어있음: {args.data_root}")

    # ── Probe (고정 검증 샘플) ────────────────────────────────────────────────
    probe_idx = 0
    probe_sample = dataset[probe_idx]
    probe_frames_np, _, probe_depth_orders, probe_meta, probe_entity_masks = probe_sample
    probe_entity_ctx = get_color_entity_context(pipe, probe_meta, device)
    _, _, probe_full_prompt = get_entity_token_positions(pipe, probe_meta)

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

    # z_pe 를 더 크게 초기화 (Phase 35의 std=0.02 → Phase 36의 std=0.3)
    torch.nn.init.normal_(proc.z_pe, std=args.z_pe_std)
    print(f"  z_pe re-init: std={args.z_pe_std:.2f}  "
          f"|z_pe|={float(proc.z_pe.norm().item()):.4f}", flush=True)

    # ── gamma 고정 (frozen) ──────────────────────────────────────────────────
    # gamma=1.0 → 순수 volumetric, standard path 희석 없음
    proc.gamma.requires_grad_(False)
    print(f"  gamma = {float(proc.gamma.item()):.4f}  (frozen)", flush=True)

    # ── 옵티마이저: z_pe 는 full LR ─────────────────────────────────────────
    param_groups = [
        {"params": list(vca_layer.parameters()), "lr": args.lr,   "name": "vca"},
        {"params": [proc.z_pe],                  "lr": args.lr,   "name": "z_pe"},
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
    best_dra   = -1.0   # -1.0 → epoch 0에서도 반드시 저장
    best_gen   = 0.0    # DRA 실패 시 gen_diff 로 fallback
    best_epoch = -1

    print(f"\n[Phase 36] 훈련 시작: {args.epochs} epochs", flush=True)
    print(f"  λ_vol={args.lambda_vol}  λ_depth={args.lambda_depth}  "
          f"λ_diff={args.lambda_diff}", flush=True)
    print(f"  λ_pe_sep={args.lambda_pe_sep}  λ_pe_norm={args.lambda_pe_norm}  "
          f"pe_target_norm={args.pe_target_norm:.1f}", flush=True)

    for epoch in range(args.epochs):
        vca_layer.train()
        proc.train()

        epoch_losses = {"total": [], "vol_attn": [], "depth": [], "diff": [],
                        "pe_sep": [], "pe_norm": []}

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

            # ── l_pe_sep, l_pe_norm (NEW) ─────────────────────────────────
            l_sep  = l_pe_sep(proc.z_pe)
            l_norm = l_pe_norm(proc.z_pe, target_norm=args.pe_target_norm)

            # ── total loss ────────────────────────────────────────────────
            loss = (args.lambda_vol    * l_vol
                  + args.lambda_depth  * l_depth
                  + args.lambda_diff   * l_diff_val
                  + args.lambda_pe_sep  * l_sep
                  + args.lambda_pe_norm * l_norm)

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss ep={epoch} step={batch_idx} "
                      f"vol={l_vol.item():.4f} sep={l_sep.item():.4f} "
                      f"norm={l_norm.item():.4f} → skip", flush=True)
                continue

            loss.backward()
            # z_pe 와 vca_layer 를 별도 clip — z_pe 가 VCA 파라미터 수에 희석되지 않도록
            torch.nn.utils.clip_grad_norm_(list(vca_layer.parameters()), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([proc.z_pe], max_norm=1.0)
            optimizer.step()

            epoch_losses["total"].append(loss.item())
            epoch_losses["vol_attn"].append(
                l_vol.item() if isinstance(l_vol, torch.Tensor) else 0.0)
            epoch_losses["depth"].append(l_depth.item())
            epoch_losses["diff"].append(l_diff_val.item())
            epoch_losses["pe_sep"].append(l_sep.item())
            epoch_losses["pe_norm"].append(l_norm.item())

        scheduler.step()

        avg = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
        z_pe_norm_val = float(proc.z_pe.norm().item())
        # cosine between bins (diagnostic)
        with torch.no_grad():
            v0 = proc.z_pe[0].float()
            v1 = proc.z_pe[1].float()
            cos_bins = float(
                (torch.dot(v0, v1) / (v0.norm() * v1.norm() + 1e-8)).item())

        print(f"[Phase 36] epoch {epoch:03d}/{args.epochs-1}  "
              f"loss={avg['total']:.4f}  vol={avg['vol_attn']:.4f}  "
              f"sep={avg['pe_sep']:.4f}  norm={avg['pe_norm']:.4f}  "
              f"|z_pe|={z_pe_norm_val:.4f}  cos={cos_bins:.3f}", flush=True)

        # ── 검증 ─────────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            vca_layer.eval()
            proc.eval()

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
                **avg,
            })

            ckpt_data = {
                "epoch": epoch,
                "vca_state_dict": vca_layer.state_dict(),
                "z_pe": proc.z_pe.detach().cpu(),
                "gamma_trained": float(proc.gamma.item()),
                "dra": dra,
                "gen_diff": gen_diff,
                "lambda_pe_sep": args.lambda_pe_sep,
                "lambda_pe_norm": args.lambda_pe_norm,
                "pe_target_norm": args.pe_target_norm,
            }
            torch.save(ckpt_data, str(save_dir / "latest.pt"))

            # DRA 평가 성공 시 DRA 기준, 실패(=0.0) 시 gen_diff 기준으로 best 저장
            save_best = False
            if dra > 0.0:
                if dra > best_dra:
                    best_dra = dra
                    save_best = True
            else:
                # DRA 측정 실패 → gen_diff 로 fallback (클수록 VCA 활성, epoch 0은 반드시 저장)
                if best_epoch < 0 or gen_diff >= best_gen:
                    best_gen  = gen_diff
                    save_best = True

            if save_best:
                best_epoch = epoch
                torch.save(ckpt_data, str(save_dir / "best.pt"))
                print(f"  ★ best epoch={best_epoch} DRA={dra:.4f} "
                      f"gen_diff={gen_diff:.1f}px → {save_dir}/best.pt", flush=True)

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

    print(f"\n[Phase 36] 훈련 완료. best epoch={best_epoch} DRA={best_dra:.4f}",
          flush=True)
    with open(debug_dir / "phase36_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  history → {debug_dir}/phase36_history.json", flush=True)
    return best_dra


# =============================================================================
# CLI
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Phase 36: Volumetric Text CA v2")
    p.add_argument("--ckpt",             type=str,   default="checkpoints/phase31/best.pt")
    p.add_argument("--data-root",        type=str,   default="toy/data_objaverse")
    p.add_argument("--save-dir",         type=str,   default="checkpoints/phase36")
    p.add_argument("--debug-dir",        type=str,   default="debug/train_phase36")
    p.add_argument("--epochs",           type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--steps-per-epoch",  type=int,   default=20)
    p.add_argument("--n-frames",         type=int,   default=8)
    p.add_argument("--t-max",            type=int,   default=300)
    p.add_argument("--lr",               type=float, default=DEFAULT_LR)
    p.add_argument("--lambda-vol",       type=float, default=DEFAULT_LAMBDA_VOL)
    p.add_argument("--lambda-depth",     type=float, default=DEFAULT_LAMBDA_DEPTH)
    p.add_argument("--lambda-diff",      type=float, default=DEFAULT_LAMBDA_DIFF)
    p.add_argument("--lambda-pe-sep",    type=float, default=DEFAULT_LAMBDA_PE_SEP)
    p.add_argument("--lambda-pe-norm",   type=float, default=DEFAULT_LAMBDA_PE_NORM)
    p.add_argument("--pe-target-norm",   type=float, default=PE_TARGET_NORM)
    p.add_argument("--z-pe-std",         type=float, default=DEFAULT_Z_PE_STD)
    p.add_argument("--eval-every",       type=int,   default=5)
    p.add_argument("--vis-every",        type=int,   default=10)
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_phase36(args)
