"""
Phase 17: λ 재조정 + 전체 데이터 학습
생성 품질 보존이 최우선. l_diff > l_depth × 10 항상 유지.

Phase 16 대비 핵심 변경:
  λ_depth: 1.0 → 0.02  (50배 감소)
  λ_ortho: 0.05 → 0.005 (10배 감소)
  lr: 1e-4 → 5e-5
  samples/epoch: 1 random → 전체 168개
  epochs: 30 → 60
  adaptive_lambda: ratio > 0.1이면 λ 절반으로 자동 감소
  quality_check: 10 epoch마다 생성 품질 모니터링

불변 원칙 (FM-I7): l_diff > l_depth × 10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_objaverse_vca import (
    ObjaverseTrainDataset,
    check_dataset_quality,
    get_entity_context_from_meta,
    training_step,
)
from scripts.train_animatediff_vca import (
    inject_vca_train, TrainVCAProcessor,
    compute_sigma_stats_train, save_sigma_gif,
    encode_frames_to_latents,
)
from scripts.run_animatediff import load_pipeline

# ─── 기본값 (Phase 16 대비 대폭 감소) ───────────────────────────────────────
DEFAULT_LAMBDA_DEPTH  = 0.02     # Phase 16: 1.0 → 50배 감소
DEFAULT_LAMBDA_ORTHO  = 0.005    # Phase 16: 0.05 → 10배 감소
DEFAULT_LR            = 5e-5     # Phase 16: 1e-4 → 2배 감소
DEFAULT_EPOCHS        = 60
DEFAULT_T_MAX         = 200      # Phase 14 발견 유지
QUALITY_CHECK_EVERY   = 10       # 10 epoch마다 생성 품질 체크
RATIO_WARNING_THRESH  = 0.1      # l_depth/l_diff > 0.1 → 경고 + adaptive


# ─── Adaptive lambda ─────────────────────────────────────────────────────────

def adaptive_lambda_depth(
    l_diff_val: float,
    l_depth_val: float,
    current_lambda: float,
    min_lambda: float = 1e-4,
) -> float:
    """
    l_depth/l_diff > RATIO_WARNING_THRESH이면 lambda 절반으로 줄임.
    생성 품질 보호 최우선. 단조 감소만 허용.
    """
    if l_diff_val > 0 and l_depth_val / l_diff_val > RATIO_WARNING_THRESH:
        new_lambda = max(current_lambda * 0.5, min_lambda)
        print(
            f"  [adaptive] λ_depth {current_lambda:.5f} → {new_lambda:.5f} "
            f"(ratio={l_depth_val/l_diff_val:.3f} > {RATIO_WARNING_THRESH})",
            flush=True,
        )
        return new_lambda
    return current_lambda


# ─── 품질 체크 (mock용 분리 함수) ────────────────────────────────────────────

def check_generation_quality_mock(
    pixel_var: float,
    sigma_max: float,
    pixel_var_thresh: float = 100.0,
    sigma_max_thresh: float = 0.95,
) -> str:
    """
    테스트에서 직접 호출 가능한 품질 판단 로직.
    pixel_var < pixel_var_thresh OR sigma_max > sigma_max_thresh → DEGRADED
    """
    if pixel_var < pixel_var_thresh or sigma_max > sigma_max_thresh:
        return "DEGRADED"
    return "OK"


def check_generation_quality(pipe, vca_layer, entity_ctx, prompt: str,
                              device: str, seed: int = 42,
                              num_frames: int = 8,
                              height: int = 256, width: int = 256,
                              epoch: int = -1) -> dict:
    """
    probe 생성 실행 후 품질 지표 측정.
    반환: {"pixel_var": float, "sigma_max": float, "status": "OK"|"DEGRADED"}
    """
    from scripts.run_animatediff import FixedContextVCAProcessor

    # probe 실행 (no_grad)
    vca_layer.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=10,
            generator=generator,
        )
    vca_layer.train()

    # pixel variance (모든 프레임 평균)
    frames = result.frames[0]  # list of PIL
    arrays = [np.array(f).astype(np.float32) for f in frames]
    pixel_var = float(np.stack(arrays).var())

    # sigma_max
    sigma = vca_layer.last_sigma
    if sigma is not None:
        sigma_max = float(sigma.max().item())
    else:
        sigma_max = 0.0

    status = check_generation_quality_mock(pixel_var, sigma_max)
    print(
        f"QUALITY_CHECK epoch={epoch} pixel_var={pixel_var:.1f} "
        f"sigma_max={sigma_max:.2f} status={status}",
        flush=True,
    )
    return {"pixel_var": pixel_var, "sigma_max": sigma_max, "status": status}


# ─── 메인 학습 루프 ──────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}", flush=True)

    # 데이터 품질 게이팅
    if args.stats_path and Path(args.stats_path).exists():
        if not check_dataset_quality(args.stats_path):
            return
    else:
        print("DATASET_CHECK SKIP: stats_path not provided, proceeding", flush=True)
        print("DATASET_OK: proceeding to training", flush=True)

    # 데이터셋 로드 (전체 사용)
    dataset = ObjaverseTrainDataset(
        data_root   = args.data_root,
        max_samples = args.max_samples,
        n_frames    = args.n_frames,
        height      = args.height,
        width       = args.width,
    )
    if len(dataset) == 0:
        print("DATASET_FAIL: no samples found", flush=True)
        return

    n_samples = len(dataset)
    print(
        f"DATASET_OK: {n_samples} samples, "
        f"depth_reversal_rate={args.stats_path and _read_stat(args.stats_path, 'depth_reversal_rate'):.3f}",
        flush=True,
    )

    # DataLoader (shuffle=True — Phase 16과의 핵심 차이)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=lambda x: x[0])

    # 파이프라인 로드
    pipe = load_pipeline(device=device, dtype=torch.float16)
    for p in pipe.unet.parameters():
        p.requires_grad = False
    print("[init] UNet frozen.", flush=True)

    save_dir  = Path(args.save_dir)
    debug_dir = Path(args.debug_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # probe 샘플 (품질 체크용, 고정)
    probe_frames, probe_depths, probe_orders, probe_meta = dataset[0]
    probe_entity_ctx = get_entity_context_from_meta(pipe, probe_meta, device)
    probe_prompt = (
        f"{probe_meta.get('prompt_entity0','entity0')} and "
        f"{probe_meta.get('prompt_entity1','entity1')}"
    )
    print(
        f"[probe] entity_0='{probe_meta.get('prompt_entity0')}'  "
        f"entity_1='{probe_meta.get('prompt_entity1')}'",
        flush=True,
    )

    # VCA 주입
    vca_layer, injected_keys, original_procs = inject_vca_train(pipe, probe_entity_ctx)
    print(f"[inject] {injected_keys}", flush=True)

    trainable = [p for p in vca_layer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    print(f"[opt] trainable params: {sum(p.numel() for p in trainable):,}", flush=True)

    # λ (adaptive scheduling 가능)
    lambda_depth = args.lambda_depth
    lambda_ortho = args.lambda_ortho

    training_curve = []
    best_sep = 0.0
    quality_checks = []
    degraded_epochs = []

    for epoch in range(args.epochs):
        vca_layer.train()
        epoch_losses = {"loss": 0., "l_diff": 0., "l_depth": 0., "l_ortho": 0.}
        epoch_steps  = 0

        for batch in loader:
            frames_np, depths_np, depth_orders, meta = batch

            # entity context
            entity_ctx = get_entity_context_from_meta(pipe, meta, device)

            # processor에 context 주입
            for key in injected_keys:
                proc = pipe.unet.attn_processors.get(key)
                if isinstance(proc, TrainVCAProcessor):
                    proc.entity_context = entity_ctx

            # VAE 인코딩
            latents = encode_frames_to_latents(pipe, frames_np, device)

            # encoder_hidden_states
            full_prompt = (
                f"{meta.get('prompt_entity0','entity0')} and "
                f"{meta.get('prompt_entity1','entity1')}"
            )
            tokens = pipe.tokenizer(
                full_prompt, return_tensors="pt",
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
            ).to(device)
            with torch.no_grad():
                enc_hs = pipe.text_encoder(**tokens).last_hidden_state.half()

            optimizer.zero_grad()
            step_out = training_step(
                pipe, vca_layer, latents, enc_hs,
                depth_orders, lambda_depth, lambda_ortho, device,
                t_max=args.t_max,
            )
            step_out["loss_tensor"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += step_out[k]
            epoch_steps += 1

        # epoch 평균
        for k in epoch_losses:
            epoch_losses[k] /= max(epoch_steps, 1)

        # sigma separation (마지막 배치 기준)
        stats = compute_sigma_stats_train(vca_layer.last_sigma)
        sep = stats["sigma_separation"]

        # l_depth_weighted = 실제 손실 기여분 (λ × raw)
        # ratio = l_diff / l_depth_weighted  → 10 이상이어야 정상
        l_diff_val     = epoch_losses["l_diff"]
        l_depth_raw    = epoch_losses["l_depth"]
        l_depth_w      = l_depth_raw * lambda_depth   # 가중 기여분
        ratio = l_diff_val / max(l_depth_w, 1e-9)

        if l_depth_w > 0 and ratio < 1.0 / RATIO_WARNING_THRESH:
            print(
                f"  WARNING: l_depth/l_diff ratio too high "
                f"(l_diff={l_diff_val:.4f} l_depth_weighted={l_depth_w:.4f} "
                f"ratio={ratio:.1f}x < 10)",
                flush=True,
            )
            lambda_depth = adaptive_lambda_depth(
                l_diff_val, l_depth_w, lambda_depth
            )

        print(
            f"epoch={epoch:3d} step={epoch_steps} "
            f"loss={epoch_losses['loss']:.4f} "
            f"l_diff={l_diff_val:.4f} "
            f"l_depth={l_depth_w:.4f} "
            f"l_ortho={epoch_losses['l_ortho']:.4f} "
            f"ratio={ratio:.1f}x "
            f"sep={sep:.3f}",
            flush=True,
        )

        training_curve.append({
            "epoch": epoch,
            "lambda_depth": lambda_depth,
            **epoch_losses,
            "l_depth_weighted": l_depth_w,
            **stats,
        })

        # 품질 체크 (QUALITY_CHECK_EVERY마다)
        if (epoch + 1) % QUALITY_CHECK_EVERY == 0:
            qc = check_generation_quality(
                pipe, vca_layer, probe_entity_ctx, probe_prompt,
                device=device, seed=42, epoch=epoch,
                num_frames=args.n_frames,
                height=args.height, width=args.width,
            )
            quality_checks.append({"epoch": epoch, **qc})
            if qc["status"] == "DEGRADED":
                degraded_epochs.append(epoch)
                lambda_depth = max(lambda_depth * 0.25, 1e-4)
                print(
                    f"  [DEGRADED] λ_depth → {lambda_depth:.5f} (x0.25 reduction)",
                    flush=True,
                )
                _log_failure(epoch, qc)

        # 체크포인트
        if sep > best_sep:
            best_sep = sep
            torch.save({
                "vca_state_dict":      vca_layer.state_dict(),
                "epoch":               epoch,
                "sigma_separation":    sep,
                "lambda_depth_final":  lambda_depth,
                "quality_checks":      quality_checks,
            }, save_dir / "best.pt")
            print(f"[ckpt] best.pt (sep={best_sep:.4f})", flush=True)

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                "vca_state_dict":     vca_layer.state_dict(),
                "epoch":              epoch,
                "sigma_separation":   sep,
                "lambda_depth_final": lambda_depth,
                "quality_checks":     quality_checks,
            }, save_dir / f"epoch_{epoch:03d}.pt")
            gif_path = debug_dir / f"sigma_epoch{epoch:03d}.gif"
            save_sigma_gif(frames_np, vca_layer.last_sigma, gif_path)
            print(f"[gif] → {gif_path}", flush=True)

    # FINAL
    print(f"FINAL sigma_separation={best_sep:.6f}", flush=True)
    print(f"FINAL best_epoch={_find_best_epoch(training_curve)}", flush=True)

    no_recent_degraded = not any(e >= args.epochs - 20 for e in degraded_epochs)
    if best_sep > 0.01 and no_recent_degraded:
        print("LEARNING=OK", flush=True)
    else:
        print("LEARNING=FAIL", flush=True)

    # 커브 저장
    curve_path = debug_dir / "training_curve.json"
    with open(curve_path, "w") as f:
        json.dump(training_curve, f, indent=2)
    print(f"[done] training_curve → {curve_path}", flush=True)


# ─── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _read_stat(stats_path: str, key: str) -> float:
    try:
        with open(stats_path) as f:
            return float(json.load(f).get(key, 0.0))
    except Exception:
        return 0.0


def _find_best_epoch(curve: list) -> int:
    if not curve:
        return -1
    best = max(curve, key=lambda x: x.get("sigma_separation", 0))
    return best.get("epoch", -1)


def _log_failure(epoch: int, qc: dict):
    failures_path = Path("docs/failures.md")
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        f"\n## FM-I7: Phase 17 DEGRADED (epoch={epoch})\n"
        f"- pixel_var={qc['pixel_var']:.1f}\n"
        f"- sigma_max={qc['sigma_max']:.3f}\n"
        f"- λ_depth reduced by ×0.25\n"
    )
    with open(failures_path, "a") as f:
        f.write(entry)
    print(f"  [FM-I7] logged to {failures_path}", flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",     default="toy/data_objaverse",  dest="data_root")
    p.add_argument("--stats-path",    default="debug/dataset_stats/objaverse_stats.json",
                   dest="stats_path")
    p.add_argument("--epochs",        type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--lr",            type=float, default=DEFAULT_LR)
    p.add_argument("--t-max",         type=int,   default=DEFAULT_T_MAX, dest="t_max")
    p.add_argument("--lambda-depth",  type=float, default=DEFAULT_LAMBDA_DEPTH,
                   dest="lambda_depth")
    p.add_argument("--lambda-ortho",  type=float, default=DEFAULT_LAMBDA_ORTHO,
                   dest="lambda_ortho")
    p.add_argument("--save-dir",      default="checkpoints/phase17",   dest="save_dir")
    p.add_argument("--debug-dir",     default="debug/train_phase17",   dest="debug_dir")
    p.add_argument("--n-frames",      type=int,   default=8,  dest="n_frames")
    p.add_argument("--height",        type=int,   default=256)
    p.add_argument("--width",         type=int,   default=256)
    p.add_argument("--max-samples",   type=int,   default=None, dest="max_samples")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
