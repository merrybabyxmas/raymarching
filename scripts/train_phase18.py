"""
Phase 18: λ_depth 그리드 탐색 — 0.1 / 0.2 / 0.3

Phase 16(λ=1.0, 파괴)과 Phase 17(λ=0.02, 미약) 사이의 최적값 탐색.
각 λ로 30 epoch 학습 후 sigma_separation 비교.

불변 원칙 (FM-I7 교훈): l_diff > l_depth_weighted × 10 유지.
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
from scripts.train_phase17 import (
    adaptive_lambda_depth,
    check_generation_quality_mock,
    RATIO_WARNING_THRESH,
    QUALITY_CHECK_EVERY,
)

LAMBDA_GRID     = [0.1, 0.2, 0.3]
DEFAULT_EPOCHS  = 30
DEFAULT_LR      = 5e-5
DEFAULT_T_MAX   = 200
DEFAULT_ORTHO   = 0.005


def train_single(args):
    """단일 λ_depth 값으로 학습 실행."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}  lambda_depth={args.lambda_depth}", flush=True)

    # 데이터 품질 게이팅
    if args.stats_path and Path(args.stats_path).exists():
        if not check_dataset_quality(args.stats_path):
            return
    else:
        print("DATASET_OK: proceeding to training", flush=True)

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

    print(f"DATASET_OK: {len(dataset)} samples", flush=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=lambda x: x[0])

    pipe = load_pipeline(device=device, dtype=torch.float16)
    for p in pipe.unet.parameters():
        p.requires_grad = False

    save_dir  = Path(args.save_dir)
    debug_dir = Path(args.debug_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    probe_frames, _, probe_orders, probe_meta = dataset[0]
    probe_entity_ctx = get_entity_context_from_meta(pipe, probe_meta, device)
    print(
        f"[probe] entity_0='{probe_meta.get('prompt_entity0')}'  "
        f"entity_1='{probe_meta.get('prompt_entity1')}'",
        flush=True,
    )

    vca_layer, injected_keys, _ = inject_vca_train(pipe, probe_entity_ctx)
    trainable = [p for p in vca_layer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    print(f"[opt] trainable params: {sum(p.numel() for p in trainable):,}", flush=True)

    lambda_depth = args.lambda_depth
    lambda_ortho = args.lambda_ortho
    best_sep     = 0.0
    training_curve = []
    quality_checks = []

    for epoch in range(args.epochs):
        vca_layer.train()
        epoch_losses = {"loss": 0., "l_diff": 0., "l_depth": 0., "l_ortho": 0.}
        epoch_steps  = 0
        last_frames_np = probe_frames

        for batch in loader:
            frames_np, depths_np, depth_orders, meta = batch
            last_frames_np = frames_np

            entity_ctx = get_entity_context_from_meta(pipe, meta, device)
            for key in injected_keys:
                proc = pipe.unet.attn_processors.get(key)
                if isinstance(proc, TrainVCAProcessor):
                    proc.entity_context = entity_ctx

            latents = encode_frames_to_latents(pipe, frames_np, device)

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

        for k in epoch_losses:
            epoch_losses[k] /= max(epoch_steps, 1)

        stats     = compute_sigma_stats_train(vca_layer.last_sigma)
        sep       = stats["sigma_separation"]
        l_diff_v  = epoch_losses["l_diff"]
        l_depth_w = epoch_losses["l_depth"] * lambda_depth
        ratio     = l_diff_v / max(l_depth_w, 1e-9)

        if l_depth_w > 0 and ratio < 1.0 / RATIO_WARNING_THRESH:
            print(
                f"  WARNING: ratio={ratio:.1f}x < 10 — adaptive λ_depth",
                flush=True,
            )
            lambda_depth = adaptive_lambda_depth(l_diff_v, l_depth_w, lambda_depth)

        print(
            f"epoch={epoch:3d} step={epoch_steps} "
            f"loss={epoch_losses['loss']:.4f} "
            f"l_diff={l_diff_v:.4f} "
            f"l_depth={l_depth_w:.4f} "
            f"l_ortho={epoch_losses['l_ortho']:.4f} "
            f"ratio={ratio:.1f}x "
            f"sep={sep:.3f}",
            flush=True,
        )

        training_curve.append({
            "epoch": epoch, "lambda_depth": lambda_depth,
            **epoch_losses, "l_depth_weighted": l_depth_w, **stats,
        })

        if sep > best_sep:
            best_sep = sep
            torch.save({
                "vca_state_dict":      vca_layer.state_dict(),
                "epoch":               epoch,
                "sigma_separation":    sep,
                "lambda_depth_init":   args.lambda_depth,
                "lambda_depth_final":  lambda_depth,
                "quality_checks":      quality_checks,
            }, save_dir / "best.pt")
            print(f"[ckpt] best.pt (sep={best_sep:.4f})", flush=True)

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            gif_path = debug_dir / f"sigma_epoch{epoch:03d}.gif"
            save_sigma_gif(last_frames_np, vca_layer.last_sigma, gif_path)
            print(f"[gif] → {gif_path}", flush=True)

    print(f"FINAL sigma_separation={best_sep:.6f}", flush=True)
    if best_sep > 0.01:
        print("LEARNING=OK", flush=True)
    else:
        print("LEARNING=FAIL", flush=True)

    curve_path = debug_dir / "training_curve.json"
    with open(curve_path, "w") as f:
        json.dump(training_curve, f, indent=2)
    print(f"[done] training_curve → {curve_path}", flush=True)


# ─── 그리드 탐색 결과 요약 ────────────────────────────────────────────────────

def summarize_grid(save_root: Path) -> dict:
    """
    각 λ 체크포인트에서 sigma_separation 읽어 winner 결정.
    반환: {"best_lambda": float, "results": [{lambda, sep, epoch}]}
    """
    results = []
    for lam in LAMBDA_GRID:
        ckpt_path = save_root / f"lambda_{lam:.1f}" / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            results.append({
                "lambda": lam,
                "sep":    ckpt.get("sigma_separation", 0.0),
                "epoch":  ckpt.get("epoch", -1),
            })

    if not results:
        return {"best_lambda": None, "results": []}

    best = max(results, key=lambda x: x["sep"])
    return {"best_lambda": best["lambda"], "results": results}


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",     default="toy/data_objaverse",  dest="data_root")
    p.add_argument("--stats-path",    default="debug/dataset_stats/objaverse_stats.json",
                   dest="stats_path")
    p.add_argument("--lambda-depth",  type=float, required=True, dest="lambda_depth")
    p.add_argument("--lambda-ortho",  type=float, default=DEFAULT_ORTHO, dest="lambda_ortho")
    p.add_argument("--epochs",        type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--lr",            type=float, default=DEFAULT_LR)
    p.add_argument("--t-max",         type=int,   default=DEFAULT_T_MAX, dest="t_max")
    p.add_argument("--save-dir",      required=True, dest="save_dir")
    p.add_argument("--debug-dir",     required=True, dest="debug_dir")
    p.add_argument("--n-frames",      type=int,   default=8,  dest="n_frames")
    p.add_argument("--height",        type=int,   default=256)
    p.add_argument("--width",         type=int,   default=256)
    p.add_argument("--max-samples",   type=int,   default=None, dest="max_samples")
    return p.parse_args()


if __name__ == "__main__":
    train_single(parse_args())
