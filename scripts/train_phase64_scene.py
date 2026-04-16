"""
Phase 64 — Stage 0 (dataset validation) + Stage 1 (backbone-agnostic scene prior).

Usage:
    CUDA_VISIBLE_DEVICES=3 conda run -n paper_env python scripts/train_phase64_scene.py \\
        --config config/phase64/stage1.yaml [--skip_stage0]

Stage 0 computes oracle stats first. Stage 1 trains the backbone-agnostic scene prior.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from data.phase64 import Phase64Dataset, make_splits
from training.phase64.stage0_validate_dataset import compute_and_save_stats
from training.phase64.stage1_train_scene_prior import Stage1Trainer


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 64 Stage 0+1: Dataset Validation + Scene Prior Training")
    parser.add_argument(
        "--config", type=str, default="config/phase64/stage1.yaml",
        help="Path to stage1 YAML config")
    parser.add_argument(
        "--skip_stage0", action="store_true",
        help="Skip Stage 0 dataset validation (use if already validated)")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="dotted.key=value overrides (e.g. training.epochs=5)")
    parser.add_argument(
        "--ckpt", type=str, default="",
        help="Checkpoint to resume from")
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to train on")
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    config = load_config(args.config, overrides=args.override)

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed = int(getattr(config.training, "seed", 42))
    _seed_all(seed)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"[Phase 64 Scene] device={device}  seed={seed}", flush=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root  = getattr(config.data, "data_root",  "toy/data_objaverse")
    n_frames   = int(getattr(config.data, "n_frames", 1))
    val_frac   = float(getattr(config.data, "val_frac", 0.2))

    print(f"[Phase 64 Scene] Loading dataset from {data_root}...", flush=True)
    dataset = Phase64Dataset(data_root, n_frames=n_frames)
    print(f"  dataset size: {len(dataset)}", flush=True)

    splits = make_splits(dataset, val_frac=val_frac, seed=seed)
    print(
        f"  train={len(splits['train'])}  val={len(splits['val'])}  "
        f"O={len(splits['split_O'])}  C={len(splits['split_C'])}  "
        f"R={len(splits['split_R'])}  X={len(splits['split_X'])}",
        flush=True,
    )

    # ── Stage 0: Dataset Validation ──────────────────────────────────────────
    if not args.skip_stage0:
        print("\n[Phase 64 Stage 0] Running dataset validation...", flush=True)
        stage0_cfg_path = Path(args.config).parent / "stage0.yaml"
        if stage0_cfg_path.exists():
            stage0_cfg = load_config(str(stage0_cfg_path))
        else:
            # Fall back to current config's data section
            stage0_cfg = config

        stats = compute_and_save_stats(
            dataset=dataset,
            splits=splits,
            config=stage0_cfg,
        )
        print(f"[Phase 64 Stage 0] Stats saved. "
              f"oracle_amo_iou={stats.get('oracle_amo_iou_mean', 0.0):.4f}",
              flush=True)
    else:
        print("[Phase 64 Stage 0] Skipped (--skip_stage0).", flush=True)

    # ── Stage 1: Scene Prior Training ────────────────────────────────────────
    print("\n[Phase 64 Stage 1] Building Stage1Trainer...", flush=True)
    trainer = Stage1Trainer(
        config=config,
        dataset=dataset,
        splits=splits,
        device=str(device),
    )

    # Optional checkpoint restore
    if args.ckpt and Path(args.ckpt).exists():
        print(f"[Phase 64 Stage 1] Resuming from {args.ckpt}", flush=True)
        trainer.load_checkpoint(args.ckpt)
    else:
        print("[Phase 64 Stage 1] Starting from scratch.", flush=True)

    trainer.train()


if __name__ == "__main__":
    main()
