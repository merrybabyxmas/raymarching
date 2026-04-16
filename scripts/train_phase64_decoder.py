"""
Phase 64 — Stage 2: Structured Decoder Training.

Requires: checkpoints/phase64/p64_stage1/best.pt

Usage:
    CUDA_VISIBLE_DEVICES=3 conda run -n paper_env python scripts/train_phase64_decoder.py \\
        --config config/phase64/stage2.yaml

Validates: scene outputs can reconstruct plausible images WITHOUT diffusion.
If reconstruction fails, do NOT proceed to Stage 3.
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
from training.phase64.stage2_train_decoder import Stage2Trainer


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
        description="Phase 64 Stage 2: Structured Decoder Training")
    parser.add_argument(
        "--config", type=str, default="config/phase64/stage2.yaml",
        help="Path to stage2 YAML config")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="dotted.key=value overrides (e.g. training.epochs=5)")
    parser.add_argument(
        "--ckpt", type=str, default="",
        help="Stage 2 checkpoint to resume from (decoder weights)")
    parser.add_argument(
        "--stage1_ckpt", type=str, default="",
        help="Override stage1 checkpoint path from config")
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
    print(f"[Phase 64 Decoder] device={device}  seed={seed}", flush=True)

    # ── Stage 1 checkpoint ───────────────────────────────────────────────────
    stage1_ckpt = args.stage1_ckpt or getattr(config, "stage1_ckpt",
                                               "checkpoints/phase64/p64_stage1/best.pt")
    if not Path(stage1_ckpt).exists():
        print(
            f"[Phase 64 Decoder] ERROR: Stage 1 checkpoint not found: {stage1_ckpt}\n"
            "  Run train_phase64_scene.py first.",
            flush=True,
        )
        sys.exit(1)
    print(f"[Phase 64 Decoder] Stage 1 ckpt: {stage1_ckpt}", flush=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root = getattr(config.data, "data_root",  "toy/data_objaverse")
    n_frames  = int(getattr(config.training, "n_frames", 1))
    val_frac  = float(getattr(config.data, "val_frac", 0.2))

    print(f"[Phase 64 Decoder] Loading dataset from {data_root}...", flush=True)
    dataset = Phase64Dataset(data_root, n_frames=n_frames)
    print(f"  dataset size: {len(dataset)}", flush=True)

    splits = make_splits(dataset, val_frac=val_frac, seed=seed)
    print(f"  train={len(splits['train'])}  val={len(splits['val'])}", flush=True)

    # ── Stage 2 Trainer ──────────────────────────────────────────────────────
    print("\n[Phase 64 Decoder] Building Stage2Trainer...", flush=True)
    trainer = Stage2Trainer(
        config=config,
        dataset=dataset,
        splits=splits,
        stage1_ckpt=stage1_ckpt,
        device=device,
    )

    # Optional resume
    if args.ckpt and Path(args.ckpt).exists():
        print(f"[Phase 64 Decoder] Resuming decoder from {args.ckpt}", flush=True)
        trainer.load_checkpoint(args.ckpt)
    else:
        print("[Phase 64 Decoder] Starting decoder from scratch.", flush=True)

    trainer.train()


if __name__ == "__main__":
    main()
