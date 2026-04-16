"""
Phase 64 — Stage 3/4: Backbone Adapter Training.

Stage 3: AnimateDiff adapter (requires stage1+stage2 checkpoints)
Stage 4: SDXL adapter (same scene prior, new adapter = transfer test)

Usage:
    # Stage 3 (AnimateDiff):
    CUDA_VISIBLE_DEVICES=3 conda run -n paper_env python scripts/train_phase64_backbone.py \\
        --config config/phase64/stage3.yaml --stage 3

    # Stage 4 (SDXL transfer):
    CUDA_VISIBLE_DEVICES=3 conda run -n paper_env python scripts/train_phase64_backbone.py \\
        --config config/phase64/stage4.yaml --stage 4
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
from training.phase64.stage3_train_adapter_backbone import Stage3Trainer
from training.phase64.stage4_transfer_eval import Stage4TransferEval


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _require_ckpt(path: str, label: str) -> None:
    """Abort with a clear message if a required checkpoint is missing."""
    if not Path(path).exists():
        print(
            f"[Phase 64 Backbone] ERROR: {label} checkpoint not found: {path}",
            flush=True,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 64 Stage 3/4: Backbone Adapter Training")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to stage3 or stage4 YAML config")
    parser.add_argument(
        "--stage", type=int, choices=[3, 4], required=True,
        help="Which stage to run: 3 (AnimateDiff) or 4 (SDXL)")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="dotted.key=value overrides")
    parser.add_argument(
        "--ckpt", type=str, default="",
        help="Checkpoint to resume adapter training from")
    parser.add_argument(
        "--stage1_ckpt", type=str, default="",
        help="Override stage1 checkpoint path from config")
    parser.add_argument(
        "--stage2_ckpt", type=str, default="",
        help="Override stage2 checkpoint path from config (stage 3 only)")
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
    print(f"[Phase 64 Backbone] stage={args.stage}  device={device}  seed={seed}",
          flush=True)

    # ── Checkpoint resolution ─────────────────────────────────────────────────
    stage1_ckpt = (args.stage1_ckpt
                   or getattr(config, "stage1_ckpt",
                               "checkpoints/phase64/p64_stage1/best.pt"))
    _require_ckpt(stage1_ckpt, "Stage 1")
    print(f"[Phase 64 Backbone] Stage 1 ckpt: {stage1_ckpt}", flush=True)

    stage2_ckpt = ""
    if args.stage == 3:
        stage2_ckpt = (args.stage2_ckpt
                       or getattr(config, "stage2_ckpt",
                                  "checkpoints/phase64/p64_stage2/best.pt"))
        _require_ckpt(stage2_ckpt, "Stage 2")
        print(f"[Phase 64 Backbone] Stage 2 ckpt: {stage2_ckpt}", flush=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root = getattr(config.data, "data_root", "toy/data_objaverse")
    n_frames  = int(getattr(config.training, "n_frames", 1))
    val_frac  = float(getattr(config.data, "val_frac", 0.2))

    print(f"[Phase 64 Backbone] Loading dataset from {data_root}...", flush=True)
    dataset = Phase64Dataset(data_root, n_frames=n_frames)
    print(f"  dataset size: {len(dataset)}", flush=True)

    splits = make_splits(dataset, val_frac=val_frac, seed=seed)
    print(f"  train={len(splits['train'])}  val={len(splits['val'])}", flush=True)

    # ── Stage dispatch ───────────────────────────────────────────────────────
    if args.stage == 3:
        print("\n[Phase 64 Backbone] Stage 3 — AnimateDiff adapter training...",
              flush=True)

        # Load AnimateDiff pipeline (same as Phase 63 approach)
        print("[Phase 64 Backbone] Loading AnimateDiff pipeline...", flush=True)
        from scripts.run_animatediff import load_pipeline as _load_pipe
        pipe = _load_pipe(device=device)
        print(f"[Phase 64 Backbone] Pipeline loaded on {device}", flush=True)

        trainer = Stage3Trainer(
            config=config,
            dataset=dataset,
            splits=splits,
            stage1_ckpt=stage1_ckpt,
            stage2_ckpt=stage2_ckpt,
            device=device,
            pipe=pipe,
        )
        if args.ckpt and Path(args.ckpt).exists():
            print(f"[Phase 64 Backbone] Resuming from {args.ckpt}", flush=True)
            trainer.load_checkpoint(args.ckpt)
        else:
            print("[Phase 64 Backbone] Starting adapter from scratch.", flush=True)
        trainer.train()

    elif args.stage == 4:
        print("\n[Phase 64 Backbone] Stage 4 — SDXL transfer adapter training...",
              flush=True)

        # Load SDXL pipeline
        print("[Phase 64 Backbone] Loading SDXL pipeline...", flush=True)
        from diffusers import AutoPipelineForText2Image
        import torch as _torch
        sdxl_model_id = getattr(
            getattr(config, "backbone", object()),
            "pipeline",
            "stabilityai/sdxl-turbo",
        )
        sdxl_pipe = AutoPipelineForText2Image.from_pretrained(
            sdxl_model_id, torch_dtype=_torch.float16, variant="fp16",
        ).to(device)
        # Reduce memory: enable attention slicing
        sdxl_pipe.enable_attention_slicing()
        print(f"[Phase 64 Backbone] SDXL pipeline loaded on {device}", flush=True)

        transfer = Stage4TransferEval(
            config=config,
            dataset=dataset,
            splits=splits,
            stage1_ckpt=stage1_ckpt,
            device=device,
            sdxl_pipe=sdxl_pipe,
        )
        if args.ckpt and Path(args.ckpt).exists():
            print(f"[Phase 64 Backbone] Resuming SDXL adapter from {args.ckpt}",
                  flush=True)
            transfer.load_checkpoint(args.ckpt)
        else:
            print("[Phase 64 Backbone] Starting SDXL adapter from scratch.", flush=True)
        transfer.train()


if __name__ == "__main__":
    main()
