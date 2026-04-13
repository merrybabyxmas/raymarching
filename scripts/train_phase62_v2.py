"""
Phase 62 — Training Entry Point (v2: Clean OOP)
=================================================

Config-driven, minimal orchestration. All logic lives in:
  - models/phase62/    — EntityVolumePredictor, FirstHitProjector, GuideFeatureAssembler, Phase62System
  - training/phase62/  — Phase62Trainer, Phase62Evaluator, Phase62RolloutRunner
  - data/phase62/      — VolumeGTBuilder, Phase62DatasetAdapter
  - config/phase62/    — YAML configs

Usage:
    python scripts/train_phase62_v2.py --config config/phase62/base.yaml
    python scripts/train_phase62_v2.py --config config/phase62/base.yaml --overlay config/phase62/train_smoke.yaml
    python scripts/train_phase62_v2.py --config config/phase62/base.yaml --override training.epochs=5 data.val_frac=0.3
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
from models.phase62.system import Phase62System
from models.phase62.backbone_adapter import (
    inject_backbone_extractors,
    BackboneManager,
    DEFAULT_INJECT_KEYS,
    BLOCK_INNER_DIMS,
)
from training.phase62.trainer import Phase62Trainer
from data.phase62.dataset_adapter import Phase62DatasetAdapter
from scripts.run_animatediff import load_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Phase 62: 3D Entity Volume + First-Hit Projection (v2)")
    parser.add_argument(
        "--config", type=str, default="config/phase62/base.yaml",
        help="Path to base YAML config")
    parser.add_argument(
        "--overlay", type=str, default=None,
        help="Path to overlay YAML (merged on top of base)")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="dotted.key=value overrides (e.g. training.epochs=5)")
    parser.add_argument(
        "--ckpt", type=str, default="",
        help="Checkpoint to resume from")
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints/phase62",
        help="Checkpoint save directory")
    parser.add_argument(
        "--debug-dir", type=str, default="outputs/phase62_debug",
        help="Debug output directory (GIFs, overlays)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, overrides=args.override, overlay=args.overlay)

    # Attach CLI paths to config
    config.save_dir = args.save_dir
    config.debug_dir = args.debug_dir

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    seed = config.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Pipeline
    print("[Phase 62 v2] Loading pipeline...", flush=True)
    pipe = load_pipeline(device=device)
    pipe.scheduler.set_timesteps(50, device=device)
    pipe.unet.enable_gradient_checkpointing()

    # Dataset
    data_root = config.data.data_root
    n_frames = config.training.n_frames
    print(f"[Phase 62 v2] Loading dataset from {data_root}...", flush=True)
    dataset = Phase62DatasetAdapter(data_root, n_frames=n_frames)
    print(f"  dataset size: {len(dataset)}", flush=True)

    # Inject backbone extractors (self-contained, no phase40 imports)
    print("[Phase 62 v2] Injecting backbone extractors...", flush=True)
    inject_keys = DEFAULT_INJECT_KEYS
    extractors, orig_procs = inject_backbone_extractors(
        pipe,
        adapter_rank=config.model.adapter_rank,
        lora_rank=config.model.lora_rank,
        inject_keys=inject_keys,
    )
    for ext in extractors:
        ext.to(device)
    backbone_mgr = BackboneManager(extractors, inject_keys, primary_idx=1)

    # Phase 62 system
    print("[Phase 62 v2] Creating Phase62 system...", flush=True)
    system = Phase62System(config).to(device)
    system.injection_mgr.register_hooks(pipe.unet)

    # Freeze base model
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    # Trainer
    trainer = Phase62Trainer(
        config=config,
        pipe=pipe,
        system=system,
        backbone_mgr=backbone_mgr,
        dataset=dataset,
        device=device,
    )

    # Checkpoint restore
    if args.ckpt and Path(args.ckpt).exists():
        trainer.load_checkpoint(args.ckpt)
    else:
        print("[Phase 62 v2] No checkpoint, starting from scratch.", flush=True)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
