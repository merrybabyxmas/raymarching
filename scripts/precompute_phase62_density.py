"""
Phase 62 density precompute
===========================

Builds actual entity density cache from rendered V_gt for the Phase 62
training split. This is the non-proxy path for entity-rich sampling.

Usage:
    python scripts/precompute_phase62_density.py \
        --config config/phase62/base.yaml \
        --overlay config/phase62/train_smoke.yaml
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
from data.phase62.dataset_adapter import Phase62DatasetAdapter
from data.phase62.volume_gt_builder import VolumeGTBuilder
from scripts.train_phase39 import (
    compute_dataset_overlap_scores,
    split_train_val,
)

MIN_VAL_SAMPLES = 3


def main():
    p = argparse.ArgumentParser(description="Precompute actual Phase62 density cache")
    p.add_argument("--config", type=str, default="config/phase62/base.yaml")
    p.add_argument("--overlay", type=str, default=None)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    cfg = load_config(args.config, overrides=args.override, overlay=args.overlay)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = Phase62DatasetAdapter(cfg.data.data_root, n_frames=cfg.training.n_frames)
    raw_ds = dataset.raw_dataset()
    overlap_scores = compute_dataset_overlap_scores(raw_ds)
    train_idx, _ = split_train_val(
        overlap_scores,
        val_frac=getattr(cfg.data, "val_frac", 0.2),
        min_val=MIN_VAL_SAMPLES,
    )

    save_dir = Path(args.out) if args.out else Path("checkpoints/phase62")
    save_dir.mkdir(parents=True, exist_ok=True)
    density_cache = save_dir / "entity_density_scores.npy"

    builder = VolumeGTBuilder(
        depth_bins=cfg.depth_bins,
        spatial_h=cfg.spatial_h,
        spatial_w=cfg.spatial_w,
        render_resolution=getattr(cfg.data, "volume_gt_render_resolution", 128),
    )

    scores = []
    for i, idx in enumerate(train_idx):
        sample = dataset[idx]
        V = builder.build_batch(
            sample["depth"][:cfg.training.n_frames],
            sample["entity_masks"][:cfg.training.n_frames],
            sample["depth_orders"][:cfg.training.n_frames],
            visible_masks=(sample["visible_masks"][:cfg.training.n_frames]
                           if sample.get("visible_masks") is not None else None),
            meta=sample["meta"],
            sample_dir=sample["sample_dir"],
        )
        density = float((V > 0).mean())
        scores.append(density)
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(train_idx):
            print(f"[{i+1:03d}/{len(train_idx):03d}] idx={idx} density={density:.6f}")

    scores = np.asarray(scores, dtype=np.float32)
    np.save(density_cache, scores)
    print(f"Saved density cache to {density_cache}")
    print(f"mean={scores.mean():.6f} min={scores.min():.6f} max={scores.max():.6f}")


if __name__ == "__main__":
    main()
