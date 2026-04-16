from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from phase65_min3d.data import Phase65Dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase65_min3d/main.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ds = Phase65Dataset(
        root=cfg["data"]["root"],
        image_size=cfg["data"].get("image_size", 256),
        mask_size=cfg["data"].get("mask_size", 64),
        num_frames=cfg["data"].get("num_frames", 16),
    )
    vis_areas_e0, vis_areas_e1, amo_areas_e0, amo_areas_e1, overlap_ratios = [], [], [], [], []
    clip_types = {}
    for i in range(len(ds)):
        s = ds[i]
        vis0 = s.visible_masks[:, 0].mean().item()
        vis1 = s.visible_masks[:, 1].mean().item()
        amo0 = s.amodal_masks[:, 0].mean().item()
        amo1 = s.amodal_masks[:, 1].mean().item()
        overlap = (s.amodal_masks[:, 0] * s.amodal_masks[:, 1]).mean().item()
        vis_areas_e0.append(vis0)
        vis_areas_e1.append(vis1)
        amo_areas_e0.append(amo0)
        amo_areas_e1.append(amo1)
        overlap_ratios.append(overlap)
        clip_types[s.clip_type] = clip_types.get(s.clip_type, 0) + 1

    summary = {
        "num_samples": len(ds),
        "visible_mean_e0": float(np.mean(vis_areas_e0)),
        "visible_mean_e1": float(np.mean(vis_areas_e1)),
        "amodal_mean_e0": float(np.mean(amo_areas_e0)),
        "amodal_mean_e1": float(np.mean(amo_areas_e1)),
        "overlap_mean": float(np.mean(overlap_ratios)),
        "overlap_nonzero_frac": float(np.mean(np.array(overlap_ratios) > 0.01)),
        "clip_types": clip_types,
    }

    print(json.dumps(summary, indent=2))
    if summary["overlap_nonzero_frac"] < 0.2:
        raise SystemExit("Dataset sanity check failed: overlap_nonzero_frac too low for contact-heavy training")
    if min(summary["visible_mean_e0"], summary["visible_mean_e1"]) < 0.005:
        raise SystemExit("Dataset sanity check failed: one entity is nearly absent on average")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
