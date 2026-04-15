"""
Phase 62 — OCC benchmark dataset generator
============================================

Generates depth-separated (occlusion) toy data in toy/data_objaverse_occ/.
Uses CrossingMode.LAYERED: entity0 is in front, entity1 is behind (partially occluded).

Needed for Priority 3 (analysis.md §12):
  - same-depth col benchmark → existing toy/data_objaverse/
  - depth-separated occ benchmark → toy/data_objaverse_occ/ (this script)

After generation, run volume_gt precompute:
  python scripts/precompute_occ_volume_gt.py --data-dir toy/data_objaverse_occ

Usage:
    python scripts/generate_occ_dataset.py [--n-cameras N] [--n-frames N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_crossing import (
    CrossingMode,
    generate_dataset,
    CAMERAS,
)


# Cameras that give good front-back depth contrast for LAYERED mode
# Using only front-facing cameras where occlusion is clearly visible
OCC_CAMERAS = [
    c for c in CAMERAS if c['name'] in {
        'front', 'front_right', 'front_left', 'top'
    }
]


def main():
    p = argparse.ArgumentParser(description="Generate occ depth-separated benchmark data")
    p.add_argument('--assets-dir',  default='toy/assets',            dest='assets_dir')
    p.add_argument('--out-dir',     default='toy/data_objaverse_occ', dest='out_dir')
    p.add_argument('--n-cameras',   type=int, default=4,              dest='n_cameras')
    p.add_argument('--n-frames',    type=int, default=16,             dest='n_frames')
    p.add_argument('--resolution',  type=int, default=256)
    p.add_argument('--seed',        type=int, default=43)   # different seed from col (42)
    p.add_argument('--pairs',       default='',
                   help='comma-separated keyword pairs, e.g. cat_dog (default: all)')
    args = p.parse_args()

    pair_filter = [x.strip() for x in args.pairs.split(',') if x.strip()] or None

    print(f"Generating OCC benchmark data → {args.out_dir}", flush=True)
    print(f"Mode: LAYERED (entity0=front y=+0.45, entity1=back y=-0.45)", flush=True)
    print(f"Cameras: {args.n_cameras}, Frames: {args.n_frames}, Seed: {args.seed}", flush=True)

    try:
        import pyvista as pv
        pv.start_xvfb()
    except Exception:
        pass

    n_ok, n_fail = generate_dataset(
        assets_dir  = Path(args.assets_dir),
        out_dir     = Path(args.out_dir),
        n_cameras   = args.n_cameras,
        n_frames    = args.n_frames,
        resolution  = args.resolution,
        modes       = [CrossingMode.LAYERED],
        pair_filter = pair_filter,
        seed        = args.seed,
    )

    print(f"\n[OCC gen done] ok={n_ok}  fail={n_fail}", flush=True)
    print(f"Output: {args.out_dir}/", flush=True)
    print("Next: run scripts/precompute_occ_volume_gt.py to build volume_gt files", flush=True)


if __name__ == '__main__':
    main()
