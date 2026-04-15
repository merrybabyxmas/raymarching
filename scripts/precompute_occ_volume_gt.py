"""
Phase 62 — OCC Volume GT precomputation
=========================================

Builds volume_gt_rendered_k8_16x16.npy for all scenes in the occ dataset.
Mirrors the logic in precompute_phase62_density.py but targets
toy/data_objaverse_occ/ specifically.

Usage:
    python scripts/precompute_occ_volume_gt.py \
        --data-dir toy/data_objaverse_occ \
        [--depth-bins 8] [--spatial 16] [--render-res 128]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.phase62.volume_gt_builder import VolumeGTBuilder


def load_depth_and_masks(scene_dir: Path, n_frames: int):
    """Load depth maps and entity masks for all frames."""
    depth_maps = []
    masks_e0 = []
    masks_e1 = []

    for fi in range(n_frames):
        d_path = scene_dir / 'depth' / f'{fi:04d}.npy'
        m0_path = scene_dir / 'mask' / f'{fi:04d}_entity0.png'
        m1_path = scene_dir / 'mask' / f'{fi:04d}_entity1.png'

        if not d_path.exists():
            return None, None, None

        depth_maps.append(np.load(str(d_path)))

        try:
            from PIL import Image
            m0 = np.array(Image.open(str(m0_path)).convert('L')) > 128
            m1 = np.array(Image.open(str(m1_path)).convert('L')) > 128
        except Exception:
            import imageio.v3 as iio
            m0 = iio.imread(str(m0_path)) > 128
            m1 = iio.imread(str(m1_path)) > 128

        masks_e0.append(m0.flatten().astype(np.uint8))
        masks_e1.append(m1.flatten().astype(np.uint8))

    return depth_maps, masks_e0, masks_e1


def infer_depth_order(depth_maps, masks_e0, masks_e1):
    """
    Determine depth ordering: which entity is in front (smaller depth = closer).
    For LAYERED mode, entity0 is always in front (y=+0.45 closer to camera).
    Verify by comparing mean depths at mask pixels.
    """
    all_e0_depths = []
    all_e1_depths = []

    for fi, (depth, m0, m1) in enumerate(zip(depth_maps, masks_e0, masks_e1)):
        H, W = depth.shape
        m0_2d = m0.reshape(H, W).astype(bool)
        m1_2d = m1.reshape(H, W).astype(bool)
        if m0_2d.any():
            all_e0_depths.append(depth[m0_2d].mean())
        if m1_2d.any():
            all_e1_depths.append(depth[m1_2d].mean())

    if not all_e0_depths or not all_e1_depths:
        return (0, 1)  # default: entity0 front

    mean_e0 = np.mean(all_e0_depths)
    mean_e1 = np.mean(all_e1_depths)

    # Smaller depth = closer to camera = front
    if mean_e0 <= mean_e1:
        return (0, 1)  # entity0 front, entity1 back
    else:
        return (1, 0)  # entity1 front, entity0 back


def process_scene(scene_dir: Path, builder: VolumeGTBuilder) -> bool:
    """Build and save volume_gt for one scene directory."""
    meta_path = scene_dir / 'meta.json'
    if not meta_path.exists():
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    n_frames = meta.get('n_frames', 16)

    depth_maps, masks_e0, masks_e1 = load_depth_and_masks(scene_dir, n_frames)
    if depth_maps is None:
        print(f"  [skip] missing depth/mask: {scene_dir.name}", flush=True)
        return False

    depth_order = infer_depth_order(depth_maps, masks_e0, masks_e1)

    # Build volume GT for each frame
    vol_gts = []
    for fi in range(n_frames):
        H, W = depth_maps[fi].shape
        entity_masks = np.stack([masks_e0[fi], masks_e1[fi]], axis=0)  # (2, H*W)
        vgt = builder.build(
            depth_map    = depth_maps[fi],
            entity_masks = entity_masks,
            depth_order  = depth_order,
        )
        vol_gts.append(vgt)

    vol_gt_batch = np.stack(vol_gts, axis=0)  # (n_frames, K, H_out, W_out)

    # Save with consistent naming convention
    K = builder.K
    H_out = builder.H_out
    out_path = scene_dir / f'volume_gt_rendered_k{K}_{H_out}x{H_out}.npy'
    np.save(str(out_path), vol_gt_batch)

    depth_tag = "occ_layered" if meta.get('mode') == 'layered' else meta.get('mode', '?')
    front_idx, back_idx = depth_order
    print(f"  [ok] {scene_dir.name}  mode={depth_tag}  front=e{front_idx}  shape={vol_gt_batch.shape}", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',    default='toy/data_objaverse_occ', dest='data_dir')
    p.add_argument('--depth-bins',  type=int, default=8,               dest='depth_bins')
    p.add_argument('--spatial',     type=int, default=16)
    p.add_argument('--render-res',  type=int, default=128,             dest='render_res')
    p.add_argument('--overwrite',   action='store_true')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist. Run generate_occ_dataset.py first.", flush=True)
        sys.exit(1)

    builder = VolumeGTBuilder(
        depth_bins        = args.depth_bins,
        spatial_h         = args.spatial,
        spatial_w         = args.spatial,
        render_resolution = args.render_res,
    )

    scenes = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"[occ vol_gt] Processing {len(scenes)} scenes in {data_dir}", flush=True)

    n_ok = n_skip = n_fail = 0
    for scene_dir in scenes:
        out_path = scene_dir / f'volume_gt_rendered_k{args.depth_bins}_{args.spatial}x{args.spatial}.npy'
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue
        ok = process_scene(scene_dir, builder)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n[done] ok={n_ok}  skip={n_skip}  fail={n_fail}", flush=True)


if __name__ == '__main__':
    main()
