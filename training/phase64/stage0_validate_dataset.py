"""
training/phase64/stage0_validate_dataset.py
=============================================
Stage 0: Dataset validation.  Must run before any training.

Computes oracle stats from Phase64Dataset:
  - visible mask coverage distribution
  - amodal-visible hidden fraction distribution
  - overlap histogram (20 bins, 0..1)
  - depth-gap histogram
  - GT object count per frame
  - same-depth vs layered breakdown
  - split O/C/R/X counts

Saves:
  outputs/phase64/stage0/dataset_stats.json
  outputs/phase64/stage0/histograms.png

Usage:
  python -m training.phase64.stage0_validate_dataset \\
      --data_root toy/data_objaverse \\
      --n_frames 8 \\
      --spatial_h 32 --spatial_w 32 \\
      --n_samples 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def plot_histograms(stats: dict, out_path: Path) -> None:
    """Save a multi-panel histogram figure to *out_path* (PNG)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[stage0] matplotlib not available — skipping histogram plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Phase 64 — Dataset Validation (Stage 0)", fontsize=14)

    def _plot(ax, hist_data, title, xlabel):
        counts = hist_data["counts"]
        edges  = hist_data["edges"]
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
        ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.85, color="steelblue", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        mean = hist_data.get("mean", float("nan"))
        ax.axvline(mean, color="red", linestyle="--", label=f"mean={mean:.3f}")
        ax.legend(fontsize=8)

    _plot(axes[0, 0], stats["visible_coverage"]["e0"],   "Visible Coverage e0",  "mean coverage")
    _plot(axes[0, 1], stats["visible_coverage"]["e1"],   "Visible Coverage e1",  "mean coverage")
    _plot(axes[0, 2], stats["overlap_hist"],             "Overlap IoU",           "overlap ratio")
    _plot(axes[1, 0], stats["hidden_fractions"]["e0"],   "Hidden Fraction e0",   "fraction")
    _plot(axes[1, 1], stats["hidden_fractions"]["e1"],   "Hidden Fraction e1",   "fraction")
    _plot(axes[1, 2], stats["depth_gap_hist"],           "Depth Gap (std)",      "std(depth)")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"[stage0] Histogram saved → {out_path}")


def print_summary(stats: dict) -> None:
    """Print a human-readable summary to stdout."""
    n = stats["n_samples"]
    print("=" * 60)
    print(f"Phase 64 — Stage 0 Dataset Validation  (n={n})")
    print("=" * 60)

    vc = stats["visible_coverage"]
    print(f"  Visible coverage e0: mean={vc['e0']['mean']:.4f}  std={vc['e0']['std']:.4f}")
    print(f"  Visible coverage e1: mean={vc['e1']['mean']:.4f}  std={vc['e1']['std']:.4f}")

    hf = stats["hidden_fractions"]
    print(f"  Hidden fraction e0:  mean={hf['e0']['mean']:.4f}  std={hf['e0']['std']:.4f}")
    print(f"  Hidden fraction e1:  mean={hf['e1']['mean']:.4f}  std={hf['e1']['std']:.4f}")

    ov = stats["overlap_hist"]
    print(f"  Overlap IoU:         mean={ov['mean']:.4f}  std={ov['std']:.4f}")

    dg = stats["depth_gap_hist"]
    print(f"  Depth gap (std):     mean={dg['mean']:.4f}  std={dg['std']:.4f}")

    oc = stats["object_counts"]
    print(f"  Mean GT objects/frame: {oc['mean_per_frame']:.3f}")
    dist = oc["distribution"]
    for k, v in sorted(dist.items()):
        print(f"    count={k}: {v} frames")

    sc = stats["split_counts"]
    total_sc = max(sum(sc.values()), 1)
    print("  Split breakdown:")
    for t_name, cnt in sc.items():
        print(f"    {t_name}: {cnt}  ({100.0 * cnt / total_sc:.1f}%)")

    print("=" * 60)


def compute_and_save_stats(dataset, splits: dict, config, out_dir: str = "outputs/phase64/stage0") -> dict:
    """Compute dataset oracle stats and save to *out_dir*.

    Called from train_phase64_scene.py before Stage 1 training starts.

    Returns the stats dict.
    """
    from data.phase64.build_scene_gt import compute_dataset_stats

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stats = compute_dataset_stats(dataset, n_samples=None)

    # Append split counts from pre-computed splits dict
    stats["split_counts"] = {
        "O": len(splits.get("split_O", [])),
        "C": len(splits.get("split_C", [])),
        "R": len(splits.get("split_R", [])),
        "X": len(splits.get("split_X", [])),
        "train": len(splits.get("train", [])),
        "val":   len(splits.get("val", [])),
    }

    json_path = out_path / "dataset_stats.json"
    import json
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[stage0] Stats saved → {json_path}")

    hist_path = out_path / "histograms.png"
    plot_histograms(stats, hist_path)
    print_summary(stats)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Phase 64 Stage 0 — Dataset Validation")
    parser.add_argument("--data_root",  default="toy/data_objaverse",
                        help="Path to the objaverse toy dataset root")
    parser.add_argument("--n_frames",   type=int, default=8,
                        help="Frames per clip")
    parser.add_argument("--spatial_h",  type=int, default=32,
                        help="Spatial height for GT masks")
    parser.add_argument("--spatial_w",  type=int, default=32,
                        help="Spatial width for GT masks")
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="Subsample N samples for faster run (None = all)")
    parser.add_argument("--out_dir",    default="outputs/phase64/stage0",
                        help="Output directory for JSON + PNG")
    args = parser.parse_args()

    # ---- add repo root to sys.path ----------------------------------------- #
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # ---- imports after path fix -------------------------------------------- #
    from data.phase64.phase64_dataset import Phase64Dataset
    from data.phase64.build_scene_gt import compute_dataset_stats

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage0] Loading dataset from {args.data_root} ...")
    dataset = Phase64Dataset(
        data_root=args.data_root,
        n_frames=args.n_frames,
        spatial_h=args.spatial_h,
        spatial_w=args.spatial_w,
    )
    print(f"[stage0] Dataset size: {len(dataset)}")

    print("[stage0] Computing statistics ...")
    stats = compute_dataset_stats(dataset, n_samples=args.n_samples)

    # ---- Save JSON --------------------------------------------------------- #
    json_path = out_dir / "dataset_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[stage0] Stats saved → {json_path}")

    # ---- Save histograms --------------------------------------------------- #
    hist_path = out_dir / "histograms.png"
    plot_histograms(stats, hist_path)

    # ---- Print summary ----------------------------------------------------- #
    print_summary(stats)

    return stats


if __name__ == "__main__":
    main()
