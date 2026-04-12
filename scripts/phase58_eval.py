"""
Phase58 v8 — Evaluation and Visualization Utilities.

Provides comparison images, detection/mask overlays, and standard output
saving for the v8 collision-aware object editing pipeline.
"""
import json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional, Any


# ─── Comparison images ──────────────────────────────────────────────────

def make_compare_image(
    before: np.ndarray,
    after: np.ndarray,
) -> np.ndarray:
    """Create side-by-side comparison image.

    Args:
        before: Original frame (H, W, 3) uint8.
        after: Edited frame (H, W, 3) uint8.

    Returns:
        Concatenated image (H, 2*W, 3) uint8.
    """
    return np.concatenate([before, after], axis=1)


# ─── Detection visualization ───────────────────────────────────────────

def make_detection_vis(
    frame: np.ndarray,
    detections: List[Dict],
    target_idx: int = -1,
) -> np.ndarray:
    """Draw detection boxes on a frame.

    Args:
        frame: Image (H, W, 3) uint8.
        detections: List of detection dicts with 'box', 'score', 'label'.
        target_idx: Index of the swap target detection (drawn in red).

    Returns:
        Annotated frame (H, W, 3) uint8.
    """
    vis = frame.copy()
    for i, det in enumerate(detections):
        x0, y0, x1, y1 = [int(c) for c in det["box"]]
        if i == target_idx:
            color = (255, 0, 0)  # Red for target
            label_prefix = "[TARGET] "
        else:
            color = (0, 255, 0)  # Green for keep
            label_prefix = "[KEEP] "

        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        text = f"{label_prefix}{det['label']} {det['score']:.2f}"
        cv2.putText(
            vis, text, (x0, max(y0 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return vis


# ─── Mask overlay ───────────────────────────────────────────────────────

def make_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 0, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a binary mask on a frame with transparency.

    Args:
        frame: Image (H, W, 3) uint8.
        mask: Binary mask (H, W) uint8, 0/255.
        color: RGB color for the overlay.
        alpha: Transparency (0=invisible, 1=opaque).

    Returns:
        Frame with mask overlay (H, W, 3) uint8.
    """
    vis = frame.copy().astype(np.float32)
    mask_bool = mask > 0

    overlay = np.zeros_like(vis)
    overlay[mask_bool] = color

    vis[mask_bool] = (1 - alpha) * vis[mask_bool] + alpha * overlay[mask_bool]
    return vis.astype(np.uint8)


# ─── Standard output saving ────────────────────────────────────────────

def save_standard_outputs(
    out_dir: str,
    stage1_frames: List[np.ndarray],
    stage2_frames: List[np.ndarray],
    dets: List[Dict],
    masks: Dict[str, np.ndarray],
    regions: Dict[str, np.ndarray],
    summary: Dict[str, Any],
) -> None:
    """Save all pipeline artifacts to a standardized output directory.

    Directory structure:
        out_dir/
            stage1/          - Stage 1 generated frames
            stage2/          - Stage 2 inpainted frames
            compare/         - Side-by-side comparisons
            masks/           - Instance masks and region decomposition
            detections.png   - Detection visualization
            summary.json     - Run summary

    Args:
        out_dir: Output directory path.
        stage1_frames: List of Stage 1 frames.
        stage2_frames: List of Stage 2 frames.
        dets: Detection results.
        masks: Dict with mask arrays (e.g. 'target', 'keep', 'overlap').
        regions: Dict from decompose_regions.
        summary: Summary dict to save as JSON.
    """
    import imageio.v2 as iio2

    out = Path(out_dir)
    (out / "stage1").mkdir(parents=True, exist_ok=True)
    (out / "stage2").mkdir(parents=True, exist_ok=True)
    (out / "compare").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    # Save Stage 1 frames + GIF
    for fi, frame in enumerate(stage1_frames):
        Image.fromarray(frame).save(str(out / "stage1" / f"f{fi:03d}.png"))
    iio2.mimwrite(str(out / "stage1.gif"), stage1_frames, fps=8, loop=0)

    # Save Stage 2 frames + GIF
    for fi, frame in enumerate(stage2_frames):
        Image.fromarray(frame).save(str(out / "stage2" / f"f{fi:03d}.png"))
    if stage2_frames:
        iio2.mimwrite(str(out / "stage2.gif"), stage2_frames, fps=8, loop=0)

    # Save comparisons
    for fi in range(min(len(stage1_frames), len(stage2_frames))):
        compare = make_compare_image(stage1_frames[fi], stage2_frames[fi])
        Image.fromarray(compare).save(str(out / "compare" / f"f{fi:03d}.png"))

    # Save masks
    for name, mask in masks.items():
        if mask is not None:
            Image.fromarray(mask).save(str(out / "masks" / f"{name}.png"))

    # Save region decomposition
    for name, region in regions.items():
        if region is not None:
            Image.fromarray(region).save(str(out / "masks" / f"region_{name}.png"))

    # Save summary JSON
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[Eval] Saved outputs to {out}", flush=True)


# ─── Summary builder ───────────────────────────────────────────────────

def build_summary(
    config: Dict[str, Any],
    dets: List[Dict],
    regions: Dict[str, np.ndarray],
    target_idx: int = -1,
    keep_idx: int = -1,
) -> Dict[str, Any]:
    """Build a summary dict for the pipeline run.

    Args:
        config: Pipeline configuration dict.
        dets: List of detection dicts.
        regions: Dict from decompose_regions.
        target_idx: Index of swap target in detections.
        keep_idx: Index of kept detection.

    Returns:
        Summary dict with config, detection info, region stats.
    """
    summary = {
        "config": config,
        "n_detections": len(dets),
        "detections": [],
        "target_idx": target_idx,
        "keep_idx": keep_idx,
        "region_stats": {},
    }

    for i, det in enumerate(dets):
        summary["detections"].append({
            "idx": i,
            "label": det.get("label", ""),
            "score": det.get("score", 0),
            "box": det.get("box", []),
            "role": "target" if i == target_idx else ("keep" if i == keep_idx else "other"),
        })

    # Region pixel counts
    total_pixels = 1
    for name, region in regions.items():
        if region is not None:
            n_pixels = int((region > 0).sum())
            if total_pixels == 1 and region.size > 0:
                total_pixels = region.size
            summary["region_stats"][name] = {
                "pixels": n_pixels,
                "ratio": round(n_pixels / max(total_pixels, 1), 4),
            }

    # Overlap ratio relative to union
    if "overlap" in regions and "front_visible" in regions and "back_visible" in regions:
        overlap_px = int((regions["overlap"] > 0).sum())
        union_px = int(((regions["front_visible"] > 0) | (regions["back_visible"] > 0)).sum())
        summary["region_stats"]["overlap_over_union"] = round(
            overlap_px / max(union_px, 1), 4
        )

    return summary
