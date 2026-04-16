"""
data/phase64/build_scene_gt.py
================================
Builds SceneGT annotations for Phase 64 from existing Phase62DatasetAdapter samples.

SceneGT is the canonical per-sample annotation that Stage 1 trains against.
It is spatial-resolution-independent (resized to spatial_h x spatial_w) and
stores both visible and amodal masks, depth, overlap, split type, and
hidden-fraction statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.phase64.phase64_splits import SplitType, categorize_sample


# --------------------------------------------------------------------------- #
#  SceneGT dataclass
# --------------------------------------------------------------------------- #

@dataclass
class SceneGT:
    """Per-frame scene annotations at (spatial_h, spatial_w) resolution.

    Fields
    ------
    vis_e0          : (T, H, W)  visible mask entity 0
    vis_e1          : (T, H, W)  visible mask entity 1
    amo_e0          : (T, H, W)  amodal mask entity 0  (present whether visible or not)
    amo_e1          : (T, H, W)  amodal mask entity 1
    depth_map       : (T, H, W)  per-frame normalized depth in [0, 1]
    overlap_ratio   : float       max overlap ratio across all frames
    split_type      : SplitType   O / C / R / X category
    hidden_fraction_e0 : float    fraction of amodal e0 that is hidden  (mean across T)
    hidden_fraction_e1 : float    fraction of amodal e1 that is hidden
    """
    vis_e0: np.ndarray            # (T, H, W) float32
    vis_e1: np.ndarray            # (T, H, W) float32
    amo_e0: np.ndarray            # (T, H, W) float32
    amo_e1: np.ndarray            # (T, H, W) float32
    depth_map: np.ndarray         # (T, H, W) float32
    overlap_ratio: float
    split_type: SplitType
    hidden_fraction_e0: float
    hidden_fraction_e1: float


# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #

def _resize_mask_sequence(
    masks_flat: np.ndarray,       # (T, S) float32, S = h*w
    src_hw: int,                  # assumed square source side
    dst_h: int,
    dst_w: int,
) -> np.ndarray:
    """Resize a (T, S) flat mask sequence to (T, dst_h, dst_w) float32."""
    from PIL import Image

    T = masks_flat.shape[0]
    out = np.zeros((T, dst_h, dst_w), dtype=np.float32)
    for t in range(T):
        flat = masks_flat[t]
        img = Image.fromarray(
            (flat.reshape(src_hw, src_hw) * 255).astype(np.uint8)
        ).resize((dst_w, dst_h), Image.BILINEAR)
        out[t] = np.array(img, dtype=np.float32) / 255.0
    return out


def _resize_depth_sequence(
    depth: np.ndarray,  # (T, H_src, W_src) float32  OR  (T, S) flat
    dst_h: int,
    dst_w: int,
) -> np.ndarray:
    """Resize a depth sequence to (T, dst_h, dst_w), normalised to [0, 1]."""
    from PIL import Image

    T = depth.shape[0]
    # Handle flat format
    if depth.ndim == 2:
        S = depth.shape[1]
        src_hw = int(S ** 0.5)
        depth = depth.reshape(T, src_hw, src_hw)

    out = np.zeros((T, dst_h, dst_w), dtype=np.float32)
    for t in range(T):
        d = depth[t].astype(np.float32)
        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            d = (d - d_min) / (d_max - d_min)
        img = Image.fromarray((d * 255).astype(np.uint8)).resize(
            (dst_w, dst_h), Image.BILINEAR
        )
        out[t] = np.array(img, dtype=np.float32) / 255.0
    return out


def _compute_overlap_ratio(amo_e0: np.ndarray, amo_e1: np.ndarray) -> float:
    """Max IoU overlap across frames from (T, H, W) amodal masks."""
    T = amo_e0.shape[0]
    max_ov = 0.0
    for t in range(T):
        a0 = amo_e0[t].ravel()
        a1 = amo_e1[t].ravel()
        inter = float((a0 * a1).sum())
        union = float(np.maximum(a0, a1).sum()) + 1e-8
        ov = inter / union
        if ov > max_ov:
            max_ov = ov
    return float(max_ov)


def _compute_hidden_fraction(
    amo: np.ndarray,  # (T, H, W)
    vis: np.ndarray,  # (T, H, W)
) -> float:
    """Mean fraction of amodal area that is occluded (hidden)."""
    hidden = np.maximum(amo - vis, 0.0)
    fracs = []
    for t in range(amo.shape[0]):
        amo_sum = float(amo[t].sum()) + 1e-8
        fracs.append(float(hidden[t].sum()) / amo_sum)
    return float(np.mean(fracs))


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def build_scene_gt(
    sample: dict,
    spatial_h: int = 32,
    spatial_w: int = 32,
) -> SceneGT:
    """
    Build a SceneGT from a Phase62DatasetAdapter sample dict.

    Expected sample keys (as returned by Phase62DatasetAdapter.__getitem__):
        frames          (T, H, W, 3) uint8
        depth           (T, H, W) float32  [optional — uses zeros if absent]
        depth_orders    list of (front_idx, back_idx)
        meta            dict
        entity_masks    (T, 2, S) float32  amodal masks
        visible_masks   (T, 2, S) float32  or None

    Parameters
    ----------
    sample      : dict from Phase62DatasetAdapter
    spatial_h   : output spatial height for all mask arrays
    spatial_w   : output spatial width for all mask arrays

    Returns
    -------
    SceneGT
    """
    entity_masks: np.ndarray = sample["entity_masks"]   # (T, 2, S)
    visible_masks: Optional[np.ndarray] = sample.get("visible_masks")
    depth_raw: Optional[np.ndarray] = sample.get("depth")
    depth_orders: list = sample.get("depth_orders", [])
    frames: np.ndarray = sample["frames"]               # (T, H, W, 3)
    meta: dict = sample.get("meta", {})

    T = entity_masks.shape[0]
    S = entity_masks.shape[2]
    src_hw = int(round(S ** 0.5))

    # ---- Amodal masks: entity_masks treated as amodal ground truth ----------
    amo_e0_raw = entity_masks[:, 0, :]   # (T, S)
    amo_e1_raw = entity_masks[:, 1, :]   # (T, S)

    amo_e0 = _resize_mask_sequence(amo_e0_raw, src_hw, spatial_h, spatial_w)
    amo_e1 = _resize_mask_sequence(amo_e1_raw, src_hw, spatial_h, spatial_w)

    # ---- Visible masks -------------------------------------------------------
    if visible_masks is not None:
        vis_e0_raw = visible_masks[:, 0, :]  # (T, S)
        vis_e1_raw = visible_masks[:, 1, :]  # (T, S)
        vis_e0 = _resize_mask_sequence(vis_e0_raw, src_hw, spatial_h, spatial_w)
        vis_e1 = _resize_mask_sequence(vis_e1_raw, src_hw, spatial_h, spatial_w)
    else:
        # Fallback: estimate visible from depth_orders (front entity is fully visible,
        # back entity is visible minus overlap).
        vis_e0 = np.zeros_like(amo_e0)
        vis_e1 = np.zeros_like(amo_e1)
        for t in range(T):
            order = depth_orders[t] if t < len(depth_orders) else (0, 1)
            front = int(order[0]) if isinstance(order, (tuple, list)) else 0
            back = 1 - front
            if front == 0:
                vis_e0[t] = amo_e0[t]
                vis_e1[t] = np.maximum(amo_e1[t] - amo_e0[t], 0.0)
            else:
                vis_e1[t] = amo_e1[t]
                vis_e0[t] = np.maximum(amo_e0[t] - amo_e1[t], 0.0)

    # ---- Depth map -----------------------------------------------------------
    if depth_raw is not None and depth_raw.ndim >= 2:
        if depth_raw.ndim == 3:
            depth_map = _resize_depth_sequence(depth_raw, spatial_h, spatial_w)
        else:
            # (T, S) flat
            depth_map = _resize_depth_sequence(depth_raw, spatial_h, spatial_w)
    else:
        # Build a synthetic depth from depth_orders: front entity = closer (lower value)
        depth_map = np.zeros((T, spatial_h, spatial_w), dtype=np.float32)
        for t in range(T):
            order = depth_orders[t] if t < len(depth_orders) else (0, 1)
            front = int(order[0]) if isinstance(order, (tuple, list)) else 0
            back = 1 - front
            front_mask = amo_e0[t] if front == 0 else amo_e1[t]
            back_mask  = amo_e1[t] if front == 0 else amo_e0[t]
            d = np.zeros((spatial_h, spatial_w), dtype=np.float32)
            d += front_mask * 0.25
            d += back_mask  * 0.75
            depth_map[t] = d

    # ---- Overlap + split type ------------------------------------------------
    overlap_ratio = _compute_overlap_ratio(amo_e0, amo_e1)

    split_type = categorize_sample(
        frames=frames,
        entity_masks=entity_masks,
        depth_orders=depth_orders,
        meta=meta,
        visible_masks=visible_masks,
    )

    # ---- Hidden fractions ----------------------------------------------------
    hidden_fraction_e0 = _compute_hidden_fraction(amo_e0, vis_e0)
    hidden_fraction_e1 = _compute_hidden_fraction(amo_e1, vis_e1)

    return SceneGT(
        vis_e0=vis_e0.astype(np.float32),
        vis_e1=vis_e1.astype(np.float32),
        amo_e0=amo_e0.astype(np.float32),
        amo_e1=amo_e1.astype(np.float32),
        depth_map=depth_map.astype(np.float32),
        overlap_ratio=overlap_ratio,
        split_type=split_type,
        hidden_fraction_e0=hidden_fraction_e0,
        hidden_fraction_e1=hidden_fraction_e1,
    )


def compute_dataset_stats(
    dataset,
    n_samples: Optional[int] = None,
) -> dict:
    """
    Stage 0 dataset statistics.  Iterates over samples and aggregates:
        - visible coverage distribution  (mean coverage per entity)
        - hidden fraction distribution
        - overlap histogram (20 bins, 0..1)
        - depth gap histogram (20 bins, 0..1)
        - GT object count per frame
        - split O/C/R/X breakdown

    Parameters
    ----------
    dataset   : Phase64Dataset
    n_samples : if not None, subsample this many indices uniformly

    Returns
    -------
    dict with keys: visible_coverage, hidden_fractions, overlap_hist,
                    depth_gap_hist, object_counts, split_counts, n_samples
    """
    import math

    n_total = len(dataset)
    indices = list(range(n_total))
    if n_samples is not None and n_samples < n_total:
        step = max(1, n_total // n_samples)
        indices = indices[::step][:n_samples]

    visible_cov_e0, visible_cov_e1 = [], []
    hidden_frac_e0, hidden_frac_e1 = [], []
    overlap_values = []
    depth_gap_values = []
    object_counts_per_frame = []
    split_counter = {t: 0 for t in SplitType}

    N_BINS = 20

    for idx in indices:
        sample = dataset[idx]
        # Support Phase64Sample dataclass or plain dict
        if hasattr(sample, "scene_gt"):
            gt = sample.scene_gt
            frames = sample.frames
        else:
            from data.phase64.build_scene_gt import build_scene_gt as _bsg
            gt = _bsg(sample)
            frames = sample.get("frames")

        T = gt.amo_e0.shape[0]

        # Visible coverage (mean over spatial dims and frames)
        vc0 = float(gt.vis_e0.mean())
        vc1 = float(gt.vis_e1.mean())
        visible_cov_e0.append(vc0)
        visible_cov_e1.append(vc1)

        hidden_frac_e0.append(gt.hidden_fraction_e0)
        hidden_frac_e1.append(gt.hidden_fraction_e1)
        overlap_values.append(gt.overlap_ratio)

        # Depth gap: compute from depth_map variance as proxy
        dmap = gt.depth_map  # (T, H, W)
        gap = float(np.std(dmap))
        depth_gap_values.append(gap)

        # Object count: for each frame, count entities with mean vis > threshold
        thresh = 0.01
        for t in range(T):
            cnt = int(gt.vis_e0[t].mean() > thresh) + int(gt.vis_e1[t].mean() > thresh)
            object_counts_per_frame.append(cnt)

        split_counter[gt.split_type] += 1

    def _hist(values, n_bins, lo=0.0, hi=1.0):
        counts, edges = np.histogram(values, bins=n_bins, range=(lo, hi))
        return {
            "counts": counts.tolist(),
            "edges": edges.tolist(),
            "mean": float(np.mean(values)) if values else 0.0,
            "std":  float(np.std(values))  if values else 0.0,
        }

    return {
        "n_samples": len(indices),
        "visible_coverage": {
            "e0": _hist(visible_cov_e0, N_BINS),
            "e1": _hist(visible_cov_e1, N_BINS),
        },
        "hidden_fractions": {
            "e0": _hist(hidden_frac_e0, N_BINS),
            "e1": _hist(hidden_frac_e1, N_BINS),
        },
        "overlap_hist": _hist(overlap_values, N_BINS),
        "depth_gap_hist": _hist(depth_gap_values, N_BINS, lo=0.0, hi=0.5),
        "object_counts": {
            "mean_per_frame": float(np.mean(object_counts_per_frame)) if object_counts_per_frame else 0.0,
            "distribution": {
                str(k): int(sum(1 for c in object_counts_per_frame if c == k))
                for k in range(4)
            },
        },
        "split_counts": {t.name: split_counter[t] for t in SplitType},
    }
