"""
data/phase64/phase64_splits.py
================================
O/C/R/X split categorization from blueprint §12.2.

Split definitions:
  O — layered_occlusion   strong front/back depth gap, clear separation
  C — contact_collision   near-depth contact, hardest (overlap > 0.08)
  R — reappearance        entity hidden then returns visible
  X — transfer_stress     new categories/shapes not well-represented in train
"""
from __future__ import annotations

import random
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
#  Split type
# --------------------------------------------------------------------------- #

class SplitType(Enum):
    O = "layered_occlusion"
    C = "contact_collision"
    R = "reappearance"
    X = "transfer_stress"


# --------------------------------------------------------------------------- #
#  X-split keywords — categories that are unusual / rarely seen in training
# --------------------------------------------------------------------------- #

_X_KEYWORDS = {
    "crystal", "gem", "mineral", "rock", "stone",
    "fungus", "mushroom", "plant", "flower", "tree",
    "vehicle", "car", "ship", "plane", "train",
    "building", "house", "tower", "bridge",
    "food", "fruit", "vegetable",
}


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _compute_overlap_ratio(
    entity_masks: np.ndarray,  # (T, 2, S) float32
) -> float:
    """Return max per-frame overlap ratio across all frames."""
    if entity_masks is None or entity_masks.ndim < 3:
        return 0.0
    # entity_masks: (T, 2, S)
    T = entity_masks.shape[0]
    max_ov = 0.0
    for t in range(T):
        m0 = entity_masks[t, 0]  # (S,)
        m1 = entity_masks[t, 1]  # (S,)
        overlap = float((m0 * m1).sum())
        union = float((np.maximum(m0, m1)).sum()) + 1e-8
        ov = overlap / union
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _compute_depth_gap(depth_orders: list) -> float:
    """
    Return fraction of frames with *consistent* front/back ordering.
    A large gap = consistent ordering across frames.
    """
    if not depth_orders:
        return 0.0
    front_counts = [0, 0]
    for order in depth_orders:
        if isinstance(order, (tuple, list)) and len(order) >= 2:
            f = int(order[0])
            if f in (0, 1):
                front_counts[f] += 1
    T = len(depth_orders)
    dominant = max(front_counts) / (T + 1e-8)
    return dominant


def _has_reappearance(
    visible_masks: Optional[np.ndarray],  # (T, 2, S) or None
    hidden_thresh: float = 0.02,
) -> bool:
    """
    Return True if either entity is near-completely hidden for at least one
    contiguous run of frames and then becomes visible again.
    """
    if visible_masks is None:
        return False
    T = entity_count = visible_masks.shape[0]
    for e in range(2):
        vis_seq = []
        for t in range(T):
            v = float(visible_masks[t, e].mean())
            vis_seq.append(v > hidden_thresh)
        # Look for pattern: True... False... True
        found_true = False
        found_hidden = False
        for v in vis_seq:
            if v:
                if found_hidden:
                    return True
                found_true = True
            else:
                if found_true:
                    found_hidden = True
    return False


def _is_x_split(meta: dict) -> bool:
    """Return True if the sample's category keywords suggest X-split."""
    for key in ("keyword0", "keyword1", "category"):
        val = str(meta.get(key, "")).lower()
        for xkw in _X_KEYWORDS:
            if xkw in val:
                return True
    return False


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def categorize_sample(
    frames: np.ndarray,
    entity_masks: np.ndarray,
    depth_orders: list,
    meta: dict,
    visible_masks: Optional[np.ndarray] = None,
    overlap_thresh: float = 0.08,
    depth_gap_thresh: float = 0.80,
) -> SplitType:
    """
    Categorize a sample into O/C/R/X.

    Priority order (highest first):
      1. C — overlap > overlap_thresh  (contact collision is most diagnostic)
      2. R — reappearance pattern detected in visible_masks
      3. X — unusual category keyword
      4. O — strong depth ordering  (default for well-separated entities)
      Falls back to O for everything else.

    Parameters
    ----------
    frames          : (T, H, W, 3) uint8
    entity_masks    : (T, 2, S) float32  amodal masks
    depth_orders    : list of (front_idx, back_idx)
    meta            : sample metadata dict
    visible_masks   : (T, 2, S) float32 or None
    overlap_thresh  : overlap IoU above which a sample is considered C
    depth_gap_thresh: dominant-front fraction above which sample is considered O
    """
    overlap = _compute_overlap_ratio(entity_masks)
    if overlap > overlap_thresh:
        return SplitType.C

    if visible_masks is not None and _has_reappearance(visible_masks):
        return SplitType.R

    if _is_x_split(meta):
        return SplitType.X

    depth_gap = _compute_depth_gap(depth_orders)
    if depth_gap >= depth_gap_thresh:
        return SplitType.O

    # Default: treat as layered occlusion
    return SplitType.O


def make_splits(
    dataset,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Build train/val split and O/C/R/X type indices.

    Parameters
    ----------
    dataset   : Phase64Dataset (or any object with __len__ + __getitem__
                returning Phase64Sample or a dict with the required fields)
    val_frac  : fraction of samples reserved for validation
    seed      : random seed for reproducible splits

    Returns
    -------
    dict with keys:
        train      - train sample indices
        val        - validation sample indices
        split_O    - indices of layered_occlusion samples (across all splits)
        split_C    - indices of contact_collision samples
        split_R    - indices of reappearance samples
        split_X    - indices of transfer_stress samples
    """
    rng = random.Random(seed)
    n = len(dataset)
    all_indices = list(range(n))

    # Categorize each sample
    type_map: Dict[int, SplitType] = {}
    for idx in all_indices:
        sample = dataset[idx]
        # Accept both Phase64Sample and plain dict
        if hasattr(sample, "scene_gt"):
            split_type = sample.scene_gt.split_type
        elif isinstance(sample, dict):
            sg = sample.get("scene_gt")
            if sg is not None and hasattr(sg, "split_type"):
                split_type = sg.split_type
            else:
                # Fallback: categorize on the fly
                entity_masks = sample.get("entity_masks")
                visible_masks = sample.get("visible_masks")
                depth_orders = sample.get("depth_orders", [])
                frames = sample.get("frames")
                meta = sample.get("meta", {})
                split_type = categorize_sample(
                    frames, entity_masks, depth_orders, meta,
                    visible_masks=visible_masks,
                )
        else:
            split_type = SplitType.O
        type_map[idx] = split_type

    # Stratified train/val split — maintain proportions across split types
    train_indices: List[int] = []
    val_indices: List[int] = []

    by_type: Dict[SplitType, List[int]] = {t: [] for t in SplitType}
    for idx, t in type_map.items():
        by_type[t].append(idx)

    for t, idxs in by_type.items():
        shuffled = idxs.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_frac)) if len(shuffled) > 1 else 0
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    # Build per-type index lists (across all train+val)
    split_O = [i for i, t in type_map.items() if t == SplitType.O]
    split_C = [i for i, t in type_map.items() if t == SplitType.C]
    split_R = [i for i, t in type_map.items() if t == SplitType.R]
    split_X = [i for i, t in type_map.items() if t == SplitType.X]

    return {
        "train":   sorted(train_indices),
        "val":     sorted(val_indices),
        "split_O": sorted(split_O),
        "split_C": sorted(split_C),
        "split_R": sorted(split_R),
        "split_X": sorted(split_X),
    }
