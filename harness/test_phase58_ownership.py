"""
Tests for Phase58 v8 ownership decomposition module.

Validates region decomposition correctness for various overlap scenarios.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure scripts dir is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from phase58_ownership import (
    compute_overlap,
    decompose_regions,
    estimate_front_back,
    build_inpaint_plan,
)


def _make_mask(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Helper: create a binary uint8 mask with a filled rectangle."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


# ─── Test: no overlap ───────────────────────────────────────────────────

def test_no_overlap_gives_empty_overlap_region():
    """When two masks do not overlap, the overlap region must be empty."""
    h, w = 64, 64
    mask_a = _make_mask(h, w, 0, 0, 30, 30)     # top-left box
    mask_b = _make_mask(h, w, 35, 35, 64, 64)    # bottom-right box

    overlap = compute_overlap(mask_a, mask_b)
    assert overlap.sum() == 0, "Overlap should be zero for non-overlapping masks"

    regions = decompose_regions(mask_a, mask_b)
    assert regions["overlap"].sum() == 0
    assert (regions["front_exclusive"] > 0).sum() == (mask_a > 0).sum()
    assert (regions["back_exclusive"] > 0).sum() == (mask_b > 0).sum()


# ─── Test: full overlap ────────────────────────────────────────────────

def test_full_overlap_gives_correct_decomposition():
    """When mask_b is entirely inside mask_a, overlap == mask_b."""
    h, w = 64, 64
    mask_a = _make_mask(h, w, 10, 10, 50, 50)   # large box
    mask_b = _make_mask(h, w, 20, 20, 40, 40)   # small box inside

    overlap = compute_overlap(mask_a, mask_b)
    # Overlap should equal mask_b exactly
    assert np.array_equal(overlap > 0, mask_b > 0), \
        "Overlap should be identical to the contained mask"

    regions = decompose_regions(mask_a, mask_b)
    # Back exclusive should be empty (mask_b is fully inside mask_a)
    assert regions["back_exclusive"].sum() == 0, \
        "Back exclusive should be empty when back is fully contained"
    # Front exclusive = mask_a minus mask_b
    front_excl_expected = ((mask_a > 0) & ~(mask_b > 0)).sum() * 255
    assert regions["front_exclusive"].sum() == front_excl_expected


# ─── Test: regions cover union ──────────────────────────────────────────

def test_front_back_regions_cover_union():
    """front_exclusive + back_exclusive + overlap should equal the union."""
    h, w = 64, 64
    mask_a = _make_mask(h, w, 5, 5, 40, 40)
    mask_b = _make_mask(h, w, 25, 25, 60, 60)

    regions = decompose_regions(mask_a, mask_b)

    # Reconstruct union from regions
    fe = regions["front_exclusive"] > 0
    be = regions["back_exclusive"] > 0
    ov = regions["overlap"] > 0
    reconstructed = fe | be | ov

    # Actual union
    union = (mask_a > 0) | (mask_b > 0)

    assert np.array_equal(reconstructed, union), \
        "Decomposed regions must exactly cover the union of both masks"


# ─── Test: non-negative values ──────────────────────────────────────────

def test_regions_are_non_negative():
    """All region masks should have only non-negative (0 or 255) values."""
    h, w = 64, 64
    mask_a = _make_mask(h, w, 0, 0, 50, 50)
    mask_b = _make_mask(h, w, 20, 10, 64, 55)

    regions = decompose_regions(mask_a, mask_b)

    for name, region in regions.items():
        unique_vals = set(np.unique(region))
        assert unique_vals.issubset({0, 255}), \
            f"Region '{name}' has unexpected values: {unique_vals}"
        assert region.min() >= 0, \
            f"Region '{name}' has negative values"


# ─── Test: inpaint plan ordering ───────────────────────────────────────

def test_inpaint_plan_back_before_front():
    """Inpaint plan should process back entity before front entity."""
    h, w = 64, 64
    mask_a = _make_mask(h, w, 5, 5, 40, 40)
    mask_b = _make_mask(h, w, 25, 25, 60, 60)

    regions = decompose_regions(mask_a, mask_b)
    plan = build_inpaint_plan(regions, "front prompt", "back prompt")

    assert len(plan) >= 1, "Plan should have at least one step"
    orders = [step[2] for step in plan]
    assert orders == sorted(orders), "Plan must be sorted by order (back=0, front=1)"

    if len(plan) >= 2:
        assert plan[0][2] == 0, "First pass should be order 0 (back)"
        assert plan[1][2] == 1, "Second pass should be order 1 (front)"


# ─── Test: estimate_front_back strategies ───────────────────────────────

def test_estimate_front_back_larger_is_front():
    """Larger box should be classified as front."""
    det_big = {"box": [0, 0, 100, 100], "score": 0.8}
    det_small = {"box": [20, 20, 50, 50], "score": 0.9}
    overlap = np.zeros((128, 128), dtype=np.uint8)

    front, back = estimate_front_back(det_big, det_small, overlap, "larger_is_front")
    assert front is det_big
    assert back is det_small


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
