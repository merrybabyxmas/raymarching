"""
Phase58 v8 — Collision-Aware Ownership Decomposition.

Given two instance masks that may overlap, decomposes the scene into
front-exclusive, back-exclusive, and overlap regions. Builds a two-pass
inpainting plan: back-entity first (on overlap + back_exclusive), then
front-entity second (on front_exclusive only, preserving the repainted back).
"""
import numpy as np
from typing import Dict, List, Tuple


# ─── Overlap computation ────────────────────────────────────────────────

def compute_overlap(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> np.ndarray:
    """Compute overlap region between two binary masks.

    Args:
        mask_a: Binary mask (H, W) uint8, values 0 or 255.
        mask_b: Binary mask (H, W) uint8, values 0 or 255.

    Returns:
        Overlap mask (H, W) uint8, values 0 or 255.
    """
    overlap = ((mask_a > 0) & (mask_b > 0)).astype(np.uint8) * 255
    return overlap


# ─── Front/back estimation ──────────────────────────────────────────────

def estimate_front_back(
    det_a: dict,
    det_b: dict,
    overlap: np.ndarray,
    strategy: str = "larger_is_front",
) -> Tuple[dict, dict]:
    """Estimate which detection is in front (occluder) vs. back (occluded).

    Strategies:
        'larger_is_front': Larger bounding box area is in front (closer to camera).
        'higher_score_front': Higher detection score is in front.
        'lower_is_front': Detection with lower y-center (higher on screen) is behind.

    Args:
        det_a: First detection dict with 'box' [x0, y0, x1, y1].
        det_b: Second detection dict.
        overlap: Overlap mask (used for context, not directly in strategies).
        strategy: Front/back estimation strategy.

    Returns:
        (front_det, back_det) tuple.
    """
    box_a = det_a["box"]
    box_b = det_b["box"]

    if strategy == "larger_is_front":
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        if area_a >= area_b:
            return det_a, det_b
        else:
            return det_b, det_a

    elif strategy == "higher_score_front":
        if det_a.get("score", 0) >= det_b.get("score", 0):
            return det_a, det_b
        else:
            return det_b, det_a

    elif strategy == "lower_is_front":
        # Lower y-center = higher on screen = further from camera = behind
        # So higher y-center = lower on screen = closer = front
        cy_a = (box_a[1] + box_a[3]) / 2
        cy_b = (box_b[1] + box_b[3]) / 2
        if cy_a >= cy_b:
            return det_a, det_b
        else:
            return det_b, det_a

    else:
        # Default: larger is front
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        if area_a >= area_b:
            return det_a, det_b
        else:
            return det_b, det_a


# ─── Region decomposition ──────────────────────────────────────────────

def decompose_regions(
    front_mask: np.ndarray,
    back_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Decompose two masks into non-overlapping ownership regions.

    Args:
        front_mask: Binary mask of the front entity (H, W) uint8, 0/255.
        back_mask: Binary mask of the back entity (H, W) uint8, 0/255.

    Returns:
        Dict with keys:
            'front_exclusive': Region only in front mask.
            'back_exclusive': Region only in back mask.
            'overlap': Region in both masks.
            'front_visible': front_exclusive + overlap (everything the front entity owns).
            'back_visible': back_exclusive + overlap (all pixels the back entity touches).
    """
    f = front_mask > 0
    b = back_mask > 0

    overlap = (f & b).astype(np.uint8) * 255
    front_exclusive = (f & ~b).astype(np.uint8) * 255
    back_exclusive = (b & ~f).astype(np.uint8) * 255
    front_visible = f.astype(np.uint8) * 255
    back_visible = b.astype(np.uint8) * 255

    return {
        "front_exclusive": front_exclusive,
        "back_exclusive": back_exclusive,
        "overlap": overlap,
        "front_visible": front_visible,
        "back_visible": back_visible,
    }


# ─── Inpaint plan ──────────────────────────────────────────────────────

def build_inpaint_plan(
    regions: Dict[str, np.ndarray],
    front_prompt: str,
    back_prompt: str,
) -> List[Tuple[np.ndarray, str, int]]:
    """Build ordered inpaint plan for two-pass collision-aware editing.

    Pass 1 (order=0): Inpaint the BACK entity.
        Mask = back_exclusive + overlap (i.e., back_visible).
        This repaints the entire back region including where it was occluded.

    Pass 2 (order=1): Inpaint the FRONT entity.
        Mask = front_exclusive only.
        The overlap region was already repainted by Pass 1 and the front
        entity's visible part just needs its exclusive pixels changed.

    Args:
        regions: Dict from decompose_regions.
        front_prompt: Inpainting prompt for the front entity.
        back_prompt: Inpainting prompt for the back entity.

    Returns:
        List of (mask, prompt, order) tuples sorted by order.
        Each mask is (H, W) uint8, 0/255.
    """
    plan = []

    # Pass 1: Back entity — repaint back_exclusive + overlap
    back_inpaint_mask = regions["back_visible"].copy()
    if back_inpaint_mask.sum() > 0:
        plan.append((back_inpaint_mask, back_prompt, 0))

    # Pass 2: Front entity — repaint front_exclusive only
    front_inpaint_mask = regions["front_exclusive"].copy()
    if front_inpaint_mask.sum() > 0:
        plan.append((front_inpaint_mask, front_prompt, 1))

    # Sort by order
    plan.sort(key=lambda x: x[2])
    return plan
