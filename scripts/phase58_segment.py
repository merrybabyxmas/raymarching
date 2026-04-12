"""
Phase58 v8 — SAM Segmentation.

Provides precise instance masks using SAM v1 with box prompts from
GroundingDINO detections. Includes mask refinement (dilation, cleanup).
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional


SAM_CHECKPOINT = str(Path.home() / ".cache/sam_vit_b.pth")
SAM_MODEL_TYPE = "vit_b"


# ─── Model loading ──────────────────────────────────────────────────────

def load_sam(
    device: str = "cuda",
    checkpoint: str = SAM_CHECKPOINT,
) -> "SamPredictor":
    """Load SAM model and return a SamPredictor.

    Args:
        device: Torch device.
        checkpoint: Path to SAM weights (vit_b).

    Returns:
        SamPredictor instance.
    """
    from segment_anything import sam_model_registry, SamPredictor

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


# ─── Instance segmentation ─────────────────────────────────────────────

def segment_instance(
    predictor: "SamPredictor",
    frame_np: np.ndarray,
    box: list,
) -> np.ndarray:
    """Segment a single instance using SAM with a box prompt.

    Args:
        predictor: SamPredictor (already loaded).
        frame_np: Frame as numpy uint8 array (H, W, 3) in RGB.
        box: Bounding box [x0, y0, x1, y1] in pixel coordinates.

    Returns:
        Binary mask as uint8 array (H, W), values 0 or 255.
    """
    predictor.set_image(frame_np)
    box_np = np.array(box)
    masks, scores, _ = predictor.predict(
        box=box_np,
        multimask_output=True,
    )
    # Pick the mask with highest predicted IoU
    best_idx = np.argmax(scores)
    mask = masks[best_idx].astype(np.uint8) * 255
    return mask


# ─── Mask refinement ────────────────────────────────────────────────────

def refine_mask(
    mask: np.ndarray,
    dilate_px: int = 20,
    min_area: int = 500,
) -> np.ndarray:
    """Refine a binary mask: dilate for inpainting boundary, remove small blobs.

    Args:
        mask: Binary mask (H, W) uint8, values 0 or 255.
        dilate_px: Dilation kernel radius in pixels.
        min_area: Minimum connected component area to keep.

    Returns:
        Refined binary mask (H, W) uint8, values 0 or 255.
    """
    refined = mask.copy()

    # Remove small connected components
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            refined, connectivity=8
        )
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                refined[labels == i] = 0

    # Dilate for better inpainting boundary coverage
    if dilate_px > 0:
        kernel_size = dilate_px * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        refined = cv2.dilate(refined, kernel, iterations=1)

    return refined
