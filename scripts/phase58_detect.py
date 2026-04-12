"""
Phase58 v8 — Animal Detection with GroundingDINO.

Detects animals in a frame using the transformers GroundingDINO API,
selects two instances via NMS, and chooses swap target.
"""
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from PIL import Image


GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


# ─── Model loading ──────────────────────────────────────────────────────

def load_detector(device: str = "cuda"):
    """Load GroundingDINO model and processor.

    Returns:
        (model, processor) tuple.
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_MODEL_ID
    ).to(device)
    model.eval()
    return model, processor


# ─── Detection ──────────────────────────────────────────────────────────

def detect_animals(
    model,
    processor,
    frame_pil: Image.Image,
    text_prompt: str = "cat . dog . animal .",
    threshold: float = 0.25,
) -> List[Dict]:
    """Detect animals in a single frame.

    Args:
        model: GroundingDINO model.
        processor: GroundingDINO processor.
        frame_pil: PIL Image of the frame.
        text_prompt: Detection text prompt (period-separated labels).
        threshold: Detection confidence threshold.

    Returns:
        List of detection dicts, each with 'box' [x0,y0,x1,y1], 'score', 'label'.
    """
    device = next(model.parameters()).device
    inputs = processor(
        images=frame_pil, text=text_prompt, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        target_sizes=[(frame_pil.height, frame_pil.width)],
    )[0]

    detections = []
    for box, score, label in zip(
        results["boxes"], results["scores"], results["labels"]
    ):
        detections.append({
            "box": box.cpu().tolist(),
            "score": float(score),
            "label": label,
        })

    return detections


# ─── Instance selection ─────────────────────────────────────────────────

def _box_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two boxes [x0, y0, x1, y1]."""
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def select_two_instances(
    detections: List[Dict],
    iou_threshold: float = 0.5,
) -> Tuple[Dict, Dict]:
    """Select two best instances via NMS + score ranking.

    Applies greedy NMS to suppress highly overlapping boxes, then picks
    the top-2 remaining detections by confidence score.

    Args:
        detections: List of detection dicts.
        iou_threshold: IoU threshold for NMS suppression.

    Returns:
        (det_a, det_b) — two detections sorted by score (highest first).

    Raises:
        ValueError: If fewer than 2 detections survive NMS.
    """
    if len(detections) < 2:
        raise ValueError(
            f"Need at least 2 detections, got {len(detections)}"
        )

    # Sort by score descending
    sorted_dets = sorted(detections, key=lambda d: d["score"], reverse=True)

    # Greedy NMS
    keep = []
    for det in sorted_dets:
        suppress = False
        for kept in keep:
            if _box_iou(det["box"], kept["box"]) > iou_threshold:
                suppress = True
                break
        if not suppress:
            keep.append(det)

    if len(keep) < 2:
        raise ValueError(
            f"Only {len(keep)} detection(s) after NMS (need 2). "
            f"Try lowering threshold or using a different prompt."
        )

    return keep[0], keep[1]


# ─── Target selection ───────────────────────────────────────────────────

def choose_target(
    det_a: Dict,
    det_b: Dict,
    strategy: str = "rightmost",
) -> Tuple[Dict, Dict]:
    """Choose which detection to swap (target) and which to keep.

    Args:
        det_a: First detection.
        det_b: Second detection.
        strategy: Selection strategy.
            'rightmost' — pick detection with larger x-center as target.
            'leftmost' — pick detection with smaller x-center as target.
            'lowest_score' — pick detection with lower confidence as target.

    Returns:
        (target_det, keep_det) tuple.
    """
    cx_a = (det_a["box"][0] + det_a["box"][2]) / 2
    cx_b = (det_b["box"][0] + det_b["box"][2]) / 2

    if strategy == "rightmost":
        if cx_a >= cx_b:
            return det_a, det_b
        else:
            return det_b, det_a
    elif strategy == "leftmost":
        if cx_a <= cx_b:
            return det_a, det_b
        else:
            return det_b, det_a
    elif strategy == "lowest_score":
        if det_a["score"] <= det_b["score"]:
            return det_a, det_b
        else:
            return det_b, det_a
    else:
        # Default: rightmost
        if cx_a >= cx_b:
            return det_a, det_b
        else:
            return det_b, det_a
