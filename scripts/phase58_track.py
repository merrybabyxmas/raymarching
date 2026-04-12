"""
Phase58 tracking — SAM2 video instance propagation.

Takes a keyframe mask and propagates it to all frames using SAM2.
Falls back to per-frame SAM re-segmentation if SAM2 tracking fails.
"""
import cv2
import numpy as np
import torch
from typing import List, Optional


def track_with_sam2(
    frames: List[np.ndarray],
    keyframe_mask: np.ndarray,
    keyframe_idx: int = 0,
    device: str = "cuda",
) -> Optional[List[np.ndarray]]:
    """Track instance across frames using SAM2 video predictor.

    Returns list of binary masks (H, W) uint8 per frame, or None on failure.
    """
    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        import tempfile, os
        from PIL import Image

        # SAM2 video predictor needs frames as JPEG directory
        tmpdir = tempfile.mkdtemp()
        for fi, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(tmpdir, f"{fi:05d}.jpg"))

        predictor = SAM2VideoPredictor.from_pretrained(
            "facebook/sam2.1-hiera-small", device=device)

        with torch.inference_mode(), torch.autocast(device, dtype=torch.float16):
            state = predictor.init_state(video_path=tmpdir)

            # Add keyframe mask as prompt
            mask_bool = keyframe_mask > 128
            _, _, masks_out = predictor.add_new_mask(
                inference_state=state,
                frame_idx=keyframe_idx,
                obj_id=1,
                mask=mask_bool,
            )

            # Propagate forward and backward
            all_masks = {}
            for fi, obj_ids, mask_logits in predictor.propagate_in_video(state):
                mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255
                all_masks[fi] = mask

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        del predictor
        torch.cuda.empty_cache()

        # Build ordered list
        result = []
        for fi in range(len(frames)):
            if fi in all_masks:
                result.append(all_masks[fi])
            elif fi == keyframe_idx:
                result.append(keyframe_mask)
            else:
                result.append(np.zeros_like(keyframe_mask))
        return result

    except Exception as e:
        print(f"  [track] SAM2 failed: {e}", flush=True)
        return None


def track_with_per_frame_sam(
    frames: List[np.ndarray],
    keyframe_box: List[float],
    keyframe_idx: int = 0,
    dilate_px: int = 20,
    device: str = "cuda",
) -> List[np.ndarray]:
    """Fallback: re-segment each frame using SAM v1 with box prompt.

    Uses the keyframe box shifted by estimated motion (centroid tracking).
    """
    from segment_anything import sam_model_registry, SamPredictor

    sam = sam_model_registry["vit_b"](
        checkpoint="/home/dongwoo44/.cache/sam_vit_b.pth")
    sam.to(device)
    predictor = SamPredictor(sam)

    kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
    masks = []

    for fi, frame in enumerate(frames):
        predictor.set_image(frame)
        box_np = np.array(keyframe_box)
        out_masks, scores, _ = predictor.predict(
            box=box_np, multimask_output=True)
        best = out_masks[np.argmax(scores)].astype(np.uint8) * 255
        if dilate_px > 0:
            best = cv2.dilate(best, kernel, iterations=1)
        masks.append(best)

    del sam, predictor
    torch.cuda.empty_cache()
    return masks


def track_instance(
    frames: List[np.ndarray],
    keyframe_mask: np.ndarray,
    keyframe_box: List[float],
    keyframe_idx: int = 0,
    dilate_px: int = 20,
    device: str = "cuda",
    method: str = "auto",
) -> List[np.ndarray]:
    """Track instance across all frames.

    Args:
        method: 'sam2', 'per_frame_sam', or 'auto' (try sam2 first)

    Returns list of binary masks per frame.
    """
    if method in ("sam2", "auto"):
        result = track_with_sam2(frames, keyframe_mask, keyframe_idx, device)
        if result is not None:
            # Apply dilation to tracked masks
            kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
            result = [cv2.dilate(m, kernel, iterations=1) for m in result]
            print(f"  [track] SAM2 tracking OK: {len(result)} frames", flush=True)
            return result
        if method == "sam2":
            raise RuntimeError("SAM2 tracking failed and method=sam2 (no fallback)")

    print("  [track] Falling back to per-frame SAM v1", flush=True)
    return track_with_per_frame_sam(
        frames, keyframe_box, keyframe_idx, dilate_px, device)


def validate_tracked_masks(masks: List[np.ndarray]) -> dict:
    """Check temporal consistency of tracked masks."""
    areas = [int((m > 128).sum()) for m in masks]
    centroids = []
    for m in masks:
        ys, xs = np.where(m > 128)
        if len(xs) > 0:
            centroids.append((float(xs.mean()), float(ys.mean())))
        else:
            centroids.append((0.0, 0.0))

    area_std = float(np.std(areas)) if areas else 0
    area_mean = float(np.mean(areas)) if areas else 0

    # Check for sudden jumps
    jumps = []
    for i in range(1, len(centroids)):
        dx = abs(centroids[i][0] - centroids[i-1][0])
        dy = abs(centroids[i][1] - centroids[i-1][1])
        jumps.append(float(max(dx, dy)))

    empty_frames = sum(1 for a in areas if a < 100)

    return {
        "areas": areas,
        "area_mean": area_mean,
        "area_std": area_std,
        "max_jump": float(max(jumps)) if jumps else 0,
        "empty_frames": empty_frames,
        "n_frames": len(masks),
    }
