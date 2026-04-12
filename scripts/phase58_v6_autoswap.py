"""
Phase58 v6 — Auto Object Detection + Identity Swap

Pipeline:
  Stage 1: Vanilla AnimateDiff → photorealistic two-animal scene
  Detect:  GroundingDINO → find 2 animals in keyframe
  Segment: SAM → precise mask for target animal
  Stage 2: Inpaint target animal with different species prompt

No GT masks needed. Works entirely in generated-image coordinates.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import imageio.v2 as iio2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

SAM_CHECKPOINT = str(Path.home() / ".cache/sam_vit_b.pth")
SAM_MODEL_TYPE = "vit_b"
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


# ─── Detection ───────────────────────────────────────────────────────────

def detect_animals(
    frame: Image.Image,
    text_prompt: str = "cat . dog . animal .",
    threshold: float = 0.25,
    device: str = "cuda",
) -> List[dict]:
    """Detect animals using GroundingDINO."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    proc = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_MODEL_ID).to(device)

    inputs = proc(images=frame, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=threshold,
        target_sizes=[(frame.height, frame.width)])[0]

    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        detections.append({
            "box": box.cpu().tolist(),
            "score": float(score),
            "label": label,
        })

    del model
    torch.cuda.empty_cache()
    return detections


def choose_swap_target(
    detections: List[dict],
    strategy: str = "rightmost",
) -> int:
    """Choose which detection to swap to a different identity.

    Strategies:
      'rightmost': pick the detection with largest x-center
      'leftmost': pick the detection with smallest x-center
      'lowest_score': pick the one with lowest detection confidence
    """
    if len(detections) < 2:
        return 0

    if strategy == "rightmost":
        centers = [(d["box"][0] + d["box"][2]) / 2 for d in detections]
        return int(np.argmax(centers))
    elif strategy == "leftmost":
        centers = [(d["box"][0] + d["box"][2]) / 2 for d in detections]
        return int(np.argmin(centers))
    elif strategy == "lowest_score":
        scores = [d["score"] for d in detections]
        return int(np.argmin(scores))
    return 0


# ─── Segmentation ────────────────────────────────────────────────────────

def segment_from_box(
    frame: np.ndarray,
    box: List[float],
    device: str = "cuda",
) -> np.ndarray:
    """Get precise mask using SAM with box prompt."""
    from segment_anything import sam_model_registry, SamPredictor

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device)
    predictor = SamPredictor(sam)

    predictor.set_image(frame)
    box_np = np.array(box)
    masks, scores, _ = predictor.predict(
        box=box_np, multimask_output=True)
    best = masks[np.argmax(scores)]

    # Dilate mask for better inpainting context at boundaries
    mask_uint8 = best.astype(np.uint8) * 255
    kernel = np.ones((41, 41), np.uint8)
    mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)

    del sam, predictor
    torch.cuda.empty_cache()
    return mask_dilated


def propagate_mask_simple(
    frames: List[np.ndarray],
    keyframe_mask: np.ndarray,
    keyframe_idx: int = 0,
) -> List[np.ndarray]:
    """Simple mask propagation: dilate keyframe mask for temporal coverage.

    For a proper implementation, use SAM2 video tracking.
    This is a placeholder that slightly dilates the keyframe mask
    to cover motion between frames.
    """
    masks = []
    for fi in range(len(frames)):
        dist = abs(fi - keyframe_idx)
        dilation = max(0, dist * 3)
        if dilation > 0:
            kernel = np.ones((dilation * 2 + 1, dilation * 2 + 1), np.uint8)
            m = cv2.dilate(keyframe_mask, kernel, iterations=1)
        else:
            m = keyframe_mask.copy()
        masks.append(m)
    return masks


# ─── Inpainting ──────────────────────────────────────────────────────────

def inpaint_frames(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    prompt: str,
    negative_prompt: str = "blurry, deformed, chimera",
    strength: float = 0.99,
    guidance_scale: float = 9.0,
    n_steps: int = 25,
    seed: int = 42,
    device: str = "cuda",
) -> List[np.ndarray]:
    """Inpaint masked region in each frame."""
    from diffusers import AutoPipelineForInpainting

    pipe = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16).to(device)

    refined = []
    for fi, (frame, mask) in enumerate(zip(frames, masks)):
        img = Image.fromarray(frame)
        mask_img = Image.fromarray(mask)
        h, w = frame.shape[:2]

        if np.array(mask_img).sum() < 100:
            refined.append(frame)
            continue

        gen = torch.Generator(device=device).manual_seed(seed + fi)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img, mask_image=mask_img,
            height=h, width=w,
            num_inference_steps=n_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=gen,
        ).images[0]
        refined.append(np.array(result))
        print(f"  [Inpaint] frame {fi}/{len(frames)}", flush=True)

    del pipe
    torch.cuda.empty_cache()
    return refined


# ─── Main pipeline ───────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Phase58 v6: Auto object detection + identity swap")
    p.add_argument("--output-dir", type=str, default="outputs/phase58_v6")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source-animal", type=str, default="cat",
                   help="Animal in Stage 1 scene (both animals will be this)")
    p.add_argument("--target-animal", type=str, default="golden retriever puppy",
                   help="Animal to swap one instance to")
    p.add_argument("--swap-strategy", type=str, default="rightmost",
                   choices=["rightmost", "leftmost", "lowest_score"])
    p.add_argument("--s2-strength", type=float, default=0.99)
    p.add_argument("--detect-threshold", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[Phase58 v6] Auto Detection + Identity Swap")
    print(f"  Source: {args.source_animal}")
    print(f"  Target: {args.target_animal}")
    print(f"  Strategy: {args.swap_strategy}")
    print("=" * 60)

    # ── Stage 1: Generate base scene ─────────────────────────────────
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

    print("[Stage1] Generating scene...", flush=True)
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3", torch_dtype=torch.float16)
    pipe1 = AnimateDiffPipeline.from_pretrained(
        "emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
    pipe1.scheduler = DDIMScheduler.from_config(
        pipe1.scheduler.config, clip_sample=False,
        timestep_spacing="linspace", beta_schedule="linear", steps_offset=1)
    pipe1.enable_vae_slicing()
    pipe1.to(args.device)

    prompt1 = (
        f"two realistic {args.source_animal}s sitting side by side "
        f"on a cozy carpet in a warm living room, photorealistic, "
        f"detailed fur, natural lighting"
    )
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    out1 = pipe1(
        prompt=prompt1,
        negative_prompt="blurry, deformed, extra limbs, chimera, fused, cartoon",
        num_frames=8, height=512, width=512,
        num_inference_steps=25, guidance_scale=8.0,
        generator=gen, output_type="np")
    base_frames = [(f * 255).astype(np.uint8) for f in out1.frames[0]]
    del pipe1, adapter
    torch.cuda.empty_cache()

    iio2.mimwrite(str(out / "stage1.gif"), base_frames, fps=8, loop=0)
    for fi in [0, 4]:
        Image.fromarray(base_frames[fi]).save(str(out / f"stage1_f{fi}.png"))
    print(f"[Stage1] Done: {len(base_frames)} frames", flush=True)

    # ── Detect animals in keyframe ───────────────────────────────────
    keyframe_idx = 0
    keyframe = base_frames[keyframe_idx]
    print(f"[Detect] Running GroundingDINO on frame {keyframe_idx}...", flush=True)

    detections = detect_animals(
        Image.fromarray(keyframe),
        text_prompt=f"{args.source_animal} . animal .",
        threshold=args.detect_threshold,
        device=args.device)

    for i, d in enumerate(detections):
        print(f"  det {i}: '{d['label']}' score={d['score']:.3f} box={[round(x) for x in d['box']]}")

    if len(detections) < 2:
        print("[WARN] Less than 2 detections! Using right-half fallback.", flush=True)
        mask_kf = np.zeros((512, 512), dtype=np.uint8)
        mask_kf[:, 256:] = 255
    else:
        target_idx = choose_swap_target(detections, args.swap_strategy)
        print(f"[Detect] Swap target: det {target_idx} ('{detections[target_idx]['label']}')", flush=True)

        # Draw detection boxes
        vis = keyframe.copy()
        for i, d in enumerate(detections):
            x0, y0, x1, y1 = [int(c) for c in d["box"]]
            color = (0, 255, 0) if i != target_idx else (255, 0, 0)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            cv2.putText(vis, f"{d['label']} {d['score']:.2f}",
                       (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        Image.fromarray(vis).save(str(out / f"det_boxes_f{keyframe_idx}.png"))

        # Segment target with SAM
        print("[Segment] Running SAM...", flush=True)
        mask_kf = segment_from_box(
            keyframe, detections[target_idx]["box"], device=args.device)
        Image.fromarray(mask_kf).save(str(out / f"mask_target_f{keyframe_idx}.png"))

    # ── Propagate mask to all frames ─────────────────────────────────
    print("[Track] Propagating mask...", flush=True)
    masks = propagate_mask_simple(base_frames, mask_kf, keyframe_idx)
    for fi in [0, 4]:
        Image.fromarray(masks[fi]).save(str(out / f"mask_f{fi}.png"))

    # ── Stage 2: Inpaint target as different animal ──────────────────
    print("[Stage2] Inpainting...", flush=True)
    target_prompt = (
        f"a realistic {args.target_animal}, "
        f"detailed fur texture, natural lighting, photorealistic"
    )
    refined = inpaint_frames(
        base_frames, masks,
        prompt=target_prompt,
        negative_prompt=f"{args.source_animal}, blurry, deformed, chimera",
        strength=args.s2_strength,
        n_steps=20, seed=args.seed,
        device=args.device)

    iio2.mimwrite(str(out / "stage2.gif"), refined, fps=8, loop=0)
    for fi in [0, 4]:
        Image.fromarray(refined[fi]).save(str(out / f"stage2_f{fi}.png"))

    # Side-by-side comparisons
    for fi in range(len(base_frames)):
        compare = np.concatenate([base_frames[fi], refined[fi]], axis=1)
        Image.fromarray(compare).save(str(out / f"compare_f{fi}.png"))

    summary = {
        "seed": args.seed,
        "source_animal": args.source_animal,
        "target_animal": args.target_animal,
        "swap_strategy": args.swap_strategy,
        "n_detections": len(detections),
        "detections": detections,
        "s2_strength": args.s2_strength,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[Phase58 v6] Done. Results: {out}", flush=True)


if __name__ == "__main__":
    main()
