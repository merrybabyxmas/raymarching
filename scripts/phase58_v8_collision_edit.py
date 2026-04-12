"""
Phase58 v8 — Collision-Aware Object Editing Pipeline.

Full orchestrator for two-animal scene generation, detection, segmentation,
ownership decomposition, and two-pass identity-swap inpainting.

Fixes the v6 failure on collision/overlap scenes by:
  1. Decomposing overlapping masks into front/back/overlap regions.
  2. Inpainting the back entity first (including overlap).
  3. Inpainting the front entity second (exclusive region only).
  4. Better Stage 1 prompts for diverse animal interaction scenarios.

Usage:
    python scripts/phase58_v8_collision_edit.py \\
        --source-animal cat --target-animal "golden retriever puppy" \\
        --scenario tangled --seed 42
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from phase58_stage1 import load_pipeline as load_stage1, build_prompt as build_stage1_prompt, generate
from phase58_detect import load_detector, detect_animals, select_two_instances, choose_target
from phase58_segment import load_sam, segment_instance, refine_mask
from phase58_ownership import compute_overlap, estimate_front_back, decompose_regions, build_inpaint_plan
from phase58_inpaint import load_inpaint_pipeline, build_prompt as build_inpaint_prompt, compute_strength, run_two_pass
from phase58_eval import (
    make_compare_image, make_detection_vis, make_mask_overlay,
    save_standard_outputs, build_summary,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase58 v8: Collision-aware object editing pipeline",
    )
    # Animals
    p.add_argument("--source-animal", type=str, default="cat",
                   help="Animal species for Stage 1 generation (both instances)")
    p.add_argument("--target-animal", type=str, default="golden retriever puppy",
                   help="Species to swap the target instance to")
    p.add_argument("--keep-animal", type=str, default="",
                   help="Species to repaint the kept instance as (empty = keep original)")

    # Scene
    p.add_argument("--scenario", type=str, default="tangled",
                   choices=["side_by_side", "tangled", "climbing", "playing"],
                   help="Scene scenario for Stage 1 prompt")
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    # Detection
    p.add_argument("--detect-threshold", type=float, default=0.25)
    p.add_argument("--detect-prompt", type=str, default="",
                   help="Custom detection prompt (default: auto from source-animal)")
    p.add_argument("--swap-strategy", type=str, default="rightmost",
                   choices=["rightmost", "leftmost", "lowest_score"])

    # Ownership
    p.add_argument("--front-back-strategy", type=str, default="larger_is_front",
                   choices=["larger_is_front", "higher_score_front", "lower_is_front"])

    # Inpainting
    p.add_argument("--s2-strength", type=float, default=0.0,
                   help="Override inpainting strength (0 = auto)")
    p.add_argument("--s2-guidance", type=float, default=9.0)
    p.add_argument("--s2-steps", type=int, default=25)
    p.add_argument("--dilate-px", type=int, default=20,
                   help="Mask dilation in pixels for inpainting boundary")

    # General
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="outputs/phase58_v8")
    p.add_argument("--keyframe", type=int, default=0,
                   help="Keyframe index for detection/segmentation")
    p.add_argument("--skip-stage1", action="store_true",
                   help="Skip Stage 1, load frames from output-dir/stage1/")

    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    print("=" * 60)
    print("[Phase58 v8] Collision-Aware Object Editing")
    print(f"  Source:   {args.source_animal}")
    print(f"  Target:   {args.target_animal}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Seed:     {args.seed}")
    print("=" * 60)

    # ── Stage 1: Generate base scene ─────────────────────────────────
    if args.skip_stage1:
        print("[Stage1] Loading existing frames...", flush=True)
        stage1_dir = out / "stage1"
        frame_paths = sorted(stage1_dir.glob("f*.png"))
        if not frame_paths:
            print("[ERROR] No Stage 1 frames found. Run without --skip-stage1.", flush=True)
            return
        base_frames = [np.array(Image.open(p)) for p in frame_paths]
        print(f"[Stage1] Loaded {len(base_frames)} frames", flush=True)
    else:
        print("[Stage1] Generating scene...", flush=True)
        pipe1 = load_stage1(args.device)
        prompt1, neg1 = build_stage1_prompt(
            args.source_animal, args.target_animal, args.scenario)
        base_frames = generate(
            pipe1, prompt1, neg1,
            n_frames=args.n_frames,
            height=args.height, width=args.width,
            steps=25, guidance=8.0, seed=args.seed,
        )
        del pipe1
        torch.cuda.empty_cache()
        print(f"[Stage1] Generated {len(base_frames)} frames", flush=True)

    # ── Detection ────────────────────────────────────────────────────
    keyframe = base_frames[args.keyframe]
    keyframe_pil = Image.fromarray(keyframe)

    print(f"[Detect] Running GroundingDINO on frame {args.keyframe}...", flush=True)
    det_model, det_proc = load_detector(args.device)

    text_prompt = args.detect_prompt or f"{args.source_animal} . animal ."
    all_dets = detect_animals(
        det_model, det_proc, keyframe_pil,
        text_prompt=text_prompt,
        threshold=args.detect_threshold,
    )
    del det_model
    torch.cuda.empty_cache()

    for i, d in enumerate(all_dets):
        print(f"  det {i}: '{d['label']}' score={d['score']:.3f} "
              f"box={[round(x) for x in d['box']]}")

    if len(all_dets) < 2:
        print("[WARN] Less than 2 detections! Using right-half fallback.", flush=True)
        h, w = keyframe.shape[:2]
        # Fabricate two detections for the fallback
        all_dets = [
            {"box": [0, 0, w // 2, h], "score": 0.5, "label": args.source_animal},
            {"box": [w // 2, 0, w, h], "score": 0.5, "label": args.source_animal},
        ]

    # Select two instances and choose target
    try:
        det_a, det_b = select_two_instances(all_dets)
    except ValueError as e:
        print(f"[WARN] {e}. Using top-2 by score.", flush=True)
        sorted_dets = sorted(all_dets, key=lambda d: d["score"], reverse=True)
        det_a, det_b = sorted_dets[0], sorted_dets[1]

    target_det, keep_det = choose_target(det_a, det_b, args.swap_strategy)
    target_idx = all_dets.index(target_det) if target_det in all_dets else 0
    keep_idx = all_dets.index(keep_det) if keep_det in all_dets else 1

    print(f"[Detect] Target: det {target_idx} ('{target_det['label']}')")
    print(f"[Detect] Keep:   det {keep_idx} ('{keep_det['label']}')")

    # Save detection visualization
    det_vis = make_detection_vis(keyframe, all_dets, target_idx)
    Image.fromarray(det_vis).save(str(out / "detections.png"))

    # ── Segmentation ─────────────────────────────────────────────────
    print("[Segment] Running SAM...", flush=True)
    predictor = load_sam(args.device)

    mask_target_raw = segment_instance(predictor, keyframe, target_det["box"])
    mask_keep_raw = segment_instance(predictor, keyframe, keep_det["box"])

    del predictor
    torch.cuda.empty_cache()

    mask_target = refine_mask(mask_target_raw, dilate_px=args.dilate_px)
    mask_keep = refine_mask(mask_keep_raw, dilate_px=args.dilate_px)

    print(f"[Segment] Target mask: {(mask_target > 0).sum()} px")
    print(f"[Segment] Keep mask:   {(mask_keep > 0).sum()} px")

    # ── Ownership decomposition ──────────────────────────────────────
    print("[Ownership] Computing overlap and regions...", flush=True)
    overlap = compute_overlap(mask_target, mask_keep)
    overlap_px = int((overlap > 0).sum())
    total_mask_px = int(((mask_target > 0) | (mask_keep > 0)).sum())
    overlap_ratio = overlap_px / max(total_mask_px, 1)
    print(f"[Ownership] Overlap: {overlap_px} px ({overlap_ratio:.1%} of union)")

    # Determine front/back
    front_det, back_det = estimate_front_back(
        target_det, keep_det, overlap, args.front_back_strategy)
    is_target_front = (front_det is target_det)

    if is_target_front:
        front_mask, back_mask = mask_target, mask_keep
        print("[Ownership] Target is FRONT, Keep is BACK")
    else:
        front_mask, back_mask = mask_keep, mask_target
        print("[Ownership] Target is BACK, Keep is FRONT")

    regions = decompose_regions(front_mask, back_mask)

    # Log region stats
    for name, region in regions.items():
        px = int((region > 0).sum())
        print(f"  {name}: {px} px")

    # Save mask/region overlays
    overlay_target = make_mask_overlay(keyframe, mask_target, color=(255, 0, 0), alpha=0.4)
    overlay_keep = make_mask_overlay(overlay_target, mask_keep, color=(0, 0, 255), alpha=0.4)
    overlay_overlap = make_mask_overlay(keyframe, overlap, color=(255, 255, 0), alpha=0.5)
    Image.fromarray(overlay_keep).save(str(out / "overlay_masks.png"))
    Image.fromarray(overlay_overlap).save(str(out / "overlay_overlap.png"))

    # ── Two-pass inpainting ──────────────────────────────────────────
    print("[Stage2] Setting up two-pass inpainting...", flush=True)

    # Build prompts for front and back entities
    # The "target" animal gets the new species; the "keep" animal stays as source
    # (unless --keep-animal is specified)
    target_prompt, target_neg = build_inpaint_prompt(
        args.target_animal, scene_hint="natural lighting")
    if args.keep_animal:
        keep_prompt, keep_neg = build_inpaint_prompt(
            args.keep_animal, scene_hint="natural lighting")
    else:
        keep_prompt = f"a realistic {args.source_animal}, detailed fur, photorealistic"
        keep_neg = "blurry, deformed, chimera"

    # Assign prompts based on front/back
    if is_target_front:
        front_prompt, front_neg = target_prompt, target_neg
        back_prompt, back_neg = keep_prompt, keep_neg
    else:
        front_prompt, front_neg = keep_prompt, keep_neg
        back_prompt, back_neg = target_prompt, target_neg

    # Build inpaint plan
    plan = build_inpaint_plan(regions, front_prompt, back_prompt)
    print(f"[Stage2] Inpaint plan: {len(plan)} passes")
    for mask_p, prompt_p, order_p in plan:
        px = int((mask_p > 0).sum())
        print(f"  Pass {order_p}: {px} px — '{prompt_p[:60]}...'")

    # Compute adaptive strengths
    img_area = args.height * args.width
    back_mask_ratio = int((regions["back_visible"] > 0).sum()) / img_area
    front_mask_ratio = int((regions["front_exclusive"] > 0).sum()) / img_area

    if args.s2_strength > 0:
        back_strength = front_strength = args.s2_strength
    else:
        back_strength = compute_strength(back_mask_ratio, overlap_ratio)
        front_strength = compute_strength(front_mask_ratio, 0.0)

    print(f"[Stage2] Strengths: back={back_strength}, front={front_strength}")

    # Load inpaint pipeline and run on all frames
    pipe2 = load_inpaint_pipeline(args.device)

    stage2_frames = []
    for fi, frame in enumerate(base_frames):
        print(f"\n[Stage2] Frame {fi}/{len(base_frames)}", flush=True)

        result = run_two_pass(
            pipe2, frame,
            back_region=regions["back_visible"],
            front_region=regions["front_exclusive"],
            back_prompt=back_prompt,
            front_prompt=front_prompt,
            back_negative=back_neg,
            front_negative=front_neg,
            back_strength=back_strength,
            front_strength=front_strength,
            guidance=args.s2_guidance,
            steps=args.s2_steps,
            seed=args.seed + fi,
        )
        stage2_frames.append(np.array(result))

    del pipe2
    torch.cuda.empty_cache()

    # ── Save outputs ─────────────────────────────────────────────────
    print("\n[Save] Writing outputs...", flush=True)

    masks_dict = {
        "target_raw": mask_target_raw,
        "keep_raw": mask_keep_raw,
        "target_refined": mask_target,
        "keep_refined": mask_keep,
        "overlap": overlap,
    }

    summary = build_summary(
        config=config,
        dets=all_dets,
        regions=regions,
        target_idx=target_idx,
        keep_idx=keep_idx,
    )
    summary["is_target_front"] = is_target_front
    summary["overlap_ratio"] = round(overlap_ratio, 4)
    summary["back_strength"] = back_strength
    summary["front_strength"] = front_strength

    save_standard_outputs(
        out_dir=str(out),
        stage1_frames=base_frames,
        stage2_frames=stage2_frames,
        dets=all_dets,
        masks=masks_dict,
        regions=regions,
        summary=summary,
    )

    print(f"\n{'=' * 60}")
    print(f"[Phase58 v8] DONE")
    print(f"  Output:        {out}")
    print(f"  Overlap ratio: {overlap_ratio:.1%}")
    print(f"  Target front:  {is_target_front}")
    print(f"  Passes:        {len(plan)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
