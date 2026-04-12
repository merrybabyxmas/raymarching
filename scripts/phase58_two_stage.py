"""
Phase 58 — Two-Stage Entity Identity Preservation

Stage 1: Layout generation (AnimateDiff + depth ControlNet)
  - Generates base video with correct spatial structure
  - Uses full prompt "a cat and a dog tangled together"
  - depth ControlNet provides layout from GT depth maps

Stage 2: Region-wise identity refinement (per-frame inpainting)
  - For each frame, inpaint entity regions with entity-specific prompts
  - Front entity region: inpaint with "a realistic cat, detailed fur"
  - Back entity visible region: inpaint with "a realistic dog, detailed fur"
  - Uses ownership masks (front-exclusive, overlap-front, overlap-back, back-exclusive)
  - Preserves background from Stage 1

Our contribution: depth-ordered visible ownership maps that decompose
each frame into non-overlapping regions for sequential inpainting.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import imageio.v2 as iio2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Ownership mask decomposition ─────────────────────────────────────────

def decompose_ownership_masks(
    entity_masks: np.ndarray,    # (T, 2, S) binary masks
    visible_masks: np.ndarray,   # (T, 2, S) float visibility
    depth_orders: list,          # [(front_idx, back_idx), ...]
    target_hw: Tuple[int, int] = (512, 512),
) -> Dict[str, List[Image.Image]]:
    """Decompose entity masks into ownership regions per frame.

    Returns dict of mask image lists:
      'front_region': where front entity is visible (exclusive + overlap-front)
      'back_region':  where back entity is visible (exclusive + overlap-back)
      'any_entity':   union of both entity regions
      'background':   where no entity is present
    """
    T = min(entity_masks.shape[0], len(depth_orders))
    S = entity_masks.shape[-1]
    H_mask = int(S ** 0.5)
    H, W = target_hw

    result = {k: [] for k in ['front_region', 'back_region', 'any_entity', 'background']}

    for fi in range(T):
        front_idx = int(depth_orders[fi][0])
        back_idx = 1 - front_idx

        m_front = entity_masks[fi, front_idx].reshape(H_mask, H_mask).astype(np.float32)
        m_back = entity_masks[fi, back_idx].reshape(H_mask, H_mask).astype(np.float32)

        if visible_masks is not None:
            v_front = visible_masks[fi, front_idx].reshape(H_mask, H_mask)
            v_back = visible_masks[fi, back_idx].reshape(H_mask, H_mask)
        else:
            v_front = m_front
            v_back = m_back

        # Front region: where front entity is visible
        front_region = cv2.resize(v_front, (W, H), interpolation=cv2.INTER_LINEAR).clip(0, 1)
        # Back region: where back entity is visible (not occluded by front)
        back_region = cv2.resize(v_back, (W, H), interpolation=cv2.INTER_LINEAR).clip(0, 1)
        # Union
        any_entity = np.maximum(front_region, back_region).clip(0, 1)
        bg = (1.0 - any_entity).clip(0, 1)

        for key, arr in [('front_region', front_region), ('back_region', back_region),
                         ('any_entity', any_entity), ('background', bg)]:
            result[key].append(Image.fromarray((arr * 255).astype(np.uint8)))

    return result


# ─── Stage 1: Layout generation ──────────────────────────────────────────

def stage1_layout_generation(
    depth_maps: np.ndarray,
    prompt: str,
    n_frames: int = 8,
    height: int = 512,
    width: int = 512,
    n_steps: int = 25,
    controlnet_scale: float = 0.7,
    guidance_scale: float = 7.5,
    seed: int = 42,
    device: str = "cuda",
) -> Tuple[List[np.ndarray], object]:
    """Generate base video with AnimateDiff + depth ControlNet."""
    from diffusers import (
        AnimateDiffControlNetPipeline, ControlNetModel,
        MotionAdapter, DDIMScheduler,
    )
    from scripts.generate_controlnet_ipadapter import depth_to_controlnet_frames

    print("[Stage1] Loading pipeline...", flush=True)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3", torch_dtype=torch.float16)
    pipe = AnimateDiffControlNetPipeline.from_pretrained(
        "emilianJR/epiCRealism", motion_adapter=adapter,
        controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, clip_sample=False,
        timestep_spacing="linspace", beta_schedule="linear", steps_offset=1)
    pipe.enable_vae_slicing()
    pipe.to(device)

    depth_cond = depth_to_controlnet_frames(depth_maps, n_frames)

    print(f"[Stage1] Generating layout: '{prompt}'", flush=True)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=prompt,
        negative_prompt="blurry, deformed, extra limbs, watermark, low quality",
        num_frames=n_frames, height=height, width=width,
        num_inference_steps=n_steps, guidance_scale=guidance_scale,
        conditioning_frames=depth_cond,
        controlnet_conditioning_scale=controlnet_scale,
        generator=gen, output_type="np",
    )
    frames = [(f * 255).astype(np.uint8) for f in out.frames[0]]
    print(f"[Stage1] Generated {len(frames)} frames", flush=True)

    del pipe, controlnet, adapter
    torch.cuda.empty_cache()

    return frames, None


# ─── Stage 2: Region-wise identity refinement ────────────────────────────

def stage2_identity_refinement(
    base_frames: List[np.ndarray],
    ownership_masks: Dict[str, List[Image.Image]],
    depth_orders: list,
    entity_prompts: Tuple[str, str],
    height: int = 512,
    width: int = 512,
    n_steps: int = 20,
    strength: float = 0.65,
    guidance_scale: float = 7.5,
    seed: int = 42,
    device: str = "cuda",
) -> List[np.ndarray]:
    """Inpaint entity regions with entity-specific prompts.

    Sequential inpainting:
    1. First inpaint BACK entity visible region with back entity prompt
    2. Then inpaint FRONT entity region with front entity prompt
    Order matters: front overwrites back in overlap.
    """
    from diffusers import AutoPipelineForInpainting

    print("[Stage2] Loading inpainting pipeline...", flush=True)
    pipe = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to(device)

    prompt_e0, prompt_e1 = entity_prompts
    refined = []

    for fi, base_frame in enumerate(base_frames):
        front_idx = int(depth_orders[fi][0]) if fi < len(depth_orders) else 0
        back_idx = 1 - front_idx

        front_prompt = prompt_e0 if front_idx == 0 else prompt_e1
        back_prompt = prompt_e1 if front_idx == 0 else prompt_e0

        base_img = Image.fromarray(base_frame).resize((width, height))
        neg = "blurry, deformed, chimera, fused, extra limbs"

        # Step 1: Inpaint back entity region
        back_mask = ownership_masks['back_region'][fi].resize((width, height))
        back_mask_np = np.array(back_mask).astype(np.float32) / 255.0
        if back_mask_np.sum() > 10:
            gen = torch.Generator(device=device).manual_seed(seed + fi * 2)
            result = pipe(
                prompt=back_prompt,
                negative_prompt=neg,
                image=base_img,
                mask_image=back_mask,
                height=height, width=width,
                num_inference_steps=n_steps,
                strength=strength,
                guidance_scale=guidance_scale,
                generator=gen,
            )
            base_img = result.images[0]

        # Step 2: Inpaint front entity region (overwrites overlap)
        front_mask = ownership_masks['front_region'][fi].resize((width, height))
        front_mask_np = np.array(front_mask).astype(np.float32) / 255.0
        if front_mask_np.sum() > 10:
            gen = torch.Generator(device=device).manual_seed(seed + fi * 2 + 1)
            result = pipe(
                prompt=front_prompt,
                negative_prompt=neg,
                image=base_img,
                mask_image=front_mask,
                height=height, width=width,
                num_inference_steps=n_steps,
                strength=strength,
                guidance_scale=guidance_scale,
                generator=gen,
            )
            base_img = result.images[0]

        refined.append(np.array(base_img))
        if fi % 2 == 0:
            print(f"  [Stage2] Frame {fi}/{len(base_frames)} refined", flush=True)

    del pipe
    torch.cuda.empty_cache()
    print(f"[Stage2] Refined {len(refined)} frames", flush=True)
    return refined


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Phase58: Two-Stage Identity Preservation")
    p.add_argument("--data-root", type=str, default="toy/data_objaverse")
    p.add_argument("--output-dir", type=str, default="outputs/phase58_twostage")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--s1-steps", type=int, default=25)
    p.add_argument("--s1-controlnet-scale", type=float, default=0.7)
    p.add_argument("--s2-steps", type=int, default=20)
    p.add_argument("--s2-strength", type=float, default=0.65)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=3)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts.generate_solo_renders import ObjaverseDatasetPhase40
    ds = ObjaverseDatasetPhase40(args.data_root, n_frames=args.n_frames)
    print(f"[Phase58] Dataset: {len(ds)} samples", flush=True)

    for si in range(min(args.n_samples, len(ds))):
        sample = ds[si]
        frames_gt = sample[0]
        depth_maps = sample[1]
        depth_orders = sample[2]
        meta = sample[3]
        entity_masks = sample[4]
        visible_masks = sample[5] if len(sample) > 5 else None

        prompt_full = meta.get("prompt_full", "a cat and a dog")
        kw0 = meta.get("keyword0", "cat")
        kw1 = meta.get("keyword1", "dog")

        prompt_e0 = f"a realistic {kw0}, detailed texture, natural lighting, photorealistic"
        prompt_e1 = f"a realistic {kw1}, detailed texture, natural lighting, photorealistic"

        print(f"\n{'='*60}", flush=True)
        print(f"[Phase58] Sample {si}: {prompt_full}", flush=True)
        print(f"  e0: '{prompt_e0}'  e1: '{prompt_e1}'", flush=True)

        sample_dir = out_dir / f"sample_{si:03d}"
        sample_dir.mkdir(exist_ok=True)

        # Decompose ownership masks
        ownership = decompose_ownership_masks(
            entity_masks, visible_masks, depth_orders,
            target_hw=(args.height, args.width))

        # Save ownership masks for inspection
        for fi in range(min(2, args.n_frames)):
            for key in ['front_region', 'back_region', 'any_entity']:
                ownership[key][fi].save(str(sample_dir / f"mask_{key}_f{fi}.png"))

        # Stage 1: Layout generation
        base_frames, _ = stage1_layout_generation(
            depth_maps, prompt_full,
            n_frames=args.n_frames, height=args.height, width=args.width,
            n_steps=args.s1_steps,
            controlnet_scale=args.s1_controlnet_scale,
            guidance_scale=args.guidance_scale,
            seed=args.seed)

        iio2.mimwrite(str(sample_dir / "stage1_layout.gif"), base_frames, fps=8, loop=0)
        for fi in [0, 4]:
            if fi < len(base_frames):
                Image.fromarray(base_frames[fi]).save(
                    str(sample_dir / f"stage1_f{fi}.png"))

        # Stage 2: Identity refinement
        refined_frames = stage2_identity_refinement(
            base_frames, ownership, depth_orders,
            entity_prompts=(prompt_e0, prompt_e1),
            height=args.height, width=args.width,
            n_steps=args.s2_steps,
            strength=args.s2_strength,
            guidance_scale=args.guidance_scale,
            seed=args.seed)

        iio2.mimwrite(str(sample_dir / "stage2_refined.gif"), refined_frames, fps=8, loop=0)
        for fi in [0, 4]:
            if fi < len(refined_frames):
                Image.fromarray(refined_frames[fi]).save(
                    str(sample_dir / f"stage2_f{fi}.png"))

        # Save GT for comparison
        for fi in [0, 4]:
            if fi < len(frames_gt):
                Image.fromarray(frames_gt[fi]).save(str(sample_dir / f"gt_f{fi}.png"))

        print(f"  Saved to {sample_dir}", flush=True)

    print(f"\n[Phase58] Done. {min(args.n_samples, len(ds))} samples processed.", flush=True)


if __name__ == "__main__":
    main()
