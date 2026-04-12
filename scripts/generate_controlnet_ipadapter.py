"""
Phase 57 — AnimateDiff + ControlNet (Depth) + IP-Adapter

Architecture:
  - AnimateDiffControlNetPipeline: scene layout via depth ControlNet
  - IP-Adapter: cat/dog identity from solo reference images
  - ip_adapter_masks: per-entity ownership masks for regional identity injection
  - Our contribution: depth-ordered visible ownership maps

No manual noise composition. No per-entity UNet passes.
The pipeline handles everything — we just provide the right conditions.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import imageio.v2 as iio2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_depth_controlnet_pipeline(device: str = "cuda"):
    """Load AnimateDiff + ControlNet (depth) + IP-Adapter pipeline."""
    from diffusers import (
        AnimateDiffControlNetPipeline,
        ControlNetModel,
        MotionAdapter,
        DDIMScheduler,
    )

    print("[Phase57] Loading ControlNet (depth)...", flush=True)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16,
    )

    print("[Phase57] Loading MotionAdapter...", flush=True)
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3",
        torch_dtype=torch.float16,
    )

    print("[Phase57] Loading AnimateDiffControlNetPipeline...", flush=True)
    pipe = AnimateDiffControlNetPipeline.from_pretrained(
        "emilianJR/epiCRealism",
        motion_adapter=adapter,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )

    print("[Phase57] Loading IP-Adapter (2 instances for 2 entities)...", flush=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name=["ip-adapter-plus_sd15.safetensors",
                     "ip-adapter-plus_sd15.safetensors"],
    )
    pipe.set_ip_adapter_scale([0.6, 0.6])

    pipe.enable_vae_slicing()
    pipe.to(device)
    print(f"[Phase57] Pipeline loaded on {device}", flush=True)
    return pipe


def depth_to_controlnet_frames(
    depth_maps: np.ndarray,
    n_frames: int = 8,
) -> List[Image.Image]:
    """Convert depth maps to ControlNet conditioning frames.

    Args:
        depth_maps: (T, H, W) float32 depth values
        n_frames: target number of frames

    Returns:
        List of PIL Images (H, W, 3) with depth visualization
    """
    frames = []
    T = min(depth_maps.shape[0], n_frames)
    for fi in range(T):
        d = depth_maps[fi]
        d_valid = d[d > 0]
        if d_valid.size == 0:
            d_norm = np.zeros_like(d, dtype=np.uint8)
        else:
            d_min, d_max = d_valid.min(), d_valid.max()
            d_range = max(d_max - d_min, 1e-6)
            d_norm = ((d - d_min) / d_range * 255).clip(0, 255).astype(np.uint8)
            d_norm[d <= 0] = 0
        d_rgb = np.stack([d_norm, d_norm, d_norm], axis=-1)
        frames.append(Image.fromarray(d_rgb))
    while len(frames) < n_frames:
        frames.append(frames[-1])
    return frames


def make_ip_adapter_masks(
    entity_masks: np.ndarray,
    visible_masks: np.ndarray,
    depth_orders: list,
    height: int = 256,
    width: int = 256,
    n_frames: int = 8,
) -> List[torch.Tensor]:
    """Build per-entity IP-Adapter masks from GT data.

    Returns list of 2 tensors (one per IP-Adapter):
    Each has shape [1, 1, H, W] as required by IPAdapterMaskProcessor.
    For video, we use the first frame's mask (IP-Adapter is frame-agnostic).
    """
    from diffusers.image_processor import IPAdapterMaskProcessor
    proc = IPAdapterMaskProcessor()

    T = min(entity_masks.shape[0], n_frames, len(depth_orders))
    S = entity_masks.shape[-1]
    H_mask = int(S ** 0.5)

    masks_out = []
    for eidx in range(2):
        if visible_masks is not None:
            m = visible_masks[0, eidx].reshape(H_mask, H_mask)
        else:
            m = entity_masks[0, eidx].reshape(H_mask, H_mask).astype(np.float32)
        m_resized = cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR)
        m_pil = Image.fromarray((m_resized.clip(0, 1) * 255).astype(np.uint8))
        processed = proc.preprocess([m_pil], height=height, width=width)
        masks_out.append(processed)

    return masks_out


def load_solo_reference_images(
    solo_e0_frames: np.ndarray,
    solo_e1_frames: np.ndarray,
    frame_idx: int = 0,
) -> Tuple[Image.Image, Image.Image]:
    """Load solo reference images for IP-Adapter."""
    ref_e0 = Image.fromarray(solo_e0_frames[frame_idx])
    ref_e1 = Image.fromarray(solo_e1_frames[frame_idx])
    return ref_e0, ref_e1


def generate_with_controlnet_ipadapter(
    pipe,
    prompt: str,
    depth_frames: List[Image.Image],
    ref_images: List[Image.Image],
    ip_masks: Optional[List[torch.Tensor]] = None,
    negative_prompt: str = "blurry, deformed, extra limbs, watermark, chimera, fused",
    num_frames: int = 8,
    height: int = 256,
    width: int = 256,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 0.8,
    seed: int = 42,
) -> List[np.ndarray]:
    """Generate video with ControlNet depth + IP-Adapter identity."""

    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    call_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        conditioning_frames=depth_frames,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        ip_adapter_image=ref_images,  # list of 2 images for 2 IP-Adapters
        generator=generator,
        output_type="np",
    )

    if ip_masks is not None:
        call_kwargs["cross_attention_kwargs"] = {
            "ip_adapter_masks": ip_masks,
        }

    output = pipe(**call_kwargs)
    frames = (output.frames[0] * 255).astype(np.uint8)
    return list(frames)


def main():
    p = argparse.ArgumentParser(description="Phase57: ControlNet+IP-Adapter generation")
    p.add_argument("--data-root", type=str, default="toy/data_objaverse")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="outputs/phase57_controlnet")
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--controlnet-scale", type=float, default=0.8)
    p.add_argument("--ip-adapter-scale", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=5,
                   help="Number of dataset samples to generate")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_depth_controlnet_pipeline(device="cuda")
    pipe.set_ip_adapter_scale([args.ip_adapter_scale, args.ip_adapter_scale])

    from scripts.generate_solo_renders import ObjaverseDatasetPhase40
    ds = ObjaverseDatasetPhase40(args.data_root, n_frames=args.n_frames)
    print(f"[Phase57] Dataset: {len(ds)} samples", flush=True)

    results = []
    for si in range(min(args.n_samples, len(ds))):
        sample = ds[si]
        frames_np = sample[0]     # (T, H, W, 3)
        depth_maps = sample[1]    # (T, H, W)
        depth_orders = sample[2]  # list of (front, back)
        meta = sample[3]
        entity_masks = sample[4]  # (T, 2, S)
        visible_masks = sample[5] if len(sample) > 5 and sample[5] is not None else None
        solo_e0 = sample[6] if len(sample) > 6 else None
        solo_e1 = sample[7] if len(sample) > 7 else None

        prompt = meta.get("prompt_full", "a cat and a dog")
        print(f"\n[Phase57] Sample {si}: {prompt}", flush=True)

        # 1. Depth conditioning frames
        depth_cond = depth_to_controlnet_frames(depth_maps, args.n_frames)

        # 2. Solo reference images for IP-Adapter
        ref_images = []
        if solo_e0 is not None and solo_e1 is not None:
            ref_e0, ref_e1 = load_solo_reference_images(solo_e0, solo_e1, frame_idx=0)
            ref_images = [ref_e0, ref_e1]
        else:
            print("  [warn] No solo renders, skipping IP-Adapter", flush=True)

        # 3. IP-Adapter masks (ownership-aware)
        ip_masks = None
        if ref_images and entity_masks is not None:
            ip_mask_list = make_ip_adapter_masks(
                entity_masks, visible_masks, depth_orders,
                args.height, args.width, args.n_frames)
            ip_masks = [m.to(pipe.device) for m in ip_mask_list]

        # 4. Generate
        try:
            gen_frames = generate_with_controlnet_ipadapter(
                pipe, prompt, depth_cond,
                ref_images=ref_images if ref_images else None,
                ip_masks=ip_masks,
                num_frames=args.n_frames,
                height=args.height, width=args.width,
                num_inference_steps=args.n_steps,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_scale,
                seed=args.seed,
            )

            # Save results
            sample_dir = out_dir / f"sample_{si:03d}"
            sample_dir.mkdir(exist_ok=True)

            # Save composite GIF
            iio2.mimwrite(str(sample_dir / "composite.gif"), gen_frames, fps=8, loop=0)

            # Save individual frames
            for fi, frame in enumerate(gen_frames):
                Image.fromarray(frame).save(str(sample_dir / f"frame_{fi:03d}.png"))

            # Save GT for comparison
            for fi in range(min(len(frames_np), args.n_frames)):
                Image.fromarray(frames_np[fi]).save(str(sample_dir / f"gt_frame_{fi:03d}.png"))

            # Save depth conditioning
            for fi, d_img in enumerate(depth_cond[:args.n_frames]):
                d_img.save(str(sample_dir / f"depth_{fi:03d}.png"))

            # Save solo refs
            if solo_e0 is not None:
                Image.fromarray(solo_e0[0]).save(str(sample_dir / "ref_e0.png"))
            if solo_e1 is not None:
                Image.fromarray(solo_e1[0]).save(str(sample_dir / "ref_e1.png"))

            print(f"  Saved to {sample_dir}", flush=True)
            results.append({
                "sample_idx": si, "prompt": prompt,
                "n_frames": len(gen_frames), "status": "ok"
            })

        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            import traceback; traceback.print_exc()
            results.append({"sample_idx": si, "prompt": prompt, "status": f"error: {e}"})

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Phase57] Done. {len(results)} samples processed.", flush=True)


if __name__ == "__main__":
    main()
