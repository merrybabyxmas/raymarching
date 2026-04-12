"""
Phase58 v5 — Reproducible Two-Stage Object Identity Swap

Captures the exact settings that produced the first successful
cat+dog separation (stage2_f4.png in phase58_v4).

Stage 1: Vanilla AnimateDiff 512px — generates photorealistic two-animal scene
Stage 2: Right-half inpainting — replaces one animal with a different species

This is the "proof of concept" script. v6 will replace the manual
right-half mask with GroundingDINO+SAM auto-detection.
"""
import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as iio2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Config that produced the successful f4 ──────────────────────────────

DEFAULT_CONFIG = {
    "stage1": {
        "model": "emilianJR/epiCRealism",
        "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-3",
        "prompt": (
            "a realistic orange tabby cat and a golden retriever puppy "
            "sitting side by side on a cozy carpet, photorealistic, "
            "detailed fur, warm interior lighting"
        ),
        "negative_prompt": (
            "blurry, deformed, extra limbs, watermark, chimera, "
            "fused animals, merged, cartoon, sketch"
        ),
        "num_frames": 8,
        "height": 512,
        "width": 512,
        "num_inference_steps": 25,
        "guidance_scale": 8.0,
    },
    "stage2": {
        "model": "runwayml/stable-diffusion-inpainting",
        "target_prompt": (
            "a realistic golden retriever puppy sitting, "
            "detailed fur, warm lighting"
        ),
        "target_negative": "cat, blurry, deformed, chimera",
        "mask_strategy": "right_half",
        "num_inference_steps": 20,
        "strength": 0.85,
        "guidance_scale": 8.0,
    },
}


def build_stage1_prompt(meta: dict) -> dict:
    """Build Stage 1 prompt from dataset metadata or defaults."""
    kw0 = meta.get("keyword0", "cat")
    kw1 = meta.get("keyword1", "dog")
    prompt = (
        f"a realistic {kw0} and a realistic {kw1} "
        f"sitting side by side on a cozy carpet, photorealistic, "
        f"detailed fur, warm interior lighting"
    )
    return {
        "prompt": prompt,
        "negative_prompt": DEFAULT_CONFIG["stage1"]["negative_prompt"],
    }


def build_stage2_prompts(meta: dict, target_entity_idx: int = 1) -> dict:
    """Build Stage 2 inpainting prompt for the target entity."""
    kw0 = meta.get("keyword0", "cat")
    kw1 = meta.get("keyword1", "dog")
    target_kw = kw1 if target_entity_idx == 1 else kw0
    source_kw = kw0 if target_entity_idx == 1 else kw1
    return {
        "target_prompt": (
            f"a realistic {target_kw}, detailed fur texture, "
            f"natural lighting, photorealistic"
        ),
        "target_negative": f"{source_kw}, blurry, deformed, chimera",
        "source_keyword": source_kw,
        "target_keyword": target_kw,
    }


def make_right_half_mask(height: int, width: int) -> Image.Image:
    """Create a right-half inpainting mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, width // 2:] = 255
    return Image.fromarray(mask)


# ─── Stage 1 ─────────────────────────────────────────────────────────────

def run_stage1(config: dict, seed: int, device: str) -> list:
    """Generate base video with vanilla AnimateDiff."""
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

    cfg = config["stage1"]
    print("[Stage1] Loading AnimateDiff...", flush=True)
    adapter = MotionAdapter.from_pretrained(cfg["motion_adapter"], torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        cfg["model"], motion_adapter=adapter, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, clip_sample=False,
        timestep_spacing="linspace", beta_schedule="linear", steps_offset=1)
    pipe.enable_vae_slicing()
    pipe.to(device)

    gen = torch.Generator(device=device).manual_seed(seed)
    print(f"[Stage1] Prompt: {cfg['prompt'][:80]}...", flush=True)
    out = pipe(
        prompt=cfg["prompt"],
        negative_prompt=cfg["negative_prompt"],
        num_frames=cfg["num_frames"],
        height=cfg["height"], width=cfg["width"],
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        generator=gen, output_type="np",
    )
    frames = [(f * 255).astype(np.uint8) for f in out.frames[0]]
    del pipe, adapter
    torch.cuda.empty_cache()
    print(f"[Stage1] Generated {len(frames)} frames", flush=True)
    return frames


# ─── Stage 2 ─────────────────────────────────────────────────────────────

def run_stage2(
    base_frames: list,
    config: dict,
    seed: int,
    device: str,
) -> list:
    """Inpaint target region in each frame."""
    from diffusers import AutoPipelineForInpainting

    cfg = config["stage2"]
    print("[Stage2] Loading inpainting pipeline...", flush=True)
    pipe = AutoPipelineForInpainting.from_pretrained(
        cfg["model"], torch_dtype=torch.float16).to(device)

    h, w = base_frames[0].shape[:2]
    mask = make_right_half_mask(h, w)

    refined = []
    for fi, bf in enumerate(base_frames):
        img = Image.fromarray(bf)
        gen = torch.Generator(device=device).manual_seed(seed + fi)
        result = pipe(
            prompt=cfg["target_prompt"],
            negative_prompt=cfg["target_negative"],
            image=img, mask_image=mask,
            height=h, width=w,
            num_inference_steps=cfg["num_inference_steps"],
            strength=cfg["strength"],
            guidance_scale=cfg["guidance_scale"],
            generator=gen,
        ).images[0]
        refined.append(np.array(result))
        print(f"  [Stage2] frame {fi}/{len(base_frames)}", flush=True)

    del pipe
    torch.cuda.empty_cache()
    return refined


# ─── Artifact saving ─────────────────────────────────────────────────────

def save_artifacts(
    out_dir: Path,
    stage1_frames: list,
    stage2_frames: list,
    config: dict,
    seed: int,
):
    """Save all outputs in standardized format."""
    out_dir.mkdir(parents=True, exist_ok=True)

    iio2.mimwrite(str(out_dir / "stage1.gif"), stage1_frames, fps=8, loop=0)
    iio2.mimwrite(str(out_dir / "stage2.gif"), stage2_frames, fps=8, loop=0)

    for fi in range(len(stage1_frames)):
        Image.fromarray(stage1_frames[fi]).save(str(out_dir / f"stage1_f{fi}.png"))
        Image.fromarray(stage2_frames[fi]).save(str(out_dir / f"stage2_f{fi}.png"))

    # Side-by-side comparison for key frames
    for fi in [0, len(stage1_frames) // 2, len(stage1_frames) - 1]:
        if fi < len(stage1_frames):
            s1 = stage1_frames[fi]
            s2 = stage2_frames[fi]
            compare = np.concatenate([s1, s2], axis=1)
            Image.fromarray(compare).save(str(out_dir / f"compare_f{fi}.png"))

    run_config = {
        "seed": seed,
        "config": config,
        "n_frames": len(stage1_frames),
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"[Save] Artifacts saved to {out_dir}", flush=True)


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Phase58 v5: Reproducible two-stage identity swap")
    p.add_argument("--output-dir", type=str, default="outputs/phase58_v5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--s2-strength", type=float, default=0.85)
    p.add_argument("--s1-prompt", type=str, default=None)
    p.add_argument("--s2-target-prompt", type=str, default=None)
    args = p.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["stage1"] = dict(DEFAULT_CONFIG["stage1"])
    config["stage2"] = dict(DEFAULT_CONFIG["stage2"])

    if args.s1_prompt:
        config["stage1"]["prompt"] = args.s1_prompt
    if args.s2_target_prompt:
        config["stage2"]["target_prompt"] = args.s2_target_prompt
    config["stage2"]["strength"] = args.s2_strength

    out_dir = Path(args.output_dir)

    print("=" * 60, flush=True)
    print("[Phase58 v5] Reproducible Two-Stage Identity Swap", flush=True)
    print(f"  Seed: {args.seed}", flush=True)
    print(f"  Stage 1: {config['stage1']['prompt'][:60]}...", flush=True)
    print(f"  Stage 2: {config['stage2']['target_prompt'][:60]}...", flush=True)
    print(f"  Mask: {config['stage2']['mask_strategy']}", flush=True)
    print("=" * 60, flush=True)

    stage1_frames = run_stage1(config, args.seed, args.device)
    stage2_frames = run_stage2(stage1_frames, config, args.seed, args.device)
    save_artifacts(out_dir, stage1_frames, stage2_frames, config, args.seed)

    print(f"\n[Phase58 v5] Done.", flush=True)
    print(f"  Compare: {out_dir}/compare_f*.png", flush=True)
    print(f"  Stage 1: {out_dir}/stage1.gif", flush=True)
    print(f"  Stage 2: {out_dir}/stage2.gif", flush=True)


if __name__ == "__main__":
    main()
