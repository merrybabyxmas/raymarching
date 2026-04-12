"""
Phase58 v8 — Stage 1: Scene Generation with AnimateDiff.

Generates a photorealistic two-animal scene using AnimateDiff with
epiCRealism backbone and DDIMScheduler.
"""
import numpy as np
import torch
from typing import List, Tuple


# ─── Pipeline loading ───────────────────────────────────────────────────

def load_pipeline(device: str = "cuda"):
    """Load AnimateDiff pipeline with motion adapter and DDIM scheduler.

    Returns:
        pipe: AnimateDiffPipeline ready for generation.
    """
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3",
        torch_dtype=torch.float16,
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        "emilianJR/epiCRealism",
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.enable_vae_slicing()
    pipe.to(device)
    return pipe


# ─── Prompt construction ────────────────────────────────────────────────

SCENARIO_TEMPLATES = {
    "side_by_side": (
        "two realistic {animal}s sitting side by side on a cozy carpet "
        "in a warm living room, photorealistic, detailed fur, natural lighting, "
        "clear separation between the two animals"
    ),
    "tangled": (
        "two realistic {animal}s playfully tangled together on a soft blanket, "
        "overlapping bodies, one partially behind the other, "
        "photorealistic, detailed fur, warm indoor lighting"
    ),
    "climbing": (
        "two realistic {animal}s climbing on a wooden cat tree, "
        "one {animal} on top partially overlapping the lower one, "
        "photorealistic, detailed fur, natural window lighting"
    ),
    "playing": (
        "two realistic {animal}s playing together on green grass in a park, "
        "chasing each other, bodies close and partially overlapping, "
        "photorealistic, detailed fur, golden hour sunlight"
    ),
}

NEGATIVE_BASE = (
    "blurry, deformed, extra limbs, chimera, fused bodies, cartoon, "
    "anime, drawing, sketch, low quality, watermark, text"
)


def build_prompt(
    source_animal: str,
    target_animal: str = "",
    scenario: str = "side_by_side",
) -> Tuple[str, str]:
    """Build generation prompt and negative prompt.

    Args:
        source_animal: Base animal species for scene generation.
        target_animal: Unused in Stage 1 (reserved for reference).
        scenario: One of 'side_by_side', 'tangled', 'climbing', 'playing'.

    Returns:
        (prompt, negative_prompt) tuple.
    """
    template = SCENARIO_TEMPLATES.get(scenario, SCENARIO_TEMPLATES["side_by_side"])
    prompt = template.format(animal=source_animal)
    negative = NEGATIVE_BASE
    return prompt, negative


# ─── Generation ─────────────────────────────────────────────────────────

def generate(
    pipe,
    prompt: str,
    negative: str,
    n_frames: int = 8,
    height: int = 512,
    width: int = 512,
    steps: int = 25,
    guidance: float = 8.0,
    seed: int = 42,
) -> List[np.ndarray]:
    """Generate frames with AnimateDiff.

    Args:
        pipe: AnimateDiffPipeline.
        prompt: Positive prompt.
        negative: Negative prompt.
        n_frames: Number of output frames.
        height: Frame height.
        width: Frame width.
        steps: Inference steps.
        guidance: Guidance scale.
        seed: Random seed.

    Returns:
        List of numpy uint8 frames (H, W, 3).
    """
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    output = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_frames=n_frames,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen,
        output_type="np",
    )
    frames = [(f * 255).astype(np.uint8) for f in output.frames[0]]
    return frames
