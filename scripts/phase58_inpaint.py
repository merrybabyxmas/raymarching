"""
Phase58 v8 — Two-Pass Collision-Aware Inpainting.

Loads stable-diffusion-inpainting and provides:
- Adaptive strength computation based on mask/overlap ratios.
- Single-region inpainting.
- Two-pass orchestration (back first, front second).
"""
import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional


# ─── Pipeline loading ───────────────────────────────────────────────────

def load_inpaint_pipeline(device: str = "cuda"):
    """Load the inpainting pipeline.

    Returns:
        Inpainting pipeline on the specified device.
    """
    from diffusers import AutoPipelineForInpainting

    pipe = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    ).to(device)
    return pipe


# ─── Prompt construction ────────────────────────────────────────────────

def build_prompt(
    animal_name: str,
    scene_hint: str = "",
) -> Tuple[str, str]:
    """Build inpainting prompt and negative prompt for an animal.

    Args:
        animal_name: Species name (e.g. 'golden retriever puppy').
        scene_hint: Optional scene context (e.g. 'on a carpet in a living room').

    Returns:
        (prompt, negative_prompt) tuple.
    """
    prompt = f"a realistic {animal_name}, detailed fur texture, photorealistic"
    if scene_hint:
        prompt += f", {scene_hint}"

    negative = (
        "blurry, deformed, extra limbs, chimera, fused, cartoon, "
        "anime, drawing, low quality, watermark"
    )
    return prompt, negative


# ─── Adaptive strength ──────────────────────────────────────────────────

def compute_strength(
    mask_area_ratio: float,
    overlap_ratio: float,
) -> float:
    """Compute adaptive inpainting strength based on mask and overlap ratios.

    Larger masks or higher overlap need stronger inpainting to fully
    replace the identity. Returns a value in [0.85, 0.99].

    Args:
        mask_area_ratio: Fraction of image covered by the inpaint mask.
        overlap_ratio: Fraction of the mask that is overlap region.

    Returns:
        Inpainting strength float in [0.85, 0.99].
    """
    # Base strength scales with mask area
    base = 0.85 + 0.10 * min(mask_area_ratio * 4, 1.0)
    # Boost for overlap — these regions need heavier repainting
    overlap_boost = 0.04 * min(overlap_ratio * 2, 1.0)
    strength = min(base + overlap_boost, 0.99)
    return round(strength, 3)


# ─── Single-region inpainting ───────────────────────────────────────────

def inpaint_region(
    pipe,
    image: Image.Image,
    mask: np.ndarray,
    prompt: str,
    negative: str,
    strength: float = 0.95,
    guidance: float = 9.0,
    steps: int = 25,
    seed: int = 42,
) -> Image.Image:
    """Inpaint a single masked region.

    Args:
        pipe: Inpainting pipeline.
        image: Input PIL Image.
        mask: Binary mask (H, W) uint8, 0/255.
        prompt: Positive prompt.
        negative: Negative prompt.
        strength: Inpainting strength (0-1).
        guidance: Guidance scale.
        steps: Number of inference steps.
        seed: Random seed.

    Returns:
        Inpainted PIL Image.
    """
    mask_pil = Image.fromarray(mask)
    h, w = np.array(image).shape[:2]

    # Skip if mask is essentially empty
    if mask.sum() < 100:
        return image.copy()

    device = pipe.device if hasattr(pipe, "device") else "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=image,
        mask_image=mask_pil,
        height=h,
        width=w,
        num_inference_steps=steps,
        strength=strength,
        guidance_scale=guidance,
        generator=gen,
    ).images[0]

    return result


# ─── Two-pass inpainting ───────────────────────────────────────────────

def run_two_pass(
    pipe,
    frame: np.ndarray,
    back_region: np.ndarray,
    front_region: np.ndarray,
    back_prompt: str,
    front_prompt: str,
    back_negative: str = "",
    front_negative: str = "",
    back_strength: float = 0.95,
    front_strength: float = 0.92,
    guidance: float = 9.0,
    steps: int = 25,
    seed: int = 42,
) -> Image.Image:
    """Run two-pass collision-aware inpainting.

    Pass 1: Inpaint back entity (back_region mask).
    Pass 2: Inpaint front entity (front_region mask) on the result of Pass 1.

    This ordering ensures the overlap region gets the back entity's identity
    first, then the front entity is painted on top only where it exclusively
    appears — matching the natural occlusion order.

    Args:
        pipe: Inpainting pipeline.
        frame: Input frame as numpy uint8 (H, W, 3).
        back_region: Back entity inpaint mask (H, W) uint8, 0/255.
        front_region: Front entity inpaint mask (H, W) uint8, 0/255.
        back_prompt: Prompt for back entity.
        front_prompt: Prompt for front entity.
        back_negative: Negative prompt for back entity.
        front_negative: Negative prompt for front entity.
        back_strength: Inpainting strength for back pass.
        front_strength: Inpainting strength for front pass.
        guidance: Guidance scale.
        steps: Inference steps.
        seed: Random seed.

    Returns:
        Final inpainted PIL Image.
    """
    current = Image.fromarray(frame)

    # Pass 1: Back entity
    if back_region.sum() > 100:
        print("  [Inpaint] Pass 1: back entity", flush=True)
        current = inpaint_region(
            pipe, current, back_region,
            prompt=back_prompt,
            negative=back_negative,
            strength=back_strength,
            guidance=guidance,
            steps=steps,
            seed=seed,
        )

    # Pass 2: Front entity
    if front_region.sum() > 100:
        print("  [Inpaint] Pass 2: front entity", flush=True)
        current = inpaint_region(
            pipe, current, front_region,
            prompt=front_prompt,
            negative=front_negative,
            strength=front_strength,
            guidance=guidance,
            steps=steps,
            seed=seed + 1,  # Different seed to avoid pattern repetition
        )

    return current
