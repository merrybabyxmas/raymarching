"""
Phase 34 — Chimera Reduction via Entity-Specific Latent Compositing
===================================================================

Method
------
Training-free, pixel-space compositing. We generate three videos per prompt:

  1. E0-only video  (prompt = "a red ball rolling to the left ...",    no VCA)
  2. E1-only video  (prompt = "a blue ball rolling to the right ...",  no VCA)
  3. Baseline video (full collision prompt, WITH Phase-31 VCA)         → gives σ

The VCA layer exposes a per-frame depth-ordering field
    σ ∈ R^(BF_total, S, N, Z)
where (BF_total = uncond_frames || cond_frames), S = h·w spatial tokens, N=2
entities, Z=2 depth bins. For each conditional frame we take

    front_mask_lowres[y,x] = 1  iff  σ[f, y*w+x, 0, 0] > σ[f, y*w+x, 1, 0]

i.e. "E0 is closer than E1 at (y,x)". We upsample that (16×16)-style mask to
full resolution with bilinear interpolation, smooth it spatially (σ=2 px)
and then temporally along the frame axis (σ=1 frame) to avoid flicker, and
finally composite:

    frame_comp[t,y,x,:] = α·E0[t,y,x,:] + (1-α)·E1[t,y,x,:]
    where α = smoothed_front_mask[t,y,x]

The hypothesis is that, because the two source videos are rendered separately,
they never share latents and cannot "fuse" colors — so the compositing step
cannot produce a red-AND-blue chimera pixel unless the mask boundary itself
lands inside a region where both source frames happen to be colored.

Outputs (per prompt, for the best seed)
---------------------------------------
  p34_{label}_baseline_frames.gif    — full prompt + VCA (reference / σ source)
  p34_{label}_e0only_frames.gif      — E0-only generation
  p34_{label}_e1only_frames.gif      — E1-only generation
  p34_{label}_composited_frames.gif  — α·E0 + (1-α)·E1
  p34_{label}_depth_mask.gif         — white = E0 front, black = E1 front
  p34_{label}_chimera_mask.gif       — yellow overlay on composited frames
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import imageio.v2 as iio2
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, gaussian_filter1d

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from scripts.run_animatediff import load_pipeline
from scripts.train_phase31 import (
    VCA_ALPHA,
    INJECT_KEY,
    INJECT_QUERY_DIM,
    AdditiveVCAInferProcessor,
    inject_vca_p21_infer,
    restore_procs,
)


# =============================================================================
# Prompts / seeds (shared with phase32 / phase33)
# =============================================================================
COLLISION_PROMPTS = [
    {
        "prompt": "a red ball and a blue ball rolling toward each other on a "
                  "wooden table, they collide in the center, cinematic lighting, "
                  "photorealistic, high quality",
        "negative": "low quality, blurry, deformed, watermark, text, chimera, "
                    "merged, fused, cartoon, painting",
        "entity0": "red ball",
        "entity1": "blue ball",
        "color0_rgb": (200, 50, 50),
        "color1_rgb": (50, 50, 200),
    },
    {
        "prompt": "a red cat and a blue cat running toward each other on a "
                  "grassy field, they meet in the middle, cinematic, "
                  "photorealistic, high detail",
        "negative": "low quality, blurry, deformed, watermark, text, chimera, "
                    "merged, fused, painting",
        "entity0": "red cat",
        "entity1": "blue cat",
        "color0_rgb": (200, 50, 50),
        "color1_rgb": (50, 50, 200),
    },
]

SEEDS = [42, 123, 456, 789, 1337]


# =============================================================================
# Metrics / GIF helpers
# =============================================================================
def chimera_score(frames: list[np.ndarray]) -> float:
    scores = []
    for f in frames:
        r = f[..., 0].astype(float)
        b = f[..., 2].astype(float)
        chimera = (r > 80) & (b > 80)
        overlap = (r > 80) | (b > 80)
        if overlap.sum() == 0:
            scores.append(0.0)
        else:
            scores.append(float(chimera.sum()) / float(overlap.sum()))
    return float(np.mean(scores))


def chimera_masks(frames: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for f in frames:
        r = f[..., 0].astype(np.int32)
        b = f[..., 2].astype(np.int32)
        out.append((r > 80) & (b > 80))
    return out


def add_label(frame_arr: np.ndarray, text: str, font_size: int = 12) -> np.ndarray:
    img = Image.fromarray(frame_arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((3, 3), text, fill=(255, 255, 255), font=font,
              stroke_width=1, stroke_fill=(0, 0, 0))
    return np.array(img)


def make_chimera_overlay(frame: np.ndarray, m: np.ndarray,
                         alpha: float = 0.6) -> np.ndarray:
    overlay = frame.copy().astype(float)
    yellow = np.array([255, 220, 0], dtype=float)
    overlay[m] = overlay[m] * (1 - alpha) + yellow * alpha
    return overlay.clip(0, 255).astype(np.uint8)


# =============================================================================
# VCA loader (mirror of phase32 / phase33)
# =============================================================================
def load_vca_checkpoint(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM, context_dim=768,
        n_heads=8, n_entities=2, z_bins=2, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=0.3,
    ).to(device)
    vca_layer.load_state_dict(ckpt["vca_state_dict"])
    vca_layer.eval()
    gamma_trained = float(ckpt.get("gamma_trained", VCA_ALPHA))
    print(f"[phase34] loaded VCA ckpt: {ckpt_path} gamma={gamma_trained:.4f}",
          flush=True)
    return vca_layer, gamma_trained


def get_entity_ctx(pipe, entity0_text: str, entity1_text: str,
                    device: str) -> torch.Tensor:
    """
    Encode each entity text through CLIP and return mean-pooled embeddings
    (excluding BOS/EOS/PAD tokens).

    Returns (1, 2, 768) on `device` as float32.
    """
    embs = []
    for text in [entity0_text, entity1_text]:
        tokens = pipe.tokenizer(
            text, return_tensors="pt", padding="max_length",
            max_length=pipe.tokenizer.model_max_length, truncation=True,
        ).to(device)
        with torch.no_grad():
            out = pipe.text_encoder(**tokens).last_hidden_state  # (1, 77, 768)
        ids = tokens.input_ids[0]
        mask = (ids != pipe.tokenizer.pad_token_id) & (ids != pipe.tokenizer.eos_token_id)
        mask[0] = False  # drop BOS
        emb = out[0][mask].mean(0)                               # (768,)
        embs.append(emb)
    return torch.stack(embs, 0).unsqueeze(0).float().to(device)  # (1, 2, 768)


# =============================================================================
# Video generation helpers
# =============================================================================
def _find_token_positions(pipe, full_prompt: str, entity_text: str) -> list[int]:
    """
    Find token index positions of `entity_text` inside `full_prompt`'s tokenization.
    Returns a list of token indices (may be empty if not found).
    """
    full_ids = pipe.tokenizer(
        full_prompt, padding=False, truncation=True,
        max_length=pipe.tokenizer.model_max_length,
    ).input_ids

    entity_ids = pipe.tokenizer(
        entity_text, add_special_tokens=False,
    ).input_ids

    positions = []
    for i in range(len(full_ids) - len(entity_ids) + 1):
        if full_ids[i:i + len(entity_ids)] == entity_ids:
            positions.extend(range(i, i + len(entity_ids)))
    return positions


def _get_entity_dominant_embeds(
    pipe, full_prompt: str, negative_prompt: str,
    suppress_entity_text: str, device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode `full_prompt` but zero-out the token embeddings that correspond
    to `suppress_entity_text`.  Same prompt structure → same scene context;
    suppressed entity → it fades from the generation.

    Returns (cond_embeds, uncond_embeds), each (1, 77, 768) float16 or float32.
    """
    unet_dtype = next(pipe.unet.parameters()).dtype

    # --- conditional embedding: suppress one entity's tokens ---------------
    text_inputs = pipe.tokenizer(
        full_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        cond = pipe.text_encoder(**text_inputs).last_hidden_state  # (1, 77, 768)

    suppress_pos = _find_token_positions(pipe, full_prompt, suppress_entity_text)
    if suppress_pos:
        print(f"    suppressing '{suppress_entity_text}' at token positions "
              f"{suppress_pos}", flush=True)
        cond = cond.clone()
        cond[0, suppress_pos, :] = 0.0          # zero out entity tokens
    else:
        print(f"    WARNING: could not find tokens for '{suppress_entity_text}' "
              f"in prompt — generation will be same as baseline", flush=True)

    # --- unconditional embedding -------------------------------------------
    neg_inputs = pipe.tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        uncond = pipe.text_encoder(**neg_inputs).last_hidden_state

    return cond.to(unet_dtype), uncond.to(unet_dtype)


def _generate_entity_dominant(
    pipe, full_prompt: str, negative_prompt: str,
    suppress_entity_text: str,
    n_frames: int, n_steps: int, height: int, width: int,
    seed: int, cfg_scale: float = 7.5,
) -> list[np.ndarray]:
    """
    Generate using the full collision prompt with same seed as baseline,
    but with one entity's text tokens zeroed so it fades out.

    Same seed + same prompt structure → same background / camera / motion;
    suppressed entity → cleaner single-entity pixels in overlap zone.
    """
    device = str(pipe.device)
    cond_embeds, uncond_embeds = _get_entity_dominant_embeds(
        pipe, full_prompt, negative_prompt, suppress_entity_text, device,
    )
    g = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        out = pipe(
            prompt_embeds=cond_embeds,
            negative_prompt_embeds=uncond_embeds,
            num_frames=n_frames,
            num_inference_steps=n_steps,
            guidance_scale=cfg_scale,
            height=height,
            width=width,
            generator=g,
            output_type="pil",
        )
    return [np.array(f) for f in out.frames[0]]


def _generate_plain(pipe, prompt: str, negative_prompt: str,
                    n_frames: int, n_steps: int, height: int, width: int,
                    seed: int, cfg_scale: float = 7.5) -> list[np.ndarray]:
    """Plain AnimateDiff generation (no VCA injected)."""
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    with torch.no_grad():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=n_frames,
            num_inference_steps=n_steps,
            guidance_scale=cfg_scale,
            height=height,
            width=width,
            generator=g,
            output_type="pil",
        )
    return [np.array(f) for f in out.frames[0]]


def _generate_with_vca(pipe, vca_layer: VCALayer, gamma_trained: float,
                       entity_ctx: torch.Tensor,
                       prompt: str, negative_prompt: str,
                       n_frames: int, n_steps: int, height: int, width: int,
                       seed: int, cfg_scale: float = 7.5,
                       ) -> tuple[list[np.ndarray], Optional[torch.Tensor]]:
    """
    Generate with the Phase-31 inference VCA processor injected at INJECT_KEY.

    Returns
    -------
    frames    : list of (H, W, 3) uint8
    last_sigma: copy of `vca_layer.last_sigma` captured AFTER the final
                UNet call during the denoising loop, shape (BF_total, S, N, Z),
                or None if never populated.
    """
    orig_procs = inject_vca_p21_infer(pipe, vca_layer, entity_ctx,
                                       gamma_trained=gamma_trained)
    try:
        g = torch.Generator(device=pipe.device).manual_seed(seed)
        with torch.no_grad():
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=n_frames,
                num_inference_steps=n_steps,
                guidance_scale=cfg_scale,
                height=height,
                width=width,
                generator=g,
                output_type="pil",
            )
        frames = [np.array(f) for f in out.frames[0]]
        last_sigma = (vca_layer.last_sigma.detach().cpu().float()
                      if vca_layer.last_sigma is not None else None)
    finally:
        restore_procs(pipe, orig_procs)

    return frames, last_sigma


# =============================================================================
# Sigma → per-frame composite mask
# =============================================================================
def _sigma_to_full_masks(
    sigma: torch.Tensor,          # (BF_total, S, N, Z), CPU fp32
    n_frames: int,
    out_h: int,
    out_w: int,
    spatial_sigma: float = 0.8,
    temporal_sigma: float = 0.5,
    mask_sharpness: float = 20.0,
) -> np.ndarray:
    """
    Convert σ → (T, out_h, out_w) float α-mask in [0,1] where 1 = E0-front.

    Uses the CONDITIONAL half of σ (the last `n_frames` entries, since
    AnimateDiff CFG batches are [uncond_frames, cond_frames]).

    mask_sharpness: sigmoid temperature applied after smoothing to push mask
    values toward 0/1 — prevents semi-transparent blending of entity pixels.
    A higher value = harder edges (less transparency). 0 = no sharpening.
    """
    assert sigma.dim() == 4, f"expected (BF,S,N,Z), got {sigma.shape}"
    BF = sigma.shape[0]
    F_ = min(n_frames, BF // 2) if BF >= 2 * n_frames else min(n_frames, BF)
    sig_cond = sigma[-F_:]                         # (F, S, N, Z)
    S = sig_cond.shape[1]
    hw = int(round(S ** 0.5))
    if hw * hw != S:
        raise RuntimeError(f"sigma spatial S={S} is not a perfect square")

    # z=0 slice: (F, S, N) → (F, N, hw, hw)
    e0 = sig_cond[:, :, 0, 0].reshape(F_, hw, hw).numpy()
    e1 = sig_cond[:, :, 1, 0].reshape(F_, hw, hw).numpy()
    front = (e0 > e1).astype(np.float32)           # (F, hw, hw)  hard mask

    # Upsample each frame (hw, hw) → (out_h, out_w) with bilinear via PIL
    # (bilinear gives smooth sub-pixel edges at the 16→256 scale boundary)
    ups = np.zeros((F_, out_h, out_w), dtype=np.float32)
    for fi in range(F_):
        im = Image.fromarray((front[fi] * 255).astype(np.uint8), mode="L")
        im = im.resize((out_w, out_h), resample=Image.BILINEAR)
        ups[fi] = np.asarray(im, dtype=np.float32) / 255.0

    # Light spatial smoothing — anti-aliasing only, NOT heavy blurring
    if spatial_sigma > 0:
        for fi in range(F_):
            ups[fi] = gaussian_filter(ups[fi], sigma=spatial_sigma)

    # Temporal smoothing across frame axis
    if temporal_sigma > 0:
        ups = gaussian_filter1d(ups, sigma=temporal_sigma, axis=0)

    # Sigmoid sharpening: pushes α → 0 or 1 away from boundaries so
    # compositing does NOT produce semi-transparent entity pixels.
    # Only boundary pixels (α near 0.5) retain a soft transition for AA.
    if mask_sharpness > 0:
        ups = 1.0 / (1.0 + np.exp(-mask_sharpness * (ups - 0.5)))

    ups = np.clip(ups, 0.0, 1.0)

    # Pad/crop to exactly `n_frames` along time axis
    if F_ < n_frames:
        pad = np.repeat(ups[-1:], n_frames - F_, axis=0)
        ups = np.concatenate([ups, pad], axis=0)
    elif F_ > n_frames:
        ups = ups[:n_frames]
    return ups  # (T, H, W)


# =============================================================================
# Main per-prompt runner
# =============================================================================
def run_compositing_for_prompt(pipe, vca_layer, gamma_trained, prompt_info: dict,
                                seed: int, device: str, args) -> dict:
    """
    Generate entity-dominant variants from the full collision prompt (same seed,
    same scene context) and repair only chimera pixels in the baseline.

    Strategy
    --------
    1. Baseline (full prompt + VCA)      → natural collision scene  + σ mask
    2. E0-dominant (full prompt, E1 suppressed, SAME seed) → E0 pixels, same bg
    3. E1-dominant (full prompt, E0 suppressed, SAME seed) → E1 pixels, same bg
    4. Chimera repair:
         result = baseline everywhere
         result[chimera_px & E0_front] = frames_e0[chimera_px & E0_front]
         result[chimera_px & E1_front] = frames_e1[chimera_px & E1_front]

    Using the same seed + same prompt for E0/E1 dominant ensures the background,
    camera angle, and motion trajectory remain coherent with the baseline.
    Only the suppressed entity fades out, so the chimera overlap region gets
    clean single-entity pixels without the "puzzle-piece" artifact.
    """
    full_prompt = prompt_info["prompt"]
    neg         = prompt_info["negative"]
    e0_text     = prompt_info["entity0"]
    e1_text     = prompt_info["entity1"]

    entity_ctx = get_entity_ctx(pipe, e0_text, e1_text, device)

    # Step 1: Baseline — natural scene + VCA sigma source
    print(f"  [seed={seed}] generating baseline (full prompt + VCA) …", flush=True)
    frames_base, last_sigma = _generate_with_vca(
        pipe, vca_layer, gamma_trained, entity_ctx,
        full_prompt, neg,
        n_frames=args.n_frames, n_steps=args.n_inference_steps,
        height=args.height, width=args.width, seed=seed,
    )
    if last_sigma is None:
        raise RuntimeError("vca_layer.last_sigma is None — injection failed?")

    # Step 2: E0-dominant — full prompt, E1 tokens zeroed, SAME seed
    print(f"  [seed={seed}] generating E0-dominant (E1 suppressed, same seed) …",
          flush=True)
    frames_e0 = _generate_entity_dominant(
        pipe, full_prompt, neg,
        suppress_entity_text=e1_text,           # zero out "blue ball" / "blue cat"
        n_frames=args.n_frames, n_steps=args.n_inference_steps,
        height=args.height, width=args.width, seed=seed,
    )

    # Step 3: E1-dominant — full prompt, E0 tokens zeroed, SAME seed
    print(f"  [seed={seed}] generating E1-dominant (E0 suppressed, same seed) …",
          flush=True)
    frames_e1 = _generate_entity_dominant(
        pipe, full_prompt, neg,
        suppress_entity_text=e0_text,           # zero out "red ball" / "red cat"
        n_frames=args.n_frames, n_steps=args.n_inference_steps,
        height=args.height, width=args.width, seed=seed,
    )

    # Step 4: sigma → per-frame binary depth mask
    H, W = args.height, args.width
    alpha = _sigma_to_full_masks(
        last_sigma, n_frames=args.n_frames, out_h=H, out_w=W,
        spatial_sigma=args.spatial_sigma,
        temporal_sigma=args.temporal_sigma,
        mask_sharpness=args.mask_sharpness,
    )                                           # (T, H, W) in [0, 1]

    # Step 5: Chimera repair — keep baseline, fix only chimera pixels
    T = min(len(frames_base), len(frames_e0), len(frames_e1), alpha.shape[0])
    composited = []
    depth_mask_rgb = []
    for fi in range(T):
        base = frames_base[fi].astype(np.float32)
        f0   = frames_e0[fi].astype(np.float32)
        f1   = frames_e1[fi].astype(np.float32)

        # Chimera pixels in baseline: R>80 AND B>80
        r = base[..., 0]
        b = base[..., 2]
        chim_hard = (r > 80) & (b > 80)                    # (H, W) bool

        # Dilate chimera mask gently for smooth boundary blending
        chim_weight = gaussian_filter(chim_hard.astype(np.float32), sigma=3.0)
        chim_weight = np.clip(chim_weight, 0.0, 1.0)[..., None]  # (H, W, 1)

        # Depth: E0-front weight
        e0_w = alpha[fi][..., None]                         # (H, W, 1)
        e1_w = 1.0 - e0_w

        # Entity composite in chimera zone: E0 where E0 is front, E1 otherwise
        entity_fill = f0 * e0_w + f1 * e1_w

        # Final: baseline outside chimera, entity_fill inside chimera
        result = base * (1.0 - chim_weight) + entity_fill * chim_weight
        composited.append(np.clip(result, 0, 255).astype(np.uint8))

        dm = (alpha[fi] * 255.0).astype(np.uint8)
        depth_mask_rgb.append(np.stack([dm, dm, dm], axis=-1))

    base_score = chimera_score(frames_base[:T])
    comp_score = chimera_score(composited)

    return {
        "seed": seed,
        "frames_e0": frames_e0[:T],
        "frames_e1": frames_e1[:T],
        "frames_base": frames_base[:T],
        "frames_comp": composited,
        "depth_mask_rgb": depth_mask_rgb,
        "base_score": base_score,
        "comp_score": comp_score,
    }


# =============================================================================
# Top-level: loop over prompts / seeds, select best, save GIFs
# =============================================================================
def run_phase34(args):
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determinism
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pipe = load_pipeline(device=device, dtype=torch.float16)
    vca_layer, gamma_trained = load_vca_checkpoint(args.ckpt, device)

    summary = []
    for p_cfg in COLLISION_PROMPTS:
        e0_text = p_cfg["entity0"]
        e1_text = p_cfg["entity1"]
        label = f"{e0_text}__vs__{e1_text}".replace(" ", "_")

        print(f"\n{'=' * 70}", flush=True)
        print(f"[phase34] prompt: {p_cfg['prompt']}", flush=True)

        best = None
        all_runs = []
        for seed in SEEDS:
            run = run_compositing_for_prompt(pipe, vca_layer, gamma_trained,
                                              p_cfg, seed, device, args)
            all_runs.append(run)
            print(f"    seed={seed}  base={run['base_score']:.4f}  "
                  f"comp={run['comp_score']:.4f}", flush=True)
            if best is None or run["comp_score"] < best["comp_score"]:
                best = run

        print(f"  >>> best seed={best['seed']}  base={best['base_score']:.4f} "
              f"comp={best['comp_score']:.4f}", flush=True)

        # -------- save GIFs for the best seed -------------------------------
        base_gif = debug_dir / f"p34_{label}_baseline_frames.gif"
        e0_gif   = debug_dir / f"p34_{label}_e0only_frames.gif"
        e1_gif   = debug_dir / f"p34_{label}_e1only_frames.gif"
        comp_gif = debug_dir / f"p34_{label}_composited_frames.gif"
        dm_gif   = debug_dir / f"p34_{label}_depth_mask.gif"
        chim_gif = debug_dir / f"p34_{label}_chimera_mask.gif"

        labeled_base = [
            add_label(f, f"BASE f{i} chim={best['base_score']:.2f}")
            for i, f in enumerate(best["frames_base"])
        ]
        labeled_e0 = [
            add_label(f, f"E0 f{i}") for i, f in enumerate(best["frames_e0"])
        ]
        labeled_e1 = [
            add_label(f, f"E1 f{i}") for i, f in enumerate(best["frames_e1"])
        ]
        labeled_comp = [
            add_label(f, f"COMP f{i} chim={best['comp_score']:.2f}")
            for i, f in enumerate(best["frames_comp"])
        ]
        labeled_dm = [
            add_label(f, f"mask f{i}") for i, f in enumerate(best["depth_mask_rgb"])
        ]
        masks = chimera_masks(best["frames_comp"])
        chim_overlay = [make_chimera_overlay(f, m)
                        for f, m in zip(best["frames_comp"], masks)]

        iio2.mimsave(str(base_gif), labeled_base, duration=250)
        iio2.mimsave(str(e0_gif),   labeled_e0,   duration=250)
        iio2.mimsave(str(e1_gif),   labeled_e1,   duration=250)
        iio2.mimsave(str(comp_gif), labeled_comp, duration=250)
        iio2.mimsave(str(dm_gif),   labeled_dm,   duration=250)
        iio2.mimsave(str(chim_gif), chim_overlay, duration=250)
        print(f"  wrote: {base_gif.name}, {e0_gif.name}, {e1_gif.name}, "
              f"{comp_gif.name}, {dm_gif.name}, {chim_gif.name}", flush=True)

        summary.append({
            "prompt": p_cfg["prompt"],
            "label": label,
            "best_seed": best["seed"],
            "base_score": best["base_score"],
            "comp_score": best["comp_score"],
            "all_runs": [(r["seed"], r["base_score"], r["comp_score"])
                         for r in all_runs],
        })

    # -------- final summary -------------------------------------------------
    print("\n===== Phase 34 summary =====", flush=True)
    for s in summary:
        print(f"  {s['prompt'][:60]}…", flush=True)
        print(f"    best seed={s['best_seed']}  "
              f"baseline chim={s['base_score']:.4f}  "
              f"composited chim={s['comp_score']:.4f}  "
              f"Δ={s['base_score'] - s['comp_score']:+.4f}", flush=True)
        for sd, bs, cs in s["all_runs"]:
            print(f"      seed={sd}: base={bs:.4f} comp={cs:.4f}", flush=True)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/phase31/best.pt")
    p.add_argument("--debug-dir", type=str, default="debug/chimera")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--n-inference-steps", type=int, default=20)
    p.add_argument("--spatial-sigma", type=float, default=0.8,
                   help="Gaussian spatial smoothing sigma for depth mask (default 0.8)")
    p.add_argument("--temporal-sigma", type=float, default=0.5,
                   help="Gaussian temporal smoothing sigma for depth mask (default 0.5)")
    p.add_argument("--mask-sharpness", type=float, default=20.0,
                   help="Sigmoid sharpening temperature for depth mask (0=off, default 20)")
    return p.parse_args()


if __name__ == "__main__":
    run_phase34(_parse_args())
