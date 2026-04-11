"""
Phase 33 — Chimera Reduction via Score / Gradient Guidance
==========================================================

Method
------
Training-free test-time guidance. At each DDIM step, after the usual
classifier-free-guidance noise prediction, we additionally compute the
gradient of a depth-ranking loss (`l_zorder_direct` from train_phase31)
with respect to the current noisy latents, and push the noise prediction
in the steepest-descent direction of that loss. The VCA layer from the
Phase-31 checkpoint is what provides the sigma field used by the loss.

    ε̂_t = ε̂_uncond + s_cfg · (ε̂_cond - ε̂_uncond)
    g_t  = ∇_{x_t}  L_zorder( σ(x_t) )
    ε̂_t ← ε̂_t  −  s_depth · g_t            (clipped)

Because AnimateDiff's public `__call__` hides the denoising loop, we run it
manually using `pipe.scheduler`, `pipe.unet`, `pipe.vae`, and the existing
`encode_prompt` helper.

Outputs
-------
  p33_baseline_frames.gif           — no guidance (VCA delta still applied)
  p33_guided_frames.gif             — with score guidance
  p33_chimera_mask.gif              — yellow chimera overlay on guided run
  p33_guidance_scale_ablation.gif   — 4 columns: s_depth ∈ {0.1, 0.5, 1.0, 2.0}
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as iio2
from PIL import Image, ImageDraw, ImageFont

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
    l_zorder_direct,
)


# =============================================================================
# Prompts / seeds (shared with phase32)
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
    },
    {
        "prompt": "a red cat and a blue cat running toward each other on a "
                  "grassy field, they meet in the middle, cinematic, "
                  "photorealistic, high detail",
        "negative": "low quality, blurry, deformed, watermark, text, chimera, "
                    "merged, fused, painting",
        "entity0": "red cat",
        "entity1": "blue cat",
    },
]


# =============================================================================
# Metrics / GIF helpers (mirror of phase32)
# =============================================================================
def chimera_score(frames: list[np.ndarray]) -> float:
    total_ch = total_ov = 0
    for f in frames:
        r = f[..., 0].astype(np.int32)
        b = f[..., 2].astype(np.int32)
        total_ch += int(((r > 80) & (b > 80)).sum())
        total_ov += int(((r > 80) | (b > 80)).sum())
    return (total_ch / total_ov) if total_ov else 0.0


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
# Entity embeddings / VCA loader (mirror of phase32)
# =============================================================================
def get_entity_ctx_simple(pipe, e0: str, e1: str, device: str) -> torch.Tensor:
    embs = []
    for text in [e0, e1]:
        tokens = pipe.tokenizer(
            text, return_tensors="pt",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            out = pipe.text_encoder(**tokens).last_hidden_state
        ids = tokens.input_ids[0]
        mask = (ids != pipe.tokenizer.pad_token_id) & (ids != pipe.tokenizer.eos_token_id)
        mask[0] = False
        embs.append(out[0][mask].mean(0))
    return torch.stack(embs, 0).unsqueeze(0).float().to(device)  # (1, 2, 768)


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
    print(f"[phase33] loaded VCA ckpt: {ckpt_path} gamma={gamma_trained:.4f}",
          flush=True)
    return vca_layer, gamma_trained


# =============================================================================
# Latent reshape helpers — AnimateDiff UNet is 2D; it expects 4D input
# =============================================================================
def _to_4d(lat5d: torch.Tensor) -> torch.Tensor:
    """(B, C, T, H, W) → (B*T, C, H, W)"""
    B, C, T, H, W = lat5d.shape
    return lat5d.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)


def _to_5d(lat4d: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """(B*T, C, H, W) → (B, C, T, H, W)"""
    BT, C, H, W = lat4d.shape
    return lat4d.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)


# =============================================================================
# Text encoding (uses the pipe's own encode_prompt)
# =============================================================================
def _encode_prompt_for_cfg(pipe, prompt: str, negative_prompt: str, n_frames: int
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (cond_hs, uncond_hs), each shaped (1, 77, 768).
    UNetMotionModel broadcasts encoder_hidden_states across frames internally,
    so we must NOT repeat over frames here.
    """
    device = pipe.device
    with torch.no_grad():
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
    # (1, 77, 768) each — do NOT repeat over frames
    return prompt_embeds, negative_embeds


# =============================================================================
# Manual decode
# =============================================================================
@torch.no_grad()
def _decode_latents_to_frames(pipe, latents: torch.Tensor,
                              height: int, width: int) -> list[np.ndarray]:
    """
    latents: (1, 4, T, lH, lW) → list of (H, W, 3) uint8 numpy.
    """
    assert latents.dim() == 5, f"expected (1,4,T,H,W), got {latents.shape}"
    T = latents.shape[2]
    lat = latents[0].permute(1, 0, 2, 3)  # (T, 4, lH, lW)
    frames = []
    scale = pipe.vae.config.scaling_factor
    for i in range(T):
        z = (lat[i:i + 1] / scale).to(pipe.vae.dtype)
        dec = pipe.vae.decode(z).sample                   # (1, 3, H, W)
        frame = dec[0].float().permute(1, 2, 0).cpu().numpy()
        frame = np.clip((frame + 1) / 2 * 255, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


# =============================================================================
# Score-guided generation
# =============================================================================
def score_guided_generation(
    pipe,
    vca_layer: VCALayer,
    vca_proc,                 # AdditiveVCAInferProcessor already installed
    prompt: str,
    negative_prompt: str,
    guidance_scale_depth: float = 1.0,
    n_steps: int = 20,
    height: int = 256,
    width: int = 256,
    n_frames: int = 8,
    seed: int = 42,
    cfg_scale: float = 7.5,
    depth_orders: Optional[list] = None,
) -> list[np.ndarray]:
    """
    Manual AnimateDiff denoising loop with depth-ordering score guidance.
    """
    device = pipe.device
    unet_dtype = next(pipe.unet.parameters()).dtype
    g = torch.Generator(device=device).manual_seed(seed)

    # --- text encoding ----------------------------------------------------
    cond_hs, uncond_hs = _encode_prompt_for_cfg(pipe, prompt, negative_prompt, n_frames)
    cond_hs = cond_hs.to(unet_dtype)
    uncond_hs = uncond_hs.to(unet_dtype)

    # --- initial noise ----------------------------------------------------
    lH, lW = height // 8, width // 8
    latents = torch.randn(
        (1, 4, n_frames, lH, lW),
        device=device, dtype=unet_dtype, generator=g,
    )
    pipe.scheduler.set_timesteps(n_steps, device=device)
    latents = latents * pipe.scheduler.init_noise_sigma

    if depth_orders is None:
        depth_orders = [(0, 1)] * n_frames

    for step_i, t in enumerate(pipe.scheduler.timesteps):
        # ---------- CFG: UNet twice (uncond, cond) ------------------------
        # UNetMotionModel expects 5D: (B, C, T, H, W).  It does B*T reshape
        # internally.  encoder_hidden_states must be (B, 77, 768) — NOT
        # repeated over frames; the UNet broadcasts across frames itself.
        latent_input = torch.cat([latents, latents], dim=0)             # (2, C, T, H, W)
        latent_input = pipe.scheduler.scale_model_input(latent_input, t)
        # cond_hs / uncond_hs are (1, 77, 768); cat → (2, 77, 768)
        enc_input = torch.cat([uncond_hs, cond_hs], dim=0)             # (2, 77, 768)

        with torch.no_grad():
            vca_layer.reset_sigma_acc()
            noise_pred_both = pipe.unet(
                latent_input, t, encoder_hidden_states=enc_input
            ).sample                                                     # (2, C, T, H, W)

        noise_pred_uncond, noise_pred_cond = noise_pred_both.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

        # ---------- Depth-ordering guidance -------------------------------
        if guidance_scale_depth > 0.0:
            try:
                latents_grad = latents.detach().float().requires_grad_(True)  # (1, C, T, H, W)
                vca_layer.reset_sigma_acc()
                with torch.enable_grad():
                    scaled = pipe.scheduler.scale_model_input(latents_grad, t).to(unet_dtype)
                    _ = pipe.unet(
                        scaled, t, encoder_hidden_states=cond_hs        # (1, 77, 768)
                    ).sample
                    sigma_acc = list(vca_layer.sigma_acc)
                    if len(sigma_acc) == 0:
                        raise RuntimeError("no sigma accumulated")
                    depth_loss = l_zorder_direct(
                        sigma_acc, depth_orders, entity_masks=None,
                    )
                if torch.isfinite(depth_loss):
                    grad = torch.autograd.grad(
                        depth_loss, latents_grad, retain_graph=False
                    )[0]
                    gn = grad.norm()
                    if gn > 0.1:
                        grad = grad * (0.1 / gn)
                    noise_pred = noise_pred - guidance_scale_depth * grad.detach().to(
                        noise_pred.dtype
                    )
            except Exception as e:
                print(f"    [phase33][warn] step {step_i}: guidance failed ({e})",
                      flush=True)

        # ---------- scheduler step ---------------------------------------
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    return _decode_latents_to_frames(pipe, latents, height, width)


# =============================================================================
# Runner
# =============================================================================
def run_one_prompt(pipe, vca_layer, gamma_trained, p_cfg: dict, args,
                    debug_dir: Path):
    device = str(pipe.device)
    full_prompt = p_cfg["prompt"]
    neg = p_cfg["negative"]
    e0 = p_cfg["entity0"]
    e1 = p_cfg["entity1"]

    print(f"\n{'='*70}", flush=True)
    print(f"[phase33] prompt: {full_prompt}", flush=True)

    entity_ctx = get_entity_ctx_simple(pipe, e0, e1, device)

    # Inject Phase-31 inference VCA processor at INJECT_KEY so that UNet
    # forwards populate vca_layer.sigma_acc.
    orig_procs = inject_vca_p21_infer(pipe, vca_layer, entity_ctx,
                                       gamma_trained=gamma_trained)
    vca_proc = pipe.unet.attn_processors[INJECT_KEY]

    label = f"{e0}__vs__{e1}".replace(" ", "_")

    try:
        # ------ baseline (no depth guidance) ----------------------------
        print("  running baseline (guidance_scale_depth=0.0) …", flush=True)
        frames_base = score_guided_generation(
            pipe, vca_layer, vca_proc,
            prompt=full_prompt, negative_prompt=neg,
            guidance_scale_depth=0.0,
            n_steps=args.n_inference_steps,
            height=args.height, width=args.width,
            n_frames=args.n_frames, seed=42, cfg_scale=7.5,
        )
        base_score = chimera_score(frames_base)
        print(f"    baseline chimera_score = {base_score:.4f}", flush=True)

        # ------ guided ---------------------------------------------------
        print(f"  running guided (guidance_scale_depth={args.guidance_scale}) …",
              flush=True)
        frames_guided = score_guided_generation(
            pipe, vca_layer, vca_proc,
            prompt=full_prompt, negative_prompt=neg,
            guidance_scale_depth=args.guidance_scale,
            n_steps=args.n_inference_steps,
            height=args.height, width=args.width,
            n_frames=args.n_frames, seed=42, cfg_scale=7.5,
        )
        guided_score = chimera_score(frames_guided)
        print(f"    guided   chimera_score = {guided_score:.4f}", flush=True)

        # ------ ablation sweep ------------------------------------------
        ablation_scales = [0.1, 0.5, 1.0, 2.0]
        ablation_runs: list[list[np.ndarray]] = []
        for s in ablation_scales:
            print(f"  ablation s={s} …", flush=True)
            fr = score_guided_generation(
                pipe, vca_layer, vca_proc,
                prompt=full_prompt, negative_prompt=neg,
                guidance_scale_depth=s,
                n_steps=args.n_inference_steps,
                height=args.height, width=args.width,
                n_frames=args.n_frames, seed=42, cfg_scale=7.5,
            )
            ablation_runs.append(fr)
    finally:
        restore_procs(pipe, orig_procs)

    # --- save GIFs --------------------------------------------------------
    base_gif = debug_dir / f"p33_{label}_baseline_frames.gif"
    guided_gif = debug_dir / f"p33_{label}_guided_frames.gif"
    chim_gif = debug_dir / f"p33_{label}_chimera_mask.gif"
    abl_gif = debug_dir / f"p33_{label}_guidance_scale_ablation.gif"

    base_labeled = [
        add_label(f, f"BASE f{i} chim={base_score:.2f}")
        for i, f in enumerate(frames_base)
    ]
    guided_labeled = [
        add_label(f, f"GUID f{i} chim={guided_score:.2f}")
        for i, f in enumerate(frames_guided)
    ]
    iio2.mimsave(str(base_gif), base_labeled, duration=120)
    iio2.mimsave(str(guided_gif), guided_labeled, duration=120)
    print(f"  wrote {base_gif.name}, {guided_gif.name}", flush=True)

    # chimera mask overlay on guided run
    H = frames_guided[0].shape[0]
    masks = chimera_masks(frames_guided)
    chim_frames = [make_chimera_overlay(f, m) for f, m in zip(frames_guided, masks)]
    iio2.mimsave(str(chim_gif), chim_frames, duration=120)
    print(f"  wrote {chim_gif.name}", flush=True)

    # 4-column ablation GIF
    n_frames = args.n_frames
    col_labels = [f"s={s}" for s in ablation_scales]
    ab_scores = [chimera_score(fr) for fr in ablation_runs]
    stacked_frames = []
    for fi in range(n_frames):
        cols = []
        for ci, fr in enumerate(ablation_runs):
            lab = f"{col_labels[ci]} ch={ab_scores[ci]:.2f}"
            cols.append(add_label(fr[fi], lab, font_size=10))
        stacked_frames.append(np.concatenate(cols, axis=1))
    iio2.mimsave(str(abl_gif), stacked_frames, duration=150)
    print(f"  wrote {abl_gif.name}  scores={ab_scores}", flush=True)

    return {
        "prompt": full_prompt,
        "score_baseline": base_score,
        "score_guided": guided_score,
        "ablation_scores": {str(s): sc for s, sc in zip(ablation_scales, ab_scores)},
    }


# =============================================================================
# Main
# =============================================================================
def run_phase33(args):
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        stats = run_one_prompt(pipe, vca_layer, gamma_trained, p_cfg,
                                args, debug_dir)
        summary.append(stats)

    print("\n===== Phase 33 summary =====", flush=True)
    for s in summary:
        print(f"  {s['prompt'][:60]}…  "
              f"base={s['score_baseline']:.4f}  guided={s['score_guided']:.4f}",
              flush=True)
        print(f"    ablation={s['ablation_scores']}", flush=True)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/phase31/best.pt")
    p.add_argument("--debug-dir", type=str, default="debug/chimera")
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--n-inference-steps", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    run_phase33(_parse_args())
