"""
Phase 63 — Collision Generation Demo
======================================

"cat + dog rolling together" 같은 dynamic shot에서
entity identity가 유지되는지 시각화 + SDEdit guided generation.

각 샘플마다 저장:
  1. input frame (RGB)
  2. entity field overlay  (vis_e0=red, vis_e1=blue 반투명)
  3. amodal field overlay  (amo_e0=pink, amo_e1=lightblue)
  4. SDEdit guided output   (guide injection으로 재생성)
  5. side-by-side 비교 grid

Usage:
    CUDA_VISIBLE_DEVICES=3 conda run -n paper_env python scripts/generate_phase63.py \\
        [--out outputs/phase63/generation_demo] [--n_samples 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from models.phase62.system import Phase63System
from models.phase62.backbone_adapter import (
    inject_backbone_extractors,
    BackboneManager,
    DEFAULT_INJECT_KEYS,
)
from data.phase62.dataset_adapter import Phase62DatasetAdapter
from scripts.train_phase39 import compute_dataset_overlap_scores
from scripts.train_animatediff_vca import encode_frames_to_latents
from training.phase62.evaluator import _encode_text
from scripts.run_phase63 import (
    _get_entity_tokens_p63,
    _build_gt_masks,
    _unpack_sample,
    _seed_all,
)

# ── Checkpoints to try (priority order) ─────────────────────────────────────
_CKPT_PRIORITY = [
    "checkpoints/phase63/p63_stage3/best.pt",
    "checkpoints/phase63/p63_stage2/best.pt",
    "checkpoints/phase63/p63_stage1/best.pt",
]

_CONFIG = "config/phase63/stage1.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Colour utils
# ─────────────────────────────────────────────────────────────────────────────

def _prob_to_rgba(prob: torch.Tensor, rgb: tuple, alpha_scale: float = 0.8
                  ) -> np.ndarray:
    """(H, W) [0,1] → (H, W, 4) uint8 RGBA."""
    p = prob.clamp(0, 1).cpu().float().numpy()
    r = np.full_like(p, rgb[0])
    g = np.full_like(p, rgb[1])
    b = np.full_like(p, rgb[2])
    a = p * alpha_scale
    return (np.stack([r, g, b, a], axis=-1) * 255).astype(np.uint8)


def _overlay_field(base_rgb: np.ndarray, field: torch.Tensor,
                   color: tuple, alpha: float = 0.55) -> np.ndarray:
    """Overlay entity field (prob map) on base RGB image. Returns uint8 (H,W,3)."""
    H, W = base_rgb.shape[:2]
    p = field.clamp(0, 1).cpu().float().numpy()
    if p.shape != (H, W):
        p_t = torch.from_numpy(p).unsqueeze(0).unsqueeze(0)
        p = F.interpolate(p_t, size=(H, W), mode="bilinear",
                          align_corners=False).squeeze().numpy()
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    mask = p[..., np.newaxis] * alpha
    out = base_rgb.astype(np.float32) / 255.0 * (1 - mask) + overlay * mask
    return (out * 255).clip(0, 255).astype(np.uint8)


def _label(img: np.ndarray, text: str, font_size: int = 14) -> np.ndarray:
    """Add text label at top-left of image."""
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, pil.width, font_size + 4], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255), font=font)
    return np.array(pil)


# ─────────────────────────────────────────────────────────────────────────────
# SDEdit guided generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_sdedit(
    pipe,
    system: Phase63System,
    backbone_mgr: BackboneManager,
    frame_rgb: np.ndarray,   # (H, W, 3) uint8
    meta: dict,
    device: str,
    n_steps: int = 20,
    strength: float = 0.6,   # how much noise to add (0=no change, 1=pure noise)
    height: int = 256,
    width: int = 256,
) -> np.ndarray:
    """
    SDEdit-style guided generation:
      1. Encode frame → latent z_0
      2. Add noise at t = strength * T
      3. Denoise with entity field guide injection
      4. Decode → RGB
    """
    # ── Encode ───────────────────────────────────────────────────────────────
    frame_resized = np.array(
        Image.fromarray(frame_rgb).convert("RGB").resize((width, height), Image.BILINEAR)
    )[np.newaxis]  # (1, H, W, 3)
    latents_clean = encode_frames_to_latents(pipe, frame_resized, device)  # (1,4,1,H/8,W/8)

    # ── Scheduler setup ──────────────────────────────────────────────────────
    scheduler = pipe.scheduler
    scheduler.set_timesteps(n_steps, device=device)
    timesteps = scheduler.timesteps

    # Find the starting timestep based on strength
    start_step = int(len(timesteps) * (1.0 - strength))
    start_step = max(0, min(start_step, len(timesteps) - 1))
    t_start = timesteps[start_step]

    # Add noise at t_start
    noise = torch.randn_like(latents_clean)
    t_tensor = t_start.unsqueeze(0)
    noisy_latents = scheduler.add_noise(latents_clean, noise, t_tensor)

    # ── Prompt ───────────────────────────────────────────────────────────────
    toks_e0, toks_e1, full_prompt = _get_entity_tokens_p63(pipe, meta, device)
    prompt_embeds = _encode_text(pipe, full_prompt, device)

    # ── Routing hints (from clean frame) ─────────────────────────────────────
    H_f, W_f = 32, 32  # spatial_h/w
    frame_t = torch.from_numpy(frame_resized.astype(np.float32)).to(device) / 255.0
    frame_t = frame_t.permute(0, 3, 1, 2)  # (1, 3, H, W)
    img_small = F.interpolate(frame_t, size=(H_f, W_f), mode="bilinear", align_corners=False)
    c0 = torch.tensor(meta.get("color0", [0.85, 0.15, 0.1]),
                      device=device, dtype=torch.float32).view(1, 3, 1, 1)
    c1 = torch.tensor(meta.get("color1", [0.1, 0.25, 0.85]),
                      device=device, dtype=torch.float32).view(1, 3, 1, 1)
    hint0 = (1.0 - (img_small - c0).abs().mean(1, keepdim=True) * 3.0).clamp(0.0, 1.0)
    hint1 = (1.0 - (img_small - c1).abs().mean(1, keepdim=True) * 3.0).clamp(0.0, 1.0)

    # ── Denoising loop ───────────────────────────────────────────────────────
    x_t = noisy_latents
    guides = {}   # will be populated after first step

    for step_i, t in enumerate(timesteps[start_step:]):
        t_batch = t.unsqueeze(0)

        backbone_mgr.set_entity_tokens(toks_e0, toks_e1)
        backbone_mgr.reset_slot_store()

        # Register guide hooks from previous step (empty on first step)
        if guides:
            system.set_guides(guides)
            system.injection_mgr.register_hooks(pipe.unet)

        # UNet forward: backbone features extracted by hooks, noise predicted
        noise_pred = pipe.unet(
            x_t, t_batch,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        if guides:
            system.injection_mgr.remove_hooks()

        # Update guides from backbone features collected in this forward
        ext = backbone_mgr.primary
        if ext.last_Fg is not None:
            F_g  = ext.last_Fg.float()
            F_e0 = ext.last_F0.float()
            F_e1 = ext.last_F1.float()
            field_out, render_out = system.forward_field_and_render(
                F_g, F_e0, F_e1, img_hint_e0=hint0, img_hint_e1=hint1)
            guides = system.encode_guide(render_out, field_out, F_e0, F_e1)

        # Scheduler step
        x_t = scheduler.step(noise_pred, t, x_t, return_dict=False)[0]

    # ── Decode ───────────────────────────────────────────────────────────────
    # x_t is (1, 4, 1, H/8, W/8)  (AnimateDiff latent with T=1)
    latent_img = x_t[:, :, 0, :, :]  # (1, 4, H/8, W/8)
    latent_img = latent_img / pipe.vae.config.scaling_factor
    img_tensor = pipe.vae.decode(latent_img).sample  # (1, 3, H, W)
    img_np = img_tensor[0].float().cpu().permute(1, 2, 0).numpy()
    img_np = ((img_np + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)

    return img_np


# ─────────────────────────────────────────────────────────────────────────────
# Entity field inference (single frame)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_field_inference(
    pipe,
    system: Phase63System,
    backbone_mgr: BackboneManager,
    frame_rgb: np.ndarray,
    meta: dict,
    device: str,
    t_val: int = 10,
    height: int = 256,
    width: int = 256,
):
    """Run EntityField on a single frame. Returns render_out."""
    frame_resized = np.array(
        Image.fromarray(frame_rgb).convert("RGB").resize((width, height), Image.BILINEAR)
    )[np.newaxis]
    latents = encode_frames_to_latents(pipe, frame_resized, device)

    toks_e0, toks_e1, full_prompt = _get_entity_tokens_p63(pipe, meta, device)
    prompt_embeds = _encode_text(pipe, full_prompt, device)
    backbone_mgr.set_entity_tokens(toks_e0, toks_e1)
    backbone_mgr.reset_slot_store()

    noise = torch.randn_like(latents)
    t_tensor = torch.tensor([t_val], device=device, dtype=torch.long)
    noisy = pipe.scheduler.add_noise(latents, noise, t_tensor)

    _ = pipe.unet(noisy, t_tensor,
                  encoder_hidden_states=prompt_embeds, return_dict=False)

    ext = backbone_mgr.primary
    if ext.last_Fg is None:
        return None
    F_g  = ext.last_Fg.float()
    F_e0 = ext.last_F0.float()
    F_e1 = ext.last_F1.float()

    H_f, W_f = 32, 32
    frame_t = torch.from_numpy(frame_resized.astype(np.float32)).to(device) / 255.0
    frame_t = frame_t.permute(0, 3, 1, 2)
    img_small = F.interpolate(frame_t, size=(H_f, W_f), mode="bilinear", align_corners=False)
    c0 = torch.tensor(meta.get("color0", [0.85, 0.15, 0.1]),
                      device=device, dtype=torch.float32).view(1, 3, 1, 1)
    c1 = torch.tensor(meta.get("color1", [0.1, 0.25, 0.85]),
                      device=device, dtype=torch.float32).view(1, 3, 1, 1)
    hint0 = (1.0 - (img_small - c0).abs().mean(1, keepdim=True) * 3.0).clamp(0.0, 1.0)
    hint1 = (1.0 - (img_small - c1).abs().mean(1, keepdim=True) * 3.0).clamp(0.0, 1.0)

    field_out, render_out = system.forward_field_and_render(
        F_g, F_e0, F_e1, img_hint_e0=hint0, img_hint_e1=hint1)

    return render_out, frame_resized[0]


# ─────────────────────────────────────────────────────────────────────────────
# Visualization grid
# ─────────────────────────────────────────────────────────────────────────────

def make_demo_grid(
    frame_orig: np.ndarray,      # (H, W, 3) original frame
    frame_gen: np.ndarray,       # (H, W, 3) SDEdit output
    vis_e0: torch.Tensor,        # (H', W') visible entity0
    vis_e1: torch.Tensor,
    amo_e0: torch.Tensor,        # (H', W') amodal entity0
    amo_e1: torch.Tensor,
    label: str,
    cell_size: int = 256,
) -> np.ndarray:
    """
    7-panel grid:
    [input | vis_e0_ov | vis_e1_ov | amo_e0_ov | amo_e1_ov | heatmap | sdedit]
    """
    def _resize(img: np.ndarray) -> np.ndarray:
        return np.array(Image.fromarray(img).resize(
            (cell_size, cell_size), Image.BILINEAR))

    def _field_to_heatmap(f: torch.Tensor, cmap: str = "hot") -> np.ndarray:
        """Convert (H,W) prob tensor → (H,W,3) uint8 heatmap."""
        import matplotlib.pyplot as plt
        p = f.clamp(0, 1).cpu().numpy()
        cm = plt.get_cmap(cmap)
        return (cm(p)[:, :, :3] * 255).astype(np.uint8)

    H, W = cell_size, cell_size
    base = _resize(frame_orig)

    # Panel 1: raw input
    p1 = _label(base.copy(), "Input")

    # Panels 2-3: visible overlays
    vis0_up = F.interpolate(vis_e0.unsqueeze(0).unsqueeze(0).float(),
                            size=(H, W), mode="bilinear", align_corners=False).squeeze()
    vis1_up = F.interpolate(vis_e1.unsqueeze(0).unsqueeze(0).float(),
                            size=(H, W), mode="bilinear", align_corners=False).squeeze()
    amo0_up = F.interpolate(amo_e0.unsqueeze(0).unsqueeze(0).float(),
                            size=(H, W), mode="bilinear", align_corners=False).squeeze()
    amo1_up = F.interpolate(amo_e1.unsqueeze(0).unsqueeze(0).float(),
                            size=(H, W), mode="bilinear", align_corners=False).squeeze()

    # Entity 0 = warm red/orange, Entity 1 = cool blue
    p2 = _label(_overlay_field(base, vis0_up, (1.0, 0.15, 0.0)), "Vis E0 (cat)")
    p3 = _label(_overlay_field(base, vis1_up, (0.0, 0.35, 1.0)), "Vis E1 (dog)")
    p4 = _label(_overlay_field(base, amo0_up, (1.0, 0.5,  0.0)), "Amo E0 (cat)")
    p5 = _label(_overlay_field(base, amo1_up, (0.0, 0.6,  1.0)), "Amo E1 (dog)")

    # Panel 6: combined depth/density heatmap (amo_e0 - amo_e1 difference)
    diff = (amo0_up - amo1_up).cpu().numpy()  # red=cat, blue=dog
    H_np, W_np = diff.shape
    cmap_img = np.zeros((H_np, W_np, 3), dtype=np.float32)
    cmap_img[..., 0] = diff.clip(0, 1)   # red channel = cat dominant
    cmap_img[..., 2] = (-diff).clip(0, 1)  # blue channel = dog dominant
    # Add neutral gray where both overlap
    both = np.minimum(amo0_up.cpu().numpy(), amo1_up.cpu().numpy())
    cmap_img += both[..., np.newaxis] * 0.3
    cmap_img = (cmap_img.clip(0, 1) * 255).astype(np.uint8)
    cmap_img_rs = np.array(Image.fromarray(cmap_img).resize((W, H), Image.BILINEAR))
    p6 = _label(cmap_img_rs, "Separation map")

    # Panel 7: SDEdit output
    p7 = _label(_resize(frame_gen), "SDEdit+guide")

    row = np.concatenate([p1, p2, p3, p4, p5, p6, p7], axis=1)

    # Title bar
    title_bar = np.zeros((28, row.shape[1], 3), dtype=np.uint8)
    pil_title = Image.fromarray(title_bar)
    draw = ImageDraw.Draw(pil_title)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 6), label, fill=(255, 255, 200), font=font)
    title_bar = np.array(pil_title)

    return np.concatenate([title_bar, row], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",        default="outputs/phase63/generation_demo")
    parser.add_argument("--n_samples",  type=int, default=8,
                        help="Number of collision samples to visualise")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--sdedit_strength", type=float, default=0.55,
                        help="SDEdit noise strength (0=no change, 1=pure noise)")
    parser.add_argument("--sdedit_steps",    type=int,   default=20)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    _seed_all(args.seed)
    device = args.device
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Checkpoint ─────────────────────────────────────────────────────────
    ckpt_path = next((p for p in _CKPT_PRIORITY if Path(p).exists()), None)
    if ckpt_path is None:
        print("[Gen] ERROR: no checkpoint found.", flush=True)
        sys.exit(1)
    print(f"[Gen] Checkpoint: {ckpt_path}", flush=True)

    # ── Config + System ───────────────────────────────────────────────────
    config = load_config(_CONFIG)
    system = Phase63System(config).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    system.field.load_state_dict(state["field_state"])
    if "guide_encoder_state" in state:
        system.guide_encoder.load_state_dict(state["guide_encoder_state"])
    system.field.eval()
    system.guide_encoder.eval()
    print(f"[Gen] Loaded epoch {state.get('epoch','?')}", flush=True)

    # ── Pipeline ──────────────────────────────────────────────────────────
    from scripts.run_animatediff import load_pipeline
    print("[Gen] Loading pipeline...", flush=True)
    pipe = load_pipeline(device=device)
    pipe.scheduler.set_timesteps(args.sdedit_steps, device=device)

    for p in pipe.unet.parameters():       p.requires_grad_(False)
    for p in pipe.text_encoder.parameters(): p.requires_grad_(False)
    for p in pipe.vae.parameters():          p.requires_grad_(False)

    adapter_rank = int(getattr(config.model, "adapter_rank", 64))
    lora_rank    = int(getattr(config.model, "lora_rank",    4))
    extractors, _ = inject_backbone_extractors(
        pipe, adapter_rank=adapter_rank, lora_rank=lora_rank,
        inject_keys=DEFAULT_INJECT_KEYS)
    for ext in extractors:
        ext.to(device)
    backbone_mgr = BackboneManager(extractors, DEFAULT_INJECT_KEYS, primary_idx=2)
    backbone_mgr.eval()

    # ── Dataset: find high-collision cat+dog samples ───────────────────────
    ds = Phase62DatasetAdapter(
        getattr(config.data, "data_root", "toy/data_objaverse"), n_frames=8)

    print("[Gen] Scoring dataset overlap...", flush=True)
    results = []
    for idx in range(len(ds)):
        try:
            s = ds[idx]
            em = s["entity_masks"].astype(float)  # (T,2,S)
            m0, m1 = em[:, 0, :], em[:, 1, :]
            inter = (m0 * m1).sum() / (m0.sum() + m1.sum() + 1e-6)
            kw = (s["meta"]["keyword0"], s["meta"]["keyword1"])
            results.append((idx, float(inter), kw))
        except Exception:
            pass
    results.sort(key=lambda x: -x[1])

    # Prefer cat+dog, then any high-overlap
    cat_dog = [(i, sc, kw) for i, sc, kw in results
               if set(kw) & {"cat", "dog"}]
    others  = [(i, sc, kw) for i, sc, kw in results
               if set(kw) & {"cat", "dog"} == set()]
    chosen = (cat_dog + others)[:args.n_samples]

    print(f"[Gen] Generating {len(chosen)} samples...", flush=True)
    grids = []

    for rank, (idx, inter_score, kw) in enumerate(chosen):
        sample = ds[idx]
        frames_np, depth_np, depth_orders, meta, sample_dir, \
            entity_masks, visible_masks = _unpack_sample(sample)

        # Pick the frame with highest entity overlap
        T = frames_np.shape[0]
        em = entity_masks.astype(float)  # (T,2,S)
        frame_overlaps = [
            (em[t, 0, :] * em[t, 1, :]).sum() / (em[t, 0, :].sum() + em[t, 1, :].sum() + 1e-6)
            for t in range(T)
        ]
        best_t = int(np.argmax(frame_overlaps))
        frame_rgb = frames_np[best_t]  # (H, W, 3)

        print(f"  [{rank+1}/{len(chosen)}] idx={idx} kw={kw} "
              f"overlap={inter_score:.3f} best_t={best_t}", flush=True)

        try:
            # ── 1. Entity field inference ────────────────────────────────
            result = run_field_inference(
                pipe, system, backbone_mgr, frame_rgb, meta, device,
                t_val=10, height=256, width=256)
            if result is None:
                print(f"    skip: field inference failed", flush=True)
                continue
            render_out, frame_256 = result

            # ── 2. SDEdit guided generation ──────────────────────────────
            frame_gen = run_sdedit(
                pipe, system, backbone_mgr, frame_rgb, meta, device,
                n_steps=args.sdedit_steps,
                strength=args.sdedit_strength,
                height=256, width=256,
            )

            # ── 3. Build visualisation grid ──────────────────────────────
            label = (f"idx={idx}  {kw[0]} + {kw[1]}  "
                     f"overlap={inter_score:.3f}  t={best_t}/{T-1}  "
                     f"[stage3 guided]")
            grid = make_demo_grid(
                frame_orig=frame_256,
                frame_gen=frame_gen,
                vis_e0=render_out.visible_e0[0],
                vis_e1=render_out.visible_e1[0],
                amo_e0=render_out.amodal_e0[0],
                amo_e1=render_out.amodal_e1[0],
                label=label,
                cell_size=256,
            )
            grids.append(grid)

            # Save individual grid
            fname = f"sample_{rank:02d}_idx{idx}_{kw[0]}_{kw[1]}.png"
            Image.fromarray(grid).save(out_dir / fname)
            print(f"    → {fname}", flush=True)

            # Also save individual field maps at higher resolution
            detail_dir = out_dir / "detail"
            detail_dir.mkdir(exist_ok=True)
            base = np.array(
                Image.fromarray(frame_256).resize((512, 512), Image.BILINEAR))
            for name, field, color in [
                ("vis_e0", render_out.visible_e0[0], (1.0, 0.15, 0.0)),
                ("vis_e1", render_out.visible_e1[0], (0.0, 0.35, 1.0)),
                ("amo_e0", render_out.amodal_e0[0],  (1.0, 0.5,  0.0)),
                ("amo_e1", render_out.amodal_e1[0],  (0.0, 0.6,  1.0)),
            ]:
                f_up = F.interpolate(
                    field.unsqueeze(0).unsqueeze(0).float(),
                    size=(512, 512), mode="bilinear", align_corners=False).squeeze()
                img_ov = _overlay_field(base, f_up, color, alpha=0.65)
                Image.fromarray(img_ov).save(
                    detail_dir / f"sample_{rank:02d}_{name}.png")

        except Exception as e:
            import traceback
            print(f"    ERROR: {e}", flush=True)
            traceback.print_exc()
            continue

    # ── Vertical montage of all grids ─────────────────────────────────────
    if grids:
        montage = np.concatenate(grids, axis=0)
        montage_path = out_dir / "montage_all.png"
        Image.fromarray(montage).save(montage_path)
        print(f"\n[Gen] Montage saved → {montage_path}", flush=True)
        print(f"[Gen] Individual panels → {out_dir}/", flush=True)
        print(f"[Gen] Detail overlays  → {out_dir}/detail/", flush=True)
    else:
        print("[Gen] No grids generated.", flush=True)


if __name__ == "__main__":
    main()
