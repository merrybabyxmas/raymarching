"""
Phase 56 — T-COMP Style 3-Pass Noise Composition Training
==========================================================

LISA T-COMP inspired: instead of single-pass decomposition heads (Phase 46-53),
run UNet 3 times per training step with entity-mode switching:
  Pass 1 (bg) : shared LoRA + full prompt      → noise_pred_bg
  Pass 2 (e0) : entity0 LoRA + entity0 prompt  → noise_pred_e0
  Pass 3 (e1) : entity1 LoRA + entity1 prompt  → noise_pred_e1

Transmittance compositing with GT masks:
  noise_composite = transmittance_compose(noise_e0, noise_e1, noise_bg, m0, m1, depth_orders)

Losses:
  l_composite = MSE(noise_composite, noise)
  l_masked_e0 = masked_MSE(noise_pred_e0, noise, m0)
  l_masked_e1 = masked_MSE(noise_pred_e1, noise, m1)
  loss = l_composite + 0.5 * l_masked_e0 + 0.5 * l_masked_e1

2-stage training:
  Stage A (8 epochs): entity LoRA only (lr=5e-5)
  Stage B (12 epochs): entity LoRA + global LoRA (lr=2e-5, 5e-6 for global)
"""
import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import imageio.v2 as iio2
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase56 import (
    Phase56Processor,
    MultiBlockSlotManagerP56,
    inject_multi_block_entity_slot_p56,
    restore_multiblock_state_p56,
    transmittance_composite_absolute,
    residual_transmittance_composite,
    outside_mask_suppression,
    masked_mse,
    solo_entity_anchor,
    BLOCK_INNER_DIMS,
    DEFAULT_INJECT_KEYS,
)
from models.entity_slot_phase40 import (
    compute_visible_iou_e0,
    compute_visible_iou_e1,
    compute_visible_masks,
)
from models.entity_slot import (
    compute_ordering_accuracy,
    compute_wrong_slot_leak,
    compute_overlap_score,
)
from scripts.run_animatediff import load_pipeline
from scripts.train_animatediff_vca import encode_frames_to_latents
from scripts.train_phase31 import (
    get_color_entity_context,
    ObjaverseDatasetWithMasks,
)
from scripts.train_phase35 import get_entity_token_positions
from scripts.train_phase39 import (
    compute_dataset_overlap_scores,
    split_train_val,
    make_sampling_weights,
)
from scripts.generate_solo_renders import ObjaverseDatasetPhase40
from scripts.prompt_identity import make_identity_prompts

# =============================================================================
# Hyperparameters
# =============================================================================
DEFAULT_STAGE_A_EPOCHS    = 8
DEFAULT_STAGE_B_EPOCHS    = 12
DEFAULT_STEPS_PER_EPOCH   = 20
DEFAULT_LA_COMPOSITE      = 1.0
DEFAULT_LA_MASKED         = 0.5
DEFAULT_LR_ENTITY_LORA    = 5e-5
DEFAULT_LR_GLOBAL_LORA    = 5e-6
DEFAULT_LR_ADAPTER        = 2e-5
DEFAULT_ADAPTER_RANK      = 64
DEFAULT_LORA_RANK         = 4
DEFAULT_COLLISION_AUG_PROB_A = 0.8
DEFAULT_COLLISION_AUG_PROB_B = 0.5
VAL_FRAC                  = 0.2
MIN_VAL_SAMPLES           = 3

DEFAULT_COLLISION_OV_MIN  = 0.005
DEFAULT_COLLISION_OV_MAX  = 0.25
DEFAULT_COLLISION_MIN_ACCEPT = 0.003


# =============================================================================
# Text encoding helpers
# =============================================================================

def encode_text(pipe, text: str, device: str) -> torch.Tensor:
    """Encode text to CLIP hidden states: (1, 77, 768) fp16."""
    tok = pipe.tokenizer(
        text, return_tensors="pt", padding="max_length",
        max_length=pipe.tokenizer.model_max_length, truncation=True,
    ).to(device)
    with torch.no_grad():
        enc = pipe.text_encoder(**tok).last_hidden_state.half()
    return enc   # (1, 77, 768)


def _append_scene_context(entity_prompt: str, scene_prompt: str) -> str:
    text = str(entity_prompt or "").strip()
    scene = str(scene_prompt or "").strip()
    if not text:
        text = "an entity"
    if not scene:
        return text
    lowered = text.lower()
    if ("same scene as" in lowered or "scene with" in lowered
            or "same environment as" in lowered):
        return text
    return f"{text}, in the same scene as {scene}"


def get_entity_prompts(meta: dict, full_prompt: str) -> Tuple[str, str]:
    """Get entity prompts while forcing shared scene context."""
    base_e0, base_e1, full_text, _, _ = make_identity_prompts(meta)
    scene_prompt = str(full_prompt or full_text).strip() or full_text

    prompt_e0 = _append_scene_context(meta.get("prompt_entity0", base_e0), scene_prompt)
    prompt_e1 = _append_scene_context(meta.get("prompt_entity1", base_e1), scene_prompt)
    return prompt_e0, prompt_e1


def get_bg_prompt(meta: dict, full_prompt: str) -> str:
    """Prompt for the shared background baseline without explicit entity ownership."""
    noun_e0, noun_e1, full_text, _, _ = make_identity_prompts(meta)
    scene_prompt = str(full_prompt or full_text).strip() or full_text
    return (
        f"background scene only, same environment as {scene_prompt}, "
        f"without {noun_e0} or {noun_e1}"
    )


# =============================================================================
# Mask downsampling to latent resolution
# =============================================================================

def downsample_masks_to_latent(
    entity_masks: np.ndarray,   # (T, 2, S) where S = H_mask * W_mask
    latent_hw:    int = 32,     # latent spatial resolution (256 // 8)
) -> torch.Tensor:
    """
    Downsample GT entity masks to latent spatial resolution.

    Returns: (T, 2, latent_hw, latent_hw) float32 tensor.
    """
    T, N, S = entity_masks.shape
    H_mask = int(S ** 0.5)
    assert H_mask * H_mask == S, f"Entity masks must be square: S={S}, sqrt={H_mask}"

    masks_4d = torch.from_numpy(entity_masks.astype(np.float32))  # (T, 2, S)
    masks_4d = masks_4d.view(T, N, H_mask, H_mask)                 # (T, 2, H, H)

    if H_mask != latent_hw:
        masks_4d = F.interpolate(
            masks_4d.float(), size=(latent_hw, latent_hw),
            mode='bilinear', align_corners=False)

    return masks_4d.clamp(0.0, 1.0)   # (T, 2, latent_hw, latent_hw)


# =============================================================================
# Collision augmentation (simplified from Phase 46)
# =============================================================================

def _try_collision_augment(
    dataset,
    sample_idx:   int,
    sample,
    overlap_min:  float = 0.08,
    overlap_max:  float = 0.25,
    max_shift:    int   = 96,
    max_tries:    int   = 24,
    min_accept:   float = 0.04,
) -> Optional[tuple]:
    """
    Build collision-heavy synthetic sample from solo frames + raw masks.
    Returns (frames_aug, depth_orders_aug, entity_masks_aug) or None.
    """
    if len(sample) < 8:
        return None
    if not hasattr(dataset, "samples") or sample_idx >= len(dataset.samples):
        return None

    frames_np, _, depth_orders, _meta, entity_masks, _visible, solo_e0, solo_e1 = sample
    seq_dir = dataset.samples[sample_idx].get("dir", None)
    if seq_dir is None:
        return None
    seq_dir = Path(seq_dir)
    mask_dir = seq_dir / "mask"
    if not mask_dir.exists():
        return None

    T = min(frames_np.shape[0], len(depth_orders), entity_masks.shape[0])
    if solo_e0 is None or solo_e1 is None:
        return None
    T = min(T, solo_e0.shape[0], solo_e1.shape[0])
    if T <= 0:
        return None

    # Load raw binary masks at render resolution
    raw0, raw1 = [], []
    for fi in range(T):
        p0 = mask_dir / f"{fi:04d}_entity0.png"
        p1 = mask_dir / f"{fi:04d}_entity1.png"
        if not p0.exists() or not p1.exists():
            return None
        m0 = (np.array(Image.open(p0).convert("L"), dtype=np.uint8) > 128).astype(np.uint8)
        m1 = (np.array(Image.open(p1).convert("L"), dtype=np.uint8) > 128).astype(np.uint8)
        raw0.append(m0)
        raw1.append(m1)

    S = entity_masks.shape[-1]
    hw = int(S ** 0.5)
    rng = np.random.default_rng()
    best = None
    best_err = 1e9

    for _ in range(max_tries):
        dy = rng.integers(-max_shift, max_shift + 1)
        dx = rng.integers(-max_shift, max_shift + 1)

        shifted_frames = []
        shifted_masks  = []
        shifted_depths = []

        for fi in range(T):
            H, W = solo_e0[fi].shape[:2]
            canvas = np.zeros((H, W, 3), dtype=np.uint8)

            m0_r = raw0[fi]
            m1_r = raw1[fi]

            # Shift entity 1
            m1_shifted = np.zeros_like(m1_r)
            s1_shifted = np.zeros_like(solo_e1[fi])
            y1s = max(0, dy); y1e = min(H, H + dy)
            x1s = max(0, dx); x1e = min(W, W + dx)
            sy  = max(0, -dy); sx = max(0, -dx)
            h_c = y1e - y1s; w_c = x1e - x1s
            if h_c > 0 and w_c > 0:
                m1_shifted[y1s:y1e, x1s:x1e] = m1_r[sy:sy+h_c, sx:sx+w_c]
                s1_shifted[y1s:y1e, x1s:x1e] = solo_e1[fi][sy:sy+h_c, sx:sx+w_c]

            # Determine depth order randomly
            front = rng.integers(0, 2)
            if front == 0:
                canvas = solo_e0[fi].copy()
                mask_bg = (m0_r == 0)
                canvas[mask_bg] = s1_shifted[mask_bg]
            else:
                canvas = s1_shifted.copy()
                mask_bg = (m1_shifted == 0)
                canvas[mask_bg] = solo_e0[fi][mask_bg]

            shifted_frames.append(canvas)
            shifted_depths.append((front, 1 - front))

            # Downsample masks to match entity_masks spatial resolution
            m0_ds = np.array(Image.fromarray(m0_r * 255).resize((hw, hw), Image.BILINEAR)) / 255.0
            m1_ds = np.array(Image.fromarray(m1_shifted * 255).resize((hw, hw), Image.BILINEAR)) / 255.0
            shifted_masks.append(np.stack([m0_ds.flatten(), m1_ds.flatten()], axis=0))

        masks_arr = np.stack(shifted_masks, axis=0)  # (T, 2, S)
        ov = float((masks_arr[:, 0, :] * masks_arr[:, 1, :]).mean())

        if ov < min_accept:
            continue

        target_ov = (overlap_min + overlap_max) / 2.0
        err = abs(ov - target_ov)
        if err < best_err:
            best_err = err
            best = (
                np.stack(shifted_frames, axis=0),
                shifted_depths,
                masks_arr,
            )

    return best


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def evaluate_val_set_p56(
    pipe,
    manager:   MultiBlockSlotManagerP56,
    dataset,
    val_idx:   list,
    device:    str,
    t_fixed:   int = 200,
) -> dict:
    """
    Evaluate Phase56 on validation set.

    Runs 3-pass composition and computes IoU metrics by comparing
    the composited noise against the original noise.
    """
    manager.eval()
    composite_mses = []
    masked_e0_mses = []
    masked_e1_mses = []

    for vi in val_idx:
        try:
            sample = dataset[vi]
            if len(sample) >= 8:
                frames_np, _, depth_orders, meta, entity_masks = sample[0], sample[1], sample[2], sample[3], sample[4]
                visible_masks = sample[5] if sample[5] is not None else entity_masks.astype(np.float32)
            else:
                frames_np, _, depth_orders, meta, entity_masks = sample[:5]
                visible_masks = entity_masks.astype(np.float32)

            full_prompt = get_entity_token_positions(pipe, meta)[2]
            bg_prompt = get_bg_prompt(meta, full_prompt)
            prompt_e0, prompt_e1 = get_entity_prompts(meta, full_prompt)

            latents = encode_frames_to_latents(pipe, frames_np, device)
            noise   = torch.randn_like(latents)
            t_tensor = torch.tensor([t_fixed], device=device).long()
            noisy   = pipe.scheduler.add_noise(latents, noise, t_tensor)

            enc_full = encode_text(pipe, bg_prompt, device)
            enc_e0   = encode_text(pipe, prompt_e0, device)
            enc_e1   = encode_text(pipe, prompt_e1, device)

            T_frames = min(frames_np.shape[0], entity_masks.shape[0])

            # 3-pass forward
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                manager.set_entity_mode('bg')
                noise_bg = pipe.unet(noisy, t_tensor, encoder_hidden_states=enc_full).sample

                manager.set_entity_mode('e0')
                noise_e0 = pipe.unet(noisy, t_tensor, encoder_hidden_states=enc_e0).sample

                manager.set_entity_mode('e1')
                noise_e1 = pipe.unet(noisy, t_tensor, encoder_hidden_states=enc_e1).sample

            # Downsample masks to latent resolution
            lat_h, lat_w = latents.shape[3], latents.shape[4]
            masks_lat = downsample_masks_to_latent(entity_masks[:T_frames], lat_h)
            m0 = masks_lat[:, 0:1].permute(1, 0, 2, 3).unsqueeze(0).float().to(device)  # (1, 1, T, H, W)
            m1 = masks_lat[:, 1:2].permute(1, 0, 2, 3).unsqueeze(0).float().to(device)
            vis_lat = downsample_masks_to_latent(visible_masks[:T_frames], lat_h)
            v0 = vis_lat[:, 0:1].permute(1, 0, 2, 3).unsqueeze(0).float().to(device)
            v1 = vis_lat[:, 1:2].permute(1, 0, 2, 3).unsqueeze(0).float().to(device)

            # Only use T_frames
            noise_bg_t = noise_bg[:, :, :T_frames].float()
            noise_e0_t = noise_e0[:, :, :T_frames].float()
            noise_e1_t = noise_e1[:, :, :T_frames].float()
            noise_t    = noise[:, :, :T_frames].float()
            delta_target = (noise_t - noise_bg_t.detach())
            delta_e0_t = noise_e0_t - noise_bg_t
            delta_e1_t = noise_e1_t - noise_bg_t

            composite = transmittance_composite_absolute(
                noise_e0_t, noise_e1_t, noise_bg_t, m0, m1, depth_orders[:T_frames])

            l_comp = F.mse_loss(composite, noise_t)
            l_me0  = masked_mse(delta_e0_t, delta_target, v0)
            l_me1  = masked_mse(delta_e1_t, delta_target, v1)

            composite_mses.append(float(l_comp.item()))
            masked_e0_mses.append(float(l_me0.item()))
            masked_e1_mses.append(float(l_me1.item()))

        except Exception as e:
            print(f"  [val warn] idx={vi}: {e}", flush=True)
            continue

    n = max(len(composite_mses), 1)
    val_composite_mse = sum(composite_mses) / n if composite_mses else 999.0
    val_masked_e0     = sum(masked_e0_mses) / n if masked_e0_mses else 999.0
    val_masked_e1     = sum(masked_e1_mses) / n if masked_e1_mses else 999.0
    val_total         = val_composite_mse + 0.5 * val_masked_e0 + 0.5 * val_masked_e1

    # val_score: lower MSE is better, so invert for "higher is better" score
    val_score = 1.0 / (1.0 + val_total)

    return {
        "val_score":        val_score,
        "val_composite_mse": val_composite_mse,
        "val_masked_e0":    val_masked_e0,
        "val_masked_e1":    val_masked_e1,
        "val_total_loss":   val_total,
        "n_samples":        len(composite_mses),
    }


# =============================================================================
# Main training function
# =============================================================================

def train_phase56(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    debug_dir = Path(args.debug_dir)
    save_dir  = Path(args.save_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Pipeline ─────────────────────────────────────────────────────────
    print("[Phase 56] Loading pipeline...", flush=True)
    pipe = load_pipeline(device=device)
    pipe.scheduler.set_timesteps(50, device=device)

    # Enable gradient checkpointing for memory efficiency (3 UNet passes)
    pipe.unet.enable_gradient_checkpointing()

    # ── Dataset ──────────────────────────────────────────────────────────
    print(f"[Phase 56] Loading dataset from {args.data_root}...", flush=True)
    dataset = ObjaverseDatasetPhase40(args.data_root, n_frames=args.n_frames)
    print(f"  dataset size: {len(dataset)}", flush=True)

    overlap_scores = compute_dataset_overlap_scores(dataset)
    train_idx, val_idx = split_train_val(overlap_scores, val_frac=args.val_frac,
                                         min_val=MIN_VAL_SAMPLES)
    sample_weights = make_sampling_weights(train_idx, overlap_scores)
    print(f"  train={len(train_idx)} val={len(val_idx)}", flush=True)

    # ── Inject Phase56 processors ────────────────────────────────────────
    inject_keys = (
        args.inject_keys.split(",") if args.inject_keys else DEFAULT_INJECT_KEYS)
    print(f"[Phase 56] Injecting processors into {inject_keys}...", flush=True)

    procs, orig_procs = inject_multi_block_entity_slot_p56(
        pipe,
        inject_keys=inject_keys,
        adapter_rank=args.adapter_rank,
        lora_rank=args.lora_rank,
    )
    manager = MultiBlockSlotManagerP56(procs, inject_keys)

    # ── Restore from checkpoint ──────────────────────────────────────────
    if args.ckpt and Path(args.ckpt).exists():
        print(f"[Phase 56] Loading checkpoint: {args.ckpt}", flush=True)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        restore_multiblock_state_p56(manager, ckpt, device)
    else:
        print("[Phase 56] No checkpoint, starting from scratch.", flush=True)

    # Move all processors to device
    for p in procs:
        p.to(device)

    # Freeze base model
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    # ── Epoch 0 no-op validation ─────────────────────────────────────────
    print("\n[Phase 56] Epoch 0 no-op validation...", flush=True)
    manager.eval()
    val_m0 = evaluate_val_set_p56(pipe, manager, dataset, val_idx, device,
                                  t_fixed=args.t_max // 2)
    print(f"  [epoch0] val_score={val_m0['val_score']:.4f}  "
          f"composite_mse={val_m0['val_composite_mse']:.4f}  "
          f"masked_e0={val_m0['val_masked_e0']:.4f}  "
          f"masked_e1={val_m0['val_masked_e1']:.4f}", flush=True)
    manager.train()

    # ── Training loop ────────────────────────────────────────────────────
    history               = []
    best_val_score        = -1.0
    best_selection_score  = -1.0
    best_epoch            = -1
    total_epochs   = args.stage_a_epochs + args.stage_b_epochs

    for epoch in range(total_epochs):
        is_stage_a = (epoch < args.stage_a_epochs)

        # ── Optimizer setup ──────────────────────────────────────────────
        if epoch == 0 and args.stage_a_epochs > 0:
            print(f"\n[Phase 56] Stage A ({args.stage_a_epochs} epochs): "
                  f"entity LoRA + adapters only", flush=True)
            # Freeze shared LoRA in stage A
            for p in manager.shared_lora_params():
                p.requires_grad_(False)
            for p in manager.entity_lora_params():
                p.requires_grad_(True)
            for p in manager.adapter_params():
                p.requires_grad_(True)

            optimizer = optim.AdamW([
                {"params": manager.entity_lora_params(),
                 "lr": args.lr_entity_lora, "name": "entity_lora"},
                {"params": manager.adapter_params(),
                 "lr": args.lr_adapter, "name": "adapters"},
            ], weight_decay=1e-4)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.stage_a_epochs), eta_min=1e-6)

        if (epoch == 0 and args.stage_a_epochs == 0) or (epoch == args.stage_a_epochs and epoch > 0):
            print(f"\n[Phase 56] Stage B ({args.stage_b_epochs} epochs): "
                  f"entity LoRA + global LoRA + adapters", flush=True)
            # Unfreeze shared LoRA
            for p in manager.shared_lora_params():
                p.requires_grad_(True)
            for p in manager.entity_lora_params():
                p.requires_grad_(True)
            for p in manager.adapter_params():
                p.requires_grad_(True)

            optimizer = optim.AdamW([
                {"params": manager.entity_lora_params(),
                 "lr": args.lr_entity_lora * 0.5, "name": "entity_lora"},
                {"params": manager.shared_lora_params(),
                 "lr": args.lr_global_lora, "name": "global_lora"},
                {"params": manager.adapter_params(),
                 "lr": args.lr_adapter, "name": "adapters"},
            ], weight_decay=1e-4)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.stage_b_epochs, eta_min=1e-6)

        manager.train()
        epoch_losses = {"total": [], "composite": [], "masked_e0": [], "masked_e1": [],
                        "leak_e0": [], "leak_e1": [], "bg_fg": [], "solo": []}
        n_collision_aug = 0
        n_collision_attempt = 0

        chosen = np.random.choice(
            len(train_idx), size=args.steps_per_epoch, replace=True, p=sample_weights)
        step_indices = [train_idx[ci] for ci in chosen]

        collision_prob = (args.collision_aug_prob_a
                          if is_stage_a else args.collision_aug_prob_b)

        for batch_idx, data_idx in enumerate(step_indices):
            sample = dataset[data_idx]
            if len(sample) >= 8:
                frames_np, _, depth_orders, meta, entity_masks = (
                    sample[0], sample[1], sample[2], sample[3], sample[4])
                visible_masks = sample[5] if sample[5] is not None else entity_masks.astype(np.float32)
            else:
                frames_np, _, depth_orders, meta, entity_masks = sample[:5]
                visible_masks = entity_masks.astype(np.float32)

            used_collision_aug = False
            # ── Collision augmentation ───────────────────────────────────
            if (collision_prob > 0.0 and
                    len(sample) >= 8 and
                    np.random.rand() < collision_prob):
                n_collision_attempt += 1
                aug = _try_collision_augment(
                    dataset, data_idx, sample,
                    overlap_min=args.collision_ov_min,
                    overlap_max=args.collision_ov_max,
                    min_accept=args.collision_min_accept)
                if aug is not None:
                    frames_np, depth_orders, entity_masks = aug
                    visible_masks = entity_masks.astype(np.float32)
                    n_collision_aug += 1
                    used_collision_aug = True

            # ── Encode frames and prepare ────────────────────────────────
            _, _, full_prompt = get_entity_token_positions(pipe, meta)
            bg_prompt = get_bg_prompt(meta, full_prompt)
            prompt_e0, prompt_e1 = get_entity_prompts(meta, full_prompt)

            with torch.no_grad():
                latents = encode_frames_to_latents(pipe, frames_np, device)
            noise = torch.randn_like(latents)
            t     = torch.randint(0, args.t_max, (1,), device=device).long()
            noisy = pipe.scheduler.add_noise(latents, noise, t)

            T_frames = min(frames_np.shape[0], entity_masks.shape[0], len(depth_orders))

            # ── Text embeddings (NOT expanded for AnimateDiff) ───────────
            enc_full = encode_text(pipe, bg_prompt, device)     # (1, 77, 768)
            enc_e0   = encode_text(pipe, prompt_e0, device)     # (1, 77, 768)
            enc_e1   = encode_text(pipe, prompt_e1, device)     # (1, 77, 768)

            # ── 3-pass forward ───────────────────────────────────────────
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                manager.set_entity_mode('bg')
                noise_pred_bg = pipe.unet(
                    noisy, t, encoder_hidden_states=enc_full).sample
                manager.set_entity_mode('e0')
                noise_pred_e0 = pipe.unet(
                    noisy, t, encoder_hidden_states=enc_e0).sample
                manager.set_entity_mode('e1')
                noise_pred_e1 = pipe.unet(
                    noisy, t, encoder_hidden_states=enc_e1).sample

            # ── Masks at latent resolution ──────────────────────────────
            lat_h = latents.shape[3]
            # Full entity masks (for composite) — includes occluded regions
            masks_full_lat = downsample_masks_to_latent(
                entity_masks[:T_frames], lat_h)
            m0_full = masks_full_lat[:, 0:1].permute(1, 0, 2, 3).unsqueeze(0).float().to(device)
            m1_full = masks_full_lat[:, 1:2].permute(1, 0, 2, 3).unsqueeze(0).float().to(device)
            # Visible masks (for masked loss) — only actually visible pixels
            vis_lat = downsample_masks_to_latent(
                visible_masks[:T_frames], lat_h)
            v0 = vis_lat[:, 0:1].permute(1, 0, 2, 3).unsqueeze(0).float().to(device)
            v1 = vis_lat[:, 1:2].permute(1, 0, 2, 3).unsqueeze(0).float().to(device)

            # Trim to T_frames
            noise_bg_t = noise_pred_bg[:, :, :T_frames].float()
            noise_e0_t = noise_pred_e0[:, :, :T_frames].float()
            noise_e1_t = noise_pred_e1[:, :, :T_frames].float()
            noise_t    = noise[:, :, :T_frames].float()
            delta_target = (noise_t - noise_bg_t.detach())
            delta_e0_t = noise_e0_t - noise_bg_t
            delta_e1_t = noise_e1_t - noise_bg_t

            # ── Residual transmittance composition (uses full masks) ────
            noise_composite = transmittance_composite_absolute(
                noise_e0_t, noise_e1_t, noise_bg_t,
                m0_full, m1_full, depth_orders[:T_frames])

            # ── Losses ──────────────────────────────────────────────────
            # 1. Composite: composited noise matches GT noise
            l_composite = F.mse_loss(noise_composite, noise_t)

            # 2. Masked entity: each entity's noise correct in its VISIBLE region
            #    (not full mask — prevents overlap double-supervision)
            l_masked_e0 = masked_mse(delta_e0_t, delta_target, v0)
            l_masked_e1 = masked_mse(delta_e1_t, delta_target, v1)

            # 3. Outside-mask leakage: entity delta must be zero outside FULL mask
            l_leak_e0 = outside_mask_suppression(noise_e0_t, noise_bg_t.detach(), m0_full)
            l_leak_e1 = outside_mask_suppression(noise_e1_t, noise_bg_t.detach(), m1_full)

            # 4. BG outside-foreground fit: anchor the shared baseline on true background
            union_fg = (m0_full + m1_full).clamp(0, 1)
            l_bg_fg = masked_mse(noise_bg_t, noise_t, 1.0 - union_fg)

            # Solo entity anchor (when solo renders available)
            l_solo = torch.tensor(0.0, device=device)
            if (len(sample) >= 8 and args.la_solo > 0.0
                    and sample[6] is not None and sample[7] is not None):
                solo_e0_np, solo_e1_np = sample[6], sample[7]
                try:
                    with torch.no_grad():
                        solo0_lat = encode_frames_to_latents(pipe, solo_e0_np, device)
                        solo1_lat = encode_frames_to_latents(pipe, solo_e1_np, device)
                    solo0_noise = torch.randn_like(solo0_lat)
                    solo1_noise = torch.randn_like(solo1_lat)
                    solo0_noisy = pipe.scheduler.add_noise(solo0_lat, solo0_noise, t)
                    solo1_noisy = pipe.scheduler.add_noise(solo1_lat, solo1_noise, t)

                    manager.set_entity_mode('e0')
                    l_solo_e0 = solo_entity_anchor(
                        pipe, solo0_noisy, solo0_noise, t, enc_e0)
                    manager.set_entity_mode('e1')
                    l_solo_e1 = solo_entity_anchor(
                        pipe, solo1_noisy, solo1_noise, t, enc_e1)
                    l_solo = 0.5 * (l_solo_e0 + l_solo_e1)
                except Exception as e:
                    print(f"  [warn] solo anchor: {e}", flush=True)

            loss = (args.la_composite * l_composite
                    + args.la_masked * l_masked_e0
                    + args.la_masked * l_masked_e1
                    + args.la_leak * l_leak_e0
                    + args.la_leak * l_leak_e1
                    + args.la_bg_fg * l_bg_fg
                    + args.la_solo * l_solo)

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss ep={epoch} step={batch_idx} -> skip",
                      flush=True)
                continue

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(manager.entity_lora_params(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(manager.adapter_params(), max_norm=1.0)
            if not is_stage_a:
                torch.nn.utils.clip_grad_norm_(manager.shared_lora_params(), max_norm=0.5)

            optimizer.step()

            # Log
            epoch_losses["total"].append(float(loss.item()))
            epoch_losses["composite"].append(float(l_composite.item()))
            epoch_losses["masked_e0"].append(float(l_masked_e0.item()))
            epoch_losses["masked_e1"].append(float(l_masked_e1.item()))
            epoch_losses["leak_e0"].append(float(l_leak_e0.item()))
            epoch_losses["leak_e1"].append(float(l_leak_e1.item()))
            epoch_losses["bg_fg"].append(float(l_bg_fg.item()))
            epoch_losses["solo"].append(float(l_solo.item()))

        lr_scheduler.step()

        # ── Epoch summary ────────────────────────────────────────────────
        avg = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
        stage_lbl = "A" if is_stage_a else "B"
        print(f"[Phase 56][Stage {stage_lbl}] epoch {epoch:03d}/{total_epochs-1}  "
              f"loss={avg['total']:.4f}  composite={avg['composite']:.4f}  "
              f"masked_e0={avg['masked_e0']:.4f}  masked_e1={avg['masked_e1']:.4f}  "
              f"leak_e0={avg['leak_e0']:.4f}  leak_e1={avg['leak_e1']:.4f}  "
              f"bg_fg={avg['bg_fg']:.4f}  solo={avg['solo']:.4f}  "
              f"aug={n_collision_aug}/{n_collision_attempt}",
              flush=True)

        # ── Validation ───────────────────────────────────────────────────
        should_eval = (
            (epoch % args.eval_every == 0) or
            (epoch == total_epochs - 1)
        )

        if should_eval:
            manager.eval()
            probe_rollout_mse = None
            probe_rollout_score = None
            selection_score = None

            val_m = evaluate_val_set_p56(
                pipe, manager, dataset, val_idx, device,
                t_fixed=args.t_max // 2)

            vs = val_m["val_score"]
            print(f"  [val] val_score={vs:.4f}  "
                  f"composite_mse={val_m['val_composite_mse']:.4f}  "
                  f"masked_e0={val_m['val_masked_e0']:.4f}  "
                  f"masked_e1={val_m['val_masked_e1']:.4f}  "
                  f"n_samples={val_m['n_samples']}",
                  flush=True)

            # GIF generation — real 3-pass composited rollout
            try:
                probe = dataset[val_idx[0]]
                probe_meta = probe[3]
                probe_masks = probe[4]  # entity_masks
                probe_depths = probe[2]  # depth_orders
                _, _, probe_prompt = get_entity_token_positions(pipe, probe_meta)
                probe_bg_prompt = get_bg_prompt(probe_meta, probe_prompt)
                probe_e0_prompt, probe_e1_prompt = get_entity_prompts(
                    probe_meta, probe_prompt)

                # Run composited denoising loop
                gen = torch.Generator(device=device).manual_seed(args.eval_seed)
                latent_shape = (1, 4, args.n_frames, args.height // 8, args.width // 8)
                latents = torch.randn(latent_shape, generator=gen, device=device,
                                      dtype=torch.float16)
                latents = latents * pipe.scheduler.init_noise_sigma

                pipe.scheduler.set_timesteps(args.n_steps, device=device)

                enc_bg_eval = encode_text(pipe, probe_bg_prompt, device)
                enc_e0_eval = encode_text(pipe, probe_e0_prompt, device)
                enc_e1_eval = encode_text(pipe, probe_e1_prompt, device)
                neg_prompt = "blurry, deformed, extra limbs, watermark"
                enc_uncond  = encode_text(pipe, neg_prompt, device)
                guidance_scale = 7.5

                T_probe = min(args.n_frames, probe_masks.shape[0], len(probe_depths))
                lat_h = args.height // 8
                pm_lat = downsample_masks_to_latent(probe_masks[:T_probe], lat_h)
                pm0 = pm_lat[:, 0:1].permute(1, 0, 2, 3).unsqueeze(0).float().to(device)
                pm1 = pm_lat[:, 1:2].permute(1, 0, 2, 3).unsqueeze(0).float().to(device)
                # Pad if T_probe < n_frames
                if T_probe < args.n_frames:
                    pad = args.n_frames - T_probe
                    pm0 = F.pad(pm0, (0,0,0,0,0,pad), mode='replicate')
                    pm1 = F.pad(pm1, (0,0,0,0,0,pad), mode='replicate')

                def _cfg_unet(latents_in, step_t, enc_cond):
                    """UNet forward with classifier-free guidance."""
                    lat2 = torch.cat([latents_in]*2, dim=0)
                    lat2 = pipe.scheduler.scale_model_input(lat2, step_t)
                    enc2 = torch.cat([enc_uncond, enc_cond], dim=0)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred = pipe.unet(lat2, step_t, encoder_hidden_states=enc2).sample
                    uncond_p, cond_p = pred.chunk(2, dim=0)
                    return uncond_p + guidance_scale * (cond_p - uncond_p)

                with torch.no_grad():
                    all_steps = list(pipe.scheduler.timesteps)
                    n_layout = 0  # Full composite from step 0
                    for si, step_t in enumerate(all_steps):
                        manager.set_entity_mode('bg')
                        n_bg = _cfg_unet(latents, step_t, enc_bg_eval)

                        if si < n_layout:
                            n_final = n_bg
                        else:
                            manager.set_entity_mode('e0')
                            n_e0 = _cfg_unet(latents, step_t, enc_e0_eval)
                            manager.set_entity_mode('e1')
                            n_e1 = _cfg_unet(latents, step_t, enc_e1_eval)

                            n_final = transmittance_composite_absolute(
                                n_e0.float(), n_e1.float(), n_bg.float(),
                                pm0, pm1, probe_depths[:args.n_frames])

                        latents = pipe.scheduler.step(
                            n_final.half(), step_t, latents, return_dict=False)[0]

                # Decode to frames
                latents_4d = latents[0].permute(1, 0, 2, 3).half()  # (T, 4, H, W) fp16
                scale = pipe.vae.config.scaling_factor
                comp_frames = []
                for fi in range(args.n_frames):
                    z_in = (latents_4d[fi:fi+1] / scale).half()
                    with torch.no_grad():
                        decoded = pipe.vae.decode(z_in).sample
                    img = ((decoded.float() / 2 + 0.5).clamp(0, 1)[0]
                           .permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    comp_frames.append(img)

                comp_gif = debug_dir / f"eval_epoch{epoch:03d}_composite.gif"
                iio2.mimwrite(str(comp_gif), comp_frames, fps=8, loop=0)
                print(f"  [gif] COMPOSITE rollout saved: {comp_gif}", flush=True)

                gt_probe = np.asarray(probe[0][:args.n_frames], dtype=np.float32) / 255.0
                pred_probe = np.asarray(comp_frames[:args.n_frames], dtype=np.float32) / 255.0
                T_cmp = min(len(gt_probe), len(pred_probe))
                if T_cmp > 0:
                    probe_rollout_mse = float(np.mean((pred_probe[:T_cmp] - gt_probe[:T_cmp]) ** 2))
                    probe_rollout_score = 1.0 / (1.0 + probe_rollout_mse)
                    print(f"  [rollout] probe_rgb_mse={probe_rollout_mse:.4f}  "
                          f"probe_score={probe_rollout_score:.4f}", flush=True)

                # Also save individual e0/e1 GIFs for comparison
                for eidx, emode in enumerate(['e0', 'e1']):
                    manager.set_entity_mode(emode)
                    prompt_e = [probe_e0_prompt, probe_e1_prompt][eidx]
                    gen_e = torch.Generator(device=device).manual_seed(args.eval_seed)
                    out_e = pipe(prompt=prompt_e, num_frames=args.n_frames,
                                num_inference_steps=args.n_steps,
                                height=args.height, width=args.width,
                                generator=gen_e, output_type="np")
                    frames_e = (out_e.frames[0] * 255).astype(np.uint8)
                    gif_e = debug_dir / f"eval_epoch{epoch:03d}_{emode}.gif"
                    iio2.mimwrite(str(gif_e), frames_e, fps=8, loop=0)
                    print(f"  [gif] {emode} saved: {gif_e}", flush=True)

            except Exception as e:
                print(f"  [warn] GIF failed: {e}", flush=True)
                import traceback; traceback.print_exc()

            selection_score = vs
            if probe_rollout_score is not None:
                selection_score = 0.75 * vs + 0.25 * probe_rollout_score
                print(f"  [select] score={selection_score:.4f}  "
                      f"(val={vs:.4f}, rollout={probe_rollout_score:.4f})", flush=True)

            # Save history
            history.append({
                "epoch": epoch,
                "stage": stage_lbl,
                **val_m,
                **avg,
                "collision_aug_count": n_collision_aug,
                "probe_rollout_mse": probe_rollout_mse,
                "probe_rollout_score": probe_rollout_score,
                "selection_score": selection_score,
            })

            # ── Checkpoint ───────────────────────────────────────────────
            ckpt_data = {
                "epoch":        epoch,
                "stage":        stage_lbl,
                "val_score":    vs,
                "selection_score": selection_score,
                "inject_keys":  inject_keys,
                "adapter_rank": args.adapter_rank,
                "lora_rank":    args.lora_rank,
                "procs_state":  [
                    {
                        "lora_k":       p.lora_k.state_dict(),
                        "lora_v":       p.lora_v.state_dict(),
                        "lora_out":     p.lora_out.state_dict(),
                        "lora_k_e0":    p.lora_k_e0.state_dict(),
                        "lora_k_e1":    p.lora_k_e1.state_dict(),
                        "lora_v_e0":    p.lora_v_e0.state_dict(),
                        "lora_v_e1":    p.lora_v_e1.state_dict(),
                        "lora_out_e0":  p.lora_out_e0.state_dict(),
                        "lora_out_e1":  p.lora_out_e1.state_dict(),
                        "slot0_adapter": p.slot0_adapter.state_dict(),
                        "slot1_adapter": p.slot1_adapter.state_dict(),
                    }
                    for p in procs
                ],
            }
            torch.save(ckpt_data, str(save_dir / "latest.pt"))

            if vs > best_val_score:
                best_val_score = vs
            if selection_score > best_selection_score:
                best_selection_score = selection_score
                best_epoch = epoch
                torch.save(ckpt_data, str(save_dir / "best.pt"))
                print(f"  * best epoch={best_epoch} val_score={vs:.4f} "
                      f" selection={selection_score:.4f} "
                      f"-> {save_dir}/best.pt", flush=True)

            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            manager.train()

    print(f"\n[Phase 56] Done. best epoch={best_epoch} val_score={best_val_score:.4f}",
          flush=True)


# =============================================================================
# argparse
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Phase 56: T-COMP Style 3-Pass Noise Composition Training")

    p.add_argument("--ckpt",       type=str, default="",
                   help="Optional checkpoint to resume from (Phase52-era or Phase56)")
    p.add_argument("--data-root",  type=str, default="toy/data_objaverse")
    p.add_argument("--save-dir",   type=str, default="checkpoints/phase56")
    p.add_argument("--debug-dir",  type=str, default="outputs/phase56_debug")

    p.add_argument("--stage-a-epochs",   type=int,   default=DEFAULT_STAGE_A_EPOCHS)
    p.add_argument("--stage-b-epochs",   type=int,   default=DEFAULT_STAGE_B_EPOCHS)
    p.add_argument("--steps-per-epoch",  type=int,   default=DEFAULT_STEPS_PER_EPOCH)
    p.add_argument("--n-frames",         type=int,   default=8)
    p.add_argument("--n-steps",          type=int,   default=20)
    p.add_argument("--t-max",            type=int,   default=300)
    p.add_argument("--height",           type=int,   default=256)
    p.add_argument("--width",            type=int,   default=256)
    p.add_argument("--adapter-rank",     type=int,   default=DEFAULT_ADAPTER_RANK)
    p.add_argument("--lora-rank",        type=int,   default=DEFAULT_LORA_RANK)
    p.add_argument("--inject-keys",      type=str,   default=None)

    # Learning rates
    p.add_argument("--lr-entity-lora",   type=float, default=DEFAULT_LR_ENTITY_LORA)
    p.add_argument("--lr-global-lora",   type=float, default=DEFAULT_LR_GLOBAL_LORA)
    p.add_argument("--lr-adapter",       type=float, default=DEFAULT_LR_ADAPTER)

    # Loss weights
    p.add_argument("--la-composite",     type=float, default=DEFAULT_LA_COMPOSITE)
    p.add_argument("--la-masked",        type=float, default=DEFAULT_LA_MASKED)
    p.add_argument("--la-leak",          type=float, default=0.3)
    p.add_argument("--la-bg-fg",         type=float, default=0.3)
    p.add_argument("--la-solo",          type=float, default=0.2)

    # Collision augmentation
    p.add_argument("--collision-aug-prob-a", type=float, default=DEFAULT_COLLISION_AUG_PROB_A)
    p.add_argument("--collision-aug-prob-b", type=float, default=DEFAULT_COLLISION_AUG_PROB_B)
    p.add_argument("--collision-ov-min",     type=float, default=DEFAULT_COLLISION_OV_MIN)
    p.add_argument("--collision-ov-max",     type=float, default=DEFAULT_COLLISION_OV_MAX)
    p.add_argument("--collision-min-accept", type=float, default=DEFAULT_COLLISION_MIN_ACCEPT)

    # Eval
    p.add_argument("--val-frac",     type=float, default=VAL_FRAC)
    p.add_argument("--eval-every",   type=int,   default=2)
    p.add_argument("--eval-seed",    type=int,   default=42)
    p.add_argument("--seed",         type=int,   default=42)

    args = p.parse_args()
    train_phase56(args)


if __name__ == "__main__":
    main()
