"""
Phase 61 — Depth-Layered Volume Diffusion Training
====================================================

Multi-entity video diffusion with depth-layered volume composition.
Each entity gets K depth bins with per-bin alpha + feature delta,
composed via NeRF-style front-to-back rendering.

3-stage training:
  Stage A (8 epochs): volume heads bootstrap
    Train: volume heads + slot adapters
    Losses: L_alpha_volume + L_visible_ownership + L_depth_expected + 0.1*L_comp

  Stage B (8 epochs): + shared LoRA
    Train: + LoRA K/V/Out
    Losses: all 4 with full weights

  Stage C (9 epochs): joint fine-tune
    Train: everything
    Losses: all 4, collision-heavy sampling
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import imageio.v2 as iio2
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.phase61_layered_volume import (
    Phase61Manager,
    inject_phase61,
    restore_phase61,
    build_depth_bin_targets,
    DEFAULT_INJECT_KEYS,
)
from models.phase61_losses import (
    loss_composite,
    loss_alpha_volume,
    loss_visible_ownership,
    loss_depth_expected,
    loss_feature_orthogonality,
)
from models.entity_slot_phase40 import (
    compute_visible_iou_e0,
    compute_visible_iou_e1,
)
from scripts.run_animatediff import load_pipeline
from scripts.train_animatediff_vca import encode_frames_to_latents
from scripts.train_phase35 import get_entity_token_positions
from scripts.train_phase39 import (
    compute_dataset_overlap_scores,
    split_train_val,
    make_sampling_weights,
)
from scripts.generate_solo_renders import ObjaverseDatasetPhase40


# =============================================================================
# Hyperparameters
# =============================================================================
DEFAULT_STAGE_A_EPOCHS  = 8
DEFAULT_STAGE_B_EPOCHS  = 8
DEFAULT_STAGE_C_EPOCHS  = 9
DEFAULT_STEPS_PER_EPOCH = 20
DEFAULT_LR_VOLUME       = 5e-5
DEFAULT_LR_ADAPTER      = 2e-5
DEFAULT_LR_LORA         = 5e-6
DEFAULT_ADAPTER_RANK    = 64
DEFAULT_LORA_RANK       = 4
DEFAULT_DEPTH_BINS      = 2

# Loss weights
DEFAULT_LA_COMP         = 1.0
DEFAULT_LA_ALPHA_VOL    = 1.0
DEFAULT_LA_OWN          = 1.0
DEFAULT_LA_DEPTH_EXP    = 0.5
DEFAULT_LA_ORTHO        = 0.1

# Collision augmentation
DEFAULT_COLLISION_PROB_A = 0.5
DEFAULT_COLLISION_PROB_B = 0.6
DEFAULT_COLLISION_PROB_C = 0.8
DEFAULT_COLLISION_OV_MIN = 0.005
DEFAULT_COLLISION_OV_MAX = 0.25
DEFAULT_COLLISION_MIN_ACC = 0.003

VAL_FRAC     = 0.2
MIN_VAL_SAMPLES = 3


# =============================================================================
# Helpers
# =============================================================================

def encode_text(pipe, text: str, device: str) -> torch.Tensor:
    """Encode text to CLIP hidden states: (1, 77, 768) fp16."""
    tok = pipe.tokenizer(
        text, return_tensors="pt", padding="max_length",
        max_length=pipe.tokenizer.model_max_length, truncation=True,
    ).to(device)
    with torch.no_grad():
        enc = pipe.text_encoder(**tok).last_hidden_state.half()
    return enc


def masks_to_feature_space(
    entity_masks: np.ndarray,  # (T, 2, S) where S = H_mask * W_mask
    spatial_dim:  int,
) -> torch.Tensor:
    """Reshape GT masks to match attention feature spatial dimension.
    Returns: (T, 2, spatial_dim) float32."""
    T, N, S = entity_masks.shape
    H_mask = int(S ** 0.5)
    target_hw = int(round(spatial_dim ** 0.5))

    masks_4d = torch.from_numpy(entity_masks.astype(np.float32))
    masks_4d = masks_4d.view(T, N, H_mask, H_mask)

    if H_mask != target_hw:
        masks_4d = F.interpolate(
            masks_4d.float(), size=(target_hw, target_hw),
            mode='bilinear', align_corners=False)

    return masks_4d.reshape(T, N, -1).clamp(0.0, 1.0)


def _try_collision_augment(
    dataset, sample_idx: int, sample,
    overlap_min: float = 0.08, overlap_max: float = 0.25,
    max_shift: int = 96, max_tries: int = 24, min_accept: float = 0.04,
):
    """Build collision-heavy synthetic sample from solo frames + raw masks."""
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

        shifted_frames, shifted_masks, shifted_depths = [], [], []
        for fi in range(T):
            H, W = solo_e0[fi].shape[:2]
            m0_r, m1_r = raw0[fi], raw1[fi]

            m1_shifted = np.zeros_like(m1_r)
            s1_shifted = np.zeros_like(solo_e1[fi])
            y1s, y1e = max(0, dy), min(H, H + dy)
            x1s, x1e = max(0, dx), min(W, W + dx)
            sy, sx = max(0, -dy), max(0, -dx)
            h_c, w_c = y1e - y1s, x1e - x1s
            if h_c > 0 and w_c > 0:
                m1_shifted[y1s:y1e, x1s:x1e] = m1_r[sy:sy+h_c, sx:sx+w_c]
                s1_shifted[y1s:y1e, x1s:x1e] = solo_e1[fi][sy:sy+h_c, sx:sx+w_c]

            front = rng.integers(0, 2)
            if front == 0:
                canvas = solo_e0[fi].copy()
                canvas[m0_r == 0] = s1_shifted[m0_r == 0]
            else:
                canvas = s1_shifted.copy()
                canvas[m1_shifted == 0] = solo_e0[fi][m1_shifted == 0]

            shifted_frames.append(canvas)
            shifted_depths.append((front, 1 - front))

            m0_ds = np.array(Image.fromarray(m0_r * 255).resize(
                (hw, hw), Image.BILINEAR)) / 255.0
            m1_ds = np.array(Image.fromarray(m1_shifted * 255).resize(
                (hw, hw), Image.BILINEAR)) / 255.0
            shifted_masks.append(np.stack(
                [m0_ds.flatten(), m1_ds.flatten()], axis=0))

        masks_arr = np.stack(shifted_masks, axis=0)
        ov = float((masks_arr[:, 0, :] * masks_arr[:, 1, :]).mean())

        if ov < min_accept:
            continue

        target_ov = (overlap_min + overlap_max) / 2.0
        err = abs(ov - target_ov)
        if err < best_err:
            best_err = err
            best = (np.stack(shifted_frames, axis=0), shifted_depths, masks_arr)

    return best


def _get_entity_tokens_with_fallback(pipe, meta, device):
    """Get entity token positions with keyword-based fallback."""
    toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)

    if not toks_e0 or not toks_e1:
        kw0 = str(meta.get("keyword0", "cat"))
        kw1 = str(meta.get("keyword1", "dog"))
        tok_ids = pipe.tokenizer.encode(full_prompt)
        kw0_ids = pipe.tokenizer.encode(kw0, add_special_tokens=False)
        kw1_ids = pipe.tokenizer.encode(kw1, add_special_tokens=False)
        if not toks_e0 and kw0_ids:
            for i in range(len(tok_ids) - len(kw0_ids) + 1):
                if tok_ids[i:i+len(kw0_ids)] == kw0_ids:
                    toks_e0 = list(range(i, i + len(kw0_ids)))
                    break
        if not toks_e1 and kw1_ids:
            for i in range(len(tok_ids) - len(kw1_ids) + 1):
                if tok_ids[i:i+len(kw1_ids)] == kw1_ids:
                    toks_e1 = list(range(i, i + len(kw1_ids)))
                    break
        if not toks_e0:
            toks_e0 = [1, 2]
        if not toks_e1:
            toks_e1 = [3, 4]

    toks_e0_t = torch.tensor(toks_e0, dtype=torch.long, device=device)
    toks_e1_t = torch.tensor(toks_e1, dtype=torch.long, device=device)
    return toks_e0_t, toks_e1_t, full_prompt


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def evaluate_val_set(
    pipe, manager: Phase61Manager, dataset, val_idx: list,
    device: str, t_fixed: int = 200, depth_bins: int = 2,
) -> dict:
    """Evaluate Phase61 on validation set."""
    manager.eval()
    comp_mses, alpha_vol_losses, own_losses, depth_exp_losses = [], [], [], []
    iou_e0s, iou_e1s = [], []

    for vi in val_idx:
        try:
            sample = dataset[vi]
            if len(sample) >= 8:
                frames_np, depth_orders = sample[0], sample[2]
                meta, entity_masks = sample[3], sample[4]
                visible_masks = (sample[5] if sample[5] is not None
                                 else entity_masks.astype(np.float32))
            else:
                frames_np, _, depth_orders, meta, entity_masks = sample[:5]
                visible_masks = entity_masks.astype(np.float32)

            toks_e0_t, toks_e1_t, full_prompt = _get_entity_tokens_with_fallback(
                pipe, meta, device)
            manager.set_entity_tokens(toks_e0_t, toks_e1_t)
            manager.reset()

            latents = encode_frames_to_latents(pipe, frames_np, device)
            noise = torch.randn_like(latents)
            t_tensor = torch.tensor([t_fixed], device=device).long()
            noisy = pipe.scheduler.add_noise(latents, noise, t_tensor)
            enc_full = encode_text(pipe, full_prompt, device)
            T_frames = min(frames_np.shape[0], entity_masks.shape[0])

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = pipe.unet(
                    noisy, t_tensor, encoder_hidden_states=enc_full).sample

            preds = manager.volume_predictions
            if preds[0] is None:
                continue

            alpha0_bins, alpha1_bins = preds[0], preds[1]
            w0_bins, w1_bins, w_bg = preds[4], preds[5], preds[6]

            B_a, S_a, K = alpha0_bins.shape

            # Composite MSE
            noise_t = noise[:, :, :T_frames].float()
            noise_pred_t = noise_pred[:, :, :T_frames].float()
            l_comp = F.mse_loss(noise_pred_t, noise_t)
            comp_mses.append(float(l_comp.item()))

            # Prepare masks at feature spatial resolution
            masks_feat = masks_to_feature_space(
                entity_masks[:T_frames], S_a).to(device).float()
            T_m = masks_feat.shape[0]
            n_repeat = max(1, B_a // T_m) if T_m > 0 else 1
            if masks_feat.shape[0] < B_a:
                masks_feat = masks_feat.repeat(n_repeat, 1, 1)[:B_a]

            vis_feat = masks_to_feature_space(
                visible_masks[:T_frames], S_a).to(device).float()
            if vis_feat.shape[0] < B_a:
                vis_feat = vis_feat.repeat(n_repeat, 1, 1)[:B_a]

            # Alpha volume loss
            tgt0, tgt1, v0, v1 = build_depth_bin_targets(
                masks_feat, depth_orders[:T_frames], depth_bins)
            l_alpha_vol = loss_alpha_volume(alpha0_bins, alpha1_bins, tgt0, tgt1, v0, v1)
            alpha_vol_losses.append(float(l_alpha_vol.item()))

            # Visible ownership loss
            l_own = loss_visible_ownership(w0_bins, w1_bins, vis_feat)
            own_losses.append(float(l_own.item()))

            # Depth expected loss
            l_depth = loss_depth_expected(
                alpha0_bins, alpha1_bins, depth_orders[:T_frames], masks_feat)
            depth_exp_losses.append(float(l_depth.item()))

            # Ownership IoU (sum weights across bins for visible weight)
            vis_w0 = w0_bins.sum(dim=2)  # (B, S)
            vis_w1 = w1_bins.sum(dim=2)  # (B, S)
            iou0 = compute_visible_iou_e0(vis_w0, masks_feat, depth_orders[:T_frames])
            iou1 = compute_visible_iou_e1(vis_w1, masks_feat, depth_orders[:T_frames])
            iou_e0s.append(iou0)
            iou_e1s.append(iou1)

        except Exception as e:
            print(f"  [val warn] idx={vi}: {e}", flush=True)
            continue

    def _avg(lst):
        return sum(lst) / max(len(lst), 1) if lst else 999.0

    val_comp_mse = _avg(comp_mses)
    val_alpha_vol = _avg(alpha_vol_losses)
    val_own_loss = _avg(own_losses)
    val_depth_exp = _avg(depth_exp_losses)
    val_iou_e0 = _avg(iou_e0s) if iou_e0s else 0.0
    val_iou_e1 = _avg(iou_e1s) if iou_e1s else 0.0

    # IoU-heavy scoring: actual visible separation matters most
    val_score = (
        0.25 * (1.0 / (1.0 + val_comp_mse))
        + 0.45 * (val_iou_e0 + val_iou_e1) / 2.0
        + 0.15 * (1.0 / (1.0 + val_alpha_vol))
        + 0.15 * (1.0 / (1.0 + val_own_loss))
    )

    return {
        "val_score":     val_score,
        "val_comp_mse":  val_comp_mse,
        "val_alpha_vol": val_alpha_vol,
        "val_own_loss":  val_own_loss,
        "val_depth_exp": val_depth_exp,
        "val_iou_e0":    val_iou_e0,
        "val_iou_e1":    val_iou_e1,
        "n_samples":     len(comp_mses),
    }


# =============================================================================
# Main training
# =============================================================================

def train_phase61(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    debug_dir = Path(args.debug_dir)
    save_dir = Path(args.save_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _save_rollout_artifacts(prefix: str, comp_frames, overlay_frames):
        comp_gif = None
        ov_gif = None
        ov_png = None
        if comp_frames:
            comp_gif = debug_dir / f"{prefix}_composite.gif"
            iio2.mimwrite(str(comp_gif), comp_frames, fps=8, loop=0)
        if overlay_frames:
            ov_png = debug_dir / f"{prefix}_overlay.png"
            Image.fromarray(overlay_frames[0]).save(str(ov_png))
            ov_gif = debug_dir / f"{prefix}_overlay.gif"
            iio2.mimwrite(str(ov_gif), overlay_frames, fps=8, loop=0)
        return comp_gif, ov_gif, ov_png

    # Pipeline
    print("[Phase 61] Loading pipeline...", flush=True)
    pipe = load_pipeline(device=device)
    pipe.scheduler.set_timesteps(50, device=device)
    pipe.unet.enable_gradient_checkpointing()

    # Dataset
    print(f"[Phase 61] Loading dataset from {args.data_root}...", flush=True)
    dataset = ObjaverseDatasetPhase40(args.data_root, n_frames=args.n_frames)
    print(f"  dataset size: {len(dataset)}", flush=True)

    overlap_scores = compute_dataset_overlap_scores(dataset)
    train_idx, val_idx = split_train_val(
        overlap_scores, val_frac=args.val_frac, min_val=MIN_VAL_SAMPLES)
    sample_weights = make_sampling_weights(train_idx, overlap_scores)
    print(f"  train={len(train_idx)} val={len(val_idx)}", flush=True)

    # Inject Phase61 processors
    inject_keys = (
        args.inject_keys.split(",") if args.inject_keys else DEFAULT_INJECT_KEYS)
    print(f"[Phase 61] Injecting processors into {inject_keys}...", flush=True)

    manager, orig_procs = inject_phase61(
        pipe,
        inject_keys=inject_keys,
        adapter_rank=args.adapter_rank,
        lora_rank=args.lora_rank,
        depth_bins=args.depth_bins,
        temperature=args.softmax_temperature,
    )

    # Restore from checkpoint
    if args.ckpt and Path(args.ckpt).exists():
        print(f"[Phase 61] Loading checkpoint: {args.ckpt}", flush=True)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        restore_phase61(manager, ckpt, device)
    else:
        print("[Phase 61] No checkpoint, starting from scratch.", flush=True)

    for p in manager.procs:
        p.to(device)

    # Freeze base model
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    # Epoch 0 validation
    print("\n[Phase 61] Epoch 0 validation...", flush=True)
    val_m0 = evaluate_val_set(
        pipe, manager, dataset, val_idx, device,
        t_fixed=args.t_max // 2, depth_bins=args.depth_bins)
    print(f"  [epoch0] val_score={val_m0['val_score']:.4f}  "
          f"comp_mse={val_m0['val_comp_mse']:.4f}  "
          f"alpha_vol={val_m0['val_alpha_vol']:.4f}  "
          f"own_loss={val_m0['val_own_loss']:.4f}  "
          f"depth_exp={val_m0['val_depth_exp']:.4f}  "
          f"iou_e0={val_m0['val_iou_e0']:.4f}  "
          f"iou_e1={val_m0['val_iou_e1']:.4f}",
          flush=True)

    # Training loop
    history = []
    best_val_score = -1.0
    best_epoch = -1
    stage_best_scores = {"A": -1.0, "B": -1.0, "C": -1.0}
    stage_best_epochs = {"A": -1, "B": -1, "C": -1}
    total_epochs = args.stage_a_epochs + args.stage_b_epochs + args.stage_c_epochs
    stage_a_end = args.stage_a_epochs
    stage_b_end = args.stage_a_epochs + args.stage_b_epochs

    optimizer = None
    lr_scheduler = None

    for epoch in range(total_epochs):
        # Determine stage
        if epoch < stage_a_end:
            stage = "A"
        elif epoch < stage_b_end:
            stage = "B"
        else:
            stage = "C"

        # Optimizer setup at stage transitions
        if epoch == 0:
            print(f"\n[Phase 61] Stage A ({args.stage_a_epochs} epochs): "
                  f"volume heads bootstrap — volume heads + adapters",
                  flush=True)
            for p in manager.shared_lora_params():
                p.requires_grad_(False)
            for p in manager.volume_head_params():
                p.requires_grad_(True)
            for p in manager.adapter_params():
                p.requires_grad_(True)

            optimizer = optim.AdamW([
                {"params": manager.volume_head_params(),
                 "lr": args.lr_volume, "name": "volume_heads"},
                {"params": manager.adapter_params(),
                 "lr": args.lr_adapter, "name": "adapters"},
            ], weight_decay=1e-4)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.stage_a_epochs), eta_min=1e-6)

        if epoch == stage_a_end and stage_a_end > 0:
            print(f"\n[Phase 61] Stage B ({args.stage_b_epochs} epochs): "
                  f"+ shared LoRA", flush=True)
            for p in manager.shared_lora_params():
                p.requires_grad_(True)
            for p in manager.volume_head_params():
                p.requires_grad_(True)
            for p in manager.adapter_params():
                p.requires_grad_(True)

            optimizer = optim.AdamW([
                {"params": manager.volume_head_params(),
                 "lr": args.lr_volume * 0.5, "name": "volume_heads"},
                {"params": manager.shared_lora_params(),
                 "lr": args.lr_lora, "name": "shared_lora"},
                {"params": manager.adapter_params(),
                 "lr": args.lr_adapter * 0.5, "name": "adapters"},
            ], weight_decay=1e-4)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.stage_b_epochs), eta_min=1e-6)

        if epoch == stage_b_end and stage_b_end > 0 and args.stage_c_epochs > 0:
            print(f"\n[Phase 61] Stage C ({args.stage_c_epochs} epochs): "
                  f"joint fine-tuning — all params", flush=True)
            for p in manager.all_params():
                p.requires_grad_(True)

            optimizer = optim.AdamW([
                {"params": manager.volume_head_params(),
                 "lr": args.lr_volume * 0.3, "name": "volume_heads"},
                {"params": manager.shared_lora_params(),
                 "lr": args.lr_lora, "name": "shared_lora"},
                {"params": manager.adapter_params(),
                 "lr": args.lr_adapter * 0.3, "name": "adapters"},
            ], weight_decay=1e-4)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.stage_c_epochs), eta_min=1e-6)

        manager.train()
        epoch_losses = {
            "total": [], "composite": [], "alpha_vol": [],
            "ownership": [], "depth_exp": [], "ortho": [],
        }
        n_collision_aug = 0
        n_collision_attempt = 0

        # Sample training indices
        chosen = np.random.choice(
            len(train_idx), size=args.steps_per_epoch,
            replace=True, p=sample_weights)
        step_indices = [train_idx[ci] for ci in chosen]

        collision_prob = {
            "A": args.collision_prob_a,
            "B": args.collision_prob_b,
            "C": args.collision_prob_c,
        }[stage]

        for batch_idx, data_idx in enumerate(step_indices):
            sample = dataset[data_idx]
            if len(sample) >= 8:
                frames_np, depth_orders = sample[0], sample[2]
                meta, entity_masks = sample[3], sample[4]
                visible_masks = (sample[5] if sample[5] is not None
                                 else entity_masks.astype(np.float32))
            else:
                frames_np, _, depth_orders, meta, entity_masks = sample[:5]
                visible_masks = entity_masks.astype(np.float32)

            # Collision augmentation
            if (collision_prob > 0.0 and len(sample) >= 8
                    and np.random.rand() < collision_prob):
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

            # Entity tokens
            toks_e0_t, toks_e1_t, full_prompt = _get_entity_tokens_with_fallback(
                pipe, meta, device)
            manager.set_entity_tokens(toks_e0_t, toks_e1_t)
            manager.reset()

            # Encode frames + noise
            with torch.no_grad():
                latents = encode_frames_to_latents(pipe, frames_np, device)
            noise = torch.randn_like(latents)
            t = torch.randint(0, args.t_max, (1,), device=device).long()
            noisy = pipe.scheduler.add_noise(latents, noise, t)

            T_frames = min(frames_np.shape[0], entity_masks.shape[0],
                           len(depth_orders))

            enc_full = encode_text(pipe, full_prompt, device)

            # UNet forward
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = pipe.unet(
                    noisy, t, encoder_hidden_states=enc_full).sample

            # Propagate primary routing to non-primary blocks
            manager.propagate_routing()

            # Get volume predictions
            preds = manager.volume_predictions
            if preds[0] is None:
                print(f"  [warn] ep={epoch} step={batch_idx}: "
                      f"no volume predictions", flush=True)
                continue

            alpha0_bins, alpha1_bins = preds[0], preds[1]  # (B, S, K)
            w0_bins, w1_bins, w_bg = preds[4], preds[5], preds[6]

            B_a, S_a, K = alpha0_bins.shape

            # Prepare GT masks at attention spatial resolution
            masks_feat = masks_to_feature_space(
                entity_masks[:T_frames], S_a).to(device).float()
            T_m = masks_feat.shape[0]
            n_repeat = max(1, B_a // T_m) if T_m > 0 else 1
            if masks_feat.shape[0] < B_a:
                masks_feat = masks_feat.repeat(n_repeat, 1, 1)[:B_a]

            vis_feat = masks_to_feature_space(
                visible_masks[:T_frames], S_a).to(device).float()
            if vis_feat.shape[0] < B_a:
                vis_feat = vis_feat.repeat(n_repeat, 1, 1)[:B_a]

            # Build depth bin targets
            tgt0, tgt1, valid0, valid1 = build_depth_bin_targets(
                masks_feat, depth_orders[:T_frames], args.depth_bins)

            # Compute losses
            noise_t = noise[:, :, :T_frames].float()
            noise_pred_t = noise_pred[:, :, :T_frames].float()

            l_comp = loss_composite(noise_pred_t, noise_t)
            l_alpha_vol = loss_alpha_volume(alpha0_bins, alpha1_bins, tgt0, tgt1, valid0, valid1)
            l_own = loss_visible_ownership(w0_bins, w1_bins, vis_feat)
            l_depth = loss_depth_expected(
                alpha0_bins, alpha1_bins, depth_orders[:T_frames], masks_feat,
                margin=args.depth_margin)
            _, feat0_base, feat1_base = manager.primary_features
            if feat0_base is not None and feat1_base is not None:
                l_ortho = loss_feature_orthogonality(feat0_base, feat1_base, masks_feat)
            else:
                l_ortho = torch.tensor(0.0, device=device)

            # Weighted total loss
            if stage == "A":
                loss = (
                    0.1 * args.la_comp * l_comp
                    + args.la_alpha_vol * l_alpha_vol
                    + args.la_own * l_own
                    + args.la_depth_exp * l_depth
                    + args.la_ortho * l_ortho
                )
            else:  # Stage B and C
                loss = (
                    args.la_comp * l_comp
                    + args.la_alpha_vol * l_alpha_vol
                    + args.la_own * l_own
                    + args.la_depth_exp * l_depth
                    + args.la_ortho * l_ortho
                )

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss ep={epoch} "
                      f"step={batch_idx} -> skip", flush=True)
                continue

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                manager.volume_head_params(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(
                manager.adapter_params(), max_norm=1.0)
            if stage in ("B", "C"):
                torch.nn.utils.clip_grad_norm_(
                    manager.shared_lora_params(), max_norm=0.5)

            optimizer.step()

            # Log losses
            epoch_losses["total"].append(float(loss.item()))
            epoch_losses["composite"].append(float(l_comp.item()))
            epoch_losses["alpha_vol"].append(float(l_alpha_vol.item()))
            epoch_losses["ownership"].append(float(l_own.item()))
            epoch_losses["depth_exp"].append(float(l_depth.item()))
            epoch_losses["ortho"].append(float(l_ortho.item()))

        lr_scheduler.step()

        # Epoch summary
        avg = {k: float(np.mean(v)) if v else 0.0
               for k, v in epoch_losses.items()}
        print(
            f"[Phase 61][Stage {stage}] epoch {epoch:03d}/{total_epochs-1}  "
            f"loss={avg['total']:.4f}  comp={avg['composite']:.4f}  "
            f"alpha_vol={avg['alpha_vol']:.4f}  own={avg['ownership']:.4f}  "
            f"depth_exp={avg['depth_exp']:.4f}  ortho={avg['ortho']:.4f}  "
            f"aug={n_collision_aug}/{n_collision_attempt}",
            flush=True)

        # Validation
        should_eval = (epoch % args.eval_every == 0) or (epoch == total_epochs - 1)

        if should_eval:
            manager.eval()
            probe_rollout_mse = None
            probe_rollout_score = None

            val_m = evaluate_val_set(
                pipe, manager, dataset, val_idx, device,
                t_fixed=args.t_max // 2, depth_bins=args.depth_bins)

            vs = val_m["val_score"]
            print(
                f"  [val] val_score={vs:.4f}  "
                f"comp_mse={val_m['val_comp_mse']:.4f}  "
                f"alpha_vol={val_m['val_alpha_vol']:.4f}  "
                f"own_loss={val_m['val_own_loss']:.4f}  "
                f"depth_exp={val_m['val_depth_exp']:.4f}  "
                f"iou_e0={val_m['val_iou_e0']:.4f}  "
                f"iou_e1={val_m['val_iou_e1']:.4f}  "
                f"n_samples={val_m['n_samples']}",
                flush=True)

            # GIF generation: CFG-enabled composite rollout
            try:
                probe = dataset[val_idx[0]]
                probe_meta = probe[3]
                probe_masks = probe[4]

                toks_e0_pt, toks_e1_pt, probe_prompt = \
                    _get_entity_tokens_with_fallback(pipe, probe_meta, device)
                manager.set_entity_tokens(toks_e0_pt, toks_e1_pt)

                gen = torch.Generator(device=device).manual_seed(args.eval_seed)
                latent_shape = (
                    1, 4, args.n_frames,
                    args.height // 8, args.width // 8)
                latents = torch.randn(
                    latent_shape, generator=gen,
                    device=device, dtype=torch.float16)
                latents = latents * pipe.scheduler.init_noise_sigma

                pipe.scheduler.set_timesteps(args.n_steps, device=device)

                enc_cond = encode_text(pipe, probe_prompt, device)
                neg_prompt = "blurry, deformed, extra limbs, watermark"
                enc_uncond = encode_text(pipe, neg_prompt, device)
                guidance_scale = 7.5

                def _cfg_unet(latents_in, step_t):
                    """UNet forward with classifier-free guidance."""
                    lat2 = torch.cat([latents_in] * 2, dim=0)  # (2, 4, T, H, W)
                    lat2 = pipe.scheduler.scale_model_input(lat2, step_t)
                    enc2 = torch.cat([enc_uncond, enc_cond], dim=0)  # (2, 77, 768)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred = pipe.unet(
                            lat2, step_t,
                            encoder_hidden_states=enc2).sample
                    uncond_p, cond_p = pred.chunk(2, dim=0)
                    return uncond_p + guidance_scale * (cond_p - uncond_p)

                with torch.no_grad():
                    for step_t in pipe.scheduler.timesteps:
                        manager.reset()
                        noise_pred_cfg = _cfg_unet(latents, step_t)
                        latents = pipe.scheduler.step(
                            noise_pred_cfg.half(), step_t, latents,
                            return_dict=False)[0]

                # Decode to frames
                latents_4d = latents[0].permute(1, 0, 2, 3).half()
                scale_f = pipe.vae.config.scaling_factor
                comp_frames = []
                for fi in range(args.n_frames):
                    z_in = (latents_4d[fi:fi+1] / scale_f).half()
                    with torch.no_grad():
                        decoded = pipe.vae.decode(z_in).sample
                    img = ((decoded.float() / 2 + 0.5).clamp(0, 1)[0]
                           .permute(1, 2, 0).cpu().numpy() * 255
                           ).astype(np.uint8)
                    comp_frames.append(img)

                overlay_frames = []
                stage_prefix = f"stage{stage}"
                epoch_prefix = f"eval_epoch{epoch:03d}"
                comp_gif, _, _ = _save_rollout_artifacts(
                    epoch_prefix, comp_frames, None)
                print(f"  [gif] COMPOSITE rollout: {comp_gif}", flush=True)

                # Overlay: ownership on generated frames
                try:
                    T_ov = min(len(comp_frames), probe_masks.shape[0])
                    for fi in range(T_ov):
                        frame = comp_frames[fi].copy()
                        S_mask = probe_masks.shape[-1]
                        hw_m = int(S_mask ** 0.5)
                        m0 = probe_masks[fi, 0].reshape(hw_m, hw_m)
                        m1 = probe_masks[fi, 1].reshape(hw_m, hw_m)
                        m0_up = np.array(Image.fromarray(
                            (m0 * 255).astype(np.uint8)).resize(
                            (args.width, args.height), Image.BILINEAR)
                        ).astype(np.float32) / 255.0
                        m1_up = np.array(Image.fromarray(
                            (m1 * 255).astype(np.uint8)).resize(
                            (args.width, args.height), Image.BILINEAR)
                        ).astype(np.float32) / 255.0
                        overlay = frame.astype(np.float32)
                        overlay[:, :, 0] += m0_up * 80   # red for e0
                        overlay[:, :, 2] += m1_up * 80   # blue for e1
                        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                        overlay_frames.append(overlay)

                    if overlay_frames:
                        _, ov_gif, _ = _save_rollout_artifacts(
                            epoch_prefix, [], overlay_frames)
                        print(f"  [gif] OVERLAY: {ov_gif}", flush=True)
                except Exception as e:
                    print(f"  [warn] overlay failed: {e}", flush=True)

                # Stage-latest aliases: quick per-phase inspection
                stage_latest_prefix = f"{stage_prefix}_latest"
                _save_rollout_artifacts(
                    stage_latest_prefix, comp_frames, overlay_frames)

                # Probe rollout MSE
                gt_probe = np.asarray(
                    probe[0][:args.n_frames], dtype=np.float32) / 255.0
                pred_probe = np.asarray(
                    comp_frames[:args.n_frames], dtype=np.float32) / 255.0
                T_cmp = min(len(gt_probe), len(pred_probe))
                if T_cmp > 0:
                    probe_rollout_mse = float(
                        np.mean((pred_probe[:T_cmp] - gt_probe[:T_cmp]) ** 2))
                    probe_rollout_score = 1.0 / (1.0 + probe_rollout_mse)
                    print(
                        f"  [rollout] probe_rgb_mse={probe_rollout_mse:.4f}  "
                        f"probe_score={probe_rollout_score:.4f}",
                        flush=True)

            except Exception as e:
                print(f"  [warn] GIF failed: {e}", flush=True)
                import traceback; traceback.print_exc()

            # Selection score
            selection_score = vs
            if probe_rollout_score is not None:
                selection_score = 0.7 * vs + 0.3 * probe_rollout_score
                print(
                    f"  [select] score={selection_score:.4f}  "
                    f"(val={vs:.4f}, rollout={probe_rollout_score:.4f})",
                    flush=True)

            # Save history
            history.append({
                "epoch": epoch,
                "stage": stage,
                **val_m,
                **avg,
                "collision_aug_count": n_collision_aug,
                "probe_rollout_mse": probe_rollout_mse,
                "probe_rollout_score": probe_rollout_score,
                "selection_score": selection_score,
            })

            # Checkpoint
            ckpt_data = {
                "epoch": epoch,
                "stage": stage,
                "val_score": vs,
                "selection_score": selection_score,
                "inject_keys": inject_keys,
                "adapter_rank": args.adapter_rank,
                "lora_rank": args.lora_rank,
                "depth_bins": args.depth_bins,
                "procs_state": [],
            }
            for p in manager.procs:
                ps = {
                    "lora_k":        p.lora_k.state_dict(),
                    "lora_v":        p.lora_v.state_dict(),
                    "lora_out":      p.lora_out.state_dict(),
                    "slot0_adapter": p.slot0_adapter.state_dict(),
                    "slot1_adapter": p.slot1_adapter.state_dict(),
                }
                if p.is_primary:
                    ps["e0_volume"] = p.e0_volume.state_dict()
                    ps["e1_volume"] = p.e1_volume.state_dict()
                ckpt_data["procs_state"].append(ps)

            torch.save(ckpt_data, str(save_dir / "latest.pt"))

            if selection_score > best_val_score:
                best_val_score = selection_score
                best_epoch = epoch
                torch.save(ckpt_data, str(save_dir / "best.pt"))
                print(
                    f"  * best epoch={best_epoch} "
                    f"val_score={vs:.4f} "
                    f"selection={selection_score:.4f} "
                    f"-> {save_dir}/best.pt", flush=True)

            if selection_score > stage_best_scores.get(stage, -1.0):
                stage_best_scores[stage] = selection_score
                stage_best_epochs[stage] = epoch
                torch.save(ckpt_data, str(save_dir / f"best_stage_{stage}.pt"))
                stage_best_prefix = f"stage{stage}_best"
                try:
                    _save_rollout_artifacts(
                        stage_best_prefix,
                        comp_frames if 'comp_frames' in locals() else [],
                        overlay_frames if 'overlay_frames' in locals() else [],
                    )
                    with open(debug_dir / f"{stage_best_prefix}.json", "w") as f:
                        json.dump({
                            "epoch": epoch,
                            "stage": stage,
                            "val_score": vs,
                            "selection_score": selection_score,
                            **val_m,
                        }, f, indent=2)
                    print(
                        f"  * stage-{stage} best epoch={epoch} "
                        f"selection={selection_score:.4f} "
                        f"-> {save_dir}/best_stage_{stage}.pt",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  [warn] stage-best artifact save failed: {e}", flush=True)

            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            manager.train()

    print(f"\n[Phase 61] Done. best epoch={best_epoch} "
          f"val_score={best_val_score:.4f}", flush=True)


# =============================================================================
# argparse
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Phase 61: Depth-Layered Volume Diffusion")

    p.add_argument("--ckpt",       type=str, default="",
                   help="Checkpoint to resume from (Phase52/60/61)")
    p.add_argument("--data-root",  type=str, default="toy/data_objaverse")
    p.add_argument("--save-dir",   type=str, default="checkpoints/phase61")
    p.add_argument("--debug-dir",  type=str, default="outputs/phase61_debug")

    # Stage epochs
    p.add_argument("--stage-a-epochs",  type=int, default=DEFAULT_STAGE_A_EPOCHS)
    p.add_argument("--stage-b-epochs",  type=int, default=DEFAULT_STAGE_B_EPOCHS)
    p.add_argument("--stage-c-epochs",  type=int, default=DEFAULT_STAGE_C_EPOCHS)
    p.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)

    # Learning rates
    p.add_argument("--lr-volume",  type=float, default=DEFAULT_LR_VOLUME)
    p.add_argument("--lr-adapter", type=float, default=DEFAULT_LR_ADAPTER)
    p.add_argument("--lr-lora",    type=float, default=DEFAULT_LR_LORA)

    # Architecture
    p.add_argument("--adapter-rank", type=int, default=DEFAULT_ADAPTER_RANK)
    p.add_argument("--lora-rank",    type=int, default=DEFAULT_LORA_RANK)
    p.add_argument("--depth-bins",   type=int, default=DEFAULT_DEPTH_BINS)
    p.add_argument("--softmax-temperature", type=float, default=0.1,
                   help="Phase61 compositor temperature; lower is harder")
    p.add_argument("--inject-keys",  type=str, default="")

    # Loss weights
    p.add_argument("--la-comp",      type=float, default=DEFAULT_LA_COMP)
    p.add_argument("--la-alpha-vol", type=float, default=DEFAULT_LA_ALPHA_VOL)
    p.add_argument("--la-own",       type=float, default=DEFAULT_LA_OWN)
    p.add_argument("--la-depth-exp", type=float, default=DEFAULT_LA_DEPTH_EXP)
    p.add_argument("--la-ortho",     type=float, default=DEFAULT_LA_ORTHO)
    p.add_argument("--depth-margin", type=float, default=0.3)

    # Collision augmentation
    p.add_argument("--collision-prob-a",    type=float, default=DEFAULT_COLLISION_PROB_A)
    p.add_argument("--collision-prob-b",    type=float, default=DEFAULT_COLLISION_PROB_B)
    p.add_argument("--collision-prob-c",    type=float, default=DEFAULT_COLLISION_PROB_C)
    p.add_argument("--collision-ov-min",    type=float, default=DEFAULT_COLLISION_OV_MIN)
    p.add_argument("--collision-ov-max",    type=float, default=DEFAULT_COLLISION_OV_MAX)
    p.add_argument("--collision-min-accept", type=float, default=DEFAULT_COLLISION_MIN_ACC)

    # General
    p.add_argument("--n-frames",   type=int,   default=8)
    p.add_argument("--height",     type=int,   default=256)
    p.add_argument("--width",      type=int,   default=256)
    p.add_argument("--t-max",      type=int,   default=500)
    p.add_argument("--n-steps",    type=int,   default=25)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--eval-seed",  type=int,   default=42)
    p.add_argument("--eval-every", type=int,   default=2)
    p.add_argument("--val-frac",   type=float, default=VAL_FRAC)

    args = p.parse_args()
    train_phase61(args)


if __name__ == "__main__":
    main()
