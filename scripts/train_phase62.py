"""
Phase 62 — 3D Entity Volume Generator + First-Hit Projection Training
======================================================================

Clean architecture: predict 3D voxel volume from UNet features,
project via first-hit scan, inject guide back into UNet.

2 losses only:
  L_diffusion: MSE(noise_pred, noise)
  L_volume_ce: CE(V_logits, V_gt)

Training step flow:
  1. Load sample, encode to latents, add noise
  2. Build V_gt from depth + masks (real 3D data)
  3. UNet forward pass #1 (no guide, extract features)
  4. EntityVolumePredictor: features -> V_logits
  5. L_volume_ce = CE(V_logits, V_gt)
  6. FirstHitProjector: V_logits -> visible_class, visible_probs
  7. VolumeGuidedInjector: build guide from visible
  8. UNet forward pass #2 (with guide injection)
  9. L_diffusion = MSE(noise_pred, noise)
  10. loss = la_diff * L_diffusion + la_vol * L_volume_ce

Eval rollout uses hybrid schedule for volume recomputation.
"""
import argparse
import json
import sys
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import imageio.v2 as iio2
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.phase62_entity_volume import EntityVolumePredictor
from models.phase62_projection import FirstHitProjector
from models.phase62_conditioning import (
    VolumeGuidedInjector,
    BLOCK_DIMS,
    BLOCK_SPATIAL,
    INJECT_CONFIGS,
    inject_guide_into_unet_features,
)
from models.phase62_losses import (
    loss_diffusion,
    loss_volume_ce,
    compute_volume_accuracy,
)
from scripts.build_volume_gt import build_volume_gt_batch

from models.entity_slot_phase40 import (
    SlotLoRA,
    Phase40Processor,
    inject_multi_block_entity_slot,
    MultiBlockSlotManager,
    BLOCK_INNER_DIMS,
    DEFAULT_INJECT_KEYS,
)
from models.entity_slot import SlotAdapter
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
DEFAULT_EPOCHS          = 25
DEFAULT_STEPS_PER_EPOCH = 20
DEFAULT_LR_VOLUME       = 5e-4
DEFAULT_LR_GUIDE        = 2e-4
DEFAULT_LR_ADAPTER      = 2e-5
DEFAULT_LR_LORA         = 5e-6
DEFAULT_ADAPTER_RANK    = 64
DEFAULT_LORA_RANK       = 4
DEFAULT_DEPTH_BINS      = 8
DEFAULT_HIDDEN_DIM      = 64
DEFAULT_SPATIAL_H       = 16
DEFAULT_SPATIAL_W       = 16

DEFAULT_LA_DIFF         = 1.0
DEFAULT_LA_VOL          = 2.0

DEFAULT_COLLISION_PROB   = 0.5
DEFAULT_COLLISION_OV_MIN = 0.005
DEFAULT_COLLISION_OV_MAX = 0.25

VAL_FRAC        = 0.2
MIN_VAL_SAMPLES = 3


# =============================================================================
# UNet guide injection hooks
# =============================================================================

class Phase62GuideHooks:
    """
    Manages forward hooks on UNet blocks to inject volume-projected guide.

    Registers hooks on mid_block and/or up_blocks that add the spatial
    guide features to the block's hidden states.
    """

    def __init__(self):
        self.hooks = []
        self.guides: dict = {}

    def set_guides(self, guides: dict):
        """Set guide features to inject at next forward pass."""
        self.guides = guides

    def clear_guides(self):
        self.guides = {}

    def _make_hook(self, block_name: str):
        def hook_fn(module, input, output):
            if block_name not in self.guides:
                return output
            guide = self.guides[block_name]

            if isinstance(output, tuple):
                h = output[0]
                h = inject_guide_into_unet_features(h, guide)
                return (h,) + output[1:]
            else:
                return inject_guide_into_unet_features(output, guide)
        return hook_fn

    def register_hooks(self, unet, inject_config: str = "mid_up2"):
        """Register forward hooks on appropriate UNet blocks."""
        block_names = INJECT_CONFIGS.get(inject_config, ["mid", "up2"])

        block_map = {
            "mid": unet.mid_block,
        }
        for i, up_block in enumerate(unet.up_blocks):
            block_map[f"up{i}"] = up_block

        for block_name in block_names:
            if block_name in block_map and block_map[block_name] is not None:
                h = block_map[block_name].register_forward_hook(
                    self._make_hook(block_name))
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


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
                if tok_ids[i:i + len(kw0_ids)] == kw0_ids:
                    toks_e0 = list(range(i, i + len(kw0_ids)))
                    break
        if not toks_e1 and kw1_ids:
            for i in range(len(tok_ids) - len(kw1_ids) + 1):
                if tok_ids[i:i + len(kw1_ids)] == kw1_ids:
                    toks_e1 = list(range(i, i + len(kw1_ids)))
                    break
        if not toks_e0:
            toks_e0 = [1, 2]
        if not toks_e1:
            toks_e1 = [3, 4]

    toks_e0_t = torch.tensor(toks_e0, dtype=torch.long, device=device)
    toks_e1_t = torch.tensor(toks_e1, dtype=torch.long, device=device)
    return toks_e0_t, toks_e1_t, full_prompt


def masks_to_feature_space(
    entity_masks: np.ndarray,  # (T, 2, S) where S = H_mask * W_mask
    spatial_dim: int,
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
                m1_shifted[y1s:y1e, x1s:x1e] = m1_r[sy:sy + h_c, sx:sx + w_c]
                s1_shifted[y1s:y1e, x1s:x1e] = solo_e1[fi][sy:sy + h_c, sx:sx + w_c]

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


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def evaluate_val_set(
    pipe, slot_manager: MultiBlockSlotManager,
    volume_pred: EntityVolumePredictor,
    projector: FirstHitProjector,
    dataset, val_idx: list,
    device: str, K: int = 8, t_fixed: int = 200,
    spatial_h: int = 16, spatial_w: int = 16,
) -> dict:
    """Evaluate Phase 62 on validation set."""
    volume_pred.eval()
    slot_manager.eval()

    diff_losses, vol_ce_losses = [], []
    vol_accs_overall, vol_accs_entity = [], []

    for vi in val_idx:
        try:
            sample = dataset[vi]
            if len(sample) >= 8:
                frames_np, depth_np, depth_orders = sample[0], sample[1], sample[2]
                meta, entity_masks = sample[3], sample[4]
            else:
                frames_np, depth_np, depth_orders, meta, entity_masks = sample[:5]

            toks_e0_t, toks_e1_t, full_prompt = _get_entity_tokens_with_fallback(
                pipe, meta, device)
            slot_manager.set_entity_tokens(toks_e0_t, toks_e1_t)
            slot_manager.reset_slot_store()

            latents = encode_frames_to_latents(pipe, frames_np, device)
            noise = torch.randn_like(latents)
            t_tensor = torch.tensor([t_fixed], device=device).long()
            noisy = pipe.scheduler.add_noise(latents, noise, t_tensor)
            enc_full = encode_text(pipe, full_prompt, device)

            T_frames = min(frames_np.shape[0], entity_masks.shape[0],
                           len(depth_orders))

            # UNet forward to extract features
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = pipe.unet(
                    noisy, t_tensor, encoder_hidden_states=enc_full).sample

            # Diffusion loss
            noise_t = noise[:, :, :T_frames].float()
            noise_pred_t = noise_pred[:, :, :T_frames].float()
            l_diff = loss_diffusion(noise_pred_t, noise_t)
            diff_losses.append(float(l_diff.item()))

            # Extract features from slot manager
            F_g = slot_manager.primary.last_Fg
            F_0 = slot_manager.primary.last_F0
            F_1 = slot_manager.primary.last_F1
            if F_g is None or F_0 is None or F_1 is None:
                continue

            # Volume prediction
            V_logits = volume_pred(F_g, F_0, F_1)  # (B, 3, K, H, W)

            # Build V_gt
            depth_maps = depth_np[:T_frames]  # (T, 256, 256)
            V_gt_np = build_volume_gt_batch(
                depth_maps, entity_masks[:T_frames],
                depth_orders[:T_frames], K=K,
                H_out=spatial_h, W_out=spatial_w)
            V_gt = torch.from_numpy(V_gt_np).to(device).long()  # (T, K, H, W)

            # Match batch dim
            B_feat = V_logits.shape[0]
            if V_gt.shape[0] < B_feat:
                n_rep = max(1, B_feat // V_gt.shape[0])
                V_gt = V_gt.repeat(n_rep, 1, 1, 1)[:B_feat]

            l_vol = loss_volume_ce(V_logits, V_gt)
            vol_ce_losses.append(float(l_vol.item()))

            acc = compute_volume_accuracy(V_logits, V_gt)
            vol_accs_overall.append(acc["overall_acc"])
            vol_accs_entity.append(acc["entity_acc"])

        except Exception as e:
            print(f"  [val warn] idx={vi}: {e}", flush=True)
            continue

    def _avg(lst):
        return sum(lst) / max(len(lst), 1) if lst else 999.0

    val_diff = _avg(diff_losses)
    val_vol_ce = _avg(vol_ce_losses)
    val_acc_all = _avg(vol_accs_overall) if vol_accs_overall else 0.0
    val_acc_ent = _avg(vol_accs_entity) if vol_accs_entity else 0.0

    # Score: volume accuracy matters most (entity class prediction)
    val_score = (
        0.30 * (1.0 / (1.0 + val_diff))
        + 0.30 * (1.0 / (1.0 + val_vol_ce))
        + 0.20 * val_acc_all
        + 0.20 * val_acc_ent
    )

    return {
        "val_score": val_score,
        "val_diff_mse": val_diff,
        "val_vol_ce": val_vol_ce,
        "val_acc_overall": val_acc_all,
        "val_acc_entity": val_acc_ent,
        "n_samples": len(diff_losses),
    }


# =============================================================================
# Main training
# =============================================================================

def train_phase62(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    debug_dir = Path(args.debug_dir)
    save_dir = Path(args.save_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline
    print("[Phase 62] Loading pipeline...", flush=True)
    pipe = load_pipeline(device=device)
    pipe.scheduler.set_timesteps(50, device=device)
    pipe.unet.enable_gradient_checkpointing()

    # Dataset
    print(f"[Phase 62] Loading dataset from {args.data_root}...", flush=True)
    dataset = ObjaverseDatasetPhase40(args.data_root, n_frames=args.n_frames)
    print(f"  dataset size: {len(dataset)}", flush=True)

    overlap_scores = compute_dataset_overlap_scores(dataset)
    train_idx, val_idx = split_train_val(
        overlap_scores, val_frac=args.val_frac, min_val=MIN_VAL_SAMPLES)
    sample_weights = make_sampling_weights(train_idx, overlap_scores)
    print(f"  train={len(train_idx)} val={len(val_idx)}", flush=True)

    # VCA layer (set to None for Phase 62 — no Porter-Duff needed)
    vca_layer = None

    # Inject multi-block slot processors (reuse Phase40 infrastructure)
    print("[Phase 62] Injecting slot processors...", flush=True)
    inject_keys = DEFAULT_INJECT_KEYS
    procs, orig_procs = inject_multi_block_entity_slot(
        pipe, vca_layer=vca_layer,
        entity_ctx=torch.zeros(1, 2, 768),  # placeholder, set per-sample
        inject_keys=inject_keys,
        slot_blend_init=0.3,
        adapter_rank=args.adapter_rank,
        lora_rank=args.lora_rank,
        use_blend_head=False,  # no blend head for Phase 62
    )
    for p in procs:
        p.to(device)
    slot_manager = MultiBlockSlotManager(procs, inject_keys, primary_idx=1)

    # Phase 62 modules
    print("[Phase 62] Creating volume predictor + projector + injector...", flush=True)
    feat_dim = BLOCK_INNER_DIMS.get(inject_keys[1], 640)  # primary block dim

    volume_pred = EntityVolumePredictor(
        feat_dim=feat_dim,
        n_classes=3,
        depth_bins=args.depth_bins,
        spatial_h=args.spatial_h,
        spatial_w=args.spatial_w,
        hidden=args.hidden_dim,
    ).to(device)

    projector = FirstHitProjector(n_classes=3, bg_class=0).to(device)

    guide_injector = VolumeGuidedInjector(
        n_classes=3,
        hidden=args.hidden_dim,
        inject_config=args.inject_config,
    ).to(device)

    # Guide hooks
    guide_hooks = Phase62GuideHooks()
    guide_hooks.register_hooks(pipe.unet, inject_config=args.inject_config)

    # Freeze base model
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    # Collect trainable parameters
    volume_params = list(volume_pred.parameters())
    guide_params = list(guide_injector.parameters())
    adapter_params = slot_manager.adapter_params()
    lora_params = slot_manager.lora_params()

    for p in volume_params:
        p.requires_grad_(True)
    for p in guide_params:
        p.requires_grad_(True)
    for p in adapter_params:
        p.requires_grad_(True)
    for p in lora_params:
        p.requires_grad_(True)

    optimizer = optim.AdamW([
        {"params": volume_params, "lr": args.lr_volume, "name": "volume_pred"},
        {"params": guide_params, "lr": args.lr_guide, "name": "guide_injector"},
        {"params": adapter_params, "lr": args.lr_adapter, "name": "adapters"},
        {"params": lora_params, "lr": args.lr_lora, "name": "lora"},
    ], weight_decay=1e-4)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=1e-6)

    # Class weights for volume CE (bg is dominant)
    # Approximate: ~95% bg, ~2.5% entity0, ~2.5% entity1
    class_weights = torch.tensor([0.1, 1.0, 1.0], device=device, dtype=torch.float32)

    # Checkpoint restore
    if args.ckpt and Path(args.ckpt).exists():
        print(f"[Phase 62] Loading checkpoint: {args.ckpt}", flush=True)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        if "volume_pred" in ckpt:
            volume_pred.load_state_dict(ckpt["volume_pred"])
        if "guide_injector" in ckpt:
            guide_injector.load_state_dict(ckpt["guide_injector"])
        if "procs_state" in ckpt:
            for i, ps in enumerate(ckpt["procs_state"]):
                if i < len(procs):
                    if "lora_k" in ps:
                        procs[i].lora_k.load_state_dict(ps["lora_k"])
                    if "lora_v" in ps:
                        procs[i].lora_v.load_state_dict(ps["lora_v"])
                    if "lora_out" in ps:
                        procs[i].lora_out.load_state_dict(ps["lora_out"])
                    if "slot0_adapter" in ps:
                        procs[i].slot0_adapter.load_state_dict(ps["slot0_adapter"])
                    if "slot1_adapter" in ps:
                        procs[i].slot1_adapter.load_state_dict(ps["slot1_adapter"])
        print("[Phase 62] Checkpoint loaded.", flush=True)
    else:
        print("[Phase 62] No checkpoint, starting from scratch.", flush=True)

    # Epoch 0 validation
    print("\n[Phase 62] Epoch 0 validation...", flush=True)
    val_m0 = evaluate_val_set(
        pipe, slot_manager, volume_pred, projector,
        dataset, val_idx, device,
        K=args.depth_bins, t_fixed=args.t_max // 2,
        spatial_h=args.spatial_h, spatial_w=args.spatial_w)
    print(f"  [epoch0] val_score={val_m0['val_score']:.4f}  "
          f"diff_mse={val_m0['val_diff_mse']:.4f}  "
          f"vol_ce={val_m0['val_vol_ce']:.4f}  "
          f"acc_all={val_m0['val_acc_overall']:.4f}  "
          f"acc_ent={val_m0['val_acc_entity']:.4f}", flush=True)

    # Training loop
    history = []
    best_val_score = -1.0
    best_epoch = -1

    for epoch in range(args.epochs):
        volume_pred.train()
        guide_injector.train()
        slot_manager.train()

        epoch_losses = {"total": [], "diffusion": [], "volume_ce": []}
        epoch_accs = {"overall": [], "entity": []}
        n_collision_aug = 0

        # Sample training indices
        chosen = np.random.choice(
            len(train_idx), size=args.steps_per_epoch,
            replace=True, p=sample_weights)
        step_indices = [train_idx[ci] for ci in chosen]

        for batch_idx, data_idx in enumerate(step_indices):
            sample = dataset[data_idx]
            if len(sample) >= 8:
                frames_np, depth_np, depth_orders = sample[0], sample[1], sample[2]
                meta, entity_masks = sample[3], sample[4]
            else:
                frames_np, depth_np, depth_orders, meta, entity_masks = sample[:5]

            # Collision augmentation
            if (args.collision_prob > 0.0 and len(sample) >= 8
                    and np.random.rand() < args.collision_prob):
                aug = _try_collision_augment(
                    dataset, data_idx, sample,
                    overlap_min=args.collision_ov_min,
                    overlap_max=args.collision_ov_max)
                if aug is not None:
                    frames_np, depth_orders, entity_masks = aug
                    n_collision_aug += 1

            # Entity tokens
            toks_e0_t, toks_e1_t, full_prompt = _get_entity_tokens_with_fallback(
                pipe, meta, device)
            slot_manager.set_entity_tokens(toks_e0_t, toks_e1_t)
            slot_manager.reset_slot_store()

            # Encode frames + noise
            with torch.no_grad():
                latents = encode_frames_to_latents(pipe, frames_np, device)
            noise = torch.randn_like(latents)
            t = torch.randint(0, args.t_max, (1,), device=device).long()
            noisy = pipe.scheduler.add_noise(latents, noise, t)

            T_frames = min(frames_np.shape[0], entity_masks.shape[0],
                           len(depth_orders))

            enc_full = encode_text(pipe, full_prompt, device)

            # AnimateDiff enc_hs broadcast: if batch=1 and B>1, expand
            B_noisy = noisy.shape[0]
            if enc_full.shape[0] == 1 and B_noisy > 1:
                enc_full = enc_full.expand(B_noisy, -1, -1)

            optimizer.zero_grad()

            # --- Pass 1: UNet forward (no guide) to extract features ---
            guide_hooks.clear_guides()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = pipe.unet(
                    noisy, t, encoder_hidden_states=enc_full).sample

            # Extract features from primary slot processor
            F_g = slot_manager.primary.last_Fg   # (B, S, D)
            F_0 = slot_manager.primary.last_F0   # (B, S, D)
            F_1 = slot_manager.primary.last_F1   # (B, S, D)

            if F_g is None or F_0 is None or F_1 is None:
                print(f"  [warn] ep={epoch} step={batch_idx}: no features", flush=True)
                continue

            # --- Volume prediction ---
            V_logits = volume_pred(F_g, F_0, F_1)  # (B, 3, K, spatial_h, spatial_w)

            # --- Build V_gt ---
            depth_maps = depth_np[:T_frames]  # (T, 256, 256)
            V_gt_np = build_volume_gt_batch(
                depth_maps, entity_masks[:T_frames],
                depth_orders[:T_frames], K=args.depth_bins,
                H_out=args.spatial_h, W_out=args.spatial_w)
            V_gt = torch.from_numpy(V_gt_np).to(device).long()  # (T, K, H, W)

            # Match batch dimension
            B_feat = V_logits.shape[0]
            if V_gt.shape[0] < B_feat:
                n_rep = max(1, B_feat // V_gt.shape[0])
                V_gt = V_gt.repeat(n_rep, 1, 1, 1)[:B_feat]

            # --- L_volume_ce ---
            l_vol = loss_volume_ce(V_logits, V_gt, class_weights=class_weights)

            # --- First-hit projection ---
            visible_class, visible_probs = projector(V_logits)  # (B, H, W), (B, 3, H, W)

            # --- Build guide ---
            guides = guide_injector.build_guide(visible_class, visible_probs)
            guide_hooks.set_guides(guides)

            # --- Pass 2: UNet forward (with guide injection) ---
            slot_manager.reset_slot_store()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = pipe.unet(
                    noisy, t, encoder_hidden_states=enc_full).sample

            # --- L_diffusion ---
            noise_t = noise[:, :, :T_frames].float()
            noise_pred_t = noise_pred[:, :, :T_frames].float()
            l_diff = loss_diffusion(noise_pred_t, noise_t)

            # --- Total loss ---
            loss = args.la_diff * l_diff + args.la_vol * l_vol

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss ep={epoch} step={batch_idx} -> skip",
                      flush=True)
                guide_hooks.clear_guides()
                continue

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(volume_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(guide_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=0.5)

            optimizer.step()

            guide_hooks.clear_guides()

            # Logging
            epoch_losses["total"].append(float(loss.item()))
            epoch_losses["diffusion"].append(float(l_diff.item()))
            epoch_losses["volume_ce"].append(float(l_vol.item()))

            with torch.no_grad():
                acc = compute_volume_accuracy(V_logits, V_gt)
                epoch_accs["overall"].append(acc["overall_acc"])
                epoch_accs["entity"].append(acc["entity_acc"])

        lr_scheduler.step()

        # Epoch summary
        avg_loss = {k: float(np.mean(v)) if v else 0.0
                    for k, v in epoch_losses.items()}
        avg_acc = {k: float(np.mean(v)) if v else 0.0
                   for k, v in epoch_accs.items()}

        print(
            f"[Phase 62] epoch {epoch:03d}/{args.epochs - 1}  "
            f"loss={avg_loss['total']:.4f}  "
            f"diff={avg_loss['diffusion']:.4f}  "
            f"vol_ce={avg_loss['volume_ce']:.4f}  "
            f"acc_all={avg_acc['overall']:.4f}  "
            f"acc_ent={avg_acc['entity']:.4f}  "
            f"aug={n_collision_aug}",
            flush=True)

        # Validation
        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs - 1)

        if should_eval:
            volume_pred.eval()
            guide_injector.eval()
            slot_manager.eval()

            val_m = evaluate_val_set(
                pipe, slot_manager, volume_pred, projector,
                dataset, val_idx, device,
                K=args.depth_bins, t_fixed=args.t_max // 2,
                spatial_h=args.spatial_h, spatial_w=args.spatial_w)

            vs = val_m["val_score"]
            print(
                f"  [val] val_score={vs:.4f}  "
                f"diff_mse={val_m['val_diff_mse']:.4f}  "
                f"vol_ce={val_m['val_vol_ce']:.4f}  "
                f"acc_all={val_m['val_acc_overall']:.4f}  "
                f"acc_ent={val_m['val_acc_entity']:.4f}  "
                f"n={val_m['n_samples']}",
                flush=True)

            # GIF generation: CFG-enabled composite rollout with hybrid schedule
            probe_rollout_mse = None
            probe_rollout_score = None

            try:
                probe = dataset[val_idx[0]]
                probe_meta = probe[3]
                probe_masks = probe[4]

                toks_e0_pt, toks_e1_pt, probe_prompt = \
                    _get_entity_tokens_with_fallback(pipe, probe_meta, device)
                slot_manager.set_entity_tokens(toks_e0_pt, toks_e1_pt)

                gen = torch.Generator(device=device).manual_seed(args.eval_seed)
                latent_shape = (
                    1, 4, args.n_frames,
                    args.height // 8, args.width // 8)
                eval_latents = torch.randn(
                    latent_shape, generator=gen,
                    device=device, dtype=torch.float16)
                eval_latents = eval_latents * pipe.scheduler.init_noise_sigma

                pipe.scheduler.set_timesteps(args.n_steps, device=device)
                timesteps = pipe.scheduler.timesteps
                n_total = len(timesteps)

                enc_cond = encode_text(pipe, probe_prompt, device)
                neg_prompt = "blurry, deformed, extra limbs, watermark"
                enc_uncond = encode_text(pipe, neg_prompt, device)
                guidance_scale = 7.5

                # Hybrid schedule boundaries
                recompute_steps = set()
                if args.update_schedule == "fixed_once":
                    recompute_steps = {0}
                elif args.update_schedule == "hybrid":
                    recompute_steps = {0, n_total // 3, 2 * n_total // 3}
                elif args.update_schedule == "every_step":
                    recompute_steps = set(range(n_total))

                current_guides = {}

                with torch.no_grad():
                    for step_idx, step_t in enumerate(timesteps):
                        slot_manager.reset_slot_store()

                        # CFG: double batch
                        lat2 = torch.cat([eval_latents] * 2, dim=0)  # (2, 4, T, H, W)
                        lat2 = pipe.scheduler.scale_model_input(lat2, step_t)
                        enc2 = torch.cat([enc_uncond, enc_cond], dim=0)  # (2, 77, 768)

                        if step_idx in recompute_steps:
                            # Pass 1: extract features (no guide)
                            guide_hooks.clear_guides()
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                _ = pipe.unet(
                                    lat2, step_t,
                                    encoder_hidden_states=enc2).sample

                            # Volume prediction (use cond batch only)
                            F_g_e = slot_manager.primary.last_Fg
                            F_0_e = slot_manager.primary.last_F0
                            F_1_e = slot_manager.primary.last_F1

                            if F_g_e is not None and F_0_e is not None and F_1_e is not None:
                                # Take cond half (second batch item)
                                B_e = F_g_e.shape[0]
                                half = B_e // 2
                                V_logits_e = volume_pred(
                                    F_g_e[half:], F_0_e[half:], F_1_e[half:])
                                vc, vp = projector(V_logits_e)
                                current_guides = guide_injector.build_guide(vc, vp)

                            # Reset for pass 2
                            slot_manager.reset_slot_store()

                        # Pass 2 (or single pass): with guide
                        guide_hooks.set_guides(current_guides)
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            pred = pipe.unet(
                                lat2, step_t,
                                encoder_hidden_states=enc2).sample

                        uncond_p, cond_p = pred.chunk(2, dim=0)
                        noise_pred_cfg = uncond_p + guidance_scale * (cond_p - uncond_p)

                        eval_latents = pipe.scheduler.step(
                            noise_pred_cfg.half(), step_t, eval_latents,
                            return_dict=False)[0]

                guide_hooks.clear_guides()

                # Decode to frames
                latents_4d = eval_latents[0].permute(1, 0, 2, 3).half()
                scale_f = pipe.vae.config.scaling_factor
                comp_frames = []
                for fi in range(args.n_frames):
                    z_in = (latents_4d[fi:fi + 1] / scale_f).half()
                    with torch.no_grad():
                        decoded = pipe.vae.decode(z_in).sample
                    img = ((decoded.float() / 2 + 0.5).clamp(0, 1)[0]
                           .permute(1, 2, 0).cpu().numpy() * 255
                           ).astype(np.uint8)
                    comp_frames.append(img)

                comp_gif = debug_dir / f"eval_epoch{epoch:03d}_composite.gif"
                iio2.mimwrite(str(comp_gif), comp_frames, fps=8, loop=0)
                print(f"  [gif] COMPOSITE rollout: {comp_gif}", flush=True)

                # Overlay: mask overlay on generated frames
                try:
                    overlay_frames = []
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
                        overlay[:, :, 0] += m0_up * 80
                        overlay[:, :, 2] += m1_up * 80
                        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                        overlay_frames.append(overlay)

                    if overlay_frames:
                        ov_png = debug_dir / f"eval_epoch{epoch:03d}_overlay.png"
                        Image.fromarray(overlay_frames[0]).save(str(ov_png))
                        ov_gif = debug_dir / f"eval_epoch{epoch:03d}_overlay.gif"
                        iio2.mimwrite(str(ov_gif), overlay_frames, fps=8, loop=0)
                        print(f"  [gif] OVERLAY: {ov_gif}", flush=True)
                except Exception as e:
                    print(f"  [warn] overlay failed: {e}", flush=True)

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
                **val_m,
                **avg_loss,
                "avg_acc_overall": avg_acc["overall"],
                "avg_acc_entity": avg_acc["entity"],
                "collision_aug_count": n_collision_aug,
                "probe_rollout_mse": probe_rollout_mse,
                "probe_rollout_score": probe_rollout_score,
                "selection_score": selection_score,
            })

            # Checkpoint
            ckpt_data = {
                "epoch": epoch,
                "val_score": vs,
                "selection_score": selection_score,
                "inject_config": args.inject_config,
                "update_schedule": args.update_schedule,
                "depth_bins": args.depth_bins,
                "hidden_dim": args.hidden_dim,
                "spatial_h": args.spatial_h,
                "spatial_w": args.spatial_w,
                "volume_pred": volume_pred.state_dict(),
                "guide_injector": guide_injector.state_dict(),
                "procs_state": [],
            }
            for p in slot_manager.procs:
                ps = {
                    "lora_k": p.lora_k.state_dict(),
                    "lora_v": p.lora_v.state_dict(),
                    "lora_out": p.lora_out.state_dict(),
                    "slot0_adapter": p.slot0_adapter.state_dict(),
                    "slot1_adapter": p.slot1_adapter.state_dict(),
                }
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

            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            volume_pred.train()
            guide_injector.train()
            slot_manager.train()

    print(f"\n[Phase 62] Done. best epoch={best_epoch} "
          f"val_score={best_val_score:.4f}", flush=True)


# =============================================================================
# argparse
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Phase 62: 3D Entity Volume + First-Hit Projection")

    p.add_argument("--ckpt", type=str, default="",
                   help="Checkpoint to resume from")
    p.add_argument("--data-root", type=str, default="toy/data_objaverse")
    p.add_argument("--save-dir", type=str, default="checkpoints/phase62")
    p.add_argument("--debug-dir", type=str, default="outputs/phase62_debug")

    # Training
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)

    # Learning rates
    p.add_argument("--lr-volume", type=float, default=DEFAULT_LR_VOLUME)
    p.add_argument("--lr-guide", type=float, default=DEFAULT_LR_GUIDE)
    p.add_argument("--lr-adapter", type=float, default=DEFAULT_LR_ADAPTER)
    p.add_argument("--lr-lora", type=float, default=DEFAULT_LR_LORA)

    # Architecture
    p.add_argument("--adapter-rank", type=int, default=DEFAULT_ADAPTER_RANK)
    p.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    p.add_argument("--depth-bins", type=int, default=DEFAULT_DEPTH_BINS,
                   help="Number of depth bins K (default: 8)")
    p.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    p.add_argument("--spatial-h", type=int, default=DEFAULT_SPATIAL_H)
    p.add_argument("--spatial-w", type=int, default=DEFAULT_SPATIAL_W)

    # Injection and update configs
    p.add_argument("--inject-config", type=str, default="mid_up2",
                   choices=["mid_only", "mid_up2", "multiscale"],
                   help="Guide injection points")
    p.add_argument("--update-schedule", type=str, default="hybrid",
                   choices=["fixed_once", "hybrid", "every_step"],
                   help="Volume recomputation schedule during eval rollout")

    # Loss weights
    p.add_argument("--la-diff", type=float, default=DEFAULT_LA_DIFF)
    p.add_argument("--la-vol", type=float, default=DEFAULT_LA_VOL)

    # Collision augmentation
    p.add_argument("--collision-prob", type=float, default=DEFAULT_COLLISION_PROB)
    p.add_argument("--collision-ov-min", type=float, default=DEFAULT_COLLISION_OV_MIN)
    p.add_argument("--collision-ov-max", type=float, default=DEFAULT_COLLISION_OV_MAX)

    # General
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--t-max", type=int, default=500)
    p.add_argument("--n-steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--val-frac", type=float, default=VAL_FRAC)

    args = p.parse_args()
    train_phase62(args)


if __name__ == "__main__":
    main()
