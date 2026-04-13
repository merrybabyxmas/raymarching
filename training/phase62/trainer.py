"""
Phase 62 — Trainer
====================

Main training loop for Phase 62: Entity Volume + First-Hit Projection.

Training step flow:
  1. Load sample, encode to latents, add noise
  2. Build V_gt from depth + masks (real 3D data)
  3. UNet forward pass #1 (no guide, extract features)
  4. EntityVolumePredictor: features -> V_logits
  5. L_volume_ce = CE(V_logits, V_gt)
  6. FirstHitProjector: V_logits -> visible_class, visible_probs
  7. GuideFeatureAssembler: build guide from visible + entity features
  8. UNet forward pass #2 (with guide injection)
  9. L_diffusion = MSE(noise_pred, noise)
  10. loss = la_diff * L_diffusion + la_vol * L_volume_ce
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from training.phase62.losses import (
    loss_diffusion,
    loss_projected_balance,
    loss_projected_global,
    loss_volume_ce,
    compute_volume_accuracy,
)
from training.phase62.evaluator import Phase62Evaluator, _encode_text, _get_entity_tokens_with_fallback
from training.phase62.rollout import Phase62RolloutRunner
from data.phase62.volume_gt_builder import VolumeGTBuilder
from scripts.train_animatediff_vca import encode_frames_to_latents
from scripts.train_phase39 import (
    compute_dataset_overlap_scores,
    split_train_val,
    make_sampling_weights,
)


MIN_VAL_SAMPLES = 3


def _try_collision_augment(
    dataset, sample_idx: int, sample,
    sample_dir: str | Path | None = None,
    overlap_min: float = 0.08, overlap_max: float = 0.25,
    max_shift: int = 96, max_tries: int = 24, min_accept: float = 0.04,
):
    """Build collision-heavy synthetic sample from solo frames + raw masks."""
    # Handle both dict and tuple samples
    if isinstance(sample, dict):
        frames_np = sample["frames"]
        depth_orders = sample["depth_orders"]
        entity_masks = sample["entity_masks"]
        solo_e0 = sample.get("solo_e0")
        solo_e1 = sample.get("solo_e1")
    elif len(sample) >= 8:
        frames_np, _, depth_orders = sample[0], sample[1], sample[2]
        entity_masks = sample[4]
        solo_e0 = sample[6]
        solo_e1 = sample[7]
    else:
        return None

    seq_dir = Path(sample_dir) if sample_dir is not None else None
    if seq_dir is None:
        raw_dataset = dataset.raw_dataset() if hasattr(dataset, 'raw_dataset') else dataset
        if not hasattr(raw_dataset, "samples") or sample_idx >= len(raw_dataset.samples):
            return None
        seq_dir = raw_dataset.samples[sample_idx].get("dir", None)
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

        shifted_frames, shifted_masks, shifted_visible, shifted_depths = [], [], [], []
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
            m0_bin = (m0_ds > 0.5).astype(np.float32)
            m1_bin = (m1_ds > 0.5).astype(np.float32)
            if front == 0:
                v0 = m0_bin
                v1 = m1_bin * (1.0 - m0_bin)
            else:
                v1 = m1_bin
                v0 = m0_bin * (1.0 - m1_bin)
            shifted_masks.append(np.stack(
                [m0_ds.flatten(), m1_ds.flatten()], axis=0))
            shifted_visible.append(np.stack(
                [v0.flatten(), v1.flatten()], axis=0))

        masks_arr = np.stack(shifted_masks, axis=0)
        visible_arr = np.stack(shifted_visible, axis=0)
        ov = float((masks_arr[:, 0, :] * masks_arr[:, 1, :]).mean())

        if ov < min_accept:
            continue

        target_ov = (overlap_min + overlap_max) / 2.0
        err = abs(ov - target_ov)
        if err < best_err:
            best_err = err
            best = (np.stack(shifted_frames, axis=0), shifted_depths, masks_arr, visible_arr)

    return best


class Phase62Trainer:
    """
    Main training loop for Phase 62.

    Orchestrates:
      - Data loading with overlap-weighted sampling
      - Two-pass UNet training (extract features -> inject guide)
      - Volume CE + diffusion loss
      - Periodic validation + GIF generation
      - Checkpoint saving (best + latest)

    Usage:
        trainer = Phase62Trainer(config, pipe, system, backbone_mgr, dataset, device)
        trainer.train()
    """

    def __init__(
        self,
        config,
        pipe,
        system,         # Phase62System
        backbone_mgr,   # BackboneManager
        dataset,        # Phase62DatasetAdapter or raw dataset
        device: str,
    ):
        self.config = config
        self.pipe = pipe
        self.system = system
        self.backbone_mgr = backbone_mgr
        self.dataset = dataset
        self.device = device

        # Config unpacking
        self.train_cfg = config.training
        self.eval_cfg = config.eval
        self.data_cfg = config.data

        # GT builder
        self.gt_builder = VolumeGTBuilder(
            depth_bins=config.depth_bins,
            spatial_h=config.spatial_h,
            spatial_w=config.spatial_w,
            render_resolution=getattr(self.data_cfg, "volume_gt_render_resolution", 128),
        )

        # Evaluator + rollout runner
        self.evaluator = Phase62Evaluator()
        self.rollout_runner = Phase62RolloutRunner()

        # Directories
        self.debug_dir = Path(getattr(config, 'debug_dir', 'outputs/phase62_debug'))
        self.save_dir = Path(getattr(config, 'save_dir', 'checkpoints/phase62'))
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.entity_density_cache_path = self.save_dir / "entity_density_scores.npy"

        # Train/val split
        raw_ds = dataset.raw_dataset() if hasattr(dataset, 'raw_dataset') else dataset
        overlap_scores = compute_dataset_overlap_scores(raw_ds)
        val_frac = getattr(self.data_cfg, 'val_frac', 0.2)
        self.train_idx, self.val_idx = split_train_val(
            overlap_scores, val_frac=val_frac, min_val=MIN_VAL_SAMPLES)
        self.entity_density_scores = self._load_entity_density_scores(self.train_idx)
        self.sample_weights = self._make_combined_sampling_weights(
            overlap_scores, self.train_idx, self.entity_density_scores)

        # Class weights for volume CE.
        # Strongly down-weight background so volume learning cannot win by
        # predicting all-background voxels.
        bg_w = float(getattr(self.train_cfg, "bg_class_weight", 0.02))
        ent_w = float(getattr(self.train_cfg, "entity_class_weight", 1.0))
        self.class_weights = torch.tensor(
            [bg_w, ent_w, ent_w], device=device, dtype=torch.float32)
        self.ignore_background_voxels = bool(
            getattr(self.train_cfg, "ignore_background_voxels", False))
        self.volume_ce_warmup_epochs = int(
            getattr(self.train_cfg, "volume_ce_warmup_epochs", 0))
        self.entity_voxel_boost = float(
            getattr(self.train_cfg, "entity_voxel_boost", 1.0))
        self.entity_voxel_boost_warmup = float(
            getattr(self.train_cfg, "entity_voxel_boost_warmup", self.entity_voxel_boost))
        self.stage1_epochs = int(getattr(self.train_cfg, "stage1_epochs", self.volume_ce_warmup_epochs))
        self.stage2_epochs = int(getattr(self.train_cfg, "stage2_epochs", max(self.stage1_epochs + 1, self.train_cfg.epochs - 1)))
        self.stage2_volume_lr_scale = float(getattr(self.train_cfg, "stage2_volume_lr_scale", 0.0))
        self.stage3_volume_lr_scale = float(getattr(self.train_cfg, "stage3_volume_lr_scale", 0.25))
        self.stage2_la_vol_scale = float(getattr(self.train_cfg, "stage2_la_vol_scale", 0.25))
        self.stage3_la_vol_scale = float(getattr(self.train_cfg, "stage3_la_vol_scale", 1.0))
        self.la_global = float(getattr(self.train_cfg, "la_global", 1.0))
        self.la_balance = float(getattr(self.train_cfg, "la_balance", 0.5))
        self.volume_negative_weight = float(getattr(self.train_cfg, "volume_negative_weight", 0.1))

        # Optimizer setup
        self._setup_optimizer()

        # History
        self.history: List[Dict] = []
        self.best_val_score = -1.0
        self.best_epoch = -1

    def _load_entity_density_scores(self, train_idx: list) -> np.ndarray | None:
        """Load actual entity voxel density cache if present, else return None."""
        if self.entity_density_cache_path.exists():
            try:
                cached = np.load(self.entity_density_cache_path)
                if cached.shape[0] == len(train_idx):
                    return cached.astype(np.float32, copy=False)
            except Exception:
                pass
        print(
            "  [warn] entity density cache missing; using overlap-only sampling. "
            "Run scripts/precompute_phase62_density.py to build the actual cache.",
            flush=True,
        )
        return None

    def _make_combined_sampling_weights(
        self,
        overlap_scores: np.ndarray,
        train_idx: list,
        entity_density_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        """Combine overlap-heavy and entity-rich sampling biases."""
        overlap_w = make_sampling_weights(train_idx, overlap_scores)
        if entity_density_scores is None or len(entity_density_scores) != len(train_idx):
            return overlap_w

        density = entity_density_scores.astype(np.float32)
        density = density - density.min()
        density = density / (density.max() - density.min() + 1e-8)
        density_sum = max(float(density.sum()), 1e-8)
        density_w = density / density_sum

        mix = float(getattr(self.train_cfg, "entity_density_sampling_weight", 1.0))
        combined = (overlap_w + mix * density_w)
        combined = combined / max(float(combined.sum()), 1e-8)
        return combined

    def _setup_optimizer(self) -> None:
        """Set up optimizer and LR scheduler."""
        volume_params = self.system.volume_params()
        assembler_params = self.system.assembler_params()
        adapter_params = self.backbone_mgr.adapter_params()
        lora_params = self.backbone_mgr.lora_params()
        for p in volume_params:
            p.requires_grad_(True)
        for p in assembler_params:
            p.requires_grad_(True)
        for p in adapter_params:
            p.requires_grad_(True)
        for p in lora_params:
            p.requires_grad_(True)

        self.param_groups = {
            "volume": volume_params,
            "assembler": assembler_params,
            "adapter": adapter_params,
            "lora": lora_params,
        }

        self.optimizer = optim.AdamW([
            {"params": volume_params, "lr": self.train_cfg.lr_volume, "name": "volume_pred"},
            {"params": assembler_params, "lr": self.train_cfg.lr_guide, "name": "assembler"},
            {"params": adapter_params, "lr": self.train_cfg.lr_adapter, "name": "adapters"},
            {"params": lora_params, "lr": self.train_cfg.lr_lora, "name": "lora"},
        ], weight_decay=1e-4)
        for group in self.optimizer.param_groups:
            group["initial_lr"] = group["lr"]

        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.train_cfg.epochs),
            eta_min=1e-6,
        )

    def _set_epoch_trainability(self, epoch: int) -> None:
        """Three-stage training schedule."""
        stage = self._get_stage(epoch)
        warmup = stage == "stage1"

        for p in self.param_groups["volume"]:
            p.requires_grad_(stage != "stage2" or self.stage2_volume_lr_scale > 0.0)
        for p in self.param_groups["assembler"]:
            p.requires_grad_(stage != "stage1")
        for p in self.param_groups["adapter"]:
            p.requires_grad_(stage != "stage1")
        for p in self.param_groups["lora"]:
            p.requires_grad_(stage != "stage1")

        for group in self.optimizer.param_groups:
            name = group.get("name", "")
            base_lr = group.get("initial_lr", group["lr"])
            if name == "volume_pred":
                if stage == "stage1":
                    group["lr"] = base_lr
                elif stage == "stage2":
                    group["lr"] = base_lr * self.stage2_volume_lr_scale
                else:
                    group["lr"] = base_lr * self.stage3_volume_lr_scale
            else:
                group["lr"] = 0.0 if warmup else base_lr

    def _get_stage(self, epoch: int) -> str:
        if epoch < self.stage1_epochs:
            return "stage1"
        if epoch < self.stage2_epochs:
            return "stage2"
        return "stage3"

    def _build_gt_visible_tensor(
        self,
        src_masks: np.ndarray,   # (T, 2, S)
        B_feat: int,
    ) -> torch.Tensor:
        """Resize GT visible masks to the feature lattice for differentiable 2D loss."""
        T = min(int(src_masks.shape[0]), int(B_feat))
        masks = src_masks[:T]
        S = masks.shape[-1]
        hw = int(round(S ** 0.5))
        gt = torch.from_numpy(masks.astype(np.float32)).to(self.device)
        gt = gt.reshape(T, 2, hw, hw)
        if hw != self.config.spatial_h or hw != self.config.spatial_w:
            gt = F.interpolate(
                gt,
                size=(self.config.spatial_h, self.config.spatial_w),
                mode="bilinear",
                align_corners=False,
            )
        return gt.clamp(0.0, 1.0)

    def _unpack_sample(self, sample):
        """Unpack sample into components (handles dict and tuple)."""
        if isinstance(sample, dict):
            return (
                sample["frames"],
                sample["depth"],
                sample["depth_orders"],
                sample["meta"],
                sample.get("sample_dir"),
                sample["entity_masks"],
                sample.get("visible_masks"),
                sample.get("solo_e0"),
                sample.get("solo_e1"),
            )
        elif len(sample) >= 8:
            return sample[0], sample[1], sample[2], sample[3], None, sample[4], sample[5], sample[6], sample[7]
        else:
            frames, depth, orders, meta, masks = sample[:5]
            return frames, depth, orders, meta, None, masks, None, None, None

    def _train_step(
        self,
        sample,
        data_idx: int,
        epoch: int,
    ) -> Optional[Dict]:
        """
        Execute one training step.

        Returns dict with step metrics or None on failure.
        """
        frames_np, depth_np, depth_orders, meta, sample_dir, entity_masks, visible_masks, solo_e0, solo_e1 = \
            self._unpack_sample(sample)

        # Collision augmentation
        collision_aug = False
        collision_prob = getattr(self.train_cfg, 'collision_prob', 0.5)
        if collision_prob > 0.0 and solo_e0 is not None and solo_e1 is not None:
            if np.random.rand() < collision_prob:
                # Build augmented sample tuple for collision function
                if isinstance(sample, dict):
                    sample_tuple = (
                        frames_np, depth_np, depth_orders, meta, entity_masks,
                        sample.get("visible_masks"), solo_e0, solo_e1)
                else:
                    sample_tuple = sample
                aug = _try_collision_augment(
                    self.dataset, data_idx, sample_tuple,
                    sample_dir=sample_dir,
                    overlap_min=getattr(self.train_cfg, 'collision_ov_min', 0.005),
                    overlap_max=getattr(self.train_cfg, 'collision_ov_max', 0.25),
                    min_accept=getattr(self.train_cfg, 'collision_min_accept', 0.01),
                    max_tries=getattr(self.train_cfg, 'collision_max_tries', 48),
                )
                if aug is not None:
                    frames_np, depth_orders, entity_masks, visible_masks = aug
                    collision_aug = True

        # Entity tokens
        toks_e0_t, toks_e1_t, full_prompt = _get_entity_tokens_with_fallback(
            self.pipe, meta, self.device)
        self.backbone_mgr.set_entity_tokens(toks_e0_t, toks_e1_t)
        self.backbone_mgr.reset_slot_store()

        # Encode frames + noise
        with torch.no_grad():
            latents = encode_frames_to_latents(self.pipe, frames_np, self.device)
        noise = torch.randn_like(latents)
        t = torch.randint(0, self.train_cfg.t_max, (1,), device=self.device).long()
        noisy = self.pipe.scheduler.add_noise(latents, noise, t)

        T_frames = min(frames_np.shape[0], entity_masks.shape[0], len(depth_orders))

        enc_full = _encode_text(self.pipe, full_prompt, self.device)
        B_noisy = noisy.shape[0]
        if enc_full.shape[0] == 1 and B_noisy > 1:
            enc_full = enc_full.expand(B_noisy, -1, -1)

        self.optimizer.zero_grad()

        # --- Pass 1: UNet forward (no guide) to extract features ---
        self.system.clear_guides()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = self.pipe.unet(
                noisy, t, encoder_hidden_states=enc_full).sample

        # Extract features from primary backbone
        F_g = self.backbone_mgr.primary.last_Fg  # (B, S, D)
        F_0 = self.backbone_mgr.primary.last_F0
        F_1 = self.backbone_mgr.primary.last_F1

        if F_g is None or F_0 is None or F_1 is None:
            return None

        # --- Volume prediction ---
        V_logits = self.system.predict_volume(F_g, F_0, F_1)  # (B, 3, K, H, W)

        # --- Build V_gt ---
        depth_maps = depth_np[:T_frames] if depth_np is not None else None
        if depth_maps is not None:
            V_gt_np = self.gt_builder.build_batch(
                depth_maps, entity_masks[:T_frames],
                depth_orders[:T_frames],
                visible_masks=(visible_masks[:T_frames] if visible_masks is not None else None),
                meta=meta,
                sample_dir=sample_dir,
            )
            V_gt = torch.from_numpy(V_gt_np).to(self.device).long()  # (T, K, H, W)
        else:
            # Fallback: all background (degenerate case for collision aug)
            V_gt = torch.zeros(
                T_frames, self.config.depth_bins,
                self.config.spatial_h, self.config.spatial_w,
                dtype=torch.long, device=self.device)

        # Match batch dimension
        B_feat = V_logits.shape[0]
        if V_gt.shape[0] < B_feat:
            n_rep = max(1, B_feat // V_gt.shape[0])
            V_gt = V_gt.repeat(n_rep, 1, 1, 1)[:B_feat]

        # --- L_volume_ce ---
        if self.ignore_background_voxels:
            voxel_weights = torch.full_like(V_gt, float(self.volume_negative_weight), dtype=torch.float32)
            voxel_weights = torch.where(V_gt > 0, torch.ones_like(voxel_weights), voxel_weights)
        else:
            voxel_weights = torch.ones_like(V_gt, dtype=torch.float32, device=self.device)

        boost = self.entity_voxel_boost_warmup if epoch < self.volume_ce_warmup_epochs else self.entity_voxel_boost
        if boost != 1.0:
            voxel_weights = torch.where(
                V_gt > 0,
                torch.full_like(voxel_weights, float(boost)),
                voxel_weights,
            )

        l_vol = loss_volume_ce(
            V_logits,
            V_gt,
            class_weights=self.class_weights,
            voxel_weights=voxel_weights,
        )

        stage = self._get_stage(epoch)
        use_diffusion = stage != "stage1"

        # Always run projection for balance loss (even in stage1)
        visible_class, front_probs, back_probs, guides = self.system.project_and_assemble(
            V_logits, F_g, F_0, F_1)

        src_masks = visible_masks if visible_masks is not None else entity_masks
        gt_visible = self._build_gt_visible_tensor(src_masks, B_feat)
        l_global = loss_projected_global(front_probs[:gt_visible.shape[0]], gt_visible)
        l_balance = loss_projected_balance(front_probs[:gt_visible.shape[0]], gt_visible)

        if use_diffusion:
            self.system.set_guides(guides)

            # --- Pass 2: UNet forward (with guide injection) ---
            self.backbone_mgr.reset_slot_store()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = self.pipe.unet(
                    noisy, t, encoder_hidden_states=enc_full).sample

            noise_t = noise[:, :, :T_frames].float()
            noise_pred_t = noise_pred[:, :, :T_frames].float()
            l_diff = loss_diffusion(noise_pred_t, noise_t)
        else:
            self.system.clear_guides()
            l_diff = torch.tensor(0.0, device=self.device)

        # --- Total loss ---
        if stage == "stage1":
            loss = self.train_cfg.la_vol * l_vol + self.la_balance * l_balance
        elif stage == "stage2":
            loss = (
                self.train_cfg.la_diff * l_diff
                + self.la_global * l_global
                + self.la_balance * l_balance
                + self.train_cfg.la_vol * self.stage2_la_vol_scale * l_vol
            )
        else:
            loss = (
                self.train_cfg.la_diff * l_diff
                + self.la_global * l_global
                + self.la_balance * l_balance
                + self.train_cfg.la_vol * self.stage3_la_vol_scale * l_vol
            )

        if not torch.isfinite(loss):
            self.system.clear_guides()
            return None

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.param_groups["volume"], max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.param_groups["assembler"], max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.param_groups["adapter"], max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.param_groups["lora"], max_norm=0.5)

        self.optimizer.step()
        self.system.clear_guides()

        # Metrics
        with torch.no_grad():
            acc = compute_volume_accuracy(V_logits, V_gt)

        return {
            "loss": float(loss.item()),
            "l_diff": float(l_diff.item()),
            "l_vol": float(l_vol.item()),
            "l_global": float(l_global.item()),
            "l_balance": float(l_balance.item()),
            "warmup": not use_diffusion,
            "acc_overall": acc["overall_acc"],
            "acc_entity": acc["entity_acc"],
            "collision_aug": collision_aug,
            "stage": stage,
        }

    def _eval_epoch(self, epoch: int) -> Dict:
        """Run validation + GIF generation for an epoch."""
        self.system.eval()
        self.backbone_mgr.eval()

        # Validation metrics
        val_m = self.evaluator.evaluate(
            self.pipe, self.system, self.backbone_mgr,
            self.dataset, self.val_idx, self.device, self.config)

        vs = val_m["val_score"]
        print(
            f"  [val] val_score={vs:.4f}  "
            f"diff_mse={val_m['val_diff_mse']:.4f}  "
            f"vol_ce={val_m['val_vol_ce']:.4f}  "
            f"acc_all={val_m['val_acc_overall']:.4f}  "
            f"acc_ent={val_m['val_acc_entity']:.4f}  "
            f"iou_e0={val_m['val_iou_e0']:.4f}  "
            f"iou_e1={val_m['val_iou_e1']:.4f}  "
            f"iou_min={val_m.get('val_iou_min', 0.0):.4f}  "
            f"n={val_m['n_samples']}",
            flush=True)

        # GIF generation
        probe_rollout_mse = None
        probe_rollout_score = None

        try:
            probe = self.dataset[self.val_idx[0]]
            if isinstance(probe, dict):
                probe_meta = probe["meta"]
                probe_masks = probe["entity_masks"]
                probe_frames = probe["frames"]
            elif len(probe) >= 8:
                probe_meta = probe[3]
                probe_masks = probe[4]
                probe_frames = probe[0]
            else:
                probe_meta = probe[3]
                probe_masks = probe[4]
                probe_frames = probe[0]

            toks_e0_pt, toks_e1_pt, probe_prompt = \
                _get_entity_tokens_with_fallback(self.pipe, probe_meta, self.device)

            rollout_result = self.rollout_runner.generate_rollout(
                self.pipe, self.system, self.backbone_mgr,
                probe_prompt, self.config, self.device,
                toks_e0=toks_e0_pt, toks_e1=toks_e1_pt,
                entity_masks=probe_masks,
                gt_frames=probe_frames,
            )

            # Save GIF
            paths = self.rollout_runner.save_rollout(
                rollout_result, self.debug_dir,
                prefix=f"eval_epoch{epoch:03d}")
            for name, path in paths.items():
                print(f"  [gif] {name}: {path}", flush=True)

            probe_rollout_mse = rollout_result.get("probe_mse")
            probe_rollout_score = rollout_result.get("probe_score")
            if probe_rollout_mse is not None:
                print(
                    f"  [rollout] probe_rgb_mse={probe_rollout_mse:.4f}  "
                    f"probe_score={probe_rollout_score:.4f}",
                    flush=True)

        except Exception as e:
            print(f"  [warn] GIF failed: {e}", flush=True)
            import traceback; traceback.print_exc()

        # Selection score
        min_iou = float(val_m.get("val_iou_min", min(val_m.get("val_iou_e0", 0.0), val_m.get("val_iou_e1", 0.0))))
        selection_score = 0.85 * vs + 0.15 * min_iou
        if probe_rollout_score is not None:
            selection_score = 0.75 * vs + 0.15 * min_iou + 0.10 * probe_rollout_score
            print(
                f"  [select] score={selection_score:.4f}  "
                f"(val={vs:.4f}, min_iou={min_iou:.4f}, rollout={probe_rollout_score:.4f})",
                flush=True)

        val_m["probe_rollout_mse"] = probe_rollout_mse
        val_m["probe_rollout_score"] = probe_rollout_score
        val_m["selection_score"] = selection_score

        return val_m

    def _save_checkpoint(self, epoch: int, val_m: Dict) -> None:
        """Save checkpoint (latest + conditionally best)."""
        selection_score = val_m.get("selection_score", val_m.get("val_score", 0.0))

        ckpt_data = {
            "epoch": epoch,
            "val_score": val_m.get("val_score", 0.0),
            "selection_score": selection_score,
            "inject_config": self.config.inject_config,
            "update_schedule": self.config.update_schedule,
            "depth_bins": self.config.depth_bins,
            "hidden_dim": self.config.hidden_dim,
            "spatial_h": self.config.spatial_h,
            "spatial_w": self.config.spatial_w,
            "system_state": {
                "volume_pred": self.system.volume_pred.state_dict(),
                "assembler": self.system.assembler.state_dict(),
            },
            "extractors_state": [],
        }

        for ext in self.backbone_mgr.extractors:
            es = {
                "lora_k": ext.lora_k.state_dict(),
                "lora_v": ext.lora_v.state_dict(),
                "lora_out": ext.lora_out.state_dict(),
                "slot0_adapter": ext.slot0_adapter.state_dict(),
                "slot1_adapter": ext.slot1_adapter.state_dict(),
            }
            ckpt_data["extractors_state"].append(es)

        torch.save(ckpt_data, str(self.save_dir / "latest.pt"))

        if selection_score > self.best_val_score:
            self.best_val_score = selection_score
            self.best_epoch = epoch
            torch.save(ckpt_data, str(self.save_dir / "best.pt"))
            print(
                f"  * best epoch={self.best_epoch} "
                f"val_score={val_m.get('val_score', 0):.4f} "
                f"selection={selection_score:.4f} "
                f"-> {self.save_dir}/best.pt", flush=True)

    def load_checkpoint(self, ckpt_path: str) -> None:
        """Load checkpoint and restore model states."""
        print(f"[Phase 62] Loading checkpoint: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "system_state" in ckpt:
            if "volume_pred" in ckpt["system_state"]:
                try:
                    self.system.volume_pred.load_state_dict(
                        ckpt["system_state"]["volume_pred"], strict=False)
                except Exception as e:
                    print(f"  [warn] volume_pred ckpt incompatible, skipping: {e}", flush=True)
            if "assembler" in ckpt["system_state"]:
                try:
                    self.system.assembler.load_state_dict(
                        ckpt["system_state"]["assembler"], strict=False)
                except Exception as e:
                    print(f"  [warn] assembler ckpt incompatible, skipping: {e}", flush=True)
        # Backward compat with old checkpoint format
        elif "volume_pred" in ckpt:
            try:
                self.system.volume_pred.load_state_dict(ckpt["volume_pred"], strict=False)
            except Exception as e:
                print(f"  [warn] old volume_pred ckpt incompatible, skipping: {e}", flush=True)
        if "guide_injector" in ckpt:
            # Old format: try to load into assembler
            try:
                self.system.assembler.load_state_dict(ckpt["guide_injector"], strict=False)
            except Exception:
                print("  [warn] guide_injector ckpt incompatible with assembler, skipping", flush=True)

        if "extractors_state" in ckpt:
            for i, es in enumerate(ckpt["extractors_state"]):
                if i < len(self.backbone_mgr.extractors):
                    ext = self.backbone_mgr.extractors[i]
                    if "lora_k" in es:
                        ext.lora_k.load_state_dict(es["lora_k"])
                    if "lora_v" in es:
                        ext.lora_v.load_state_dict(es["lora_v"])
                    if "lora_out" in es:
                        ext.lora_out.load_state_dict(es["lora_out"])
                    if "slot0_adapter" in es:
                        ext.slot0_adapter.load_state_dict(es["slot0_adapter"])
                    if "slot1_adapter" in es:
                        ext.slot1_adapter.load_state_dict(es["slot1_adapter"])
        # Backward compat with old format
        elif "procs_state" in ckpt:
            for i, ps in enumerate(ckpt["procs_state"]):
                if i < len(self.backbone_mgr.extractors):
                    ext = self.backbone_mgr.extractors[i]
                    if "lora_k" in ps:
                        ext.lora_k.load_state_dict(ps["lora_k"])
                    if "lora_v" in ps:
                        ext.lora_v.load_state_dict(ps["lora_v"])
                    if "lora_out" in ps:
                        ext.lora_out.load_state_dict(ps["lora_out"])
                    if "slot0_adapter" in ps:
                        ext.slot0_adapter.load_state_dict(ps["slot0_adapter"])
                    if "slot1_adapter" in ps:
                        ext.slot1_adapter.load_state_dict(ps["slot1_adapter"])

        print("[Phase 62] Checkpoint loaded.", flush=True)

    def train(self) -> None:
        """Run the full training loop."""
        device = self.device
        epochs = self.train_cfg.epochs
        steps_per_epoch = self.train_cfg.steps_per_epoch
        eval_every = self.train_cfg.eval_every

        print(f"[Phase 62] Starting training: {epochs} epochs, "
              f"{steps_per_epoch} steps/epoch, "
              f"train={len(self.train_idx)} val={len(self.val_idx)}", flush=True)

        # Epoch 0 validation
        print("\n[Phase 62] Epoch 0 validation...", flush=True)
        val_m0 = self.evaluator.evaluate(
            self.pipe, self.system, self.backbone_mgr,
            self.dataset, self.val_idx, device, self.config)
        print(f"  [epoch0] val_score={val_m0['val_score']:.4f}  "
              f"diff_mse={val_m0['val_diff_mse']:.4f}  "
              f"vol_ce={val_m0['val_vol_ce']:.4f}  "
              f"acc_all={val_m0['val_acc_overall']:.4f}  "
              f"acc_ent={val_m0['val_acc_entity']:.4f}", flush=True)

        for epoch in range(epochs):
            self.system.train()
            self.backbone_mgr.train()
            self._set_epoch_trainability(epoch)

            epoch_metrics: Dict[str, List[float]] = {
                "total": [], "diffusion": [], "volume_ce": [],
                "acc_overall": [], "acc_entity": [], "global": [], "balance": [],
                "warmup_steps": [],
            }
            n_collision_aug = 0

            # Sample training indices
            chosen = np.random.choice(
                len(self.train_idx), size=steps_per_epoch,
                replace=True, p=self.sample_weights)
            step_indices = [self.train_idx[ci] for ci in chosen]

            for batch_idx, data_idx in enumerate(step_indices):
                sample = self.dataset[data_idx]
                step_result = self._train_step(sample, data_idx, epoch)

                if step_result is None:
                    print(f"  [warn] ep={epoch} step={batch_idx}: skipped", flush=True)
                    continue

                epoch_metrics["total"].append(step_result["loss"])
                epoch_metrics["diffusion"].append(step_result["l_diff"])
                epoch_metrics["volume_ce"].append(step_result["l_vol"])
                epoch_metrics["global"].append(step_result["l_global"])
                epoch_metrics["balance"].append(step_result["l_balance"])
                epoch_metrics["acc_overall"].append(step_result["acc_overall"])
                epoch_metrics["acc_entity"].append(step_result["acc_entity"])
                epoch_metrics["warmup_steps"].append(1.0 if step_result.get("warmup") else 0.0)
                if step_result["collision_aug"]:
                    n_collision_aug += 1

            self.lr_scheduler.step()

            # Epoch summary
            avg = {k: float(np.mean(v)) if v else 0.0
                   for k, v in epoch_metrics.items()}

            print(
                f"[Phase 62] epoch {epoch:03d}/{epochs - 1}  "
                f"loss={avg['total']:.4f}  "
                f"diff={avg['diffusion']:.4f}  "
                f"vol_ce={avg['volume_ce']:.4f}  "
                f"glob={avg['global']:.4f}  "
                f"bal={avg['balance']:.4f}  "
                f"acc_all={avg['acc_overall']:.4f}  "
                f"acc_ent={avg['acc_entity']:.4f}  "
                f"warmup={avg['warmup_steps']:.2f}  "
                f"aug={n_collision_aug}  "
                f"stage={self._get_stage(epoch)}",
                flush=True)

            # Validation
            should_eval = (epoch % eval_every == 0) or (epoch == epochs - 1)
            if should_eval:
                val_m = self._eval_epoch(epoch)

                self.history.append({
                    "epoch": epoch,
                    **val_m,
                    **avg,
                    "collision_aug_count": n_collision_aug,
                })

                self._save_checkpoint(epoch, val_m)

                with open(self.save_dir / "history.json", "w") as f:
                    json.dump(self.history, f, indent=2)

                self.system.train()
                self.backbone_mgr.train()

        print(f"\n[Phase 62] Done. best epoch={self.best_epoch} "
              f"val_score={self.best_val_score:.4f}", flush=True)
