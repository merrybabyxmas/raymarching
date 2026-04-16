"""
training/phase64/stage2_train_decoder.py
==========================================
Stage 2: Train StructuredDecoder from frozen ScenePriorModule.

Proves: structured scene outputs → plausible coarse RGB
without any diffusion backbone.

If Stage 2 fails (composite PSNR < threshold), do NOT proceed to Stage 3.

Usage:
  python -m training.phase64.stage2_train_decoder \\
      --scene_prior_ckpt outputs/phase64/stage1/best_scene_prior.pt \\
      --data_root toy/data_objaverse
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.phase64.phase64_dataset import Phase64Dataset, Phase64Sample
from training.phase64.evaluator_phase64 import Phase64Evaluator


# --------------------------------------------------------------------------- #
#  StructuredDecoder — lightweight backbone-agnostic RGB decoder
# --------------------------------------------------------------------------- #

class StructuredDecoder(nn.Module):
    """
    Lightweight convolutional decoder: SceneOutputs (8 channels) → coarse RGB.

    Takes the canonical 8-channel scene representation from SceneOutputs and
    decodes it to a (3, H, W) image at the same spatial resolution as the
    scene prior.

    In Stage 2 the ScenePriorModule is frozen; only this decoder is trained.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden: int = 64,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, scene_tensor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        scene_tensor : (B, 8, H, W) from SceneOutputs.to_canonical_tensor()

        Returns
        -------
        (B, 3, H, W) predicted RGB in [0, 1]
        """
        return self.net(scene_tensor)


# --------------------------------------------------------------------------- #
#  Loss helpers
# --------------------------------------------------------------------------- #

def reconstruction_loss(
    pred_rgb: torch.Tensor,   # (B, 3, H, W) in [0, 1]
    gt_rgb: torch.Tensor,     # (B, 3, H, W) in [0, 1]
) -> torch.Tensor:
    """L1 + perceptual proxy (L2 in feature space approximated by L2 on RGB)."""
    l1 = F.l1_loss(pred_rgb, gt_rgb)
    l2 = F.mse_loss(pred_rgb, gt_rgb)
    return l1 + 0.5 * l2


def isolation_loss(
    pred_rgb: torch.Tensor,   # (B, 3, H, W)
    vis_e0: torch.Tensor,     # (B, 1, H, W) or (B, H, W) — entity 0 mask
    vis_e1: torch.Tensor,     # (B, 1, H, W) or (B, H, W) — entity 1 mask
) -> torch.Tensor:
    """
    Both entities must contribute non-trivially to the predicted image.

    Penalises the model when either entity region is blank (all-zero or all-one).
    """
    if vis_e0.dim() == 3:
        vis_e0 = vis_e0.unsqueeze(1)
    if vis_e1.dim() == 3:
        vis_e1 = vis_e1.unsqueeze(1)

    # Mean brightness under each entity mask
    eps = 1e-8
    bright_e0 = (pred_rgb * vis_e0).sum(dim=(2, 3)) / (vis_e0.sum(dim=(2, 3)) + eps)
    bright_e1 = (pred_rgb * vis_e1).sum(dim=(2, 3)) / (vis_e1.sum(dim=(2, 3)) + eps)

    # Both entities must be non-trivial (> 0.05 mean brightness)
    thresh = 0.05
    penalty = (F.relu(thresh - bright_e0.mean()) +
               F.relu(thresh - bright_e1.mean()))
    return penalty


# --------------------------------------------------------------------------- #
#  Stage2Trainer
# --------------------------------------------------------------------------- #

class Stage2Trainer:
    """
    Trains StructuredDecoder from frozen ScenePriorModule.

    Parameters
    ----------
    config             : config object (model + training sections)
    dataset            : Phase64Dataset
    splits             : dict from make_splits() with 'train'/'val' keys
                         (optional; if omitted the dataset must expose
                         ``get_split_indices()``)
    stage1_ckpt        : path to saved Stage 1 scene prior checkpoint
                         (alias: scene_prior_ckpt for backward compat)
    device             : torch device string or torch.device
    """

    def __init__(
        self,
        config,
        dataset: Phase64Dataset,
        splits: Optional[Dict] = None,
        stage1_ckpt: str = "",
        device: str = "cuda",
        # backward-compat alias
        scene_prior_ckpt: str = "",
    ) -> None:
        self.config  = config
        self.dataset = dataset
        self.device  = torch.device(device) if isinstance(device, str) else device
        self._splits = splits  # may be None if dataset has get_split_indices()

        # Resolve checkpoint path (new name takes priority)
        _stage1_ckpt = stage1_ckpt or scene_prior_ckpt

        from scene_prior import ScenePriorModule, EntityRenderer

        model_cfg = config.model
        self.scene_prior = ScenePriorModule(
            depth_bins=model_cfg.depth_bins,
            hidden_dim=model_cfg.hidden_dim,
            id_dim=model_cfg.id_dim,
            pose_dim=model_cfg.pose_dim,
            spatial_h=model_cfg.spatial_h,
            spatial_w=model_cfg.spatial_w,
            slot_dim=getattr(model_cfg, "slot_dim", 64),
        ).to(device)

        self.renderer = EntityRenderer(depth_bins=model_cfg.depth_bins).to(device)

        # Load Stage 1 checkpoint (accept multiple common key layouts)
        ckpt = torch.load(_stage1_ckpt, weights_only=False, map_location=self.device)
        _sp_key = next(
            (k for k in ("scene_prior", "scene_prior_state", "field_state",
                         "model_state") if k in ckpt),
            None,
        )
        if _sp_key is not None:
            self.scene_prior.load_state_dict(ckpt[_sp_key])
        else:
            # direct state dict
            self.scene_prior.load_state_dict(ckpt)
        if "renderer" in ckpt:
            self.renderer.load_state_dict(ckpt["renderer"])
        print(f"[stage2] Loaded scene prior from {_stage1_ckpt}")

        # Freeze scene prior + renderer
        for p in self.scene_prior.parameters():
            p.requires_grad_(False)
        for p in self.renderer.parameters():
            p.requires_grad_(False)
        self.scene_prior.eval()
        self.renderer.eval()

        # Decoder (the only trainable module in Stage 2)
        self.decoder = StructuredDecoder(
            in_channels=8,
            hidden=getattr(model_cfg, "hidden_dim", 64),
        ).to(device)

        train_cfg = config.training
        self.optimizer = optim.AdamW(
            self.decoder.parameters(),
            lr=getattr(train_cfg, "lr_decoder", 3e-4),
            weight_decay=1e-4,
        )

        self.evaluator = Phase64Evaluator()
        self.out_dir = Path(getattr(config, "out_dir", "outputs/phase64/stage2"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._best_psnr: float = 0.0
        self._step: int = 0
        self._train_log: list = []

    # ---------------------------------------------------------------------- #

    def _train_step(self, sample: Phase64Sample) -> Dict[str, float]:
        """
        1. Run frozen ScenePriorModule
        2. Get SceneOutputs → canonical tensor
        3. Run StructuredDecoder → pred_rgb
        4. Compute reconstruction + isolation losses
        """
        self.decoder.train()
        self.optimizer.zero_grad()

        frames = sample.frames
        routing_e0 = sample.routing_e0
        routing_e1 = sample.routing_e1
        meta = sample.meta
        scene_gt = sample.scene_gt

        frame_t = torch.from_numpy(frames.astype(np.float32) / 255.0).to(self.device)
        img_chw = frame_t.mean(dim=0).unsqueeze(0).permute(0, 3, 1, 2)  # (1, 3, H, W)

        r0 = torch.from_numpy(routing_e0.mean(axis=0)).float().to(self.device).unsqueeze(0).unsqueeze(0)
        r1 = torch.from_numpy(routing_e1.mean(axis=0)).float().to(self.device).unsqueeze(0).unsqueeze(0)
        routing_hints = torch.cat([r0, r1], dim=1)

        entity_names = [
            str(meta.get("keyword0", "entity0")),
            str(meta.get("keyword1", "entity1")),
        ]

        with torch.no_grad():
            density_fields = self.scene_prior(
                img=img_chw,
                entity_names=entity_names,
                routing_hints=routing_hints,
            )
            scene_out = self.renderer(density_fields)

        scene_tensor = scene_out.to_canonical_tensor()  # (1, 8, H, W)

        # Decode to RGB
        pred_rgb = self.decoder(scene_tensor)  # (1, 3, H, W)

        # GT RGB: resize mean frame to spatial resolution
        model_cfg = self.config.model
        H_sp, W_sp = model_cfg.spatial_h, model_cfg.spatial_w
        gt_rgb = F.interpolate(
            img_chw, size=(H_sp, W_sp), mode="bilinear", align_corners=False
        )

        rec_loss = reconstruction_loss(pred_rgb, gt_rgb)

        # Isolation: use predicted visible maps as entity region proxies
        iso_loss = isolation_loss(
            pred_rgb,
            scene_out.visible_e0,   # (1, H, W)
            scene_out.visible_e1,
        )

        total_loss = rec_loss + 0.5 * iso_loss

        total_loss.backward()

        grad_clip = getattr(self.config.training, "grad_clip", 1.0)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), grad_clip)
        self.optimizer.step()

        return {
            "total": float(total_loss.item()),
            "rec":   float(rec_loss.item()),
            "iso":   float(iso_loss.item()),
        }

    def _validate(self, epoch: int) -> dict:
        """Validation: compute PSNR on val split."""
        self.decoder.eval()

        split_info = (self._splits if self._splits is not None
                      else self.dataset.get_split_indices()
                      if hasattr(self.dataset, "get_split_indices") else {})
        val_indices = split_info.get("val", [])
        if not val_indices:
            return {}

        pred_rgb_list, gt_rgb_list = [], []

        with torch.no_grad():
            for idx in val_indices[:30]:
                sample = self.dataset[idx]
                frames = sample.frames
                routing_e0 = sample.routing_e0
                routing_e1 = sample.routing_e1
                meta = sample.meta

                frame_t = torch.from_numpy(frames.astype(np.float32) / 255.0).to(self.device)
                img_chw = frame_t.mean(dim=0).unsqueeze(0).permute(0, 3, 1, 2)

                r0 = torch.from_numpy(routing_e0.mean(axis=0)).float().to(self.device).unsqueeze(0).unsqueeze(0)
                r1 = torch.from_numpy(routing_e1.mean(axis=0)).float().to(self.device).unsqueeze(0).unsqueeze(0)
                routing_hints = torch.cat([r0, r1], dim=1)

                entity_names = [
                    str(meta.get("keyword0", "entity0")),
                    str(meta.get("keyword1", "entity1")),
                ]

                density_fields = self.scene_prior(
                    img=img_chw, entity_names=entity_names, routing_hints=routing_hints,
                )
                scene_out = self.renderer(density_fields)
                scene_tensor = scene_out.to_canonical_tensor()
                pred_rgb = self.decoder(scene_tensor)  # (1, 3, H, W)

                model_cfg = self.config.model
                H_sp, W_sp = model_cfg.spatial_h, model_cfg.spatial_w
                gt_rgb = F.interpolate(img_chw, size=(H_sp, W_sp), mode="bilinear", align_corners=False)

                pred_rgb_list.append(pred_rgb[0].permute(1, 2, 0).cpu().numpy())
                gt_rgb_list.append(gt_rgb[0].permute(1, 2, 0).cpu().numpy())

        dec_metrics = self.evaluator.eval_decoder(pred_rgb_list, gt_rgb_list)
        psnr = dec_metrics.get("composite_psnr", 0.0)
        l1   = dec_metrics.get("composite_l1", float("nan"))
        print(f"  [val epoch {epoch}] PSNR={psnr:.2f} dB  L1={l1:.4f}")

        if psnr > self._best_psnr:
            self._best_psnr = psnr
            ckpt_path = self.out_dir / "best_decoder.pt"
            torch.save({
                "epoch": epoch,
                "step": self._step,
                "decoder": self.decoder.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "val_metrics": dec_metrics,
            }, ckpt_path)
            print(f"  [val] New best PSNR={psnr:.2f} dB → saved to {ckpt_path}")

        return dec_metrics

    # ---------------------------------------------------------------------- #

    def load_checkpoint(self, path: str) -> None:
        """Load decoder + optimizer state (called by train_phase64_decoder.py)."""
        ckpt = torch.load(path, weights_only=False, map_location=self.device)
        key = "decoder_state" if "decoder_state" in ckpt else "decoder"
        self.decoder.load_state_dict(ckpt[key])
        opt_key = "optimizer_state" if "optimizer_state" in ckpt else "optimizer"
        if opt_key in ckpt:
            self.optimizer.load_state_dict(ckpt[opt_key])
        self._step = ckpt.get("step", 0)
        print(f"[stage2] Loaded checkpoint from {path}  (step={self._step})")

    # ---------------------------------------------------------------------- #

    def train(self, resume_from: Optional[str] = None) -> None:
        """Main training loop for Stage 2."""
        if resume_from is not None:
            ckpt = torch.load(resume_from, weights_only=False, map_location=self.device)
            self.decoder.load_state_dict(ckpt["decoder"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self._step = ckpt.get("step", 0)
            print(f"[stage2] Resumed from {resume_from}  (step={self._step})")

        train_cfg = self.config.training
        epochs          = train_cfg.epochs
        steps_per_epoch = train_cfg.steps_per_epoch
        eval_every      = train_cfg.eval_every

        split_info = (self._splits if self._splits is not None
                      else self.dataset.get_split_indices()
                      if hasattr(self.dataset, "get_split_indices") else {})
        train_indices = split_info.get("train", list(range(len(self.dataset))))

        print(f"[stage2] Training decoder  epochs={epochs}  "
              f"steps/epoch={steps_per_epoch}  n_train={len(train_indices)}")

        rng = np.random.default_rng(seed=43)

        for epoch in range(1, epochs + 1):
            epoch_metrics: list = []
            idxs = rng.choice(train_indices, size=min(steps_per_epoch, len(train_indices)),
                              replace=False).tolist()

            for local_i, idx in enumerate(idxs):
                sample = self.dataset[idx]
                step_m = self._train_step(sample)
                epoch_metrics.append(step_m)
                self._step += 1

            avg_total = float(np.mean([m["total"] for m in epoch_metrics]))
            avg_rec   = float(np.mean([m["rec"]   for m in epoch_metrics]))
            print(f"[epoch {epoch}] total={avg_total:.4f}  rec={avg_rec:.4f}")

            self._train_log.append({
                "epoch": epoch,
                "step": self._step,
                "loss": avg_total,
                "rec_loss": avg_rec,
            })

            if epoch % eval_every == 0 or epoch == epochs:
                val_m = self._validate(epoch)
                self._train_log[-1].update({f"val_{k}": v for k, v in val_m.items()})

            log_path = self.out_dir / "train_log.json"
            with open(log_path, "w") as f:
                json.dump(self._train_log, f, indent=2, default=str)

        print(f"[stage2] Complete.  Best PSNR={self._best_psnr:.2f} dB")
        print(f"[stage2] Logs → {self.out_dir}")

        # Gating check
        psnr_thresh = getattr(getattr(self.config, "eval", object()), "min_decoder_psnr", 18.0)
        if self._best_psnr < psnr_thresh:
            print(f"[stage2] WARNING: best PSNR {self._best_psnr:.2f} < {psnr_thresh:.1f} "
                  "— do NOT proceed to Stage 3 until decoder quality is sufficient.")
        else:
            print(f"[stage2] Decoder PSNR {self._best_psnr:.2f} ≥ {psnr_thresh:.1f} — OK to proceed to Stage 3.")
