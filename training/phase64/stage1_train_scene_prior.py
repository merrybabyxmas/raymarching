"""
training/phase64/stage1_train_scene_prior.py
=============================================
Stage 1: Train backbone-agnostic scene prior (EntityField).

NO backbone (AnimateDiff, SDXL) is loaded or used.

The scene prior learns purely from:
  - direct image RGB
  - entity colour routing maps
  - GT visible/amodal masks from SceneGT

Key difference from Phase 63: no UNet dependency.

Config expected attributes:
  model:
    depth_bins, hidden_dim, id_dim, pose_dim, spatial_h, spatial_w, slot_dim
  training:
    epochs, steps_per_epoch, eval_every, lr_field, lr_encoder, lr_memory,
    grad_clip
  training losses:
    lambda_vis, lambda_amo, lambda_occ, lambda_surv, lambda_color, lambda_sep
  eval:
    visible_survival_thresh, min_survival
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.phase64.phase64_dataset import Phase64Dataset, Phase64Sample
from training.phase64.evaluator_phase64 import Phase64Evaluator


# --------------------------------------------------------------------------- #
#  Loss helpers
# --------------------------------------------------------------------------- #

def _bce_balanced(pred: torch.Tensor, gt: torch.Tensor, pos_weight: float = 10.0) -> torch.Tensor:
    """Balanced BCE: equal weight on positive and negative pixels."""
    gt = gt.float()
    pos_mask = gt > 0.5
    neg_mask = ~pos_mask
    loss = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), gt, reduction="none")
    n_pos = pos_mask.float().sum().clamp(min=1.0)
    n_neg = neg_mask.float().sum().clamp(min=1.0)
    l_pos = (loss * pos_mask.float()).sum() / n_pos
    l_neg = (loss * neg_mask.float()).sum() / n_neg
    return pos_weight * l_pos + l_neg


def _iou_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft IoU loss."""
    pred = pred.clamp(0.0, 1.0)
    gt = gt.float().clamp(0.0, 1.0)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return 1.0 - (inter + eps) / (union + eps)


def total_scene_loss(
    scene_out,      # SceneOutputs
    scene_gt,       # SceneGT
    device: torch.device,
    lambda_vis:   float = 1.0,
    lambda_amo:   float = 1.0,
    lambda_occ:   float = 0.5,
    lambda_surv:  float = 2.0,
    lambda_color: float = 0.2,
    lambda_sep:   float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Compute all scene-prior training losses.

    Returns a dict of named loss tensors.  The caller sums them with their
    lambda weights to form the total scalar loss.
    """
    # GT masks as tensors on device — shape (H, W)
    # scene_out has shape (B=1, H, W) — squeeze batch dim
    def _gt(arr: np.ndarray) -> torch.Tensor:
        # arr: (T, H, W) or (H, W) — use mean over T as single-frame GT
        t = torch.from_numpy(arr.mean(axis=0) if arr.ndim == 3 else arr).float().to(device)
        return t

    gt_vis_e0 = _gt(scene_gt.vis_e0)
    gt_vis_e1 = _gt(scene_gt.vis_e1)
    gt_amo_e0 = _gt(scene_gt.amo_e0)
    gt_amo_e1 = _gt(scene_gt.amo_e1)

    # Predicted fields — squeeze batch dim if B=1
    pred_vis_e0 = scene_out.visible_e0.squeeze(0)
    pred_vis_e1 = scene_out.visible_e1.squeeze(0)
    pred_amo_e0 = scene_out.amodal_e0.squeeze(0)
    pred_amo_e1 = scene_out.amodal_e1.squeeze(0)
    pred_sep    = scene_out.sep_map.squeeze(0)

    losses: Dict[str, torch.Tensor] = {}

    # Visible IoU
    losses["vis_e0"] = _iou_loss(pred_vis_e0, gt_vis_e0)
    losses["vis_e1"] = _iou_loss(pred_vis_e1, gt_vis_e1)

    # Amodal BCE
    losses["amo_e0"] = _bce_balanced(pred_amo_e0, gt_amo_e0)
    losses["amo_e1"] = _bce_balanced(pred_amo_e1, gt_amo_e1)

    # Occlusion consistency: amodal >= visible everywhere
    occ_e0 = F.relu(pred_vis_e0 - pred_amo_e0).mean()
    occ_e1 = F.relu(pred_vis_e1 - pred_amo_e1).mean()
    losses["occ"] = occ_e0 + occ_e1

    # Survival loss: both entities must have non-trivial visible coverage
    surv_thresh = 0.01
    survival_e0 = pred_vis_e0.mean()
    survival_e1 = pred_vis_e1.mean()
    losses["surv"] = (F.relu(surv_thresh - survival_e0) + F.relu(surv_thresh - survival_e1))

    # Separation consistency: sep_map should match (visible_e0 - visible_e1)
    gt_sep = gt_vis_e0 - gt_vis_e1
    losses["sep"] = F.mse_loss(pred_sep, gt_sep)

    # Total (weighted)
    total = (
        lambda_vis  * (losses["vis_e0"] + losses["vis_e1"])
        + lambda_amo  * (losses["amo_e0"] + losses["amo_e1"])
        + lambda_occ  * losses["occ"]
        + lambda_surv * losses["surv"]
        + lambda_sep  * losses["sep"]
    )
    losses["total"] = total
    return losses


# --------------------------------------------------------------------------- #
#  Stage1Trainer
# --------------------------------------------------------------------------- #

class Stage1Trainer:
    """
    Trains ScenePriorModule (backbone-agnostic entity field) on Phase64Dataset.

    NO backbone (AnimateDiff, SDXL) is loaded or used.

    Parameters
    ----------
    config  : object with attributes described in module docstring
    dataset : Phase64Dataset
    splits  : dict returned by make_splits (train/val/split_O/C/R/X)
    device  : torch.device
    """

    def __init__(
        self,
        config,
        dataset: Phase64Dataset,
        splits: Optional[Dict[str, list]] = None,
        device: str = "cuda",
    ) -> None:
        self.config = config
        self.dataset = dataset
        # Store splits; fall back to dataset.get_split_indices() if not provided
        self._splits = splits

        if isinstance(device, str):
            device = torch.device(device) if torch.cuda.is_available() or device == "cpu" else torch.device("cpu")
        self.device = device

        # Lazy import to avoid hard dependency at module level
        from scene_prior import ScenePriorModule

        model_cfg = config.model
        # NOTE: ScenePriorModule uses `hidden` (not `hidden_dim`)
        self.scene_prior = ScenePriorModule(
            depth_bins=int(model_cfg.depth_bins),
            hidden=int(model_cfg.hidden_dim),
            id_dim=int(model_cfg.id_dim),
            pose_dim=int(model_cfg.pose_dim),
            ctx_dim=int(getattr(model_cfg, "ctx_dim", model_cfg.hidden_dim)),
            spatial_h=int(model_cfg.spatial_h),
            spatial_w=int(model_cfg.spatial_w),
            slot_dim=int(getattr(model_cfg, "slot_dim", 64)),
        ).to(device)
        # ScenePriorModule already contains an EntityRenderer — no separate renderer needed.

        train_cfg = config.training
        param_groups = [
            {"params": self.scene_prior.field_params(),   "lr": float(train_cfg.lr_field)},
            {"params": self.scene_prior.encoder_params(), "lr": float(train_cfg.lr_encoder)},
            {"params": self.scene_prior.memory_params(),  "lr": float(train_cfg.lr_memory)},
        ]
        self.optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

        loss_cfg = config.training
        self.lambda_vis   = getattr(loss_cfg, "lambda_vis",   1.0)
        self.lambda_amo   = getattr(loss_cfg, "lambda_amo",   1.0)
        self.lambda_occ   = getattr(loss_cfg, "lambda_occ",   0.5)
        self.lambda_surv  = getattr(loss_cfg, "lambda_surv",  2.0)
        self.lambda_color = getattr(loss_cfg, "lambda_color", 0.2)
        self.lambda_sep   = getattr(loss_cfg, "lambda_sep",   0.5)

        self.evaluator = Phase64Evaluator(
            visible_thresh=getattr(config.eval, "visible_survival_thresh", 0.02),
            amodal_thresh=0.02,
        )

        run_name = getattr(config, "run_name", "p64_stage1")
        self.out_dir = Path(f"checkpoints/phase64/{run_name}")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._best_vis_iou_min: float = 0.0
        self._step: int = 0
        self._train_log: list = []

    # ---------------------------------------------------------------------- #

    def _train_step(self, sample: Phase64Sample) -> Dict[str, float]:
        """
        Single training step.

        1. Get frame + routing hints from sample
        2. Run ScenePriorModule(img, entity_names, routing_hints)
        3. EntityRenderer → SceneOutputs
        4. Compute losses via total_scene_loss
        5. Backward + optimizer step

        Returns metrics dict.
        """
        self.scene_prior.train()
        self.optimizer.zero_grad()

        frames = sample.frames          # (T, H, W, 3) uint8
        routing_e0 = sample.routing_e0  # (T, H, W) float32
        routing_e1 = sample.routing_e1  # (T, H, W) float32
        meta = sample.meta
        scene_gt = sample.scene_gt

        # Use mean frame as representative image for this clip
        T = frames.shape[0]
        frame_t = torch.from_numpy(frames.astype(np.float32) / 255.0).to(self.device)  # (T, H, W, 3)
        img = frame_t.mean(dim=0).unsqueeze(0)  # (1, H, W, 3) → rearrange below

        # (1, 3, H, W)
        img_chw = img.permute(0, 3, 1, 2)

        # Routing hints: mean over T, then (1, 2, H, W)
        r0 = torch.from_numpy(routing_e0.mean(axis=0)).float().to(self.device).unsqueeze(0).unsqueeze(0)
        r1 = torch.from_numpy(routing_e1.mean(axis=0)).float().to(self.device).unsqueeze(0).unsqueeze(0)
        routing_hints = torch.cat([r0, r1], dim=1)  # (1, 2, H, W)

        # Entity names
        entity_name_e0 = str(meta.get("keyword0", "unknown"))
        entity_name_e1 = str(meta.get("keyword1", "unknown"))

        # Forward pass: ScenePriorModule renders internally → returns (SceneOutputs, mem_e0, mem_e1)
        scene_out, _, _ = self.scene_prior(
            img=img_chw,
            entity_name_e0=entity_name_e0,
            entity_name_e1=entity_name_e1,
            routing_hint_e0=r0,
            routing_hint_e1=r1,
        )

        # Compute losses
        loss_dict = total_scene_loss(
            scene_out,
            scene_gt,
            self.device,
            lambda_vis=self.lambda_vis,
            lambda_amo=self.lambda_amo,
            lambda_occ=self.lambda_occ,
            lambda_surv=self.lambda_surv,
            lambda_color=self.lambda_color,
            lambda_sep=self.lambda_sep,
        )

        total_loss = loss_dict["total"]
        total_loss.backward()

        grad_clip = getattr(self.config.training, "grad_clip", 1.0)
        nn.utils.clip_grad_norm_(self.scene_prior.parameters(), grad_clip)
        self.optimizer.step()

        metrics = {k: float(v.item()) for k, v in loss_dict.items()}
        metrics["vis_iou_e0"] = float(
            1.0 - loss_dict["vis_e0"].item()
        )
        metrics["vis_iou_e1"] = float(
            1.0 - loss_dict["vis_e1"].item()
        )
        return metrics

    def _validate(self, epoch: int) -> dict:
        """Run evaluation on the validation split."""
        self.scene_prior.eval()

        splits = self._splits or self.dataset.get_split_indices()
        val_indices = splits.get("val", [])
        if not val_indices:
            return {}

        preds, gts = [], []
        with torch.no_grad():
            for idx in val_indices[:50]:  # cap at 50 for speed
                sample = self.dataset[idx]
                frames = sample.frames
                routing_e0 = sample.routing_e0
                routing_e1 = sample.routing_e1
                meta = sample.meta
                scene_gt = sample.scene_gt

                frame_t = torch.from_numpy(frames.astype(np.float32) / 255.0).to(self.device)
                img_chw = frame_t.mean(dim=0).unsqueeze(0).permute(0, 3, 1, 2)

                r0 = torch.from_numpy(routing_e0.mean(axis=0)).float().to(self.device).unsqueeze(0).unsqueeze(0)
                r1 = torch.from_numpy(routing_e1.mean(axis=0)).float().to(self.device).unsqueeze(0).unsqueeze(0)

                entity_name_e0 = str(meta.get("keyword0", "unknown"))
                entity_name_e1 = str(meta.get("keyword1", "unknown"))

                scene_out, _, _ = self.scene_prior(
                    img=img_chw,
                    entity_name_e0=entity_name_e0,
                    entity_name_e1=entity_name_e1,
                    routing_hint_e0=r0,
                    routing_hint_e1=r1,
                )

                preds.append({
                    "visible_e0": scene_out.visible_e0.squeeze(0).cpu(),
                    "visible_e1": scene_out.visible_e1.squeeze(0).cpu(),
                    "amodal_e0":  scene_out.amodal_e0.squeeze(0).cpu(),
                    "amodal_e1":  scene_out.amodal_e1.squeeze(0).cpu(),
                    "sep_map":    scene_out.sep_map.squeeze(0).cpu(),
                    "hidden_fraction_e0": float(scene_out.hidden_e0.squeeze(0).mean()),
                    "hidden_fraction_e1": float(scene_out.hidden_e1.squeeze(0).mean()),
                })
                gts.append({
                    "visible_e0": torch.from_numpy(scene_gt.vis_e0.mean(axis=0)),
                    "visible_e1": torch.from_numpy(scene_gt.vis_e1.mean(axis=0)),
                    "amodal_e0":  torch.from_numpy(scene_gt.amo_e0.mean(axis=0)),
                    "amodal_e1":  torch.from_numpy(scene_gt.amo_e1.mean(axis=0)),
                    "overlap_ratio": scene_gt.overlap_ratio,
                    "hidden_fraction_e0": scene_gt.hidden_fraction_e0,
                    "hidden_fraction_e1": scene_gt.hidden_fraction_e1,
                    "is_reappearance": (scene_gt.split_type.name == "R"),
                })

        val_metrics = self.evaluator.eval_scene_prior(preds, gts)

        # Print compact summary
        vis_min = val_metrics.get("visible_iou_min", float("nan"))
        amo_min = val_metrics.get("amodal_iou_min", float("nan"))
        surv_min = val_metrics.get("visible_survival_min", float("nan"))
        print(f"  [val epoch {epoch}] vis_iou_min={vis_min:.4f}  "
              f"amo_iou_min={amo_min:.4f}  surv_min={surv_min:.4f}")

        # Save checkpoint if best
        if not math.isnan(vis_min) and vis_min > self._best_vis_iou_min:
            self._best_vis_iou_min = vis_min
            ckpt_path = self.out_dir / "best_scene_prior.pt"
            torch.save({
                "epoch": epoch,
                "step": self._step,
                "scene_prior": self.scene_prior.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  [val] New best vis_iou_min={vis_min:.4f} → saved to {ckpt_path}")

        return val_metrics

    # ---------------------------------------------------------------------- #

    def train(self, resume_from: Optional[str] = None) -> None:
        """
        Main training loop.

        Parameters
        ----------
        resume_from : optional path to a checkpoint dict to resume from
        """
        if resume_from is not None:
            ckpt = torch.load(resume_from, weights_only=False, map_location=self.device)
            self.scene_prior.load_state_dict(ckpt["scene_prior"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self._step = ckpt.get("step", 0)
            print(f"[stage1] Resumed from {resume_from}  (step={self._step})")

        train_cfg = self.config.training
        epochs          = train_cfg.epochs
        steps_per_epoch = train_cfg.steps_per_epoch
        eval_every      = train_cfg.eval_every

        split_info = self._splits or self.dataset.get_split_indices()
        train_indices = split_info.get("train", list(range(len(self.dataset))))

        print(f"[stage1] Training  epochs={epochs}  "
              f"steps/epoch={steps_per_epoch}  "
              f"n_train={len(train_indices)}")

        rng = np.random.default_rng(seed=42)

        for epoch in range(1, epochs + 1):
            epoch_metrics: list = []
            idxs = rng.choice(train_indices, size=min(steps_per_epoch, len(train_indices)),
                              replace=False).tolist()

            for local_i, idx in enumerate(idxs):
                sample = self.dataset[idx]
                step_metrics = self._train_step(sample)
                epoch_metrics.append(step_metrics)
                self._step += 1

                if (local_i + 1) % max(1, steps_per_epoch // 5) == 0:
                    loss_mean = np.mean([m["total"] for m in epoch_metrics[-10:]])
                    print(f"  epoch {epoch}/{epochs}  "
                          f"step {local_i + 1}/{len(idxs)}  "
                          f"loss={loss_mean:.4f}")

            # Epoch summary
            avg_total = float(np.mean([m["total"] for m in epoch_metrics]))
            avg_vis_e0 = float(np.mean([m.get("vis_iou_e0", 0) for m in epoch_metrics]))
            avg_vis_e1 = float(np.mean([m.get("vis_iou_e1", 0) for m in epoch_metrics]))
            print(f"[epoch {epoch}] loss={avg_total:.4f}  "
                  f"vis_iou_e0={avg_vis_e0:.4f}  vis_iou_e1={avg_vis_e1:.4f}")

            self._train_log.append({
                "epoch": epoch,
                "step": self._step,
                "loss": avg_total,
                "vis_iou_e0": avg_vis_e0,
                "vis_iou_e1": avg_vis_e1,
            })

            # Validation
            if epoch % eval_every == 0 or epoch == epochs:
                val_m = self._validate(epoch)
                self._train_log[-1].update(
                    {f"val_{k}": v for k, v in val_m.items()}
                )

            # Save training log
            log_path = self.out_dir / "train_log.json"
            with open(log_path, "w") as f:
                json.dump(self._train_log, f, indent=2, default=str)

        print(f"[stage1] Training complete.  Best vis_iou_min={self._best_vis_iou_min:.4f}")
        print(f"[stage1] Logs saved to {self.out_dir}")
