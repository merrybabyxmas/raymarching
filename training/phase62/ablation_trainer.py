"""
Phase 62 — Ablation Trainer
==============================

Config-driven trainer supporting all objective families, guide families,
and training schedules.

Objective families: independent_bce, factorized_fg_id, projected_visible_only,
                    projected_amodal_only, center_offset
Guide families:     none, front_only, dual, four_stream
Schedules:          S0 (volume_only), S1 (freeze_bind), S2 (low_lr_bind), S3 (short_joint)
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

from training.phase62.objectives import build_objective
from training.phase62.objectives.base import VolumeOutputs
from training.phase62.losses import loss_diffusion, loss_feature_separation, compute_volume_accuracy
from training.phase62.evaluator import Phase62Evaluator, _encode_text, _get_entity_tokens_with_fallback
from training.phase62.rollout import Phase62RolloutRunner
from training.phase62.metrics import compute_projected_class_iou
from data.phase62.volume_gt_builder import VolumeGTBuilder
from scripts.train_animatediff_vca import encode_frames_to_latents
from scripts.train_phase39 import (
    compute_dataset_overlap_scores,
    split_train_val,
    make_sampling_weights,
)

MIN_VAL_SAMPLES = 3


class AblationTrainer:
    """
    Config-driven ablation trainer for Phase 62.

    Config keys:
        objective:     str  — which objective family
        guide_family:  str  — which guide family
        schedule:      str  — S0 | S1 | S2 | S3
        representation: str — independent | factorized_fg_id | center_offset
    """

    def __init__(self, config, pipe, system, backbone_mgr, dataset, device: str):
        self.config = config
        self.pipe = pipe
        self.system = system
        self.backbone_mgr = backbone_mgr
        self.dataset = dataset
        self.device = device

        self.train_cfg = config.training
        self.eval_cfg = config.eval
        self.data_cfg = config.data

        self.schedule = getattr(config, "schedule", "S1")
        self.objective_name = getattr(config, "objective", "independent_bce")

        # Build objective
        obj_kwargs = {}
        if self.objective_name == "factorized_fg_id":
            obj_kwargs["lambda_id"] = float(getattr(self.train_cfg, "lambda_id", 1.0))
            obj_kwargs["fg_pos_weight"] = float(getattr(self.train_cfg, "fg_pos_weight", 20.0))
            obj_kwargs["lambda_vis"] = float(getattr(self.train_cfg, "lambda_vis", 0.5))
        elif self.objective_name == "independent_bce":
            obj_kwargs["entity_pos_weight"] = float(getattr(self.train_cfg, "entity_pos_weight", 50.0))
        elif self.objective_name == "center_offset":
            obj_kwargs["fg_pos_weight"] = float(getattr(self.train_cfg, "fg_pos_weight", 20.0))
        self.objective = build_objective(self.objective_name, **obj_kwargs)

        self.gt_builder = VolumeGTBuilder(
            depth_bins=config.depth_bins,
            spatial_h=config.spatial_h,
            spatial_w=config.spatial_w,
            render_resolution=getattr(self.data_cfg, "volume_gt_render_resolution", 128),
        )

        self.evaluator = Phase62Evaluator()
        self.rollout_runner = Phase62RolloutRunner()

        self.debug_dir = Path(getattr(config, 'debug_dir', 'outputs/phase62_debug'))
        self.save_dir = Path(getattr(config, 'save_dir', 'checkpoints/phase62'))
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        raw_ds = dataset.raw_dataset() if hasattr(dataset, 'raw_dataset') else dataset
        overlap_scores = compute_dataset_overlap_scores(raw_ds)
        val_frac = getattr(self.data_cfg, 'val_frac', 0.2)
        self.train_idx, self.val_idx = split_train_val(
            overlap_scores, val_frac=val_frac, min_val=MIN_VAL_SAMPLES)
        self.sample_weights = make_sampling_weights(self.train_idx, overlap_scores)

        self._parse_schedule()
        self._setup_optimizer()

        self.history: List[Dict] = []
        self.best_val_score = -1.0
        self.best_epoch = -1

    def _parse_schedule(self):
        epochs = self.train_cfg.epochs
        if self.schedule == "S0":
            self.stage1_end = epochs
            self.stage2_end = epochs
        elif self.schedule == "S1":
            self.stage1_end = max(1, epochs // 3)
            self.stage2_end = epochs
        elif self.schedule == "S2":
            self.stage1_end = max(1, epochs // 3)
            self.stage2_end = epochs
        elif self.schedule == "S3":
            self.stage1_end = max(1, epochs // 4)
            self.stage2_end = max(self.stage1_end + 1, 3 * epochs // 4)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def _get_stage(self, epoch: int) -> str:
        if epoch < self.stage1_end:
            return "stage1"
        if epoch < self.stage2_end:
            return "stage2"
        return "stage3"

    def _setup_optimizer(self):
        volume_params = self.system.volume_params()
        assembler_params = self.system.assembler_params()
        adapter_params = self.backbone_mgr.adapter_params()
        lora_params = self.backbone_mgr.lora_params()
        for p in volume_params + assembler_params + adapter_params + lora_params:
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
            self.optimizer, T_max=max(1, self.train_cfg.epochs), eta_min=1e-6)

    def _set_epoch_trainability(self, epoch: int):
        stage = self._get_stage(epoch)

        if self.schedule == "S0":
            for p in self.param_groups["volume"]:
                p.requires_grad_(True)
            for p in self.param_groups["assembler"] + self.param_groups["adapter"] + self.param_groups["lora"]:
                p.requires_grad_(False)
            for group in self.optimizer.param_groups:
                name = group.get("name", "")
                if name == "volume_pred":
                    group["lr"] = group["initial_lr"]
                else:
                    group["lr"] = 0.0
            return

        if stage == "stage1":
            for p in self.param_groups["volume"]:
                p.requires_grad_(True)
            for p in self.param_groups["assembler"] + self.param_groups["adapter"] + self.param_groups["lora"]:
                p.requires_grad_(False)
            for group in self.optimizer.param_groups:
                name = group.get("name", "")
                group["lr"] = group["initial_lr"] if name == "volume_pred" else 0.0

        elif stage == "stage2":
            freeze_vol = (self.schedule == "S1")
            low_lr_vol = (self.schedule == "S2")
            for p in self.param_groups["volume"]:
                p.requires_grad_(not freeze_vol)
            for p in self.param_groups["assembler"] + self.param_groups["adapter"] + self.param_groups["lora"]:
                p.requires_grad_(True)
            for group in self.optimizer.param_groups:
                name = group.get("name", "")
                if name == "volume_pred":
                    if freeze_vol:
                        group["lr"] = 0.0
                    elif low_lr_vol:
                        group["lr"] = group["initial_lr"] * 0.1
                    else:
                        group["lr"] = group["initial_lr"]
                else:
                    group["lr"] = group["initial_lr"]

        elif stage == "stage3":
            for p in self.param_groups["volume"]:
                p.requires_grad_(True)
            for p in self.param_groups["assembler"] + self.param_groups["adapter"] + self.param_groups["lora"]:
                p.requires_grad_(True)
            for group in self.optimizer.param_groups:
                name = group.get("name", "")
                if name == "volume_pred":
                    group["lr"] = group["initial_lr"] * 0.25
                else:
                    group["lr"] = group["initial_lr"]

    def _build_gt_tensor(self, src_masks, B_feat: int) -> torch.Tensor:
        T = min(int(src_masks.shape[0]), B_feat)
        masks = src_masks[:T]
        S = masks.shape[-1]
        hw = int(round(S ** 0.5))
        gt = torch.from_numpy(masks.astype(np.float32)).to(self.device)
        gt = gt.reshape(T, 2, hw, hw)
        if hw != self.config.spatial_h or hw != self.config.spatial_w:
            gt = F.interpolate(gt, size=(self.config.spatial_h, self.config.spatial_w),
                               mode="bilinear", align_corners=False)
        return gt.clamp(0.0, 1.0)

    def _unpack_sample(self, sample):
        if isinstance(sample, dict):
            return (
                sample["frames"], sample["depth"], sample["depth_orders"],
                sample["meta"], sample.get("sample_dir"),
                sample["entity_masks"], sample.get("visible_masks"),
            )
        elif len(sample) >= 8:
            return sample[0], sample[1], sample[2], sample[3], None, sample[4], sample[5]
        else:
            return sample[0], sample[1], sample[2], sample[3], None, sample[4], None

    def _train_step(self, sample, data_idx: int, epoch: int) -> Optional[Dict]:
        frames_np, depth_np, depth_orders, meta, sample_dir, entity_masks, visible_masks = \
            self._unpack_sample(sample)

        toks_e0_t, toks_e1_t, full_prompt = _get_entity_tokens_with_fallback(
            self.pipe, meta, self.device)
        self.backbone_mgr.set_entity_tokens(toks_e0_t, toks_e1_t)
        self.backbone_mgr.reset_slot_store()

        with torch.no_grad():
            latents = encode_frames_to_latents(self.pipe, frames_np, self.device)
        noise = torch.randn_like(latents)
        t = torch.randint(0, self.train_cfg.t_max, (1,), device=self.device).long()
        noisy = self.pipe.scheduler.add_noise(latents, noise, t)

        T_frames = min(frames_np.shape[0], entity_masks.shape[0], len(depth_orders))
        enc_full = _encode_text(self.pipe, full_prompt, self.device)
        if enc_full.shape[0] == 1 and noisy.shape[0] > 1:
            enc_full = enc_full.expand(noisy.shape[0], -1, -1)

        self.optimizer.zero_grad()

        # Pass 1: extract features
        self.system.clear_guides()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = self.pipe.unet(noisy, t, encoder_hidden_states=enc_full).sample

        F_g = self.backbone_mgr.primary.last_Fg
        F_0 = self.backbone_mgr.primary.last_F0
        F_1 = self.backbone_mgr.primary.last_F1
        if F_g is None or F_0 is None or F_1 is None:
            return None

        # Volume prediction
        vol_outputs = self.system.predict_volume(F_g, F_0, F_1)

        # Build V_gt
        depth_maps = depth_np[:T_frames] if depth_np is not None else None
        if depth_maps is not None:
            V_gt_np = self.gt_builder.build_batch(
                depth_maps, entity_masks[:T_frames], depth_orders[:T_frames],
                visible_masks=(visible_masks[:T_frames] if visible_masks is not None else None),
                meta=meta, sample_dir=sample_dir)
            V_gt = torch.from_numpy(V_gt_np).to(self.device).long()
        else:
            V_gt = torch.zeros(T_frames, self.config.depth_bins,
                               self.config.spatial_h, self.config.spatial_w,
                               dtype=torch.long, device=self.device)

        B_feat = vol_outputs.entity_probs.shape[0]
        if V_gt.shape[0] < B_feat:
            n_rep = max(1, B_feat // V_gt.shape[0])
            V_gt = V_gt.repeat(n_rep, 1, 1, 1)[:B_feat]

        # Projection: only assemble guides when they'll actually be used
        stage = self._get_stage(epoch)
        use_diffusion = (stage != "stage1") and (self.schedule != "S0")

        if use_diffusion:
            vol_outputs, guides = self.system.project_and_assemble(vol_outputs, F_g, F_0, F_1)
        else:
            vol_outputs = self.system.projector(vol_outputs)
            guides = {}

        # Build GT tensors
        gt_amodal = self._build_gt_tensor(entity_masks, B_feat)
        src_vis = visible_masks if visible_masks is not None else entity_masks
        gt_visible = self._build_gt_tensor(src_vis, B_feat)

        obj_result = self.objective(vol_outputs, V_gt, gt_visible=gt_visible, gt_amodal=gt_amodal)
        l_struct = obj_result["total"].clamp(max=50.0)

        # Feature separation loss (Issue 2 fix): push F_0 and F_1 apart
        la_sep = float(getattr(self.train_cfg, "la_feature_sep", 0.1))
        if la_sep > 0:
            l_sep = loss_feature_separation(F_0, F_1)
            l_struct = l_struct + la_sep * l_sep

        if use_diffusion:
            self.system.set_guides(guides)
            self.backbone_mgr.reset_slot_store()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = self.pipe.unet(noisy, t, encoder_hidden_states=enc_full).sample

            noise_t = noise[:, :, :T_frames].float()
            noise_pred_t = noise_pred[:, :, :T_frames].float()
            l_diff = loss_diffusion(noise_pred_t, noise_t)
        else:
            self.system.clear_guides()
            l_diff = torch.tensor(0.0, device=self.device)

        # Total loss
        la_vol = float(getattr(self.train_cfg, "la_vol", 2.0))
        la_diff = float(getattr(self.train_cfg, "la_diff", 1.0))

        if stage == "stage1" or self.schedule == "S0":
            loss = la_vol * l_struct
        elif stage == "stage2":
            vol_scale = 0.0 if self.schedule == "S1" else 0.25
            loss = la_diff * l_diff + la_vol * vol_scale * l_struct
        else:
            loss = la_diff * l_diff + la_vol * 0.25 * l_struct

        if not torch.isfinite(loss) or loss.item() > 100.0:
            self.system.clear_guides()
            return None

        loss.backward()

        has_nan = False
        for pg in self.param_groups.values():
            for p in pg:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    has_nan = True
                    p.grad.zero_()
        if has_nan:
            self.optimizer.zero_grad()
            self.system.clear_guides()
            return None

        torch.nn.utils.clip_grad_norm_(self.param_groups["volume"], max_norm=0.3)
        torch.nn.utils.clip_grad_norm_(self.param_groups["assembler"], max_norm=0.3)
        torch.nn.utils.clip_grad_norm_(self.param_groups["adapter"], max_norm=0.3)
        torch.nn.utils.clip_grad_norm_(self.param_groups["lora"], max_norm=0.2)

        self.optimizer.step()
        self.system.clear_guides()

        # Metrics
        with torch.no_grad():
            # Compute accuracy from entity_probs
            ep = vol_outputs.entity_probs[:B_feat]
            p0 = ep[:, 0]
            p1 = ep[:, 1]
            pred_class = torch.zeros_like(V_gt.long())
            has_ent = (p0 > 0.5) | (p1 > 0.5)
            pred_class = torch.where(has_ent & (p0 >= p1), torch.ones_like(pred_class), pred_class)
            pred_class = torch.where(has_ent & (p1 > p0), torch.full_like(pred_class, 2), pred_class)
            correct = (pred_class == V_gt.long())
            acc_overall = correct.float().mean().item()
            ent_mask = (V_gt > 0)
            acc_entity = correct[ent_mask].float().mean().item() if ent_mask.any() else 0.0

        result = {
            "loss": float(loss.item()),
            "l_diff": float(l_diff.item()),
            "l_struct": float(l_struct.item()),
            "stage": stage,
            "acc_overall": acc_overall,
            "acc_entity": acc_entity,
        }
        for k, v in obj_result.items():
            if k != "total" and isinstance(v, torch.Tensor):
                result[k] = float(v.item())
        return result

    @torch.no_grad()
    def _eval_epoch(self, epoch: int) -> Dict:
        self.system.eval()
        self.backbone_mgr.eval()

        diff_losses, struct_losses = [], []
        accs_overall, accs_entity = [], []
        ious_e0, ious_e1 = [], []

        for vi in self.val_idx:
            try:
                sample = self.dataset[vi]
                frames_np, depth_np, depth_orders, meta, sample_dir, entity_masks, visible_masks = \
                    self._unpack_sample(sample)

                toks_e0_t, toks_e1_t, full_prompt = _get_entity_tokens_with_fallback(
                    self.pipe, meta, self.device)
                self.backbone_mgr.set_entity_tokens(toks_e0_t, toks_e1_t)
                self.backbone_mgr.reset_slot_store()

                latents = encode_frames_to_latents(self.pipe, frames_np, self.device)
                noise = torch.randn_like(latents)
                t_fixed = getattr(self.train_cfg, 't_max', 300) // 2
                t_tensor = torch.tensor([t_fixed], device=self.device).long()
                noisy = self.pipe.scheduler.add_noise(latents, noise, t_tensor)
                enc_full = _encode_text(self.pipe, full_prompt, self.device)

                T_frames = min(frames_np.shape[0], entity_masks.shape[0], len(depth_orders))

                self.system.clear_guides()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    noise_pred = self.pipe.unet(noisy, t_tensor, encoder_hidden_states=enc_full).sample

                l_diff = loss_diffusion(
                    noise_pred[:, :, :T_frames].float(),
                    noise[:, :, :T_frames].float())
                diff_losses.append(float(l_diff.item()))

                F_g = self.backbone_mgr.primary.last_Fg
                F_0 = self.backbone_mgr.primary.last_F0
                F_1 = self.backbone_mgr.primary.last_F1
                if F_g is None or F_0 is None or F_1 is None:
                    continue

                vol_outputs = self.system.predict_volume(F_g, F_0, F_1)

                depth_maps = depth_np[:T_frames]
                V_gt_np = self.gt_builder.build_batch(
                    depth_maps, entity_masks[:T_frames], depth_orders[:T_frames],
                    visible_masks=(visible_masks[:T_frames] if visible_masks is not None else None),
                    meta=meta, sample_dir=sample_dir)
                V_gt = torch.from_numpy(V_gt_np).to(self.device).long()

                B_feat = vol_outputs.entity_probs.shape[0]
                if V_gt.shape[0] < B_feat:
                    n_rep = max(1, B_feat // V_gt.shape[0])
                    V_gt = V_gt.repeat(n_rep, 1, 1, 1)[:B_feat]

                gt_amodal = self._build_gt_tensor(entity_masks, B_feat)
                src_vis = visible_masks if visible_masks is not None else entity_masks
                gt_visible = self._build_gt_tensor(src_vis, B_feat)

                # Project first (populates amodal/visible needed by some objectives)
                vol_outputs = self.system.projector(vol_outputs)

                obj_result = self.objective(vol_outputs, V_gt, gt_visible=gt_visible, gt_amodal=gt_amodal)
                struct_losses.append(float(obj_result["total"].item()))

                # Accuracy
                ep = vol_outputs.entity_probs[:B_feat]
                p0, p1 = ep[:, 0], ep[:, 1]
                pred_class = torch.zeros_like(V_gt.long())
                has_ent = (p0 > 0.5) | (p1 > 0.5)
                pred_class = torch.where(has_ent & (p0 >= p1), torch.ones_like(pred_class), pred_class)
                pred_class = torch.where(has_ent & (p1 > p0), torch.full_like(pred_class, 2), pred_class)
                correct = (pred_class == V_gt.long())
                accs_overall.append(correct.float().mean().item())
                ent_mask = (V_gt > 0)
                accs_entity.append(correct[ent_mask].float().mean().item() if ent_mask.any() else 0.0)

                # Projected IoU (already projected above)
                vc = vol_outputs.visible_class

                src_masks = visible_masks if visible_masks is not None else entity_masks
                n_eval = min(int(vc.shape[0]), int(src_masks.shape[0]))
                if n_eval > 0:
                    m0_gt = torch.from_numpy(src_masks[:n_eval, 0].astype(np.float32)).to(self.device)
                    m1_gt = torch.from_numpy(src_masks[:n_eval, 1].astype(np.float32)).to(self.device)
                    ious_e0.append(compute_projected_class_iou(
                        vc[:n_eval], m0_gt, entity_idx=1,
                        spatial_h=self.config.spatial_h, spatial_w=self.config.spatial_w))
                    ious_e1.append(compute_projected_class_iou(
                        vc[:n_eval], m1_gt, entity_idx=2,
                        spatial_h=self.config.spatial_h, spatial_w=self.config.spatial_w))

            except Exception as e:
                print(f"  [val warn] idx={vi}: {e}", flush=True)
                continue

        def _avg(lst):
            return sum(lst) / max(len(lst), 1) if lst else 0.0

        val_iou_e0 = _avg(ious_e0)
        val_iou_e1 = _avg(ious_e1)
        val_iou_min = min(val_iou_e0, val_iou_e1)
        val_acc_ent = _avg(accs_entity)
        val_struct = _avg(struct_losses)

        val_score = (
            0.10 * (1.0 / (1.0 + _avg(diff_losses)))
            + 0.10 * (1.0 / (1.0 + val_struct))
            + 0.20 * val_acc_ent
            + 0.15 * val_iou_e0
            + 0.15 * val_iou_e1
            + 0.30 * val_iou_min
        )

        result = {
            "val_score": val_score,
            "val_diff_mse": _avg(diff_losses),
            "val_struct": val_struct,
            "val_acc_overall": _avg(accs_overall),
            "val_acc_entity": val_acc_ent,
            "val_iou_e0": val_iou_e0,
            "val_iou_e1": val_iou_e1,
            "val_iou_min": val_iou_min,
            "n_samples": len(diff_losses),
        }

        print(
            f"  [val] score={val_score:.4f}  "
            f"struct={val_struct:.4f}  "
            f"acc_ent={val_acc_ent:.4f}  "
            f"iou_e0={val_iou_e0:.4f}  "
            f"iou_e1={val_iou_e1:.4f}  "
            f"iou_min={val_iou_min:.4f}  "
            f"n={len(diff_losses)}",
            flush=True)

        # GIF
        try:
            probe = self.dataset[self.val_idx[0]]
            if isinstance(probe, dict):
                probe_meta, probe_masks, probe_frames = probe["meta"], probe["entity_masks"], probe["frames"]
            else:
                probe_meta, probe_masks, probe_frames = probe[3], probe[4], probe[0]

            toks_e0_pt, toks_e1_pt, probe_prompt = \
                _get_entity_tokens_with_fallback(self.pipe, probe_meta, self.device)

            rollout_result = self.rollout_runner.generate_rollout(
                self.pipe, self.system, self.backbone_mgr,
                probe_prompt, self.config, self.device,
                toks_e0=toks_e0_pt, toks_e1=toks_e1_pt,
                entity_masks=probe_masks, gt_frames=probe_frames)

            paths = self.rollout_runner.save_rollout(
                rollout_result, self.debug_dir, prefix=f"eval_epoch{epoch:03d}")
            for name, path in paths.items():
                print(f"  [gif] {name}: {path}", flush=True)
        except Exception as e:
            print(f"  [warn] GIF failed: {e}", flush=True)

        return result

    def load_checkpoint(self, ckpt_path: str):
        """Load checkpoint and restore model states."""
        print(f"[Ablation] Loading checkpoint: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "system_state" in ckpt:
            if "volume_pred" in ckpt["system_state"]:
                self.system.volume_pred.load_state_dict(
                    ckpt["system_state"]["volume_pred"], strict=False)
            if "assembler" in ckpt["system_state"]:
                try:
                    self.system.assembler.load_state_dict(
                        ckpt["system_state"]["assembler"], strict=False)
                except Exception:
                    print("  [warn] assembler ckpt incompatible, skipping", flush=True)
        if "extractors_state" in ckpt:
            for i, es in enumerate(ckpt["extractors_state"]):
                if i < len(self.backbone_mgr.extractors):
                    ext = self.backbone_mgr.extractors[i]
                    for key in ("lora_k", "lora_v", "lora_out", "slot0_adapter", "slot1_adapter"):
                        if key in es:
                            getattr(ext, key).load_state_dict(es[key])
        print(f"[Ablation] Checkpoint loaded (epoch={ckpt.get('epoch', '?')})", flush=True)

    def _save_checkpoint(self, epoch: int, val_m: Dict):
        sel = val_m.get("val_score", 0.0)
        ckpt = {
            "epoch": epoch,
            "val_score": sel,
            "objective": self.objective_name,
            "schedule": self.schedule,
            "system_state": {
                "volume_pred": self.system.volume_pred.state_dict(),
                "assembler": self.system.assembler.state_dict(),
            },
            "extractors_state": [],
        }
        for ext in self.backbone_mgr.extractors:
            ckpt["extractors_state"].append({
                "lora_k": ext.lora_k.state_dict(),
                "lora_v": ext.lora_v.state_dict(),
                "lora_out": ext.lora_out.state_dict(),
                "slot0_adapter": ext.slot0_adapter.state_dict(),
                "slot1_adapter": ext.slot1_adapter.state_dict(),
            })
        torch.save(ckpt, str(self.save_dir / "latest.pt"))
        if sel > self.best_val_score:
            self.best_val_score = sel
            self.best_epoch = epoch
            torch.save(ckpt, str(self.save_dir / "best.pt"))
            print(f"  * best epoch={epoch} val_score={sel:.4f}", flush=True)

    def train(self):
        epochs = self.train_cfg.epochs
        steps_per_epoch = self.train_cfg.steps_per_epoch
        eval_every = self.train_cfg.eval_every
        seed = int(getattr(self.train_cfg, 'seed', 42))
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"[Ablation] {self.objective_name} / {self.schedule} / "
              f"guide={getattr(self.config, 'guide_family', 'dual')}", flush=True)
        print(f"  epochs={epochs} steps/ep={steps_per_epoch} "
              f"train={len(self.train_idx)} val={len(self.val_idx)}", flush=True)

        # Temperature annealing: start soft (1.0), anneal to hard (0.1)
        temp_start = float(getattr(self.train_cfg, "temp_start", 1.0))
        temp_end = float(getattr(self.train_cfg, "temp_end", 0.1))

        for epoch in range(epochs):
            # Anneal temperature: linear from temp_start to temp_end
            progress = epoch / max(1, epochs - 1)
            temp = temp_start + (temp_end - temp_start) * progress
            self.system.projector.set_temperature(temp)

            self.system.train()
            self.backbone_mgr.train()
            self._set_epoch_trainability(epoch)

            losses, accs_ent = [], []
            chosen = np.random.choice(len(self.train_idx), size=steps_per_epoch,
                                      replace=True, p=self.sample_weights)
            step_indices = [self.train_idx[ci] for ci in chosen]

            for batch_idx, data_idx in enumerate(step_indices):
                sample = self.dataset[data_idx]
                r = self._train_step(sample, data_idx, epoch)
                if r is None:
                    continue
                losses.append(r["loss"])
                accs_ent.append(r["acc_entity"])

            stage = self._get_stage(epoch)
            avg_loss = sum(losses) / max(len(losses), 1) if losses else float("nan")
            avg_acc = sum(accs_ent) / max(len(accs_ent), 1) if accs_ent else 0.0
            print(f"  [ep {epoch:3d}] {stage}  loss={avg_loss:.4f}  "
                  f"acc_ent={avg_acc:.4f}  steps={len(losses)}/{steps_per_epoch}",
                  flush=True)

            self.lr_scheduler.step()

            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                val_m = self._eval_epoch(epoch)
                self._save_checkpoint(epoch, val_m)
                self.history.append({"epoch": epoch, **val_m})

        # Save history
        hist_path = self.save_dir / "history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Ablation] Done. Best epoch={self.best_epoch} "
              f"val_score={self.best_val_score:.4f}", flush=True)
