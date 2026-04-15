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
from training.phase62.losses import (
    loss_diffusion, loss_feature_separation, compute_volume_accuracy,
    loss_spatial_coherence, loss_fg_coverage_prior, loss_permutation_consistency,
    loss_amodal_entity_coverage, loss_temporal_centroid_consistency,
)
from training.phase62.evaluator import Phase62Evaluator, _encode_text, _get_entity_tokens_with_fallback
from training.phase62.rollout import Phase62RolloutRunner
from training.phase62.metrics import compute_projected_class_iou
from training.phase62.debug_viz import TrainingDebugViz
from training.phase62.contract import DebugContract
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
            obj_kwargs["lambda_compact"] = float(getattr(self.train_cfg, "lambda_compact", 0.5))
            obj_kwargs["lambda_depth_ce"] = float(getattr(self.train_cfg, "lambda_depth_ce", 3.0))
            obj_kwargs["lambda_depth_vis"] = float(getattr(self.train_cfg, "lambda_depth_vis", 0.0))
            obj_kwargs["lambda_balance"] = float(getattr(self.train_cfg, "lambda_balance", 0.0))
            obj_kwargs["detach_fg_from_entity_losses"] = bool(getattr(self.train_cfg, "detach_fg_from_entity_losses", False))
            obj_kwargs["lambda_overlay_preserve"] = 0.0  # starts at 0; activated at stage3 entry via _set_epoch_trainability
            # Legacy params (kept for backwards compat if config includes them):
            obj_kwargs["lambda_dice"] = float(getattr(self.train_cfg, "lambda_dice", 0.0))
            obj_kwargs["lambda_hinge"] = float(getattr(self.train_cfg, "lambda_hinge", 0.0))
            obj_kwargs["hinge_margin"] = float(getattr(self.train_cfg, "hinge_margin", 1.0))
            obj_kwargs["hinge_density_thresh"] = float(getattr(self.train_cfg, "hinge_density_thresh", 0.20))
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

        # Training debug visualizer (writes to debug_dir/training/)
        self.train_viz = TrainingDebugViz(self.debug_dir, config)

        # Debug contract: formal pass/fail criteria per stage
        self.contract = DebugContract()
        self._last_contract_metrics = None  # updated on each eval

        # Eval outputs go to debug_dir/eval/
        (self.debug_dir / "eval").mkdir(parents=True, exist_ok=True)

        raw_ds = dataset.raw_dataset() if hasattr(dataset, 'raw_dataset') else dataset
        overlap_scores = compute_dataset_overlap_scores(raw_ds)
        val_frac = getattr(self.data_cfg, 'val_frac', 0.2)
        self.train_idx, self.val_idx = split_train_val(
            overlap_scores, val_frac=val_frac, min_val=MIN_VAL_SAMPLES)
        self.sample_weights = make_sampling_weights(self.train_idx, overlap_scores)

        self._parse_schedule()
        self._setup_optimizer()

        self.history: List[Dict] = []
        self.train_step_history: List[Dict] = []  # per-step metrics for loss curve
        self.val_history: List[Dict] = []
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
            # v24: stage1_epochs_override allows longer stage1 for compact growth.
            # Default: epochs//4 (30ep for 120ep). Override: e.g. 40 for better compact.
            s1_override = int(getattr(self.config, "stage1_epochs_override", 0))
            self.stage1_end = s1_override if s1_override > 0 else max(1, epochs // 4)
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

        # v15: Split fg_spatial_head into its own param group so it can be
        # selectively unfrozen in stage2 while the deep volume stays frozen.
        # fg_spatial_head is a tiny 2D head — safe to train incrementally.
        fg_spatial_param_ids = set()
        fg_spatial_params = []
        deep_volume_params = []
        if hasattr(self.system.volume_pred, 'fg_spatial_head'):
            for p in self.system.volume_pred.fg_spatial_head.parameters():
                fg_spatial_params.append(p)
                fg_spatial_param_ids.add(id(p))
        for p in volume_params:
            if id(p) not in fg_spatial_param_ids:
                deep_volume_params.append(p)

        for p in volume_params + assembler_params + adapter_params + lora_params:
            p.requires_grad_(True)

        self.param_groups = {
            "volume": deep_volume_params,
            "fg_spatial": fg_spatial_params,
            "assembler": assembler_params,
            "adapter": adapter_params,
            "lora": lora_params,
        }

        self.optimizer = optim.AdamW([
            {"params": deep_volume_params, "lr": self.train_cfg.lr_volume, "name": "volume_pred"},
            {"params": fg_spatial_params, "lr": self.train_cfg.lr_volume, "name": "fg_spatial"},
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
            for p in self.param_groups["volume"] + self.param_groups["fg_spatial"]:
                p.requires_grad_(True)
            for p in self.param_groups["assembler"] + self.param_groups["adapter"] + self.param_groups["lora"]:
                p.requires_grad_(False)
            for group in self.optimizer.param_groups:
                name = group.get("name", "")
                if name in ("volume_pred", "fg_spatial"):
                    group["lr"] = group["initial_lr"]
                else:
                    group["lr"] = 0.0
            return

        if stage == "stage1":
            for p in self.param_groups["volume"] + self.param_groups["fg_spatial"]:
                p.requires_grad_(True)
            for p in self.param_groups["assembler"] + self.param_groups["adapter"] + self.param_groups["lora"]:
                p.requires_grad_(False)
            for group in self.optimizer.param_groups:
                name = group.get("name", "")
                group["lr"] = group["initial_lr"] if name in ("volume_pred", "fg_spatial") else 0.0

        elif stage == "stage2":
            # v15: Freeze deep volume (fg_logit_vol / depth_attn) to protect compact.
            # ONLY unfreeze fg_spatial_head — lets fg_magnitude co-adapt as backbone
            # features improve through adapter training. Avoids v13 collapse where
            # full volume un-freeze destabilised depth_attn.
            # v25: vol_stage2_lr_factor > 0 allows vol to track backbone drift at ultra-low lr
            #      instead of being fully frozen (prevents compact decline from backbone drift).
            freeze_deep_vol = (self.schedule in ("S1", "S3"))
            low_lr_vol = (self.schedule == "S2")
            vol_s2_lr_factor = float(getattr(self.config, "vol_stage2_lr_factor", 0.0))
            # If vol_stage2_lr_factor > 0, vol trains at that fraction of lr_volume instead
            # of being fully frozen (only applies when freeze_deep_vol would be True)
            if freeze_deep_vol and vol_s2_lr_factor > 0:
                freeze_deep_vol = False   # override: allow vol to train at ultra-low lr
            for p in self.param_groups["volume"]:
                p.requires_grad_(not freeze_deep_vol)
            # fg_spatial always trains in stage2 (it's a tiny 2D head, safe to update)
            for p in self.param_groups["fg_spatial"]:
                p.requires_grad_(True)
            for p in self.param_groups["assembler"] + self.param_groups["adapter"] + self.param_groups["lora"]:
                p.requires_grad_(True)
            for group in self.optimizer.param_groups:
                name = group.get("name", "")
                if name == "volume_pred":
                    if vol_s2_lr_factor > 0:
                        # ultra-low lr: track backbone drift without overfitting
                        group["lr"] = group["initial_lr"] * vol_s2_lr_factor
                    elif freeze_deep_vol:
                        group["lr"] = 0.0
                    elif low_lr_vol:
                        group["lr"] = group["initial_lr"] * 0.1
                    else:
                        group["lr"] = group["initial_lr"]
                elif name == "fg_spatial":
                    # fg_spatial trains at reduced lr during stage2 (adapts with backbone)
                    group["lr"] = group["initial_lr"] * 0.3
                else:
                    group["lr"] = group["initial_lr"]

        elif stage == "stage3":
            # v18: freeze volume + fg_spatial entirely in stage3.
            # v17 showed iou_min oscillation (0.141→0.089→0.109) when volume trains
            # jointly with adapters in stage3. Freezing protects S1+S2 metrics while
            # guide (assembler + adapters + lora) continues to learn.
            # v24: separate freeze control for volume vs fg_spatial.
            # freeze_vol_stage3=false + freeze_fg_spatial_stage3=true: trains depth-related
            # vol params (for compact) while freezing spatial fg map (for overlay stability).
            # v23 showed that unfreezing fg_spatial in stage3 drops overlay 0.28→0.18.
            freeze_vol_s3 = bool(getattr(self.config, "freeze_vol_stage3", False))
            freeze_fg_s3 = bool(getattr(self.config, "freeze_fg_spatial_stage3", freeze_vol_s3))
            for p in self.param_groups["volume"]:
                p.requires_grad_(not freeze_vol_s3)
            for p in self.param_groups["fg_spatial"]:
                p.requires_grad_(not freeze_fg_s3)
            for p in self.param_groups["assembler"] + self.param_groups["adapter"] + self.param_groups["lora"]:
                p.requires_grad_(True)
            for group in self.optimizer.param_groups:
                name = group.get("name", "")
                if name == "volume_pred":
                    # v25: 0.10× lr (was 0.05×) for faster compact recovery in stage3
                    group["lr"] = 0.0 if freeze_vol_s3 else group["initial_lr"] * 0.10
                elif name == "fg_spatial":
                    group["lr"] = 0.0 if freeze_fg_s3 else group["initial_lr"] * 0.10
                else:
                    group["lr"] = group["initial_lr"]

            # v35: Activate overlay-preserving loss in stage3.
            # Rendered fg-coverage Dice(front_probs[:, 1:].sum, GT_fg_any) supervises
            # fg_spatial to maintain fg coverage as UNet features drift.
            # Only fires when fg_spatial is unfrozen (otherwise gradient can't reach fg_spatial_head).
            if hasattr(self, "objective") and hasattr(self.objective, "lambda_overlay_preserve"):
                lambda_op_s3 = float(getattr(self.config, "lambda_overlay_preserve_s3", 0.0))
                self.objective.lambda_overlay_preserve = lambda_op_s3

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

    def _unpack_solo_frames(self, sample):
        """Return (frames_solo0, frames_solo1) or (None, None) if unavailable."""
        if isinstance(sample, dict):
            return sample.get("frames_solo0"), sample.get("frames_solo1")
        # Tuple-style samples don't carry solo frames
        return None, None

    def _train_step(self, sample, data_idx: int, epoch: int, current_epoch: int = 0,
                    debug_step: bool = False, step_idx: int = 0) -> Optional[Dict]:
        frames_np, depth_np, depth_orders, meta, sample_dir, entity_masks, visible_masks = \
            self._unpack_sample(sample)
        frames_solo0, frames_solo1 = self._unpack_solo_frames(sample)

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

        # Build scene depth hint: (B_feat, H_vol, W_vol) ∈ [0,1]
        # Provides per-pixel depth cues so the 3D predictor can learn compact blobs.
        depth_hint_train = None
        if depth_np is not None:
            B_feat = F_g.shape[0]
            dmap = depth_np[:B_feat].astype(np.float32)        # (T', H_scene, W_scene)
            # MUST match VolumeGTBuilder's normalization: (depth - d_min) / (d_max - d_min)
            # Using just dmap/dmax misaligns depth_hint K-bins with GT V_gt K-bins → corrupt signal.
            dmin = float(dmap.min())
            dmax = float(dmap.max())
            drange = max(dmax - dmin, 1e-6)
            dmap_norm = torch.from_numpy((dmap - dmin) / drange).to(self.device)  # (T', H, W)
            H_vol, W_vol = self.config.spatial_h, self.config.spatial_w
            if dmap_norm.shape[-2:] != (H_vol, W_vol):
                dmap_norm = F.interpolate(
                    dmap_norm.unsqueeze(1).float(),
                    size=(H_vol, W_vol), mode="bilinear", align_corners=False,
                ).squeeze(1)
            if dmap_norm.shape[0] < B_feat:
                n_rep = max(1, B_feat // dmap_norm.shape[0])
                dmap_norm = dmap_norm.repeat(n_rep, 1, 1)[:B_feat]
            depth_hint_train = dmap_norm  # (B_feat, H_vol, W_vol)

        # Volume prediction
        vol_outputs = self.system.predict_volume(F_g, F_0, F_1,
                                                  depth_hint=depth_hint_train)

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
            # v39b: log guide feature norm and effective injected delta norm
            # guide_feature_norm = mean L2 norm of assembled guide tensors (before gate)
            # injected_delta_norm = mean norm after gate scaling (what actually reaches UNet)
            _guide_feature_norm = 0.0
            _injected_delta_norm = 0.0
            if guides:
                _norms = []
                _delta_norms = []
                for _bn, _gfeat in guides.items():
                    _gn = float(_gfeat.detach().norm(dim=1).mean().item())
                    _norms.append(_gn)
                    # Effective delta = guide_norm * tanh(gate)
                    if hasattr(self.system.assembler, "guide_gates") and _bn in self.system.assembler.guide_gates:
                        _gate_val = float(torch.tanh(self.system.assembler.guide_gates[_bn]).item())
                        _delta_norms.append(_gn * abs(_gate_val))
                    else:
                        _delta_norms.append(_gn)
                _guide_feature_norm = sum(_norms) / len(_norms) if _norms else 0.0
                _injected_delta_norm = sum(_delta_norms) / len(_delta_norms) if _delta_norms else 0.0
            # Accumulate per-step into epoch-level running average (stored on self)
            if not hasattr(self, "_step_guide_norm_acc"):
                self._step_guide_norm_acc = []
                self._step_delta_norm_acc = []
            self._step_guide_norm_acc.append(_guide_feature_norm)
            self._step_delta_norm_acc.append(_injected_delta_norm)
        else:
            vol_outputs = self.system.projector(vol_outputs)
            guides = {}

        # Build GT tensors
        gt_amodal = self._build_gt_tensor(entity_masks, B_feat)
        src_vis = visible_masks if visible_masks is not None else entity_masks
        gt_visible = self._build_gt_tensor(src_vis, B_feat)

        # Compact warmup: don't apply L_compact until fg_logit has had time to develop
        # spatial specificity. Firing L_compact from epoch 0 creates a destructive conflict:
        #   epoch 0: entity_probs = 0.5*softmax(init_id) ≈ uniform slab → L_compact fires
        #   L_compact concentrates slab before spatial fg locations are learned
        #   → collapse before spatial learning can succeed
        compact_warmup_epoch = int(getattr(self.train_cfg, "compact_warmup_epoch", 20))
        saved_lambda_compact = self.objective.lambda_compact
        if current_epoch < compact_warmup_epoch:
            self.objective.lambda_compact = 0.0

        # v27: balance_warmup_epoch — disable L_balance during stage1 so L_compact can
        # grow freely without opposing force. L_balance (vis_e0 ≈ vis_e1) acts on depth_attn
        # via vis_e = fg_magnitude × (depth_attn ⊙ q).sum(depth), pulling depth_attn to
        # spread across both entity bins — exactly opposing L_compact concentration.
        # With lambda_balance = lambda_compact = 3.0, equilibrium compact ≈ 0.40.
        # Setting lambda_balance=0 in stage1 removes this opposing force → compact grows to 0.65+.
        balance_warmup_epoch = int(getattr(self.train_cfg, "balance_warmup_epoch", 0))
        saved_lambda_balance = getattr(self.objective, "lambda_balance", 0.0)
        if balance_warmup_epoch > 0 and current_epoch < balance_warmup_epoch:
            if hasattr(self.objective, "lambda_balance"):
                self.objective.lambda_balance = 0.0

        obj_result = self.objective(vol_outputs, V_gt, gt_visible=gt_visible, gt_amodal=gt_amodal)
        l_struct = obj_result["total"].clamp(max=50.0)

        # Restore lambda_compact and lambda_balance after forward pass
        self.objective.lambda_compact = saved_lambda_compact
        if hasattr(self.objective, "lambda_balance"):
            self.objective.lambda_balance = saved_lambda_balance

        # Feature separation loss (Issue 2 fix): push F_0 and F_1 apart
        la_sep = float(getattr(self.train_cfg, "la_feature_sep", 0.1))
        if la_sep > 0:
            l_sep = loss_feature_separation(F_0, F_1)
            l_struct = l_struct + la_sep * l_sep

        # ── Priority 4 losses ─────────────────────────────────────────────────
        # Spatial coherence: TV regularization to encourage connected entity regions
        la_sc = float(getattr(self.train_cfg, "lambda_spatial_coherence", 0.0))
        if la_sc > 0 and vol_outputs.entity_probs is not None:
            l_sc = loss_spatial_coherence(vol_outputs.entity_probs)
            l_struct = l_struct + la_sc * l_sc

        # FG coverage prior: prevent all-background collapse
        la_fg_prior = float(getattr(self.train_cfg, "lambda_fg_prior", 0.0))
        if la_fg_prior > 0 and vol_outputs.entity_probs is not None:
            _min_fg = float(getattr(self.train_cfg, "min_fg_fraction", 0.05))
            l_fg_prior = loss_fg_coverage_prior(vol_outputs.entity_probs, _min_fg)
            l_struct = l_struct + la_fg_prior * l_fg_prior

        # Permutation consistency: penalise frame-to-frame entity label flips
        # Requires storing entity_probs from previous training step
        la_perm = float(getattr(self.train_cfg, "lambda_perm_consist", 0.0))
        if la_perm > 0 and vol_outputs.entity_probs is not None:
            if not hasattr(self, "_prev_entity_probs") or self._prev_entity_probs is None:
                self._prev_entity_probs = vol_outputs.entity_probs.detach().clone()
            else:
                if self._prev_entity_probs.shape == vol_outputs.entity_probs.shape:
                    l_perm = loss_permutation_consistency(
                        vol_outputs.entity_probs, self._prev_entity_probs)
                    l_struct = l_struct + la_perm * l_perm
                self._prev_entity_probs = vol_outputs.entity_probs.detach().clone()

        # ── Amodal entity coverage: ensure BOTH entities have amodal presence ──
        # Critical for four_stream guide: back_e0/back_e1 streams carry occluded
        # entity signal. If either entity's amodal field is all-zero, its back
        # stream is dead and guide provides no identity-preserving signal for
        # that entity during contact/occlusion frames.
        la_amo = float(getattr(self.train_cfg, "lambda_amodal_coverage", 0.0))
        if la_amo > 0 and vol_outputs.entity_probs is not None:
            _min_amo = float(getattr(self.train_cfg, "min_amodal_coverage", 0.02))
            l_amo = loss_amodal_entity_coverage(vol_outputs, min_coverage=_min_amo)
            l_struct = l_struct + la_amo * l_amo

        # ── Temporal centroid consistency: entity identity across frames ────────
        # Centroid-based version of permutation consistency.
        # More robust than volumetric overlap during contact frames (where
        # spatial overlap of two entities is expected and valid).
        # Requires frame-level entity_probs; uses _frame_ep_buffer to accumulate.
        la_tc = float(getattr(self.train_cfg, "lambda_temporal_centroid", 0.0))
        if la_tc > 0 and vol_outputs.entity_probs is not None:
            if not hasattr(self, "_frame_ep_buffer"):
                self._frame_ep_buffer = []
            self._frame_ep_buffer.append(vol_outputs.entity_probs.detach())
            # Keep last 4 frames for temporal consistency
            if len(self._frame_ep_buffer) > 4:
                self._frame_ep_buffer = self._frame_ep_buffer[-4:]
            if len(self._frame_ep_buffer) >= 2:
                # Use current frame + 1 previous to compute centroid shift
                _ep_pair = [self._frame_ep_buffer[-2], vol_outputs.entity_probs]
                l_tc = loss_temporal_centroid_consistency(_ep_pair)
                l_struct = l_struct + la_tc * l_tc

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

        # v38: Entity-isolated diffusion loss (L_iso)
        # Forces UNet to generate each entity separately when solo frames are available.
        # Only active in stage2/3 when use_diffusion=True and lambda_iso > 0.
        # Run every 4 steps to limit additional compute (~25% overhead vs full every step).
        l_iso = torch.tensor(0.0, device=self.device)
        lambda_iso = float(getattr(self.train_cfg, "lambda_iso", 0.0))
        _iso_run = (
            lambda_iso > 0.0 and
            use_diffusion and
            stage in ("stage2", "stage3") and
            (step_idx % 4 == 0) and  # every 4 steps to limit overhead
            vol_outputs.entity_probs is not None
        )
        if _iso_run:
            _iso_losses = []
            _solo_frames_list = [frames_solo0, frames_solo1]
            for _ent_idx in range(2):
                _solo_frames = _solo_frames_list[_ent_idx]
                if _solo_frames is None:
                    continue
                try:
                    # Encode solo frames as latents (entity rendered alone)
                    with torch.no_grad():
                        _solo_latents = encode_frames_to_latents(
                            self.pipe, _solo_frames, self.device)
                    _solo_noise = torch.randn_like(_solo_latents)
                    _solo_noisy = self.pipe.scheduler.add_noise(_solo_latents, _solo_noise, t)

                    # Build masked entity_probs: zero out the OTHER entity
                    # entity_probs shape: (B, 2, K, H, W)
                    _ep_masked = vol_outputs.entity_probs.clone()
                    _other_idx = 1 - _ent_idx
                    _ep_masked[:, _other_idx] = 0.0  # mask out the other entity

                    # Build masked VolumeOutputs for the isolated pass
                    from training.phase62.objectives.base import VolumeOutputs as _VO
                    _vol_masked = _VO(
                        entity_probs=_ep_masked,
                        entity_logits=vol_outputs.entity_logits,
                        fg_logit=vol_outputs.fg_logit,
                        fg_spatial_logit=vol_outputs.fg_spatial_logit,
                        id_logits=vol_outputs.id_logits,
                        visible_class=vol_outputs.visible_class,
                        visible=vol_outputs.visible,
                        amodal=vol_outputs.amodal,
                        front_probs=vol_outputs.front_probs,
                        back_probs=vol_outputs.back_probs,
                    )

                    # Re-project and assemble guide with masked entity_probs
                    _vol_masked_proj, _guides_iso = self.system.project_and_assemble(
                        _vol_masked, F_g, F_0, F_1)

                    # Run isolated diffusion pass
                    self.system.set_guides(_guides_iso)
                    self.backbone_mgr.reset_slot_store()
                    _T_solo = min(_solo_frames.shape[0], T_frames)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        _solo_pred = self.pipe.unet(
                            _solo_noisy, t, encoder_hidden_states=enc_full).sample

                    _solo_noise_t = _solo_noise[:, :, :_T_solo].float()
                    _solo_pred_t = _solo_pred[:, :, :_T_solo].float()
                    _l_iso_e = loss_diffusion(_solo_pred_t, _solo_noise_t)
                    _iso_losses.append(_l_iso_e)

                    self.system.clear_guides()
                except Exception as _iso_e:
                    self.system.clear_guides()
                    continue  # non-fatal: skip this entity's isolated pass

            if _iso_losses:
                l_iso = torch.stack(_iso_losses).mean()

        # Total loss
        la_vol = float(getattr(self.train_cfg, "la_vol", 2.0))
        la_diff = float(getattr(self.train_cfg, "la_diff", 1.0))

        # Solution 2: Gradual diffusion loss warm-up at stage transition
        # Prevents cold-start explosion when guide network first activates
        diff_warmup_epochs = int(getattr(self.train_cfg, "diff_warmup_epochs", 5))
        diff_weight = 0.0  # default (stage1 / S0)

        if stage == "stage1" or self.schedule == "S0":
            loss = la_vol * l_struct
        elif stage == "stage2":
            vol_scale = 0.0 if self.schedule == "S1" else 0.25
            current_stage_epoch = current_epoch - self.stage1_end
            diff_weight = min(1.0, (current_stage_epoch + 1) / max(1, diff_warmup_epochs))
            # Gate-adaptive: if guide gate hasn't opened, slow down diffusion ramp
            gate_open, _ = self.contract._gate_open(self.system.assembler)
            diff_weight = self.contract.adaptive_diff_weight(diff_weight, gate_open)
            loss = diff_weight * la_diff * l_diff + la_vol * vol_scale * l_struct
        else:
            # stage3: full joint, brief warmup when entering from stage2
            current_stage_epoch = current_epoch - self.stage2_end
            diff_weight = min(1.0, (current_stage_epoch + 1) / max(1, diff_warmup_epochs // 2))
            gate_open, _ = self.contract._gate_open(self.system.assembler)
            diff_weight = self.contract.adaptive_diff_weight(diff_weight, gate_open)
            loss = diff_weight * la_diff * l_diff + la_vol * 0.25 * l_struct

        # v38: Add entity-isolated loss (L_iso) — only in stage2/3 when l_iso > 0
        if lambda_iso > 0.0 and l_iso.item() > 0.0:
            loss = loss + lambda_iso * l_iso

        # v18: Gate push loss — directly maximises tanh(gate_param) to bypass the
        # std-normalisation in inject_guide_into_unet_features which cancels the gate
        # gradient: guide_std = gate × proj_std → norm rescales by 1/gate → dL/d_gate≈0.
        # This auxiliary loss provides a clean gradient path independent of diffusion quality.
        # Only active in stage2+ (guide is assembled); zero in stage1 / S0.
        lambda_gate_push = float(getattr(self.train_cfg, "lambda_gate_push", 0.0))
        if lambda_gate_push > 0 and use_diffusion and hasattr(self.system.assembler, "guide_gates"):
            gate_vals = [torch.tanh(g) for g in self.system.assembler.guide_gates.values()]
            if gate_vals:
                l_gate_push = -lambda_gate_push * torch.stack(gate_vals).mean()
                loss = loss + l_gate_push

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

        # Solution 3: Gradient clipping — per-group then global safety net
        # fg_spatial is clipped TOGETHER with volume so its tiny param group (65 params)
        # does not get ~40× larger per-param updates than v14 (which had all volume
        # params in one group). Separate lr control in stage2 still uses the split group.
        torch.nn.utils.clip_grad_norm_(
            self.param_groups["volume"] + self.param_groups["fg_spatial"], max_norm=0.3)
        torch.nn.utils.clip_grad_norm_(self.param_groups["assembler"], max_norm=0.3)
        torch.nn.utils.clip_grad_norm_(self.param_groups["adapter"], max_norm=0.3)
        torch.nn.utils.clip_grad_norm_(self.param_groups["lora"], max_norm=0.2)
        all_params = (self.param_groups["volume"] + self.param_groups["fg_spatial"]
                      + self.param_groups["assembler"]
                      + self.param_groups["adapter"] + self.param_groups["lora"])
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

        self.optimizer.step()

        # v20: Gate clamp in stage3 — force gate_param ≥ atanh(min_gate_stage3).
        # Rationale: diffusion gradient brings gate to equilibrium ≈0.042 (v17 pattern),
        # which is below the S3 threshold of 0.05. gate_push (v18/v19) over-grows gate
        # and hurts iou_min because diffusion gradient dominates regardless of lambda.
        # Clamp bypasses both problems: gate is forced ≥ target without affecting
        # stage2 gradient dynamics (iou_min grows freely as in v17).
        min_gate = float(getattr(self.train_cfg, "min_gate_stage3", 0.0))
        if min_gate > 0 and stage == "stage3" and hasattr(self.system.assembler, "guide_gates"):
            import math
            # atanh(x) = 0.5 * ln((1+x)/(1-x))
            min_gate_param = 0.5 * math.log((1.0 + min_gate) / (1.0 - min_gate))
            for gate_p in self.system.assembler.guide_gates.values():
                gate_p.data.clamp_(min=min_gate_param)

        # v23: Gate ceiling clamp — force gate_param ≤ atanh(max_gate) in stage2+stage3.
        # Gate grows freely through diffusion gradient. Without an upper bound it overshoots
        # CGUIDE_GATE_HI=0.35 (v22 reached 0.48+), over-injecting guide and collapsing
        # overlay (0.38→0.31 as gate: 0.31→0.48). Hard clamp is the simplest correct fix —
        # d(guide_eff)/d(gate) stays non-zero below ceiling; clamp only activates at boundary.
        max_gate = float(getattr(self.train_cfg, "max_gate", 1.0))
        if max_gate < 1.0 and stage in ("stage2", "stage3") and hasattr(self.system.assembler, "guide_gates"):
            import math
            max_gate_param = 0.5 * math.log((1.0 + max_gate) / (1.0 - max_gate))
            for gate_p in self.system.assembler.guide_gates.values():
                gate_p.data.clamp_(max=max_gate_param)

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
            "l_iso": float(l_iso.item()),
            "stage": stage,
            "acc_overall": acc_overall,
            "acc_entity": acc_entity,
            "diff_weight": float(diff_weight),
        }
        for k, v in obj_result.items():
            if k != "total" and isinstance(v, torch.Tensor):
                result[k] = float(v.item())

        # ── Training debug visualisation (once per epoch, step 0)
        if debug_step:
            try:
                _cm = self._last_contract_metrics  # may be None early on
                with torch.no_grad():
                    self.train_viz.save_volume_debug(
                        vol_outputs, V_gt, gt_visible, gt_amodal,
                        epoch=epoch, step=step_idx, stage=stage,
                        contract_metrics=_cm,
                        frames_np=frames_np)
                    if use_diffusion and guides:
                        self.train_viz.save_guide_debug(
                            guides, self.system.assembler, epoch=epoch, stage=stage,
                            contract_metrics=_cm)
                    if use_diffusion and 'noise_pred' in locals():
                        noise_t_dbg = noise[:, :, :T_frames].float()
                        np_t_dbg = noise_pred[:, :, :T_frames].float()
                        self.train_viz.save_diffusion_debug(
                            np_t_dbg, noise_t_dbg,
                            epoch=epoch, step=step_idx, stage=stage,
                            diff_weight=float(diff_weight))
            except Exception as _e:
                pass  # never crash training for debug viz

        return result

    @torch.no_grad()
    def _eval_epoch(self, epoch: int) -> Dict:
        self.system.eval()
        self.backbone_mgr.eval()
        self._first_val_cached = False  # reset per-eval cache

        diff_losses, struct_losses = [], []
        accs_overall, accs_entity = [], []
        ious_e0, ious_e1 = [], []
        amo_dice_e0, amo_dice_e1 = [], []  # v25: amodal Dice per entity
        compacts_e0, compacts_e1 = [], []  # per-sample compact for averaged metric
        lcc_e0_vals, lcc_e1_vals = [], []  # v26: LCC per sample for stable averaged metric
        cos_F_list: list = []             # v38: cos(F_0, F_1) at entity overlap region

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

                # Build depth hint for val step
                depth_hint_val = None
                if depth_np is not None:
                    B_feat_hint = F_g.shape[0]
                    dmap_v = depth_np[:B_feat_hint].astype(np.float32)
                    # Match VolumeGTBuilder: min-max normalization
                    dmin_v = float(dmap_v.min())
                    dmax_v = float(dmap_v.max())
                    drange_v = max(dmax_v - dmin_v, 1e-6)
                    dmap_v_norm = torch.from_numpy((dmap_v - dmin_v) / drange_v).to(self.device)
                    H_vol, W_vol = self.config.spatial_h, self.config.spatial_w
                    if dmap_v_norm.shape[-2:] != (H_vol, W_vol):
                        dmap_v_norm = F.interpolate(
                            dmap_v_norm.unsqueeze(1).float(),
                            size=(H_vol, W_vol), mode="bilinear", align_corners=False,
                        ).squeeze(1)
                    if dmap_v_norm.shape[0] < B_feat_hint:
                        n_rep = max(1, B_feat_hint // dmap_v_norm.shape[0])
                        dmap_v_norm = dmap_v_norm.repeat(n_rep, 1, 1)[:B_feat_hint]
                    depth_hint_val = dmap_v_norm

                vol_outputs = self.system.predict_volume(F_g, F_0, F_1,
                                                          depth_hint=depth_hint_val)

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

                # Cache FIRST val sample for contract (representative, not last)
                if not hasattr(self, "_first_val_cached") or not self._first_val_cached:
                    self._last_val_vol_outputs = vol_outputs
                    self._last_val_gt_visible  = gt_visible
                    self._first_val_cached = True

                obj_result = self.objective(vol_outputs, V_gt, gt_visible=gt_visible, gt_amodal=gt_amodal)
                struct_losses.append(float(obj_result["total"].item()))

                # Accumulate compact and LCC per sample for averaged metrics
                if vol_outputs.entity_probs is not None:
                    fg_sp = (V_gt > 0).any(dim=1)  # (B, H, W)
                    for b in range(vol_outputs.entity_probs.shape[0]):
                        ep_b = vol_outputs.entity_probs[b]  # (2, K, H, W)
                        fg_b = fg_sp[b] if fg_sp.shape[0] > b else None
                        from training.phase62.contract import DebugContract
                        c_e0 = DebugContract._depth_compactness(ep_b[0], fg_b)
                        c_e1 = DebugContract._depth_compactness(ep_b[1], fg_b)
                        compacts_e0.append(c_e0)
                        compacts_e1.append(c_e1)
                        # v26: LCC averaged over all eval samples (single-sample LCC is too noisy)
                        lcc_e0_vals.append(DebugContract._compute_lcc(ep_b[0]))
                        lcc_e1_vals.append(DebugContract._compute_lcc(ep_b[1]))

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

                    # v25: amodal Dice — compare vol amodal projection to entity GT mask.
                    # entity_masks are at spatial_h×spatial_w (downsampled), shape (T, 2, S).
                    # amodal["e0"] = 1 - prod(1 - entity_probs[:, 0]) over depth bins.
                    if "e0" in vol_outputs.amodal and "e1" in vol_outputs.amodal:
                        n_amo = min(n_eval, int(entity_masks.shape[0]))
                        if n_amo > 0:
                            amo_e0 = vol_outputs.amodal["e0"][:n_amo]  # (B, H, W)
                            amo_e1 = vol_outputs.amodal["e1"][:n_amo]
                            amo_gt0 = torch.from_numpy(
                                entity_masks[:n_amo, 0].astype(np.float32)).to(self.device)
                            amo_gt1 = torch.from_numpy(
                                entity_masks[:n_amo, 1].astype(np.float32)).to(self.device)
                            # Reshape GT to (B, H, W): entity_masks stored as (T, 2, S=HW)
                            S_amo = amo_gt0.shape[-1]
                            hw_amo = int(round(S_amo ** 0.5))
                            if hw_amo * hw_amo == S_amo:
                                amo_gt0 = amo_gt0.reshape(n_amo, hw_amo, hw_amo)
                                amo_gt1 = amo_gt1.reshape(n_amo, hw_amo, hw_amo)
                                # Dice: 2*inter / (pred_sum + gt_sum)
                                eps = 1e-6
                                inter0 = (amo_e0.float() * amo_gt0).sum(dim=(1, 2))
                                denom0 = amo_e0.float().sum(dim=(1, 2)) + amo_gt0.sum(dim=(1, 2))
                                d0 = float(((2.0 * inter0 + eps) / (denom0 + eps)).mean().item())
                                inter1 = (amo_e1.float() * amo_gt1).sum(dim=(1, 2))
                                denom1 = amo_e1.float().sum(dim=(1, 2)) + amo_gt1.sum(dim=(1, 2))
                                d1 = float(((2.0 * inter1 + eps) / (denom1 + eps)).mean().item())
                                amo_dice_e0.append(d0)
                                amo_dice_e1.append(d1)

                # v38: cos(F_0, F_1) at entity overlap region
                # overlap = pixels where BOTH entity GT masks are nonzero.
                # Low value = good feature separation; high = model conflates entities.
                if F_0 is not None and F_1 is not None and entity_masks.shape[0] > 0:
                    try:
                        B_cos = min(F_0.shape[0], entity_masks.shape[0])
                        if B_cos > 0:
                            # F_0, F_1: (B, S, D) — spatial features per frame
                            f0 = F.normalize(F_0[:B_cos].float(), dim=-1, eps=1e-6)
                            f1 = F.normalize(F_1[:B_cos].float(), dim=-1, eps=1e-6)
                            cos_all = (f0 * f1).sum(dim=-1)  # (B, S)

                            # Build overlap mask from entity_masks (T, 2, S)
                            m0_cos = torch.from_numpy(
                                entity_masks[:B_cos, 0].astype(np.float32)).to(self.device)
                            m1_cos = torch.from_numpy(
                                entity_masks[:B_cos, 1].astype(np.float32)).to(self.device)
                            overlap_cos = (m0_cos > 0.5) & (m1_cos > 0.5)  # (B, S)

                            if overlap_cos.any():
                                cos_at_overlap = cos_all[overlap_cos]
                                cos_F_list.append(float(cos_at_overlap.mean().clamp(0, 1).item()))
                            else:
                                cos_F_list.append(float(cos_all.mean().clamp(0, 1).item()))
                    except Exception:
                        pass  # non-fatal

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

        # Average compact over all val samples (more reliable than single-sample measurement)
        val_compact = min(_avg(compacts_e0), _avg(compacts_e1))

        # v25: store amodal dice for contract (getattr default=0.0)
        self._val_amo_dice_e0 = _avg(amo_dice_e0)
        self._val_amo_dice_e1 = _avg(amo_dice_e1)
        # v26: store averaged LCC for contract (more stable than single-sample)
        self._val_lcc_e0 = _avg(lcc_e0_vals)
        self._val_lcc_e1 = _avg(lcc_e1_vals)
        # v38: store averaged cos_F_overlap (computed inline above, not via Phase62Evaluator)
        self._val_cos_F_overlap = _avg(cos_F_list)

        # ── Compute last vol_outputs for contract (reuse last val sample)
        _contract_vol = getattr(self, "_last_val_vol_outputs", None)
        _contract_gt_vis = getattr(self, "_last_val_gt_visible", None)
        if _contract_vol is not None and _contract_gt_vis is not None:
            _cstage = self._get_stage(epoch)
            # v39b: pull accumulated guide/delta norms from training steps this epoch
            _guide_norm_avg = float(sum(self._step_guide_norm_acc) / max(1, len(self._step_guide_norm_acc))) \
                if getattr(self, "_step_guide_norm_acc", None) else 0.0
            _delta_norm_avg = float(sum(self._step_delta_norm_acc) / max(1, len(self._step_delta_norm_acc))) \
                if getattr(self, "_step_delta_norm_acc", None) else 0.0
            # Reset accumulators for next epoch
            self._step_guide_norm_acc = []
            self._step_delta_norm_acc = []
            # v39c: feature_sep_active = la_feature_sep > 0
            _feature_sep_active = float(getattr(self.train_cfg, "la_feature_sep", 0.0)) > 0.0
            # scene_type: "occ" for depth-separated data, "col" for same-depth (default)
            _scene_type = getattr(self.train_cfg, "scene_type", "col")
            _contract_metrics = self.contract.compute(
                vol_outputs=_contract_vol,
                assembler=self.system.assembler,
                gt_visible=_contract_gt_vis,
                val_metrics={
                    "val_iou_min":   val_iou_min,
                    "val_iou_e0":    val_iou_e0,
                    "val_iou_e1":    val_iou_e1,
                    "val_diff_mse":  _avg(diff_losses),
                    "val_compact":   val_compact,
                    # v22/v38 new metrics
                    "val_amo_dice_e0": getattr(self, "_val_amo_dice_e0", 0.0),
                    "val_amo_dice_e1": getattr(self, "_val_amo_dice_e1", 0.0),
                    "val_cos_F_overlap": getattr(self, "_val_cos_F_overlap", 0.0),
                    "val_lcc_e0": getattr(self, "_val_lcc_e0", 1.0),
                    "val_lcc_e1": getattr(self, "_val_lcc_e1", 1.0),
                    "val_pass_rate_clips": getattr(self, "_val_pass_rate_clips", 0.0),
                    # v39b: injection path diagnostics
                    "val_guide_feature_norm": _guide_norm_avg,
                    "val_injected_delta_norm": _delta_norm_avg,
                    # v39c: feature sep flag
                    "feature_sep_active": _feature_sep_active,
                },
                epoch=epoch,
                stage=_cstage,
                scene_type=_scene_type,
            )
            self._last_contract_metrics = _contract_metrics
            self.contract.log(_contract_metrics)

        result = {
            "val_score": val_score,
            "val_diff_mse": _avg(diff_losses),
            "val_struct": val_struct,
            "val_acc_overall": _avg(accs_overall),
            "val_acc_entity": val_acc_ent,
            "val_iou_e0": val_iou_e0,
            "val_iou_e1": val_iou_e1,
            "val_iou_min": val_iou_min,
            "val_compact": val_compact,
            "n_samples": len(diff_losses),
        }

        print(
            f"  [val] score={val_score:.4f}  "
            f"struct={val_struct:.4f}  "
            f"acc_ent={val_acc_ent:.4f}  "
            f"iou_e0={val_iou_e0:.4f}  "
            f"iou_e1={val_iou_e1:.4f}  "
            f"iou_min={val_iou_min:.4f}  "
            f"compact={val_compact:.3f}  "
            f"n={len(diff_losses)}",
            flush=True)

        # GIF + isolated rollouts (v38: C_render metrics)
        render_metrics = None
        try:
            probe = self.dataset[self.val_idx[0]]
            if isinstance(probe, dict):
                probe_meta = probe["meta"]
                probe_masks = probe["entity_masks"]
                probe_frames = probe["frames"]
                probe_solo0 = probe.get("frames_solo0")
                probe_solo1 = probe.get("frames_solo1")
            else:
                probe_meta, probe_masks, probe_frames = probe[3], probe[4], probe[0]
                probe_solo0 = probe_solo1 = None

            toks_e0_pt, toks_e1_pt, probe_prompt = \
                _get_entity_tokens_with_fallback(self.pipe, probe_meta, self.device)

            rollout_result = self.rollout_runner.generate_rollout(
                self.pipe, self.system, self.backbone_mgr,
                probe_prompt, self.config, self.device,
                toks_e0=toks_e0_pt, toks_e1=toks_e1_pt,
                entity_masks=probe_masks, gt_frames=probe_frames)

            eval_out_dir = self.debug_dir / "eval"
            eval_out_dir.mkdir(parents=True, exist_ok=True)
            paths = self.rollout_runner.save_rollout(
                rollout_result, eval_out_dir, prefix=f"eval_epoch{epoch:03d}")
            for name, path in paths.items():
                print(f"  [gif] {name}: {path}", flush=True)

            # v38: Isolated rollouts for C_render evaluation
            # Only run if solo frames are available (to compute render_iou_min)
            train_cfg = self.train_cfg
            height = getattr(train_cfg, 'height', 256)
            width = getattr(train_cfg, 'width', 256)
            iso_result_e0 = None
            iso_result_e1 = None
            try:
                iso_result_e0 = self.rollout_runner.generate_isolated_rollout(
                    self.pipe, self.system, self.backbone_mgr,
                    probe_prompt, entity_idx=0, config=self.config, device=self.device,
                    toks_e0=toks_e0_pt, toks_e1=toks_e1_pt)
                # Save isolated rollout GIF
                if iso_result_e0 and iso_result_e0.get("frames"):
                    import imageio.v2 as _iio2
                    iso_gif = eval_out_dir / f"eval_epoch{epoch:03d}_iso_e0.gif"
                    _iio2.mimwrite(str(iso_gif), iso_result_e0["frames"], fps=8, loop=0)
                    print(f"  [gif] iso_e0: {iso_gif}", flush=True)
            except Exception as _ie:
                print(f"  [warn] isolated rollout e0 failed: {_ie}", flush=True)
            try:
                iso_result_e1 = self.rollout_runner.generate_isolated_rollout(
                    self.pipe, self.system, self.backbone_mgr,
                    probe_prompt, entity_idx=1, config=self.config, device=self.device,
                    toks_e0=toks_e0_pt, toks_e1=toks_e1_pt)
                if iso_result_e1 and iso_result_e1.get("frames"):
                    import imageio.v2 as _iio2
                    iso_gif = eval_out_dir / f"eval_epoch{epoch:03d}_iso_e1.gif"
                    _iio2.mimwrite(str(iso_gif), iso_result_e1["frames"], fps=8, loop=0)
                    print(f"  [gif] iso_e1: {iso_gif}", flush=True)
            except Exception as _ie:
                print(f"  [warn] isolated rollout e1 failed: {_ie}", flush=True)

            # Compute C_render metrics if any rollout succeeded
            if iso_result_e0 is not None or iso_result_e1 is not None:
                render_metrics = self.rollout_runner.compute_render_metrics(
                    composite_result=rollout_result,
                    iso_result_e0=iso_result_e0,
                    iso_result_e1=iso_result_e1,
                    solo_frames_e0=probe_solo0,
                    solo_frames_e1=probe_solo1,
                    entity_masks=probe_masks,
                    height=height,
                    width=width,
                )
                print(
                    f"  [render] P_2obj={render_metrics['P_2obj']:.3f}  "
                    f"chimera={render_metrics['R_chimera']:.3f}  "
                    f"M_id={render_metrics['M_id_min']:.3f}  "
                    f"iou_min={render_metrics['render_iou_min']:.3f}",
                    flush=True)

        except Exception as e:
            print(f"  [warn] GIF failed: {e}", flush=True)

        # Re-compute contract with render_metrics if available
        if render_metrics is not None:
            _contract_vol = getattr(self, "_last_val_vol_outputs", None)
            _contract_gt_vis = getattr(self, "_last_val_gt_visible", None)
            if _contract_vol is not None and _contract_gt_vis is not None:
                _cstage = self._get_stage(epoch)
                _feature_sep_active2 = float(getattr(self.train_cfg, "la_feature_sep", 0.0)) > 0.0
                _scene_type2 = getattr(self.train_cfg, "scene_type", "col")
                _guide_norm_r = getattr(self, "_last_guide_norm_for_contract", 0.0)
                _delta_norm_r = getattr(self, "_last_delta_norm_for_contract", 0.0)
                _contract_metrics = self.contract.compute(
                    vol_outputs=_contract_vol,
                    assembler=self.system.assembler,
                    gt_visible=_contract_gt_vis,
                    val_metrics={
                        "val_iou_min":   val_iou_min,
                        "val_iou_e0":    val_iou_e0,
                        "val_iou_e1":    val_iou_e1,
                        "val_diff_mse":  _avg(diff_losses),
                        "val_compact":   val_compact,
                        "val_amo_dice_e0": getattr(self, "_val_amo_dice_e0", 0.0),
                        "val_amo_dice_e1": getattr(self, "_val_amo_dice_e1", 0.0),
                        "val_cos_F_overlap": getattr(self, "_val_cos_F_overlap", 0.0),
                        "val_lcc_e0": getattr(self, "_val_lcc_e0", 1.0),
                        "val_lcc_e1": getattr(self, "_val_lcc_e1", 1.0),
                        "val_pass_rate_clips": getattr(self, "_val_pass_rate_clips", 0.0),
                        "val_guide_feature_norm": _guide_norm_r,
                        "val_injected_delta_norm": _delta_norm_r,
                        "feature_sep_active": _feature_sep_active2,
                    },
                    epoch=epoch,
                    stage=_cstage,
                    scene_type=_scene_type2,
                    render_metrics=render_metrics,
                )
                self._last_contract_metrics = _contract_metrics
                print(f"  [contract+render] recomputed with C_render metrics", flush=True)
                self.contract.log(_contract_metrics)

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
        # Multi-objective checkpoint selection:
        # 60% contract_score (compactness + gate + iou_min + two-color + stability)
        # 40% legacy val_score
        _cm = self._last_contract_metrics
        contract_score = _cm.contract_score if _cm is not None else 0.0
        sel = 0.60 * contract_score + 0.40 * val_m.get("val_score", 0.0)
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

            torch.cuda.empty_cache()
            self.system.train()
            self.backbone_mgr.train()
            self._set_epoch_trainability(epoch)

            losses, accs_ent = [], []
            chosen = np.random.choice(len(self.train_idx), size=steps_per_epoch,
                                      replace=True, p=self.sample_weights)
            step_indices = [self.train_idx[ci] for ci in chosen]

            for batch_idx, data_idx in enumerate(step_indices):
                sample = self.dataset[data_idx]
                is_debug = (batch_idx == 0)  # save debug viz on first step of each epoch
                r = self._train_step(sample, data_idx, epoch, current_epoch=epoch,
                                     debug_step=is_debug, step_idx=batch_idx)
                if r is None:
                    continue
                losses.append(r["loss"])
                accs_ent.append(r["acc_entity"])
                self.train_step_history.append({"epoch": epoch, **r})

            stage = self._get_stage(epoch)
            avg_loss = sum(losses) / max(len(losses), 1) if losses else float("nan")
            avg_acc = sum(accs_ent) / max(len(accs_ent), 1) if accs_ent else 0.0
            # Compute diff_weight for logging (mirrors _train_step logic)
            _warmup = int(getattr(self.train_cfg, "diff_warmup_epochs", 5))
            if stage == "stage1" or self.schedule == "S0":
                _dw = 0.0
            elif stage == "stage2":
                _dw = min(1.0, (epoch - self.stage1_end + 1) / max(1, _warmup))
            else:
                _dw = min(1.0, (epoch - self.stage2_end + 1) / max(1, _warmup // 2))
            print(f"  [ep {epoch:3d}] {stage}  loss={avg_loss:.4f}  "
                  f"acc_ent={avg_acc:.4f}  diff_w={_dw:.2f}  steps={len(losses)}/{steps_per_epoch}",
                  flush=True)

            self.lr_scheduler.step()

            # Epoch-level loss curve update (using per-step history)
            epoch_summary = {
                "epoch": epoch,
                "loss": avg_loss if np.isfinite(avg_loss) else float("nan"),
                "acc_entity": avg_acc,
                "stage": stage,
                "diff_weight": _dw,
            }
            # Pull component losses from last step in this epoch
            last_steps = [r for r in self.train_step_history if r["epoch"] == epoch]
            if last_steps:
                for key in ("l_diff", "l_struct"):
                    vals = [r[key] for r in last_steps if key in r and np.isfinite(r[key])]
                    epoch_summary[key] = sum(vals) / max(len(vals), 1) if vals else float("nan")
            self.train_step_history_epoch = getattr(self, "train_step_history_epoch", [])
            self.train_step_history_epoch.append(epoch_summary)

            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                val_m = self._eval_epoch(epoch)
                self._save_checkpoint(epoch, val_m)
                self.history.append({"epoch": epoch, **val_m})
                self.val_history.append({"epoch": epoch, **val_m})

                # Update loss curve with latest val data
                stage_bounds = {
                    "s1→s2": self.stage1_end,
                    "s2→s3": self.stage2_end,
                }
                self.train_viz.update_loss_curve(
                    self.train_step_history_epoch,
                    self.val_history,
                    stage_bounds,
                )

        # Save history
        hist_path = self.save_dir / "history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Ablation] Done. Best epoch={self.best_epoch} "
              f"val_score={self.best_val_score:.4f}", flush=True)
        print(self.contract.summary(), flush=True)
