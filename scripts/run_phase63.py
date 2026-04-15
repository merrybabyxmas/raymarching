"""
Phase 63 — Dynamic Entity Fields Training Script
================================================

Stage 1: EntityField + TransmittanceRenderer pretrain (no diffusion)
  - Independent density per entity (no class competition)
  - losses: visible_dice + amodal_dice + occlusion_consistency + entity_survival

Usage:
    CUDA_VISIBLE_DEVICES=3 conda run -n paper_env python scripts/run_phase63.py \\
        --config config/phase63/stage1.yaml [--ckpt path/to/checkpoint.pt]

Output:
    checkpoints/phase63/<run_name>/   — model checkpoints
    outputs/phase63/<run_name>/       — debug visualizations + metrics JSON
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from models.phase62.system import Phase63System
from models.phase62.backbone_adapter import (
    inject_backbone_extractors,
    BackboneManager,
    DEFAULT_INJECT_KEYS,
)
from data.phase62.dataset_adapter import Phase62DatasetAdapter
from data.phase62.volume_gt_builder import VolumeGTBuilder
from training.losses_entity import (
    loss_visible_dice,
    loss_amodal_dice,
    loss_occlusion_consistency,
    loss_entity_survival,
    loss_identity_separation,
    loss_temporal_slot_consistency,
    compute_entity_metrics,
)
from training.evaluator_entity import EntityEvaluator
from scripts.train_phase39 import (
    compute_dataset_overlap_scores,
    split_train_val,
    make_sampling_weights,
)
from scripts.train_animatediff_vca import encode_frames_to_latents
from training.phase62.evaluator import _encode_text
from scripts.prompt_identity import make_identity_prompts
from models.vca_volumetric import find_entity_token_positions

try:
    import imageio.v2 as iio2
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False

MIN_VAL_SAMPLES = 3


def _get_entity_tokens_p63(pipe, meta: dict, device: str):
    """Phase 63 entity token finder: prioritizes color-specific entity prompts.

    For same-keyword pairs (cat+cat), the generic fallback assigns both entities
    to the SAME token position → F_e0 ≡ F_e1.  This version searches
    prompt_entity0/prompt_entity1 (e.g. "a red cat", "a blue cat") FIRST to get
    distinct positions, then falls back to keyword search.

    Also returns full_prompt so the caller can encode the same text.
    """
    _, _, full_prompt, _, _ = make_identity_prompts(meta)

    def _find(prompt_key: str, kw_key: str) -> list[int]:
        # 1. Try entity-specific color prompt: "a red cat" → ["a red cat", "red cat"]
        specific = str(meta.get(prompt_key, "")).strip()
        if specific:
            toks = find_entity_token_positions(pipe.tokenizer, full_prompt, specific)
            if toks:
                return toks
            for prefix in ("a ", "an ", "the "):
                if specific.lower().startswith(prefix):
                    stripped = specific[len(prefix):].strip()
                    toks = find_entity_token_positions(pipe.tokenizer, full_prompt, stripped)
                    if toks:
                        return toks
                    break
        # 2. Fall back to keyword
        kw = str(meta.get(kw_key, "")).strip()
        if kw:
            toks = find_entity_token_positions(pipe.tokenizer, full_prompt, kw)
            if toks:
                return toks
        return [1, 2]  # absolute fallback

    toks_e0 = _find("prompt_entity0", "keyword0")
    toks_e1 = _find("prompt_entity1", "keyword1")

    # If same keyword caused collision (cat+cat), find second occurrence for e1
    if toks_e0 == toks_e1:
        kw1 = str(meta.get("keyword1", "")).strip()
        if kw1:
            all_ids = pipe.tokenizer(full_prompt, add_special_tokens=True).input_ids
            kw1_ids = pipe.tokenizer(kw1, add_special_tokens=False).input_ids
            n = len(kw1_ids)
            found_first = False
            for i in range(len(all_ids) - n + 1):
                if all_ids[i:i + n] == kw1_ids:
                    if not found_first:
                        found_first = True
                    else:
                        toks_e1 = list(range(i, i + n))
                        break

    toks_e0_t = torch.tensor(toks_e0, dtype=torch.long, device=device)
    toks_e1_t = torch.tensor(toks_e1, dtype=torch.long, device=device)
    return toks_e0_t, toks_e1_t, full_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_gt_masks(src_masks: np.ndarray, spatial_h: int, spatial_w: int,
                    device: str) -> torch.Tensor:
    """Resize (T, 2, S) masks → (T, 2, spatial_h, spatial_w) float tensor."""
    T, _, S = src_masks.shape
    hw = int(round(S ** 0.5))
    gt = torch.from_numpy(src_masks.astype(np.float32)).to(device)
    gt = gt.reshape(T, 2, hw, hw)
    if hw != spatial_h or hw != spatial_w:
        gt = F.interpolate(gt, size=(spatial_h, spatial_w),
                           mode="bilinear", align_corners=False)
    return gt.clamp(0.0, 1.0)


def _unpack_sample(sample):
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


def _save_gif(frames_bgr: list, path: Path, fps: int = 8):
    if not _HAS_IMAGEIO or not frames_bgr:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    iio2.mimwrite(str(path), [np.array(f) for f in frames_bgr], fps=fps, loop=0)


def _prob_to_rgb(prob: torch.Tensor, color: tuple) -> np.ndarray:
    """(H, W) in [0,1] → (H, W, 3) uint8 tinted image."""
    p = prob.clamp(0, 1).cpu().numpy()
    img = np.stack([p * color[0], p * color[1], p * color[2]], axis=-1)
    return (img * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 63 Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Phase63Trainer:
    """
    Stage 1 trainer: EntityField + TransmittanceRenderer only (no diffusion).

    Analysis.md Stage 1 goal: stable V_i(t), A_i(t) from scratch.
    Both entities must survive and be separable before diffusion is introduced.
    """

    def __init__(
        self,
        config,
        pipe,
        backbone_mgr: BackboneManager,
        dataset,
        device: str,
    ):
        self.config = config
        self.pipe = pipe
        self.backbone_mgr = backbone_mgr
        self.dataset = dataset
        self.device = device
        self.tc = config.training
        self.run_name = getattr(config, "run_name", "p63_stage1")

        self.save_dir = Path(f"checkpoints/phase63/{self.run_name}")
        self.debug_dir = Path(f"outputs/phase63/{self.run_name}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.debug_dir / "eval").mkdir(parents=True, exist_ok=True)
        (self.debug_dir / "train_viz").mkdir(parents=True, exist_ok=True)

        # Phase 63 System
        self.system = Phase63System(config).to(device)
        self.evaluator = EntityEvaluator(
            visible_survival_thresh=float(getattr(config.eval, "visible_survival_thresh", 0.02))
        )
        self.gt_builder = VolumeGTBuilder(
            depth_bins=config.depth_bins,
            spatial_h=config.spatial_h,
            spatial_w=config.spatial_w,
            render_resolution=int(getattr(config.data, "volume_gt_render_resolution", 128)),
        )

        # Dataset splits
        raw_ds = dataset.raw_dataset() if hasattr(dataset, "raw_dataset") else dataset
        overlap_scores = compute_dataset_overlap_scores(raw_ds)
        val_frac = getattr(config.data, "val_frac", 0.2)
        self.train_idx, self.val_idx = split_train_val(
            overlap_scores, val_frac=val_frac, min_val=MIN_VAL_SAMPLES)
        self.sample_weights = make_sampling_weights(self.train_idx, overlap_scores)
        print(f"[Data] train={len(self.train_idx)} val={len(self.val_idx)}", flush=True)

        self._setup_optimizer()
        self.history: List[Dict] = []
        self.best_val_score = -1.0
        self.best_epoch = -1

    def _setup_optimizer(self):
        field_params = list(self.system.field.parameters())
        adapter_params = self.backbone_mgr.adapter_params()
        lora_params = self.backbone_mgr.lora_params()
        # Guide encoder not trained in stage1
        for p in list(self.system.guide_encoder.parameters()):
            p.requires_grad_(False)

        for p in field_params + adapter_params + lora_params:
            p.requires_grad_(True)

        lr_field = float(getattr(self.tc, "lr_field", 5e-4))
        lr_adapter = float(getattr(self.tc, "lr_adapter", 1e-4))
        lr_lora = float(getattr(self.tc, "lr_lora", 5e-5))

        self.optimizer = optim.AdamW([
            {"params": field_params,   "lr": lr_field,   "name": "field"},
            {"params": adapter_params, "lr": lr_adapter, "name": "adapters"},
            {"params": lora_params,    "lr": lr_lora,    "name": "lora"},
        ], weight_decay=1e-4)
        for g in self.optimizer.param_groups:
            g["initial_lr"] = g["lr"]

        total_epochs = int(self.tc.epochs)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_epochs), eta_min=1e-6)

    # ── Training step ────────────────────────────────────────────────────────

    def _train_step(self, sample) -> Optional[Dict]:
        frames_np, depth_np, depth_orders, meta, sample_dir, \
            entity_masks, visible_masks = _unpack_sample(sample)

        n_frames = int(getattr(self.tc, "n_frames", 8))
        T = min(int(frames_np.shape[0]), n_frames)
        if T < 1:
            return None

        frames_np = frames_np[:T]
        depth_np = depth_np[:T]
        depth_orders = list(depth_orders)[:T]
        entity_masks = entity_masks[:T]

        gt_vis_np = visible_masks[:T] if visible_masks is not None else None
        V_gt_np = self.gt_builder.build_batch(
            depth_maps=depth_np,
            entity_masks=entity_masks,
            depth_orders=depth_orders,
            visible_masks=gt_vis_np,
            meta=meta,
            sample_dir=sample_dir,
        )

        # GT masks: visible + amodal
        if visible_masks is not None:
            gt_vis_tensor = _build_gt_masks(visible_masks[:T], self.config.spatial_h,
                                            self.config.spatial_w, self.device)
        else:
            # fallback: derive from entity_masks + depth_order
            gt_vis_tensor = _build_gt_masks(entity_masks[:T], self.config.spatial_h,
                                            self.config.spatial_w, self.device)

        gt_amo_tensor = _build_gt_masks(entity_masks[:T], self.config.spatial_h,
                                        self.config.spatial_w, self.device)

        # Encode frames → latents → UNet forward (extract backbone features)
        height = int(getattr(self.tc, "height", 256))
        width  = int(getattr(self.tc, "width", 256))
        t_max  = int(getattr(self.tc, "t_max", 300))

        frames_rgb = [Image.fromarray(frames_np[t]).convert("RGB") for t in range(T)]
        # Resize frames to target resolution and encode
        frames_resized = np.stack([
            np.array(Image.fromarray(frames_np[t]).convert("RGB").resize(
                (width, height), Image.BILINEAR))
            for t in range(T)
        ])  # (T, H, W, 3)
        latents = encode_frames_to_latents(self.pipe, frames_resized, self.device)  # (1, 4, T, H/8, W/8)

        # Encode text + set entity tokens
        # CRITICAL: encode the SAME prompt whose tokens are used for entity routing
        toks_e0, toks_e1, full_prompt = _get_entity_tokens_p63(
            self.pipe, meta, self.device)
        prompt_embeds = _encode_text(self.pipe, full_prompt, self.device)
        self.backbone_mgr.set_entity_tokens(toks_e0, toks_e1)
        self.backbone_mgr.reset_slot_store()

        # Add noise
        noise = torch.randn_like(latents)
        t_idx = torch.randint(0, t_max, (1,), device=self.device).long()
        t_tensor = t_idx.view(1)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, t_tensor)

        # UNet forward (backbone feature extraction, NO guide injection in stage1)
        _ = self.pipe.unet(
            noisy_latents,
            t_tensor,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )

        # Collect backbone features
        ext = self.backbone_mgr.primary
        if ext.last_Fg is None:
            return None
        F_g  = ext.last_Fg.float()    # (B, S, D)
        F_e0 = ext.last_F0.float()
        F_e1 = ext.last_F1.float()

        # ── Losses ──────────────────────────────────────────────────────────
        B = F_g.shape[0]
        # Spatial size of GT masks
        H_gt = gt_vis_tensor.shape[-2]
        W_gt = gt_vis_tensor.shape[-1]

        # Compute color routing maps for direct spatial supervision.
        # These serve double duty: (1) img_hint to EntityField for spatial
        # localization bootstrap, (2) auxiliary color routing loss target.
        # Computed at GT spatial resolution (spatial_h × spatial_w = 32×32).
        color0 = meta.get("color0", [0.85, 0.15, 0.1])
        color1 = meta.get("color1", [0.1, 0.25, 0.85])
        img_t_f = torch.from_numpy(frames_resized[:1].astype(np.float32)).to(self.device) / 255.0
        img_t_f = img_t_f.permute(0, 3, 1, 2)  # (1, 3, H, W)
        img_small = F.interpolate(img_t_f, size=(H_gt, W_gt), mode="bilinear", align_corners=False)
        c0 = torch.tensor(color0, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        c1 = torch.tensor(color1, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        dist0 = (img_small - c0).abs().mean(dim=1, keepdim=True)  # (1, 1, H_gt, W_gt)
        dist1 = (img_small - c1).abs().mean(dim=1, keepdim=True)
        routing0_hint = (1.0 - dist0 * 3.0).clamp(0.0, 1.0)  # (1, 1, H_gt, W_gt)
        routing1_hint = (1.0 - dist1 * 3.0).clamp(0.0, 1.0)
        # Squeezed for loss computation: (1, H_gt, W_gt)
        routing0 = routing0_hint.squeeze(1)
        routing1 = routing1_hint.squeeze(1)

        # Phase 63 forward: EntityField + TransmittanceRenderer
        # img_hint_e0/e1 give EntityField direct color-based spatial features
        field_out, render_out = self.system.forward_field_and_render(
            F_g, F_e0, F_e1,
            img_hint_e0=routing0_hint,
            img_hint_e1=routing1_hint,
        )

        def _maybe_resize(x: torch.Tensor) -> torch.Tensor:
            if x.shape[-2:] != (H_gt, W_gt):
                return F.interpolate(x.unsqueeze(1), size=(H_gt, W_gt),
                                     mode="bilinear", align_corners=False).squeeze(1)
            return x

        vis_e0 = _maybe_resize(render_out.visible_e0)
        vis_e1 = _maybe_resize(render_out.visible_e1)
        amo_e0 = _maybe_resize(render_out.amodal_e0)
        amo_e1 = _maybe_resize(render_out.amodal_e1)

        lam = self.tc
        loss_vis  = loss_visible_dice(vis_e0, vis_e1, gt_vis_tensor[:B, 0], gt_vis_tensor[:B, 1])
        loss_amo  = loss_amodal_dice(amo_e0, amo_e1, gt_amo_tensor[:B, 0], gt_amo_tensor[:B, 1])
        loss_occ  = loss_occlusion_consistency(vis_e0, vis_e1, amo_e0, amo_e1)
        min_surv = float(getattr(self.config.eval, "min_survival", 0.02))
        loss_surv = loss_entity_survival(vis_e0, vis_e1, min_survival=min_surv)

        total = (float(getattr(lam, "lambda_vis",      2.0)) * loss_vis
               + float(getattr(lam, "lambda_amo",      1.0)) * loss_amo
               + float(getattr(lam, "lambda_occ",      0.5)) * loss_occ
               + float(getattr(lam, "lambda_survival", 1.0)) * loss_surv)

        # Color routing auxiliary loss (Stage 1 only)
        # routing0/routing1 were computed above from the image color similarity maps.
        # amodal_e0 should match red-similarity map, amodal_e1 should match blue-similarity map.
        lambda_color = float(getattr(lam, "lambda_color", 0.0))
        if lambda_color > 0.0:
            loss_color = 0.5 * (F.mse_loss(amo_e0, routing0) + F.mse_loss(amo_e1, routing1))
            total = total + lambda_color * loss_color

        # Temporal slot consistency if T > 1
        if T > 1 and float(getattr(lam, "lambda_temp", 0.2)) > 0:
            # Re-derive density list from stored field_out
            ep = field_out.entity_probs   # (B, 2, K, H, W)
            # Slice into per-frame tensors (B==1 in stage1, T frames stacked)
            ep_frames = [ep] * T  # simple approach: reuse same
            loss_temp = loss_temporal_slot_consistency(ep_frames)
            total = total + float(getattr(lam, "lambda_temp", 0.2)) * loss_temp
        else:
            loss_temp = torch.zeros(1, device=self.device)

        # Backward + optimizer
        self.optimizer.zero_grad()
        total.backward()
        grad_clip = float(getattr(lam, "grad_clip", 0.5))
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g["params"]
             if p.requires_grad and p.grad is not None],
            grad_clip,
        )
        self.optimizer.step()

        # Metrics
        with torch.no_grad():
            m = compute_entity_metrics(
                vis_e0, vis_e1, amo_e0, amo_e1,
                gt_vis_tensor[:B], gt_amo_tensor[:B],
            )

        return {
            "loss_total":   float(total.item()),
            "loss_vis":     float(loss_vis.item()),
            "loss_amo":     float(loss_amo.item()),
            "loss_occ":     float(loss_occ.item()),
            "loss_surv":    float(loss_surv.item()),
            "loss_temp":    float(loss_temp.item()) if hasattr(loss_temp, "item") else 0.0,
            **{k: float(v) for k, v in m.items()},
        }

    # ── Validation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict:
        self.backbone_mgr.eval()
        preds, gts = [], []

        for idx in self.val_idx[:8]:  # quick validation
            try:
                sample = self.dataset[idx]
            except Exception:
                continue

            frames_np, depth_np, depth_orders, meta, sample_dir, \
                entity_masks, visible_masks = _unpack_sample(sample)

            n_frames = int(getattr(self.tc, "n_frames", 8))
            T = min(int(frames_np.shape[0]), n_frames)
            if T < 1:
                continue

            gt_vis_tensor = _build_gt_masks(
                (visible_masks[:T] if visible_masks is not None else entity_masks[:T]),
                self.config.spatial_h, self.config.spatial_w, self.device)
            gt_amo_tensor = _build_gt_masks(
                entity_masks[:T], self.config.spatial_h, self.config.spatial_w, self.device)

            toks_e0, toks_e1, full_prompt = _get_entity_tokens_p63(
                self.pipe, meta, self.device)
            prompt_embeds = _encode_text(self.pipe, full_prompt, self.device)
            self.backbone_mgr.set_entity_tokens(toks_e0, toks_e1)
            self.backbone_mgr.reset_slot_store()

            height = int(getattr(self.tc, "height", 256))
            width  = int(getattr(self.tc, "width", 256))
            # Single-frame UNet forward to avoid AnimateDiff T>1 temporal attention bug
            frame_0 = np.array(
                Image.fromarray(frames_np[0]).convert("RGB").resize(
                    (width, height), Image.BILINEAR)
            )[np.newaxis]  # (1, H, W, 3)
            latents = encode_frames_to_latents(self.pipe, frame_0, self.device)
            noise = torch.randn_like(latents)
            t_max_val = int(getattr(self.tc, "t_max", 20))
            t_val = torch.randint(0, max(1, t_max_val), (1,), device=self.device).long()
            noisy = self.pipe.scheduler.add_noise(latents, noise, t_val)
            _ = self.pipe.unet(noisy, t_val,
                               encoder_hidden_states=prompt_embeds, return_dict=False)

            ext = self.backbone_mgr.primary
            if ext.last_Fg is None:
                continue
            F_g  = ext.last_Fg.float()
            F_e0 = ext.last_F0.float()
            F_e1 = ext.last_F1.float()

            # Compute color routing hints for validation (same as training)
            H_gt_v = gt_vis_tensor.shape[-2]
            W_gt_v = gt_vis_tensor.shape[-1]
            color0_v = meta.get("color0", [0.85, 0.15, 0.1])
            color1_v = meta.get("color1", [0.1, 0.25, 0.85])
            frame0_t = torch.from_numpy(frame_0.astype(np.float32)).to(self.device) / 255.0
            frame0_t = frame0_t.permute(0, 3, 1, 2)  # (1, 3, H, W)
            img_small_v = F.interpolate(frame0_t, size=(H_gt_v, W_gt_v),
                                        mode="bilinear", align_corners=False)
            c0_v = torch.tensor(color0_v, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
            c1_v = torch.tensor(color1_v, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
            hint0 = (1.0 - (img_small_v - c0_v).abs().mean(1, keepdim=True) * 3.0).clamp(0.0, 1.0)
            hint1 = (1.0 - (img_small_v - c1_v).abs().mean(1, keepdim=True) * 3.0).clamp(0.0, 1.0)

            field_out, render_out = self.system.forward_field_and_render(
                F_g, F_e0, F_e1, img_hint_e0=hint0, img_hint_e1=hint1)

            # Squeeze batch dim (B=1) so evaluator receives (H, W) tensors
            preds.append({
                "visible_e0": render_out.visible_e0[0],   # (H, W)
                "visible_e1": render_out.visible_e1[0],
                "amodal_e0":  render_out.amodal_e0[0],
                "amodal_e1":  render_out.amodal_e1[0],
            })
            gts.append({
                "visible_e0": gt_vis_tensor[0, 0],        # (H, W)
                "visible_e1": gt_vis_tensor[0, 1],
                "amodal_e0":  gt_amo_tensor[0, 0],
                "amodal_e1":  gt_amo_tensor[0, 1],
            })

            # Save debug viz for first sample
            if len(preds) == 1:
                self._save_debug_viz(epoch, render_out, gt_vis_tensor)

        self.backbone_mgr.train()
        if not preds:
            return {"visible_iou_min": 0.0, "visible_survival_min": 0.0}
        return self.evaluator.evaluate_sequence(preds, gts)

    def _save_debug_viz(self, epoch: int, render_out, gt_vis: torch.Tensor):
        """Save side-by-side debug images for visible/amodal maps."""
        try:
            H, W = render_out.visible_e0.shape[-2:]
            v0 = _prob_to_rgb(render_out.visible_e0[0], (1, 0.3, 0.3))   # red tint
            v1 = _prob_to_rgb(render_out.visible_e1[0], (0.3, 0.3, 1))   # blue tint
            a0 = _prob_to_rgb(render_out.amodal_e0[0],  (1, 0.6, 0.6))
            a1 = _prob_to_rgb(render_out.amodal_e1[0],  (0.6, 0.6, 1))
            gt0 = _prob_to_rgb(gt_vis[0, 0], (0.8, 0.8, 0))              # yellow
            gt1 = _prob_to_rgb(gt_vis[0, 1], (0, 0.8, 0.8))

            row1 = np.concatenate([v0, v1, a0, a1], axis=1)
            row2 = np.concatenate([gt0, gt1,
                                   np.zeros_like(a0), np.zeros_like(a1)], axis=1)
            grid = np.concatenate([row1, row2], axis=0)
            img = Image.fromarray(grid)
            img.save(self.debug_dir / "train_viz" / f"ep{epoch:04d}_fields.png")
        except Exception as e:
            print(f"[viz] save failed: {e}", flush=True)

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        state = {
            "epoch": epoch,
            "field_state": self.system.field.state_dict(),
            "guide_encoder_state": self.system.guide_encoder.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": {k: v for k, v in vars(self.config).items()
                       if not callable(v)},
        }
        ckpt_path = self.save_dir / f"ep{epoch:04d}.pt"
        torch.save(state, ckpt_path)
        if is_best:
            torch.save(state, self.save_dir / "best.pt")
            print(f"  ✓ Best checkpoint saved (epoch {epoch})", flush=True)

    def _load_checkpoint(self, ckpt_path: str):
        state = torch.load(ckpt_path, map_location=self.device)
        self.system.field.load_state_dict(state["field_state"])
        if "guide_encoder_state" in state:
            self.system.guide_encoder.load_state_dict(state["guide_encoder_state"])
        print(f"[Resume] Loaded from {ckpt_path} (epoch {state['epoch']})", flush=True)
        return int(state.get("epoch", 0))

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, resume_from: Optional[str] = None):
        _seed_all(int(getattr(self.tc, "seed", 42)))
        start_epoch = 0
        if resume_from:
            start_epoch = self._load_checkpoint(resume_from) + 1

        epochs = int(self.tc.epochs)
        steps_per_epoch = int(getattr(self.tc, "steps_per_epoch", 30))
        eval_every = int(getattr(self.tc, "eval_every", 5))

        print(f"\n{'='*60}", flush=True)
        print(f"[Phase63] Stage 1 — EntityField + TransmittanceRenderer", flush=True)
        print(f"  epochs={epochs}  steps/ep={steps_per_epoch}", flush=True)
        print(f"  save_dir={self.save_dir}", flush=True)
        print(f"{'='*60}\n", flush=True)

        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            self.backbone_mgr.train()
            self.system.field.train()

            # Sample indices for this epoch (weighted by collision overlap)
            if self.sample_weights is not None:
                # sample_weights is positional array aligned to train_idx
                ep_idx = random.choices(
                    self.train_idx,
                    weights=self.sample_weights.tolist(),
                    k=steps_per_epoch,
                )
            else:
                ep_idx = [random.choice(self.train_idx) for _ in range(steps_per_epoch)]

            step_metrics: List[Dict] = []
            for step, idx in enumerate(ep_idx):
                try:
                    sample = self.dataset[idx]
                    m = self._train_step(sample)
                    if m is not None:
                        step_metrics.append(m)
                except Exception as exc:
                    import traceback
                    print(f"  [step {step}] error: {exc}", flush=True)
                    if step == 0:
                        traceback.print_exc()
                    continue

            self.lr_scheduler.step()

            if not step_metrics:
                print(f"  [epoch {epoch}] no valid steps", flush=True)
                continue

            # Aggregate epoch metrics
            ep_m = {k: float(np.mean([s[k] for s in step_metrics if k in s]))
                    for k in step_metrics[0]}
            ep_m["epoch"] = epoch
            ep_m["time_s"] = time.time() - t0

            # Evaluation
            if epoch % eval_every == 0 or epoch == epochs - 1:
                self.system.field.eval()
                val_m = self._validate(epoch)
                self.system.field.train()
                ep_m.update({f"val_{k}": v for k, v in val_m.items()})

                # Use amodal_iou_min as primary score for Stage 1.
                # Visible IoU can be 0 due to occlusion even when the field is correct.
                # Amodal IoU measures whether the entity body is correctly predicted.
                val_score = float(val_m.get("amodal_iou_min", 0.0))
                is_best = val_score > self.best_val_score
                if is_best:
                    self.best_val_score = val_score
                    self.best_epoch = epoch
                self._save_checkpoint(epoch, ep_m, is_best=is_best)

                print(
                    f"[ep {epoch:3d}] "
                    f"loss={ep_m['loss_total']:.4f} "
                    f"vis={ep_m.get('loss_vis', 0):.4f} "
                    f"surv={ep_m.get('loss_surv', 0):.4f} | "
                    f"val_vis_iou={val_m.get('visible_iou_min', 0):.4f} "
                    f"val_amo_iou={val_m.get('amodal_iou_min', 0):.4f} "
                    f"val_surv={val_m.get('visible_survival_min', 0):.4f} "
                    f"val_bal={val_m.get('entity_balance', 0):.3f} "
                    f"({ep_m['time_s']:.1f}s)",
                    flush=True,
                )
            else:
                print(
                    f"[ep {epoch:3d}] "
                    f"loss={ep_m['loss_total']:.4f} "
                    f"vis={ep_m.get('loss_vis', 0):.4f} "
                    f"amo={ep_m.get('loss_amo', 0):.4f} "
                    f"vis_iou={ep_m.get('visible_iou_min', 0):.4f} "
                    f"amo_iou={ep_m.get('amodal_iou_min', 0):.4f} "
                    f"({ep_m['time_s']:.1f}s)",
                    flush=True,
                )

            self.history.append(ep_m)

        # Save history
        hist_path = self.debug_dir / "history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n[Done] best epoch={self.best_epoch} amodal_iou_min={self.best_val_score:.4f}", flush=True)
        print(f"  history → {hist_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase63/stage1.yaml")
    parser.add_argument("--ckpt",   default=None, help="Resume from checkpoint")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    config = load_config(args.config)

    _seed_all(int(getattr(getattr(config, "training", config), "seed", 42)))

    # ── Pipeline ──────────────────────────────────────────────────────────────
    from scripts.run_animatediff import load_pipeline
    print("[Phase63] Loading AnimateDiff pipeline...", flush=True)
    pipe = load_pipeline(device=device)
    pipe.scheduler.set_timesteps(50, device=device)
    pipe.unet.enable_gradient_checkpointing()

    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    # ── Backbone extractors ────────────────────────────────────────────────────
    print("[Phase63] Injecting backbone feature extractors...", flush=True)
    adapter_rank = int(getattr(config.model, "adapter_rank", 64))
    lora_rank    = int(getattr(config.model, "lora_rank",    4))
    extractors, _ = inject_backbone_extractors(
        pipe,
        adapter_rank=adapter_rank,
        lora_rank=lora_rank,
        inject_keys=DEFAULT_INJECT_KEYS,
    )
    for ext in extractors:
        ext.to(device)
    backbone_mgr = BackboneManager(
        extractors, DEFAULT_INJECT_KEYS, primary_idx=2)  # up_blocks.3: 32×32, feat_dim=320

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_root = getattr(config.data, "data_root", "toy/data_objaverse")
    n_frames  = int(getattr(config.training, "n_frames", 8))
    print(f"[Phase63] Loading dataset from {data_root}...", flush=True)
    dataset = Phase62DatasetAdapter(data_root, n_frames=n_frames)
    print(f"[Phase63] Dataset size: {len(dataset)}", flush=True)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Phase63Trainer(config, pipe, backbone_mgr, dataset, device)
    trainer.train(resume_from=args.ckpt)


if __name__ == "__main__":
    main()
