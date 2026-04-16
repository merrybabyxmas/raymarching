"""
training/phase64/stage3_train_adapter_backbone.py
===================================================
Stage 3: Train AnimateDiffAdapter with frozen/lightly fine-tuned scene prior.

Uses SDEdit-style training:
  - Encode real frame + add noise + denoise with guide injection
  - Loss: diffusion MSE + reconstruction against decoder output + survival

Prerequisites:
  - Stage 1 checkpoint (scene prior)
  - Stage 2 checkpoint (decoder)
  - AnimateDiff pipeline (loaded by caller or from config)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.phase64.phase64_dataset import Phase64Dataset, Phase64Sample
from training.phase64.evaluator_phase64 import Phase64Evaluator
from training.phase64.stage2_train_decoder import StructuredDecoder


# --------------------------------------------------------------------------- #
#  Stage3Trainer
# --------------------------------------------------------------------------- #

class Stage3Trainer:
    """
    Trains AnimateDiffAdapter with frozen/lightly fine-tuned scene prior.

    Training flow per step:
      1. Run frozen ScenePriorModule + EntityRenderer → SceneOutputs
      2. Run SceneGuideEncoder(SceneOutputs) → guide features
      3. Build AnimateDiffAdapter guides
      4. Encode real frame → latent z0
      5. Add noise: z_t = sqrt(alpha_t) * z0 + sqrt(1-alpha_t) * eps
      6. UNet denoise with guide injection: eps_pred = unet(z_t, t, guide)
      7. Diffusion MSE loss: mse(eps_pred, eps)
      8. (Optional) decode scene outputs → pred_rgb; add reconstruction loss

    Parameters
    ----------
    config            : config object
    dataset           : Phase64Dataset
    scene_prior_ckpt  : path to Stage 1 checkpoint
    decoder_ckpt      : path to Stage 2 checkpoint
    pipe              : AnimateDiff pipeline (loaded externally)
    device            : torch.device
    """

    def __init__(
        self,
        config,
        dataset: Phase64Dataset,
        splits: Optional[dict] = None,
        stage1_ckpt: str = "",
        stage2_ckpt: str = "",
        device: str = "cuda",
        # backward-compat aliases
        scene_prior_ckpt: str = "",
        decoder_ckpt: str = "",
        pipe=None,
    ) -> None:
        self.config  = config
        self.dataset = dataset
        self.device  = torch.device(device) if isinstance(device, str) else device
        self._splits = splits
        self.pipe    = pipe  # optional; needed only for SDEdit diffusion loss

        # Resolve checkpoint paths (new names take priority)
        _stage1_ckpt = stage1_ckpt or scene_prior_ckpt
        _stage2_ckpt = stage2_ckpt or decoder_ckpt

        from scene_prior import ScenePriorModule, EntityRenderer
        from adapters import SceneGuideEncoder, AnimateDiffAdapter

        model_cfg = config.model
        self.scene_prior = ScenePriorModule(
            depth_bins=model_cfg.depth_bins,
            hidden_dim=getattr(model_cfg, "hidden_dim", 64),
            id_dim=model_cfg.id_dim,
            pose_dim=model_cfg.pose_dim,
            spatial_h=model_cfg.spatial_h,
            spatial_w=model_cfg.spatial_w,
            slot_dim=getattr(model_cfg, "slot_dim", 64),
        ).to(self.device)

        self.renderer = EntityRenderer(depth_bins=model_cfg.depth_bins).to(self.device)

        # Load Stage 1 checkpoint
        ckpt1 = torch.load(_stage1_ckpt, weights_only=False, map_location=self.device)
        _sp_key = next(
            (k for k in ("scene_prior", "scene_prior_state", "field_state",
                         "model_state") if k in ckpt1),
            None,
        )
        if _sp_key:
            self.scene_prior.load_state_dict(ckpt1[_sp_key])
        else:
            self.scene_prior.load_state_dict(ckpt1)
        if "renderer" in ckpt1:
            self.renderer.load_state_dict(ckpt1["renderer"])
        print(f"[stage3] Loaded scene prior from {_stage1_ckpt}")

        # Load Stage 2 decoder (optional — may not exist yet during testing)
        self.decoder = StructuredDecoder(
            in_channels=8,
            hidden=getattr(model_cfg, "hidden_dim", 64),
        ).to(self.device)
        if _stage2_ckpt and Path(_stage2_ckpt).exists():
            ckpt2 = torch.load(_stage2_ckpt, weights_only=False, map_location=self.device)
            dec_key = next(
                (k for k in ("decoder", "decoder_state") if k in ckpt2),
                None,
            )
            if dec_key:
                self.decoder.load_state_dict(ckpt2[dec_key])
            print(f"[stage3] Loaded decoder from {_stage2_ckpt}")
        else:
            print(f"[stage3] Stage 2 ckpt not provided / not found — decoder init from scratch")

        # Guide encoder + adapter (trainable)
        self.guide_encoder = SceneGuideEncoder(
            in_ch=8,
            hidden=getattr(model_cfg, "hidden_dim", 64),
        ).to(self.device)

        self.adapter = AnimateDiffAdapter(
            guide_channels=getattr(model_cfg, "hidden_dim", 64),
            guide_max_ratio=float(getattr(model_cfg, "guide_max_ratio", 0.15)),
            inject_blocks=list(getattr(model_cfg, "inject_blocks", ["up1", "up2", "up3"])),
        ).to(self.device)

        # Freeze scene prior + renderer; decoder frozen too
        for p in self.scene_prior.parameters():
            p.requires_grad_(False)
        for p in self.renderer.parameters():
            p.requires_grad_(False)
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.scene_prior.eval()
        self.renderer.eval()
        self.decoder.eval()

        # Lightly fine-tune: allow small LR on encoder memory if configured
        train_cfg = config.training
        trainable_prior_lr = getattr(train_cfg, "lr_prior_finetune", 0.0)
        param_groups = [
            {"params": self.guide_encoder.parameters(), "lr": getattr(train_cfg, "lr_guide_enc", 3e-4)},
            {"params": self.adapter.parameters(),       "lr": getattr(train_cfg, "lr_adapter",   1e-4)},
        ]
        if trainable_prior_lr > 0:
            for p in self.scene_prior.memory_params():
                p.requires_grad_(True)
            param_groups.append({
                "params": list(self.scene_prior.memory_params()),
                "lr": trainable_prior_lr,
            })

        self.optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

        self.evaluator = Phase64Evaluator()
        run_name = getattr(config, "run_name", "p64_stage3_animatediff")
        self.out_dir = Path(getattr(config, "out_dir",
                                    f"checkpoints/phase64/{run_name}"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._best_val: float = 0.0
        self._step: int = 0
        self._train_log: list = []

    # ---------------------------------------------------------------------- #

    def load_checkpoint(self, path: str) -> None:
        """Load guide_encoder + adapter + (optionally) optimizer state."""
        ckpt = torch.load(path, weights_only=False, map_location=self.device)
        for key_pair in [("guide_encoder", "guide_encoder"),
                         ("adapter", "adapter"),
                         ("guide_encoder_state", "guide_encoder"),
                         ("adapter_state", "adapter")]:
            src, tgt = key_pair
            if src in ckpt and hasattr(self, tgt):
                getattr(self, tgt).load_state_dict(ckpt[src])
        for opt_key in ("optimizer", "optimizer_state"):
            if opt_key in ckpt:
                self.optimizer.load_state_dict(ckpt[opt_key])
                break
        self._step = ckpt.get("step", 0)
        print(f"[stage3] Loaded checkpoint from {path}  (step={self._step})")

    # ---------------------------------------------------------------------- #

    def _encode_frames_to_latents(
        self,
        frames: np.ndarray,   # (T, H, W, 3) uint8
    ) -> torch.Tensor:
        """Encode frames to VAE latents using the pipeline's VAE.

        Returns: (1, 4, h_lat, w_lat) float16 latents.
        """
        from PIL import Image

        # Use mean frame
        mean_frame = frames.mean(axis=0).astype(np.uint8)
        img = Image.fromarray(mean_frame).resize((256, 256))
        img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(self.device) * 2.0 - 1.0

        with torch.no_grad():
            latents = self.pipe.vae.encode(img_t.half()).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor
        return latents  # (1, 4, h, w)

    def _add_noise(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
    ):
        """SDEdit: sample noise and add it to latents at timestep t."""
        noise = torch.randn_like(latents)
        scheduler = self.pipe.scheduler
        noisy = scheduler.add_noise(latents, noise, t)
        return noisy, noise

    def _train_step(self, sample: Phase64Sample) -> dict:
        """Single training step for Stage 3."""
        self.guide_encoder.train()
        self.adapter.train()
        self.optimizer.zero_grad()

        frames = sample.frames
        routing_e0 = sample.routing_e0
        routing_e1 = sample.routing_e1
        meta = sample.meta

        # ---- Scene prior forward (frozen) ---------------------------------- #
        frame_t = torch.from_numpy(frames.astype(np.float32) / 255.0).to(self.device)
        img_chw = frame_t.mean(dim=0).unsqueeze(0).permute(0, 3, 1, 2)

        r0 = torch.from_numpy(routing_e0.mean(axis=0)).float().to(self.device).unsqueeze(0).unsqueeze(0)
        r1 = torch.from_numpy(routing_e1.mean(axis=0)).float().to(self.device).unsqueeze(0).unsqueeze(0)
        routing_hints = torch.cat([r0, r1], dim=1)

        entity_names = [
            str(meta.get("keyword0", "entity0")),
            str(meta.get("keyword1", "entity1")),
        ]

        with torch.no_grad():
            density_fields = self.scene_prior(
                img=img_chw, entity_names=entity_names, routing_hints=routing_hints,
            )
            scene_out = self.renderer(density_fields)

        scene_tensor = scene_out.to_canonical_tensor()  # (1, 8, H, W)

        # ---- Guide encoding ------------------------------------------------ #
        guide_features = self.guide_encoder(scene_tensor)  # (1, hidden, H, W)
        guide_dict = self.adapter.build_guides(guide_features)

        # ---- SDEdit: encode + noise + denoise ------------------------------ #
        latents = self._encode_frames_to_latents(frames)

        # Sample random timestep
        T_max = self.pipe.scheduler.config.num_train_timesteps
        t = torch.randint(0, T_max, (1,), device=self.device).long()
        noisy_latents, noise = self._add_noise(latents, t)

        # Text embedding
        prompt = str(meta.get("prompt", "two objects"))
        text_enc = self.pipe.text_encoder(
            self.pipe.tokenizer(
                prompt, return_tensors="pt", padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length, truncation=True,
            ).input_ids.to(self.device)
        ).last_hidden_state.half()

        # Register adapter hooks + UNet forward
        self.adapter.register_hooks(self.pipe.unet)
        try:
            noise_pred = self.pipe.unet(
                noisy_latents.half(), t, encoder_hidden_states=text_enc,
            ).sample
        finally:
            self.adapter.remove_hooks()

        # Diffusion MSE loss
        loss_diff = F.mse_loss(noise_pred.float(), noise.float())

        # Reconstruction auxiliary: decoder output should match GT
        with torch.no_grad():
            pred_rgb = self.decoder(scene_tensor)   # (1, 3, H, W)

        model_cfg = self.config.model
        H_sp, W_sp = model_cfg.spatial_h, model_cfg.spatial_w
        gt_rgb = F.interpolate(
            img_chw.float(), size=(H_sp, W_sp), mode="bilinear", align_corners=False
        )
        loss_rec = F.l1_loss(pred_rgb, gt_rgb)

        # Survival: both entities must be present in scene outputs
        surv_thresh = 0.01
        loss_surv = (
            F.relu(surv_thresh - scene_out.visible_e0.mean()) +
            F.relu(surv_thresh - scene_out.visible_e1.mean())
        )

        lambda_diff = getattr(self.config.training, "lambda_diff", 1.0)
        lambda_rec  = getattr(self.config.training, "lambda_rec",  0.2)
        lambda_surv = getattr(self.config.training, "lambda_surv", 1.0)

        total_loss = lambda_diff * loss_diff + lambda_rec * loss_rec + lambda_surv * loss_surv
        total_loss.backward()

        grad_clip = getattr(self.config.training, "grad_clip", 1.0)
        trainable_params = (
            list(self.guide_encoder.parameters()) +
            list(self.adapter.parameters())
        )
        nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        self.optimizer.step()

        return {
            "total":     float(total_loss.item()),
            "diff":      float(loss_diff.item()),
            "rec":       float(loss_rec.item()),
            "surv":      float(loss_surv.item()),
        }

    # ---------------------------------------------------------------------- #

    def train(self, resume_from: Optional[str] = None) -> None:
        """Main Stage 3 training loop."""
        if resume_from is not None:
            ckpt = torch.load(resume_from, weights_only=False, map_location=self.device)
            self.guide_encoder.load_state_dict(ckpt["guide_encoder"])
            self.adapter.load_state_dict(ckpt["adapter"])
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            self._step = ckpt.get("step", 0)
            print(f"[stage3] Resumed from {resume_from}  (step={self._step})")

        train_cfg = self.config.training
        epochs          = train_cfg.epochs
        steps_per_epoch = train_cfg.steps_per_epoch
        eval_every      = train_cfg.eval_every

        split_info = (self._splits if self._splits is not None
                      else self.dataset.get_split_indices()
                      if hasattr(self.dataset, "get_split_indices") else {})
        train_indices = split_info.get("train", list(range(len(self.dataset))))

        print(f"[stage3] Training adapter  epochs={epochs}  "
              f"steps/epoch={steps_per_epoch}  n_train={len(train_indices)}")

        rng = np.random.default_rng(seed=44)

        for epoch in range(1, epochs + 1):
            epoch_metrics: list = []
            idxs = rng.choice(train_indices, size=min(steps_per_epoch, len(train_indices)),
                              replace=False).tolist()

            for idx in idxs:
                sample = self.dataset[idx]
                step_m = self._train_step(sample)
                epoch_metrics.append(step_m)
                self._step += 1

            avg_total = float(np.mean([m["total"] for m in epoch_metrics]))
            avg_diff  = float(np.mean([m["diff"]  for m in epoch_metrics]))
            print(f"[epoch {epoch}] total={avg_total:.4f}  diff={avg_diff:.4f}")

            self._train_log.append({
                "epoch": epoch, "step": self._step,
                "loss": avg_total, "diff_loss": avg_diff,
            })

            if epoch % eval_every == 0 or epoch == epochs:
                ckpt_path = self.out_dir / f"stage3_epoch{epoch:04d}.pt"
                torch.save({
                    "epoch": epoch, "step": self._step,
                    "guide_encoder": self.guide_encoder.state_dict(),
                    "adapter": self.adapter.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }, ckpt_path)
                print(f"  [stage3] Checkpoint → {ckpt_path}")

            log_path = self.out_dir / "train_log.json"
            with open(log_path, "w") as f:
                json.dump(self._train_log, f, indent=2, default=str)

        print(f"[stage3] Complete.  Logs → {self.out_dir}")
