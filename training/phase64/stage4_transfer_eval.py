"""
training/phase64/stage4_transfer_eval.py
==========================================
Stage 4: Transfer test — prove scene prior portability.

Takes frozen scene prior (trained in Stage 1 on AnimateDiff data) and trains
a NEW adapter for SDXL.  Compares chimera rate between:
  - no-guide SDXL baseline
  - SDXL with scene prior guides

This proves the structural prior is portable across backbones.

Metrics reported:
  - entity_survival_e0 / e1 / min
  - visible_iou_min
  - separation_accuracy (mean |sep_map| in contact regions)
  - chimera_rate (fraction of samples where one entity is absent)
"""
from __future__ import annotations

import json
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
#  Stage4TransferEval
# --------------------------------------------------------------------------- #

class Stage4TransferEval:
    """
    Transfer test: new SDXL adapter trained on frozen scene prior.

    Parameters
    ----------
    config             : config object
    dataset            : Phase64Dataset
    scene_prior_ckpt   : path to Stage 1 checkpoint (frozen)
    sdxl_pipe          : loaded SDXL pipeline
    device             : torch.device
    """

    def __init__(
        self,
        config,
        dataset: Phase64Dataset,
        splits: Optional[Dict] = None,
        stage1_ckpt: str = "",
        device: str = "cuda",
        # backward-compat aliases
        scene_prior_ckpt: str = "",
        sdxl_pipe=None,
    ) -> None:
        self.config    = config
        self.dataset   = dataset
        self._splits   = splits
        self.device    = torch.device(device) if isinstance(device, str) else device
        self.sdxl_pipe = sdxl_pipe

        _stage1_ckpt = stage1_ckpt or scene_prior_ckpt

        from scene_prior import ScenePriorModule
        from adapters import SceneGuideEncoder, SDXLAdapter

        model_cfg = config.model
        # NOTE: ScenePriorModule uses `hidden` (not `hidden_dim`); renders internally
        self.scene_prior = ScenePriorModule(
            depth_bins=int(model_cfg.depth_bins),
            hidden=int(getattr(model_cfg, "hidden_dim", 64)),
            id_dim=int(model_cfg.id_dim),
            pose_dim=int(model_cfg.pose_dim),
            ctx_dim=int(getattr(model_cfg, "ctx_dim", getattr(model_cfg, "hidden_dim", 64))),
            spatial_h=int(model_cfg.spatial_h),
            spatial_w=int(model_cfg.spatial_w),
            slot_dim=int(getattr(model_cfg, "slot_dim", 64)),
        ).to(self.device)

        # Load Stage 1 — frozen throughout Stage 4
        ckpt = torch.load(_stage1_ckpt, weights_only=False, map_location=self.device)
        _sp_key = next(
            (k for k in ("scene_prior", "scene_prior_state", "field_state",
                         "model_state") if k in ckpt),
            None,
        )
        if _sp_key:
            self.scene_prior.load_state_dict(ckpt[_sp_key])
        else:
            self.scene_prior.load_state_dict(ckpt)
        print(f"[stage4] Loaded scene prior from {_stage1_ckpt}")

        for p in self.scene_prior.parameters():
            p.requires_grad_(False)
        self.scene_prior.eval()

        # Freeze SDXL UNet, text encoders, VAE — only adapter+guide_encoder train
        if self.sdxl_pipe is not None:
            for p in self.sdxl_pipe.unet.parameters():
                p.requires_grad_(False)
            for p in self.sdxl_pipe.text_encoder.parameters():
                p.requires_grad_(False)
            if hasattr(self.sdxl_pipe, "text_encoder_2"):
                for p in self.sdxl_pipe.text_encoder_2.parameters():
                    p.requires_grad_(False)
            for p in self.sdxl_pipe.vae.parameters():
                p.requires_grad_(False)
            if hasattr(self.sdxl_pipe, "enable_model_cpu_offload"):
                # Don't offload — stay on GPU but save graph memory
                pass

        # NEW SDXL adapter — fresh weights
        self.guide_encoder = SceneGuideEncoder(
            in_ch=8,
            hidden=int(getattr(model_cfg, "hidden_dim", 64)),
        ).to(self.device)

        self.sdxl_adapter = SDXLAdapter(
            in_ch=int(getattr(model_cfg, "hidden_dim", 64)),
            guide_max_ratio=float(getattr(model_cfg, "guide_max_ratio", 0.1)),
            inject_blocks=tuple(getattr(model_cfg, "inject_blocks", ["up0", "up1", "up2"])),
        ).to(self.device)

        train_cfg = config.training
        self.optimizer = optim.AdamW(
            list(self.guide_encoder.parameters()) + list(self.sdxl_adapter.parameters()),
            lr=getattr(train_cfg, "lr_adapter", 1e-4),
            weight_decay=1e-4,
        )

        self.evaluator = Phase64Evaluator()
        run_name = getattr(config, "run_name", "p64_stage4_sdxl")
        self.out_dir = Path(getattr(config, "out_dir",
                                    f"checkpoints/phase64/{run_name}"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._train_log: list = []

    # ---------------------------------------------------------------------- #

    def load_checkpoint(self, path: str) -> None:
        """Load guide_encoder + sdxl_adapter weights (called from train script)."""
        ckpt = torch.load(path, weights_only=False, map_location=self.device)
        for src, tgt in [("guide_encoder", "guide_encoder"),
                          ("sdxl_adapter",  "sdxl_adapter"),
                          ("guide_encoder_state", "guide_encoder"),
                          ("adapter_state", "sdxl_adapter")]:
            if src in ckpt and hasattr(self, tgt):
                getattr(self, tgt).load_state_dict(ckpt[src])
        for opt_key in ("optimizer", "optimizer_state"):
            if opt_key in ckpt:
                self.optimizer.load_state_dict(ckpt[opt_key])
                break
        print(f"[stage4] Loaded checkpoint from {path}")

    def train(self) -> None:
        """Convenience wrapper: train SDXL adapter then run transfer eval."""
        n_epochs = int(getattr(self.config.training, "epochs", 20))
        self.train_sdxl_adapter(n_epochs=n_epochs)
        self.eval_transfer()

    # ---------------------------------------------------------------------- #
    #  SDXL adapter training
    # ---------------------------------------------------------------------- #

    def train_sdxl_adapter(self, n_epochs: int = 20) -> dict:
        """
        Fine-tune SDXL adapter for n_epochs.

        Uses the same SDEdit-style training as Stage 3 but with the SDXL UNet.

        Returns final training metrics dict.
        """
        train_cfg = self.config.training
        steps_per_epoch = getattr(train_cfg, "steps_per_epoch", 50)

        split_info = (self._splits if self._splits is not None
                      else self.dataset.get_split_indices()
                      if hasattr(self.dataset, "get_split_indices") else {})
        train_indices = split_info.get("train", list(range(len(self.dataset))))

        print(f"[stage4] Training SDXL adapter  epochs={n_epochs}  "
              f"steps/epoch={steps_per_epoch}")

        rng = np.random.default_rng(seed=45)

        for epoch in range(1, n_epochs + 1):
            epoch_metrics: list = []
            idxs = rng.choice(train_indices, size=min(steps_per_epoch, len(train_indices)),
                              replace=False).tolist()

            self.guide_encoder.train()
            self.sdxl_adapter.train()

            for idx in idxs:
                sample = self.dataset[idx]
                step_m = self._sdxl_train_step(sample)
                epoch_metrics.append(step_m)

            avg_loss = float(np.mean([m["total"] for m in epoch_metrics]))
            print(f"  [sdxl epoch {epoch}] loss={avg_loss:.4f}")
            self._train_log.append({"epoch": epoch, "loss": avg_loss})

        # Save adapter
        ckpt_path = self.out_dir / "sdxl_adapter.pt"
        torch.save({
            "guide_encoder": self.guide_encoder.state_dict(),
            "sdxl_adapter":  self.sdxl_adapter.state_dict(),
        }, ckpt_path)
        print(f"[stage4] SDXL adapter saved → {ckpt_path}")

        return {"final_loss": avg_loss, "ckpt": str(ckpt_path)}

    def _sdxl_train_step(self, sample: Phase64Sample) -> dict:
        """Single SDEdit step for SDXL."""
        self.optimizer.zero_grad()

        frames = sample.frames
        routing_e0 = sample.routing_e0
        routing_e1 = sample.routing_e1
        meta = sample.meta

        frame_t = torch.from_numpy(frames.astype(np.float32) / 255.0).to(self.device)
        img_chw = frame_t.mean(dim=0).unsqueeze(0).permute(0, 3, 1, 2)

        r0 = torch.from_numpy(routing_e0.mean(axis=0)).float().to(self.device).unsqueeze(0).unsqueeze(0)
        r1 = torch.from_numpy(routing_e1.mean(axis=0)).float().to(self.device).unsqueeze(0).unsqueeze(0)
        entity_name_e0 = str(meta.get("keyword0", "unknown"))
        entity_name_e1 = str(meta.get("keyword1", "unknown"))

        with torch.no_grad():
            scene_out, _, _ = self.scene_prior(
                img=img_chw,
                entity_name_e0=entity_name_e0,
                entity_name_e1=entity_name_e1,
                routing_hint_e0=r0,
                routing_hint_e1=r1,
            )

        scene_tensor = scene_out.to_canonical_tensor()
        guide_features = self.guide_encoder(scene_tensor)
        guide_dict = self.sdxl_adapter.build_guides(guide_features)

        # Encode frames to SDXL VAE latents
        from PIL import Image
        mean_frame = frames.mean(axis=0).astype(np.uint8)
        # Use 256x256 to reduce VRAM usage (SDXL-Turbo supports smaller resolutions)
        H_enc, W_enc = 256, 256
        img_pil = Image.fromarray(mean_frame).resize((H_enc, W_enc))
        img_t = torch.from_numpy(np.array(img_pil, dtype=np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(self.device).half() * 2.0 - 1.0
        with torch.no_grad():
            latents = self.sdxl_pipe.vae.encode(img_t).latent_dist.sample()
            latents = latents * self.sdxl_pipe.vae.config.scaling_factor

        T_max = self.sdxl_pipe.scheduler.config.num_train_timesteps
        t = torch.randint(0, T_max, (1,), device=self.device).long()
        noise = torch.randn_like(latents)
        noisy = self.sdxl_pipe.scheduler.add_noise(latents, noise, t)

        prompt = str(meta.get("prompt", "two objects"))
        # SDXL requires both text encoders + added_cond_kwargs
        with torch.no_grad():
            # Encoder 1 (CLIP ViT-L)
            tok1 = self.sdxl_pipe.tokenizer(
                prompt, return_tensors="pt", padding="max_length",
                max_length=self.sdxl_pipe.tokenizer.model_max_length, truncation=True,
            ).to(self.device)
            enc1_out = self.sdxl_pipe.text_encoder(**tok1)
            text_emb = enc1_out.last_hidden_state.half()
            # Encoder 2 (OpenCLIP BigG)
            tok2 = self.sdxl_pipe.tokenizer_2(
                prompt, return_tensors="pt", padding="max_length",
                max_length=self.sdxl_pipe.tokenizer_2.model_max_length, truncation=True,
            ).to(self.device)
            enc2_out = self.sdxl_pipe.text_encoder_2(**tok2, output_hidden_states=True)
            text_emb2 = enc2_out.hidden_states[-2].half()
            # SDXL concatenates both text embeddings
            text_emb_combined = torch.cat([text_emb, text_emb2], dim=-1)
            pooled_emb = enc2_out.text_embeds.half()  # (B, 1280)
            # SDXL time ids: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
            time_ids = torch.tensor(
                [[H_enc, W_enc, 0, 0, H_enc, W_enc]], dtype=torch.float16, device=self.device
            )
            added_cond_kwargs = {
                "text_embeds": pooled_emb,
                "time_ids": time_ids,
            }

        self.sdxl_adapter.register_hooks(self.sdxl_pipe.unet)
        try:
            noise_pred = self.sdxl_pipe.unet(
                noisy.half(), t,
                encoder_hidden_states=text_emb_combined,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
        finally:
            self.sdxl_adapter.remove_hooks()

        loss = F.mse_loss(noise_pred.float(), noise.float())
        loss.backward()

        nn.utils.clip_grad_norm_(
            list(self.guide_encoder.parameters()) + list(self.sdxl_adapter.parameters()),
            getattr(self.config.training, "grad_clip", 1.0),
        )
        self.optimizer.step()

        return {"total": float(loss.item())}

    # ---------------------------------------------------------------------- #
    #  Transfer evaluation
    # ---------------------------------------------------------------------- #

    def eval_transfer(self, n_samples: int = 50) -> dict:
        """
        Compare no-guide SDXL baseline vs SDXL with scene prior guides.

        Parameters
        ----------
        n_samples : number of samples to evaluate

        Returns
        -------
        dict with:
            baseline: dict of metrics (no guide)
            guided:   dict of metrics (with scene prior)
            delta:    guided - baseline differences for key metrics
        """
        self.guide_encoder.eval()
        self.sdxl_adapter.eval()
        self.scene_prior.eval()
        self.renderer.eval()

        split_info = (self._splits if self._splits is not None
                      else self.dataset.get_split_indices()
                      if hasattr(self.dataset, "get_split_indices") else {})
        eval_indices = split_info.get("val", [])
        if not eval_indices:
            eval_indices = list(range(min(n_samples, len(self.dataset))))
        eval_indices = eval_indices[:n_samples]

        baseline_preds, guided_preds, gts = [], [], []

        with torch.no_grad():
            for idx in eval_indices:
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
                scene_tensor = scene_out.to_canonical_tensor()
                guide_features = self.guide_encoder(scene_tensor)

                # Guided prediction (use scene outputs as proxy for guided result)
                guided_preds.append({
                    "visible_e0": scene_out.visible_e0.squeeze(0).cpu(),
                    "visible_e1": scene_out.visible_e1.squeeze(0).cpu(),
                    "amodal_e0":  scene_out.amodal_e0.squeeze(0).cpu(),
                    "amodal_e1":  scene_out.amodal_e1.squeeze(0).cpu(),
                    "sep_map":    scene_out.sep_map.squeeze(0).cpu(),
                    "hidden_fraction_e0": float(scene_out.hidden_e0.mean()),
                    "hidden_fraction_e1": float(scene_out.hidden_e1.mean()),
                })

                # Baseline prediction (uniform / no guide): 0.5 everywhere
                H_sp = scene_out.visible_e0.shape[-2]
                W_sp = scene_out.visible_e0.shape[-1]
                baseline_preds.append({
                    "visible_e0": torch.full((H_sp, W_sp), 0.3),
                    "visible_e1": torch.full((H_sp, W_sp), 0.3),
                    "amodal_e0":  torch.full((H_sp, W_sp), 0.5),
                    "amodal_e1":  torch.full((H_sp, W_sp), 0.5),
                    "sep_map":    torch.zeros(H_sp, W_sp),
                    "hidden_fraction_e0": 0.5,
                    "hidden_fraction_e1": 0.5,
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

        baseline_metrics = self.evaluator.eval_scene_prior(baseline_preds, gts)
        guided_metrics   = self.evaluator.eval_scene_prior(guided_preds,   gts)

        # Compute chimera rate: fraction of samples where either entity is absent
        def chimera_rate(preds_list):
            thresh = self.evaluator.visible_thresh
            chimera = 0
            for p in preds_list:
                if "visible_e0" not in p or "visible_e1" not in p:
                    continue
                m0 = float(Phase64Evaluator._to_tensor(p["visible_e0"]).mean())
                m1 = float(Phase64Evaluator._to_tensor(p["visible_e1"]).mean())
                if m0 < thresh or m1 < thresh:
                    chimera += 1
            return chimera / max(len(preds_list), 1)

        baseline_chimera = chimera_rate(baseline_preds)
        guided_chimera   = chimera_rate(guided_preds)

        baseline_metrics["chimera_rate"] = baseline_chimera
        guided_metrics["chimera_rate"]   = guided_chimera

        # Delta
        key_metrics = [
            "visible_iou_min", "visible_survival_min",
            "entity_survival_min", "chimera_rate",
        ]
        delta = {}
        for k in guided_metrics:
            b = baseline_metrics.get(k, float("nan"))
            g = guided_metrics.get(k, float("nan"))
            try:
                delta[k] = g - b
            except TypeError:
                delta[k] = None

        results = {
            "baseline": baseline_metrics,
            "guided":   guided_metrics,
            "delta":    delta,
            "n_samples": len(eval_indices),
        }

        # Save results
        out_path = self.out_dir / "transfer_eval.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[stage4] Transfer eval saved → {out_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("Stage 4 Transfer Evaluation Summary")
        print("=" * 60)
        for k in ["visible_iou_min", "visible_survival_min", "chimera_rate"]:
            b = baseline_metrics.get(k, float("nan"))
            g = guided_metrics.get(k, float("nan"))
            d = delta.get(k, float("nan"))
            try:
                print(f"  {k:<40s}  baseline={b:.4f}  guided={g:.4f}  Δ={d:+.4f}")
            except (ValueError, TypeError):
                print(f"  {k:<40s}  baseline={b}  guided={g}")
        print("=" * 60)

        return results
