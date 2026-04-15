"""
Phase 63 — Evaluation Script
==============================

Loads the best available checkpoint (stage3 > stage2 > stage1), runs the
EntityField + TransmittanceRenderer over the validation set, and reports:
  - per-sample metrics JSON
  - aggregate summary JSON
  - side-by-side visualisations (GT vs pred entity maps)

Usage:
    CUDA_VISIBLE_DEVICES=3 conda run -n paper_env python scripts/eval_phase63.py \\
        [--ckpt checkpoints/phase63/p63_stage3/best.pt] \\
        [--config config/phase63/stage1.yaml] \\
        [--out outputs/phase63/eval_final]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
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
from training.losses_entity import compute_entity_metrics
from training.evaluator_entity import EntityEvaluator
from scripts.train_phase39 import (
    compute_dataset_overlap_scores,
    split_train_val,
)
from scripts.train_animatediff_vca import encode_frames_to_latents
from training.phase62.evaluator import _encode_text
from scripts.run_phase63 import (
    _get_entity_tokens_p63,
    _build_gt_masks,
    _unpack_sample,
    _seed_all,
    _prob_to_rgb,
    MIN_VAL_SAMPLES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint priority
# ─────────────────────────────────────────────────────────────────────────────

_CKPT_CANDIDATES = [
    "checkpoints/phase63/p63_stage3/best.pt",
    "checkpoints/phase63/p63_stage2/best.pt",
    "checkpoints/phase63/p63_stage1/best.pt",
]


def find_best_checkpoint(override: Optional[str] = None) -> Optional[str]:
    if override and Path(override).exists():
        return override
    for p in _CKPT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(
    config,
    pipe,
    backbone_mgr: BackboneManager,
    dataset,
    system: Phase63System,
    device: str,
    out_dir: Path,
    max_samples: int = 50,
) -> Dict:
    evaluator = EntityEvaluator(visible_survival_thresh=0.02)
    gt_builder = VolumeGTBuilder(
        depth_bins=config.depth_bins,
        spatial_h=config.spatial_h,
        spatial_w=config.spatial_w,
        render_resolution=int(getattr(config.data, "volume_gt_render_resolution", 128)),
    )

    raw_ds = dataset.raw_dataset() if hasattr(dataset, "raw_dataset") else dataset
    overlap_scores = compute_dataset_overlap_scores(raw_ds)
    val_frac = getattr(config.data, "val_frac", 0.2)
    _, val_idx = split_train_val(overlap_scores, val_frac=val_frac, min_val=MIN_VAL_SAMPLES)

    tc = config.training
    height = int(getattr(tc, "height", 256))
    width  = int(getattr(tc, "width",  256))
    t_max  = int(getattr(tc, "t_max",  20))

    (out_dir / "viz").mkdir(parents=True, exist_ok=True)

    all_preds, all_gts = [], []
    per_sample_metrics: List[Dict] = []

    system.field.eval()
    system.guide_encoder.eval()
    backbone_mgr.eval()

    print(f"\n[Eval] {len(val_idx)} val samples → evaluating up to {max_samples}", flush=True)

    for sample_i, idx in enumerate(val_idx[:max_samples]):
        try:
            sample = dataset[idx]
        except Exception as e:
            print(f"  [sample {sample_i}] load error: {e}", flush=True)
            continue

        try:
            frames_np, depth_np, depth_orders, meta, sample_dir, \
                entity_masks, visible_masks = _unpack_sample(sample)

            n_frames = int(getattr(tc, "n_frames", 1))
            T = min(int(frames_np.shape[0]), n_frames)
            if T < 1:
                continue

            gt_vis_tensor = _build_gt_masks(
                (visible_masks[:T] if visible_masks is not None else entity_masks[:T]),
                config.spatial_h, config.spatial_w, device)
            gt_amo_tensor = _build_gt_masks(
                entity_masks[:T], config.spatial_h, config.spatial_w, device)

            frame_0 = np.array(
                Image.fromarray(frames_np[0]).convert("RGB").resize(
                    (width, height), Image.BILINEAR)
            )[np.newaxis]
            latents = encode_frames_to_latents(pipe, frame_0, device)

            noise = torch.randn_like(latents)
            t_val = torch.randint(0, max(1, t_max), (1,), device=device).long()
            noisy = pipe.scheduler.add_noise(latents, noise, t_val)

            toks_e0, toks_e1, full_prompt = _get_entity_tokens_p63(pipe, meta, device)
            prompt_embeds = _encode_text(pipe, full_prompt, device)
            backbone_mgr.set_entity_tokens(toks_e0, toks_e1)
            backbone_mgr.reset_slot_store()

            with torch.no_grad():
                _ = pipe.unet(noisy, t_val,
                              encoder_hidden_states=prompt_embeds, return_dict=False)
                ext = backbone_mgr.primary
                if ext.last_Fg is None:
                    continue
                F_g  = ext.last_Fg.float()
                F_e0 = ext.last_F0.float()
                F_e1 = ext.last_F1.float()

                H_gt = gt_vis_tensor.shape[-2]
                W_gt = gt_vis_tensor.shape[-1]
                color0 = meta.get("color0", [0.85, 0.15, 0.1])
                color1 = meta.get("color1", [0.1, 0.25, 0.85])
                frame0_t = torch.from_numpy(frame_0.astype(np.float32)).to(device) / 255.0
                frame0_t = frame0_t.permute(0, 3, 1, 2)
                img_small = F.interpolate(frame0_t, size=(H_gt, W_gt),
                                          mode="bilinear", align_corners=False)
                c0 = torch.tensor(color0, device=device, dtype=torch.float32).view(1, 3, 1, 1)
                c1 = torch.tensor(color1, device=device, dtype=torch.float32).view(1, 3, 1, 1)
                hint0 = (1.0 - (img_small - c0).abs().mean(1, keepdim=True) * 3.0).clamp(0.0, 1.0)
                hint1 = (1.0 - (img_small - c1).abs().mean(1, keepdim=True) * 3.0).clamp(0.0, 1.0)

                field_out, render_out = system.forward_field_and_render(
                    F_g, F_e0, F_e1, img_hint_e0=hint0, img_hint_e1=hint1)

            pred = {
                "visible_e0": render_out.visible_e0[0],
                "visible_e1": render_out.visible_e1[0],
                "amodal_e0":  render_out.amodal_e0[0],
                "amodal_e1":  render_out.amodal_e1[0],
            }
            gt = {
                "visible_e0": gt_vis_tensor[0, 0],
                "visible_e1": gt_vis_tensor[0, 1],
                "amodal_e0":  gt_amo_tensor[0, 0],
                "amodal_e1":  gt_amo_tensor[0, 1],
            }

            # Per-sample metrics
            with torch.no_grad():
                def _mr(x):
                    if x.shape[-2:] != (H_gt, W_gt):
                        return F.interpolate(x.unsqueeze(0).unsqueeze(0),
                                             size=(H_gt, W_gt),
                                             mode="bilinear",
                                             align_corners=False).squeeze()
                    return x

                sm = compute_entity_metrics(
                    _mr(pred["visible_e0"]).unsqueeze(0),
                    _mr(pred["visible_e1"]).unsqueeze(0),
                    _mr(pred["amodal_e0"]).unsqueeze(0),
                    _mr(pred["amodal_e1"]).unsqueeze(0),
                    gt_vis_tensor[:1],
                    gt_amo_tensor[:1],
                )
            sm_f = {k: float(v) for k, v in sm.items()}
            sm_f["sample_idx"] = int(idx)
            per_sample_metrics.append(sm_f)
            all_preds.append(pred)
            all_gts.append(gt)

            # ── Visualisation ───────────────────────────────────────────────
            try:
                v0  = _prob_to_rgb(pred["visible_e0"], (1.0, 0.3, 0.3))    # red
                v1  = _prob_to_rgb(pred["visible_e1"], (0.3, 0.3, 1.0))    # blue
                a0  = _prob_to_rgb(pred["amodal_e0"],  (1.0, 0.6, 0.6))
                a1  = _prob_to_rgb(pred["amodal_e1"],  (0.6, 0.6, 1.0))
                gv0 = _prob_to_rgb(gt["visible_e0"], (0.8, 0.8, 0.0))      # yellow
                gv1 = _prob_to_rgb(gt["visible_e1"], (0.0, 0.8, 0.8))      # cyan
                ga0 = _prob_to_rgb(gt["amodal_e0"],  (0.6, 0.6, 0.0))
                ga1 = _prob_to_rgb(gt["amodal_e1"],  (0.0, 0.6, 0.6))

                # Row 1: pred visible e0/e1, pred amodal e0/e1
                # Row 2: GT   visible e0/e1, GT  amodal e0/e1
                row1 = np.concatenate([v0, v1, a0, a1], axis=1)
                row2 = np.concatenate([gv0, gv1, ga0, ga1], axis=1)
                grid = np.concatenate([row1, row2], axis=0)

                # Upsample for visibility
                img = Image.fromarray(grid).resize(
                    (grid.shape[1] * 4, grid.shape[0] * 4), Image.NEAREST)
                img.save(out_dir / "viz" / f"sample_{sample_i:04d}.png")
            except Exception as e:
                print(f"  [viz {sample_i}] failed: {e}", flush=True)

            amo_iou = sm_f.get("amodal_iou_min", 0.0)
            vis_iou = sm_f.get("visible_iou_min", 0.0)
            print(f"  [{sample_i:3d}] amo_iou={amo_iou:.4f}  vis_iou={vis_iou:.4f}", flush=True)

        except Exception as e:
            import traceback
            print(f"  [sample {sample_i}] error: {e}", flush=True)
            traceback.print_exc()
            continue

    if not all_preds:
        print("[Eval] No valid samples — returning empty metrics.", flush=True)
        return {}

    # ── Aggregate over full val set ─────────────────────────────────────────
    agg = evaluator.evaluate_sequence(all_preds, all_gts)
    agg_f = {k: float(v) for k, v in agg.items()}
    agg_f["n_samples"] = len(all_preds)

    # Means from per-sample results
    metric_keys = [k for k in per_sample_metrics[0] if k != "sample_idx"]
    for k in metric_keys:
        vals = [s[k] for s in per_sample_metrics if k in s]
        agg_f[f"mean_{k}"] = float(np.mean(vals)) if vals else 0.0

    # ── Save results ─────────────────────────────────────────────────────────
    with open(out_dir / "per_sample_metrics.json", "w") as f:
        json.dump(per_sample_metrics, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(agg_f, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"[Eval Summary]  n={agg_f['n_samples']}", flush=True)
    for k in sorted(agg_f):
        if k != "n_samples" and not k.startswith("mean_"):
            print(f"  {k:35s} = {agg_f[k]:.4f}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  → summary.json: {out_dir / 'summary.json'}", flush=True)
    print(f"  → viz/        : {out_dir / 'viz'}", flush=True)

    return agg_f


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 63 Evaluation")
    parser.add_argument("--config", default="config/phase63/stage1.yaml",
                        help="YAML config (model architecture)")
    parser.add_argument("--ckpt",   default=None,
                        help="Checkpoint to load (auto-selects best if omitted)")
    parser.add_argument("--out",    default="outputs/phase63/eval_final",
                        help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Max validation samples to evaluate")
    args = parser.parse_args()

    device = args.device
    _seed_all(42)

    config = load_config(args.config)

    # ── Find checkpoint ──────────────────────────────────────────────────────
    ckpt_path = find_best_checkpoint(args.ckpt)
    if ckpt_path is None:
        print("[Eval] ERROR: no checkpoint found. "
              "Run Stage 1/2/3 training first.", flush=True)
        sys.exit(1)
    print(f"[Eval] Using checkpoint: {ckpt_path}", flush=True)

    # ── Pipeline ─────────────────────────────────────────────────────────────
    from scripts.run_animatediff import load_pipeline
    print("[Eval] Loading AnimateDiff pipeline...", flush=True)
    pipe = load_pipeline(device=device)
    pipe.scheduler.set_timesteps(50, device=device)

    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    # ── Backbone ─────────────────────────────────────────────────────────────
    print("[Eval] Injecting backbone feature extractors...", flush=True)
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
    backbone_mgr = BackboneManager(extractors, DEFAULT_INJECT_KEYS, primary_idx=2)

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root = getattr(config.data, "data_root", "toy/data_objaverse")
    n_frames  = int(getattr(config.training, "n_frames", 1))
    print(f"[Eval] Loading dataset from {data_root}...", flush=True)
    dataset = Phase62DatasetAdapter(data_root, n_frames=n_frames)
    print(f"[Eval] Dataset size: {len(dataset)}", flush=True)

    # ── System ───────────────────────────────────────────────────────────────
    system = Phase63System(config).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    system.field.load_state_dict(state["field_state"])
    if "guide_encoder_state" in state:
        system.guide_encoder.load_state_dict(state["guide_encoder_state"])
    print(f"[Eval] Loaded checkpoint (epoch {state.get('epoch', '?')})", flush=True)

    # ── Eval ─────────────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_eval(
        config=config,
        pipe=pipe,
        backbone_mgr=backbone_mgr,
        dataset=dataset,
        system=system,
        device=device,
        out_dir=out_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
