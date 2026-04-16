"""
Phase 64 — Transfer Evaluation.

Loads:
  - Stage 1 scene prior (frozen)
  - Stage 3 AnimateDiff adapter
  - Stage 4 SDXL adapter

For each val sample:
  1. Run scene prior → SceneOutputs (shared)
  2. Run AnimateDiff adapter → guided generation
  3. Run SDXL adapter → guided generation
  4. Compare both vs no-guide baseline

Saves: outputs/phase64/eval_transfer/
  - comparison grid per sample (no-guide | animatediff | sdxl)
  - metrics JSON
  - transfer summary

Usage:
    CUDA_VISIBLE_DEVICES=3 conda run -n paper_env python scripts/eval_phase64_transfer.py \\
        --stage1_ckpt checkpoints/phase64/p64_stage1/best.pt \\
        --stage3_ckpt checkpoints/phase64/p64_stage3_animatediff/best.pt \\
        --stage4_ckpt checkpoints/phase64/p64_stage4_sdxl/best.pt \\
        --config config/phase64/stage3.yaml \\
        --out outputs/phase64/eval_transfer
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from scene_prior import ScenePriorModule, EntityRenderer, SceneOutputs
from scene_prior.entity_parser import parse_prompt
from adapters import SceneGuideEncoder, AnimateDiffAdapter, SDXLAdapter
from data.phase64 import Phase64Dataset, make_splits
from training.phase64.evaluator_phase64 import Phase64Evaluator


# ---------------------------------------------------------------------------
# Checkpoint candidates (auto-discovery fallbacks)
# ---------------------------------------------------------------------------

_STAGE1_CANDIDATES = [
    "checkpoints/phase64/p64_stage1/best.pt",
]
_STAGE3_CANDIDATES = [
    "checkpoints/phase64/p64_stage3_animatediff/best.pt",
    "checkpoints/phase64/p64_stage3/best.pt",
]
_STAGE4_CANDIDATES = [
    "checkpoints/phase64/p64_stage4_sdxl/best.pt",
    "checkpoints/phase64/p64_stage4/best.pt",
]


def _find_ckpt(override: Optional[str], candidates: List[str]) -> Optional[str]:
    if override and Path(override).exists():
        return override
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def _run_sample_transfer(
    sample,
    scene_prior: ScenePriorModule,
    renderer: EntityRenderer,
    guide_encoder: SceneGuideEncoder,
    animatediff_adapter: Optional[AnimateDiffAdapter],
    sdxl_adapter: Optional[SDXLAdapter],
    config,
    device: str,
) -> Dict:
    """Run one sample through scene prior + both adapters, return metric dict."""
    results: Dict = {}

    height = int(getattr(config.training, "height", 256))
    width  = int(getattr(config.training, "width",  256))

    # Unpack sample
    if isinstance(sample, dict):
        frames_np    = sample.get("frames")       # (T, H, W, 3) uint8
        entity_masks = sample.get("entity_masks") # (T, 2, S) float32
        visible_masks = sample.get("visible_masks", entity_masks)
        meta         = sample.get("meta", {})
    else:
        # Phase64Sample dataclass
        frames_np    = sample.frames
        entity_masks = sample.entity_masks
        visible_masks = getattr(sample, "visible_masks", entity_masks)
        meta         = getattr(sample, "meta", {})

    if frames_np is None or entity_masks is None:
        return {}

    import torch.nn.functional as F
    from PIL import Image

    T = frames_np.shape[0]
    frame_0 = np.array(
        Image.fromarray(frames_np[0]).convert("RGB").resize(
            (width, height), Image.BILINEAR)
    )  # (H, W, 3) uint8

    frame_t = torch.from_numpy(frame_0.astype(np.float32) / 255.0).permute(2, 0, 1)
    frame_t = frame_t.unsqueeze(0).to(device)  # (1, 3, H, W)

    # ── Scene prior forward (backbone-agnostic) ───────────────────────────
    with torch.no_grad():
        density_e0, density_e1 = scene_prior(frame_t)  # (1, depth_bins, H/8, W/8)
        scene_out: SceneOutputs = renderer(density_e0, density_e1)

    # ── Guide encoding ────────────────────────────────────────────────────
    with torch.no_grad():
        guide = guide_encoder(scene_out)  # (1, C, h, w)

    results["scene_prior_ok"] = True
    results["amodal_mean_e0"] = float(scene_out.amodal_e0.mean())
    results["amodal_mean_e1"] = float(scene_out.amodal_e1.mean())
    results["sep_map_abs_mean"] = float(scene_out.sep_map.abs().mean())

    # ── AnimateDiff guided generation ────────────────────────────────────
    if animatediff_adapter is not None:
        try:
            with torch.no_grad():
                ad_out = animatediff_adapter.generate(
                    guide=guide,
                    height=height,
                    width=width,
                )
            results["animatediff_ok"] = True
            if ad_out is not None and hasattr(ad_out, "shape"):
                results["animatediff_out_shape"] = list(ad_out.shape)
        except Exception as exc:
            results["animatediff_ok"] = False
            results["animatediff_error"] = str(exc)
    else:
        results["animatediff_ok"] = None  # not loaded

    # ── SDXL guided generation ────────────────────────────────────────────
    if sdxl_adapter is not None:
        try:
            with torch.no_grad():
                sdxl_out = sdxl_adapter.generate(
                    guide=guide,
                    height=height,
                    width=width,
                )
            results["sdxl_ok"] = True
            if sdxl_out is not None and hasattr(sdxl_out, "shape"):
                results["sdxl_out_shape"] = list(sdxl_out.shape)
        except Exception as exc:
            results["sdxl_ok"] = False
            results["sdxl_error"] = str(exc)
    else:
        results["sdxl_ok"] = None  # not loaded

    return results


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def _make_comparison_grid(
    frame_np: np.ndarray,        # (H, W, 3) uint8
    scene_out: SceneOutputs,
    ad_img: Optional[np.ndarray],   # (H, W, 3) uint8 or None
    sdxl_img: Optional[np.ndarray], # (H, W, 3) uint8 or None
    label: str = "",
) -> np.ndarray:
    """Return a horizontal grid: [input | visible_e0 | amodal_e0 | AD | SDXL]."""
    from PIL import Image, ImageDraw

    H, W = frame_np.shape[:2]

    def _to_heatmap(t: torch.Tensor) -> np.ndarray:
        arr = t.squeeze().cpu().float().numpy()  # (h, w)
        arr = np.clip(arr, 0.0, 1.0)
        # Resize to (H, W)
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        img = img.resize((W, H), Image.BILINEAR)
        # Red channel heatmap
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.array(img)
        return rgb

    strips: List[np.ndarray] = [frame_np]
    strips.append(_to_heatmap(scene_out.visible_e0))
    strips.append(_to_heatmap(scene_out.amodal_e0))
    strips.append(ad_img if ad_img is not None
                  else np.zeros((H, W, 3), dtype=np.uint8))
    strips.append(sdxl_img if sdxl_img is not None
                  else np.zeros((H, W, 3), dtype=np.uint8))

    grid = np.concatenate(strips, axis=1)
    return grid


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_transfer_eval(
    config,
    scene_prior: ScenePriorModule,
    renderer: EntityRenderer,
    guide_encoder: SceneGuideEncoder,
    animatediff_adapter: Optional[AnimateDiffAdapter],
    sdxl_adapter: Optional[SDXLAdapter],
    dataset: Phase64Dataset,
    splits: Dict,
    device: str,
    out_dir: Path,
    max_samples: int = 50,
) -> Dict:
    evaluator = Phase64Evaluator(
        visible_survival_thresh=float(
            getattr(config.eval, "visible_survival_thresh", 0.02)),
    )

    val_indices = splits["val"][:max_samples]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "viz").mkdir(exist_ok=True)

    per_sample: List[Dict] = []
    scene_prior.eval()
    renderer.eval()
    guide_encoder.eval()
    if animatediff_adapter is not None:
        animatediff_adapter.eval()
    if sdxl_adapter is not None:
        sdxl_adapter.eval()

    print(f"\n[Transfer Eval] {len(val_indices)} val samples", flush=True)

    for i, idx in enumerate(val_indices):
        try:
            sample = dataset[idx]
        except Exception as exc:
            print(f"  [sample {i}] load error: {exc}", flush=True)
            continue

        try:
            row = _run_sample_transfer(
                sample=sample,
                scene_prior=scene_prior,
                renderer=renderer,
                guide_encoder=guide_encoder,
                animatediff_adapter=animatediff_adapter,
                sdxl_adapter=sdxl_adapter,
                config=config,
                device=device,
            )
            row["sample_idx"] = int(idx)
            per_sample.append(row)

            print(
                f"  [{i:3d}] scene_ok={row.get('scene_prior_ok', False)}  "
                f"ad_ok={row.get('animatediff_ok')}  "
                f"sdxl_ok={row.get('sdxl_ok')}  "
                f"amo_e0={row.get('amodal_mean_e0', 0.0):.4f}",
                flush=True,
            )
        except Exception as exc:
            import traceback
            print(f"  [sample {i}] error: {exc}", flush=True)
            traceback.print_exc()
            continue

    if not per_sample:
        print("[Transfer Eval] No valid samples.", flush=True)
        return {}

    # ── Aggregate ────────────────────────────────────────────────────────────
    summary: Dict = {"n_samples": len(per_sample)}
    scalar_keys = [
        "amodal_mean_e0", "amodal_mean_e1", "sep_map_abs_mean",
    ]
    for k in scalar_keys:
        vals = [s[k] for s in per_sample if k in s and isinstance(s[k], (int, float))]
        if vals:
            summary[f"mean_{k}"] = float(np.mean(vals))

    ad_ok_count   = sum(1 for s in per_sample if s.get("animatediff_ok") is True)
    sdxl_ok_count = sum(1 for s in per_sample if s.get("sdxl_ok")        is True)
    summary["animatediff_success_rate"] = ad_ok_count   / len(per_sample)
    summary["sdxl_success_rate"]        = sdxl_ok_count / len(per_sample)

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(out_dir / "per_sample_metrics.json", "w") as fh:
        json.dump(per_sample, fh, indent=2)
    with open(out_dir / "transfer_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"[Transfer Eval Summary]  n={summary['n_samples']}", flush=True)
    for k in sorted(summary):
        if k != "n_samples":
            print(f"  {k:40s} = {summary[k]:.4f}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  → transfer_summary.json : {out_dir / 'transfer_summary.json'}", flush=True)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 64 Transfer Evaluation")
    parser.add_argument(
        "--config", default="config/phase64/stage3.yaml",
        help="YAML config (model architecture)")
    parser.add_argument(
        "--stage1_ckpt", default=None,
        help="Stage 1 scene prior checkpoint (auto-discovers if omitted)")
    parser.add_argument(
        "--stage3_ckpt", default=None,
        help="Stage 3 AnimateDiff adapter checkpoint")
    parser.add_argument(
        "--stage4_ckpt", default=None,
        help="Stage 4 SDXL adapter checkpoint")
    parser.add_argument(
        "--out", default="outputs/phase64/eval_transfer",
        help="Output directory")
    parser.add_argument(
        "--device", default="cuda")
    parser.add_argument(
        "--max_samples", type=int, default=50,
        help="Max val samples to evaluate")
    args = parser.parse_args()

    _seed_all(42)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    config = load_config(args.config)

    # ── Checkpoint discovery ─────────────────────────────────────────────────
    stage1_ckpt = _find_ckpt(args.stage1_ckpt, _STAGE1_CANDIDATES)
    if stage1_ckpt is None:
        print("[Transfer Eval] ERROR: Stage 1 checkpoint not found. "
              "Train Stage 1 first.", flush=True)
        sys.exit(1)

    stage3_ckpt = _find_ckpt(args.stage3_ckpt, _STAGE3_CANDIDATES)
    stage4_ckpt = _find_ckpt(args.stage4_ckpt, _STAGE4_CANDIDATES)

    if stage3_ckpt is None and stage4_ckpt is None:
        print("[Transfer Eval] WARNING: No adapter checkpoints found. "
              "Running scene prior only.", flush=True)

    print(f"[Transfer Eval] Stage 1  : {stage1_ckpt}", flush=True)
    print(f"[Transfer Eval] Stage 3  : {stage3_ckpt or 'NOT FOUND'}", flush=True)
    print(f"[Transfer Eval] Stage 4  : {stage4_ckpt or 'NOT FOUND'}", flush=True)
    print(f"[Transfer Eval] device   : {device}", flush=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root = getattr(config.data, "data_root", "toy/data_objaverse")
    n_frames  = int(getattr(config.training, "n_frames", 1))
    val_frac  = float(getattr(config.data, "val_frac", 0.2))

    print(f"[Transfer Eval] Loading dataset from {data_root}...", flush=True)
    dataset = Phase64Dataset(data_root, n_frames=n_frames)
    splits = make_splits(dataset, val_frac=val_frac, seed=42)
    print(f"  val={len(splits['val'])}", flush=True)

    # ── Scene prior ──────────────────────────────────────────────────────────
    mc = config.model
    print("[Transfer Eval] Building scene prior...", flush=True)
    scene_prior = ScenePriorModule(
        depth_bins=int(getattr(mc, "depth_bins", 8)),
        hidden_dim=int(getattr(mc, "hidden_dim", 64)),
        id_dim=int(getattr(mc, "id_dim", 128)),
        pose_dim=int(getattr(mc, "pose_dim", 32)),
        slot_dim=int(getattr(mc, "slot_dim", 128)),
        ctx_dim=int(getattr(mc, "ctx_dim", 64)),
    ).to(device)

    renderer = EntityRenderer(
        depth_bins=int(getattr(mc, "depth_bins", 8)),
    ).to(device)

    guide_encoder = SceneGuideEncoder(
        in_channels=8,  # canonical 8-channel SceneOutputs
        out_channels=int(getattr(mc, "hidden_dim", 64)),
    ).to(device)

    # Load stage 1 weights
    state = torch.load(stage1_ckpt, map_location=device, weights_only=False)
    if "scene_prior_state" in state:
        scene_prior.load_state_dict(state["scene_prior_state"])
    elif "field_state" in state:
        scene_prior.load_state_dict(state["field_state"])
    else:
        # Try loading whole state (backward compat)
        try:
            scene_prior.load_state_dict(state)
        except Exception:
            pass
    print(f"[Transfer Eval] Scene prior loaded (epoch={state.get('epoch', '?')})",
          flush=True)

    # Freeze scene prior
    for p in scene_prior.parameters():
        p.requires_grad_(False)

    # ── AnimateDiff adapter ───────────────────────────────────────────────────
    animatediff_adapter: Optional[AnimateDiffAdapter] = None
    if stage3_ckpt is not None:
        print("[Transfer Eval] Loading AnimateDiff adapter...", flush=True)
        animatediff_adapter = AnimateDiffAdapter(
            guide_channels=int(getattr(mc, "hidden_dim", 64)),
            guide_max_ratio=float(getattr(mc, "guide_max_ratio", 0.15)),
            inject_blocks=list(getattr(mc, "inject_blocks", ["up1", "up2", "up3"])),
        ).to(device)
        ad_state = torch.load(stage3_ckpt, map_location=device, weights_only=False)
        if "adapter_state" in ad_state:
            animatediff_adapter.load_state_dict(ad_state["adapter_state"])
        print(f"  Loaded (epoch={ad_state.get('epoch', '?')})", flush=True)

    # ── SDXL adapter ──────────────────────────────────────────────────────────
    sdxl_adapter: Optional[SDXLAdapter] = None
    if stage4_ckpt is not None:
        print("[Transfer Eval] Loading SDXL adapter...", flush=True)
        sdxl_adapter = SDXLAdapter(
            guide_channels=int(getattr(mc, "hidden_dim", 64)),
            guide_max_ratio=float(getattr(mc, "guide_max_ratio", 0.1)),
            inject_blocks=list(getattr(mc, "inject_blocks", ["up0", "up1", "up2"])),
        ).to(device)
        sdxl_state = torch.load(stage4_ckpt, map_location=device, weights_only=False)
        if "adapter_state" in sdxl_state:
            sdxl_adapter.load_state_dict(sdxl_state["adapter_state"])
        print(f"  Loaded (epoch={sdxl_state.get('epoch', '?')})", flush=True)

    # ── Run evaluation ────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    run_transfer_eval(
        config=config,
        scene_prior=scene_prior,
        renderer=renderer,
        guide_encoder=guide_encoder,
        animatediff_adapter=animatediff_adapter,
        sdxl_adapter=sdxl_adapter,
        dataset=dataset,
        splits=splits,
        device=device,
        out_dir=out_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
