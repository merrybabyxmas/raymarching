"""
Phase 64 — Export Scene Outputs.

Runs the backbone-agnostic scene prior on val samples and saves:
- SceneOutputs as 8-channel PNG (false-color)
- visible/amodal overlay PNGs
- separation map heatmaps
- coarse decoder RGB (if decoder checkpoint available)

This demonstrates the scene prior is independent of any backbone.

Usage:
    CUDA_VISIBLE_DEVICES=3 conda run -n paper_env python scripts/export_scene_outputs.py \\
        --config config/phase64/stage1.yaml \\
        --ckpt checkpoints/phase64/p64_stage1/best.pt \\
        --out outputs/phase64/scene_exports
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from scene_prior import ScenePriorModule, EntityRenderer, SceneOutputs
from backbones import StructuredDecoder
from data.phase64 import Phase64Dataset, make_splits


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(H, W) or (1, H, W) float tensor → (H, W) uint8."""
    arr = t.squeeze().detach().cpu().float().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _heatmap_rgb(arr_u8: np.ndarray, color: tuple) -> np.ndarray:
    """Apply a colour tint to a greyscale uint8 array.

    Parameters
    ----------
    arr_u8 : (H, W) uint8
    color  : (R, G, B) in [0, 1]

    Returns
    -------
    (H, W, 3) uint8
    """
    H, W = arr_u8.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for c, ch_val in enumerate(color):
        rgb[:, :, c] = (arr_u8.astype(np.float32) * ch_val).astype(np.uint8)
    return rgb


def _save_channel_map(
    tensor: torch.Tensor,
    path: Path,
    color: tuple = (1.0, 1.0, 1.0),
    upsample_to: Optional[int] = None,
) -> None:
    """Save a single (H, W) probability/depth tensor as a coloured PNG."""
    from PIL import Image

    arr = _tensor_to_uint8(tensor)
    if upsample_to is not None:
        h, w = arr.shape
        if h != upsample_to or w != upsample_to:
            pil = Image.fromarray(arr, mode="L")
            pil = pil.resize((upsample_to, upsample_to), Image.BILINEAR)
            arr = np.array(pil)
    rgb = _heatmap_rgb(arr, color)
    Image.fromarray(rgb).save(path)


def _save_overlay(
    frame_np: np.ndarray,        # (H, W, 3) uint8 — original frame
    visible_e0: torch.Tensor,    # (H, W) float
    visible_e1: torch.Tensor,    # (H, W) float
    amodal_e0: torch.Tensor,     # (H, W) float
    amodal_e1: torch.Tensor,     # (H, W) float
    path: Path,
) -> None:
    """Save an overlay of predicted masks on top of the input frame."""
    from PIL import Image, ImageDraw

    H, W = frame_np.shape[:2]

    def _resize_mask(t: torch.Tensor) -> np.ndarray:
        arr = t.squeeze().detach().cpu().float().numpy()  # (h, w)
        pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
        pil = pil.resize((W, H), Image.BILINEAR)
        return np.array(pil).astype(np.float32) / 255.0

    v0 = _resize_mask(visible_e0)  # (H, W)
    v1 = _resize_mask(visible_e1)
    a0 = _resize_mask(amodal_e0)
    a1 = _resize_mask(amodal_e1)

    # Compose overlay
    overlay = frame_np.astype(np.float32).copy()
    # Entity 0 — red tint for visible, orange for amodal
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + a0 * 80, 0, 255)    # amodal: soft red
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + v0 * 120, 0, 255)   # visible: stronger red
    # Entity 1 — blue tint for visible, cyan for amodal
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] + a1 * 80, 0, 255)    # amodal: soft blue
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] + v1 * 120, 0, 255)   # visible: stronger blue

    Image.fromarray(overlay.astype(np.uint8)).save(path)


def _make_8channel_png(
    scene_out: SceneOutputs,
    upsample_h: int,
    upsample_w: int,
) -> np.ndarray:
    """Stack all 8 channels into a false-color (H, W*8, 3) strip for inspection."""
    from PIL import Image

    channels = [
        (scene_out.visible_e0, (1.0, 0.3, 0.3)),   # red
        (scene_out.visible_e1, (0.3, 0.3, 1.0)),   # blue
        (scene_out.amodal_e0,  (1.0, 0.6, 0.6)),   # pink
        (scene_out.amodal_e1,  (0.6, 0.6, 1.0)),   # lavender
        (scene_out.depth_map,  (0.5, 1.0, 0.5)),   # green
        ((scene_out.sep_map + 1.0) / 2.0,
                               (1.0, 1.0, 0.3)),   # yellow (re-centred)
        (scene_out.hidden_e0,  (0.8, 0.4, 0.0)),   # orange
        (scene_out.hidden_e1,  (0.0, 0.8, 0.8)),   # cyan
    ]

    strips: List[np.ndarray] = []
    for tensor, color in channels:
        arr = _tensor_to_uint8(tensor)  # (h, w)
        pil = Image.fromarray(arr, mode="L").resize(
            (upsample_w, upsample_h), Image.BILINEAR)
        arr_up = np.array(pil)
        strips.append(_heatmap_rgb(arr_up, color))

    return np.concatenate(strips, axis=1)  # (H, W*8, 3)


# ---------------------------------------------------------------------------
# Main export loop
# ---------------------------------------------------------------------------

def run_export(
    config,
    scene_prior: ScenePriorModule,
    renderer: EntityRenderer,
    decoder: Optional[StructuredDecoder],
    dataset: Phase64Dataset,
    splits: Dict,
    device: str,
    out_dir: Path,
    max_samples: int = 50,
    upsample_to: int = 256,
) -> None:
    val_indices = splits["val"][:max_samples]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "heatmaps").mkdir(exist_ok=True)
    (out_dir / "overlays").mkdir(exist_ok=True)
    (out_dir / "8channel").mkdir(exist_ok=True)
    if decoder is not None:
        (out_dir / "decoder_rgb").mkdir(exist_ok=True)

    scene_prior.eval()
    renderer.eval()
    if decoder is not None:
        decoder.eval()

    height = int(getattr(config.training, "height", 256))
    width  = int(getattr(config.training, "width",  256))

    from PIL import Image

    export_meta: List[Dict] = []
    print(f"\n[Export] {len(val_indices)} val samples → {out_dir}", flush=True)

    for i, idx in enumerate(val_indices):
        try:
            sample = dataset[idx]
        except Exception as exc:
            print(f"  [sample {i}] load error: {exc}", flush=True)
            continue

        try:
            # Unpack
            if isinstance(sample, dict):
                frames_np = sample.get("frames")
                entity_masks = sample.get("entity_masks")
                visible_masks = sample.get("visible_masks", entity_masks)
            else:
                frames_np    = sample.frames
                entity_masks = sample.entity_masks
                visible_masks = getattr(sample, "visible_masks", entity_masks)

            if frames_np is None:
                continue

            frame_0 = np.array(
                Image.fromarray(frames_np[0]).convert("RGB").resize(
                    (width, height), Image.BILINEAR)
            )  # (H, W, 3) uint8

            frame_t = torch.from_numpy(
                frame_0.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)

            # ── Scene prior forward ───────────────────────────────────────
            with torch.no_grad():
                density_e0, density_e1 = scene_prior(frame_t)
                scene_out: SceneOutputs = renderer(density_e0, density_e1)

            tag = f"sample_{i:04d}_idx{idx}"

            # ── 8-channel false-color strip ───────────────────────────────
            strip = _make_8channel_png(scene_out, upsample_h=upsample_to,
                                       upsample_w=upsample_to)
            Image.fromarray(strip).save(out_dir / "8channel" / f"{tag}_8ch.png")

            # ── Individual heatmaps ───────────────────────────────────────
            for ch_name, tensor, color in [
                ("visible_e0", scene_out.visible_e0, (1.0, 0.3, 0.3)),
                ("visible_e1", scene_out.visible_e1, (0.3, 0.3, 1.0)),
                ("amodal_e0",  scene_out.amodal_e0,  (1.0, 0.6, 0.6)),
                ("amodal_e1",  scene_out.amodal_e1,  (0.6, 0.6, 1.0)),
                ("depth_map",  scene_out.depth_map,  (0.5, 1.0, 0.5)),
                ("sep_map",    (scene_out.sep_map + 1.0) / 2.0,
                               (1.0, 1.0, 0.3)),
                ("hidden_e0",  scene_out.hidden_e0,  (0.8, 0.4, 0.0)),
                ("hidden_e1",  scene_out.hidden_e1,  (0.0, 0.8, 0.8)),
            ]:
                _save_channel_map(
                    tensor,
                    out_dir / "heatmaps" / f"{tag}_{ch_name}.png",
                    color=color,
                    upsample_to=upsample_to,
                )

            # ── Overlay on input frame ────────────────────────────────────
            _save_overlay(
                frame_np=frame_0,
                visible_e0=scene_out.visible_e0,
                visible_e1=scene_out.visible_e1,
                amodal_e0=scene_out.amodal_e0,
                amodal_e1=scene_out.amodal_e1,
                path=out_dir / "overlays" / f"{tag}_overlay.png",
            )

            # ── Input frame ───────────────────────────────────────────────
            Image.fromarray(frame_0).save(
                out_dir / "overlays" / f"{tag}_input.png")

            # ── Coarse decoder RGB ────────────────────────────────────────
            decoder_ok = False
            if decoder is not None:
                try:
                    canonical = scene_out.to_canonical_tensor()  # (1, 8, h, w)
                    with torch.no_grad():
                        recon = decoder(canonical)  # (1, 3, H', W')
                    recon_np = (
                        recon.squeeze(0).permute(1, 2, 0)
                        .clamp(0.0, 1.0)
                        .cpu().float().numpy()
                    )
                    recon_u8 = (recon_np * 255).astype(np.uint8)
                    Image.fromarray(recon_u8).save(
                        out_dir / "decoder_rgb" / f"{tag}_decoder_rgb.png")
                    decoder_ok = True
                except Exception as exc:
                    print(f"  [sample {i}] decoder failed: {exc}", flush=True)

            export_meta.append({
                "sample_i": i,
                "dataset_idx": int(idx),
                "tag": tag,
                "amodal_mean_e0": float(scene_out.amodal_e0.mean()),
                "amodal_mean_e1": float(scene_out.amodal_e1.mean()),
                "sep_map_abs_mean": float(scene_out.sep_map.abs().mean()),
                "decoder_ok": decoder_ok,
            })

            print(
                f"  [{i:3d}] amo_e0={export_meta[-1]['amodal_mean_e0']:.4f}  "
                f"amo_e1={export_meta[-1]['amodal_mean_e1']:.4f}  "
                f"sep={export_meta[-1]['sep_map_abs_mean']:.4f}  "
                f"decoder={decoder_ok}",
                flush=True,
            )

        except Exception as exc:
            import traceback
            print(f"  [sample {i}] error: {exc}", flush=True)
            traceback.print_exc()
            continue

    # ── Save export metadata ──────────────────────────────────────────────────
    with open(out_dir / "export_meta.json", "w") as fh:
        json.dump(export_meta, fh, indent=2)

    print(f"\n[Export] Done.  {len(export_meta)} samples exported to {out_dir}",
          flush=True)
    print(f"  8channel/  : {out_dir / '8channel'}", flush=True)
    print(f"  overlays/  : {out_dir / 'overlays'}", flush=True)
    print(f"  heatmaps/  : {out_dir / 'heatmaps'}", flush=True)
    if decoder is not None:
        print(f"  decoder_rgb/: {out_dir / 'decoder_rgb'}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 64 — Export Scene Outputs")
    parser.add_argument(
        "--config", default="config/phase64/stage1.yaml",
        help="YAML config (model architecture)")
    parser.add_argument(
        "--ckpt", required=True,
        help="Stage 1 scene prior checkpoint (.pt)")
    parser.add_argument(
        "--decoder_ckpt", default=None,
        help="Stage 2 decoder checkpoint (.pt) — optional")
    parser.add_argument(
        "--out", default="outputs/phase64/scene_exports",
        help="Output directory")
    parser.add_argument(
        "--device", default="cuda")
    parser.add_argument(
        "--max_samples", type=int, default=50,
        help="Max val samples to export")
    parser.add_argument(
        "--upsample_to", type=int, default=256,
        help="Target pixel size for saved images")
    args = parser.parse_args()

    _seed_all(42)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    config = load_config(args.config)
    mc = config.model

    # ── Checkpoint check ──────────────────────────────────────────────────────
    if not Path(args.ckpt).exists():
        print(f"[Export] ERROR: checkpoint not found: {args.ckpt}", flush=True)
        sys.exit(1)

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root = getattr(config.data, "data_root", "toy/data_objaverse")
    n_frames  = int(getattr(config.data, "n_frames", 1))
    val_frac  = float(getattr(config.data, "val_frac", 0.2))

    print(f"[Export] Loading dataset from {data_root}...", flush=True)
    dataset = Phase64Dataset(data_root, n_frames=n_frames)
    splits = make_splits(dataset, val_frac=val_frac, seed=42)
    print(f"  dataset={len(dataset)}  val={len(splits['val'])}", flush=True)

    # ── Scene prior ──────────────────────────────────────────────────────────
    print("[Export] Building scene prior...", flush=True)
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

    # Load checkpoint
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    # Try common state dict keys
    loaded = False
    for key in ("scene_prior_state", "field_state", "model_state"):
        if key in state:
            scene_prior.load_state_dict(state[key])
            loaded = True
            break
    if not loaded:
        # Attempt direct load
        try:
            scene_prior.load_state_dict(state)
        except Exception as exc:
            print(f"[Export] WARNING: could not load state dict directly: {exc}",
                  flush=True)

    epoch_str = state.get("epoch", "?") if isinstance(state, dict) else "?"
    print(f"[Export] Scene prior loaded (epoch={epoch_str})", flush=True)

    scene_prior.eval()
    renderer.eval()
    for p in scene_prior.parameters():
        p.requires_grad_(False)
    for p in renderer.parameters():
        p.requires_grad_(False)

    # ── Optional decoder ─────────────────────────────────────────────────────
    decoder: Optional[StructuredDecoder] = None
    if args.decoder_ckpt and Path(args.decoder_ckpt).exists():
        print(f"[Export] Loading StructuredDecoder from {args.decoder_ckpt}...",
              flush=True)
        decoder = StructuredDecoder(
            in_channels=8,  # canonical 8-channel input
            upsample_to=int(getattr(mc, "decoder_upsample_to", 256)),
        ).to(device)
        dec_state = torch.load(
            args.decoder_ckpt, map_location=device, weights_only=False)
        dec_key = "decoder_state"
        if dec_key in dec_state:
            decoder.load_state_dict(dec_state[dec_key])
        else:
            try:
                decoder.load_state_dict(dec_state)
            except Exception as exc:
                print(f"[Export] WARNING: decoder load failed: {exc}", flush=True)
                decoder = None
        if decoder is not None:
            decoder.eval()
            for p in decoder.parameters():
                p.requires_grad_(False)
            print("[Export] Decoder ready.", flush=True)
    elif args.decoder_ckpt:
        print(f"[Export] WARNING: decoder_ckpt not found: {args.decoder_ckpt}  "
              "(skipping decoder)", flush=True)

    # ── Export ───────────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    run_export(
        config=config,
        scene_prior=scene_prior,
        renderer=renderer,
        decoder=decoder,
        dataset=dataset,
        splits=splits,
        device=device,
        out_dir=out_dir,
        max_samples=args.max_samples,
        upsample_to=args.upsample_to,
    )


if __name__ == "__main__":
    main()
