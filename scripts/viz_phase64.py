"""
Phase 64 — Visualization Script

Generates per-sample GIF/PNG grids from the scene prior outputs.

For each val sample, saves:
  outputs/phase64/viz/
    sample_XXXX/
      scene_maps.png    — [input | vis_e0 | vis_e1 | amo_e0 | amo_e1 | depth | sep]
      visible_e0.png
      visible_e1.png
      amodal_e0.png
      amodal_e1.png
      overlay.png       — GT masks overlaid on input

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n paper_env python scripts/viz_phase64.py \\
        --stage1_ckpt checkpoints/phase64/p64_stage1/best_scene_prior.pt \\
        --out outputs/phase64/viz \\
        --n_samples 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from scene_prior import ScenePriorModule
from data.phase64 import Phase64Dataset, make_splits


def _to_heatmap_rgb(t: torch.Tensor, cmap: str = "red") -> np.ndarray:
    """(H, W) float [0,1] → (H, W, 3) uint8 heatmap."""
    arr = t.squeeze().cpu().float().clamp(0, 1).numpy()
    h, w = arr.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    v = (arr * 255).astype(np.uint8)
    if cmap == "red":
        out[:, :, 0] = v
        out[:, :, 1] = (v * 0.2).astype(np.uint8)
    elif cmap == "blue":
        out[:, :, 2] = v
        out[:, :, 1] = (v * 0.2).astype(np.uint8)
    elif cmap == "green":
        out[:, :, 1] = v
    elif cmap == "gray":
        out[:, :, 0] = v; out[:, :, 1] = v; out[:, :, 2] = v
    elif cmap == "diverge":  # signed sep_map
        arr_s = t.squeeze().cpu().float().clamp(-1, 1).numpy()
        pos = np.clip(arr_s,  0, 1)
        neg = np.clip(-arr_s, 0, 1)
        out[:, :, 0] = (pos * 255).astype(np.uint8)
        out[:, :, 2] = (neg * 255).astype(np.uint8)
    return out


def _overlay_mask(frame: np.ndarray, mask: torch.Tensor,
                  color: tuple, alpha: float = 0.45) -> np.ndarray:
    """Draw semi-transparent colored mask over frame (H, W, 3) uint8."""
    from PIL import Image as PIL
    m = mask.squeeze().cpu().float().clamp(0, 1).numpy()
    from PIL import Image
    m_img = Image.fromarray((m * 255).astype(np.uint8), "L")
    m_img = m_img.resize((frame.shape[1], frame.shape[0]), Image.BILINEAR)
    m = np.array(m_img).astype(np.float32) / 255.0

    out = frame.copy().astype(np.float32)
    for c, col in enumerate(color):
        out[:, :, c] = (1 - alpha * m) * out[:, :, c] + alpha * m * col
    return out.clip(0, 255).astype(np.uint8)


def _hstack(imgs: list, gap: int = 4) -> np.ndarray:
    """Horizontally stack uint8 images with a white gap."""
    h = max(img.shape[0] for img in imgs)
    parts = []
    white = np.ones((h, gap, 3), dtype=np.uint8) * 240
    for i, img in enumerate(imgs):
        if img.shape[0] < h:
            pad = np.ones((h - img.shape[0], img.shape[1], 3), dtype=np.uint8) * 240
            img = np.vstack([img, pad])
        parts.append(img)
        if i < len(imgs) - 1:
            parts.append(white)
    return np.hstack(parts)


def main():
    parser = argparse.ArgumentParser(description="Phase 64 Visualization")
    parser.add_argument("--config", default="config/phase64/stage1.yaml")
    parser.add_argument("--stage1_ckpt", default="checkpoints/phase64/p64_stage1/best_scene_prior.pt")
    parser.add_argument("--out", default="outputs/phase64/viz")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from PIL import Image
    device = args.device if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Load config + model ─────────────────────────────────────────────────
    config = load_config(args.config)
    mc = config.model

    scene_prior = ScenePriorModule(
        depth_bins=int(getattr(mc, "depth_bins", 8)),
        hidden=int(getattr(mc, "hidden_dim", 64)),
        id_dim=int(getattr(mc, "id_dim", 128)),
        pose_dim=int(getattr(mc, "pose_dim", 32)),
        slot_dim=int(getattr(mc, "slot_dim", 128)),
        ctx_dim=int(getattr(mc, "ctx_dim", 64)),
    ).to(device).eval()

    state = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
    key = "scene_prior_state" if "scene_prior_state" in state else "field_state"
    if key in state:
        scene_prior.load_state_dict(state[key])
    print(f"Loaded scene prior from {args.stage1_ckpt}  (epoch={state.get('epoch','?')})")

    # ── Dataset ─────────────────────────────────────────────────────────────
    data_root = getattr(config.data, "data_root", "toy/data_objaverse")
    height = int(getattr(config.training, "height", 256))
    width  = int(getattr(config.training, "width",  256))
    dataset = Phase64Dataset(data_root, n_frames=1)
    splits  = make_splits(dataset, val_frac=0.2, seed=42)
    indices = splits["val"][:args.n_samples]
    print(f"Visualizing {len(indices)} val samples → {out_root}")

    summary_frames = []  # for overview GIF

    for rank, idx in enumerate(indices):
        sample = dataset[idx]
        meta       = getattr(sample, "meta", {})
        frames_np  = sample.frames          # (T, H, W, 3) uint8
        scene_gt   = sample.scene_gt
        routing_e0 = sample.routing_e0      # (T, H, W)
        routing_e1 = sample.routing_e1

        kw0 = meta.get("keyword0", "entity0")
        kw1 = meta.get("keyword1", "entity1")

        frame_0 = np.array(
            Image.fromarray(frames_np[0]).convert("RGB").resize((width, height), Image.BILINEAR)
        )
        frame_t = torch.from_numpy(frame_0.astype(np.float32) / 255.0
                                   ).permute(2, 0, 1).unsqueeze(0).to(device)

        # Routing hints
        def _rt(r_np):
            r = r_np[0]
            ri = Image.fromarray((r * 255).clip(0, 255).astype(np.uint8), "L")
            ri = ri.resize((width, height), Image.BILINEAR)
            return torch.from_numpy(np.array(ri).astype(np.float32) / 255.0
                                    ).unsqueeze(0).unsqueeze(0).to(device)

        r0_t = _rt(routing_e0)
        r1_t = _rt(routing_e1)

        with torch.no_grad():
            scene_out, _, _ = scene_prior(
                img=frame_t,
                entity_name_e0=kw0,
                entity_name_e1=kw1,
                routing_hint_e0=r0_t,
                routing_hint_e1=r1_t,
            )

        # ── Resize scene outputs to display resolution ──────────────────────
        def _up(t: torch.Tensor) -> torch.Tensor:
            """Upsample (1,H,W) or (H,W) to (height, width)."""
            t4 = t.float().squeeze().unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
            return torch.nn.functional.interpolate(
                t4, size=(height, width), mode="bilinear", align_corners=False
            ).squeeze(0)  # (1, H, W)

        # ── Build panels ────────────────────────────────────────────────────
        vis_e0 = _to_heatmap_rgb(_up(scene_out.visible_e0), "red")
        vis_e1 = _to_heatmap_rgb(_up(scene_out.visible_e1), "blue")
        amo_e0 = _to_heatmap_rgb(_up(scene_out.amodal_e0), "red")
        amo_e1 = _to_heatmap_rgb(_up(scene_out.amodal_e1), "blue")
        depth  = _to_heatmap_rgb(_up(scene_out.depth_map), "gray")
        sep    = _to_heatmap_rgb(_up(scene_out.sep_map),   "diverge")
        hidden_e0 = _to_heatmap_rgb(_up(scene_out.hidden_e0), "red")
        hidden_e1 = _to_heatmap_rgb(_up(scene_out.hidden_e1), "blue")

        # GT overlay
        gt_overlay = frame_0.copy()
        if scene_gt is not None:
            gt_vis_e0 = torch.from_numpy(scene_gt.vis_e0[0])
            gt_vis_e1 = torch.from_numpy(scene_gt.vis_e1[0])
            gt_overlay = _overlay_mask(gt_overlay, gt_vis_e0, (255, 80, 80))
            gt_overlay = _overlay_mask(gt_overlay, gt_vis_e1, (80, 80, 255))

        # Pred overlay
        pred_overlay = _overlay_mask(frame_0.copy(), scene_out.visible_e0, (255, 80, 80))
        pred_overlay = _overlay_mask(pred_overlay,   scene_out.visible_e1, (80, 80, 255))

        # Row 1: input | vis_e0(R) | vis_e1(B) | amo_e0 | amo_e1 | depth | sep
        # Row 2: gt_overlay | pred_overlay | hidden_e0 | hidden_e1 | (padding to 7)
        padding = np.ones_like(frame_0) * 240
        row1 = _hstack([frame_0, vis_e0, vis_e1, amo_e0, amo_e1, depth, sep])
        row2 = _hstack([gt_overlay, pred_overlay, hidden_e0, hidden_e1,
                        padding, padding, padding])

        divider = np.ones((4, row1.shape[1], 3), np.uint8) * 200
        grid = np.vstack([row1, divider, row2])

        # ── Save ────────────────────────────────────────────────────────────
        sample_dir = out_root / f"sample_{idx:04d}_{kw0}_{kw1}"
        sample_dir.mkdir(exist_ok=True)

        Image.fromarray(grid).save(sample_dir / "scene_maps.png")
        Image.fromarray(vis_e0).save(sample_dir / "visible_e0.png")
        Image.fromarray(vis_e1).save(sample_dir / "visible_e1.png")
        Image.fromarray(amo_e0).save(sample_dir / "amodal_e0.png")
        Image.fromarray(amo_e1).save(sample_dir / "amodal_e1.png")
        Image.fromarray(pred_overlay).save(sample_dir / "overlay.png")

        summary_frames.append(
            Image.fromarray(np.array(
                Image.fromarray(grid).resize((grid.shape[1] // 2, grid.shape[0] // 2), Image.BILINEAR)
            ))
        )
        print(f"  [{rank:3d}] {kw0}+{kw1} → {sample_dir.name}")

    # ── Overview GIF ────────────────────────────────────────────────────────
    if summary_frames:
        gif_path = out_root / "overview.gif"
        summary_frames[0].save(
            gif_path,
            save_all=True,
            append_images=summary_frames[1:],
            duration=600,
            loop=0,
        )
        print(f"\nOverview GIF saved: {gif_path}  ({len(summary_frames)} frames)")

    print(f"\nAll visualizations saved to: {out_root}")


if __name__ == "__main__":
    main()
