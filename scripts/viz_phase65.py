from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from phase65_min3d.data import Phase65Dataset, phase65_collate_fn
from phase65_min3d.evaluator import Phase65Evaluator
from phase65_min3d.scene_module import SceneModule


def _to_rgb_map(x: torch.Tensor, color: str = "red") -> np.ndarray:
    arr = x.detach().cpu().float().squeeze().clamp(0, 1).numpy()
    h, w = arr.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    v = (arr * 255).astype(np.uint8)
    if color == "red":
        out[..., 0] = v
    elif color == "blue":
        out[..., 2] = v
    elif color == "gray":
        out[..., :] = v[..., None]
    elif color == "green":
        out[..., 1] = v
    return out


def _overlay(frame: np.ndarray, mask: torch.Tensor, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    m = mask.detach().cpu().float().squeeze().clamp(0, 1).numpy()
    img = Image.fromarray((m * 255).astype(np.uint8), mode="L").resize((frame.shape[1], frame.shape[0]), Image.BILINEAR)
    m = np.asarray(img).astype(np.float32) / 255.0
    out = frame.astype(np.float32).copy()
    for c, col in enumerate(color):
        out[..., c] = (1 - alpha * m) * out[..., c] + alpha * m * col
    return out.clip(0, 255).astype(np.uint8)


def _hstack(imgs: list[np.ndarray], gap: int = 4) -> np.ndarray:
    h = max(im.shape[0] for im in imgs)
    pad = np.ones((h, gap, 3), dtype=np.uint8) * 245
    parts = []
    for i, im in enumerate(imgs):
        if im.shape[0] < h:
            extra = np.ones((h - im.shape[0], im.shape[1], 3), dtype=np.uint8) * 245
            im = np.vstack([im, extra])
        parts.append(im)
        if i < len(imgs) - 1:
            parts.append(pad)
    return np.hstack(parts)


def _annotate(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([(0, 0), (pil.size[0], 18)], fill=(255, 255, 255))
    draw.text((4, 2), text, fill=(0, 0, 0))
    return np.array(pil)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase65_min3d/stage1.yaml")
    parser.add_argument("--ckpt", default="outputs/phase65/stage1/best_scene_module.pt")
    parser.add_argument("--out", default="outputs/phase65/viz")
    parser.add_argument("--n_samples", type=int, default=8)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dataset = Phase65Dataset(
        root=cfg["data"]["root"],
        image_size=cfg["data"].get("image_size", 256),
        mask_size=cfg["data"].get("mask_size", 64),
        num_frames=cfg["data"].get("num_frames", 16),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=phase65_collate_fn)
    model = SceneModule(
        slot_dim=cfg["model"].get("slot_dim", 256),
        feat_dim=cfg["model"].get("feat_dim", 64),
        hidden_dim=cfg["model"].get("hidden_dim", 128),
        Hs=cfg["model"].get("Hs", 64),
        Ws=cfg["model"].get("Ws", 64),
        Hf=cfg["model"].get("Hf", 32),
        Wf=cfg["model"].get("Wf", 32),
    ).to(device).eval()
    evaluator = Phase65Evaluator()
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    saved = 0
    prev_state = None
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            visible = batch["visible_masks"].to(device)
            amodal = batch["amodal_masks"].to(device)
            entity_names = batch["entity_names"][0]
            prompt = batch["text_prompts"][0]
            T = frames.shape[1]
            sample_dir = out_root / f"sample_{saved:04d}_{entity_names[0]}_{entity_names[1]}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for t in range(min(T, 3)):
                scene = model(entity_names=entity_names, text_prompt=prompt, prev_state=prev_state, prev_frame=None if t == 0 else frames[:, t - 1], t_index=t)
                metrics = evaluator.evaluate_scene(scene, visible[:, t], amodal[:, t], prev_state=prev_state)
                prev_state = scene.detach()
                frame = (frames[0, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                vis0 = _to_rgb_map(scene.maps.visible_e0[0], "red")
                vis1 = _to_rgb_map(scene.maps.visible_e1[0], "blue")
                hid0 = _to_rgb_map(scene.maps.hidden_e0[0], "red")
                hid1 = _to_rgb_map(scene.maps.hidden_e1[0], "blue")
                dep = _to_rgb_map(scene.maps.depth_e0[0], "gray")

                pred_overlay = _overlay(_overlay(frame.copy(), scene.maps.visible_e0[0], (255, 80, 80)), scene.maps.visible_e1[0], (80, 80, 255))
                gt_overlay = _overlay(_overlay(frame.copy(), visible[0, t, 0], (255, 80, 80)), visible[0, t, 1], (80, 80, 255))

                vis0 = _annotate(vis0, f"pred vis e0")
                vis1 = _annotate(vis1, f"pred vis e1")
                hid0 = _annotate(hid0, f"pred hid e0")
                hid1 = _annotate(hid1, f"pred hid e1")
                dep = _annotate(dep, f"depth e0")
                pred_overlay = _annotate(pred_overlay, f"pred overlay")
                gt_overlay = _annotate(gt_overlay, f"gt overlay")
                frame_annot = _annotate(frame, f"frame t={t} vis_min={metrics['visible_iou_min']:.3f}")

                grid = _hstack([frame_annot, gt_overlay, pred_overlay, vis0, vis1, hid0, hid1, dep])
                Image.fromarray(grid).save(sample_dir / f"frame_{t:02d}.png")
            saved += 1
            prev_state = None
            if saved >= args.n_samples:
                break
    print(f"Saved Phase 65 visualizations to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
