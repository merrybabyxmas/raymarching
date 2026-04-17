"""Generate GIF visualizations of Stage 1 model predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import imageio
import numpy as np
import torch
import yaml

from phase65_min3d.data import Phase65Dataset
from phase65_min3d.scene_module import SceneModule


def overlay_masks(frame: np.ndarray, mask_e0: np.ndarray, mask_e1: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay entity masks on frame with distinct colors."""
    out = frame.copy().astype(np.float32)
    out[..., 0] = out[..., 0] * (1 - alpha * mask_e0) + 255 * alpha * mask_e0
    out[..., 2] = out[..., 2] * (1 - alpha * mask_e1) + 255 * alpha * mask_e1
    return np.clip(out, 0, 255).astype(np.uint8)


def create_comparison_frame(
    frame: np.ndarray,
    pred_vis_e0: np.ndarray,
    pred_vis_e1: np.ndarray,
    gt_vis_e0: np.ndarray,
    gt_vis_e1: np.ndarray,
) -> np.ndarray:
    gt_overlay = overlay_masks(frame, gt_vis_e0, gt_vis_e1)
    pred_overlay = overlay_masks(frame, pred_vis_e0, pred_vis_e1)
    combined = np.concatenate([gt_overlay, pred_overlay], axis=1)
    return combined


@torch.no_grad()
def visualize_sample(
    model: SceneModule,
    dataset: Phase65Dataset,
    sample_idx: int,
    device: str,
    output_path: Path,
) -> None:
    sample = dataset[sample_idx]
    frames = sample.frames
    visible = sample.visible_masks
    T = frames.shape[0]

    model.eval()
    pred_frames: List[np.ndarray] = []
    prev_state = None
    camera_context = sample.camera_vec.unsqueeze(0).to(device)

    for t in range(T):
        frame_t = frames[t].unsqueeze(0).to(device)
        prev_frame = torch.zeros_like(frame_t) if t == 0 else frames[t - 1].unsqueeze(0).to(device)

        state = model(
            entity_names=sample.entity_names,
            text_prompt=sample.text_prompt,
            prev_state=prev_state,
            prev_frame=prev_frame,
            t_index=t,
            camera_context=camera_context,
        )
        prev_state = state.detach()

        pred_vis_e0 = state.maps.visible_e0[0, 0].cpu().numpy()
        pred_vis_e1 = state.maps.visible_e1[0, 0].cpu().numpy()
        gt_vis_e0 = visible[t, 0].numpy()
        gt_vis_e1 = visible[t, 1].numpy()
        frame_np = (frames[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        h, w = frame_np.shape[:2]
        from PIL import Image

        def resize_mask(mask: np.ndarray, size: tuple) -> np.ndarray:
            return np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(size, Image.BILINEAR)) / 255.0

        pred_vis_e0 = resize_mask(pred_vis_e0, (w, h))
        pred_vis_e1 = resize_mask(pred_vis_e1, (w, h))
        gt_vis_e0 = resize_mask(gt_vis_e0, (w, h))
        gt_vis_e1 = resize_mask(gt_vis_e1, (w, h))

        comp_frame = create_comparison_frame(frame_np, pred_vis_e0, pred_vis_e1, gt_vis_e0, gt_vis_e1)
        pred_frames.append(comp_frame)

    imageio.mimsave(output_path, pred_frames, fps=8, loop=0)
    print(f'Saved: {output_path}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pt file')
    parser.add_argument('--config', type=str, required=True, help='Path to config .yaml file')
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Phase65Dataset(
        root=cfg['data']['root'],
        image_size=cfg['data'].get('image_size', 256),
        mask_size=cfg['data'].get('mask_size', 64),
        num_frames=cfg['data'].get('num_frames', 16),
    )

    model = SceneModule(
        slot_dim=cfg['model'].get('slot_dim', 256),
        feat_dim=cfg['model'].get('feat_dim', 64),
        hidden_dim=cfg['model'].get('hidden_dim', 128),
        Hs=cfg['model'].get('Hs', 64),
        Ws=cfg['model'].get('Ws', 64),
        Hf=cfg['model'].get('Hf', 32),
        Wf=cfg['model'].get('Wf', 32),
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    for idx in indices:
        output_path = output_dir / f'sample_{idx:04d}.gif'
        visualize_sample(model, dataset, idx, device, output_path)

    print(f'Visualizations saved to {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
