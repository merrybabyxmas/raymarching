"""Visualize ablation results - compare A0/A1/A2/A3 predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import yaml

from phase65_min3d.data import Phase65Dataset
from phase65_min3d.scene_module import SceneModule


def load_model(checkpoint_path: str, config_path: str, device: str):
    """Load scene module from checkpoint."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = SceneModule(
        slot_dim=cfg['model'].get('slot_dim', 256),
        feat_dim=cfg['model'].get('feat_dim', 64),
        hidden_dim=cfg['model'].get('hidden_dim', 128),
        Hs=cfg['model'].get('Hs', 64),
        Ws=cfg['model'].get('Ws', 64),
        Hf=cfg['model'].get('Hf', 32),
        Wf=cfg['model'].get('Wf', 32),
    )
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()
    return model, cfg


def visualize_sample(models: dict, sample, device: str, output_path: Path):
    """Create visualization comparing all ablations."""
    frames = sample.frames.unsqueeze(0).to(device)  # (1, T, 3, H, W)
    camera_vec = sample.camera_vec.unsqueeze(0).to(device)  # (1, Dc)
    T = frames.shape[1]

    # Collect predictions from all models
    predictions = {}

    for name, (model, cfg) in models.items():
        use_camera = cfg.get('ablation', {}).get('use_camera_cond', True)
        cam_input = camera_vec if use_camera else None

        prev_state = None
        pred_vis_e0, pred_vis_e1 = [], []
        pred_amo_e0, pred_amo_e1 = [], []

        with torch.no_grad():
            for t in range(T):
                prev_frame = torch.zeros_like(frames[:, 0]) if t == 0 else frames[:, t - 1]

                state = model(
                    entity_names=sample.entity_names,
                    text_prompt=sample.text_prompt,
                    prev_state=prev_state,
                    prev_frame=prev_frame,
                    t_index=t,
                    camera_context=cam_input,
                )
                prev_state = state.detach()

                # Upsample masks to frame size
                h, w = frames.shape[-2:]
                vis_e0 = F.interpolate(state.maps.visible_e0, size=(h, w), mode='bilinear', align_corners=False)
                vis_e1 = F.interpolate(state.maps.visible_e1, size=(h, w), mode='bilinear', align_corners=False)
                amo_e0 = F.interpolate(state.maps.amodal_e0, size=(h, w), mode='bilinear', align_corners=False)
                amo_e1 = F.interpolate(state.maps.amodal_e1, size=(h, w), mode='bilinear', align_corners=False)

                pred_vis_e0.append(vis_e0)
                pred_vis_e1.append(vis_e1)
                pred_amo_e0.append(amo_e0)
                pred_amo_e1.append(amo_e1)

        predictions[name] = {
            'vis_e0': torch.stack(pred_vis_e0, dim=1),  # (1, T, 1, H, W)
            'vis_e1': torch.stack(pred_vis_e1, dim=1),
            'amo_e0': torch.stack(pred_amo_e0, dim=1),
            'amo_e1': torch.stack(pred_amo_e1, dim=1),
        }

    # Upsample GT masks to frame size
    # sample.visible_masks: (T, 2, Hm, Wm)
    h, w = frames.shape[-2:]
    vis_masks = sample.visible_masks.unsqueeze(0).to(device)  # (1, T, 2, Hm, Wm)
    amo_masks = sample.amodal_masks.unsqueeze(0).to(device)

    # Reshape for interpolation: (1*T, 1, Hm, Wm)
    gt_vis_e0 = vis_masks[:, :, 0:1]  # (1, T, 1, Hm, Wm)
    gt_vis_e1 = vis_masks[:, :, 1:2]
    gt_amo_e0 = amo_masks[:, :, 0:1]
    gt_amo_e1 = amo_masks[:, :, 1:2]

    # Create frames for GIF
    gif_frames = []

    for t in range(T):
        # Layout:
        # Row 1: GT Frame | GT Vis E0 | GT Vis E1 | GT Amo E0 | GT Amo E1
        # Row 2: A0 Vis E0 | A0 Vis E1 | A0 Amo E0 | A0 Amo E1
        # Row 3: A1 Vis E0 | A1 Vis E1 | A1 Amo E0 | A1 Amo E1
        # Row 4: A2 Vis E0 | A2 Vis E1 | A2 Amo E0 | A2 Amo E1
        # Row 5: A3 Vis E0 | A3 Vis E1 | A3 Amo E0 | A3 Amo E1

        cell_h, cell_w = h, w
        n_cols = 5
        n_rows = 5
        margin = 40  # Space for labels

        canvas = Image.new('RGB', (n_cols * cell_w + margin, n_rows * cell_h + margin), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Helper to paste image
        def paste_tensor(img_tensor, row, col):
            # img_tensor: (1, 1, H, W) or (1, 3, H, W)
            img_np = img_tensor[0].cpu().numpy()
            if img_np.shape[0] == 1:  # Grayscale mask
                img_np = np.repeat(img_np, 3, axis=0)
            img_np = (img_np.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            canvas.paste(img_pil, (col * cell_w + margin, row * cell_h + margin))

        # Row 0: GT
        paste_tensor(frames[0, t:t+1], 0, 0)  # GT Frame
        # Upsample GT masks for this frame
        gt_v0_t = F.interpolate(gt_vis_e0[0, t:t+1], size=(h, w), mode='nearest')
        gt_v1_t = F.interpolate(gt_vis_e1[0, t:t+1], size=(h, w), mode='nearest')
        gt_a0_t = F.interpolate(gt_amo_e0[0, t:t+1], size=(h, w), mode='nearest')
        gt_a1_t = F.interpolate(gt_amo_e1[0, t:t+1], size=(h, w), mode='nearest')
        paste_tensor(gt_v0_t, 0, 1)
        paste_tensor(gt_v1_t, 0, 2)
        paste_tensor(gt_a0_t, 0, 3)
        paste_tensor(gt_a1_t, 0, 4)

        # Rows 1-4: A0, A1, A2, A3
        for i, name in enumerate(['A0', 'A1', 'A2', 'A3']):
            row = i + 1
            preds = predictions[name]
            paste_tensor(preds['vis_e0'][0, t:t+1], row, 1)
            paste_tensor(preds['vis_e1'][0, t:t+1], row, 2)
            paste_tensor(preds['amo_e0'][0, t:t+1], row, 3)
            paste_tensor(preds['amo_e1'][0, t:t+1], row, 4)
            # Paste frame in first column for reference
            paste_tensor(frames[0, t:t+1], row, 0)

        # Add labels
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()

        # Column headers
        labels = ['Frame', 'Vis E0', 'Vis E1', 'Amo E0', 'Amo E1']
        for col, label in enumerate(labels):
            draw.text((col * cell_w + margin + 5, 5), label, fill=(0, 0, 0), font=font)

        # Row headers
        row_labels = ['GT', 'A0 (no cam, no hard)', 'A1 (cam only)', 'A2 (hard only)', 'A3 (cam+hard)']
        for row, label in enumerate(row_labels):
            draw.text((5, row * cell_h + margin + 5), label, fill=(0, 0, 0), font=font)

        # Frame number
        draw.text((5, n_rows * cell_h + margin + 5), f'Frame {t+1}/{T}', fill=(0, 0, 0), font=font)

        gif_frames.append(canvas)

    # Save GIF
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=200,
        loop=0,
    )
    print(f"Saved ablation comparison to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_id', type=str, default='sample_000001')
    parser.add_argument('--output_dir', type=str, default='outputs/phase65/visualizations')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load all models
    models = {}
    for name in ['A0', 'A1', 'A2', 'A3']:
        checkpoint = f'outputs/phase65/ablation_{name}/best_scene_module.pt'
        config = f'config/phase65_min3d/ablation_{name}.yaml'
        if Path(checkpoint).exists() and Path(config).exists():
            models[name] = load_model(checkpoint, config, device)
            print(f"Loaded {name}")

    if not models:
        print("No models found!")
        return 1

    # Load dataset
    dataset = Phase65Dataset(
        root='data/phase65',
        image_size=256,
        mask_size=64,
        num_frames=16,
    )

    # Find sample
    sample = None
    for s in dataset.samples:
        if s.name == args.sample_id:
            idx = dataset.samples.index(s)
            sample = dataset[idx]
            break

    if sample is None:
        print(f"Sample {args.sample_id} not found!")
        return 1

    print(f"Visualizing {args.sample_id}: {sample.entity_names}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualization
    output_path = output_dir / f'ablation_comparison_{args.sample_id}.gif'
    visualize_sample(models, sample, device, output_path)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
