"""Visualize Cat/Dog failure cases."""
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
    return model


def visualize_failure(model, sample, device: str, output_path: Path, sample_info: dict):
    """Create visualization for a single failure case."""
    frames = sample.frames.unsqueeze(0).to(device)  # (1, T, 3, H, W)
    camera_vec = sample.camera_vec.unsqueeze(0).to(device)
    T = frames.shape[1]

    # Get predictions
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
                camera_context=camera_vec,
            )
            prev_state = state.detach()

            # Upsample to frame size
            h, w = frames.shape[-2:]
            vis_e0 = F.interpolate(state.maps.visible_e0, size=(h, w), mode='bilinear', align_corners=False)
            vis_e1 = F.interpolate(state.maps.visible_e1, size=(h, w), mode='bilinear', align_corners=False)
            amo_e0 = F.interpolate(state.maps.amodal_e0, size=(h, w), mode='bilinear', align_corners=False)
            amo_e1 = F.interpolate(state.maps.amodal_e1, size=(h, w), mode='bilinear', align_corners=False)

            pred_vis_e0.append(vis_e0)
            pred_vis_e1.append(vis_e1)
            pred_amo_e0.append(amo_e0)
            pred_amo_e1.append(amo_e1)

    pred_vis_e0 = torch.stack(pred_vis_e0, dim=1)
    pred_vis_e1 = torch.stack(pred_vis_e1, dim=1)
    pred_amo_e0 = torch.stack(pred_amo_e0, dim=1)
    pred_amo_e1 = torch.stack(pred_amo_e1, dim=1)

    # GT masks
    # sample.visible_masks: (T, 2, Hm, Wm)
    h, w = frames.shape[-2:]
    vis_masks = sample.visible_masks.unsqueeze(0).to(device)  # (1, T, 2, Hm, Wm)
    amo_masks = sample.amodal_masks.unsqueeze(0).to(device)

    gt_vis_e0 = vis_masks[:, :, 0:1]  # (1, T, 1, Hm, Wm)
    gt_vis_e1 = vis_masks[:, :, 1:2]
    gt_amo_e0 = amo_masks[:, :, 0:1]
    gt_amo_e1 = amo_masks[:, :, 1:2]

    # Create GIF frames
    gif_frames = []

    for t in range(T):
        # Layout:
        # Row 1: Frame | GT Vis E0 | GT Vis E1 | GT Amo E0 | GT Amo E1
        # Row 2: Frame | Pred Vis E0 | Pred Vis E1 | Pred Amo E0 | Pred Amo E1

        cell_h, cell_w = h, w
        n_cols = 5
        n_rows = 2
        margin = 60

        canvas = Image.new('RGB', (n_cols * cell_w + margin, n_rows * cell_h + margin * 2), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        def paste_tensor(img_tensor, row, col):
            img_np = img_tensor[0].cpu().numpy()
            if img_np.shape[0] == 1:
                img_np = np.repeat(img_np, 3, axis=0)
            img_np = (img_np.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            canvas.paste(img_pil, (col * cell_w + margin, row * cell_h + margin))

        # Row 0: GT
        paste_tensor(frames[0, t:t+1], 0, 0)
        # Upsample GT masks for this frame
        gt_v0_t = F.interpolate(gt_vis_e0[0, t:t+1], size=(h, w), mode='nearest')
        gt_v1_t = F.interpolate(gt_vis_e1[0, t:t+1], size=(h, w), mode='nearest')
        gt_a0_t = F.interpolate(gt_amo_e0[0, t:t+1], size=(h, w), mode='nearest')
        gt_a1_t = F.interpolate(gt_amo_e1[0, t:t+1], size=(h, w), mode='nearest')
        paste_tensor(gt_v0_t, 0, 1)
        paste_tensor(gt_v1_t, 0, 2)
        paste_tensor(gt_a0_t, 0, 3)
        paste_tensor(gt_a1_t, 0, 4)

        # Row 1: Pred
        paste_tensor(frames[0, t:t+1], 1, 0)
        paste_tensor(pred_vis_e0[0, t:t+1], 1, 1)
        paste_tensor(pred_vis_e1[0, t:t+1], 1, 2)
        paste_tensor(pred_amo_e0[0, t:t+1], 1, 3)
        paste_tensor(pred_amo_e1[0, t:t+1], 1, 4)

        # Labels
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = font

        # Title
        title = f"{sample_info['sample_id']}: {sample.entity_names}"
        draw.text((margin, 5), title, fill=(0, 0, 0), font=font)

        # IoU scores
        iou_text = f"Vis IoU: {sample_info['vis_iou_e0']:.3f}/{sample_info['vis_iou_e1']:.3f} | Amo IoU: {sample_info['amo_iou_e0']:.3f}/{sample_info['amo_iou_e1']:.3f}"
        draw.text((margin, 25), iou_text, fill=(200, 0, 0), font=font_small)

        # Column headers
        labels = ['Frame', 'Vis E0', 'Vis E1', 'Amo E0', 'Amo E1']
        for col, label in enumerate(labels):
            draw.text((col * cell_w + margin + 5, margin - 20), label, fill=(0, 0, 0), font=font_small)

        # Row headers
        draw.text((5, 0 * cell_h + margin + 5), 'GT', fill=(0, 0, 0), font=font_small)
        draw.text((5, 1 * cell_h + margin + 5), 'Pred', fill=(0, 0, 0), font=font_small)

        # Frame counter
        draw.text((5, n_rows * cell_h + margin + 10), f'Frame {t+1}/{T}', fill=(0, 0, 0), font=font_small)

        gif_frames.append(canvas)

    # Save GIF
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=200,
        loop=0,
    )
    print(f"Saved {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/phase65/ablation_A2/best_scene_module.pt')
    parser.add_argument('--config', type=str, default='config/phase65_min3d/ablation_A2.yaml')
    parser.add_argument('--catdog_json', type=str, default='outputs/phase65/catdog_pilot_A2.json')
    parser.add_argument('--output_dir', type=str, default='outputs/phase65/visualizations')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = load_model(args.checkpoint, args.config, device)
    print(f"Loaded model from {args.checkpoint}")

    # Load cat/dog results
    catdog_results = json.loads(Path(args.catdog_json).read_text())
    samples = catdog_results['samples']

    # Load dataset
    dataset = Phase65Dataset(
        root='data/phase65',
        image_size=256,
        mask_size=64,
        num_frames=16,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize first N samples
    count = 0
    for sample_info in samples:
        if count >= args.num_samples:
            break

        sample_id = sample_info['sample_id']

        # Find sample in dataset
        sample = None
        for s in dataset.samples:
            if s.name == sample_id:
                idx = dataset.samples.index(s)
                sample = dataset[idx]
                break

        if sample is None:
            print(f"Sample {sample_id} not found, skipping")
            continue

        print(f"Visualizing {sample_id}: {sample.entity_names}")

        output_path = output_dir / f'catdog_failure_{sample_id}.gif'
        visualize_failure(model, sample, device, output_path, sample_info)

        count += 1

    print(f"\nCreated {count} failure visualizations in {output_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
