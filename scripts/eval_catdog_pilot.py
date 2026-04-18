"""Cat/Dog Pilot - Analyze failure types on cat and dog samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from phase65_min3d.data import Phase65Dataset, phase65_collate_fn
from phase65_min3d.scene_module import SceneModule


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute IoU between predicted and target masks."""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    return (intersection / (union + 1e-6)).item()


def classify_failure(vis_iou_e0: float, vis_iou_e1: float,
                     amo_iou_e0: float, amo_iou_e1: float,
                     threshold: float = 0.3) -> str:
    """Classify failure type based on IoU metrics."""
    vis_ok = vis_iou_e0 > threshold or vis_iou_e1 > threshold
    amo_ok = amo_iou_e0 > threshold or amo_iou_e1 > threshold

    if vis_ok and amo_ok:
        return "success"
    elif not vis_ok and not amo_ok:
        return "both_fail"
    elif not vis_ok:
        return "visible_fail"
    else:
        return "amodal_fail"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=30)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
    print(f"Loaded model from {args.checkpoint}")

    # Find cat/dog samples
    data_root = Path(cfg['data']['root'])
    catdog_samples = []
    for sample_dir in sorted(data_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        meta_file = sample_dir / 'meta.json'
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        entity_names = meta.get('entity_names', [])
        # Check if any entity is cat or dog
        if any('cat' in name.lower() or 'dog' in name.lower() for name in entity_names):
            catdog_samples.append(sample_dir.name)
        if len(catdog_samples) >= args.num_samples * 2:
            break

    print(f"Found {len(catdog_samples)} cat/dog samples")

    # Load dataset and create subset
    dataset = Phase65Dataset(
        root=cfg['data']['root'],
        image_size=cfg['data'].get('image_size', 256),
        mask_size=cfg['data'].get('mask_size', 64),
        num_frames=cfg['data'].get('num_frames', 16),
    )

    id_to_idx = {p.name: i for i, p in enumerate(dataset.samples)}
    indices = [id_to_idx[s] for s in catdog_samples if s in id_to_idx][:args.num_samples]
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=phase65_collate_fn)
    print(f"Testing {len(subset)} samples")

    # Run inference and classify failures
    results = {
        'samples': [],
        'failure_counts': {
            'success': 0,
            'visible_fail': 0,
            'amodal_fail': 0,
            'both_fail': 0,
        },
        'entity_stats': {
            'cat': {'total': 0, 'success': 0},
            'dog': {'total': 0, 'success': 0},
        }
    }

    with torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(device)
            camera_vecs = batch['camera_vecs'].to(device)
            # visible_masks: (B, T, 2, H, W) where 2 is [e0, e1]
            vis_masks = batch['visible_masks'].to(device)
            amo_masks = batch['amodal_masks'].to(device)
            vis_e0_gt = vis_masks[:, :, 0]  # (B, T, H, W)
            vis_e1_gt = vis_masks[:, :, 1]
            amo_e0_gt = amo_masks[:, :, 0]
            amo_e1_gt = amo_masks[:, :, 1]
            entity_names = batch['entity_names'][0]
            # Get sample_id from the dataset index
            sample_idx = loader.dataset.indices[len(results['samples'])] if hasattr(loader.dataset, 'indices') else len(results['samples'])
            sample_id = dataset.samples[sample_idx].name if sample_idx < len(dataset.samples) else f"sample_{len(results['samples'])}"

            B, T = frames.shape[:2]

            # Run through all timesteps
            prev_state = None
            vis_ious_e0, vis_ious_e1 = [], []
            amo_ious_e0, amo_ious_e1 = [], []

            for t in range(T):
                prev_frame = torch.zeros_like(frames[:, 0]) if t == 0 else frames[:, t - 1]

                state = model(
                    entity_names=entity_names,
                    text_prompt=batch['text_prompts'][0],
                    prev_state=prev_state,
                    prev_frame=prev_frame,
                    t_index=t,
                    camera_context=camera_vecs,
                )
                prev_state = state.detach()

                # Compute IoU for this timestep
                vis_ious_e0.append(compute_iou(state.maps.visible_e0, vis_e0_gt[:, t]))
                vis_ious_e1.append(compute_iou(state.maps.visible_e1, vis_e1_gt[:, t]))
                amo_ious_e0.append(compute_iou(state.maps.amodal_e0, amo_e0_gt[:, t]))
                amo_ious_e1.append(compute_iou(state.maps.amodal_e1, amo_e1_gt[:, t]))

            # Average IoU across timesteps
            avg_vis_e0 = np.mean(vis_ious_e0)
            avg_vis_e1 = np.mean(vis_ious_e1)
            avg_amo_e0 = np.mean(amo_ious_e0)
            avg_amo_e1 = np.mean(amo_ious_e1)

            failure_type = classify_failure(avg_vis_e0, avg_vis_e1, avg_amo_e0, avg_amo_e1)

            sample_result = {
                'sample_id': sample_id,
                'entity_names': entity_names,
                'vis_iou_e0': float(avg_vis_e0),
                'vis_iou_e1': float(avg_vis_e1),
                'amo_iou_e0': float(avg_amo_e0),
                'amo_iou_e1': float(avg_amo_e1),
                'failure_type': failure_type,
            }
            results['samples'].append(sample_result)
            results['failure_counts'][failure_type] += 1

            # Update entity stats
            for name in entity_names:
                entity_type = 'cat' if 'cat' in name.lower() else 'dog' if 'dog' in name.lower() else None
                if entity_type:
                    results['entity_stats'][entity_type]['total'] += 1
                    if failure_type == 'success':
                        results['entity_stats'][entity_type]['success'] += 1

            print(f"[{sample_id}] {entity_names} -> {failure_type} "
                  f"(vis:{avg_vis_e0:.3f}/{avg_vis_e1:.3f}, amo:{avg_amo_e0:.3f}/{avg_amo_e1:.3f})")

    # Compute summary statistics
    results['summary'] = {
        'total_samples': len(results['samples']),
        'success_rate': results['failure_counts']['success'] / len(results['samples']) if results['samples'] else 0,
        'visible_fail_rate': results['failure_counts']['visible_fail'] / len(results['samples']) if results['samples'] else 0,
        'amodal_fail_rate': results['failure_counts']['amodal_fail'] / len(results['samples']) if results['samples'] else 0,
        'both_fail_rate': results['failure_counts']['both_fail'] / len(results['samples']) if results['samples'] else 0,
    }

    for entity_type in ['cat', 'dog']:
        total = results['entity_stats'][entity_type]['total']
        if total > 0:
            results['summary'][f'{entity_type}_success_rate'] = results['entity_stats'][entity_type]['success'] / total

    print("\n=== Summary ===")
    print(f"Total samples: {results['summary']['total_samples']}")
    print(f"Success rate: {results['summary']['success_rate']:.2%}")
    print(f"Visible fail: {results['summary']['visible_fail_rate']:.2%}")
    print(f"Amodal fail: {results['summary']['amodal_fail_rate']:.2%}")
    print(f"Both fail: {results['summary']['both_fail_rate']:.2%}")

    # Save results
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.output_json}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
