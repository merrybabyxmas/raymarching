"""Evaluate multi-view consistency of scene module.

For the same scene, check if slot embeddings are consistent across different camera views.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from phase65_min3d.data import Phase65Dataset, phase65_collate_fn
from phase65_min3d.scene_module import SceneModule


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    a_flat = a.view(-1)
    b_flat = b.view(-1)
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=50)
    args = parser.parse_args()

    set_seed(42)

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

    # Load dataset
    dataset = Phase65Dataset(
        root=cfg['data']['root'],
        image_size=cfg['data'].get('image_size', 256),
        mask_size=cfg['data'].get('mask_size', 64),
        num_frames=cfg['data'].get('num_frames', 16),
    )

    # Group samples by base scene (samples with same entities but different cameras)
    # For now, we'll use synthetic approach: same scene with different camera vectors
    results = {
        'same_slot_cosine': [],  # e0(A) vs e0(B), e1(A) vs e1(B)
        'cross_slot_cosine': [],  # e0(A) vs e1(B), e1(A) vs e0(B)
        'slot_separation': [],  # ratio of same_slot to cross_slot
    }

    num_tested = 0
    with torch.no_grad():
        for idx in range(min(len(dataset), args.num_samples * 2)):
            sample = dataset[idx]
            if sample is None:
                continue

            frames = sample.frames.unsqueeze(0).to(device)
            camera_A = sample.camera_vec.unsqueeze(0).to(device)

            # Create a different camera by perturbing
            camera_B = camera_A.clone()
            camera_B[:, :3] += torch.randn_like(camera_B[:, :3]) * 0.5  # Perturb position
            camera_B = camera_B / (camera_B.norm(dim=-1, keepdim=True) + 1e-6) * camera_A.norm(dim=-1, keepdim=True)

            # Run with camera A
            prev_state_A = None
            slots_A = []
            for t in range(frames.shape[1]):
                prev_frame = torch.zeros_like(frames[:, 0]) if t == 0 else frames[:, t - 1]
                state_A = model(
                    entity_names=sample.entity_names,
                    text_prompt=sample.text_prompt,
                    prev_state=prev_state_A,
                    prev_frame=prev_frame,
                    t_index=t,
                    camera_context=camera_A,
                )
                slots_A.append((state_A.features.feat_e0.clone(), state_A.features.feat_e1.clone()))
                prev_state_A = state_A.detach()

            # Run with camera B
            prev_state_B = None
            slots_B = []
            for t in range(frames.shape[1]):
                prev_frame = torch.zeros_like(frames[:, 0]) if t == 0 else frames[:, t - 1]
                state_B = model(
                    entity_names=sample.entity_names,
                    text_prompt=sample.text_prompt,
                    prev_state=prev_state_B,
                    prev_frame=prev_frame,
                    t_index=t,
                    camera_context=camera_B,
                )
                slots_B.append((state_B.features.feat_e0.clone(), state_B.features.feat_e1.clone()))
                prev_state_B = state_B.detach()

            # Compute consistency metrics at middle timestep
            t_mid = len(slots_A) // 2
            e0_A, e1_A = slots_A[t_mid]
            e0_B, e1_B = slots_B[t_mid]

            # Same slot similarity (should be high)
            same_e0 = cosine_similarity(e0_A, e0_B)
            same_e1 = cosine_similarity(e1_A, e1_B)

            # Cross slot similarity (should be low)
            cross_01 = cosine_similarity(e0_A, e1_B)
            cross_10 = cosine_similarity(e1_A, e0_B)

            results['same_slot_cosine'].extend([same_e0, same_e1])
            results['cross_slot_cosine'].extend([cross_01, cross_10])

            avg_same = (same_e0 + same_e1) / 2
            avg_cross = (cross_01 + cross_10) / 2
            if abs(avg_cross) > 1e-6:
                results['slot_separation'].append(avg_same / avg_cross)

            num_tested += 1
            if num_tested >= args.num_samples:
                break

    # Aggregate results
    summary = {
        'num_samples': num_tested,
        'same_slot_cosine_mean': float(np.mean(results['same_slot_cosine'])),
        'same_slot_cosine_std': float(np.std(results['same_slot_cosine'])),
        'cross_slot_cosine_mean': float(np.mean(results['cross_slot_cosine'])),
        'cross_slot_cosine_std': float(np.std(results['cross_slot_cosine'])),
        'slot_separation_mean': float(np.mean(results['slot_separation'])) if results['slot_separation'] else 0.0,
    }

    print(f"Multi-view Consistency Results:")
    print(f"  Same slot cosine: {summary['same_slot_cosine_mean']:.4f} ± {summary['same_slot_cosine_std']:.4f}")
    print(f"  Cross slot cosine: {summary['cross_slot_cosine_mean']:.4f} ± {summary['cross_slot_cosine_std']:.4f}")
    print(f"  Slot separation ratio: {summary['slot_separation_mean']:.4f}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Results saved to {args.output_json}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
