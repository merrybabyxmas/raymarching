from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from phase65_min3d.data import Phase65Dataset, phase65_collate_fn
from phase65_min3d.scene_module import SceneModule
from phase65_min3d.evaluator import Phase65Evaluator


def _make_subset(dataset: Phase65Dataset, ids: list[str]) -> Subset:
    id_to_idx = {p.name: i for i, p in enumerate(dataset.samples)}
    indices = [id_to_idx[x] for x in ids if x in id_to_idx]
    return Subset(dataset, indices)


def _mean_dict(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(sum(r[k] for r in rows) / len(rows)) for k in keys}


@torch.no_grad()
def evaluate_loader(model: SceneModule, loader: DataLoader, device: str) -> tuple[dict, dict]:
    model.eval()
    evaluator = Phase65Evaluator()
    all_metrics = []
    by_clip_type = defaultdict(list)
    for batch in loader:
        frames = batch['frames'].to(device)
        visible = batch['visible_masks'].to(device)
        amodal = batch['amodal_masks'].to(device)
        clip_types = batch['clip_types']
        _B, T = frames.shape[:2]
        prev_state = None
        for t in range(T):
            prev_frame_t = torch.zeros_like(frames[:, 0]) if t == 0 else frames[:, t - 1]
            scene_state = model(
                entity_names=batch['entity_names'][0],
                text_prompt=batch['text_prompts'][0],
                prev_state=prev_state,
                prev_frame=prev_frame_t,
                t_index=t,
            )
            metrics = evaluator.evaluate_scene(scene_state, visible[:, t], amodal[:, t], prev_state=prev_state)
            all_metrics.append(metrics)
            for clip_type in clip_types:
                by_clip_type[clip_type].append(metrics)
            prev_state = scene_state.detach()
    grouped = {k: _mean_dict(v) for k, v in by_clip_type.items()}
    return _mean_dict(all_metrics), grouped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--split_json', type=str, required=True)
    parser.add_argument('--split_name', type=str, default='test')
    parser.add_argument('--output_json', type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Phase65Dataset(
        root=cfg['data']['root'],
        image_size=cfg['data'].get('image_size', 256),
        mask_size=cfg['data'].get('mask_size', 64),
        num_frames=cfg['data'].get('num_frames', 16),
    )
    splits = json.loads(Path(args.split_json).read_text())
    subset = _make_subset(dataset, splits[args.split_name])
    loader = DataLoader(subset, batch_size=cfg['train'].get('batch_size', 2), shuffle=False, collate_fn=phase65_collate_fn)

    model = SceneModule(
        slot_dim=cfg['model'].get('slot_dim', 256),
        feat_dim=cfg['model'].get('feat_dim', 64),
        hidden_dim=cfg['model'].get('hidden_dim', 128),
        Hs=cfg['model'].get('Hs', 64),
        Ws=cfg['model'].get('Ws', 64),
        Hf=cfg['model'].get('Hf', 32),
        Wf=cfg['model'].get('Wf', 32),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])

    aggregate, grouped = evaluate_loader(model, loader, device)
    report = {
        'checkpoint': args.checkpoint,
        'split_json': args.split_json,
        'split_name': args.split_name,
        'aggregate': aggregate,
        'by_clip_type': grouped,
        'num_samples': len(subset),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
