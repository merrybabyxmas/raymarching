"""Training script for ablation study with camera conditioning toggle."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from phase65_min3d.data import Phase65Dataset, phase65_collate_fn
from phase65_min3d.scene_module import SceneModule
from phase65_min3d.trainer_stage1 import Stage1Batch, Stage1Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean_metrics(epoch_metrics: list[dict]) -> dict:
    return {k: float(sum(m[k] for m in epoch_metrics) / max(len(epoch_metrics), 1)) for k in epoch_metrics[0].keys()}


def _make_subset(dataset: Phase65Dataset, ids: list[str]) -> Subset:
    id_to_idx = {p.name: i for i, p in enumerate(dataset.samples)}
    indices = [id_to_idx[x] for x in ids if x in id_to_idx]
    return Subset(dataset, indices)


@torch.no_grad()
def evaluate_epoch(trainer: Stage1Trainer, loader: DataLoader, device: str, use_camera_cond: bool) -> dict:
    trainer.scene_module.eval()
    all_metrics = []
    for batch in loader:
        frames = batch['frames'].to(device)
        visible = batch['visible_masks'].to(device)
        amodal = batch['amodal_masks'].to(device)
        camera_vecs = batch['camera_vecs'].to(device) if use_camera_cond else None
        _B, T = frames.shape[:2]
        prev_state = None
        for t in range(T):
            prev_frame_t = torch.zeros_like(frames[:, 0]) if t == 0 else frames[:, t - 1]
            scene_state = trainer.scene_module(
                entity_names=batch['entity_names'][0],
                text_prompt=batch['text_prompts'][0],
                prev_state=prev_state,
                prev_frame=prev_frame_t,
                t_index=t,
                camera_context=camera_vecs,
            )
            metrics = trainer.evaluator.evaluate_scene(scene_state, visible[:, t], amodal[:, t], prev_state=prev_state)
            all_metrics.append(metrics)
            prev_state = scene_state.detach()
    return _mean_metrics(all_metrics)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg['train'].get('seed', 42))
    set_seed(seed)
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Ablation flags
    use_camera_cond = cfg.get('ablation', {}).get('use_camera_cond', True)
    print(f"[Ablation] use_camera_cond={use_camera_cond}")

    dataset = Phase65Dataset(
        root=cfg['data']['root'],
        image_size=cfg['data'].get('image_size', 256),
        mask_size=cfg['data'].get('mask_size', 64),
        num_frames=cfg['data'].get('num_frames', 16),
    )

    split_path = Path(cfg['data']['split_json'])
    split_obj = json.loads(split_path.read_text())
    train_ds = _make_subset(dataset, split_obj[cfg['data'].get('split_name', 'train')])
    val_name = cfg.get('benchmark', {}).get('val_split_name', 'val')
    val_ds = _make_subset(dataset, split_obj[val_name])

    print(f"[Data] train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg['train'].get('batch_size', 2), shuffle=True, collate_fn=phase65_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg['train'].get('batch_size', 2), shuffle=False, collate_fn=phase65_collate_fn)

    model = SceneModule(
        slot_dim=cfg['model'].get('slot_dim', 256),
        feat_dim=cfg['model'].get('feat_dim', 64),
        hidden_dim=cfg['model'].get('hidden_dim', 128),
        Hs=cfg['model'].get('Hs', 64),
        Ws=cfg['model'].get('Ws', 64),
        Hf=cfg['model'].get('Hf', 32),
        Wf=cfg['model'].get('Wf', 32),
    )
    trainer = Stage1Trainer(
        scene_module=model,
        device=device,
        lr=cfg['train'].get('lr_scene_stage1', 1e-4),
        lambda_vis=cfg['loss'].get('lambda_vis', 1.0),
        lambda_amo=cfg['loss'].get('lambda_amo', 1.0),
        lambda_temp=cfg['loss'].get('lambda_temp', 0.05),
        lambda_depth=cfg['loss'].get('lambda_depth', 0.1),
        grad_clip=cfg['train'].get('grad_clip', 1.0),
    )

    out_dir = Path(cfg.get('output_dir', 'outputs/phase65/ablation'))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1e9
    history = []
    for epoch in range(cfg['train'].get('stage1_epochs', 50)):
        trainer.scene_module.train()
        epoch_metrics = []
        for batch in train_loader:
            frames = batch['frames'].to(device)
            visible = batch['visible_masks'].to(device)
            amodal = batch['amodal_masks'].to(device)
            camera_vecs = batch['camera_vecs'].to(device) if use_camera_cond else None
            _B, T = frames.shape[:2]
            prev_state = None
            for t in range(T):
                prev_frame_t = torch.zeros_like(frames[:, 0]) if t == 0 else frames[:, t - 1]
                step_batch = Stage1Batch(
                    entity_names=batch['entity_names'][0],
                    text_prompt=batch['text_prompts'][0],
                    prev_frame=prev_frame_t,
                    gt_visible=visible[:, t],
                    gt_amodal=amodal[:, t],
                    gt_front_idx=None,
                    t_index=t,
                    camera_context=camera_vecs,
                )
                metrics, prev_state = trainer.step(step_batch, prev_state=prev_state)
                epoch_metrics.append(metrics)

        train_mean = _mean_metrics(epoch_metrics)
        val_mean = evaluate_epoch(trainer, val_loader, device=device, use_camera_cond=use_camera_cond)
        merged = {f'train_{k}': v for k, v in train_mean.items()}
        merged.update({f'val_{k}': v for k, v in val_mean.items()})
        merged['epoch'] = epoch + 1
        history.append(merged)
        score = trainer.evaluator.score_for_checkpoint(val_mean)

        payload = {
            'model': model.state_dict(),
            'config': cfg,
            'metrics': merged,
            'history': history,
            'split_file': str(split_path),
            'score': score,
        }
        torch.save(payload, out_dir / 'latest_scene_module.pt')
        if score > best_score:
            best_score = score
            torch.save(payload, out_dir / 'best_scene_module.pt')

        (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
        print(f"[epoch {epoch + 1}] val_score={score:.4f} entity_balance={val_mean.get('entity_balance', 0.0):.4f} survival_min={val_mean.get('visible_survival_min', 0.0):.4f}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
