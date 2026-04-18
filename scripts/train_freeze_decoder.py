"""Freeze scene module and train a small CNN decoder to reconstruct RGB."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml

from phase65_min3d.data import Phase65Dataset, phase65_collate_fn
from phase65_min3d.scene_module import SceneModule


class SmallRGBDecoder(nn.Module):
    """Small CNN decoder to reconstruct RGB from scene module outputs."""

    def __init__(self, in_channels: int = 6, hidden_dim: int = 64, out_size: int = 256):
        super().__init__()
        self.out_size = out_size

        # Input: concatenated visible_e0, visible_e1, amodal_e0, amodal_e1 (4 channels)
        # Or just visible masks (2 channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, mask_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask_maps: (B, C, H, W) concatenated mask predictions
        Returns:
            rgb: (B, 3, out_size, out_size) reconstructed RGB
        """
        x = self.encoder(mask_maps)
        x = self.decoder(x)
        # Upsample to output size
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
        return x


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_subset(dataset: Phase65Dataset, ids: list[str]) -> Subset:
    id_to_idx = {p.name: i for i, p in enumerate(dataset.samples)}
    indices = [id_to_idx[x] for x in ids if x in id_to_idx]
    return Subset(dataset, indices)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Scene module checkpoint to freeze')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/phase65/freeze_decoder')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg['train'].get('seed', 42))
    set_seed(seed)
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Load frozen scene module
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    scene_module = SceneModule(
        slot_dim=cfg['model'].get('slot_dim', 256),
        feat_dim=cfg['model'].get('feat_dim', 64),
        hidden_dim=cfg['model'].get('hidden_dim', 128),
        Hs=cfg['model'].get('Hs', 64),
        Ws=cfg['model'].get('Ws', 64),
        Hf=cfg['model'].get('Hf', 32),
        Wf=cfg['model'].get('Wf', 32),
    )
    scene_module.load_state_dict(ckpt['model'])
    scene_module = scene_module.to(device)
    scene_module.eval()
    for p in scene_module.parameters():
        p.requires_grad = False
    print(f"Loaded and froze scene module from {args.checkpoint}")

    # Create decoder
    # Input: visible_e0, visible_e1, amodal_e0, amodal_e1 = 4 channels
    decoder = SmallRGBDecoder(in_channels=4, hidden_dim=64, out_size=256).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    # Data
    dataset = Phase65Dataset(
        root=cfg['data']['root'],
        image_size=cfg['data'].get('image_size', 256),
        mask_size=cfg['data'].get('mask_size', 64),
        num_frames=cfg['data'].get('num_frames', 16),
    )
    split_path = Path(cfg['data']['split_json'])
    split_obj = json.loads(split_path.read_text())
    train_ds = _make_subset(dataset, split_obj[cfg['data'].get('split_name', 'train')])
    val_ds = _make_subset(dataset, split_obj.get('val', split_obj.get('test', [])))

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=phase65_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=phase65_collate_fn)
    print(f"Data: train={len(train_ds)}, val={len(val_ds)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_loss = float('inf')

    for epoch in range(args.epochs):
        decoder.train()
        train_losses = []

        for batch in train_loader:
            frames = batch['frames'].to(device)
            camera_vecs = batch['camera_vecs'].to(device)
            _B, T = frames.shape[:2]

            # Process each timestep
            prev_state = None
            for t in range(T):
                prev_frame_t = torch.zeros_like(frames[:, 0]) if t == 0 else frames[:, t - 1]
                gt_frame = frames[:, t]  # (B, 3, H, W)

                with torch.no_grad():
                    scene_state = scene_module(
                        entity_names=batch['entity_names'][0],
                        text_prompt=batch['text_prompts'][0],
                        prev_state=prev_state,
                        prev_frame=prev_frame_t,
                        t_index=t,
                        camera_context=camera_vecs,
                    )
                    prev_state = scene_state.detach()

                # Concatenate mask predictions (maps are already probabilities)
                mask_input = torch.cat([
                    scene_state.maps.visible_e0,
                    scene_state.maps.visible_e1,
                    scene_state.maps.amodal_e0,
                    scene_state.maps.amodal_e1,
                ], dim=1)  # (B, 4, Hs, Ws)

                # Decode to RGB
                pred_rgb = decoder(mask_input)  # (B, 3, 256, 256)

                # Reconstruction loss
                loss = F.mse_loss(pred_rgb, gt_frame)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

        # Validation
        decoder.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(device)
                camera_vecs = batch['camera_vecs'].to(device)
                _B, T = frames.shape[:2]

                prev_state = None
                for t in range(T):
                    prev_frame_t = torch.zeros_like(frames[:, 0]) if t == 0 else frames[:, t - 1]
                    gt_frame = frames[:, t]

                    scene_state = scene_module(
                        entity_names=batch['entity_names'][0],
                        text_prompt=batch['text_prompts'][0],
                        prev_state=prev_state,
                        prev_frame=prev_frame_t,
                        t_index=t,
                        camera_context=camera_vecs,
                    )
                    prev_state = scene_state.detach()

                    mask_input = torch.cat([
                        scene_state.maps.visible_e0,
                        scene_state.maps.visible_e1,
                        scene_state.maps.amodal_e0,
                        scene_state.maps.amodal_e1,
                    ], dim=1)

                    pred_rgb = decoder(mask_input)
                    loss = F.mse_loss(pred_rgb, gt_frame)
                    val_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
        })

        print(f"[epoch {epoch + 1}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # Save
        torch.save({
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'history': history,
        }, out_dir / 'latest_decoder.pt')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'decoder': decoder.state_dict(),
                'epoch': epoch + 1,
                'val_loss': val_loss,
            }, out_dir / 'best_decoder.pt')

        (out_dir / 'history.json').write_text(json.dumps(history, indent=2))

    print(f"Training complete. Best val_loss: {best_loss:.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
