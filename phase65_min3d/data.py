from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Phase65Sample:
    frames: torch.Tensor           # (T, 3, H, W)
    visible_masks: torch.Tensor    # (T, 2, Hm, Wm)
    amodal_masks: torch.Tensor     # (T, 2, Hm, Wm)
    depth_maps: torch.Tensor       # (T, 1, Hd, Wd)
    entity_names: tuple[str, str]
    text_prompt: str
    clip_type: str


class Phase65Dataset(Dataset):
    """Filesystem dataset for Phase 65 synthetic clips.

    Expected layout per sample directory:
      sample_xxxxx/
        meta.json
        frames/0000.png ...
        visible_masks/0000_e0.png, 0000_e1.png ...
        amodal_masks/0000_e0.png, 0000_e1.png ...
        depth/0000.npy ...
    """

    def __init__(self, root: str, image_size: int = 256, mask_size: int = 64, num_frames: int = 16):
        self.root = Path(root)
        self.image_size = image_size
        self.mask_size = mask_size
        self.num_frames = num_frames
        self.samples = sorted([p for p in self.root.iterdir() if p.is_dir()])
        if not self.samples:
            raise FileNotFoundError(f"No sample directories found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, path: Path, size: int) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _read_mask(self, path: Path, size: int) -> torch.Tensor:
        img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _read_depth(self, path: Path, size: int) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            pass
        elif arr.ndim == 3:
            arr = arr[..., 0]
        img = Image.fromarray(arr)
        img = img.resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx: int) -> Phase65Sample:
        sample_dir = self.samples[idx]
        meta = json.loads((sample_dir / "meta.json").read_text())
        T = min(int(meta.get("num_frames", self.num_frames)), self.num_frames)
        frames, visible, amodal, depth = [], [], [], []
        for t in range(T):
            stem = f"{t:04d}"
            frames.append(self._read_image(sample_dir / "frames" / f"{stem}.png", self.image_size))
            visible.append(torch.cat([
                self._read_mask(sample_dir / "visible_masks" / f"{stem}_e0.png", self.mask_size),
                self._read_mask(sample_dir / "visible_masks" / f"{stem}_e1.png", self.mask_size),
            ], dim=0))
            amodal.append(torch.cat([
                self._read_mask(sample_dir / "amodal_masks" / f"{stem}_e0.png", self.mask_size),
                self._read_mask(sample_dir / "amodal_masks" / f"{stem}_e1.png", self.mask_size),
            ], dim=0))
            depth.append(self._read_depth(sample_dir / "depth" / f"{stem}.npy", self.mask_size))
        return Phase65Sample(
            frames=torch.stack(frames, dim=0),
            visible_masks=torch.stack(visible, dim=0),
            amodal_masks=torch.stack(amodal, dim=0),
            depth_maps=torch.stack(depth, dim=0),
            entity_names=tuple(meta["entity_names"]),
            text_prompt=meta.get("text_prompt", f"{meta['entity_names'][0]} and {meta['entity_names'][1]}"),
            clip_type=meta.get("clip_type", "unknown"),
        )


def phase65_collate_fn(batch: List[Phase65Sample]) -> Dict[str, torch.Tensor | list | tuple]:
    frames = torch.stack([b.frames for b in batch], dim=0)
    visible = torch.stack([b.visible_masks for b in batch], dim=0)
    amodal = torch.stack([b.amodal_masks for b in batch], dim=0)
    depth = torch.stack([b.depth_maps for b in batch], dim=0)
    entity_names = [b.entity_names for b in batch]
    text_prompts = [b.text_prompt for b in batch]
    clip_types = [b.clip_type for b in batch]
    return {
        "frames": frames,
        "visible_masks": visible,
        "amodal_masks": amodal,
        "depth_maps": depth,
        "entity_names": entity_names,
        "text_prompts": text_prompts,
        "clip_types": clip_types,
    }
