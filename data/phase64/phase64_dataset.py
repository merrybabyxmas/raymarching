"""
data/phase64/phase64_dataset.py
================================
Phase 64 dataset: wraps Phase62DatasetAdapter and adds SceneGT + routing maps.

Routing maps are precomputed from frame RGB + entity color hints.
They serve as the primary spatial signal for the backbone-agnostic EntityField.

The dataset returns Phase64Sample dataclass objects, which contain:
  - raw frames
  - SceneGT (visible/amodal masks, depth, split type)
  - per-entity routing color maps
  - metadata
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.phase62.dataset_adapter import Phase62DatasetAdapter
from data.phase64.build_scene_gt import SceneGT, build_scene_gt
from data.phase64.phase64_splits import SplitType, make_splits
from scripts.generate_solo_renders import ObjaverseDatasetPhase40


# --------------------------------------------------------------------------- #
#  Sample dataclass
# --------------------------------------------------------------------------- #

@dataclass
class Phase64Sample:
    """A single Phase 64 sample.

    Fields
    ------
    idx         : dataset index
    frames      : (T, H, W, 3) uint8  composite frames
    scene_gt    : SceneGT  per-frame annotations at (spatial_h, spatial_w)
    meta        : metadata dict (keyword0, keyword1, prompt, …)
    routing_e0  : (T, H, W) float32  colour routing map for entity 0
    routing_e1  : (T, H, W) float32  colour routing map for entity 1
    """
    idx: int
    frames: np.ndarray          # (T, H, W, 3) uint8
    scene_gt: SceneGT
    meta: dict
    routing_e0: np.ndarray      # (T, H, W) float32
    routing_e1: np.ndarray      # (T, H, W) float32


# --------------------------------------------------------------------------- #
#  Routing map builder
# --------------------------------------------------------------------------- #

# Default entity slot colours (must match scene_prior/entity_parser.py defaults)
_ENTITY_COLORS = [
    np.array([0.85, 0.15, 0.10], dtype=np.float32),   # entity 0 → red
    np.array([0.10, 0.20, 0.85], dtype=np.float32),   # entity 1 → blue
]


def _build_routing_map(
    frames: np.ndarray,   # (T, H, W, 3) uint8
    entity_color: np.ndarray,   # (3,) float32  in [0, 1]
    spatial_h: int,
    spatial_w: int,
    sigma: float = 0.12,
) -> np.ndarray:
    """
    Compute a per-entity routing map from frame colours.

    For each pixel, we measure colour affinity to the entity colour using a
    Gaussian kernel in RGB space.  The map is normalised per-frame to [0, 1].

    Returns: (T, spatial_h, spatial_w) float32
    """
    T, H, W, _ = frames.shape
    routing = np.zeros((T, spatial_h, spatial_w), dtype=np.float32)

    for t in range(T):
        frame = frames[t].astype(np.float32) / 255.0  # (H, W, 3)
        diff = frame - entity_color[None, None, :]     # (H, W, 3)
        dist2 = (diff ** 2).sum(axis=-1)               # (H, W)
        affinity = np.exp(-dist2 / (2.0 * sigma ** 2)) # (H, W)

        # Resize to spatial resolution
        aff_img = Image.fromarray((affinity * 255).astype(np.uint8))
        aff_small = aff_img.resize((spatial_w, spatial_h), Image.BILINEAR)
        aff_np = np.array(aff_small, dtype=np.float32) / 255.0

        routing[t] = aff_np

    return routing


def _get_entity_color(meta: dict, entity_idx: int) -> np.ndarray:
    """Extract entity colour from metadata or fall back to defaults."""
    from scene_prior.entity_parser import parse_prompt, _COLOR_MAP

    prompt = meta.get("prompt", "")
    if prompt:
        try:
            sp = parse_prompt(prompt)
            if entity_idx < len(sp.entities):
                color = sp.entities[entity_idx].color
                return np.array(color, dtype=np.float32)
        except Exception:
            pass

    return _ENTITY_COLORS[entity_idx].copy()


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #

class Phase64Dataset(Dataset):
    """
    Phase 64 dataset.

    Wraps Phase62DatasetAdapter and enriches each sample with:
      1. SceneGT  — visible/amodal/depth annotations at (spatial_h, spatial_w)
      2. Routing maps — per-entity colour affinity maps at (spatial_h, spatial_w)

    These together form the supervision signal for the backbone-agnostic
    EntityField trained in Stage 1.

    Parameters
    ----------
    data_root       : path to toy/data_objaverse (or similar ObjaverseDatasetPhase40 root)
    n_frames        : number of frames per clip (passed to Phase62DatasetAdapter)
    spatial_h       : height of all spatial GT fields
    spatial_w       : width of all spatial GT fields
    split_indices   : if given, only expose these dataset indices
    """

    def __init__(
        self,
        data_root: str,
        n_frames: int = 8,
        spatial_h: int = 32,
        spatial_w: int = 32,
        split_indices: Optional[List[int]] = None,
    ) -> None:
        self.adapter = Phase62DatasetAdapter(data_root, n_frames=n_frames)
        self.n_frames = n_frames
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w

        # Index mapping: dataset position → underlying adapter index
        if split_indices is not None:
            self._indices = list(split_indices)
        else:
            self._indices = list(range(len(self.adapter)))

        # Lazy split cache — populated on first call to get_split_indices()
        self._split_cache: Optional[Dict[str, List[int]]] = None

    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Phase64Sample:
        """Return a Phase64Sample for position *idx* in this dataset."""
        underlying_idx = self._indices[idx]
        raw = self.adapter[underlying_idx]

        frames: np.ndarray = raw["frames"]       # (T, H, W, 3)
        meta: dict = raw.get("meta", {})

        # Build SceneGT
        scene_gt = build_scene_gt(raw, spatial_h=self.spatial_h, spatial_w=self.spatial_w)

        # Build routing maps from frame colours
        color_e0 = _get_entity_color(meta, 0)
        color_e1 = _get_entity_color(meta, 1)

        routing_e0 = _build_routing_map(
            frames, color_e0, self.spatial_h, self.spatial_w
        )
        routing_e1 = _build_routing_map(
            frames, color_e1, self.spatial_h, self.spatial_w
        )

        return Phase64Sample(
            idx=underlying_idx,
            frames=frames,
            scene_gt=scene_gt,
            meta=meta,
            routing_e0=routing_e0,
            routing_e1=routing_e1,
        )

    # ---------------------------------------------------------------------- #
    #  Accessors
    # ---------------------------------------------------------------------- #

    def raw_dataset(self) -> ObjaverseDatasetPhase40:
        """Return the underlying ObjaverseDatasetPhase40 instance."""
        return self.adapter.raw_dataset()

    def get_split_indices(self) -> Dict[str, List[int]]:
        """Compute and cache O/C/R/X split indices for this dataset.

        Note: indices are in the *Phase64Dataset* index space (not the
        underlying adapter index space).
        """
        if self._split_cache is None:
            self._split_cache = make_splits(self)
        return self._split_cache
