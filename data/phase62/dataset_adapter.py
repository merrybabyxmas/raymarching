"""
Phase 62 — Dataset Adapter
============================

Wraps ObjaverseDatasetPhase40 for Phase62 training.
Provides a clean interface returning structured sample dicts.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from scripts.generate_solo_renders import ObjaverseDatasetPhase40


class Phase62DatasetAdapter:
    """
    Wraps ObjaverseDatasetPhase40 for Phase62 training.

    Returns dict-based samples with all fields needed by the trainer.

    Usage:
        adapter = Phase62DatasetAdapter('toy/data_objaverse', n_frames=8)
        sample = adapter[idx]
        # sample keys: frames, depth, depth_orders, meta, entity_masks,
        #              visible_masks, solo_e0, solo_e1
    """

    def __init__(self, data_root: str, n_frames: int = 8):
        self.dataset = ObjaverseDatasetPhase40(data_root, n_frames=n_frames)

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_sample_dir(self, idx: int):
        return self.dataset.samples[idx]["dir"]

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns dict with:
            frames:        (T, H, W, 3) uint8
            depth:         (T, H, W) float32
            depth_orders:  list of (front_idx, back_idx) per frame
            meta:          dict with keyword0, keyword1, prompt etc.
            entity_masks:  (T, 2, S) float32  where S = H_mask * W_mask
            visible_masks: (T, 2, S) float32
            solo_e0:       (T, H, W, 3) uint8 or None
            solo_e1:       (T, H, W, 3) uint8 or None
        """
        sample = self.dataset[idx]

        # ObjaverseDatasetPhase40 returns 8-tuple:
        # frames, depths, depth_orders, meta, entity_masks, visible_masks, solo_e0, solo_e1
        if len(sample) >= 8:
            frames_np, depth_np, depth_orders, meta, entity_masks, \
                visible_masks, solo_e0, solo_e1 = sample
        else:
            # Fallback for older 5-tuple datasets
            frames_np, depth_np, depth_orders, meta, entity_masks = sample[:5]
            visible_masks = None
            solo_e0 = None
            solo_e1 = None

        return {
            "frames": frames_np,
            "depth": depth_np,
            "depth_orders": depth_orders,
            "meta": meta,
            "sample_dir": str(self._get_sample_dir(idx)),
            "entity_masks": entity_masks,
            "visible_masks": visible_masks,
            "solo_e0": solo_e0,
            "solo_e1": solo_e1,
        }

    @property
    def samples(self):
        """Proxy to underlying dataset samples list (for collision augment)."""
        return getattr(self.dataset, "samples", [])

    def raw_dataset(self) -> ObjaverseDatasetPhase40:
        """Return the underlying ObjaverseDatasetPhase40 instance."""
        return self.dataset
