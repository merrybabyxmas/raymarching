"""
Phase 62 — Volume-Guided UNet Injection
=========================================

Projects the 2D visible class map into spatial guide features and
injects them into UNet blocks via spatial addition.

Supports ablation configs:
  - mid_only:   inject at mid_block only
  - mid_up2:    mid_block + up_blocks.2
  - multiscale: mid + up_blocks.1 + up_blocks.2 + up_blocks.3

Straight-through gradient flow: hard class for embedding lookup,
multiplied by soft probs so gradients flow through the volume predictor.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# SD1.5 AnimateDiff block channel dims
BLOCK_DIMS = {
    "mid":  1280,
    "up1":  1280,
    "up2":  640,
    "up3":  320,
}

# Spatial resolution per block (at 256x256 input, latent 32x32)
# mid: 4x4, up1: 8x8, up2: 16x16, up3: 32x32
BLOCK_SPATIAL = {
    "mid":  (4, 4),
    "up1":  (8, 8),
    "up2":  (16, 16),
    "up3":  (32, 32),
}

# Injection config presets
INJECT_CONFIGS = {
    "mid_only":   ["mid"],
    "mid_up2":    ["mid", "up2"],
    "multiscale": ["mid", "up1", "up2", "up3"],
}


class VolumeGuidedInjector(nn.Module):
    """
    Injects projected 2D entity-class guide into UNet blocks.

    For each injection point, produces a spatial feature map from
    the projected visible classes and adds it to the UNet hidden states.
    """

    def __init__(
        self,
        n_classes: int = 3,
        hidden: int = 64,
        inject_config: str = "mid_up2",
    ):
        super().__init__()
        self.n_classes = n_classes
        self.hidden = hidden
        self.inject_config = inject_config

        block_names = INJECT_CONFIGS.get(inject_config, ["mid", "up2"])

        # Per-class learnable embeddings
        self.class_embed = nn.Embedding(n_classes, hidden)

        # Per-block projection: hidden -> block_dim
        self.projectors = nn.ModuleDict()
        for block_name in block_names:
            block_dim = BLOCK_DIMS[block_name]
            self.projectors[block_name] = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden, block_dim, kernel_size=1, bias=True),
            )

        # Zero-init output layers for smooth training start
        for block_name in block_names:
            proj = self.projectors[block_name]
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)

    def build_guide(
        self,
        visible_class: torch.Tensor,  # (B, H, W) int64
        visible_probs: torch.Tensor,  # (B, C, H, W) float
    ) -> Dict[str, torch.Tensor]:
        """
        Build 2D spatial guide features for each injection block.

        Uses straight-through: hard class for embedding lookup,
        but multiplied by soft probs for gradient flow through volume.

        Args:
            visible_class: (B, H, W) — hard class index per pixel
            visible_probs: (B, C, H, W) — differentiable softmax probs

        Returns:
            dict[block_name -> (B, block_dim, H_block, W_block)] guide features
        """
        B, H, W = visible_class.shape
        device = visible_class.device

        # --- Straight-through class embedding ---
        # Hard lookup: (B, H, W) -> (B, H, W, hidden)
        embed_hard = self.class_embed(visible_class)  # (B, H, W, hidden)
        embed_hard = embed_hard.permute(0, 3, 1, 2)   # (B, hidden, H, W)

        # Soft weighting for gradient flow:
        # For each pixel, weight the embedding by the probability of the
        # selected class. This creates the straight-through estimator:
        # forward uses hard class, backward flows through soft probs.
        # Gather the prob of the hard-selected class
        cls_idx = visible_class.unsqueeze(1)  # (B, 1, H, W)
        selected_prob = visible_probs.gather(dim=1, index=cls_idx)  # (B, 1, H, W)

        # Straight-through: detach hard embedding, add soft gradient path
        # guide_base = embed_hard.detach() * selected_prob + embed_hard * (1 - selected_prob.detach())
        # Simplified: multiply by prob (gradient flows through prob -> softmax -> V_logits)
        guide_base = embed_hard * selected_prob  # (B, hidden, H, W)

        # --- Project to each block's resolution and dim ---
        guides: Dict[str, torch.Tensor] = {}
        for block_name, proj in self.projectors.items():
            h_block, w_block = BLOCK_SPATIAL[block_name]

            # Resize guide to block spatial resolution
            if h_block != H or w_block != W:
                guide_resized = F.interpolate(
                    guide_base,
                    size=(h_block, w_block),
                    mode="bilinear",
                    align_corners=False,
                )  # (B, hidden, h_block, w_block)
            else:
                guide_resized = guide_base

            # Project to block dim: (B, block_dim, h_block, w_block)
            guides[block_name] = proj(guide_resized)

        return guides


def inject_guide_into_unet_features(
    hidden_states: torch.Tensor,   # (B, C_block, T, H_block, W_block) for video
    guide: torch.Tensor,           # (B, C_block, H_block, W_block)
) -> torch.Tensor:
    """
    Add guide features to UNet hidden states via spatial addition.

    Handles both 4D (image) and 5D (video) hidden states.
    For video: broadcasts guide across time dimension.
    """
    if hidden_states.dim() == 5:
        # Video: (B, C, T, H, W) — broadcast guide across T
        T = hidden_states.shape[2]
        H_block, W_block = hidden_states.shape[3], hidden_states.shape[4]
        # Resize guide to match block resolution if needed
        if guide.shape[2] != H_block or guide.shape[3] != W_block:
            guide = F.interpolate(guide, size=(H_block, W_block), mode='nearest')
        guide_5d = guide.unsqueeze(2).expand(-1, -1, T, -1, -1)  # (B, C, T, H, W)
        return hidden_states + guide_5d
    else:
        # Image: (B, C, H, W)
        H_block, W_block = hidden_states.shape[2], hidden_states.shape[3]
        if guide.shape[2] != H_block or guide.shape[3] != W_block:
            guide = F.interpolate(guide, size=(H_block, W_block), mode='nearest')
        return hidden_states + guide
