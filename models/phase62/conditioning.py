"""
Phase 62 — Entity-Conditioned Guide Assembly + UNet Injection
==============================================================

GuideFeatureAssembler: builds dual-stream 2D feature guides from first-hit
projection:

    front layer: visible winner stream
    back layer:  occluded / behind-front stream

    F_front(h,w) = F_n(h,w)  where n = visible_class(h,w)
    F_back(h,w)  = weighted feature mixture for entities behind the winner

Straight-through: hard selection forward, soft gradient backward via
visible_probs (differentiable through the volume softmax).

GuideInjectionManager: registers forward hooks on UNet blocks to inject
the assembled dual-stream guide features via spatial addition.

Supports injection configs:
  - mid_only:   inject at mid_block only
  - mid_up2:    mid_block + up_blocks.2
  - multiscale: mid + up_blocks.1 + up_blocks.2 + up_blocks.3
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# SD1.5 AnimateDiff block channel dims
BLOCK_DIMS: Dict[str, int] = {
    "mid":  1280,
    "up1":  1280,
    "up2":  640,
    "up3":  320,
}

# Spatial resolution per block (at 256x256 input, latent 32x32)
# mid: 4x4, up1: 8x8, up2: 16x16, up3: 32x32
BLOCK_SPATIAL: Dict[str, tuple] = {
    "mid":  (4, 4),
    "up1":  (8, 8),
    "up2":  (16, 16),
    "up3":  (32, 32),
}

# Injection config presets
INJECT_CONFIGS: Dict[str, List[str]] = {
    "mid_only":   ["mid"],
    "mid_up2":    ["mid", "up2"],
    "multiscale": ["mid", "up1", "up2", "up3"],
}


class GuideFeatureAssembler(nn.Module):
    """
    Assemble dual-stream 2D feature guide from first-hit visible class selection.

    Instead of just embedding class indices, this selects the actual
    entity-conditioned features at each pixel based on which entity
    is visible (from first-hit projection).

    Front stream uses straight-through first-hit selection.
    Back stream uses differentiable soft mass from occluded / behind-front bins.

    Shape flow:
        F_g, F_0, F_1: (B, S, D) where S = H * W
        -> proj: (B, S, hidden) -> reshape (B, hidden, H, W)
        -> front layer (B, hidden, H, W)
        -> back layer  (B, hidden, H, W)
        -> concat to guide_base (B, hidden*2, H, W)
        -> per-block projection to (B, block_dim, H_block, W_block)
    """

    def __init__(
        self,
        feat_dim: int = 640,
        hidden: int = 64,
        spatial_h: int = 16,
        spatial_w: int = 16,
        n_classes: int = 3,
        inject_config: str = "mid_up2",
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.n_classes = n_classes

        # Feature projectors: one per stream
        self.proj_g = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
        )
        self.proj_e0 = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
        )
        self.proj_e1 = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
        )

        # Per-block output projection: hidden*2 -> block_dim
        block_names = INJECT_CONFIGS.get(inject_config, ["mid", "up2"])
        self.block_names = block_names

        self.block_projectors = nn.ModuleDict()
        for block_name in block_names:
            block_dim = BLOCK_DIMS[block_name]
            self.block_projectors[block_name] = nn.Sequential(
                nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden * 2, block_dim, kernel_size=1, bias=True),
            )

        # Zero-init output layers for smooth training start
        for block_name in block_names:
            proj = self.block_projectors[block_name]
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)

    def forward(
        self,
        visible_class: torch.Tensor,  # (B, H, W) int64
        front_probs: torch.Tensor,    # (B, C, H, W) float — straight-through first-hit probs
        back_probs: torch.Tensor,     # (B, C, H, W) float — behind-front soft probs
        F_g: torch.Tensor,            # (B, S, D)
        F_0: torch.Tensor,            # (B, S, D)
        F_1: torch.Tensor,            # (B, S, D)
    ) -> Dict[str, torch.Tensor]:
        """
        Assemble entity-conditioned guide features.

        For each pixel, select the projected feature of the visible entity.
        Uses straight-through estimation: hard selection in forward pass,
        soft probs provide gradient backward through the volume predictor.

        Args:
            visible_class: (B, H, W) int64 — hard class index (0=bg, 1=e0, 2=e1)
            visible_probs: (B, C, H, W) float — differentiable softmax probs
            F_g:  (B, S, D) — global cross-attention features
            F_0:  (B, S, D) — entity-0 slot features
            F_1:  (B, S, D) — entity-1 slot features

        Returns:
            dict[block_name -> (B, block_dim, H_block, W_block)] guide features
        """
        B = F_g.shape[0]
        H, W = self.spatial_h, self.spatial_w
        device = F_g.device

        # 1. Project each stream: (B, S, D) -> (B, S, hidden) -> (B, hidden, H, W)
        h_g = self.proj_g(F_g.float())     # (B, S, hidden)
        h_e0 = self.proj_e0(F_0.float())   # (B, S, hidden)
        h_e1 = self.proj_e1(F_1.float())   # (B, S, hidden)

        h_g = h_g.permute(0, 2, 1).reshape(B, self.hidden, H, W)     # (B, hidden, H, W)
        h_e0 = h_e0.permute(0, 2, 1).reshape(B, self.hidden, H, W)   # (B, hidden, H, W)
        h_e1 = h_e1.permute(0, 2, 1).reshape(B, self.hidden, H, W)   # (B, hidden, H, W)

        # 2. Stack features: (B, 3, hidden, H, W) — class 0=bg, 1=e0, 2=e1
        feat_stack = torch.stack([h_g, h_e0, h_e1], dim=1)  # (B, 3, hidden, H, W)

        # 3. Front stream: hard-forward / soft-backward first-hit.
        front_layer = (feat_stack * front_probs.unsqueeze(2)).sum(dim=1)  # (B, hidden, H, W)

        # 4. Back stream: soft mixture of occluded / behind-front entities.
        back_layer = (feat_stack * back_probs.unsqueeze(2)).sum(dim=1)  # (B, hidden, H, W)

        guide_base = torch.cat([front_layer, back_layer], dim=1)  # (B, hidden*2, H, W)

        # 5. Project to each block's resolution and dim
        guides: Dict[str, torch.Tensor] = {}
        for block_name in self.block_names:
            proj = self.block_projectors[block_name]
            h_block, w_block = BLOCK_SPATIAL[block_name]

            if h_block != H or w_block != W:
                guide_resized = F.interpolate(
                    guide_base,
                    size=(h_block, w_block),
                    mode="bilinear",
                    align_corners=False,
                )  # (B, hidden, h_block, w_block)
            else:
                guide_resized = guide_base

            guides[block_name] = proj(guide_resized)  # (B, block_dim, h_block, w_block)

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
        # Video: (B_hs, C, T, H, W)
        B_hs = hidden_states.shape[0]
        T = hidden_states.shape[2]
        H_block, W_block = hidden_states.shape[3], hidden_states.shape[4]
        # Spatial resize if needed
        if guide.shape[2] != H_block or guide.shape[3] != W_block:
            guide = F.interpolate(guide, size=(H_block, W_block), mode='nearest')
        # Batch broadcast: CFG doubles the batch (uncond + cond)
        if guide.shape[0] != B_hs:
            guide = guide.repeat(B_hs // max(guide.shape[0], 1), 1, 1, 1)
        guide_5d = guide.unsqueeze(2).expand(-1, -1, T, -1, -1)  # (B, C, T, H, W)
        return hidden_states + guide_5d
    else:
        # Image: (B_hs, C, H, W)
        B_hs = hidden_states.shape[0]
        H_block, W_block = hidden_states.shape[2], hidden_states.shape[3]
        if guide.shape[2] != H_block or guide.shape[3] != W_block:
            guide = F.interpolate(guide, size=(H_block, W_block), mode='nearest')
        if guide.shape[0] != B_hs:
            guide = guide.repeat(B_hs // max(guide.shape[0], 1), 1, 1, 1)
        return hidden_states + guide


class GuideInjectionManager:
    """
    Manages forward hooks on UNet blocks for guide injection.

    Usage:
        mgr = GuideInjectionManager('mid_up2')
        mgr.register_hooks(unet)
        mgr.set_guides(guides_dict)    # before UNet forward
        output = unet(...)             # hooks inject guides
        mgr.clear_guides()             # after forward
        mgr.remove_hooks()             # cleanup
    """

    def __init__(self, inject_config: str = "mid_up2"):
        self.inject_config = inject_config
        self.block_names = INJECT_CONFIGS.get(inject_config, ["mid", "up2"])
        self._hooks: list = []
        self._guides: Dict[str, torch.Tensor] = {}

    def set_guides(self, guides: Dict[str, torch.Tensor]) -> None:
        """Set guide features to inject at next forward pass."""
        self._guides = guides

    def clear_guides(self) -> None:
        """Clear stored guides (no injection on next forward)."""
        self._guides = {}

    def _make_hook(self, block_name: str):
        """Create a forward hook that adds the guide to block output."""
        def hook_fn(module, input, output):
            if block_name not in self._guides:
                return output
            guide = self._guides[block_name]

            if isinstance(output, tuple):
                h = output[0]
                h = inject_guide_into_unet_features(h, guide)
                return (h,) + output[1:]
            else:
                return inject_guide_into_unet_features(output, guide)
        return hook_fn

    def register_hooks(self, unet) -> None:
        """Register forward hooks on appropriate UNet blocks."""
        block_map = {
            "mid": unet.mid_block,
        }
        for i, up_block in enumerate(unet.up_blocks):
            block_map[f"up{i}"] = up_block

        for block_name in self.block_names:
            if block_name in block_map and block_map[block_name] is not None:
                h = block_map[block_name].register_forward_hook(
                    self._make_hook(block_name))
                self._hooks.append(h)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
