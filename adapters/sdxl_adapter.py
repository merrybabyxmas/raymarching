"""adapters/sdxl_adapter.py
===========================
Thin adapter for Stable Diffusion XL (SDXL) UNet.

Same architecture and injection logic as :class:`~adapters.animatediff_adapter.AnimateDiffAdapter`
but with SDXL-specific block names, channel dimensions, and spatial resolutions.

SDXL UNet structure (512×512 base resolution):
    mid_block       : 1280 ch,  8× 8 spatial
    up_blocks[0]    : 1280 ch, 16×16 spatial
    up_blocks[1]    :  640 ch, 32×32 spatial
    up_blocks[2]    :  320 ch, 64×64 spatial
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters.base_adapter import BaseBackboneAdapter
from adapters.animatediff_adapter import inject_guide_into_unet_features


# ---------------------------------------------------------------------------
# Block layout
# ---------------------------------------------------------------------------

SDXL_BLOCK_DIMS: dict[str, int] = {
    "mid":  1280,
    "up0":  1280,
    "up1":   640,
    "up2":   320,
}

SDXL_BLOCK_SPATIAL: dict[str, tuple[int, int]] = {
    "mid":  (8,  8),
    "up0":  (16, 16),
    "up1":  (32, 32),
    "up2":  (64, 64),
}

DEFAULT_INJECT_BLOCKS: tuple[str, ...] = ("up0", "up1", "up2")


# ---------------------------------------------------------------------------
# SDXLAdapter
# ---------------------------------------------------------------------------

class SDXLAdapter(nn.Module, BaseBackboneAdapter):
    """Thin adapter: scene features (B, in_ch, H, W) → SDXL UNet conditioning.

    Uses forward hooks on SDXL UNet blocks.  Gate initialised to zeros so the
    adapter starts as identity and learns to use the scene prior.

    Args:
        in_ch:           Input channels from SceneGuideEncoder (default: 64).
        inject_blocks:   Which SDXL blocks to inject into.
        guide_max_ratio: Maximum guide amplitude as fraction of block std.
    """

    def __init__(
        self,
        in_ch: int = 64,
        inject_blocks: tuple[str, ...] = DEFAULT_INJECT_BLOCKS,
        guide_max_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.inject_blocks   = list(inject_blocks)
        self.guide_max_ratio = float(guide_max_ratio)

        # Per-block projectors: scene_features → (B, block_dim, H_b, W_b)
        self.projectors = nn.ModuleDict()
        for block_name in self.inject_blocks:
            block_dim = SDXL_BLOCK_DIMS[block_name]
            self.projectors[block_name] = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Conv2d(in_ch, block_dim, kernel_size=1, bias=True),
            )

        # Per-block learnable gates initialised to 0 → tanh(0) = 0 (identity).
        self.gates = nn.ParameterDict({
            bn: nn.Parameter(torch.zeros(1)) for bn in self.inject_blocks
        })

        # State
        self._hooks:  list = []
        self._guides: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # BaseBackboneAdapter implementation
    # ------------------------------------------------------------------

    def build_guides(
        self, scene_features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Project scene features to per-block guide tensors.

        Args:
            scene_features: (B, in_ch, H, W) from SceneGuideEncoder.

        Returns:
            Dict mapping block name → (B, block_dim, H_b, W_b) guide tensor.
        """
        guides: dict[str, torch.Tensor] = {}
        for block_name in self.inject_blocks:
            h_b, w_b = SDXL_BLOCK_SPATIAL[block_name]
            if scene_features.shape[2] != h_b or scene_features.shape[3] != w_b:
                feat = F.interpolate(
                    scene_features, size=(h_b, w_b), mode="bilinear", align_corners=False
                )
            else:
                feat = scene_features
            guides[block_name] = self.projectors[block_name](feat)
        self._guides = guides
        return guides

    def register_hooks(self, unet) -> None:
        """Register forward hooks on the SDXL UNet mid/up blocks.

        Args:
            unet: SDXL UNet model (must have ``mid_block`` and ``up_blocks``).
        """
        self.remove_hooks()

        block_map: dict[str, object] = {"mid": unet.mid_block}
        for i, up_block in enumerate(unet.up_blocks):
            block_map[f"up{i}"] = up_block

        for block_name in self.inject_blocks:
            module = block_map.get(block_name)
            if module is None:
                continue
            hook = module.register_forward_hook(self._make_hook(block_name))
            self._hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def inject(self, h: torch.Tensor, block_name: str) -> torch.Tensor:
        """Directly inject guide for *block_name* into feature map *h*.

        Args:
            h:          (B, C, H, W) feature tensor.
            block_name: Block name key; must exist in current guides dict.

        Returns:
            Injected tensor, same shape as *h*.
        """
        if block_name not in self._guides:
            return h
        guide = self._guides[block_name]
        gate  = torch.tanh(self.gates[block_name])
        return inject_guide_into_unet_features(
            h, guide, gate=gate, max_ratio=self.guide_max_ratio
        )

    # ------------------------------------------------------------------
    # Guide management helpers
    # ------------------------------------------------------------------

    def set_guides(self, guides: dict[str, torch.Tensor]) -> None:
        """Manually set guides."""
        self._guides = guides

    def clear_guides(self) -> None:
        """Clear stored guides."""
        self._guides = {}

    # ------------------------------------------------------------------
    # Hook factory
    # ------------------------------------------------------------------

    def _make_hook(self, block_name: str):
        def hook_fn(module, input, output):
            if block_name not in self._guides:
                return output
            guide = self._guides[block_name]
            gate  = torch.tanh(self.gates[block_name])
            if isinstance(output, tuple):
                h_out = output[0]
                h_out = inject_guide_into_unet_features(
                    h_out, guide, gate=gate, max_ratio=self.guide_max_ratio
                )
                return (h_out,) + output[1:]
            return inject_guide_into_unet_features(
                output, guide, gate=gate, max_ratio=self.guide_max_ratio
            )
        return hook_fn
