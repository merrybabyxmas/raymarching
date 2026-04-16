"""adapters/guide_encoders.py
=============================
Backbone-agnostic encoder that converts the canonical 8-channel
``SceneOutputs`` representation into a compact feature map.

The output of :class:`SceneGuideEncoder` is then fed to a backbone-specific
adapter (e.g. :class:`AnimateDiffAdapter`) which projects it further to match
the UNet block dimensions.

Input channel layout (fixed canonical order from ``SceneOutputs``):
    0  vis_e0    — visible contribution of entity 0
    1  vis_e1    — visible contribution of entity 1
    2  amo_e0    — amodal presence of entity 0
    3  amo_e1    — amodal presence of entity 1
    4  depth_map — expected depth in [0, 1]
    5  sep_map   — signed separation (vis_e0 - vis_e1) in [-1, 1]
    6  hidden_e0 — occluded fraction of entity 0
    7  hidden_e1 — occluded fraction of entity 1
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from scene_prior.scene_outputs import SceneOutputs


class SceneGuideEncoder(nn.Module):
    """Encode the canonical 8-channel scene representation into a dense feature map.

    Takes a ``SceneOutputs`` object (or a pre-stacked (B, 8, H, W) tensor) and
    produces a (B, hidden, H, W) feature map that is backbone-agnostic.  Each
    backbone adapter then projects this map further to fit its own block dims.

    Args:
        in_ch:  Number of input channels.  Must match the number of channels in
                the canonical ``SceneOutputs`` tensor (default: 8).
        hidden: Number of output channels for the scene feature map.
    """

    def __init__(self, in_ch: int = 8, hidden: int = 64) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.hidden = hidden

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        scene_out: "SceneOutputs | torch.Tensor",
    ) -> torch.Tensor:
        """Encode scene outputs into a feature map.

        Args:
            scene_out: Either a ``SceneOutputs`` dataclass instance or a
                       pre-stacked (B, 8, H, W) tensor.

        Returns:
            (B, hidden, H, W) float32 feature map.
        """
        if isinstance(scene_out, torch.Tensor):
            x = scene_out  # already stacked
        else:
            # SceneOutputs → (B, 8, H, W) via the canonical helper
            x = scene_out.to_canonical_tensor()

        x = x.float()
        return self.net(x)
