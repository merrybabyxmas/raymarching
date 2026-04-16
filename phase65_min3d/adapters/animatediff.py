from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSceneAdapter
from ..scene_outputs import SceneState


BLOCK_DIMS = {
    "mid": 1280,
    "up1": 1280,
    "up2": 640,
    "up3": 320,
}
BLOCK_SPATIAL = {
    "mid": (4, 4),
    "up1": (8, 8),
    "up2": (16, 16),
    "up3": (32, 32),
}


class AnimateDiffSceneAdapter(BaseSceneAdapter):
    """Project SceneState into multiscale UNet conditioning tensors.

    This module does not register hooks itself; it only produces block-specific
    tensors for a caller-side injection manager.
    """

    def __init__(self, feat_dim: int = 64, hidden_dim: int = 128, blocks: Iterable[str] = ("mid", "up1", "up2", "up3")):
        super().__init__()
        self.blocks: List[str] = list(blocks)
        self.map_encoder_e0 = nn.Sequential(nn.Conv2d(4, hidden_dim, 3, padding=1), nn.GELU())
        self.map_encoder_e1 = nn.Sequential(nn.Conv2d(4, hidden_dim, 3, padding=1), nn.GELU())
        self.depth_encoder = nn.Sequential(nn.Conv2d(2, hidden_dim // 2, 3, padding=1), nn.GELU())
        self.feat_encoder_e0 = nn.Sequential(nn.Conv2d(feat_dim, hidden_dim, 3, padding=1), nn.GELU())
        self.feat_encoder_e1 = nn.Sequential(nn.Conv2d(feat_dim, hidden_dim, 3, padding=1), nn.GELU())
        self.trunk = nn.Sequential(
            nn.Conv2d(hidden_dim * 4 + hidden_dim // 2, hidden_dim * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
            nn.GELU(),
        )
        self.projectors = nn.ModuleDict({
            b: nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim * 2, BLOCK_DIMS[b], 1),
            )
            for b in self.blocks
        })
        self.gates = nn.ParameterDict({b: nn.Parameter(torch.zeros(1)) for b in self.blocks})

    def _pack_e0(self, scene_state: SceneState) -> torch.Tensor:
        m = scene_state.maps
        return torch.cat([m.visible_e0, m.hidden_e0, m.amodal_e0, m.contact if m.contact is not None else torch.zeros_like(m.visible_e0)], dim=1)

    def _pack_e1(self, scene_state: SceneState) -> torch.Tensor:
        m = scene_state.maps
        return torch.cat([m.visible_e1, m.hidden_e1, m.amodal_e1, m.contact if m.contact is not None else torch.zeros_like(m.visible_e1)], dim=1)

    def forward(self, scene_state: SceneState) -> Dict[str, torch.Tensor]:
        m = scene_state.maps
        e0_maps = self.map_encoder_e0(self._pack_e0(scene_state))
        e1_maps = self.map_encoder_e1(self._pack_e1(scene_state))
        depth_maps = self.depth_encoder(torch.cat([m.depth_e0, m.depth_e1], dim=1))
        feat_e0 = self.feat_encoder_e0(scene_state.features.feat_e0)
        feat_e1 = self.feat_encoder_e1(scene_state.features.feat_e1)
        if feat_e0.shape[-2:] != e0_maps.shape[-2:]:
            feat_e0 = F.interpolate(feat_e0, size=e0_maps.shape[-2:], mode="bilinear", align_corners=False)
            feat_e1 = F.interpolate(feat_e1, size=e0_maps.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.trunk(torch.cat([e0_maps, e1_maps, depth_maps, feat_e0, feat_e1], dim=1))
        out: Dict[str, torch.Tensor] = {}
        for b in self.blocks:
            h, w = BLOCK_SPATIAL[b]
            x = fused if fused.shape[-2:] == (h, w) else F.interpolate(fused, size=(h, w), mode="bilinear", align_corners=False)
            out[b] = self.projectors[b](x) * torch.tanh(self.gates[b])
        return out
