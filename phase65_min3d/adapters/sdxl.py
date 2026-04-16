from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSceneAdapter
from ..scene_outputs import SceneState


BLOCK_DIMS = {
    "down": 320,
    "mid": 1280,
    "up": 640,
}
BLOCK_SPATIAL = {
    "down": (32, 32),
    "mid": (16, 16),
    "up": (32, 32),
}


class SDXLSceneAdapter(BaseSceneAdapter):
    """Generic SDXL-style scene adapter producing multiscale residuals."""

    def __init__(self, feat_dim: int = 64, hidden_dim: int = 128, blocks: Iterable[str] = ("down", "mid", "up")):
        super().__init__()
        self.blocks: List[str] = list(blocks)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(8, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.feat_encoder = nn.Sequential(
            nn.Conv2d(feat_dim * 2, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
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

    def _pack_maps(self, scene_state: SceneState) -> torch.Tensor:
        m = scene_state.maps
        return torch.cat([
            m.visible_e0, m.visible_e1,
            m.hidden_e0, m.hidden_e1,
            m.amodal_e0, m.amodal_e1,
            m.depth_e0, m.depth_e1,
        ], dim=1)

    def forward(self, scene_state: SceneState) -> Dict[str, torch.Tensor]:
        maps = self.map_encoder(self._pack_maps(scene_state))
        feats = self.feat_encoder(torch.cat([scene_state.features.feat_e0, scene_state.features.feat_e1], dim=1))
        if feats.shape[-2:] != maps.shape[-2:]:
            feats = F.interpolate(feats, size=maps.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.trunk(torch.cat([maps, feats], dim=1))
        out: Dict[str, torch.Tensor] = {}
        for b in self.blocks:
            h, w = BLOCK_SPATIAL[b]
            x = fused if fused.shape[-2:] == (h, w) else F.interpolate(fused, size=(h, w), mode="bilinear", align_corners=False)
            out[b] = self.projectors[b](x) * torch.tanh(self.gates[b])
        return out
