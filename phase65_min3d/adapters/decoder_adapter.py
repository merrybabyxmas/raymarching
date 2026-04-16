from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSceneAdapter
from ..scene_outputs import SceneState


class DecoderSceneAdapter(BaseSceneAdapter):
    """Adapter for simple reconstruction decoders.

    Produces a fused feature tensor from explicit scene maps and latent features.
    """

    def __init__(self, feat_dim: int = 64, out_dim: int = 128, map_size: int = 64, feat_size: int = 32):
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.feat_encoder_e0 = nn.Sequential(
            nn.Conv2d(feat_dim, out_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.feat_encoder_e1 = nn.Sequential(
            nn.Conv2d(feat_dim, out_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(64 + out_dim, out_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
        )
        self.map_size = map_size
        self.feat_size = feat_size

    def _pack_maps(self, scene_state: SceneState) -> torch.Tensor:
        m = scene_state.maps
        return torch.cat([
            m.visible_e0, m.visible_e1,
            m.hidden_e0, m.hidden_e1,
            m.amodal_e0, m.amodal_e1,
            m.depth_e0, m.depth_e1,
        ], dim=1)

    def forward(self, scene_state: SceneState) -> Dict[str, torch.Tensor]:
        maps = self._pack_maps(scene_state)
        feat_e0 = self.feat_encoder_e0(scene_state.features.feat_e0)
        feat_e1 = self.feat_encoder_e1(scene_state.features.feat_e1)
        feats = torch.cat([feat_e0, feat_e1], dim=1)
        if feats.shape[-2:] != maps.shape[-2:]:
            feats = F.interpolate(feats, size=maps.shape[-2:], mode="bilinear", align_corners=False)
        map_feat = self.map_encoder(maps)
        fused = self.out_proj(torch.cat([map_feat, feats], dim=1))
        return {"decoder_cond": fused}
