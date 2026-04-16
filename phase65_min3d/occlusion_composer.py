from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .scene_outputs import SceneFeatures, SceneMaps, SceneState


class OcclusionComposer(nn.Module):
    """Compose per-entity layered outputs into a coherent SceneState.

    The composer preserves both entities, resolves front/back competition via
    per-entity depth, and reassigns suppressed visible mass into hidden mass.
    """

    def __init__(self, sharpness: float = 8.0):
        super().__init__()
        self.sharpness = sharpness

    def forward(
        self,
        raw_visible_e0: torch.Tensor,
        raw_hidden_e0: torch.Tensor,
        raw_depth_e0: torch.Tensor,
        feat_e0: torch.Tensor,
        raw_visible_e1: torch.Tensor,
        raw_hidden_e1: torch.Tensor,
        raw_depth_e1: torch.Tensor,
        feat_e1: torch.Tensor,
        global_feat: Optional[torch.Tensor] = None,
        mem_e0: Optional[torch.Tensor] = None,
        mem_e1: Optional[torch.Tensor] = None,
    ) -> SceneState:
        v0 = torch.sigmoid(raw_visible_e0)
        v1 = torch.sigmoid(raw_visible_e1)
        h0 = torch.sigmoid(raw_hidden_e0)
        h1 = torch.sigmoid(raw_hidden_e1)
        d0 = torch.sigmoid(raw_depth_e0)
        d1 = torch.sigmoid(raw_depth_e1)

        # Smaller depth means closer / more front. Convert to frontness gates.
        f0 = torch.sigmoid(self.sharpness * (d1 - d0))
        f1 = torch.sigmoid(self.sharpness * (d0 - d1))

        v0_front = v0 * f0
        v1_front = v1 * f1
        vis_sum = (v0_front + v1_front).clamp(min=1.0)
        v0_final = v0_front / vis_sum
        v1_final = v1_front / vis_sum

        a0 = (v0 + h0).clamp(0.0, 1.0)
        a1 = (v1 + h1).clamp(0.0, 1.0)
        h0_final = (a0 - v0_final).clamp(0.0, 1.0)
        h1_final = (a1 - v1_final).clamp(0.0, 1.0)
        contact = (a0 * a1).clamp(0.0, 1.0)

        maps = SceneMaps(
            visible_e0=v0_final,
            visible_e1=v1_final,
            hidden_e0=h0_final,
            hidden_e1=h1_final,
            amodal_e0=a0,
            amodal_e1=a1,
            depth_e0=d0,
            depth_e1=d1,
            contact=contact,
        )
        feats = SceneFeatures(feat_e0=feat_e0, feat_e1=feat_e1, global_feat=global_feat)
        return SceneState(maps=maps, features=feats, mem_e0=mem_e0, mem_e1=mem_e1)
