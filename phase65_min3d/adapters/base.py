from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..scene_outputs import SceneState


class BaseSceneAdapter(nn.Module):
    """Base interface for backbone-specific scene adapters."""

    def forward(self, scene_state: SceneState) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
