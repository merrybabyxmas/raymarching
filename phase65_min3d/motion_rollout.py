from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class MotionRollout(nn.Module):
    """Predict coarse 2.5D layout state for each entity.

    Outputs center, scale, frontness, and orientation. The module is deliberately
    lightweight and does not attempt full physical motion simulation.
    """

    def __init__(self, slot_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slot_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head_center = nn.Linear(hidden_dim, 2)
        self.head_scale = nn.Linear(hidden_dim, 1)
        self.head_front = nn.Linear(hidden_dim, 1)
        self.head_orient = nn.Linear(hidden_dim, 1)

    def _forward_entity(self, slot: torch.Tensor, mem: torch.Tensor, t_index: int) -> Dict[str, torch.Tensor]:
        t = torch.full((slot.shape[0], 1), float(t_index), device=slot.device, dtype=slot.dtype)
        t_sin = torch.sin(t / 10.0)
        t_cos = torch.cos(t / 10.0)
        h = self.net(torch.cat([slot, mem, t_sin, t_cos], dim=-1))
        center = torch.tanh(self.head_center(h))
        scale = torch.sigmoid(self.head_scale(h))
        frontness = torch.sigmoid(self.head_front(h))
        orientation = torch.tanh(self.head_orient(h))
        return {
            "center": center,
            "scale": scale,
            "frontness": frontness,
            "orientation": orientation,
        }

    def forward(
        self,
        slot_e0: torch.Tensor,
        slot_e1: torch.Tensor,
        mem_e0: torch.Tensor,
        mem_e1: torch.Tensor,
        t_index: int,
        global_context: Optional[torch.Tensor] = None,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # global_context intentionally unused in the minimal implementation
        return self._forward_entity(slot_e0, mem_e0, t_index), self._forward_entity(slot_e1, mem_e1, t_index)
