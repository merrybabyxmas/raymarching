from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class MotionRollout(nn.Module):
    """Predict coarse 2.5D layout state for each entity.

    Outputs center, scale, frontness, and orientation. The module is deliberately
    lightweight and does not attempt full physical motion simulation.

    v2: supports optional global context (for example camera pose embeddings)
    so that the scene prior can become more view-aware instead of implicitly
    overfitting to the training projection patterns.
    """

    def __init__(self, slot_dim: int = 256, hidden_dim: int = 256, context_dim: int = 0):
        super().__init__()
        self.context_dim = int(context_dim)
        in_dim = slot_dim * 2 + 2 + self.context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head_center = nn.Linear(hidden_dim, 2)
        self.head_scale = nn.Linear(hidden_dim, 1)
        self.head_front = nn.Linear(hidden_dim, 1)
        self.head_orient = nn.Linear(hidden_dim, 1)

    def _forward_entity(
        self,
        slot: torch.Tensor,
        mem: torch.Tensor,
        t_index: int,
        global_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        t = torch.full((slot.shape[0], 1), float(t_index), device=slot.device, dtype=slot.dtype)
        t_sin = torch.sin(t / 10.0)
        t_cos = torch.cos(t / 10.0)
        pieces = [slot, mem, t_sin, t_cos]
        if self.context_dim > 0:
            if global_context is None:
                global_context = torch.zeros(slot.shape[0], self.context_dim, device=slot.device, dtype=slot.dtype)
            pieces.append(global_context)
        h = self.net(torch.cat(pieces, dim=-1))
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
        return (
            self._forward_entity(slot_e0, mem_e0, t_index, global_context=global_context),
            self._forward_entity(slot_e1, mem_e1, t_index, global_context=global_context),
        )
