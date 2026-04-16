from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TemporalSlotMemory(nn.Module):
    """Per-entity GRU-like memory update.

    Keeps entity memories separate until after the update to reduce slot mixing.
    """

    def __init__(self, slot_dim: int = 256, obs_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.slot_dim = slot_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        in_dim = slot_dim + obs_dim + hidden_dim
        self.update_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def _default_obs(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros(ref.shape[0], self.obs_dim, device=ref.device, dtype=ref.dtype)

    def _default_mem(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros(ref.shape[0], self.hidden_dim, device=ref.device, dtype=ref.dtype)

    def _step(self, prev_mem: Optional[torch.Tensor], slot: torch.Tensor, obs: Optional[torch.Tensor]) -> torch.Tensor:
        if prev_mem is None:
            prev_mem = self._default_mem(slot)
        if obs is None:
            obs = self._default_obs(slot)
        x = torch.cat([prev_mem, slot, obs], dim=-1)
        proposal = self.update_net(x)
        gate = self.gate_net(x)
        new_mem = gate * prev_mem + (1.0 - gate) * proposal
        return self.out_norm(new_mem)

    def forward(
        self,
        prev_mem_e0: Optional[torch.Tensor],
        prev_mem_e1: Optional[torch.Tensor],
        slot_e0: torch.Tensor,
        slot_e1: torch.Tensor,
        obs_e0: Optional[torch.Tensor] = None,
        obs_e1: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mem_e0 = self._step(prev_mem_e0, slot_e0, obs_e0)
        mem_e1 = self._step(prev_mem_e1, slot_e1, obs_e1)
        return mem_e0, mem_e1
