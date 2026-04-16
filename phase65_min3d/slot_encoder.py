from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class EntitySlotEncoder(nn.Module):
    """Encode per-entity identity slots from names and optional text context.

    This implementation is intentionally lightweight:
    - hashes entity names into a fixed-size embedding table,
    - optionally fuses a shared text context vector,
    - returns two stable entity slots.
    """

    def __init__(self, slot_dim: int = 256, vocab_size: int = 8192, text_dim: int = 768):
        super().__init__()
        self.slot_dim = slot_dim
        self.vocab_size = vocab_size
        self.name_embed = nn.Embedding(vocab_size, slot_dim)
        self.text_proj = nn.Linear(text_dim, slot_dim)
        self.fuse = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, slot_dim),
        )
        self.norm = nn.LayerNorm(slot_dim)

    def _name_to_index(self, name: str) -> int:
        return abs(hash(name.lower().strip())) % self.vocab_size

    def _embed_name(self, names: Sequence[str], device: torch.device) -> torch.Tensor:
        idx = torch.tensor([self._name_to_index(n) for n in names], device=device, dtype=torch.long)
        return self.name_embed(idx)

    def forward(
        self,
        entity_names: Sequence[str],
        text_context: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(entity_names) != 2:
            raise ValueError(f"Expected exactly 2 entity names, got {len(entity_names)}")
        if device is None:
            device = self.name_embed.weight.device

        name_slots = self._embed_name(entity_names, device=device)  # (2, D)
        if batch_size > 1:
            name_slots = name_slots.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 2, D)
        else:
            name_slots = name_slots.unsqueeze(0)

        if text_context is not None:
            if text_context.dim() == 1:
                text_context = text_context.unsqueeze(0)
            if text_context.shape[0] == 1 and batch_size > 1:
                text_context = text_context.expand(batch_size, -1)
            text_slot = self.text_proj(text_context).unsqueeze(1).expand(-1, 2, -1)
            fused = self.fuse(torch.cat([name_slots, text_slot], dim=-1))
            slots = self.norm(name_slots + fused)
        else:
            slots = self.norm(name_slots)

        return slots[:, 0], slots[:, 1]
