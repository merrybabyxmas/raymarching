from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class VolumeOutputs:
    """Standardized outputs from volume prediction + projection."""
    fg_logit: Optional[torch.Tensor] = None       # (B, 1, K, H, W)
    id_logits: Optional[torch.Tensor] = None       # (B, 2, K, H, W)
    entity_logits: Optional[torch.Tensor] = None   # (B, 2, K, H, W) independent per-entity
    entity_probs: Optional[torch.Tensor] = None    # (B, 2, K, H, W)
    visible_class: Optional[torch.Tensor] = None   # (B, H, W)
    front_probs: Optional[torch.Tensor] = None     # (B, C, H, W)
    back_probs: Optional[torch.Tensor] = None      # (B, C, H, W)
    amodal: Dict[str, torch.Tensor] = field(default_factory=dict)
    visible: Dict[str, torch.Tensor] = field(default_factory=dict)


class VolumeObjective(nn.Module):
    """Base class for volume objective families."""

    def forward(
        self,
        outputs: VolumeOutputs,
        V_gt: torch.Tensor,           # (B, K, H, W) class indices
        gt_visible: Optional[torch.Tensor] = None,  # (B, 2, H, W)
        gt_amodal: Optional[torch.Tensor] = None,   # (B, 2, H, W)
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss dict from volume outputs and GT.

        Returns:
            dict with 'total' key (scalar) and any component keys for logging.
        """
        raise NotImplementedError
