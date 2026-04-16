"""scene_prior/scene_outputs.py
================================
Canonical 8-channel portable scene representation dataclass.

All tensors are (B, H, W) float32.  The 8 channels encode everything
needed by downstream losses and guidance modules without requiring any
specific backbone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# SceneOutputs — the canonical 8-channel scene representation
# ---------------------------------------------------------------------------

@dataclass
class SceneOutputs:
    """Canonical 8-channel scene representation.

    Fields
    ------
    visible_e0, visible_e1 : (B, H, W)
        Visible (first-hit) contribution of each entity, rendered via
        transmittance compositing.  In [0, 1].
    amodal_e0, amodal_e1 : (B, H, W)
        Amodal presence — probability that the entity occupies a pixel at
        ANY depth bin, regardless of occlusion.  In [0, 1].
    depth_map : (B, H, W)
        Expected depth normalised to [0, 1].
    sep_map : (B, H, W)
        Signed separation: visible_e0 - visible_e1.  In [-1, 1].
    hidden_e0, hidden_e1 : (B, H, W)
        Occluded fraction: relu(amodal_ei - visible_ei) ≥ 0.
    """

    visible_e0: torch.Tensor   # (B, H, W)
    visible_e1: torch.Tensor   # (B, H, W)
    amodal_e0:  torch.Tensor   # (B, H, W)
    amodal_e1:  torch.Tensor   # (B, H, W)
    depth_map:  torch.Tensor   # (B, H, W)
    sep_map:    torch.Tensor   # (B, H, W)
    hidden_e0:  torch.Tensor   # (B, H, W)
    hidden_e1:  torch.Tensor   # (B, H, W)

    def to_canonical_tensor(self) -> torch.Tensor:
        """Stack all 8 channels into (B, 8, H, W) in canonical order.

        Channel order (fixed):
          0  visible_e0
          1  visible_e1
          2  amodal_e0
          3  amodal_e1
          4  depth_map
          5  sep_map
          6  hidden_e0
          7  hidden_e1
        """
        return torch.stack(
            [
                self.visible_e0,
                self.visible_e1,
                self.amodal_e0,
                self.amodal_e1,
                self.depth_map,
                self.sep_map,
                self.hidden_e0,
                self.hidden_e1,
            ],
            dim=1,
        )  # (B, 8, H, W)


# ---------------------------------------------------------------------------
# RendererOutputs — intermediate tensors from the renderer before assembly
# ---------------------------------------------------------------------------

@dataclass
class RendererOutputs:
    """Intermediate outputs from the EntityRenderer before SceneOutputs assembly.

    Exposed for debugging and loss calculation on raw density fields.
    """

    density_e0: torch.Tensor   # (B, depth_bins, H, W)  input densities
    density_e1: torch.Tensor   # (B, depth_bins, H, W)
    transmittance: torch.Tensor  # (B, depth_bins, H, W)  T_k per bin
