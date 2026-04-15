"""
Phase 63 — Structured Guide Encoder
====================================

Replaces Phase 62's ``GuideFeatureAssembler``.  The new design removes the
"guide_family" switch entirely: there is a single structured guide composed
of five per-pixel streams, each L2-normalised independently so that neither
entity can dominate the guide on pure magnitude.

Streams
-------
Given the renderer's ``visible_i``, ``amodal_i`` and the per-entity feature
maps ``F_ei`` (projected to the hidden channel width), we build::

    s1 = V_0 * h_e0                # entity-0 visible
    s2 = V_1 * h_e1                # entity-1 visible
    s3 = A_0 * h_e0                # entity-0 amodal (includes occluded part)
    s4 = A_1 * h_e1                # entity-1 amodal
    s5 = depth_emb(D)              # learned depth positional encoding

Each stream is L2-normalised along its channel dimension independently::

    s_hat_i = s_i / ||s_i||_2

Then concatenated into a (B, hidden*5, H, W) block feature and projected per
UNet block (same per-block Conv2d stack as Phase 62) to produce the final
guide dict.

Gate mechanism
--------------
We reuse v22 gate semantics: the forward pass stores a per-block scalar gate
tensor in ``_current_gates`` (grad-connected, via ``tanh(guide_gate)``) and
the injection manager multiplies the gate AFTER amplitude normalisation —
preserving gate gradient flow.

The ``GuideInjectionManager`` and ``inject_guide_into_unet_features``
primitives from ``models.phase62.conditioning`` are reused directly.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Re-export injection primitives + constants so callers can simply import from
# guide_encoder.
from models.phase62.conditioning import (
    BLOCK_DIMS,
    BLOCK_SPATIAL,
    INJECT_CONFIGS,
    GuideInjectionManager,
    inject_guide_into_unet_features,
)

from models.entity_field import EntityFieldOutputs
from models.renderer import RendererOutputs


# ---------------------------------------------------------------------------
# Depth positional embedding (sinusoidal, learnable residual)
# ---------------------------------------------------------------------------
class DepthPositionalEmbedding(nn.Module):
    """Encode a scalar depth map (B, H, W) into (B, hidden, H, W).

    Uses a fixed sinusoidal basis across ``n_freqs`` frequencies followed by
    a learned 1x1 conv to map to ``hidden`` channels.  This gives depth a
    structured representation independent of the density fields themselves.
    """

    def __init__(self, hidden: int, n_freqs: int = 8):
        super().__init__()
        self.hidden = hidden
        self.n_freqs = n_freqs
        # Frequencies 2^0 ... 2^(n_freqs-1) scaled by pi.
        freqs = 2.0 ** torch.arange(n_freqs).float() * math.pi          # (n_freqs,)
        self.register_buffer("freqs", freqs.view(1, n_freqs, 1, 1), persistent=False)
        # 2 * n_freqs channels (sin + cos) -> hidden.
        self.proj = nn.Conv2d(2 * n_freqs, hidden, kernel_size=1, bias=True)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        # depth: (B, H, W) in [0, 1]
        d = depth.unsqueeze(1)                                          # (B, 1, H, W)
        angles = d * self.freqs                                         # (B, n_freqs, H, W)
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (B, 2*nf, H, W)
        return self.proj(enc)                                           # (B, hidden, H, W)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
def _l2_normalise_stream(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise along the channel dimension (dim=1)."""
    return x / x.norm(dim=1, keepdim=True).clamp(min=eps)


class StructuredGuideEncoder(nn.Module):
    """
    Build a multi-scale guide dict from renderer outputs + entity features.

    Parameters
    ----------
    feat_dim : int
        Backbone feature dim for F_e0/F_e1 (tokens are (B, S, feat_dim)).
    hidden : int
        Per-stream channel width.  The concatenated guide has hidden*5 channels.
    spatial_h, spatial_w : int
        Token grid size.
    inject_config : str
        One of ``INJECT_CONFIGS`` keys — which UNet blocks to inject into.
    gate_warm_start : float
        Initial value of tanh(guide_gate) ∈ [0, 0.95).  0.0 means guide starts
        at magnitude zero (safe), higher values let the guide act earlier.
    """

    def __init__(
        self,
        feat_dim: int = 640,
        hidden: int = 64,
        spatial_h: int = 16,
        spatial_w: int = 16,
        inject_config: str = "multiscale",
        gate_warm_start: float = 0.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w

        # Project token features to ``hidden`` channels.  Only the per-entity
        # features are needed here — global F_g is consumed earlier in the
        # field decoder.
        self.proj_e0 = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU())
        self.proj_e1 = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU())

        # Depth embedding.
        self.depth_emb = DepthPositionalEmbedding(hidden=hidden, n_freqs=8)

        # Per-block projector: (B, hidden*5, H, W) -> (B, block_dim, H_b, W_b)
        block_names = INJECT_CONFIGS.get(inject_config, ["mid", "up2"])
        self.block_names = block_names
        in_ch = hidden * 5

        self.block_projectors = nn.ModuleDict()
        for bn in block_names:
            block_dim = BLOCK_DIMS[bn]
            self.block_projectors[bn] = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Conv2d(in_ch, block_dim, kernel_size=1, bias=True),
            )

        # v22 gate: stored as a pre-tanh parameter; the effective gate is
        # tanh(guide_gate).  gate_warm_start lets us initialise above zero.
        _ws = max(0.0, min(float(gate_warm_start), 0.95))
        _gate_init = math.atanh(_ws) if _ws > 0.0 else 0.0
        self.guide_gates = nn.ParameterDict({
            bn: nn.Parameter(torch.full((1,), _gate_init))
            for bn in block_names
        })

        # Filled by forward(), read by the injection manager's hook callback.
        self._current_gates: Dict[str, torch.Tensor] = {}

        # Diagnostics (detached scalars) — useful for debugging entity balance.
        self._diag_stream_norms: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _tokens_to_2d(
        self, x: torch.Tensor, proj: nn.Module,
    ) -> torch.Tensor:
        B = x.shape[0]
        h = proj(x.float())                                             # (B, S, hidden)
        return rearrange(h, "b (h w) c -> b c h w",
                         h=self.spatial_h, w=self.spatial_w)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        renderer_out: RendererOutputs,
        field_out: EntityFieldOutputs,
        F_e0: torch.Tensor,                                             # (B, S, feat_dim)
        F_e1: torch.Tensor,                                             # (B, S, feat_dim)
    ) -> Dict[str, torch.Tensor]:
        """Return a dict ``{block_name: guide_tensor}`` (gate NOT applied)."""

        B = F_e0.shape[0]
        H, W = self.spatial_h, self.spatial_w

        # Per-entity feature maps.
        h_e0 = self._tokens_to_2d(F_e0, self.proj_e0)                   # (B, hidden, H, W)
        h_e1 = self._tokens_to_2d(F_e1, self.proj_e1)

        # Ensure spatial sizes of renderer outputs match token grid.  The
        # renderer usually runs at the same (H, W), but interpolate just in
        # case the feature field was produced at a different resolution.
        def _match(x: torch.Tensor) -> torch.Tensor:
            # (B, H, W) -> (B, 1, H, W) -> maybe resize -> (B, 1, H, W)
            if x.shape[-2:] != (H, W):
                x = F.interpolate(
                    x.unsqueeze(1).float(), size=(H, W),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
            return x.unsqueeze(1).float()                               # (B, 1, H, W)

        vis_e0 = _match(renderer_out.visible_e0)
        vis_e1 = _match(renderer_out.visible_e1)
        amo_e0 = _match(renderer_out.amodal_e0)
        amo_e1 = _match(renderer_out.amodal_e1)

        depth = renderer_out.depth
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(
                depth.unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1)

        # Build 5 streams.
        s_vis_e0 = vis_e0 * h_e0
        s_vis_e1 = vis_e1 * h_e1
        s_amo_e0 = amo_e0 * h_e0
        s_amo_e1 = amo_e1 * h_e1
        s_depth = self.depth_emb(depth)                                 # (B, hidden, H, W)

        # Diagnostics — pre-normalisation magnitudes (detached).
        with torch.no_grad():
            self._diag_stream_norms = {
                "vis_e0": s_vis_e0.norm(dim=1).mean().item(),
                "vis_e1": s_vis_e1.norm(dim=1).mean().item(),
                "amo_e0": s_amo_e0.norm(dim=1).mean().item(),
                "amo_e1": s_amo_e1.norm(dim=1).mean().item(),
                "depth":  s_depth.norm(dim=1).mean().item(),
            }

        # L2-normalise each stream independently so no entity can dominate by
        # raw magnitude alone.
        s_vis_e0 = _l2_normalise_stream(s_vis_e0)
        s_vis_e1 = _l2_normalise_stream(s_vis_e1)
        s_amo_e0 = _l2_normalise_stream(s_amo_e0)
        s_amo_e1 = _l2_normalise_stream(s_amo_e1)
        s_depth = _l2_normalise_stream(s_depth)

        guide_base = torch.cat(
            [s_vis_e0, s_vis_e1, s_amo_e0, s_amo_e1, s_depth], dim=1,
        )                                                               # (B, hidden*5, H, W)

        # Per-block projection + resize.
        guides: Dict[str, torch.Tensor] = {}
        self._current_gates = {}
        for bn in self.block_names:
            proj = self.block_projectors[bn]
            h_b, w_b = BLOCK_SPATIAL[bn]
            if (h_b, w_b) != (H, W):
                resized = F.interpolate(
                    guide_base, size=(h_b, w_b),
                    mode="bilinear", align_corners=False,
                )
            else:
                resized = guide_base

            proj_out = proj(resized)                                    # (B, block_dim, h_b, w_b)
            gate = torch.tanh(self.guide_gates[bn])                     # scalar, grad-connected

            guides[bn] = proj_out                                       # gate NOT applied (v22)
            self._current_gates[bn] = gate

        return guides

    # ------------------------------------------------------------------
    # API expected by GuideInjectionManager
    # ------------------------------------------------------------------
    def get_gate(self, block_name: str) -> Optional[torch.Tensor]:
        """Return the scalar gate tensor for a given block.

        Called by the injection manager's hook callback to apply the gate
        AFTER amplitude normalisation (v22).
        """
        return self._current_gates.get(block_name, None)


__all__ = [
    "StructuredGuideEncoder",
    "GuideInjectionManager",
    "inject_guide_into_unet_features",
    "BLOCK_DIMS",
    "BLOCK_SPATIAL",
    "INJECT_CONFIGS",
]
