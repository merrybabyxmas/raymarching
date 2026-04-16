"""backbones/reconstruction_decoder.py
======================================
Stage-2 supervised decoder: scene features → coarse composite RGB.

:class:`StructuredDecoder` proves the scene prior (entity fields, depth,
separation) contains enough information to reconstruct plausible images
WITHOUT a diffusion backbone.  It is trained end-to-end with
:class:`DecoderLosses` before diffusion refinement is added.

Architecture: skip-free transposed-conv decoder
    Input  : (B, 64, H_in, W_in)  — output of SceneGuideEncoder
    Output : (B,  3, H_out, W_out)  — composite RGB in [0, 1]

    Spatial upsampling steps (defaults):
        32 → 64 → 128 → 256

Each step doubles spatial resolution via ConvTranspose2d + GELU, without
skip connections so the model is forced to rely entirely on the scene prior.

Optional separate decoding of isolated entity renders (iso_e0 / iso_e1) can
be enabled via ``separate=True`` in the forward call.  These are used by
:meth:`DecoderLosses.isolation_consistency_loss`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# StructuredDecoder
# ---------------------------------------------------------------------------

class StructuredDecoder(nn.Module):
    """Stage-2 decoder: scene features → coarse composite RGB (and optionally
    isolated entity renders).

    Args:
        in_ch:         Number of input channels (must match SceneGuideEncoder
                       hidden dim, default 64).
        out_ch:        Number of output channels (3 for RGB).
        upsample_to:   Target spatial resolution.  The decoder doubles spatial
                       resolution at each step; ``upsample_to`` determines how
                       many steps are needed relative to the input size.
                       Default: 256.  Input assumed to be (B, in_ch, 32, 32).
    """

    def __init__(
        self,
        in_ch: int = 64,
        out_ch: int = 3,
        upsample_to: int = 256,
    ) -> None:
        super().__init__()
        self.in_ch       = in_ch
        self.out_ch      = out_ch
        self.upsample_to = upsample_to

        # Determine how many 2× upsample steps we need.
        # We start from an assumed 32×32 input and double to upsample_to.
        import math
        steps = int(round(math.log2(upsample_to / 32)))
        steps = max(steps, 1)
        self._n_steps = steps

        # Build the decoder layers.
        # Channel schedule: in_ch → 128 → 64 → 32 → out_ch
        channel_schedule = self._make_channel_schedule(in_ch, steps)

        layers: list[nn.Module] = []
        for i in range(steps):
            c_in, c_out = channel_schedule[i], channel_schedule[i + 1]
            layers.append(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=True)
            )
            layers.append(nn.GELU())
        # Final 1×1 conv to out_ch with sigmoid for [0, 1] output
        layers.append(nn.Conv2d(channel_schedule[-1], out_ch, kernel_size=1, bias=True))
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

        # Optional separate entity decoders (shared architecture, separate weights)
        # Only created on demand (via separate=True in forward)
        self.iso_decoder_e0: Optional[nn.Sequential] = None
        self.iso_decoder_e1: Optional[nn.Sequential] = None
        self._channel_schedule = channel_schedule

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_channel_schedule(in_ch: int, steps: int) -> list[int]:
        """Produce a descending channel schedule of length steps+1."""
        # Always end just before out_ch (out_ch added separately)
        targets = [128, 64, 32, 16]
        schedule = [in_ch]
        for i in range(steps):
            if i < len(targets):
                schedule.append(targets[i])
            else:
                schedule.append(max(schedule[-1] // 2, 8))
        return schedule

    def _build_iso_decoder(self) -> nn.Sequential:
        """Build an isolated entity decoder with the same architecture."""
        channel_schedule = self._channel_schedule
        steps = self._n_steps
        layers: list[nn.Module] = []
        for i in range(steps):
            c_in, c_out = channel_schedule[i], channel_schedule[i + 1]
            layers.append(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=True)
            )
            layers.append(nn.GELU())
        layers.append(nn.Conv2d(channel_schedule[-1], self.out_ch, kernel_size=1, bias=True))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        scene_features: torch.Tensor,
        separate: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Decode scene features to coarse composite RGB.

        Args:
            scene_features: (B, in_ch, H, W) from SceneGuideEncoder.
                            Expected H == W == 32 for the default schedule.
            separate:       If True, also decode isolated entity renders
                            ``iso_e0`` and ``iso_e1``.  The iso decoders share
                            the same architecture but have independent weights.

        Returns:
            Dict with keys:
                ``"composite"`` — (B, 3, H_out, W_out) in [0, 1].
                ``"iso_e0"``    — (B, 3, H_out, W_out) if separate=True.
                ``"iso_e1"``    — (B, 3, H_out, W_out) if separate=True.
        """
        composite = self.decoder(scene_features)
        out: dict[str, torch.Tensor] = {"composite": composite}

        if separate:
            # Lazy initialisation of iso decoders on the same device/dtype
            if self.iso_decoder_e0 is None:
                self.iso_decoder_e0 = self._build_iso_decoder().to(
                    device=scene_features.device, dtype=scene_features.dtype
                )
                # Register as proper submodule
                self.add_module("iso_decoder_e0", self.iso_decoder_e0)
            if self.iso_decoder_e1 is None:
                self.iso_decoder_e1 = self._build_iso_decoder().to(
                    device=scene_features.device, dtype=scene_features.dtype
                )
                self.add_module("iso_decoder_e1", self.iso_decoder_e1)

            # Use vis_e0/vis_e1 weighting from first two channels of scene_features
            # Channel 0 = vis_e0, Channel 1 = vis_e1 (canonical order from SceneOutputs)
            vis_e0 = scene_features[:, 0:1, :, :]  # (B, 1, H, W)
            vis_e1 = scene_features[:, 1:2, :, :]

            # Weight the scene features by each entity's visibility mask
            feat_e0 = scene_features * vis_e0
            feat_e1 = scene_features * vis_e1

            out["iso_e0"] = self.iso_decoder_e0(feat_e0)
            out["iso_e1"] = self.iso_decoder_e1(feat_e1)

        return out


# ---------------------------------------------------------------------------
# DecoderLosses
# ---------------------------------------------------------------------------

class DecoderLosses:
    """Stage-2 training losses for :class:`StructuredDecoder`.

    All methods are static; no instantiation required.
    """

    @staticmethod
    def reconstruction_loss(
        pred_rgb: torch.Tensor,
        gt_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """L1 reconstruction loss between predicted and ground-truth RGB.

        Args:
            pred_rgb: (B, 3, H, W) predicted composite, values in [0, 1].
            gt_rgb:   (B, 3, H, W) ground-truth image, values in [0, 1].

        Returns:
            Scalar loss tensor.

        Note:
            LPIPS can be added here in the future.  The current implementation
            uses L1 only for efficiency and differentiability.
        """
        return F.l1_loss(pred_rgb, gt_rgb)

    @staticmethod
    def isolation_consistency_loss(
        iso_e0: torch.Tensor,
        iso_e1: torch.Tensor,
        composite: torch.Tensor,
    ) -> torch.Tensor:
        """Penalise when the two entity isolates do not sum to the composite.

        Encourages the decoder to produce entity renders whose weighted sum
        explains the composite output, preventing entity collapse.

        Args:
            iso_e0:    (B, 3, H, W) isolated render of entity 0.
            iso_e1:    (B, 3, H, W) isolated render of entity 1.
            composite: (B, 3, H, W) composite render.

        Returns:
            Scalar loss tensor.
        """
        # Average of iso renders should approximate composite
        approx = (iso_e0 + iso_e1) * 0.5
        return F.l1_loss(approx, composite.detach())

    @staticmethod
    def object_count_loss(
        composite: torch.Tensor,
        min_entities: int = 2,
    ) -> torch.Tensor:
        """Penalise if the composite appears to contain fewer than ``min_entities``.

        Proxy measure: the composite must exhibit non-trivial pixel variance in
        at least two spatially distinct regions.  We partition the image into
        four quadrants and require that at least ``min_entities`` quadrants have
        per-channel standard deviation above a small threshold.

        A low variance in a quadrant suggests empty space or a single flat
        entity covering the whole image.

        Args:
            composite:    (B, 3, H, W) composite render.
            min_entities: Minimum number of "active" quadrants (default 2).

        Returns:
            Scalar loss tensor (0 if criterion met, positive otherwise).
        """
        B, C, H, W = composite.shape
        h2, w2 = H // 2, W // 2

        # Four quadrants
        quadrants = [
            composite[:, :, :h2,  :w2],   # top-left
            composite[:, :, :h2,  w2:],   # top-right
            composite[:, :, h2:,  :w2],   # bottom-left
            composite[:, :, h2:,  w2:],   # bottom-right
        ]

        threshold = 0.02  # min std for a quadrant to be "active"
        stds = torch.stack(
            [q.std(dim=(1, 2, 3)) for q in quadrants], dim=1
        )  # (B, 4)

        active = (stds > threshold).float()   # (B, 4)
        n_active = active.sum(dim=1)           # (B,)

        # Penalise samples where fewer than min_entities quadrants are active
        deficit = F.relu(torch.tensor(float(min_entities), device=composite.device) - n_active)
        return deficit.mean()
