"""backbones/interface.py
=========================
Abstract interface that every backbone wrapper must implement.

A backbone receives a coarse composite RGB image (produced by
:class:`~backbones.reconstruction_decoder.StructuredDecoder` or a raymarcher)
together with a text prompt and per-block guide tensors, and returns a refined
image.

This decouples the scene prior (entity fields, depth, separation) from the
specific diffusion model used for refinement, making it easy to swap between
AnimateDiff, SDXL, or future DiT-based refiners.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from adapters.base_adapter import BaseBackboneAdapter


class BackboneInterface(ABC):
    """Abstract interface for diffusion backbone refiners.

    Concrete implementations (e.g. :class:`~backbones.animatediff_refiner.AnimateDiffRefiner`)
    wrap a pretrained diffusion pipeline and inject scene prior guides via
    their corresponding :class:`~adapters.base_adapter.BaseBackboneAdapter`.
    """

    @abstractmethod
    def refine(
        self,
        coarse_rgb: torch.Tensor,
        prompt: str,
        guides: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Refine a coarse composite image using scene prior guides.

        Args:
            coarse_rgb: (B, 3, H, W) float32 composite from the decoder,
                        values in [0, 1].
            prompt:     Text prompt string shared across the batch.
            guides:     Per-block guide tensors returned by
                        :meth:`~adapters.base_adapter.BaseBackboneAdapter.build_guides`.

        Returns:
            Refined (B, 3, H, W) float32 tensor, values in [0, 1].
        """
        raise NotImplementedError

    @abstractmethod
    def get_adapter(self) -> "BaseBackboneAdapter":
        """Return the adapter instance bound to this backbone.

        The adapter is used to inject scene prior guides into the backbone's
        internal UNet or DiT blocks.
        """
        raise NotImplementedError
