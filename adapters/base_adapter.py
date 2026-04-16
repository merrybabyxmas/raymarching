"""adapters/base_adapter.py
==========================
Abstract base class for all backbone-specific thin adapter layers.

Every concrete adapter must implement:
  - build_guides:   scene features → per-block guide tensors
  - register_hooks: attach forward hooks to the target model
  - remove_hooks:   detach all hooks
  - inject:         single-call injection for a named block (used in testing)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseBackboneAdapter(ABC):
    """Abstract interface that all backbone adapters must satisfy."""

    @abstractmethod
    def build_guides(
        self, scene_features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Convert (B, hidden, H, W) scene features into per-block guide tensors.

        Returns a dict mapping block name → guide tensor ready for injection.
        """
        raise NotImplementedError

    @abstractmethod
    def register_hooks(self, model) -> None:
        """Attach forward hooks to ``model`` so guides are injected during the
        forward pass.  Must be called after :meth:`build_guides` / before
        the backbone forward call.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_hooks(self) -> None:
        """Remove all previously registered forward hooks."""
        raise NotImplementedError

    @abstractmethod
    def inject(self, h: torch.Tensor, block_name: str) -> torch.Tensor:
        """Directly inject the stored guide for *block_name* into *h*.

        Useful for unit tests and custom forward loops where hooks are not used.

        Args:
            h:          UNet intermediate feature map (B, C, H, W).
            block_name: Key matching an entry in the current guides dict.

        Returns:
            Injected feature map with the same shape as *h*.
        """
        raise NotImplementedError
