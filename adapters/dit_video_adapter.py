"""adapters/dit_video_adapter.py
================================
Adapter for DiT-style video models (e.g. CogVideoX, Open-Sora).

DiT architectures use sequences of tokens rather than spatial feature maps.
Scene features are therefore projected to a sequence of conditioning tokens
which are either concatenated to the model's token sequence (prepend mode) or
injected via cross-attention (xattn mode).

This module is intentionally stubbed with reasonable defaults for future use;
the ``register_hooks`` / ``remove_hooks`` methods require a concrete DiT
implementation to be wired up.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters.base_adapter import BaseBackboneAdapter


class DiTVideoAdapter(nn.Module, BaseBackboneAdapter):
    """Adapter for DiT-style video models (e.g. CogVideoX, Open-Sora).

    DiT uses a flat sequence of tokens, not spatial feature maps.  This
    adapter projects the (B, in_ch, H, W) scene feature map to a short
    sequence of conditioning tokens (B, n_tokens, hidden) that can be
    injected as cross-attention conditioning or prepended to the token
    sequence.

    Args:
        in_ch:           Input channels from SceneGuideEncoder.
        hidden:          Dimension of each conditioning token.
        n_tokens:        Number of conditioning tokens to produce.
        guide_max_ratio: Amplitude cap for additive injection mode.
    """

    def __init__(
        self,
        in_ch: int = 64,
        hidden: int = 256,
        n_tokens: int = 256,
        guide_max_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_ch           = in_ch
        self.hidden          = hidden
        self.n_tokens        = n_tokens
        self.guide_max_ratio = float(guide_max_ratio)

        # Spatial → token projection:
        # Conv to compress channels, then adaptive pool to n_token grid, flatten.
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
        )

        # Determine grid size for adaptive pooling (closest square root).
        import math
        side = int(math.isqrt(n_tokens))
        self._grid = (side, side)
        actual_tokens = side * side
        if actual_tokens != n_tokens:
            # Use a linear projection to go from actual_tokens → n_tokens
            self.token_proj: Optional[nn.Linear] = nn.Linear(actual_tokens, n_tokens)
        else:
            self.token_proj = None

        # Learnable gate (applied after amplitude normalisation)
        self.gate = nn.Parameter(torch.zeros(1))

        # State
        self._hooks:  list = []
        self._guides: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Scene features → token sequence
    # ------------------------------------------------------------------

    def scene_to_tokens(self, scene_features: torch.Tensor) -> torch.Tensor:
        """Convert (B, in_ch, H, W) scene features to (B, n_tokens, hidden) tokens.

        Args:
            scene_features: Output of SceneGuideEncoder.

        Returns:
            (B, n_tokens, hidden) conditioning token sequence.
        """
        x = self.spatial_proj(scene_features)           # (B, hidden, H, W)
        x = F.adaptive_avg_pool2d(x, self._grid)        # (B, hidden, g, g)
        B, C, g1, g2 = x.shape
        x = x.flatten(2)                                 # (B, hidden, g*g)
        x = x.transpose(1, 2)                            # (B, g*g, hidden)
        if self.token_proj is not None:
            x = x.transpose(1, 2)                        # (B, hidden, g*g)
            x = self.token_proj(x)                       # (B, hidden, n_tokens)
            x = x.transpose(1, 2)                        # (B, n_tokens, hidden)
        return x  # (B, n_tokens, hidden)

    # ------------------------------------------------------------------
    # BaseBackboneAdapter implementation
    # ------------------------------------------------------------------

    def build_guides(
        self, scene_features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Convert scene features to conditioning token sequence.

        Stores result under the key ``"tokens"`` for downstream use.

        Args:
            scene_features: (B, in_ch, H, W).

        Returns:
            ``{"tokens": (B, n_tokens, hidden)}``
        """
        tokens = self.scene_to_tokens(scene_features)
        gate   = torch.tanh(self.gate)
        # Apply gate-weighted amplitude (scale by gate, no spatial normalisation
        # here since token norms vary widely across models)
        tokens = tokens * gate
        self._guides = {"tokens": tokens}
        return self._guides

    def register_hooks(self, model) -> None:
        """Register hooks on a DiT model.

        NOTE: This is a stub.  Concrete implementations must override this
        method with the specific hook attachment logic for their DiT variant
        (CogVideoX, Open-Sora, etc.).
        """
        # No-op stub — override in concrete subclass.
        pass

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def inject(self, h: torch.Tensor, block_name: str) -> torch.Tensor:
        """Inject tokens into a token sequence *h*.

        For DiT this is an additive injection on the token sequence.

        Args:
            h:          (B, seq_len, hidden) token tensor.
            block_name: Key in guides dict (usually ``"tokens"``).

        Returns:
            Tensor with same shape as *h* with conditioning added.
        """
        if block_name not in self._guides:
            return h
        tokens = self._guides[block_name]  # (B, n_tokens, hidden)

        # Broadcast / truncate tokens to h's sequence length if needed
        seq_len = h.shape[1]
        if tokens.shape[1] != seq_len:
            tokens = tokens[:, :seq_len, :]

        # Amplitude normalisation
        h_std   = h.float().std().clamp(min=1e-6)
        t_std   = tokens.float().std().clamp(min=1e-8)
        ratio   = self.guide_max_ratio
        t_norm  = tokens * (ratio * h_std / t_std)
        return h + t_norm.to(h.dtype)
