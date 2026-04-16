"""adapters/animatediff_adapter.py
===================================
Thin adapter that bridges the backbone-agnostic scene feature map produced by
:class:`~adapters.guide_encoders.SceneGuideEncoder` with an AnimateDiff UNet.

Design decisions (mirrored from models/phase62/conditioning.py v22):
- Per-block projectors resize + project scene features to each UNet block dim.
- Gate initialised to 0 (tanh(0) = 0) → identity at init; learned during training.
- Gate is applied AFTER amplitude normalisation so its gradient is preserved:
    proj_norm = proj(x) * (max_ratio * hs_std / proj_std)   ← gate-free
    guide_eff = proj_norm * gate                              ← gate here
  Without this separation the gate cancels from the gradient path entirely.
- Forward hooks inject guides during the UNet forward pass; call remove_hooks()
  after generation to avoid stale state.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters.base_adapter import BaseBackboneAdapter


# ---------------------------------------------------------------------------
# Block layout
# ---------------------------------------------------------------------------

BLOCK_DIMS: dict[str, int] = {
    "mid":  1280,
    "up1":  1280,
    "up2":   640,
    "up3":   320,
}

BLOCK_SPATIAL: dict[str, tuple[int, int]] = {
    "mid":  (4, 4),
    "up1":  (8, 8),
    "up2":  (16, 16),
    "up3":  (32, 32),
}

DEFAULT_INJECT_BLOCKS: tuple[str, ...] = ("up1", "up2", "up3")  # skip mid by default


# ---------------------------------------------------------------------------
# Amplitude-normalised guide injection (self-contained, no phase62 imports)
# ---------------------------------------------------------------------------

def inject_guide_into_unet_features(
    hidden_states: torch.Tensor,
    guide: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    max_ratio: float = 0.15,
) -> torch.Tensor:
    """Add guide features to UNet hidden states with amplitude normalisation.

    The injection amplitude is bounded by ``max_ratio * std(hidden_states)``
    so the guide cannot dominate the UNet activations regardless of scale.

    When a ``gate`` tensor is provided (v22 path) it is applied *after*
    normalisation so ``d(loss)/d(gate_param)`` remains non-zero throughout
    training.

    Args:
        hidden_states: UNet intermediate features.
                       Shape: (B, C, H, W) or (B, C, T, H, W).
        guide:         Gate-free projected guide. Shape: (B, C, H_b, W_b).
        gate:          Optional scalar gate tensor (on the computation graph).
                       If ``None``, legacy path: gate assumed already in guide.
        max_ratio:     Max guide amplitude as fraction of ``std(hidden_states)``.

    Returns:
        Tensor with the same shape as ``hidden_states``.
    """
    hs_std    = hidden_states.float().std().clamp(min=1e-6)
    guide_std = guide.float().std().clamp(min=1e-8)

    if gate is not None:
        # v22 path: normalise proj BEFORE applying gate → clean gradient.
        guide_normalized = guide * (max_ratio * hs_std / guide_std)
        guide_eff = guide_normalized * gate
    else:
        # Legacy path: gate already baked in; just clip if over-amplitude.
        if guide_std > max_ratio * hs_std:
            guide = guide * (max_ratio * hs_std / guide_std)
        guide_eff = guide

    guide_eff = guide_eff.to(dtype=hidden_states.dtype)

    if hidden_states.dim() == 5:
        # (B, C, T, H, W) — video UNet
        B_hs = hidden_states.shape[0]
        T    = hidden_states.shape[2]
        H_b, W_b = hidden_states.shape[3], hidden_states.shape[4]
        if guide_eff.shape[2] != H_b or guide_eff.shape[3] != W_b:
            guide_eff = F.interpolate(guide_eff, size=(H_b, W_b), mode="nearest")
        if guide_eff.shape[0] != B_hs:
            guide_eff = guide_eff.repeat(B_hs // max(guide_eff.shape[0], 1), 1, 1, 1)
        guide_5d = guide_eff.unsqueeze(2).expand(-1, -1, T, -1, -1)
        return hidden_states + guide_5d
    else:
        # (B, C, H, W)
        B_hs = hidden_states.shape[0]
        H_b, W_b = hidden_states.shape[2], hidden_states.shape[3]
        if guide_eff.shape[2] != H_b or guide_eff.shape[3] != W_b:
            guide_eff = F.interpolate(guide_eff, size=(H_b, W_b), mode="nearest")
        if guide_eff.shape[0] != B_hs:
            guide_eff = guide_eff.repeat(B_hs // max(guide_eff.shape[0], 1), 1, 1, 1)
        return hidden_states + guide_eff


# ---------------------------------------------------------------------------
# AnimateDiffAdapter
# ---------------------------------------------------------------------------

class AnimateDiffAdapter(nn.Module, BaseBackboneAdapter):
    """Thin adapter: scene features (B, in_ch, H, W) → AnimateDiff UNet conditioning.

    Uses forward hooks on AnimateDiff UNet blocks (same pattern as
    ``models/phase62/conditioning.py:GuideInjectionManager``).

    Gate initialised to zeros → starts as identity, learns to use guides.

    Args:
        in_ch:          Input channels from :class:`~adapters.guide_encoders.SceneGuideEncoder`.
        inject_blocks:  Which UNet blocks to inject into.
        guide_max_ratio: Maximum guide amplitude as fraction of block ``std``.
    """

    def __init__(
        self,
        in_ch: int = 64,
        inject_blocks: tuple[str, ...] = DEFAULT_INJECT_BLOCKS,
        guide_max_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        self.inject_blocks    = list(inject_blocks)
        self.guide_max_ratio  = float(guide_max_ratio)

        # Per-block projectors: scene_features → (B, block_dim, H_b, W_b)
        self.projectors = nn.ModuleDict()
        for block_name in self.inject_blocks:
            block_dim = BLOCK_DIMS[block_name]
            self.projectors[block_name] = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Conv2d(in_ch, block_dim, kernel_size=1, bias=True),
            )

        # Per-block learnable gates initialised to 0 → tanh(0) = 0 (identity).
        self.gates = nn.ParameterDict({
            bn: nn.Parameter(torch.zeros(1)) for bn in self.inject_blocks
        })

        # State
        self._hooks:  list  = []
        self._guides: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # BaseBackboneAdapter implementation
    # ------------------------------------------------------------------

    def build_guides(
        self, scene_features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Project scene features to per-block guide tensors.

        Args:
            scene_features: (B, in_ch, H, W) from SceneGuideEncoder.

        Returns:
            Dict mapping block name → (B, block_dim, H_b, W_b) guide tensor.
            Gates are NOT applied here; they are applied during injection so
            the gradient path is preserved.
        """
        guides: dict[str, torch.Tensor] = {}
        for block_name in self.inject_blocks:
            h_b, w_b = BLOCK_SPATIAL[block_name]
            # Resize scene features to block spatial resolution
            if scene_features.shape[2] != h_b or scene_features.shape[3] != w_b:
                feat = F.interpolate(
                    scene_features, size=(h_b, w_b), mode="bilinear", align_corners=False
                )
            else:
                feat = scene_features
            # Project to block channel dim (gate-free output)
            guides[block_name] = self.projectors[block_name](feat)
        self._guides = guides
        return guides

    def register_hooks(self, unet) -> None:
        """Register forward hooks on the AnimateDiff UNet's mid/up blocks.

        Call :meth:`build_guides` before the UNet forward pass so guides are
        ready when the hooks fire.

        Args:
            unet: AnimateDiff UNet model (must have ``mid_block`` and
                  ``up_blocks`` attributes).
        """
        self.remove_hooks()  # clean any stale hooks

        # Map block names → nn.Module references
        block_map: dict[str, object] = {"mid": unet.mid_block}
        for i, up_block in enumerate(unet.up_blocks):
            block_map[f"up{i}"] = up_block

        for block_name in self.inject_blocks:
            module = block_map.get(block_name)
            if module is None:
                continue
            hook = module.register_forward_hook(self._make_hook(block_name))
            self._hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def inject(self, h: torch.Tensor, block_name: str) -> torch.Tensor:
        """Directly inject guide for *block_name* into feature map *h*.

        Useful for unit-testing injection logic without running a full UNet.

        Args:
            h:          (B, C, H, W) feature tensor.
            block_name: Block name key; must exist in current guides dict.

        Returns:
            Injected tensor, same shape as *h*.
        """
        if block_name not in self._guides:
            return h
        guide = self._guides[block_name]
        gate  = torch.tanh(self.gates[block_name])
        return inject_guide_into_unet_features(
            h, guide, gate=gate, max_ratio=self.guide_max_ratio
        )

    # ------------------------------------------------------------------
    # Guide management helpers
    # ------------------------------------------------------------------

    def set_guides(self, guides: dict[str, torch.Tensor]) -> None:
        """Manually set guides (e.g. pre-computed outside build_guides)."""
        self._guides = guides

    def clear_guides(self) -> None:
        """Clear stored guides (call after generation to free memory)."""
        self._guides = {}

    # ------------------------------------------------------------------
    # Hook factory
    # ------------------------------------------------------------------

    def _make_hook(self, block_name: str):
        def hook_fn(module, input, output):
            if block_name not in self._guides:
                return output
            guide = self._guides[block_name]
            gate  = torch.tanh(self.gates[block_name])
            if isinstance(output, tuple):
                h_out = output[0]
                h_out = inject_guide_into_unet_features(
                    h_out, guide, gate=gate, max_ratio=self.guide_max_ratio
                )
                return (h_out,) + output[1:]
            return inject_guide_into_unet_features(
                output, guide, gate=gate, max_ratio=self.guide_max_ratio
            )
        return hook_fn
