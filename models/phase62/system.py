"""
Phase 62 — System Module
==========================

Phase62System: single entry point wrapping volume predictor, first-hit
projector, guide assembler, and injection manager.

Does NOT contain the UNet — it wraps around UNet-extracted features.

Usage:
    system = Phase62System(config)
    system.injection_mgr.register_hooks(unet)

    # After UNet forward (features extracted by BackboneFeatureExtractor):
    V_logits = system.predict_volume(F_g, F_0, F_1)
    visible_class, front_probs, back_probs, guides = system.project_and_assemble(
        V_logits, F_g, F_0, F_1)
    system.set_guides(guides)
    # Next UNet forward will have guides injected via hooks
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.phase62.entity_volume import EntityVolumePredictor
from models.phase62.projection import FirstHitProjector
from models.phase62.conditioning import GuideFeatureAssembler, GuideInjectionManager


class Phase62System(nn.Module):
    """
    Complete Phase62 system: volume prediction + projection + guide assembly.

    This is the single entry point for the Phase62 model.
    Does NOT contain the UNet — it wraps around UNet features.

    Components:
        volume_pred:    EntityVolumePredictor — F_g,F_0,F_1 -> V_logits (B,3,K,H,W)
        projector:      FirstHitProjector — V_logits -> visible_class, visible_probs
        assembler:      GuideFeatureAssembler — entity-conditioned feature selection
        injection_mgr:  GuideInjectionManager — forward hooks for UNet injection
    """

    def __init__(self, config):
        """
        Initialize Phase62 system from config namespace.

        Expected config attributes:
            config.depth_bins: int (8)
            config.hidden_dim: int (64)
            config.spatial_h: int (16)
            config.spatial_w: int (16)
            config.n_classes: int (3)
            config.inject_config: str ('mid_up2')
        """
        super().__init__()

        # Resolve feat_dim from backbone primary block
        # up_blocks.2 (primary) has inner_dim=640
        feat_dim = 640

        self.volume_pred = EntityVolumePredictor(
            feat_dim=feat_dim,
            n_classes=config.n_classes,
            depth_bins=config.depth_bins,
            spatial_h=config.spatial_h,
            spatial_w=config.spatial_w,
            hidden=config.hidden_dim,
        )

        self.projector = FirstHitProjector(
            n_classes=config.n_classes,
            bg_class=0,
        )

        self.assembler = GuideFeatureAssembler(
            feat_dim=feat_dim,
            hidden=config.hidden_dim,
            spatial_h=config.spatial_h,
            spatial_w=config.spatial_w,
            n_classes=config.n_classes,
            inject_config=config.inject_config,
        )

        self.injection_mgr = GuideInjectionManager(
            inject_config=config.inject_config,
        )

    def predict_volume(
        self,
        F_g: torch.Tensor,   # (B, S, D)
        F_0: torch.Tensor,   # (B, S, D)
        F_1: torch.Tensor,   # (B, S, D)
    ) -> torch.Tensor:
        """
        Predict 3D entity volume from cross-attention features.

        Returns:
            V_logits: (B, n_classes, K, H, W)
        """
        return self.volume_pred(F_g, F_0, F_1)

    def project_and_assemble(
        self,
        V_logits: torch.Tensor,   # (B, C, K, H, W)
        F_g: torch.Tensor,        # (B, S, D)
        F_0: torch.Tensor,        # (B, S, D)
        F_1: torch.Tensor,        # (B, S, D)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        First-hit project then assemble entity-conditioned guide.

        Returns:
            visible_class:  (B, H, W) int64
            front_probs:    (B, C, H, W) float
            back_probs:     (B, C, H, W) float
            guides:         dict[block_name -> (B, block_dim, H_b, W_b)]
        """
        visible_class, front_probs, back_probs = self.projector(V_logits)
        guides = self.assembler(visible_class, front_probs, back_probs, F_g, F_0, F_1)
        return visible_class, front_probs, back_probs, guides

    def set_guides(self, guides: Dict[str, torch.Tensor]) -> None:
        """Set guide features for injection at next UNet forward."""
        self.injection_mgr.set_guides(guides)

    def clear_guides(self) -> None:
        """Clear guide features (no injection)."""
        self.injection_mgr.clear_guides()

    def volume_params(self):
        """Parameters of the volume predictor."""
        return list(self.volume_pred.parameters())

    def assembler_params(self):
        """Parameters of the guide assembler (replaces guide_injector params)."""
        return list(self.assembler.parameters())
