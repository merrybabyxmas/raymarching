"""
Phase 62 — System Module
==========================

Phase62System: single entry point wrapping volume predictor, first-hit
projector, guide assembler, and injection manager.

Supports multiple representations, guide families, and objective families
via config-driven construction.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from models.phase62.entity_volume import EntityVolumePredictor
from models.phase62.projection import FirstHitProjector
from models.phase62.conditioning import GuideFeatureAssembler, GuideInjectionManager
from training.phase62.objectives.base import VolumeOutputs


class Phase62System(nn.Module):

    def __init__(self, config):
        super().__init__()

        feat_dim = 640
        representation = getattr(config, "representation", "independent")
        guide_family = getattr(config, "guide_family", "dual") or "none"

        self.volume_pred = EntityVolumePredictor(
            feat_dim=feat_dim,
            n_classes=config.n_classes,
            depth_bins=config.depth_bins,
            spatial_h=config.spatial_h,
            spatial_w=config.spatial_w,
            hidden=config.hidden_dim,
            representation=representation,
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
            guide_family=guide_family,
        )

        self.injection_mgr = GuideInjectionManager(
            inject_config=config.inject_config,
        )

    def predict_volume(
        self,
        F_g: torch.Tensor,
        F_0: torch.Tensor,
        F_1: torch.Tensor,
    ) -> VolumeOutputs:
        return self.volume_pred(F_g, F_0, F_1)

    def project_and_assemble(
        self,
        vol_outputs: VolumeOutputs,
        F_g: torch.Tensor,
        F_0: torch.Tensor,
        F_1: torch.Tensor,
    ) -> Tuple[VolumeOutputs, Dict[str, torch.Tensor]]:
        vol_outputs = self.projector(vol_outputs)
        guides = self.assembler(vol_outputs, F_g, F_0, F_1)
        return vol_outputs, guides

    def set_guides(self, guides: Dict[str, torch.Tensor]) -> None:
        self.injection_mgr.set_guides(guides)

    def clear_guides(self) -> None:
        self.injection_mgr.clear_guides()

    def volume_params(self):
        return list(self.volume_pred.parameters())

    def assembler_params(self):
        return list(self.assembler.parameters())
