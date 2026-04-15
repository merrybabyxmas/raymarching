"""
Phase 62 — System Module
==========================

Phase62System: single entry point wrapping volume predictor, first-hit
projector, guide assembler, and injection manager.

Supports multiple representations, guide families, and objective families
via config-driven construction.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.phase62.entity_volume import EntityVolumePredictor
from models.phase62.projection import FirstHitProjector
from models.phase62.conditioning import GuideFeatureAssembler, GuideInjectionManager
from training.phase62.objectives.base import VolumeOutputs

# Phase 63 modules are imported lazily inside Phase63System.__init__ to avoid
# a circular import chain (models.guide_encoder imports from
# models.phase62.conditioning, whose package __init__ pulls in this file).


class Phase62System(nn.Module):

    def __init__(self, config):
        super().__init__()

        feat_dim = int(getattr(config, "feat_dim", 640))
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

        gate_warm_start = float(getattr(config, "gate_warm_start", 0.0))
        self.assembler = GuideFeatureAssembler(
            feat_dim=feat_dim,
            hidden=config.hidden_dim,
            spatial_h=config.spatial_h,
            spatial_w=config.spatial_w,
            n_classes=config.n_classes,
            inject_config=config.inject_config,
            guide_family=guide_family,
            gate_warm_start=gate_warm_start,
        )

        guide_max_ratio = float(getattr(config, "guide_max_ratio", 0.1))
        self.injection_mgr = GuideInjectionManager(
            inject_config=config.inject_config,
            guide_max_ratio=guide_max_ratio,
        )

    def predict_volume(
        self,
        F_g: torch.Tensor,
        F_0: torch.Tensor,
        F_1: torch.Tensor,
        depth_hint: "Optional[torch.Tensor]" = None,  # (B, H_vol, W_vol) ∈ [0,1]
    ) -> VolumeOutputs:
        return self.volume_pred(F_g, F_0, F_1, depth_hint=depth_hint)

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
        # v22: pass gate_fn so injection manager applies gate AFTER amplitude
        # normalisation, preserving the gate gradient path.
        def gate_fn(block_name: str) -> "Optional[torch.Tensor]":
            return self.assembler.get_gate(block_name)
        self.injection_mgr.set_guides(guides, gate_fn=gate_fn)

    def clear_guides(self) -> None:
        self.injection_mgr.clear_guides()

    def volume_params(self):
        return list(self.volume_pred.parameters())

    def assembler_params(self):
        return list(self.assembler.parameters())


# ===========================================================================
# Phase 63 — Independent Entity Fields + Transmittance Rendering
# ===========================================================================
class Phase63System(nn.Module):
    """
    Phase 63 system: replaces Phase 62's class-competition volume with fully
    independent per-entity density fields and a differentiable transmittance
    renderer.

    Pipeline
    --------
        EntityField          # independent sigma_0, sigma_1
            -> TransmittanceRenderer       # visible / amodal / depth
            -> StructuredGuideEncoder      # 5-stream normalised guide
            -> GuideInjectionManager       # inject into UNet (hooks)

    Config attributes expected
    --------------------------
    feat_dim, hidden_dim, depth_bins, spatial_h, spatial_w,
    inject_config, guide_max_ratio (default 0.1),
    gate_warm_start (default 0.0),
    app_dim (default 32),
    n_refine_blocks (default 3)
    """

    def __init__(self, config):
        super().__init__()

        # Lazy imports to avoid circular imports at package load time.
        from models.entity_field import EntityField
        from models.renderer import TransmittanceRenderer
        from models.guide_encoder import StructuredGuideEncoder

        feat_dim = int(getattr(config, "feat_dim", 640))
        hidden = int(getattr(config, "hidden_dim", 64))
        depth_bins = int(getattr(config, "depth_bins", 8))
        spatial_h = int(getattr(config, "spatial_h", 16))
        spatial_w = int(getattr(config, "spatial_w", 16))
        inject_config = getattr(config, "inject_config", "multiscale")
        guide_max_ratio = float(getattr(config, "guide_max_ratio", 0.1))
        gate_warm_start = float(getattr(config, "gate_warm_start", 0.0))
        app_dim = int(getattr(config, "app_dim", 32))
        n_refine_blocks = int(getattr(config, "n_refine_blocks", 3))

        self.field = EntityField(
            feat_dim=feat_dim,
            hidden=hidden,
            depth_bins=depth_bins,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            app_dim=app_dim,
            n_refine_blocks=n_refine_blocks,
        )

        self.renderer = TransmittanceRenderer(bg_class=0)

        self.guide_encoder = StructuredGuideEncoder(
            feat_dim=feat_dim,
            hidden=hidden,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            inject_config=inject_config,
            gate_warm_start=gate_warm_start,
        )

        self.injection_mgr = GuideInjectionManager(
            inject_config=inject_config,
            guide_max_ratio=guide_max_ratio,
        )

    # ------------------------------------------------------------------
    # Forward helpers (mirror Phase62System's split API)
    # ------------------------------------------------------------------
    def forward_field_and_render(
        self,
        F_g: torch.Tensor,
        F_e0: torch.Tensor,
        F_e1: torch.Tensor,
        depth_hint: "Optional[torch.Tensor]" = None,
        img_hint_e0: "Optional[torch.Tensor]" = None,  # (B,1,H,W) color routing map
        img_hint_e1: "Optional[torch.Tensor]" = None,
    ) -> "Tuple[EntityFieldOutputs, RendererOutputs]":
        field_out = self.field(
            F_g, F_e0, F_e1, depth_hint=depth_hint,
            img_hint_e0=img_hint_e0, img_hint_e1=img_hint_e1,
        )
        render_out = self.renderer(field_out.density_e0, field_out.density_e1)
        return field_out, render_out

    def encode_guide(
        self,
        render_out: "RendererOutputs",
        field_out: "EntityFieldOutputs",
        F_e0: torch.Tensor,
        F_e1: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.guide_encoder(render_out, field_out, F_e0, F_e1)

    # ------------------------------------------------------------------
    # Injection management (same semantics as Phase62System)
    # ------------------------------------------------------------------
    def set_guides(self, guides: Dict[str, torch.Tensor]) -> None:
        def gate_fn(block_name: str) -> "Optional[torch.Tensor]":
            return self.guide_encoder.get_gate(block_name)
        self.injection_mgr.set_guides(guides, gate_fn=gate_fn)

    def clear_guides(self) -> None:
        self.injection_mgr.clear_guides()

    # ------------------------------------------------------------------
    # Parameter groups — useful for per-group LR scheduling.
    # ------------------------------------------------------------------
    def field_params(self):
        return list(self.field.parameters())

    def guide_encoder_params(self):
        return list(self.guide_encoder.parameters())
