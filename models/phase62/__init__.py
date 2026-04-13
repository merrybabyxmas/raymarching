"""
Phase 62 — Entity Volume Prediction + First-Hit Projection + Guide Assembly.

Modules:
    EntityVolumePredictor  — 3D conv volume from UNet cross-attn features
    FirstHitProjector      — front-to-back scan to 2D visible class
    GuideFeatureAssembler  — entity-conditioned feature selection per pixel
    GuideInjectionManager  — forward hooks for UNet guide injection
    Phase62System          — unified entry point wrapping all above
    BackboneFeatureExtractor — LoRA + slot cross-attention processor
"""
from models.phase62.entity_volume import EntityVolumePredictor
from models.phase62.projection import FirstHitProjector
from models.phase62.conditioning import GuideFeatureAssembler, GuideInjectionManager
from models.phase62.system import Phase62System
from models.phase62.backbone_adapter import BackboneFeatureExtractor

__all__ = [
    "EntityVolumePredictor",
    "FirstHitProjector",
    "GuideFeatureAssembler",
    "GuideInjectionManager",
    "Phase62System",
    "BackboneFeatureExtractor",
]
