"""Phase 65 Minimal 3D package.

A clean implementation of an entity-separated layered 2.5D scene model
intended to reduce chimera in contact-heavy multi-entity video generation.
"""

from .scene_outputs import SceneMaps, SceneFeatures, SceneState
from .slot_encoder import EntitySlotEncoder
from .temporal_slots import TemporalSlotMemory
from .motion_rollout import MotionRollout
from .layered_decoder import LayeredEntityDecoder
from .occlusion_composer import OcclusionComposer
from .scene_module import SceneModule

__all__ = [
    "SceneMaps",
    "SceneFeatures",
    "SceneState",
    "EntitySlotEncoder",
    "TemporalSlotMemory",
    "MotionRollout",
    "LayeredEntityDecoder",
    "OcclusionComposer",
    "SceneModule",
]
