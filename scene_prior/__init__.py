"""scene_prior — backbone-agnostic scene decomposition system for Phase 64.

Public API
----------
SceneOutputs          — canonical 8-channel portable scene representation
EntitySpec            — single entity specification (name, attributes, color)
ScenePrompt           — parsed two-entity scene description
parse_prompt          — text → ScenePrompt
ScenePriorModule      — top-level backbone-agnostic scene prior (main model)
EntityRenderer        — differentiable transmittance renderer
TemporalSlotMemory    — per-entity GRU slot memory across frames
Losses                — see individual functions below

Key design principle:  nothing in this package imports from AnimateDiff,
diffusion UNet, or any specific generative backbone.
"""
from __future__ import annotations

from scene_prior.scene_outputs import SceneOutputs, RendererOutputs
from scene_prior.entity_parser import EntitySpec, ScenePrompt, parse_prompt
from scene_prior.entity_field import ScenePriorModule
from scene_prior.renderer import EntityRenderer
from scene_prior.temporal_memory import TemporalSlotMemory
from scene_prior.losses import (
    dice_loss,
    loss_visible,
    loss_amodal,
    loss_occlusion,
    loss_survival,
    loss_identity_contrastive,
    loss_reappearance,
    loss_separation,
    loss_color_routing,
    total_scene_loss,
)

__all__ = [
    # Data
    "SceneOutputs",
    "RendererOutputs",
    # Parser
    "EntitySpec",
    "ScenePrompt",
    "parse_prompt",
    # Model
    "ScenePriorModule",
    "EntityRenderer",
    "TemporalSlotMemory",
    # Losses
    "dice_loss",
    "loss_visible",
    "loss_amodal",
    "loss_occlusion",
    "loss_survival",
    "loss_identity_contrastive",
    "loss_reappearance",
    "loss_separation",
    "loss_color_routing",
    "total_scene_loss",
]
