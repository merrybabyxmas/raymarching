"""scene_prior/entity_field.py
================================
Backbone-agnostic entity field modules for Phase 64.

Key principle: NO AnimateDiff UNet features are used here.
Context is extracted via a lightweight standalone CNN (ImageContextEncoder),
NOT from any diffusion backbone.

Classes
-------
  ImageContextEncoder          — lightweight 3-stage CNN, 1/8 resolution output
  EntityEmbedding              — learnable per-category identity embeddings
  BackboneAgnosticFieldDecoder — density field from (ctx, id, pose, routing)
  ScenePriorModule             — top-level backbone-agnostic scene prior
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scene_prior.scene_outputs import SceneOutputs
from scene_prior.motion_model import MotionModel
from scene_prior.renderer import EntityRenderer
from scene_prior.temporal_memory import TemporalSlotMemory


# ---------------------------------------------------------------------------
# ImageContextEncoder
# ---------------------------------------------------------------------------

class ImageContextEncoder(nn.Module):
    """Lightweight standalone CNN.  NOT a UNet backbone.

    Architecture: 3 stride-2 conv stages.
      RGB (H, W) → (H/2, W/2) → (H/4, W/4) → (H/8, W/8)
    Output channels: ``hidden`` (default 64).

    Parameters
    ----------
    hidden : int
        Output channel count.  Intermediate channels are hidden//2.
    """

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        mid = max(hidden // 2, 32)

        self.net = nn.Sequential(
            # Stage 1: 3 → 32, stride 2
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # Stage 2: 32 → mid, stride 2
            nn.Conv2d(32, mid, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            # Stage 3: mid → hidden, stride 2
            nn.Conv2d(mid, hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img : (B, 3, H, W)

        Returns
        -------
        (B, hidden, H//8, W//8)
        """
        return self.net(img)


# ---------------------------------------------------------------------------
# EntityEmbedding
# ---------------------------------------------------------------------------

class EntityEmbedding(nn.Module):
    """Learnable per-category identity embeddings.

    Supports the 8 known categories listed in ``KNOWN``.  Unknown tokens are
    mapped to a shared "unknown" embedding.

    Parameters
    ----------
    id_dim : int
        Embedding dimension.
    """

    KNOWN: List[str] = [
        "cat", "dog", "wolf", "snake", "alligator",
        "person", "sword", "unknown",
    ]

    def __init__(self, id_dim: int = 128) -> None:
        super().__init__()
        self.id_dim = id_dim
        n = len(self.KNOWN)
        self.emb = nn.Embedding(n, id_dim)
        # index map
        self._idx: dict[str, int] = {k: i for i, k in enumerate(self.KNOWN)}

    def _name_to_idx(self, name: str) -> int:
        return self._idx.get(name.lower(), self._idx["unknown"])

    def embed(self, name: str) -> torch.Tensor:
        """Return embedding for a single entity name.

        Returns
        -------
        (1, id_dim)
        """
        idx = torch.tensor([self._name_to_idx(name)], device=self.emb.weight.device)
        return self.emb(idx)  # (1, id_dim)

    def forward(self, names: List[str]) -> torch.Tensor:
        """Batch embedding lookup.

        Parameters
        ----------
        names : List[str]  length B

        Returns
        -------
        (B, id_dim)
        """
        device = self.emb.weight.device
        idxs = torch.tensor(
            [self._name_to_idx(n) for n in names], device=device
        )  # (B,)
        return self.emb(idxs)  # (B, id_dim)


# ---------------------------------------------------------------------------
# BackboneAgnosticFieldDecoder
# ---------------------------------------------------------------------------

class BackboneAgnosticFieldDecoder(nn.Module):
    """Predict per-entity density field.

    Inputs
    ------
    ctx_feat     : (B, ctx_dim, H, W)   from ImageContextEncoder
    id_feat      : (B, id_dim)          entity identity
    pose_code    : (B, pose_dim)        from MotionModel
    routing_hint : (B, 1, H, W)        color-based spatial prior

    Output
    ------
    density : (B, depth_bins, H, W)  values in [0, 1]

    Implementation note
    -------------------
    id_feat and pose_code are expanded spatially and concatenated with
    ctx_feat.  A small conv stack produces the density logit.  The routing
    hint is converted to logit space and added:
        logit_prior = clamp(log(rh/(1-rh)), -6, 6)
        density = sigmoid(density_logit + logit_prior)
    This gives the model a strong spatial prior at initialisation while still
    allowing the conv layers to learn spatial residuals.
    """

    def __init__(
        self,
        ctx_dim: int = 64,
        id_dim: int = 128,
        pose_dim: int = 32,
        depth_bins: int = 8,
        spatial_h: int = 32,
        spatial_w: int = 32,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.depth_bins = depth_bins
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w

        # Project id + pose to hidden channels
        self.id_proj = nn.Linear(id_dim, hidden)
        self.pose_proj = nn.Linear(pose_dim, hidden)

        # Conv trunk: (ctx_dim + 2*hidden, H, W) → depth logit (depth_bins, H, W)
        in_ch = ctx_dim + 2 * hidden
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, depth_bins, kernel_size=1, bias=True),
        )

        # Routing hint projection (1 → depth_bins)
        self.routing_proj = nn.Conv2d(1, depth_bins, kernel_size=1, bias=False)

    def forward(
        self,
        ctx_feat: torch.Tensor,          # (B, ctx_dim, H, W)
        id_feat: torch.Tensor,           # (B, id_dim)
        pose_code: torch.Tensor,         # (B, pose_dim)
        routing_hint: Optional[torch.Tensor] = None,  # (B, 1, H, W)
    ) -> torch.Tensor:                   # (B, depth_bins, H, W)

        B, _, H, W = ctx_feat.shape

        # Expand id and pose spatially
        id_proj  = self.id_proj(id_feat)                  # (B, hidden)
        pose_prj = self.pose_proj(pose_code)              # (B, hidden)

        id_map   = id_proj.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)    # (B, hidden, H, W)
        pose_map = pose_prj.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)   # (B, hidden, H, W)

        # Concatenate context
        feat = torch.cat([ctx_feat, id_map, pose_map], dim=1)   # (B, ctx+2*hidden, H, W)

        # Density logit from conv trunk
        density_logit = self.trunk(feat)                         # (B, depth_bins, H, W)

        # Add routing logit prior
        if routing_hint is not None:
            # routing_hint ∈ [0, 1]; convert to logit, clamp to avoid ±∞
            rh = routing_hint.float().clamp(1e-4, 1.0 - 1e-4)
            logit_prior = torch.log(rh / (1.0 - rh)).clamp(-6.0, 6.0)  # (B, 1, H, W)
            # project 1-ch prior to depth_bins
            logit_prior_k = self.routing_proj(logit_prior)              # (B, depth_bins, H, W)
            density_logit = density_logit + logit_prior_k

        return torch.sigmoid(density_logit)   # (B, depth_bins, H, W)


# ---------------------------------------------------------------------------
# ScenePriorModule
# ---------------------------------------------------------------------------

class ScenePriorModule(nn.Module):
    """Top-level backbone-agnostic scene prior.

    Inputs
    ------
    img              : (B, 3, H, W)   RGB frame — NOT backbone features
    entity_name_e0   : str (or list[str] for batch)
    entity_name_e1   : str (or list[str] for batch)
    routing_hint_e0  : (B, 1, H, W)  optional color routing map
    routing_hint_e1  : (B, 1, H, W)
    pose_code_e0     : (B, pose_dim)  optional from MotionModel
    pose_code_e1     : (B, pose_dim)
    memory_e0        : (B, slot_dim)  optional GRU state
    memory_e1        : (B, slot_dim)

    Outputs
    -------
    (SceneOutputs, updated_memory_e0, updated_memory_e1)
    """

    def __init__(
        self,
        ctx_dim:    int = 64,
        id_dim:     int = 128,
        pose_dim:   int = 32,
        depth_bins: int = 8,
        spatial_h:  int = 32,
        spatial_w:  int = 32,
        hidden:     int = 64,
        slot_dim:   int = 128,
    ) -> None:
        super().__init__()
        self.ctx_dim    = ctx_dim
        self.id_dim     = id_dim
        self.pose_dim   = pose_dim
        self.depth_bins = depth_bins
        self.spatial_h  = spatial_h
        self.spatial_w  = spatial_w
        self.slot_dim   = slot_dim

        # Sub-modules
        self.ctx_encoder = ImageContextEncoder(hidden=ctx_dim)
        self.entity_emb  = EntityEmbedding(id_dim=id_dim)
        self.motion      = MotionModel(
            id_dim=id_dim, hidden_dim=hidden * 4,
            pose_dim=pose_dim,
        )

        self.decoder_e0 = BackboneAgnosticFieldDecoder(
            ctx_dim=ctx_dim, id_dim=id_dim, pose_dim=pose_dim,
            depth_bins=depth_bins, spatial_h=spatial_h, spatial_w=spatial_w,
            hidden=hidden,
        )
        self.decoder_e1 = BackboneAgnosticFieldDecoder(
            ctx_dim=ctx_dim, id_dim=id_dim, pose_dim=pose_dim,
            depth_bins=depth_bins, spatial_h=spatial_h, spatial_w=spatial_w,
            hidden=hidden,
        )

        self.renderer = EntityRenderer(depth_bins=depth_bins)

        # Temporal slot memories (one per entity)
        # obs_dim = ctx_dim (we pool the context feature map)
        self.memory_e0 = TemporalSlotMemory(slot_dim=slot_dim, obs_dim=ctx_dim)
        self.memory_e1 = TemporalSlotMemory(slot_dim=slot_dim, obs_dim=ctx_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_batch(
        self, name: str | List[str], B: int
    ) -> List[str]:
        if isinstance(name, str):
            return [name] * B
        return list(name)

    def _default_pose(
        self, B: int, device: torch.device
    ) -> torch.Tensor:
        return torch.zeros(B, self.pose_dim, device=device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        img:             torch.Tensor,
        entity_name_e0:  str | List[str],
        entity_name_e1:  str | List[str],
        routing_hint_e0: Optional[torch.Tensor] = None,
        routing_hint_e1: Optional[torch.Tensor] = None,
        pose_code_e0:    Optional[torch.Tensor] = None,
        pose_code_e1:    Optional[torch.Tensor] = None,
        memory_e0:       Optional[torch.Tensor] = None,
        memory_e1:       Optional[torch.Tensor] = None,
    ) -> Tuple[SceneOutputs, torch.Tensor, torch.Tensor]:
        """Run the full backbone-agnostic scene prior for one frame.

        Returns
        -------
        scene_out        : SceneOutputs
        updated_mem_e0   : (B, slot_dim)
        updated_mem_e1   : (B, slot_dim)
        """
        B, _, H, W = img.shape
        device = img.device

        names_e0 = self._ensure_batch(entity_name_e0, B)
        names_e1 = self._ensure_batch(entity_name_e1, B)

        # 1. Encode image context (backbone-agnostic)
        ctx = self.ctx_encoder(img)  # (B, ctx_dim, H/8, W/8)

        # Resize ctx to target spatial resolution if needed
        if ctx.shape[-2:] != (self.spatial_h, self.spatial_w):
            ctx = F.interpolate(
                ctx, size=(self.spatial_h, self.spatial_w),
                mode="bilinear", align_corners=False,
            )  # (B, ctx_dim, spatial_h, spatial_w)

        # 2. Entity identity embeddings
        id_e0 = self.entity_emb(names_e0)   # (B, id_dim)
        id_e1 = self.entity_emb(names_e1)   # (B, id_dim)

        # 3. Pose codes (use provided or generate t=0 pose)
        if pose_code_e0 is None:
            t_zero = torch.zeros(B, 1, device=device)
            pose_code_e0 = self.motion(id_e0, t_zero)   # (B, pose_dim)
        if pose_code_e1 is None:
            t_zero = torch.zeros(B, 1, device=device)
            pose_code_e1 = self.motion(id_e1, t_zero)

        # 4. Resize routing hints to spatial resolution
        if routing_hint_e0 is not None:
            routing_hint_e0 = F.interpolate(
                routing_hint_e0, size=(self.spatial_h, self.spatial_w),
                mode="bilinear", align_corners=False,
            )
        if routing_hint_e1 is not None:
            routing_hint_e1 = F.interpolate(
                routing_hint_e1, size=(self.spatial_h, self.spatial_w),
                mode="bilinear", align_corners=False,
            )

        # 5. Decode density fields
        density_e0 = self.decoder_e0(ctx, id_e0, pose_code_e0, routing_hint_e0)
        density_e1 = self.decoder_e1(ctx, id_e1, pose_code_e1, routing_hint_e1)
        # both: (B, depth_bins, spatial_h, spatial_w)

        # 6. Render scene outputs
        scene_out = self.renderer(density_e0, density_e1)

        # 7. Update temporal slot memories
        if memory_e0 is None:
            memory_e0 = self.memory_e0.init_state(B, device)
        if memory_e1 is None:
            memory_e1 = self.memory_e1.init_state(B, device)

        updated_mem_e0 = self.memory_e0(memory_e0, ctx)
        updated_mem_e1 = self.memory_e1(memory_e1, ctx)

        return scene_out, updated_mem_e0, updated_mem_e1

    # ------------------------------------------------------------------
    # Parameter group helpers (for optimiser construction)
    # ------------------------------------------------------------------

    def field_params(self):
        """Return decoder parameters (BackboneAgnosticFieldDecoder × 2)."""
        return (
            list(self.decoder_e0.parameters())
            + list(self.decoder_e1.parameters())
        )

    def encoder_params(self):
        """Return context encoder + entity embedding parameters."""
        return (
            list(self.ctx_encoder.parameters())
            + list(self.entity_emb.parameters())
            + list(self.motion.parameters())
        )

    def memory_params(self):
        """Return temporal slot memory parameters."""
        return (
            list(self.memory_e0.parameters())
            + list(self.memory_e1.parameters())
        )
