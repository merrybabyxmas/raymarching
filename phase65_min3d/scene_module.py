from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layered_decoder import LayeredEntityDecoder
from .motion_rollout import MotionRollout
from .occlusion_composer import OcclusionComposer
from .scene_outputs import SceneState
from .slot_encoder import EntitySlotEncoder
from .temporal_slots import TemporalSlotMemory


class SceneModule(nn.Module):
    """Top-level entity-separated layered 2.5D scene module.

    v2 adds an explicit camera-conditioning path so the scene prior is forced
    to use view metadata instead of implicitly baking training-view shortcuts
    into the temporal state and decoder.
    """

    def __init__(
        self,
        slot_dim: int = 256,
        feat_dim: int = 64,
        hidden_dim: int = 128,
        Hs: int = 64,
        Ws: int = 64,
        Hf: int = 32,
        Wf: int = 32,
        text_dim: int = 768,
        camera_dim: int = 8,
    ):
        super().__init__()
        self.camera_dim = int(camera_dim)
        self.slot_encoder = EntitySlotEncoder(slot_dim=slot_dim, text_dim=text_dim)
        self.memory = TemporalSlotMemory(slot_dim=slot_dim, obs_dim=slot_dim, hidden_dim=slot_dim)
        self.motion = MotionRollout(slot_dim=slot_dim, hidden_dim=hidden_dim, context_dim=slot_dim)
        self.decoder = LayeredEntityDecoder(
            slot_dim=slot_dim,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            Hs=Hs,
            Ws=Ws,
            Hf=Hf,
            Wf=Wf,
            context_dim=slot_dim,
        )
        self.composer = OcclusionComposer(sharpness=8.0)
        self.prev_frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.obs_proj = nn.Linear(64, slot_dim)
        self.global_proj = nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1)
        self.camera_proj = nn.Sequential(
            nn.Linear(self.camera_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, slot_dim),
        )

    def _encode_prev_obs(self, prev_frame: Optional[torch.Tensor], batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if prev_frame is None:
            z = torch.zeros(batch_size, self.obs_proj.out_features, device=device, dtype=dtype)
            return z, z
        h = self.prev_frame_encoder(prev_frame).flatten(1)
        z = self.obs_proj(h)
        return z, z

    def _encode_camera(self, camera_context: Optional[torch.Tensor], batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.camera_dim <= 0:
            return torch.zeros(batch_size, self.obs_proj.out_features, device=device, dtype=dtype)
        if camera_context is None:
            camera_context = torch.zeros(batch_size, self.camera_dim, device=device, dtype=dtype)
        return self.camera_proj(camera_context.to(device=device, dtype=dtype))

    def forward(
        self,
        entity_names: Sequence[str],
        text_prompt: str,
        prev_state: Optional[SceneState] = None,
        prev_frame: Optional[torch.Tensor] = None,
        t_index: int = 0,
        text_context: Optional[torch.Tensor] = None,
        camera_context: Optional[torch.Tensor] = None,
    ) -> SceneState:
        del text_prompt  # reserved for future richer conditioning
        if prev_frame is not None:
            batch_size = prev_frame.shape[0]
            device = prev_frame.device
            dtype = prev_frame.dtype
        elif prev_state is not None:
            batch_size = prev_state.maps.visible_e0.shape[0]
            device = prev_state.maps.visible_e0.device
            dtype = prev_state.maps.visible_e0.dtype
        else:
            batch_size = 1
            device = self.slot_encoder.name_embed.weight.device
            dtype = self.slot_encoder.name_embed.weight.dtype

        slot_e0, slot_e1 = self.slot_encoder(entity_names, text_context=text_context, batch_size=batch_size, device=device)
        camera_latent = self._encode_camera(camera_context, batch_size, device, dtype)
        # Inject a small amount of camera context directly into the slots to make
        # view-aware decoding easier under held-out camera evaluation.
        slot_e0 = slot_e0 + 0.15 * camera_latent
        slot_e1 = slot_e1 + 0.15 * camera_latent

        obs_e0, obs_e1 = self._encode_prev_obs(prev_frame, batch_size, device, dtype)
        prev_mem_e0 = None if prev_state is None else prev_state.mem_e0
        prev_mem_e1 = None if prev_state is None else prev_state.mem_e1
        mem_e0, mem_e1 = self.memory(prev_mem_e0, prev_mem_e1, slot_e0, slot_e1, obs_e0=obs_e0, obs_e1=obs_e1)
        layout_e0, layout_e1 = self.motion(slot_e0, slot_e1, mem_e0, mem_e1, t_index=t_index, global_context=camera_latent)
        raw_v0, raw_h0, raw_d0, feat_e0 = self.decoder(slot_e0, mem_e0, layout_e0, global_context=camera_latent)
        raw_v1, raw_h1, raw_d1, feat_e1 = self.decoder(slot_e1, mem_e1, layout_e1, global_context=camera_latent)
        global_feat = self.global_proj(torch.cat([feat_e0, feat_e1], dim=1))
        return self.composer(
            raw_v0, raw_h0, raw_d0, feat_e0,
            raw_v1, raw_h1, raw_d1, feat_e1,
            global_feat=global_feat,
            mem_e0=mem_e0,
            mem_e1=mem_e1,
        )
