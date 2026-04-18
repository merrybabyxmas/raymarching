from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class SceneMaps:
    visible_e0: torch.Tensor
    visible_e1: torch.Tensor
    hidden_e0: torch.Tensor
    hidden_e1: torch.Tensor
    amodal_e0: torch.Tensor
    amodal_e1: torch.Tensor
    depth_e0: torch.Tensor
    depth_e1: torch.Tensor
    contact: Optional[torch.Tensor] = None

    def as_dict(self) -> Dict[str, torch.Tensor]:
        out = {
            'visible_e0': self.visible_e0,
            'visible_e1': self.visible_e1,
            'hidden_e0': self.hidden_e0,
            'hidden_e1': self.hidden_e1,
            'amodal_e0': self.amodal_e0,
            'amodal_e1': self.amodal_e1,
            'depth_e0': self.depth_e0,
            'depth_e1': self.depth_e1,
        }
        if self.contact is not None:
            out['contact'] = self.contact
        return out


@dataclass
class SceneFeatures:
    feat_e0: torch.Tensor
    feat_e1: torch.Tensor
    global_feat: Optional[torch.Tensor] = None

    def as_dict(self) -> Dict[str, torch.Tensor]:
        out = {
            'feat_e0': self.feat_e0,
            'feat_e1': self.feat_e1,
        }
        if self.global_feat is not None:
            out['global_feat'] = self.global_feat
        return out


@dataclass
class SceneState:
    maps: SceneMaps
    features: SceneFeatures
    mem_e0: Optional[torch.Tensor] = None
    mem_e1: Optional[torch.Tensor] = None
    slot_e0: Optional[torch.Tensor] = None
    slot_e1: Optional[torch.Tensor] = None

    def detach(self) -> 'SceneState':
        def _d(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if x is None else x.detach()

        maps = SceneMaps(**{k: _d(v) for k, v in self.maps.as_dict().items()})
        feats = SceneFeatures(**{k: _d(v) for k, v in self.features.as_dict().items()})
        return SceneState(
            maps=maps,
            features=feats,
            mem_e0=_d(self.mem_e0),
            mem_e1=_d(self.mem_e1),
            slot_e0=_d(self.slot_e0),
            slot_e1=_d(self.slot_e1),
        )
