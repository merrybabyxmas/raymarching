from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import torch

from .scene_outputs import SceneState


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def scene_state_to_cpu_dict(scene_state: SceneState) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in scene_state.maps.as_dict().items():
        out[f"maps/{k}"] = v.detach().cpu()
    for k, v in scene_state.features.as_dict().items():
        out[f"features/{k}"] = v.detach().cpu()
    if scene_state.mem_e0 is not None:
        out["mem_e0"] = scene_state.mem_e0.detach().cpu()
    if scene_state.mem_e1 is not None:
        out["mem_e1"] = scene_state.mem_e1.detach().cpu()
    return out


def move_scene_state(scene_state: SceneState, device: str | torch.device) -> SceneState:
    maps = type(scene_state.maps)(**{k: v.to(device) for k, v in scene_state.maps.as_dict().items()})
    feats = type(scene_state.features)(**{k: v.to(device) for k, v in scene_state.features.as_dict().items()})
    mem_e0 = None if scene_state.mem_e0 is None else scene_state.mem_e0.to(device)
    mem_e1 = None if scene_state.mem_e1 is None else scene_state.mem_e1.to(device)
    return SceneState(maps=maps, features=feats, mem_e0=mem_e0, mem_e1=mem_e1)
