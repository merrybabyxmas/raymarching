from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .evaluator import Phase65Evaluator
from .losses import (
    amodal_loss,
    depth_ordering_loss,
    occlusion_consistency_loss,
    slot_separation_loss,
    temporal_identity_loss,
    visible_loss,
)
from .scene_module import SceneModule
from .scene_outputs import SceneState


@dataclass
class Stage1Batch:
    entity_names: tuple[str, str]
    text_prompt: str
    prev_frame: Optional[torch.Tensor]
    gt_visible: torch.Tensor
    gt_amodal: torch.Tensor
    gt_front_idx: Optional[torch.Tensor] = None
    t_index: int = 0
    camera_context: Optional[torch.Tensor] = None


class Stage1Trainer:
    """Minimal stage-1 trainer for the Scene Module only."""

    def __init__(
        self,
        scene_module: SceneModule,
        device: str = 'cuda',
        lr: float = 3e-4,
        lambda_vis: float = 1.0,
        lambda_amo: float = 1.0,
        lambda_temp: float = 0.25,
        lambda_depth: float = 0.1,
        lambda_sep: float = 0.0,
        grad_clip: float = 1.0,
    ):
        self.scene_module = scene_module.to(device)
        self.device = device
        self.optimizer = optim.AdamW(scene_module.parameters(), lr=lr, weight_decay=1e-4)
        self.evaluator = Phase65Evaluator()
        self.lambda_vis = lambda_vis
        self.lambda_amo = lambda_amo
        self.lambda_temp = lambda_temp
        self.lambda_depth = lambda_depth
        self.lambda_sep = lambda_sep
        self.grad_clip = grad_clip

    def step(self, batch: Stage1Batch, prev_state: Optional[SceneState] = None) -> tuple[Dict[str, float], SceneState]:
        self.scene_module.train()
        gt_visible = batch.gt_visible.to(self.device)
        gt_amodal = batch.gt_amodal.to(self.device)
        prev_frame = None if batch.prev_frame is None else batch.prev_frame.to(self.device)
        gt_front_idx = None if batch.gt_front_idx is None else batch.gt_front_idx.to(self.device)
        camera_context = None if batch.camera_context is None else batch.camera_context.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        scene_state = self.scene_module(
            entity_names=batch.entity_names,
            text_prompt=batch.text_prompt,
            prev_state=prev_state,
            prev_frame=prev_frame,
            t_index=batch.t_index,
            camera_context=camera_context,
        )
        l_vis = visible_loss(scene_state, gt_visible)
        l_amo = amodal_loss(scene_state, gt_amodal)
        l_occ = occlusion_consistency_loss(scene_state)
        l_temp = temporal_identity_loss(scene_state, prev_state)
        l_depth = depth_ordering_loss(scene_state, gt_front_idx=gt_front_idx)
        l_sep = slot_separation_loss(scene_state)
        loss = (
            self.lambda_vis * l_vis
            + self.lambda_amo * (l_amo + 0.25 * l_occ)
            + self.lambda_temp * l_temp
            + self.lambda_depth * l_depth
            + self.lambda_sep * l_sep
        )
        loss.backward()
        nn.utils.clip_grad_norm_(self.scene_module.parameters(), self.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            metrics = self.evaluator.evaluate_scene(scene_state, gt_visible, gt_amodal, prev_state=prev_state)
        metrics.update({
            'loss': float(loss.item()),
            'l_vis': float(l_vis.item()),
            'l_amo': float(l_amo.item()),
            'l_occ': float(l_occ.item()),
            'l_temp': float(l_temp.item()),
            'l_depth': float(l_depth.item()),
            'l_sep': float(l_sep.item()),
        })
        return metrics, scene_state.detach()
