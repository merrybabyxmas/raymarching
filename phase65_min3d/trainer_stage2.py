from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .adapters.base import BaseSceneAdapter
from .backbones.reconstruction_decoder import ReconstructionDecoder
from .evaluator import Phase65Evaluator
from .losses import amodal_loss, depth_ordering_loss, occlusion_consistency_loss, reconstruction_loss, temporal_identity_loss, visible_loss
from .scene_module import SceneModule
from .scene_outputs import SceneState


@dataclass
class Stage2Batch:
    entity_names: tuple[str, str]
    text_prompt: str
    prev_frame: Optional[torch.Tensor]
    gt_visible: torch.Tensor
    gt_amodal: torch.Tensor
    gt_rgb: torch.Tensor
    gt_front_idx: Optional[torch.Tensor] = None
    t_index: int = 0


class Stage2Trainer:
    """Align a backbone with the learned Scene Module.

    The default backbone is a small reconstruction decoder baseline. Scene losses
    remain active so the backbone cannot drag the scene representation into a
    one-object local optimum.
    """

    def __init__(
        self,
        scene_module: SceneModule,
        adapter: BaseSceneAdapter,
        backbone: ReconstructionDecoder,
        device: str = "cuda",
        lr_scene: float = 5e-5,
        lr_adapter: float = 2e-4,
        lr_backbone: float = 1e-4,
        lambda_backbone: float = 1.0,
        lambda_vis: float = 0.5,
        lambda_amo: float = 0.5,
        lambda_temp: float = 0.1,
        lambda_depth: float = 0.05,
        grad_clip: float = 1.0,
    ):
        self.scene_module = scene_module.to(device)
        self.adapter = adapter.to(device)
        self.backbone = backbone.to(device)
        self.device = device
        self.optimizer = optim.AdamW([
            {"params": self.scene_module.parameters(), "lr": lr_scene},
            {"params": self.adapter.parameters(), "lr": lr_adapter},
            {"params": self.backbone.parameters(), "lr": lr_backbone},
        ], weight_decay=1e-4)
        self.evaluator = Phase65Evaluator()
        self.lambda_backbone = lambda_backbone
        self.lambda_vis = lambda_vis
        self.lambda_amo = lambda_amo
        self.lambda_temp = lambda_temp
        self.lambda_depth = lambda_depth
        self.grad_clip = grad_clip

    def step(self, batch: Stage2Batch, prev_state: Optional[SceneState] = None) -> tuple[Dict[str, float], SceneState, torch.Tensor]:
        self.scene_module.train()
        self.adapter.train()
        self.backbone.train()

        gt_visible = batch.gt_visible.to(self.device)
        gt_amodal = batch.gt_amodal.to(self.device)
        gt_rgb = batch.gt_rgb.to(self.device)
        prev_frame = None if batch.prev_frame is None else batch.prev_frame.to(self.device)
        gt_front_idx = None if batch.gt_front_idx is None else batch.gt_front_idx.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        scene_state = self.scene_module(
            entity_names=batch.entity_names,
            text_prompt=batch.text_prompt,
            prev_state=prev_state,
            prev_frame=prev_frame,
            t_index=batch.t_index,
        )
        adapter_out = self.adapter(scene_state)
        if "decoder_cond" not in adapter_out:
            raise KeyError("Stage2Trainer currently expects adapter output key 'decoder_cond'")
        pred_rgb = self.backbone(adapter_out["decoder_cond"])

        l_backbone = reconstruction_loss(pred_rgb, gt_rgb)
        l_vis = visible_loss(scene_state, gt_visible)
        l_amo = amodal_loss(scene_state, gt_amodal)
        l_occ = occlusion_consistency_loss(scene_state)
        l_temp = temporal_identity_loss(scene_state, prev_state)
        l_depth = depth_ordering_loss(scene_state, gt_front_idx=gt_front_idx)

        loss = (
            self.lambda_backbone * l_backbone
            + self.lambda_vis * l_vis
            + self.lambda_amo * (l_amo + 0.25 * l_occ)
            + self.lambda_temp * l_temp
            + self.lambda_depth * l_depth
        )
        loss.backward()
        nn.utils.clip_grad_norm_(self.scene_module.parameters(), self.grad_clip)
        nn.utils.clip_grad_norm_(self.adapter.parameters(), self.grad_clip)
        nn.utils.clip_grad_norm_(self.backbone.parameters(), self.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            metrics = self.evaluator.evaluate_scene(scene_state, gt_visible, gt_amodal, prev_state=prev_state)
        metrics.update({
            "loss": float(loss.item()),
            "l_backbone": float(l_backbone.item()),
            "l_vis": float(l_vis.item()),
            "l_amo": float(l_amo.item()),
            "l_occ": float(l_occ.item()),
            "l_temp": float(l_temp.item()),
            "l_depth": float(l_depth.item()),
        })
        return metrics, scene_state.detach(), pred_rgb.detach()
