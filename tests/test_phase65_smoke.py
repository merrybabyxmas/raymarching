from __future__ import annotations

import torch

from phase65_min3d.adapters.decoder_adapter import DecoderSceneAdapter
from phase65_min3d.backbones.reconstruction_decoder import ReconstructionDecoder
from phase65_min3d.scene_module import SceneModule
from phase65_min3d.trainer_stage1 import Stage1Batch, Stage1Trainer
from phase65_min3d.trainer_stage2 import Stage2Batch, Stage2Trainer


def test_phase65_scene_module_smoke() -> None:
    device = "cpu"
    model = SceneModule(slot_dim=64, feat_dim=16, hidden_dim=32, Hs=16, Ws=16, Hf=8, Wf=8).to(device)
    prev_frame = torch.randn(2, 3, 64, 64, device=device)
    scene = model(entity_names=("cat", "dog"), text_prompt="a cat and a dog", prev_frame=prev_frame, t_index=0)
    assert scene.maps.visible_e0.shape == (2, 1, 16, 16)
    assert scene.maps.visible_e1.shape == (2, 1, 16, 16)
    assert scene.maps.amodal_e0.shape == (2, 1, 16, 16)
    assert scene.maps.depth_e0.shape == (2, 1, 16, 16)
    assert scene.features.feat_e0.shape == (2, 16, 8, 8)
    assert scene.mem_e0.shape[0] == 2


def test_phase65_decoder_baseline_smoke() -> None:
    device = "cpu"
    model = SceneModule(slot_dim=64, feat_dim=16, hidden_dim=32, Hs=16, Ws=16, Hf=8, Wf=8).to(device)
    adapter = DecoderSceneAdapter(feat_dim=16, out_dim=32)
    decoder = ReconstructionDecoder(in_dim=32, hidden_dim=32)
    prev_frame = torch.randn(2, 3, 64, 64, device=device)
    scene = model(entity_names=("cat", "dog"), text_prompt="a cat and a dog", prev_frame=prev_frame, t_index=0)
    adapter_out = adapter(scene)
    rgb = decoder(adapter_out["decoder_cond"])
    assert rgb.shape[0] == 2
    assert rgb.shape[1] == 3


def test_phase65_stage1_trainer_smoke() -> None:
    device = "cpu"
    model = SceneModule(slot_dim=32, feat_dim=8, hidden_dim=16, Hs=8, Ws=8, Hf=4, Wf=4).to(device)
    trainer = Stage1Trainer(scene_module=model, device=device, lr=1e-3)
    batch = Stage1Batch(
        entity_names=("cat", "dog"),
        text_prompt="a cat and a dog",
        prev_frame=torch.randn(2, 3, 32, 32),
        gt_visible=torch.rand(2, 2, 8, 8),
        gt_amodal=torch.rand(2, 2, 8, 8),
        t_index=0,
    )
    metrics, state = trainer.step(batch, prev_state=None)
    assert "loss" in metrics
    assert state.maps.visible_e0.shape == (2, 1, 8, 8)


def test_phase65_stage2_trainer_smoke() -> None:
    device = "cpu"
    model = SceneModule(slot_dim=32, feat_dim=8, hidden_dim=16, Hs=8, Ws=8, Hf=4, Wf=4).to(device)
    adapter = DecoderSceneAdapter(feat_dim=8, out_dim=16)
    backbone = ReconstructionDecoder(in_dim=16, hidden_dim=16)
    trainer = Stage2Trainer(scene_module=model, adapter=adapter, backbone=backbone, device=device, lr_scene=1e-3, lr_adapter=1e-3, lr_backbone=1e-3)
    batch = Stage2Batch(
        entity_names=("cat", "dog"),
        text_prompt="a cat and a dog",
        prev_frame=torch.randn(2, 3, 32, 32),
        gt_visible=torch.rand(2, 2, 8, 8),
        gt_amodal=torch.rand(2, 2, 8, 8),
        gt_rgb=torch.rand(2, 3, 8, 8),
        t_index=0,
    )
    metrics, state, pred_rgb = trainer.step(batch, prev_state=None)
    assert "l_backbone" in metrics
    assert pred_rgb.shape[1] == 3
    assert state.maps.amodal_e1.shape == (2, 1, 8, 8)
