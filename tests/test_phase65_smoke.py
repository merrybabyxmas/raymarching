from __future__ import annotations

import torch

from phase65_min3d.adapters.decoder_adapter import DecoderSceneAdapter
from phase65_min3d.backbones.reconstruction_decoder import ReconstructionDecoder
from phase65_min3d.scene_module import SceneModule


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
