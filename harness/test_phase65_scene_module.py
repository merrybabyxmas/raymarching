from __future__ import annotations

import torch

from phase65_min3d.scene_module import SceneModule


def test_phase65_scene_module_shapes() -> None:
    model = SceneModule(slot_dim=128, feat_dim=32, hidden_dim=64, Hs=32, Ws=32, Hf=16, Wf=16)
    prev_frame = torch.randn(2, 3, 128, 128)
    out = model(
        entity_names=("cat", "dog"),
        text_prompt="a cat and a dog",
        prev_state=None,
        prev_frame=prev_frame,
        t_index=0,
    )
    assert out.maps.visible_e0.shape == (2, 1, 32, 32)
    assert out.maps.visible_e1.shape == (2, 1, 32, 32)
    assert out.maps.hidden_e0.shape == (2, 1, 32, 32)
    assert out.maps.hidden_e1.shape == (2, 1, 32, 32)
    assert out.maps.amodal_e0.shape == (2, 1, 32, 32)
    assert out.maps.amodal_e1.shape == (2, 1, 32, 32)
    assert out.maps.depth_e0.shape == (2, 1, 32, 32)
    assert out.maps.depth_e1.shape == (2, 1, 32, 32)
    assert out.features.feat_e0.shape == (2, 32, 16, 16)
    assert out.features.feat_e1.shape == (2, 32, 16, 16)
    assert out.mem_e0.shape == (2, 128)
    assert out.mem_e1.shape == (2, 128)
