"""
test_phase53_decomposition.py — Phase53 explicit decomposition 검증

검증 항목:
  1. decompose_entity_weights는 음수 없이 정규화되며 합이 1이다.
  2. overlap > exclusive > bg ordering이 유지된다.
  3. restore_multiblock_state_p53는 phase53 ckpt의 decomp_heads를 복원한다.
  4. phase52-style ckpt( decomp_heads 없음 )에서는 fresh init이 유지된다.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase53 import (
    DecompositionHeads,
    decompose_entity_weights,
    restore_multiblock_state_p53,
)


class DummyProc(nn.Module):
    def __init__(self, in_dim: int = 8):
        super().__init__()
        self.slot_blend_raw = nn.Parameter(torch.tensor(0.0))
        self.decomp_heads = DecompositionHeads(in_dim=in_dim, proj_dim=4, hidden=8)


def _make_manager(n_procs: int = 2):
    procs = [DummyProc() for _ in range(n_procs)]
    return SimpleNamespace(procs=procs, vca_layer=None, primary=procs[0])


class TestDecomposeEntityWeights:
    def test_normalized_and_non_negative(self):
        p0 = torch.tensor([[0.9, 0.8, 0.2]])
        p1 = torch.tensor([[0.8, 0.1, 0.2]])
        pov = torch.tensor([[0.7, 0.05, 0.0]])
        pfront = torch.tensor([[0.9, 0.2, 0.5]])

        w0, w1, wbg = decompose_entity_weights(p0, p1, pov, pfront)

        total = w0 + w1 + wbg
        assert torch.all(w0 >= 0)
        assert torch.all(w1 >= 0)
        assert torch.all(wbg >= 0)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-6)

    def test_product_floor_prevents_overlap_collapse(self):
        p0 = torch.tensor([[0.95]])
        p1 = torch.tensor([[0.90]])
        pov = torch.tensor([[0.05]])
        pfront = torch.tensor([[0.80]])

        w0, w1, _ = decompose_entity_weights(p0, p1, pov, pfront)
        overlap_mass = (w0 + w1).item()

        assert overlap_mass > 0.7, \
            f"product floor should preserve overlap routing, got {overlap_mass:.4f}"

    def test_union_gate_suppresses_bg_in_foreground(self):
        p0 = torch.tensor([[0.92]])
        p1 = torch.tensor([[0.88]])
        pov = torch.tensor([[0.80]])
        pfront = torch.tensor([[0.75]])

        _, _, wbg = decompose_entity_weights(p0, p1, pov, pfront)
        assert wbg.item() < 0.1, f"foreground union gate should keep bg small, got {wbg.item():.4f}"

    def test_overlap_orders_higher_than_exclusive_and_bg(self):
        p0 = torch.tensor([[0.9, 0.9, 0.1]])
        p1 = torch.tensor([[0.9, 0.1, 0.1]])
        pov = torch.tensor([[0.8, 0.1, 0.0]])
        pfront = torch.tensor([[0.8, 0.8, 0.5]])

        w0, w1, _ = decompose_entity_weights(p0, p1, pov, pfront)

        overlap_score = (w0[0, 0] + w1[0, 0]).item()
        exclusive_score = (w0[0, 1] + w1[0, 1]).item()
        bg_score = (w0[0, 2] + w1[0, 2]).item()

        assert overlap_score > exclusive_score > bg_score


class TestRestoreMultiblockStateP53:
    def test_decomp_heads_loaded_from_phase53_ckpt(self):
        manager_src = _make_manager()
        manager_dst = _make_manager()

        with torch.no_grad():
            manager_src.procs[0].decomp_heads.p0_head[-1].bias.fill_(0.37)
            manager_src.procs[0].decomp_heads.pov_head[-1].weight.fill_(0.11)

        ckpt = {
            "procs_state": [
                {
                    "slot_blend_raw": manager_src.procs[0].slot_blend_raw.detach().clone(),
                    "decomp_heads": manager_src.procs[0].decomp_heads.state_dict(),
                },
                {
                    "slot_blend_raw": manager_src.procs[1].slot_blend_raw.detach().clone(),
                    "decomp_heads": manager_src.procs[1].decomp_heads.state_dict(),
                },
            ]
        }

        restore_multiblock_state_p53(manager_dst, ckpt)

        src_b = manager_src.procs[0].decomp_heads.p0_head[-1].bias
        dst_b = manager_dst.procs[0].decomp_heads.p0_head[-1].bias
        src_w = manager_src.procs[0].decomp_heads.pov_head[-1].weight
        dst_w = manager_dst.procs[0].decomp_heads.pov_head[-1].weight

        assert torch.allclose(dst_b, src_b, atol=1e-6)
        assert torch.allclose(dst_w, src_w, atol=1e-6)

    def test_missing_decomp_heads_keeps_fresh_init(self):
        manager = _make_manager()
        ckpt = {
            "procs_state": [
                {"slot_blend_raw": manager.procs[0].slot_blend_raw.detach().clone()},
                {"slot_blend_raw": manager.procs[1].slot_blend_raw.detach().clone()},
            ]
        }

        restore_multiblock_state_p53(manager, ckpt)

        feat = torch.randn(2, 3, 8)
        with torch.no_grad():
            p0, p1, pov, pfront = manager.procs[0].decomp_heads(feat, feat, feat)

        for p in (p0, p1, pov, pfront):
            assert torch.allclose(p, torch.full_like(p, 0.5), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
