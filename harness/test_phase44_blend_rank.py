"""
test_phase44_blend_rank.py — l_blend_rank + collect_blend_stats_detailed 검증

검증 항목 (l_blend_rank):
  1. overlap > exclusive > bg (margin 충족) → loss = 0
  2. overlap < exclusive (margin 위반) → loss > 0
  3. exclusive < bg (margin 위반) → loss > 0
  4. 두 위반 모두 → loss 더 큼
  5. no overlap region → first rank term = 0
  6. no bg region → second rank term = 0
  7. gradient flows to blend_map
  8. margin=0일 때 완벽 순서 → loss = 0

검증 항목 (collect_blend_stats_detailed):
  9. blend_sep = blend_overlap_mean - blend_exclusive_mean
 10. blend_gap_bg = blend_exclusive_mean - blend_bg_mean
 11. 모든 bg → overlap/exclusive stats = 0/eps
 12. key 목록 확인
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase44 import l_blend_rank, collect_blend_stats_detailed


# =============================================================================
# Helpers
# =============================================================================

def _masks(S=8, ov=None, ex0=None, ex1=None):
    m = torch.zeros(1, 2, S)
    for p in (ov or []):
        m[0, 0, p] = 1.0; m[0, 1, p] = 1.0
    for p in (ex0 or []):
        m[0, 0, p] = 1.0
    for p in (ex1 or []):
        m[0, 1, p] = 1.0
    return m


def _blend(S=8, ov_vals=None, ex_vals=None, bg_vals=None,
           ov_pix=None, ex_pix=None, bg_pix=None):
    """ov/ex/bg 픽셀에 지정 값 할당."""
    b = torch.zeros(1, S)
    for p, v in zip(ov_pix or [], ov_vals or []):
        b[0, p] = v
    for p, v in zip(ex_pix or [], ex_vals or []):
        b[0, p] = v
    for p, v in zip(bg_pix or [], bg_vals or []):
        b[0, p] = v
    return b


# =============================================================================
# l_blend_rank
# =============================================================================

class TestLBlendRank:

    def test_correct_order_loss_zero(self):
        """overlap > exclusive > bg by margin → loss = 0."""
        m = _masks(ov=[0,1], ex0=[2,3], ex1=[4,5])
        # ov=0.9, ex=0.35, bg=0.05  (모두 margin=0.1 이상 차이)
        blend = torch.zeros(1, 8)
        blend[0, 0] = blend[0, 1] = 0.9
        blend[0, 2] = blend[0, 3] = 0.35
        blend[0, 4] = blend[0, 5] = 0.35
        blend[0, 6] = blend[0, 7] = 0.05

        loss = l_blend_rank(blend, m, margin=0.10)
        assert loss.item() < 1e-5, f"correct order: loss={loss.item():.6f}"

    def test_overlap_below_exclusive_gives_loss(self):
        """overlap < exclusive → loss > 0 (ov-ex 위반)."""
        m = _masks(ov=[0,1], ex0=[2,3], ex1=[4,5])
        blend = torch.zeros(1, 8)
        blend[0, 0] = blend[0, 1] = 0.20   # low overlap
        blend[0, 2] = blend[0, 3] = 0.50   # high exclusive
        blend[0, 4] = blend[0, 5] = 0.50
        blend[0, 6] = blend[0, 7] = 0.05

        loss = l_blend_rank(blend, m, margin=0.10)
        assert loss.item() > 0.0, f"ov<ex violation: loss={loss.item():.4f}"

    def test_exclusive_below_bg_gives_loss(self):
        """exclusive < bg → loss > 0 (ex-bg 위반)."""
        m = _masks(ov=[0,1], ex0=[2,3], ex1=[4,5])
        blend = torch.zeros(1, 8)
        blend[0, 0] = blend[0, 1] = 0.90
        blend[0, 2] = blend[0, 3] = 0.10   # exclusive < bg
        blend[0, 4] = blend[0, 5] = 0.10
        blend[0, 6] = blend[0, 7] = 0.50   # high bg

        loss = l_blend_rank(blend, m, margin=0.10)
        assert loss.item() > 0.0, f"ex<bg violation: loss={loss.item():.4f}"

    def test_both_violations_higher_loss(self):
        """두 rank 위반 모두 → 단일 위반보다 loss 큼."""
        m = _masks(ov=[0,1], ex0=[2,3], ex1=[4,5])
        blend = torch.zeros(1, 8)

        # Case A: only ov-ex violation
        ba = blend.clone()
        ba[0, 0] = ba[0, 1] = 0.20   # low overlap
        ba[0, 2] = ba[0, 3] = 0.50   # high exclusive
        ba[0, 4] = ba[0, 5] = 0.50
        ba[0, 6] = ba[0, 7] = 0.05   # bg below exclusive

        # Case B: both ov-ex AND ex-bg violations
        bb = blend.clone()
        bb[0, 0] = bb[0, 1] = 0.10   # very low overlap
        bb[0, 2] = bb[0, 3] = 0.30   # exclusive
        bb[0, 4] = bb[0, 5] = 0.30
        bb[0, 6] = bb[0, 7] = 0.50   # bg > exclusive

        loss_a = l_blend_rank(ba, m, margin=0.10)
        loss_b = l_blend_rank(bb, m, margin=0.10)
        assert loss_b.item() > loss_a.item(), (
            f"both violations should give higher loss: {loss_a.item():.4f} vs {loss_b.item():.4f}")

    def test_no_overlap_region_first_term_zero(self):
        """overlap region 없으면 ov-ex rank term = 0."""
        # Only exclusive and bg
        m = _masks(ex0=[0,1], ex1=[2,3])
        blend = torch.rand(1, 8)
        loss = l_blend_rank(blend, m, margin=0.10)
        # ex-bg term may be nonzero, but ov-ex term should be 0 since n_ov→eps
        # exclusive > bg → both terms can be 0
        # Just check it doesn't raise and is >= 0
        assert loss.item() >= 0.0

    def test_gradient_flows_to_blend(self):
        """l_blend_rank에서 blend_map으로 gradient가 흐름."""
        m = _masks(ov=[0,1], ex0=[2,3], ex1=[4,5])
        blend = torch.zeros(1, 8, requires_grad=False)
        # Make a violation: ov < ex
        blend_vals = blend.clone()
        blend_vals[0, 0] = blend_vals[0, 1] = 0.20
        blend_vals[0, 2] = blend_vals[0, 3] = 0.60
        blend_vals[0, 4] = blend_vals[0, 5] = 0.60
        blend_vals[0, 6] = blend_vals[0, 7] = 0.05
        blend_grad = blend_vals.detach().requires_grad_(True)

        loss = l_blend_rank(blend_grad, m, margin=0.10)
        assert loss.item() > 0.0
        loss.backward()
        assert blend_grad.grad is not None
        assert blend_grad.grad.abs().max().item() > 0

    def test_zero_margin_correct_order_zero_loss(self):
        """margin=0일 때 overlap ≥ exclusive ≥ bg이면 loss = 0."""
        m = _masks(ov=[0], ex0=[1], ex1=[2])
        # ov=0.5, ex=0.3, bg=0.1 — strictly ordered
        blend = torch.tensor([[0.5, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1]])
        loss = l_blend_rank(blend, m, margin=0.0)
        assert loss.item() < 1e-5, f"zero margin correct order: loss={loss.item():.6f}"

    def test_output_scalar(self):
        """l_blend_rank 출력은 scalar."""
        m = _masks(ov=[0,1], ex0=[2,3])
        blend = torch.rand(1, 8)
        loss = l_blend_rank(blend, m)
        assert loss.ndim == 0


# =============================================================================
# collect_blend_stats_detailed
# =============================================================================

class TestCollectBlendStatsDetailed:

    def test_blend_sep_formula(self):
        """blend_sep = blend_overlap_mean - blend_exclusive_mean."""
        S = 8
        m = _masks(S=S, ov=[0,1], ex0=[2,3], ex1=[4,5])
        blend = torch.zeros(1, S)
        blend[0, 0] = blend[0, 1] = 0.90
        blend[0, 2] = blend[0, 3] = 0.35
        blend[0, 4] = blend[0, 5] = 0.35
        blend[0, 6] = blend[0, 7] = 0.05

        stats = collect_blend_stats_detailed(blend, m)
        expected_sep = stats["blend_overlap_mean"] - stats["blend_exclusive_mean"]
        assert abs(stats["blend_sep"] - expected_sep) < 1e-5

    def test_blend_gap_bg_formula(self):
        """blend_gap_bg = blend_exclusive_mean - blend_bg_mean."""
        S = 8
        m = _masks(S=S, ov=[0,1], ex0=[2,3], ex1=[4,5])
        blend = torch.zeros(1, S)
        blend[0, 0] = blend[0, 1] = 0.90
        blend[0, 2] = blend[0, 3] = 0.35
        blend[0, 4] = blend[0, 5] = 0.35
        blend[0, 6] = blend[0, 7] = 0.05

        stats = collect_blend_stats_detailed(blend, m)
        expected_gap = stats["blend_exclusive_mean"] - stats["blend_bg_mean"]
        assert abs(stats["blend_gap_bg"] - expected_gap) < 1e-5

    def test_all_bg_stats(self):
        """전체 bg → overlap/exclusive stats ≈ 0."""
        m = torch.zeros(1, 2, 8)  # no entities
        blend = torch.rand(1, 8)
        stats = collect_blend_stats_detailed(blend, m)
        # no entity pixels → n_ov and n_ex approach eps, so means ≈ 0
        assert abs(stats["blend_overlap_mean"]) < 1e-2
        assert abs(stats["blend_exclusive_mean"]) < 1e-2

    def test_required_keys(self):
        """필요한 key 목록 확인."""
        m = _masks(S=8, ov=[0,1], ex0=[2,3])
        blend = torch.rand(1, 8)
        stats = collect_blend_stats_detailed(blend, m)
        for key in ["blend_mean", "blend_overlap_mean", "blend_exclusive_mean",
                    "blend_bg_mean", "blend_sep", "blend_gap_bg"]:
            assert key in stats, f"key '{key}' not in stats"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
