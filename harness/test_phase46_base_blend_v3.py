"""
test_phase46_base_blend_v3.py — compute_base_blend_v3 + val_score_phase46 검증

검증 항목 (compute_base_blend_v3):
  1. overlap (o0=1, o1=1): output ≈ 0.05 + 0.25*1 + 0.65*1 = 0.95 (clamped)
  2. exclusive (o0=1, o1=0): output ≈ 0.05 + 0.25*1 = 0.30
  3. bg (o0=0, o1=0): output ≈ 0.05
  4. Phase45 collapse (o0=0.92, o1=0.69): v3 < v2 (less inflation)
  5. correct exclusive (o0=0.92, o1=0.08): v3 ≈ v2 (unchanged for correct case)
  6. output ∈ [0.05, 0.95] (clamp 검증)
  7. ordering: v3(overlap) > v3(exclusive) > v3(bg)
  8. soft threshold: o1=0.49 (below 0.5) → soft_o1=0 → product=0

검증 항목 (val_score_phase46):
  9. has_rollout=True weights sum to 1.0
 10. has_rollout=False weights sum to 1.0
 11. has_rollout=False → rollout값 무시됨 (rollout=0 vs 1이어도 점수 동일)
 12. blend_sep=0.15 → blend_score=1.0 (max)
 13. blend_sep=-0.15 → blend_score=0.0 (min)
 14. blend_sep=0.00 → blend_score=0.5 (midpoint: (0+0.15)/0.30=0.5)
 15. 단조 증가: blend_sep 증가 → val_score 증가
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase46 import compute_base_blend_v3, val_score_phase46
from models.entity_slot_phase45 import compute_base_blend_v2


class TestComputeBaseBlendV3:

    def test_overlap_high(self):
        """o0=1, o1=1 → output ≈ 0.95 (clamped)."""
        o0 = torch.ones(2, 64)
        o1 = torch.ones(2, 64)
        out = compute_base_blend_v3(o0, o1)
        # 0.05 + 0.25*1 + 0.65*1 = 0.95
        assert out.allclose(torch.full_like(out, 0.95), atol=1e-5), \
            f"overlap(1,1): expected 0.95, got {out.mean().item():.4f}"

    def test_exclusive_e0(self):
        """o0=1, o1=0 → output ≈ 0.30."""
        o0 = torch.ones(2, 64)
        o1 = torch.zeros(2, 64)
        out = compute_base_blend_v3(o0, o1)
        # soft_o0=(2*(1-0.5)).clamp(0,1)=1, soft_o1=0 → product=0
        # output = 0.05 + 0.25*max(1,0) + 0.65*0 = 0.30
        assert out.allclose(torch.full_like(out, 0.30), atol=1e-5), \
            f"exclusive(1,0): expected 0.30, got {out.mean().item():.4f}"

    def test_bg_low(self):
        """o0=0, o1=0 → output ≈ 0.05."""
        o0 = torch.zeros(2, 64)
        o1 = torch.zeros(2, 64)
        out = compute_base_blend_v3(o0, o1)
        assert out.allclose(torch.full_like(out, 0.05), atol=1e-5), \
            f"bg(0,0): expected 0.05, got {out.mean().item():.4f}"

    def test_collapse_v3_less_than_v2(self):
        """Phase45 collapse (o0=0.92, o1=0.69): v3 < v2 (26% reduction)."""
        o0 = torch.full((1, 1), 0.92)
        o1 = torch.full((1, 1), 0.69)
        v2 = compute_base_blend_v2(o0, o1).item()
        v3 = compute_base_blend_v3(o0, o1).item()
        assert v3 < v2, f"collapse: v3={v3:.4f} should be < v2={v2:.4f}"
        # v2 ≈ 0.661, v3 ≈ 0.488
        assert v3 < 0.55, f"v3 should be noticeably lower than 0.55, got {v3:.4f}"

    def test_correct_exclusive_v3_approx_v2(self):
        """Correct exclusive (o0=0.92, o1=0.08): v3 ≈ v2 (both give ~0.28)."""
        o0 = torch.full((1, 1), 0.92)
        o1 = torch.full((1, 1), 0.08)
        v2 = compute_base_blend_v2(o0, o1).item()
        v3 = compute_base_blend_v3(o0, o1).item()
        assert abs(v3 - v2) < 0.05, \
            f"correct exclusive: v3={v3:.4f} and v2={v2:.4f} should be close"

    def test_output_in_range(self):
        """Output ∈ [0.05, 0.95]."""
        o0 = torch.rand(4, 256)
        o1 = torch.rand(4, 256)
        out = compute_base_blend_v3(o0, o1)
        assert out.min().item() >= 0.05 - 1e-6, f"min below 0.05: {out.min().item():.4f}"
        assert out.max().item() <= 0.95 + 1e-6, f"max above 0.95: {out.max().item():.4f}"

    def test_ordering(self):
        """v3(overlap) > v3(exclusive) > v3(bg)."""
        v_ov = compute_base_blend_v3(
            torch.tensor([[0.9]]), torch.tensor([[0.9]])).item()
        v_ex = compute_base_blend_v3(
            torch.tensor([[0.9]]), torch.tensor([[0.1]])).item()
        v_bg = compute_base_blend_v3(
            torch.tensor([[0.1]]), torch.tensor([[0.1]])).item()
        assert v_ov > v_ex, f"overlap({v_ov:.4f}) should > exclusive({v_ex:.4f})"
        assert v_ex > v_bg, f"exclusive({v_ex:.4f}) should > bg({v_bg:.4f})"

    def test_soft_threshold_below_half(self):
        """o1 just below 0.5 → soft_o1=0 → product=0 (no overlap proxy)."""
        o0 = torch.tensor([[0.9]])
        o1 = torch.tensor([[0.49]])   # just below 0.5
        out = compute_base_blend_v3(o0, o1)
        # soft_o1 = (2*(0.49-0.5)).clamp(0,1) = clamp(-0.02,0,1) = 0
        # product = soft_o0 * 0 = 0
        # output = 0.05 + 0.25*max(0.9, 0.49) + 0.65*0 = 0.05 + 0.25*0.9 = 0.275
        expected = 0.05 + 0.25 * 0.9
        assert abs(out.item() - expected) < 1e-5, \
            f"o1=0.49 → soft_o1=0: expected {expected:.4f}, got {out.item():.4f}"


class TestValScorePhase46:

    def test_has_rollout_weights_sum_one(self):
        """has_rollout=True weights sum to 1.0."""
        # 0.15 + 0.15 + 0.10 + 0.10 + 0.20 + 0.20 + 0.10 = 1.00
        total_w = 0.15 + 0.15 + 0.10 + 0.10 + 0.20 + 0.20 + 0.10
        assert abs(total_w - 1.0) < 1e-9, f"has_rollout=True weights sum: {total_w}"

    def test_no_rollout_weights_sum_one(self):
        """has_rollout=False weights sum to 1.0."""
        # 0.25 + 0.25 + 0.15 + 0.15 + 0.20 = 1.00
        total_w = 0.25 + 0.25 + 0.15 + 0.15 + 0.20
        assert abs(total_w - 1.0) < 1e-9, f"has_rollout=False weights sum: {total_w}"

    def test_no_rollout_ignores_rollout_values(self):
        """has_rollout=False: rollout_iou=0 and rollout_iou=1 give same score."""
        s1 = val_score_phase46(0.5, 0.5, 0.5, 0.5,
                               rollout_iou_e0=0.0, rollout_iou_e1=0.0,
                               blend_sep=0.05, has_rollout=False)
        s2 = val_score_phase46(0.5, 0.5, 0.5, 0.5,
                               rollout_iou_e0=1.0, rollout_iou_e1=1.0,
                               blend_sep=0.05, has_rollout=False)
        assert abs(s1 - s2) < 1e-9, \
            f"has_rollout=False should ignore rollout: {s1:.6f} vs {s2:.6f}"

    def test_blend_sep_max(self):
        """blend_sep=0.15 → blend_score=1.0."""
        # blend_score = min(1, max(0, (0.15+0.15)/0.30)) = min(1, 1.0) = 1.0
        s = val_score_phase46(0.5, 0.5, 0.5, 0.5,
                              blend_sep=0.15, has_rollout=False)
        s_max = val_score_phase46(0.5, 0.5, 0.5, 0.5,
                                  blend_sep=100.0, has_rollout=False)
        assert abs(s - s_max) < 1e-6, \
            f"blend_sep=0.15 should reach max blend_score: {s:.6f} vs max {s_max:.6f}"

    def test_blend_sep_min(self):
        """blend_sep=-0.15 → blend_score=0.0."""
        # blend_score = min(1, max(0, (-0.15+0.15)/0.30)) = 0.0
        s = val_score_phase46(0.5, 0.5, 0.5, 0.5,
                              blend_sep=-0.15, has_rollout=False)
        s_min = val_score_phase46(0.5, 0.5, 0.5, 0.5,
                                  blend_sep=-100.0, has_rollout=False)
        assert abs(s - s_min) < 1e-6, \
            f"blend_sep=-0.15 should reach min blend_score: {s:.6f} vs min {s_min:.6f}"

    def test_blend_sep_midpoint(self):
        """blend_sep=0.0 → blend_score=0.5."""
        # blend_score = (0+0.15)/0.30 = 0.5
        s = val_score_phase46(0.5, 0.5, 0.5, 0.5,
                              blend_sep=0.0, has_rollout=False)
        # With has_rollout=False: 0.25*0.5+0.25*0.5+0.15*0.5+0.15*0.5+0.20*0.5 = 0.5
        expected_blend_contrib = 0.20 * 0.5
        base = 0.25*0.5 + 0.25*0.5 + 0.15*0.5 + 0.15*0.5
        expected = base + expected_blend_contrib
        assert abs(s - expected) < 1e-6, \
            f"blend_sep=0: expected {expected:.4f}, got {s:.4f}"

    def test_blend_sep_monotone(self):
        """blend_sep 증가 → val_score 단조 증가."""
        sep_vals = [-0.20, -0.10, 0.0, 0.05, 0.15, 0.30]
        scores = [val_score_phase46(0.5, 0.5, 0.5, 0.5,
                                    blend_sep=sep, has_rollout=False)
                  for sep in sep_vals]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i+1] + 1e-9, \
                f"blend_sep={sep_vals[i]}→{sep_vals[i+1]}: score should not decrease: {scores[i]:.4f}→{scores[i+1]:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
