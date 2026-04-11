"""
test_phase45_base_blend_v2.py — compute_base_blend_v2 검증

검증 항목:
  1. bg (o0=0, o1=0) → base_blend ≈ 0.05
  2. single entity (o0=1, o1=0) → 0.05 + 0.25*1 + 0 = 0.30
  3. overlap (o0=1, o1=1) → 0.05 + 0.25*1 + 0.60*1 = 0.90
  4. 순서 확인: overlap > exclusive > bg
  5. 클램프 [0.05, 0.95]
  6. 대칭: compute_base_blend_v2(o0, o1) == compute_base_blend_v2(o1, o0)
  7. o0*o1 항이 실제로 기여하는지 (partial overlap 확인)
  8. 단조성: o1 고정 시 o0 증가 → base_blend 단조 증가
  9. Phase44 compute_base_blend(alpha)보다 올바른 순서 제공 (overlap > exclusive)
 10. 공식 직접 검증: 0.05 + 0.25*max(o0,o1) + 0.60*(o0*o1)
 11. 배치 shape 유지: (B, S) → (B, S)
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase45 import compute_base_blend_v2


# =============================================================================
# compute_base_blend_v2
# =============================================================================

class TestComputeBaseBlendV2:

    def test_background_approx_005(self):
        """o0=0, o1=0 → base_blend ≈ 0.05 (background)."""
        o0 = torch.zeros(2, 16)
        o1 = torch.zeros(2, 16)
        bb = compute_base_blend_v2(o0, o1)
        assert bb.allclose(torch.full_like(bb, 0.05), atol=1e-5), \
            f"bg: expected 0.05, got max={bb.max().item():.6f}"

    def test_exclusive_entity_030(self):
        """o0=1, o1=0 → 0.05 + 0.25*1 + 0.60*0 = 0.30 (exclusive)."""
        o0 = torch.ones(2, 16)
        o1 = torch.zeros(2, 16)
        bb = compute_base_blend_v2(o0, o1)
        expected = 0.30
        assert bb.allclose(torch.full_like(bb, expected), atol=1e-5), \
            f"exclusive: expected {expected}, got {bb[0,0].item():.6f}"

    def test_overlap_090(self):
        """o0=1, o1=1 → 0.05 + 0.25 + 0.60 = 0.90 (overlap)."""
        o0 = torch.ones(2, 16)
        o1 = torch.ones(2, 16)
        bb = compute_base_blend_v2(o0, o1)
        expected = 0.90
        assert bb.allclose(torch.full_like(bb, expected), atol=1e-5), \
            f"overlap: expected {expected}, got {bb[0,0].item():.6f}"

    def test_correct_ordering_overlap_exclusive_bg(self):
        """overlap > exclusive > bg — Phase45의 핵심 수정 사항."""
        bb_ov = compute_base_blend_v2(
            torch.ones(1, 1), torch.ones(1, 1)).item()    # 0.90
        bb_ex = compute_base_blend_v2(
            torch.ones(1, 1), torch.zeros(1, 1)).item()   # 0.30
        bb_bg = compute_base_blend_v2(
            torch.zeros(1, 1), torch.zeros(1, 1)).item()  # 0.05

        assert bb_ov > bb_ex, \
            f"overlap({bb_ov:.4f}) should > exclusive({bb_ex:.4f})"
        assert bb_ex > bb_bg, \
            f"exclusive({bb_ex:.4f}) should > bg({bb_bg:.4f})"

    def test_clamped_to_005_095(self):
        """output ∈ [0.05, 0.95] — clamp 확인."""
        o0 = torch.rand(4, 64)
        o1 = torch.rand(4, 64)
        bb = compute_base_blend_v2(o0, o1)
        assert bb.min().item() >= 0.05 - 1e-6, f"min={bb.min().item():.6f}"
        assert bb.max().item() <= 0.95 + 1e-6, f"max={bb.max().item():.6f}"

    def test_symmetry(self):
        """compute_base_blend_v2(o0, o1) == compute_base_blend_v2(o1, o0)."""
        o0 = torch.rand(2, 16)
        o1 = torch.rand(2, 16)
        bb_01 = compute_base_blend_v2(o0, o1)
        bb_10 = compute_base_blend_v2(o1, o0)
        assert bb_01.allclose(bb_10, atol=1e-6), \
            "base_blend_v2 should be symmetric"

    def test_overlap_product_contributes(self):
        """partial overlap: o0=0.8, o1=0.8 → 0.05 + 0.25*0.8 + 0.60*0.64 = 0.634."""
        o0 = torch.full((1, 1), 0.8)
        o1 = torch.full((1, 1), 0.8)
        bb = compute_base_blend_v2(o0, o1)
        expected = 0.05 + 0.25 * 0.8 + 0.60 * (0.8 * 0.8)
        assert abs(bb.item() - expected) < 1e-5, \
            f"partial overlap: expected {expected:.5f}, got {bb.item():.5f}"

    def test_monotone_with_o0(self):
        """o1 고정 시 o0 증가 → base_blend 단조 증가."""
        o1 = torch.full((1, 1), 0.6)
        vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bbs = [compute_base_blend_v2(
            torch.full((1, 1), v), o1).item() for v in vals]
        for i in range(1, len(bbs)):
            assert bbs[i] >= bbs[i-1] - 1e-6, \
                f"non-monotone at o0={vals[i]}: {bbs[i-1]:.4f} → {bbs[i]:.4f}"

    def test_formula_direct_verification(self):
        """공식 직접 검증: 0.05 + 0.25*max(o0,o1) + 0.60*(o0*o1)."""
        o0 = torch.tensor([[0.3, 0.7, 1.0]])
        o1 = torch.tensor([[0.5, 0.2, 1.0]])
        bb = compute_base_blend_v2(o0, o1)

        # Manual computation
        overlap_proxy = o0 * o1
        entity_proxy  = torch.maximum(o0, o1)
        expected = (0.05 + 0.25 * entity_proxy + 0.60 * overlap_proxy).clamp(0.05, 0.95)
        assert bb.allclose(expected, atol=1e-6), \
            f"formula mismatch: got {bb}, expected {expected}"

    def test_batch_shape_preserved(self):
        """(B, S) → (B, S) shape 유지."""
        B, S = 3, 64
        o0 = torch.rand(B, S)
        o1 = torch.rand(B, S)
        bb = compute_base_blend_v2(o0, o1)
        assert bb.shape == (B, S), f"expected ({B},{S}), got {bb.shape}"

    def test_v2_vs_phase44_ordering(self):
        """
        Phase44 compute_base_blend (alpha-based) 실패 패턴 재현 후
        v2 (occupancy-based)가 올바른 순서를 제공하는지 확인.

        Phase44 실패:
          exclusive region: alpha_0=alpha_1=0.7 (shared attention)
          → compute_base_blend(0.7, 0.7) = 0.05 + 0.25*0.7 + 0.60*0.49 = 0.519
          overlap region: alpha_0=alpha_1=0.5
          → compute_base_blend(0.5, 0.5) = 0.05 + 0.25*0.5 + 0.60*0.25 = 0.325
          → exclusive > overlap (WRONG)

        Phase45 수정 (occupancy):
          exclusive: o0=1, o1=0 → 0.30
          overlap: o0=1, o1=1 → 0.90
          → overlap > exclusive (CORRECT)
        """
        from models.entity_slot_phase44 import compute_base_blend as compute_base_blend_v1

        # Phase44 실패 패턴
        a_ex  = torch.tensor([[0.7]])
        a_ov  = torch.tensor([[0.5]])
        bb_v1_ex = compute_base_blend_v1(a_ex, a_ex).item()
        bb_v1_ov = compute_base_blend_v1(a_ov, a_ov).item()
        # Phase44 inverted (exclusive > overlap when sigma shared)
        assert bb_v1_ex > bb_v1_ov, \
            "Phase44 bug: exclusive alpha=0.7 should give higher blend than overlap alpha=0.5"

        # Phase45 수정
        o_ex0 = torch.tensor([[1.0]]); o_ex1 = torch.tensor([[0.0]])
        o_ov0 = torch.tensor([[1.0]]); o_ov1 = torch.tensor([[1.0]])
        bb_v2_ex = compute_base_blend_v2(o_ex0, o_ex1).item()
        bb_v2_ov = compute_base_blend_v2(o_ov0, o_ov1).item()
        assert bb_v2_ov > bb_v2_ex, \
            f"Phase45: overlap({bb_v2_ov:.4f}) should > exclusive({bb_v2_ex:.4f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
