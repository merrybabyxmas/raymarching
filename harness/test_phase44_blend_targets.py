"""
test_phase44_blend_targets.py — build_blend_targets + l_blend_target 검증

검증 항목:
  1. overlap 영역 → target = overlap_val (0.90)
  2. exclusive e0만 → target = exclusive_val (0.35)
  3. exclusive e1만 → target = exclusive_val (0.35)
  4. background → target = bg_val (0.05)
  5. 혼합 픽셀 집합 — 각 target 값 올바름
  6. target 값 clamp [0, 1]
  7. custom overlap/exclusive/bg_val 파라미터 작동
  8. l_blend_target: 완벽 예측 → loss 최소 (near 0)
  9. l_blend_target: 역방향 예측 → loss > 완벽 예측
 10. l_blend_target: overlap 영역 가중치 → overlap 오류가 non-overlap 오류보다 loss ↑
 11. l_blend_target: gradient가 blend_map으로 흐름
 12. l_blend_target: output shape = scalar
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase44 import build_blend_targets, l_blend_target


# =============================================================================
# Helpers
# =============================================================================

def _masks(B=1, S=8, ov=None, ex0=None, ex1=None):
    """(B, 2, S) 마스크 생성."""
    m = torch.zeros(B, 2, S)
    for p in (ov or []):
        m[:, 0, p] = 1.0; m[:, 1, p] = 1.0
    for p in (ex0 or []):
        m[:, 0, p] = 1.0
    for p in (ex1 or []):
        m[:, 1, p] = 1.0
    return m


# =============================================================================
# build_blend_targets
# =============================================================================

class TestBuildBlendTargets:

    def test_overlap_region_gets_overlap_val(self):
        """overlap 픽셀 → target = overlap_val."""
        m = _masks(S=4, ov=[0, 1])
        t = build_blend_targets(m)
        assert abs(t[0, 0].item() - 0.90) < 1e-5
        assert abs(t[0, 1].item() - 0.90) < 1e-5

    def test_exclusive_e0_gets_exclusive_val(self):
        """e0 전용 픽셀 → target = exclusive_val."""
        m = _masks(S=4, ex0=[0, 1])
        t = build_blend_targets(m)
        assert abs(t[0, 0].item() - 0.35) < 1e-5
        assert abs(t[0, 1].item() - 0.35) < 1e-5

    def test_exclusive_e1_gets_exclusive_val(self):
        """e1 전용 픽셀 → target = exclusive_val."""
        m = _masks(S=4, ex1=[2, 3])
        t = build_blend_targets(m)
        assert abs(t[0, 2].item() - 0.35) < 1e-5
        assert abs(t[0, 3].item() - 0.35) < 1e-5

    def test_background_gets_bg_val(self):
        """어떤 entity도 없는 픽셀 → target = bg_val."""
        m = torch.zeros(1, 2, 4)  # all background
        t = build_blend_targets(m)
        for p in range(4):
            assert abs(t[0, p].item() - 0.05) < 1e-5

    def test_mixed_pixel_set(self):
        """overlap/exclusive/bg 모두 포함 — 각 target 올바름."""
        # pixel 0: overlap; pixel 1: ex0; pixel 2: ex1; pixel 3: bg
        m = _masks(S=4, ov=[0], ex0=[1], ex1=[2])
        t = build_blend_targets(m)
        assert abs(t[0, 0].item() - 0.90) < 1e-5, "overlap"
        assert abs(t[0, 1].item() - 0.35) < 1e-5, "exclusive e0"
        assert abs(t[0, 2].item() - 0.35) < 1e-5, "exclusive e1"
        assert abs(t[0, 3].item() - 0.05) < 1e-5, "background"

    def test_output_clamped_01(self):
        """target values ∈ [0, 1]."""
        m = _masks(S=8, ov=[0,1], ex0=[2,3], ex1=[4,5])
        t = build_blend_targets(m)
        assert t.min().item() >= 0.0
        assert t.max().item() <= 1.0

    def test_custom_values(self):
        """custom overlap/exclusive/bg_val 파라미터 작동."""
        m = _masks(S=3, ov=[0], ex0=[1])
        t = build_blend_targets(m, overlap_val=0.80, exclusive_val=0.50, bg_val=0.10)
        assert abs(t[0, 0].item() - 0.80) < 1e-5, "custom overlap_val"
        assert abs(t[0, 1].item() - 0.50) < 1e-5, "custom exclusive_val"
        assert abs(t[0, 2].item() - 0.10) < 1e-5, "custom bg_val"

    def test_output_shape(self):
        """출력 shape = (B, S)."""
        m = _masks(B=2, S=16, ov=[0,1])
        t = build_blend_targets(m)
        assert t.shape == (2, 16), f"expected (2,16), got {t.shape}"


# =============================================================================
# l_blend_target
# =============================================================================

class TestLBlendTarget:

    def test_perfect_prediction_low_loss(self):
        """blend_map = target → loss ≈ 0."""
        m = _masks(S=8, ov=[0,1], ex0=[2,3], ex1=[4,5])
        target = build_blend_targets(m)
        loss = l_blend_target(target, m)
        assert loss.item() < 1e-4, f"perfect prediction: loss={loss.item():.6f}"

    def test_wrong_prediction_high_loss(self):
        """blend_map ≈ 1-target → loss 큼."""
        m = _masks(S=8, ov=[0,1], ex0=[2,3], ex1=[4,5])
        target = build_blend_targets(m)
        wrong  = 1.0 - target
        loss = l_blend_target(wrong, m)
        assert loss.item() > 0.1, f"wrong prediction: loss={loss.item():.4f}"

    def test_overlap_region_upweighted(self):
        """overlap 오류가 exclusive 오류보다 loss에 더 큰 영향."""
        # 4 픽셀: 0→overlap, 1→exclusive, 2→bg, 3→bg
        m = _masks(S=4, ov=[0], ex0=[1])

        # Case A: overlap 오류, exclusive 정확
        target = build_blend_targets(m)
        blend_a = target.clone()
        blend_a[0, 0] = 1.0 - target[0, 0]  # flip overlap pixel

        # Case B: exclusive 오류, overlap 정확
        blend_b = target.clone()
        blend_b[0, 1] = 1.0 - target[0, 1]  # flip exclusive pixel

        loss_a = l_blend_target(blend_a, m)
        loss_b = l_blend_target(blend_b, m)

        assert loss_a.item() > loss_b.item(), (
            f"overlap error (loss={loss_a.item():.4f}) should > "
            f"exclusive error (loss={loss_b.item():.4f})")

    def test_gradient_flows_to_blend_map(self):
        """l_blend_target에서 blend_map으로 gradient가 흐름."""
        m = _masks(S=8, ov=[0,1], ex0=[2,3], ex1=[4,5])
        target = build_blend_targets(m)
        blend = (1.0 - target).detach().requires_grad_(True)  # loss > 0

        loss = l_blend_target(blend, m)
        loss.backward()
        assert blend.grad is not None
        assert blend.grad.abs().max().item() > 0

    def test_output_scalar(self):
        """l_blend_target 출력은 scalar."""
        m = _masks(S=8, ov=[0,1], ex0=[2,3])
        blend = torch.rand(1, 8)
        loss = l_blend_target(blend, m)
        assert loss.ndim == 0, f"expected scalar, got shape {loss.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
