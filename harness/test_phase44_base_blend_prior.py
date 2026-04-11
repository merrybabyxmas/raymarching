"""
test_phase44_base_blend_prior.py — compute_base_blend + OverlapBlendHead 검증

검증 항목 (compute_base_blend):
  1. alpha_0=0, alpha_1=0 → base_blend ≈ 0.05 (background)
  2. single entity alpha≈0.5 → ~0.175 (중간)
  3. overlap alpha=0.9 → > single entity (overlap 높음)
  4. base_blend always ∈ [0.05, 0.95]
  5. overlap 증가 → base_blend 단조 증가
  6. 대칭: compute_base_blend(a0, a1) == compute_base_blend(a1, a0)

검증 항목 (OverlapBlendHead):
  7. 초기화 후 delta ≈ 0 → blend_map ≈ base_blend (zero-init 보장)
  8. 출력 shape: (B, S, 1)
  9. gradient가 마지막 Linear weight로 흐름
 10. in_features=8 입력 수용 (8 feature channels)
 11. hidden 크기 파라미터 작동
 12. non-zero grad update 후 delta ≠ 0
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase44 import compute_base_blend, OverlapBlendHead


# =============================================================================
# compute_base_blend
# =============================================================================

class TestComputeBaseBlend:

    def test_background_approx_005(self):
        """alpha_0=alpha_1=0 → base_blend ≈ 0.05."""
        a0 = torch.zeros(1, 4)
        a1 = torch.zeros(1, 4)
        bb = compute_base_blend(a0, a1)
        assert bb.allclose(torch.full_like(bb, 0.05), atol=1e-5), \
            f"background: expected 0.05, got {bb[0,0].item():.6f}"

    def test_single_entity_mid_range(self):
        """single entity alpha=0.5, other=0 → ~0.175."""
        # base = 0.05 + 0.25*max(0.5,0) + 0.60*(0.5*0) = 0.05 + 0.125 = 0.175
        a0 = torch.full((1, 4), 0.5)
        a1 = torch.zeros(1, 4)
        bb = compute_base_blend(a0, a1)
        assert abs(bb[0, 0].item() - 0.175) < 1e-5, \
            f"single entity: expected 0.175, got {bb[0,0].item():.6f}"

    def test_overlap_higher_than_single(self):
        """overlap alpha_0=alpha_1=0.9 → higher than single entity."""
        a_single0 = torch.full((1, 4), 0.9)
        a_single1 = torch.zeros(1, 4)
        bb_single = compute_base_blend(a_single0, a_single1)

        a_ov0 = torch.full((1, 4), 0.9)
        a_ov1 = torch.full((1, 4), 0.9)
        bb_ov = compute_base_blend(a_ov0, a_ov1)

        assert (bb_ov > bb_single).all(), \
            f"overlap should be higher than single: {bb_ov[0,0]:.4f} vs {bb_single[0,0]:.4f}"

    def test_clamped_to_005_095(self):
        """base_blend always ∈ [0.05, 0.95]."""
        a0 = torch.rand(4, 64)
        a1 = torch.rand(4, 64)
        bb = compute_base_blend(a0, a1)
        assert bb.min().item() >= 0.05 - 1e-6
        assert bb.max().item() <= 0.95 + 1e-6

    def test_monotone_with_overlap(self):
        """overlap 증가 → base_blend 단조 증가."""
        # Fix a0=0.8, increase a1
        a0 = torch.full((1, 1), 0.8)
        vals = [0.0, 0.2, 0.5, 0.8, 1.0]
        bbs  = [compute_base_blend(a0, torch.full((1, 1), v)).item() for v in vals]
        for i in range(1, len(bbs)):
            assert bbs[i] >= bbs[i-1], \
                f"non-monotone at a1={vals[i]}: {bbs[i-1]:.4f} → {bbs[i]:.4f}"

    def test_symmetry(self):
        """compute_base_blend(a0, a1) == compute_base_blend(a1, a0)."""
        a0 = torch.rand(2, 16)
        a1 = torch.rand(2, 16)
        bb_01 = compute_base_blend(a0, a1)
        bb_10 = compute_base_blend(a1, a0)
        assert bb_01.allclose(bb_10, atol=1e-6), "base_blend should be symmetric"


# =============================================================================
# OverlapBlendHead
# =============================================================================

class TestOverlapBlendHead:

    def test_zero_init_delta_near_zero(self):
        """초기화 직후 delta ≈ 0 → blend_map ≈ base_blend."""
        head = OverlapBlendHead(in_features=8, hidden=32)
        head.eval()

        B, S = 2, 16
        feat = torch.randn(B, S, 8)
        with torch.no_grad():
            delta = head(feat)
        # last layer is zero-initialized → delta ≈ 0
        assert delta.abs().max().item() < 1e-6, \
            f"zero-init: delta should be 0, got max={delta.abs().max().item():.2e}"

    def test_output_shape(self):
        """출력 shape = (B, S, 1)."""
        head = OverlapBlendHead(in_features=8, hidden=32)
        feat = torch.randn(3, 64, 8)
        out = head(feat)
        assert out.shape == (3, 64, 1), f"expected (3,64,1), got {out.shape}"

    def test_gradient_to_last_layer(self):
        """gradient가 마지막 Linear weight로 흐름."""
        head = OverlapBlendHead(in_features=8, hidden=32)
        # Perturb last layer so delta ≠ 0 and there's a non-trivial gradient
        with torch.no_grad():
            head.net[2].weight.fill_(0.01)

        feat = torch.randn(1, 8, 8)
        delta = head(feat)
        loss  = delta.sum()
        loss.backward()
        assert head.net[2].weight.grad is not None
        assert head.net[2].weight.grad.abs().max().item() > 0

    def test_accepts_8_features(self):
        """in_features=8 수용."""
        head = OverlapBlendHead(in_features=8, hidden=32)
        feat = torch.randn(1, 10, 8)
        out = head(feat)  # should not raise
        assert out.shape[-1] == 1

    def test_custom_hidden_size(self):
        """hidden 크기 파라미터 작동."""
        head16 = OverlapBlendHead(in_features=8, hidden=16)
        head64 = OverlapBlendHead(in_features=8, hidden=64)
        feat = torch.randn(1, 4, 8)
        # Both should produce (1, 4, 1)
        assert head16(feat).shape == (1, 4, 1)
        assert head64(feat).shape == (1, 4, 1)

    def test_nonzero_update_changes_delta(self):
        """파라미터 업데이트 후 delta ≠ 0."""
        head = OverlapBlendHead(in_features=8, hidden=32)
        feat = torch.randn(1, 8, 8)

        # Update: set last weight to non-zero
        with torch.no_grad():
            head.net[2].weight.fill_(0.1)
            head.net[2].bias.fill_(0.0)

        delta = head(feat)
        assert delta.abs().max().item() > 0, "after weight update, delta should be non-zero"

    def test_blend_map_from_base_plus_delta(self):
        """blend_map = sigmoid(logit(base_blend) + delta) 계산 검증."""
        head = OverlapBlendHead(in_features=8, hidden=32)

        a0 = torch.full((1, 4), 0.9)
        a1 = torch.full((1, 4), 0.9)
        from models.entity_slot_phase44 import compute_base_blend
        base_blend = compute_base_blend(a0, a1)  # (1, 4)

        # With zero-init head, delta=0, so blend_map ≈ base_blend
        feat = torch.randn(1, 4, 8)
        with torch.no_grad():
            delta = head(feat)  # ≈ 0

        blend_logit = torch.logit(base_blend, eps=1e-6).unsqueeze(-1)
        blend_map   = torch.sigmoid(blend_logit + delta).squeeze(-1)

        assert blend_map.allclose(base_blend, atol=1e-5), \
            "zero-init: blend_map should equal base_blend"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
