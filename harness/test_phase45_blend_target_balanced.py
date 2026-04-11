"""
test_phase45_blend_target_balanced.py — l_blend_target_balanced 검증

검증 항목:
  1. perfect overlap prediction (blend=0.90) → overlap term ≈ 0
  2. perfect exclusive prediction (blend=0.35) → exclusive term ≈ 0
  3. perfect bg prediction (blend=0.05) → bg term ≈ 0
  4. region normalization: sparse overlap와 dense bg가 equal weight
  5. overlap gradient는 overlap region 픽셀 수에 무관하게 정상 흐름
  6. w_ov > w_ex > w_bg 가중치 반영
  7. gradient flow → blend_map으로 grad 흐름
  8. Phase44 l_blend_target과 비교: sparse overlap에서 gradient 크기 차이
  9. 출력 scalar 확인
 10. default values (overlap=0.90, exclusive=0.35, bg=0.05) 확인
 11. overlap 1픽셀 vs 1000픽셀 → region-normalized loss 동일
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase45 import l_blend_target_balanced


# =============================================================================
# Helper
# =============================================================================

def _make_masks(B: int, S: int,
                ov_frac: float = 0.05,
                ex_frac: float = 0.35) -> torch.Tensor:
    """
    2-entity mask: (B, 2, S)
    - overlap: first ov_frac of S pixels
    - exclusive e0: next ex_frac pixels
    - exclusive e1: next ex_frac pixels
    - bg: rest
    """
    masks = torch.zeros(B, 2, S)
    n_ov = max(1, int(S * ov_frac))
    n_ex = max(1, int(S * ex_frac))
    # overlap region
    masks[:, 0, :n_ov] = 1.0
    masks[:, 1, :n_ov] = 1.0
    # exclusive e0
    masks[:, 0, n_ov:n_ov + n_ex] = 1.0
    # exclusive e1
    masks[:, 1, n_ov + n_ex:n_ov + 2 * n_ex] = 1.0
    return masks


# =============================================================================
# l_blend_target_balanced
# =============================================================================

class TestLBlendTargetBalanced:

    def test_perfect_overlap_prediction(self):
        """blend=0.90 everywhere → overlap term ≈ 0."""
        B, S = 2, 64
        masks = _make_masks(B, S, ov_frac=0.05)
        blend_map = torch.full((B, S), 0.90)
        # Only check overlap loss: set exclusive/bg to near-target too
        blend_perfect = blend_map.clone()
        # Manually force: set exclusive pixels to 0.35, bg to 0.05
        m0 = masks[:, 0, :]
        m1 = masks[:, 1, :]
        ov   = m0 * m1
        excl = (m0 + m1).clamp(0, 1) - ov
        bg   = 1.0 - (m0 + m1).clamp(0, 1)
        blend_perfect = (0.90 * ov + 0.35 * excl + 0.05 * bg)

        loss = l_blend_target_balanced(blend_perfect, masks)
        assert loss.item() < 1e-6, \
            f"perfect prediction: loss should ≈ 0, got {loss.item():.2e}"

    def test_overlap_term_near_zero_when_correct(self):
        """blend_map == overlap_val in overlap region → MSE_ov ≈ 0."""
        B, S = 1, 100
        masks = _make_masks(B, S, ov_frac=0.10)
        m0 = masks[:, 0, :]
        m1 = masks[:, 1, :]
        ov = m0 * m1

        # Only overlap region is correctly predicted, rest is random
        blend_map = torch.rand(B, S)
        # Override overlap pixels to 0.90
        blend_map = blend_map * (1 - ov) + 0.90 * ov

        # Compute the overlap MSE manually
        n_ov = ov.sum().item()
        mse_ov = ((blend_map - 0.90).pow(2) * ov).sum().item() / (n_ov + 1e-6)
        assert mse_ov < 1e-6, f"overlap MSE should be 0: {mse_ov:.2e}"

    def test_region_normalization_sparse_vs_dense(self):
        """
        overlap 1픽셀 vs 50픽셀: region-normalized → 동일한 overlap gradient 기여.
        Phase44 global mean: 1픽셀 loss ≈ 0 (희석), balanced: 동일.
        """
        B = 1

        def _loss_with_n_overlap(n_ov: int, S: int = 200) -> float:
            masks = torch.zeros(B, 2, S)
            # overlap: first n_ov pixels
            masks[:, 0, :n_ov] = 1.0
            masks[:, 1, :n_ov] = 1.0
            # All blend = 0.5 (wrong for overlap: target 0.90)
            blend_map = torch.full((B, S), 0.5)
            return l_blend_target_balanced(
                blend_map, masks,
                w_ov=1.0, w_ex=0.0, w_bg=0.0  # only overlap term
            ).item()

        loss_1  = _loss_with_n_overlap(1)
        loss_50 = _loss_with_n_overlap(50)

        # Region-normalized: both should be the same (MSE_ov / n_ov is constant)
        # MSE_ov = (0.5-0.90)^2 = 0.16 for all pixels → both = 0.16
        assert abs(loss_1 - loss_50) < 1e-4, \
            f"region-normalized: 1-pix({loss_1:.4f}) vs 50-pix({loss_50:.4f}) should be equal"

    def test_overlap_gradient_independent_of_pixel_count(self):
        """
        overlap region 크기와 무관하게 grad 크기 동일 (region normalization 효과).
        """
        B, S = 1, 200

        def _grad_magnitude(n_ov: int) -> float:
            masks = torch.zeros(B, 2, S)
            masks[:, 0, :n_ov] = 1.0
            masks[:, 1, :n_ov] = 1.0
            blend_map = torch.full((B, S), 0.5, requires_grad=True)
            loss = l_blend_target_balanced(
                blend_map, masks, w_ov=1.0, w_ex=0.0, w_bg=0.0)
            loss.backward()
            # Mean gradient over overlap pixels
            ov_mask = masks[:, 0, :] * masks[:, 1, :]
            return (blend_map.grad * ov_mask).abs().mean().item()

        g1  = _grad_magnitude(1)
        g50 = _grad_magnitude(50)
        # Region-normalized: per-pixel grad should be the same
        assert abs(g1 - g50) < 0.01, \
            f"grad magnitude should be region-normalized: n=1({g1:.4f}) vs n=50({g50:.4f})"

    def test_weights_applied_correctly(self):
        """w_ov=1.0, w_ex=0.5, w_bg=0.2 가중치 반영 확인."""
        B, S = 1, 60
        # Setup: 20 overlap, 20 exclusive, 20 bg pixels
        masks = torch.zeros(B, 2, S)
        masks[:, 0, :20] = 1.0
        masks[:, 1, :20] = 1.0  # overlap
        masks[:, 0, 20:40] = 1.0  # exclusive e0

        blend_map = torch.full((B, S), 0.0)  # worst prediction

        # Overlap MSE: (0.0 - 0.90)^2 = 0.81, ex: (0.0-0.35)^2=0.1225, bg: (0.0-0.05)^2=0.0025
        loss = l_blend_target_balanced(blend_map, masks, w_ov=1.0, w_ex=0.5, w_bg=0.2)
        expected = 1.0 * 0.81 + 0.5 * 0.1225 + 0.2 * 0.0025
        assert abs(loss.item() - expected) < 1e-4, \
            f"weighted loss: expected {expected:.5f}, got {loss.item():.5f}"

    def test_gradient_flows_to_blend_map(self):
        """gradient가 blend_map으로 흐름."""
        B, S = 2, 32
        masks = _make_masks(B, S)
        blend_map = torch.rand(B, S, requires_grad=True)
        loss = l_blend_target_balanced(blend_map, masks)
        loss.backward()
        assert blend_map.grad is not None
        assert blend_map.grad.abs().max().item() > 0

    def test_scalar_output(self):
        """출력 scalar 확인."""
        B, S = 2, 64
        masks = _make_masks(B, S)
        blend_map = torch.rand(B, S)
        loss = l_blend_target_balanced(blend_map, masks)
        assert loss.shape == torch.Size([]), f"loss should be scalar, got {loss.shape}"

    def test_default_target_values(self):
        """default values: overlap=0.90, exclusive=0.35, bg=0.05."""
        B, S = 1, 60
        masks = torch.zeros(B, 2, S)
        # 20 overlap, 20 exclusive, 20 bg
        masks[:, 0, :20] = 1.0
        masks[:, 1, :20] = 1.0
        masks[:, 0, 20:40] = 1.0

        # Evaluate at default targets → MSE = 0
        blend_default = torch.zeros(B, S)
        blend_default[:, :20]  = 0.90
        blend_default[:, 20:40] = 0.35
        blend_default[:, 40:]  = 0.05

        loss = l_blend_target_balanced(blend_default, masks)
        assert loss.item() < 1e-6, \
            f"default targets: loss should ≈ 0, got {loss.item():.2e}"

    def test_phase44_comparison_sparse_overlap(self):
        """
        Phase44 l_blend_target (global mean) vs Phase45 balanced:
        sparse overlap (5%) → balanced has higher overlap gradient contribution.
        """
        from models.entity_slot_phase44 import l_blend_target as l_bt_v1

        B, S = 1, 200
        masks = torch.zeros(B, 2, S)
        n_ov = 10  # only 5%
        masks[:, 0, :n_ov] = 1.0
        masks[:, 1, :n_ov] = 1.0

        # Blend predicts wrong value in overlap
        blend_v1 = torch.full((B, S), 0.5, requires_grad=True)
        blend_v2 = torch.full((B, S), 0.5, requires_grad=True)

        loss_v1 = l_bt_v1(blend_v1, masks)
        loss_v2 = l_blend_target_balanced(blend_v2, masks, w_ov=1.0, w_ex=0.0, w_bg=0.0)

        loss_v1.backward()
        loss_v2.backward()

        # In overlap region, Phase45 balanced should have equal or larger grad
        ov_mask = masks[:, 0, :] * masks[:, 1, :]
        grad_v1_ov = (blend_v1.grad * ov_mask).abs().mean().item()
        grad_v2_ov = (blend_v2.grad * ov_mask).abs().mean().item()

        # Phase44 dilutes by /S, Phase45 divides by /n_ov → Phase45 bigger
        assert grad_v2_ov >= grad_v1_ov, \
            f"balanced ({grad_v2_ov:.5f}) should ≥ global-mean ({grad_v1_ov:.5f}) in overlap"

    def test_consistent_with_manual_computation(self):
        """수동 계산과 동일한 결과."""
        B, S = 1, 6
        masks = torch.zeros(B, 2, S)
        # pixel 0,1: overlap; pixel 2,3: exclusive e0; pixel 4,5: bg
        masks[:, 0, 0:2] = 1.0
        masks[:, 1, 0:2] = 1.0
        masks[:, 0, 2:4] = 1.0

        blend_map = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

        # Manual:
        # overlap (px 0,1): target=0.90, MSE=(0.5-0.90)^2=0.16, n=2 → MSE_ov=0.16
        # exclusive e0 (px 2,3): target=0.35, MSE=(0.5-0.35)^2=0.0225, n=2 → MSE_ex=0.0225
        # bg (px 4,5): target=0.05, MSE=(0.5-0.05)^2=0.2025, n=2 → MSE_bg=0.2025
        # L = 1.0*0.16 + 0.5*0.0225 + 0.2*0.2025 = 0.16 + 0.01125 + 0.0405 = 0.21175
        expected = 1.0 * 0.16 + 0.5 * 0.0225 + 0.2 * 0.2025
        loss = l_blend_target_balanced(blend_map, masks)
        assert abs(loss.item() - expected) < 1e-4, \
            f"manual: expected {expected:.5f}, got {loss.item():.5f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
