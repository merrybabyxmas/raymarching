"""
test_phase43_blend_stats.py — collect_blend_stats + l_blend_overlap 검증

검증 항목:
  1. collect_blend_stats: overlap/nonoverlap/separation 계산 검증
  2. collect_blend_stats: blend_separation > 0 when blend higher in overlap
  3. collect_blend_stats: blend_separation < 0 when blend lower in overlap
  4. collect_blend_stats: 모든 background → separation ≈ 0
  5. l_blend_overlap: separation > margin → loss = 0
  6. l_blend_overlap: separation < margin → loss > 0
  7. l_blend_overlap: no overlap region → loss = 0 (no-op)
  8. val_score_phase43 수식 검증
  9. val_score_phase43 rollout 포함 여부
"""
import sys
import math
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase43 import (
    collect_blend_stats,
    l_blend_overlap,
    val_score_phase43,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_masks(B=1, S=8, overlap_pixels=None, excl0_pixels=None, excl1_pixels=None):
    """
    (B, 2, S) 마스크 생성.
    overlap_pixels: list of pixel indices that belong to both entities
    excl0_pixels:   list of pixel indices belonging only to entity0
    excl1_pixels:   list of pixel indices belonging only to entity1
    """
    masks = torch.zeros(B, 2, S)
    if overlap_pixels:
        for p in overlap_pixels:
            masks[:, 0, p] = 1.0
            masks[:, 1, p] = 1.0
    if excl0_pixels:
        for p in excl0_pixels:
            masks[:, 0, p] = 1.0
    if excl1_pixels:
        for p in excl1_pixels:
            masks[:, 1, p] = 1.0
    return masks


# =============================================================================
# collect_blend_stats 검증
# =============================================================================

class TestCollectBlendStats:

    def test_blend_higher_in_overlap(self):
        """overlap 영역 blend가 non-overlap보다 높으면 blend_separation > 0."""
        S = 8
        # pixels 0,1: overlap
        # pixels 2,3: excl0; pixels 4,5: excl1; pixels 6,7: bg
        masks = _make_masks(S=S, overlap_pixels=[0, 1],
                            excl0_pixels=[2, 3], excl1_pixels=[4, 5])
        # blend high in overlap, low elsewhere
        blend = torch.zeros(1, S)
        blend[0, 0] = 0.9
        blend[0, 1] = 0.8
        blend[0, 2] = 0.2
        blend[0, 3] = 0.1
        blend[0, 4] = 0.2
        blend[0, 5] = 0.1

        stats = collect_blend_stats(blend, masks)
        assert stats["blend_separation"] > 0, \
            f"blend higher in overlap: separation should be > 0, got {stats['blend_separation']:.4f}"
        assert stats["blend_overlap_mean"] > stats["blend_nonoverlap_mean"], \
            "overlap blend should be > nonoverlap blend"

    def test_blend_lower_in_overlap(self):
        """overlap 영역 blend가 non-overlap보다 낮으면 blend_separation < 0."""
        S = 8
        masks = _make_masks(S=S, overlap_pixels=[0, 1],
                            excl0_pixels=[2, 3], excl1_pixels=[4, 5])
        blend = torch.zeros(1, S)
        blend[0, 0] = 0.1
        blend[0, 1] = 0.1
        blend[0, 2] = 0.8
        blend[0, 3] = 0.9
        blend[0, 4] = 0.8
        blend[0, 5] = 0.9

        stats = collect_blend_stats(blend, masks)
        assert stats["blend_separation"] < 0, \
            f"blend lower in overlap: separation should be < 0, got {stats['blend_separation']:.4f}"

    def test_all_background_separation_near_zero(self):
        """모든 pixel이 background → nonoverlap=0, overlap=0 → separation ≈ 0."""
        S = 8
        masks = torch.zeros(1, 2, S)  # no entities
        blend = torch.rand(1, S)
        stats = collect_blend_stats(blend, masks)
        # When no entities, overlap=0 and nonoverlap=0, so stats are 0/eps
        # separation = 0/(eps) - 0/(eps) = 0
        assert abs(stats["blend_separation"]) < 1e-3, \
            f"all background: separation should ≈ 0, got {stats['blend_separation']:.4f}"

    def test_stats_keys_present(self):
        """collect_blend_stats가 필요한 key들 반환."""
        masks = _make_masks(S=4, overlap_pixels=[0], excl0_pixels=[1], excl1_pixels=[2])
        blend = torch.rand(1, 4)
        stats = collect_blend_stats(blend, masks)
        for key in ["blend_mean", "blend_std", "blend_overlap_mean",
                    "blend_nonoverlap_mean", "blend_separation"]:
            assert key in stats, f"key '{key}' not in stats"

    def test_blend_mean_correct(self):
        """blend_mean = mean of all blend values."""
        S = 4
        masks = _make_masks(S=S, overlap_pixels=[0], excl0_pixels=[1])
        blend = torch.tensor([[0.2, 0.4, 0.6, 0.8]])
        stats = collect_blend_stats(blend, masks)
        expected_mean = blend.mean().item()
        assert abs(stats["blend_mean"] - expected_mean) < 1e-5, \
            f"blend_mean: expected {expected_mean:.4f}, got {stats['blend_mean']:.4f}"


# =============================================================================
# l_blend_overlap 검증
# =============================================================================

class TestLBlendOverlap:

    def test_separation_above_margin_gives_zero_loss(self):
        """blend_overlap - blend_nonoverlap > margin → loss = 0."""
        S = 8
        masks = _make_masks(S=S, overlap_pixels=[0, 1],
                            excl0_pixels=[2, 3], excl1_pixels=[4, 5])
        # overlap blend = 0.9, nonoverlap blend = 0.1 → separation = 0.8
        blend = torch.zeros(1, S)
        blend[0, 0] = 0.9; blend[0, 1] = 0.9
        blend[0, 2] = 0.1; blend[0, 3] = 0.1
        blend[0, 4] = 0.1; blend[0, 5] = 0.1

        loss = l_blend_overlap(blend, masks, margin=0.05)
        assert loss.item() == 0.0, \
            f"separation=0.8 > margin=0.05: loss should be 0, got {loss.item():.4f}"

    def test_negative_separation_gives_positive_loss(self):
        """blend_overlap < blend_nonoverlap → loss > 0."""
        S = 8
        masks = _make_masks(S=S, overlap_pixels=[0, 1],
                            excl0_pixels=[2, 3], excl1_pixels=[4, 5])
        # overlap blend = 0.1, nonoverlap blend = 0.9 → separation = -0.8
        blend = torch.zeros(1, S)
        blend[0, 0] = 0.1; blend[0, 1] = 0.1
        blend[0, 2] = 0.9; blend[0, 3] = 0.9
        blend[0, 4] = 0.9; blend[0, 5] = 0.9

        loss = l_blend_overlap(blend, masks, margin=0.05)
        assert loss.item() > 0.0, \
            f"separation=-0.8: loss should be > 0, got {loss.item():.4f}"

    def test_no_overlap_region_gives_zero_loss(self):
        """overlap region 없음 → l_blend_overlap = 0 (no-op)."""
        S = 4
        # Only exclusive pixels, no overlap
        masks = _make_masks(S=S, excl0_pixels=[0, 1], excl1_pixels=[2, 3])
        blend = torch.rand(1, S)
        loss  = l_blend_overlap(blend, masks, margin=0.05)
        # n_ov < 1 → returns 0
        assert loss.item() == 0.0, \
            f"no overlap: loss should be 0, got {loss.item():.4f}"

    def test_gradient_flows_to_blend(self):
        """l_blend_overlap에서 blend_map으로 gradient가 흐르는지."""
        S = 8
        masks = _make_masks(S=S, overlap_pixels=[0, 1],
                            excl0_pixels=[2, 3], excl1_pixels=[4, 5])
        # Make loss > 0: overlap blend = 0.1, nonoverlap blend = 0.9
        blend = torch.zeros(1, S, requires_grad=True)
        with torch.no_grad():
            blend_init = blend.clone()
            blend_init[0, 0] = 0.1; blend_init[0, 1] = 0.1
            blend_init[0, 2] = 0.9; blend_init[0, 3] = 0.9
            blend_init[0, 4] = 0.9; blend_init[0, 5] = 0.9
        blend = blend_init.detach().requires_grad_(True)

        loss = l_blend_overlap(blend, masks, margin=0.05)
        assert loss.item() > 0.0, "Loss must be > 0 for gradient test"
        loss.backward()
        assert blend.grad is not None, "gradient not flowing to blend_map"
        assert blend.grad.abs().max() > 0, "blend_map gradient is zero"

    def test_loss_with_margin_boundary(self):
        """
        separation = margin exactly → loss ≈ 0.
        relu(margin - margin) = relu(0) = 0.
        """
        S = 8
        masks  = _make_masks(S=S, overlap_pixels=[0, 1],
                              excl0_pixels=[2, 3], excl1_pixels=[4, 5])
        margin = 0.1
        # blend_ov = 0.5, blend_non = 0.4 → separation = 0.1 = margin
        blend = torch.zeros(1, S)
        blend[0, 0] = 0.5; blend[0, 1] = 0.5
        blend[0, 2] = 0.4; blend[0, 3] = 0.4
        blend[0, 4] = 0.4; blend[0, 5] = 0.4

        loss = l_blend_overlap(blend, masks, margin=margin)
        assert loss.item() < 1e-5, \
            f"separation=margin: loss should be ≈ 0, got {loss.item():.4f}"


# =============================================================================
# val_score_phase43 검증
# =============================================================================

class TestValScorePhase43:

    def test_formula(self):
        """val_score = 0.20*iou0 + 0.20*iou1 + 0.15*ord + 0.15*(1-wrong) + 0.15*r0 + 0.15*r1."""
        iou0, iou1 = 0.5, 0.4
        ord_, wrong = 0.7, 0.1
        r0, r1 = 0.3, 0.2
        expected = 0.20*0.5 + 0.20*0.4 + 0.15*0.7 + 0.15*(1-0.1) + 0.15*0.3 + 0.15*0.2
        result   = val_score_phase43(iou0, iou1, ord_, wrong, r0, r1)
        assert abs(result - expected) < 1e-6, \
            f"val_score formula: expected={expected:.6f}, got={result:.6f}"

    def test_perfect_score(self):
        """iou=1, ord=1, wrong=0, rollout=1 → 1.0."""
        result = val_score_phase43(1.0, 1.0, 1.0, 0.0, 1.0, 1.0)
        assert abs(result - 1.0) < 1e-6, f"perfect score: expected 1.0, got {result:.6f}"

    def test_worst_score(self):
        """iou=0, ord=0, wrong=1, rollout=0 → 0.0."""
        result = val_score_phase43(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        assert abs(result - 0.0) < 1e-6, f"worst score: expected 0.0, got {result:.6f}"

    def test_rollout_included(self):
        """rollout iou가 0일 때와 0.5일 때 val_score가 다름."""
        s_no_rollout   = val_score_phase43(0.5, 0.5, 0.7, 0.1, 0.0, 0.0)
        s_with_rollout = val_score_phase43(0.5, 0.5, 0.7, 0.1, 0.5, 0.5)
        assert s_with_rollout > s_no_rollout, \
            f"rollout should increase score: {s_no_rollout:.4f} vs {s_with_rollout:.4f}"

    def test_weights_sum_to_one(self):
        """가중치 합 = 1.0 (0.20+0.20+0.15+0.15+0.15+0.15 = 1.00)."""
        total = 0.20 + 0.20 + 0.15 + 0.15 + 0.15 + 0.15
        assert abs(total - 1.0) < 1e-6, f"weights don't sum to 1.0: {total}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
