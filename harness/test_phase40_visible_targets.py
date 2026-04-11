"""
Phase 40 — Test: per-entity visible IoU, val_score_phase40, visible mask computation
=====================================================================================

Phase 40에서 추가된 per-entity 지표와 스코어 함수 검증.

  compute_visible_masks  : front=fully visible, back=visible where front=0
  compute_visible_iou_e0 : w0 vs GT visible mask for entity 0
  compute_visible_iou_e1 : w1 vs GT visible mask for entity 1
  val_score_phase40      : 6-term weighted score
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase40 import (
    compute_visible_masks,
    compute_visible_iou_e0,
    compute_visible_iou_e1,
    compute_id_feature_margin,
    val_score_phase40,
)

pytestmark = pytest.mark.phase40


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def overlap_masks():
    """
    B=2, T=1, S=8 masks:
      batch 0: depth_order=(e0_front, e1_back)
        pos 0-2: e0 exclusive
        pos 3-5: e1 exclusive
        pos 6-7: overlap (both entities)
      batch 1: depth_order=(e1_front, e0_back)
        pos 0-3: e0 mask
        pos 2-7: e1 mask (overlap at 2-3)
    """
    # (B, T, 2, S) → squeeze T for per-frame usage
    m = torch.zeros(2, 2, 8)
    # batch 0
    m[0, 0, 0:3] = 1.0   # e0 exclusive
    m[0, 1, 3:6] = 1.0   # e1 exclusive
    m[0, 0, 6:8] = 1.0   # overlap
    m[0, 1, 6:8] = 1.0
    # batch 1
    m[1, 0, 0:4] = 1.0   # e0
    m[1, 1, 2:8] = 1.0   # e1 (overlap at 2-3)
    return m   # (B, 2, S)


# =============================================================================
# Tests: compute_visible_masks
# =============================================================================

class TestComputeVisibleMasks:

    def test_front_entity_fully_visible(self):
        """front entity visible mask = full entity mask."""
        # T=1, N=2, S=6
        masks = np.zeros((1, 2, 6))
        masks[0, 0, :3] = 1.0   # e0
        masks[0, 1, 2:] = 1.0   # e1 (overlap at 2)
        depth_orders = [(0, 1)]  # e0 front

        vis = compute_visible_masks(masks, depth_orders)  # (T, 2, S)
        # e0 (front): visible == masks[:,0,:]
        np.testing.assert_allclose(vis[0, 0], masks[0, 0], atol=1e-6,
                                   err_msg="front entity visible = full mask")

    def test_back_entity_occluded_in_overlap(self):
        """back entity visible only where front mask = 0."""
        masks = np.zeros((1, 2, 6))
        masks[0, 0, :4] = 1.0   # e0 (front)
        masks[0, 1, 2:] = 1.0   # e1 (back), overlap at 2-3
        depth_orders = [(0, 1)]  # e0 front, e1 back

        vis = compute_visible_masks(masks, depth_orders)
        # e1 visible = e1_mask * (1 - e0_mask)
        expected_e1 = masks[0, 1] * (1.0 - masks[0, 0])
        np.testing.assert_allclose(vis[0, 1], expected_e1, atol=1e-6,
                                   err_msg="back entity occluded in overlap region")

    def test_no_overlap_both_fully_visible(self):
        """겹침 없으면 두 entity 모두 완전히 visible."""
        masks = np.zeros((2, 2, 8))
        masks[:, 0, :4] = 1.0   # e0 left
        masks[:, 1, 4:] = 1.0   # e1 right
        depth_orders = [(0, 1), (1, 0)]

        vis = compute_visible_masks(masks, depth_orders)
        np.testing.assert_allclose(vis[:, 0], masks[:, 0], atol=1e-6)
        np.testing.assert_allclose(vis[:, 1], masks[:, 1], atol=1e-6)

    def test_output_shape(self):
        """출력 shape = (T, 2, S)."""
        T, S = 4, 16
        masks = np.random.rand(T, 2, S) > 0.5
        depth_orders = [(0, 1)] * T
        vis = compute_visible_masks(masks.astype(float), depth_orders)
        assert vis.shape == (T, 2, S)

    def test_output_range_0_to_1(self):
        """visible mask 값이 [0, 1]."""
        rng = np.random.default_rng(42)
        masks = (rng.random((4, 2, 16)) > 0.5).astype(float)
        depth_orders = [(0, 1)] * 4
        vis = compute_visible_masks(masks, depth_orders)
        assert vis.min() >= 0.0
        assert vis.max() <= 1.0

    def test_reversing_depth_order_flips_visibility(self):
        """depth order 반전 시 visible region이 바뀌는지."""
        masks = np.zeros((1, 2, 8))
        masks[0, 0, 2:6] = 1.0
        masks[0, 1, 4:8] = 1.0   # overlap at 4-5

        vis_e0_front = compute_visible_masks(masks, [(0, 1)])
        vis_e1_front = compute_visible_masks(masks, [(1, 0)])

        # When e0 front: overlap region visible for e0
        e0_front_overlap_vis = vis_e0_front[0, 0, 4:6].sum()
        # When e1 front: overlap region occluded for e0
        e1_front_overlap_vis = vis_e1_front[0, 0, 4:6].sum()
        assert e0_front_overlap_vis > e1_front_overlap_vis, \
            "e0 should be more visible when it is front"


# =============================================================================
# Tests: compute_visible_iou_e0 / compute_visible_iou_e1
# =============================================================================

class TestComputeVisibleIoU:

    def _make_weights(self, B, S, entity_idx, masks_BNS):
        """entity_idx에 대한 perfect weight (visible GT와 일치)."""
        depth_orders = [(0, 1)] * B
        vis = compute_visible_masks(
            masks_BNS.numpy().reshape(1, B, 2, S)[0]
            if masks_BNS.ndim == 3 else masks_BNS.numpy(),
            depth_orders,
        )
        # Return the visible mask as "predicted" weight
        w = torch.tensor(vis[:, entity_idx, :], dtype=torch.float32)
        return w

    def test_perfect_w0_iou_is_1(self, overlap_masks):
        """w0 = GT visible → iou_e0 = 1."""
        B, N, S = overlap_masks.shape
        depth_orders = [(0, 1)] * B
        # GT visible for e0 with e0-front
        masks_np = overlap_masks.numpy()
        vis = compute_visible_masks(masks_np, depth_orders)
        w0 = vis[:, 0, :].float().detach().clone()

        iou = compute_visible_iou_e0(w0, overlap_masks, depth_orders)
        assert iou == pytest.approx(1.0, abs=0.01), \
            f"Perfect w0 prediction should give iou_e0≈1.0, got {iou:.4f}"

    def test_perfect_w1_iou_is_1(self, overlap_masks):
        """w1 = GT visible → iou_e1 = 1."""
        B, N, S = overlap_masks.shape
        depth_orders = [(0, 1)] * B
        masks_np = overlap_masks.numpy()
        vis = compute_visible_masks(masks_np, depth_orders)
        w1 = vis[:, 1, :].float().detach().clone()

        iou = compute_visible_iou_e1(w1, overlap_masks, depth_orders)
        assert iou == pytest.approx(1.0, abs=0.01), \
            f"Perfect w1 prediction should give iou_e1≈1.0, got {iou:.4f}"

    def test_zero_prediction_zero_iou(self, overlap_masks):
        """w0 = 0 everywhere → iou_e0 near 0 (no true positive)."""
        B, N, S = overlap_masks.shape
        depth_orders = [(0, 1)] * B
        w0 = torch.zeros(B, S)
        iou = compute_visible_iou_e0(w0, overlap_masks, depth_orders)
        assert iou <= 0.01, f"Zero w0 should give near-0 iou_e0, got {iou:.4f}"

    def test_iou_range_0_to_1(self, overlap_masks):
        """IoU 값이 [0, 1] 범위인지."""
        B, N, S = overlap_masks.shape
        depth_orders = [(0, 1)] * B
        w0 = torch.rand(B, S)
        w1 = torch.rand(B, S)
        iou0 = compute_visible_iou_e0(w0, overlap_masks, depth_orders)
        iou1 = compute_visible_iou_e1(w1, overlap_masks, depth_orders)
        assert 0.0 <= iou0 <= 1.0
        assert 0.0 <= iou1 <= 1.0

    def test_e0_and_e1_iou_independent(self, overlap_masks):
        """iou_e0와 iou_e1는 서로 독립적으로 계산되어야 함."""
        B, N, S = overlap_masks.shape
        depth_orders = [(0, 1)] * B
        masks_np = overlap_masks.numpy()
        vis = compute_visible_masks(masks_np, depth_orders)

        # Perfect w0, zero w1
        w0_perfect = vis[:, 0, :].float().detach().clone()
        w1_zero = torch.zeros(B, S)

        iou_e0 = compute_visible_iou_e0(w0_perfect, overlap_masks, depth_orders)
        iou_e1 = compute_visible_iou_e1(w1_zero, overlap_masks, depth_orders)

        # iou_e0 should be high, iou_e1 should be low
        assert iou_e0 > 0.5, f"iou_e0 should be high with perfect w0: {iou_e0:.4f}"
        assert iou_e1 < 0.5, f"iou_e1 should be low with zero w1: {iou_e1:.4f}"


# =============================================================================
# Tests: val_score_phase40
# =============================================================================

class TestValScorePhase40:

    def test_perfect_score(self):
        """iou_e0=1, iou_e1=1, ordering=1, wrong_leak=0, id_margin=1, rollout_id=1 → 1.0."""
        s = val_score_phase40(1.0, 1.0, 1.0, 0.0, 1.0, rollout_id=1.0)
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_zero_score(self):
        """모든 지표 최악 → 0.0."""
        s = val_score_phase40(0.0, 0.0, 0.0, 1.0, 0.0, rollout_id=0.0)
        assert s == pytest.approx(0.0, abs=1e-6)

    def test_weights_sum_to_1(self):
        """각 term weight 합 = 1.0."""
        # val_score = 0.20*iou_e0 + 0.20*iou_e1 + 0.20*ord + 0.15*(1-leak) + 0.15*id + 0.10*rollout
        weight_sum = 0.20 + 0.20 + 0.20 + 0.15 + 0.15 + 0.10
        assert weight_sum == pytest.approx(1.0, abs=1e-6)

    def test_iou_e0_weight_020(self):
        """iou_e0만 1 개선 시 score += 0.20."""
        s_base = val_score_phase40(0.0, 0.5, 0.5, 0.5, 0.5, rollout_id=0.5)
        s_high = val_score_phase40(1.0, 0.5, 0.5, 0.5, 0.5, rollout_id=0.5)
        assert s_high - s_base == pytest.approx(0.20, abs=1e-6)

    def test_iou_e1_weight_020(self):
        """iou_e1만 1 개선 시 score += 0.20."""
        s_base = val_score_phase40(0.5, 0.0, 0.5, 0.5, 0.5, rollout_id=0.5)
        s_high = val_score_phase40(0.5, 1.0, 0.5, 0.5, 0.5, rollout_id=0.5)
        assert s_high - s_base == pytest.approx(0.20, abs=1e-6)

    def test_ordering_acc_weight_020(self):
        """ordering_acc만 1 개선 시 score += 0.20."""
        s_base = val_score_phase40(0.5, 0.5, 0.0, 0.5, 0.5, rollout_id=0.5)
        s_high = val_score_phase40(0.5, 0.5, 1.0, 0.5, 0.5, rollout_id=0.5)
        assert s_high - s_base == pytest.approx(0.20, abs=1e-6)

    def test_wrong_leak_weight_015(self):
        """wrong_leak 0→1 시 score -= 0.15."""
        s_low_leak  = val_score_phase40(0.5, 0.5, 0.5, 0.0, 0.5, rollout_id=0.5)
        s_high_leak = val_score_phase40(0.5, 0.5, 0.5, 1.0, 0.5, rollout_id=0.5)
        assert s_low_leak - s_high_leak == pytest.approx(0.15, abs=1e-6)

    def test_id_margin_weight_015(self):
        """id_margin만 1 개선 시 score += 0.15."""
        s_base = val_score_phase40(0.5, 0.5, 0.5, 0.5, 0.0, rollout_id=0.5)
        s_high = val_score_phase40(0.5, 0.5, 0.5, 0.5, 1.0, rollout_id=0.5)
        assert s_high - s_base == pytest.approx(0.15, abs=1e-6)

    def test_rollout_id_weight_010(self):
        """rollout_id만 1 개선 시 score += 0.10."""
        s_base = val_score_phase40(0.5, 0.5, 0.5, 0.5, 0.5, rollout_id=0.0)
        s_high = val_score_phase40(0.5, 0.5, 0.5, 0.5, 0.5, rollout_id=1.0)
        assert s_high - s_base == pytest.approx(0.10, abs=1e-6)

    def test_rollout_id_defaults_to_zero(self):
        """rollout_id 미제공 시 기본값 0.0으로 처리."""
        s_no_rollout   = val_score_phase40(0.5, 0.5, 0.5, 0.5, 0.5)
        s_zero_rollout = val_score_phase40(0.5, 0.5, 0.5, 0.5, 0.5, rollout_id=0.0)
        assert s_no_rollout == pytest.approx(s_zero_rollout, abs=1e-6)

    def test_score_monotone_in_iou(self):
        """iou_e0, iou_e1 증가 → score 증가."""
        scores = [val_score_phase40(v, v, 0.5, 0.3, 0.5) for v in [0.0, 0.25, 0.5, 0.75, 1.0]]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], \
                f"Score should increase with iou: {scores}"


# =============================================================================
# Tests: compute_id_feature_margin
# =============================================================================

class TestComputeIdFeatureMargin:

    def _make_features(self, B, S, D=32):
        """Random normalized feature vectors."""
        F = torch.randn(B, S, D)
        F = F / (F.norm(dim=-1, keepdim=True) + 1e-8)
        return F

    def test_identical_features_zero_margin(self):
        """
        F0_slot == F0_ref, F1_slot == F1_ref로 cos sim 모두 1이면
        margin = 0 (same_sim == other_sim일 때).
        실제로는 margin > 0인 경우도 있으므로 >= 0만 검증.
        """
        B, S, D = 2, 8, 32
        F = self._make_features(B, S, D)
        mask_e0 = torch.ones(B, S)
        mask_e1 = torch.ones(B, S)
        margin = compute_id_feature_margin(F, F, F, F, mask_e0, mask_e1)
        assert margin >= 0.0, f"Margin should be >= 0, got {margin:.4f}"

    def test_distinct_entities_positive_margin(self):
        """
        F0_slot ≈ F0_ref (서로 같은 entity), F1_slot ≈ F0_ref와 orthogonal이면
        id_margin > 0.
        """
        B, S, D = 2, 8, 64
        F0 = self._make_features(B, S, D)
        # F1: orthogonal to F0 (approx)
        F1 = torch.randn(B, S, D)
        F1 = F1 - (F1 * F0).sum(-1, keepdim=True) * F0
        F1 = F1 / (F1.norm(dim=-1, keepdim=True) + 1e-8)

        mask = torch.ones(B, S)
        # same = (F0_slot, F0_ref), other = (F0_slot, F1_ref)
        margin = compute_id_feature_margin(F0, F1, F0, F1, mask, mask)
        # cos(F0, F0) ≈ 1, cos(F0, F1) ≈ 0 → margin ≈ 1
        assert margin > 0.3, f"Orthogonal entities should have positive margin, got {margin:.4f}"

    def test_output_is_scalar(self):
        """반환값이 scalar float인지."""
        B, S, D = 2, 8, 32
        F = self._make_features(B, S, D)
        mask = torch.ones(B, S)
        margin = compute_id_feature_margin(F, F * 0.9, F, F * 0.9, mask, mask)
        assert isinstance(margin, float), f"Expected float, got {type(margin)}"

    def test_range_minus1_to_1(self):
        """margin ∈ [-1, 1] (cosine similarity 기반)."""
        B, S, D = 4, 16, 32
        F0 = self._make_features(B, S, D)
        F1 = self._make_features(B, S, D)
        mask = torch.ones(B, S)
        margin = compute_id_feature_margin(F0, F1, F0, F1, mask, mask)
        assert -1.0 <= margin <= 1.0 + 1e-4
