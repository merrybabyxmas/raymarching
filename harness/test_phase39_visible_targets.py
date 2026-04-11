"""
Phase 39 — Test: build_visible_targets correctness
====================================================

build_visible_targets가 아래 규칙을 정확히 따르는지 검증.

  exclusive e0 (m0=1, m1=0): w0_target=1, w1_target=0
  exclusive e1 (m0=0, m1=1): w0_target=0, w1_target=1
  overlap    (m0=1, m1=1):   front entity target=1, back entity target=0
  background (m0=0, m1=0):   w0_target=0, w1_target=0
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot import (
    build_visible_targets,
    l_visible_weights,
    l_wrong_slot_suppression,
    l_sigma_spatial,
    compute_overlap_score,
    val_slot_score,
)

pytestmark = pytest.mark.phase39


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_masks():
    """
    B=1, S=6 마스크:
      pos 0: e0 exclusive (m0=1, m1=0)
      pos 1: e1 exclusive (m0=0, m1=1)
      pos 2: overlap      (m0=1, m1=1)
      pos 3: background   (m0=0, m1=0)
      pos 4: e0 exclusive
      pos 5: overlap
    """
    m = torch.zeros(1, 2, 6)
    m[0, 0, 0] = 1.0   # e0 exclusive
    m[0, 1, 1] = 1.0   # e1 exclusive
    m[0, 0, 2] = 1.0   # overlap
    m[0, 1, 2] = 1.0
    # pos 3: background
    m[0, 0, 4] = 1.0   # e0 exclusive
    m[0, 0, 5] = 1.0   # overlap
    m[0, 1, 5] = 1.0
    return m


@pytest.fixture
def batch_masks():
    """B=2 배치 마스크."""
    m = torch.zeros(2, 2, 4)
    # batch 0: e0 front
    m[0, 0, 0] = 1.0   # e0 excl
    m[0, 1, 1] = 1.0   # e1 excl
    m[0, 0, 2] = 1.0   # overlap
    m[0, 1, 2] = 1.0
    # batch 1: e1 front
    m[1, 0, 0] = 1.0   # e0 excl
    m[1, 1, 1] = 1.0   # e1 excl
    m[1, 0, 3] = 1.0   # overlap
    m[1, 1, 3] = 1.0
    return m


# =============================================================================
# Tests: build_visible_targets
# =============================================================================

class TestBuildVisibleTargets:

    def test_exclusive_e0_w0_target_is_1(self, simple_masks):
        """e0 exclusive 위치에서 w0_target=1."""
        w0t, w1t = build_visible_targets(simple_masks, [(0, 1)])
        assert w0t[0, 0] == pytest.approx(1.0), "e0 exclusive: w0_target should be 1"
        assert w0t[0, 4] == pytest.approx(1.0), "e0 exclusive: w0_target should be 1"

    def test_exclusive_e0_w1_target_is_0(self, simple_masks):
        """e0 exclusive 위치에서 w1_target=0."""
        w0t, w1t = build_visible_targets(simple_masks, [(0, 1)])
        assert w1t[0, 0] == pytest.approx(0.0), "e0 exclusive: w1_target should be 0"
        assert w1t[0, 4] == pytest.approx(0.0), "e0 exclusive: w1_target should be 0"

    def test_exclusive_e1_w1_target_is_1(self, simple_masks):
        """e1 exclusive 위치에서 w1_target=1."""
        w0t, w1t = build_visible_targets(simple_masks, [(0, 1)])
        assert w1t[0, 1] == pytest.approx(1.0), "e1 exclusive: w1_target should be 1"

    def test_exclusive_e1_w0_target_is_0(self, simple_masks):
        """e1 exclusive 위치에서 w0_target=0."""
        w0t, w1t = build_visible_targets(simple_masks, [(0, 1)])
        assert w0t[0, 1] == pytest.approx(0.0), "e1 exclusive: w0_target should be 0"

    def test_background_both_zero(self, simple_masks):
        """background 위치에서 w0_target=w1_target=0."""
        w0t, w1t = build_visible_targets(simple_masks, [(0, 1)])
        assert w0t[0, 3] == pytest.approx(0.0), "background: w0_target should be 0"
        assert w1t[0, 3] == pytest.approx(0.0), "background: w1_target should be 0"

    def test_overlap_e0_front_w0_is_1(self, simple_masks):
        """e0가 front일 때, overlap 위치에서 w0_target=1."""
        w0t, w1t = build_visible_targets(simple_masks, [(0, 1)])  # front=e0
        assert w0t[0, 2] == pytest.approx(1.0), "overlap, e0 front: w0_target=1"
        assert w0t[0, 5] == pytest.approx(1.0), "overlap, e0 front: w0_target=1"

    def test_overlap_e0_front_w1_is_0(self, simple_masks):
        """e0가 front일 때, overlap 위치에서 w1_target=0."""
        w0t, w1t = build_visible_targets(simple_masks, [(0, 1)])
        assert w1t[0, 2] == pytest.approx(0.0), "overlap, e0 front: w1_target=0"

    def test_overlap_e1_front_w1_is_1(self, simple_masks):
        """e1가 front일 때, overlap 위치에서 w1_target=1."""
        m = torch.zeros(1, 2, 4)
        m[0, 0, 0] = 1.0; m[0, 1, 0] = 1.0   # overlap
        w0t, w1t = build_visible_targets(m, [(1, 0)])   # front=e1
        assert w1t[0, 0] == pytest.approx(1.0), "overlap, e1 front: w1_target=1"
        assert w0t[0, 0] == pytest.approx(0.0), "overlap, e1 front: w0_target=0"

    def test_batch_independent(self, batch_masks):
        """배치별로 다른 depth_order가 올바르게 적용되는지."""
        w0t, w1t = build_visible_targets(batch_masks, [(0, 1), (1, 0)])
        # batch 0: e0 front → overlap pos 2에서 w0=1, w1=0
        assert w0t[0, 2] == pytest.approx(1.0)
        assert w1t[0, 2] == pytest.approx(0.0)
        # batch 1: e1 front → overlap pos 3에서 w1=1, w0=0
        assert w1t[1, 3] == pytest.approx(1.0)
        assert w0t[1, 3] == pytest.approx(0.0)

    def test_output_shape(self, simple_masks):
        """출력 shape이 (B, S)인지."""
        B, _, S = simple_masks.shape
        w0t, w1t = build_visible_targets(simple_masks, [(0, 1)])
        assert w0t.shape == (B, S)
        assert w1t.shape == (B, S)

    def test_targets_are_binary(self, simple_masks):
        """w0_target, w1_target가 0 또는 1만 포함하는지."""
        w0t, w1t = build_visible_targets(simple_masks, [(0, 1)])
        vals = torch.cat([w0t.flatten(), w1t.flatten()]).unique()
        for v in vals:
            assert v.item() in (0.0, 1.0), f"Target should be binary, got {v.item()}"


# =============================================================================
# Tests: l_visible_weights
# =============================================================================

class TestLVisibleWeights:

    def test_perfect_prediction_zero_loss(self, simple_masks):
        """GT target과 완벽히 일치하면 loss=0."""
        w0t, w1t = build_visible_targets(simple_masks, [(0, 1)])
        loss = l_visible_weights(w0t, w1t, simple_masks, [(0, 1)])
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-5)

    def test_wrong_prediction_positive_loss(self, simple_masks):
        """GT target과 다르면 loss > 0."""
        # 모두 0.5 예측
        B, _, S = simple_masks.shape
        w0 = torch.full((B, S), 0.5)
        w1 = torch.full((B, S), 0.5)
        loss = l_visible_weights(w0, w1, simple_masks, [(0, 1)])
        assert float(loss.item()) > 0.0

    def test_backward_computes_gradient(self, simple_masks):
        """gradient가 w0, w1로 흐르는지."""
        B, _, S = simple_masks.shape
        w0 = torch.zeros(B, S, requires_grad=True)
        w1 = torch.zeros(B, S, requires_grad=True)
        loss = l_visible_weights(w0, w1, simple_masks, [(0, 1)])
        loss.backward()
        assert w0.grad is not None
        assert w1.grad is not None
        assert not torch.all(w0.grad == 0), "grad should be non-zero"

    def test_background_ignored(self):
        """background 픽셀은 loss에 기여하지 않아야 함."""
        m = torch.zeros(1, 2, 4)   # 전부 background
        w0 = torch.ones(1, 4) * 0.9
        w1 = torch.ones(1, 4) * 0.9
        loss = l_visible_weights(w0, w1, m, [(0, 1)])
        # background만이면 n = eps → loss≈0
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-4)

    def test_loss_is_finite(self, simple_masks):
        """loss가 finite한지."""
        B, _, S = simple_masks.shape
        w0 = torch.rand(B, S)
        w1 = torch.rand(B, S)
        loss = l_visible_weights(w0, w1, simple_masks, [(0, 1)])
        assert torch.isfinite(loss)


# =============================================================================
# Tests: l_wrong_slot_suppression
# =============================================================================

class TestLWrongSlot:

    def test_perfect_separation_zero_loss(self, simple_masks):
        """
        e0 exclusive에서 w1=0, e1 exclusive에서 w0=0이면 loss=0.
        """
        B, _, S = simple_masks.shape
        w0 = torch.zeros(B, S)
        w1 = torch.zeros(B, S)
        # e0 exclusive에서 w0=1 (올바른 entity), w1=0
        w0[0, 0] = 1.0; w0[0, 4] = 1.0
        # e1 exclusive에서 w1=1 (올바른 entity), w0=0
        w1[0, 1] = 1.0
        loss = l_wrong_slot_suppression(w0, w1, simple_masks)
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-5)

    def test_wrong_entity_in_exclusive_penalized(self, simple_masks):
        """
        e0 exclusive에서 w1 > 0이면 loss > 0.
        """
        B, _, S = simple_masks.shape
        w0 = torch.zeros(B, S)
        w1 = torch.zeros(B, S)
        w1[0, 0] = 0.5   # e0 exclusive에 wrong entity weight
        loss = l_wrong_slot_suppression(w0, w1, simple_masks)
        assert float(loss.item()) > 0.0

    def test_backward_ok(self, simple_masks):
        """gradient가 흐르는지."""
        B, _, S = simple_masks.shape
        w0 = torch.rand(B, S, requires_grad=True)
        w1 = torch.rand(B, S, requires_grad=True)
        loss = l_wrong_slot_suppression(w0, w1, simple_masks)
        loss.backward()
        assert w0.grad is not None
        assert w1.grad is not None


# =============================================================================
# Tests: compute_overlap_score
# =============================================================================

class TestComputeOverlapScore:

    def test_no_overlap_is_zero(self):
        """겹침이 없으면 score=0."""
        m = np.zeros((4, 2, 16))
        m[:, 0, :8]  = 1.0   # e0 left half
        m[:, 1, 8:]  = 1.0   # e1 right half
        score = compute_overlap_score(m)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_full_overlap_is_one(self):
        """완전히 겹치면 score=1."""
        m = np.ones((4, 2, 16))
        score = compute_overlap_score(m)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_partial_overlap(self):
        """부분 겹침은 0 < score < 1."""
        m = np.zeros((4, 2, 8))
        m[:, 0, :4] = 1.0   # e0: pos 0-3
        m[:, 1, 2:6] = 1.0  # e1: pos 2-5, overlap at 2-3
        score = compute_overlap_score(m)
        assert 0.0 < score < 1.0

    def test_output_range(self):
        """score ∈ [0, 1]."""
        rng = np.random.default_rng(42)
        m = (rng.random((8, 2, 32)) > 0.5).astype(np.float32)
        score = compute_overlap_score(m)
        assert 0.0 <= score <= 1.0


# =============================================================================
# Tests: val_slot_score
# =============================================================================

class TestValSlotScore:

    def test_perfect_score(self):
        """visible_iou=1, ordering_acc=1, wrong_leak=0, dra=1 → score=1."""
        s = val_slot_score(1.0, 1.0, 0.0, 1.0)
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_zero_score(self):
        """visible_iou=0, ordering_acc=0, wrong_leak=1, dra=0 → score=0."""
        s = val_slot_score(0.0, 0.0, 1.0, 0.0)
        assert s == pytest.approx(0.0, abs=1e-6)

    def test_visible_iou_most_important(self):
        """visible_iou weight=0.4 (가장 큼)."""
        # visible_iou만 개선
        s_base = val_slot_score(0.0, 0.5, 0.5, 0.5)
        s_vis  = val_slot_score(1.0, 0.5, 0.5, 0.5)
        assert s_vis - s_base == pytest.approx(0.4, abs=1e-6)

    def test_ordering_acc_second(self):
        """ordering_acc weight=0.3."""
        s_base = val_slot_score(0.5, 0.0, 0.5, 0.5)
        s_ord  = val_slot_score(0.5, 1.0, 0.5, 0.5)
        assert s_ord - s_base == pytest.approx(0.3, abs=1e-6)


# =============================================================================
# l_sigma_spatial tests
# =============================================================================

class TestSigmaSpatial:

    def test_zero_loss_for_perfect_prediction(self):
        """alpha = mask → loss = 0."""
        m = torch.zeros(2, 2, 8)
        m[0, 0, :4] = 1.0
        m[1, 1, 4:] = 1.0
        alpha0 = m[:, 0, :]
        alpha1 = m[:, 1, :]
        loss = l_sigma_spatial(alpha0, alpha1, m)
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_loss_for_mismatch(self):
        """alpha = 0.5 everywhere, mask = binary → loss > 0."""
        m = torch.zeros(1, 2, 8)
        m[0, 0, :4] = 1.0
        m[0, 1, 4:] = 1.0
        alpha0 = torch.full((1, 8), 0.5)
        alpha1 = torch.full((1, 8), 0.5)
        loss = l_sigma_spatial(alpha0, alpha1, m)
        assert float(loss.item()) > 0.0

    def test_gradient_flows_to_alpha0_alpha1(self):
        """loss.backward()가 alpha0, alpha1로 gradient를 전달하는지."""
        m = torch.zeros(2, 2, 6)
        m[:, 0, :3] = 1.0
        m[:, 1, 3:] = 1.0
        alpha0 = torch.rand(2, 6, requires_grad=True)
        alpha1 = torch.rand(2, 6, requires_grad=True)
        loss = l_sigma_spatial(alpha0, alpha1, m)
        loss.backward()
        assert alpha0.grad is not None
        assert alpha1.grad is not None
        assert not torch.all(alpha0.grad == 0)

    def test_loss_drives_alpha_toward_mask(self):
        """Gradient descent로 alpha가 mask에 수렴하는지."""
        import torch.optim as optim
        m = torch.zeros(1, 2, 8)
        m[0, 0, :4] = 1.0
        m[0, 1, 4:] = 1.0
        alpha0 = torch.nn.Parameter(torch.zeros(1, 8))
        alpha1 = torch.nn.Parameter(torch.zeros(1, 8))
        opt = optim.Adam([alpha0, alpha1], lr=0.1)
        init_loss = float(l_sigma_spatial(alpha0, alpha1, m).item())
        for _ in range(200):
            opt.zero_grad()
            loss = l_sigma_spatial(alpha0, alpha1, m)
            loss.backward()
            opt.step()
        final_loss = float(l_sigma_spatial(alpha0, alpha1, m).item())
        assert final_loss < init_loss * 0.01, \
            f"Loss should drop: {init_loss:.4f} → {final_loss:.4f}"
