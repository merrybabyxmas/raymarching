"""
test_phase43_soft_targets.py — Soft visible targets 검증

검증 항목:
  1. overlap front entity: target = front_val (0.85)
  2. overlap back entity: target = back_val (0.05)
  3. exclusive entity0: target = 1.0
  4. exclusive entity1: target = 1.0
  5. background: target = 0.0
  6. l_visible_weights_soft: perfect alignment → loss < threshold
  7. l_visible_weights_soft: confused assignment → loss > threshold
  8. front_val/back_val 파라미터가 실제 target에 반영되는지
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase43 import (
    build_visible_targets_soft,
    l_visible_weights_soft,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_toy_masks_and_orders():
    """
    4 pixels:
      pixel 0: entity0 only (excl_0)
      pixel 1: entity1 only (excl_1)
      pixel 2: both (overlap)
      pixel 3: background
    depth order: entity0 front
    """
    B, N, S = 1, 2, 4
    masks = torch.zeros(B, N, S)
    masks[0, 0, 0] = 1.0  # e0 exclusive
    masks[0, 0, 2] = 1.0  # e0 overlap
    masks[0, 1, 1] = 1.0  # e1 exclusive
    masks[0, 1, 2] = 1.0  # e1 overlap
    depth_orders = [(0, 1)]  # e0 front
    return masks, depth_orders


# =============================================================================
# build_visible_targets_soft 검증
# =============================================================================

class TestBuildVisibleTargetsSoft:

    def test_exclusive_e0_target_is_one(self):
        """e0 exclusive pixel: w0_target = 1.0."""
        masks, do = _make_toy_masks_and_orders()
        w0_target, w1_target = build_visible_targets_soft(masks, do)
        # pixel 0 = e0 exclusive
        assert abs(w0_target[0, 0].item() - 1.0) < 1e-5, \
            f"e0 exclusive: w0_target={w0_target[0,0].item():.4f}, expected 1.0"

    def test_exclusive_e1_target_is_one(self):
        """e1 exclusive pixel: w1_target = 1.0."""
        masks, do = _make_toy_masks_and_orders()
        w0_target, w1_target = build_visible_targets_soft(masks, do)
        # pixel 1 = e1 exclusive
        assert abs(w1_target[0, 1].item() - 1.0) < 1e-5, \
            f"e1 exclusive: w1_target={w1_target[0,1].item():.4f}, expected 1.0"

    def test_overlap_front_entity_target(self):
        """overlap pixel: front entity (e0) target = front_val."""
        masks, do = _make_toy_masks_and_orders()
        front_val, back_val = 0.85, 0.05
        w0_target, w1_target = build_visible_targets_soft(
            masks, do, front_val=front_val, back_val=back_val)
        # pixel 2 = overlap, e0 front
        assert abs(w0_target[0, 2].item() - front_val) < 1e-5, \
            f"overlap front (e0): w0_target={w0_target[0,2].item():.4f}, expected {front_val}"

    def test_overlap_back_entity_target(self):
        """overlap pixel: back entity (e1) target = back_val."""
        masks, do = _make_toy_masks_and_orders()
        front_val, back_val = 0.85, 0.05
        w0_target, w1_target = build_visible_targets_soft(
            masks, do, front_val=front_val, back_val=back_val)
        # pixel 2 = overlap, e1 back
        assert abs(w1_target[0, 2].item() - back_val) < 1e-5, \
            f"overlap back (e1): w1_target={w1_target[0,2].item():.4f}, expected {back_val}"

    def test_background_target_is_zero(self):
        """background pixel: both w0_target and w1_target = 0."""
        masks, do = _make_toy_masks_and_orders()
        w0_target, w1_target = build_visible_targets_soft(masks, do)
        # pixel 3 = background
        assert abs(w0_target[0, 3].item()) < 1e-5, \
            f"background: w0_target={w0_target[0,3].item():.4f}, expected 0"
        assert abs(w1_target[0, 3].item()) < 1e-5, \
            f"background: w1_target={w1_target[0,3].item():.4f}, expected 0"

    def test_e1_front_flips_assignment(self):
        """e1이 front이면: overlap에서 w1_target=front_val, w0_target=back_val."""
        masks, _ = _make_toy_masks_and_orders()
        do_e1_front = [(1, 0)]
        front_val, back_val = 0.85, 0.05
        w0_target, w1_target = build_visible_targets_soft(
            masks, do_e1_front, front_val=front_val, back_val=back_val)
        # pixel 2 = overlap, e1 front
        assert abs(w1_target[0, 2].item() - front_val) < 1e-5, \
            f"e1 front overlap: w1_target={w1_target[0,2].item():.4f}, expected {front_val}"
        assert abs(w0_target[0, 2].item() - back_val) < 1e-5, \
            f"e1 front overlap: w0_target={w0_target[0,2].item():.4f}, expected {back_val}"

    def test_custom_front_back_values(self):
        """front_val=0.9, back_val=0.1 파라미터 반영."""
        masks, do = _make_toy_masks_and_orders()
        w0_target, w1_target = build_visible_targets_soft(
            masks, do, front_val=0.9, back_val=0.1)
        assert abs(w0_target[0, 2].item() - 0.9) < 1e-5, \
            f"front_val=0.9: got {w0_target[0,2].item():.4f}"
        assert abs(w1_target[0, 2].item() - 0.1) < 1e-5, \
            f"back_val=0.1: got {w1_target[0,2].item():.4f}"


# =============================================================================
# l_visible_weights_soft 검증
# =============================================================================

class TestLVisibleWeightsSoft:

    def test_perfect_w0_low_loss(self):
        """
        w0이 soft target과 정확히 일치 → loss가 낮음.
        """
        masks, do = _make_toy_masks_and_orders()
        front_val, back_val = 0.85, 0.05
        w0_target, w1_target = build_visible_targets_soft(
            masks, do, front_val=front_val, back_val=back_val)

        # Perfect assignment: w0 = soft target
        loss = l_visible_weights_soft(
            w0_target, w1_target, masks, do,
            front_val=front_val, back_val=back_val)
        assert loss.item() < 0.05, \
            f"perfect soft targets: loss should be near 0, got {loss.item():.4f}"

    def test_confused_w_high_loss(self):
        """
        w0이 e1 영역에 높고 w1이 e0 영역에 높으면 → loss가 큼.
        """
        masks, do = _make_toy_masks_and_orders()
        B, N, S = 1, 2, 4
        # Confused: w0 = [0, 1, 0, 0] (e1 position), w1 = [1, 0, 0, 0] (e0 position)
        w0_confused = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        w1_confused = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        loss = l_visible_weights_soft(
            w0_confused, w1_confused, masks, do)
        assert loss.item() > 0.1, \
            f"confused assignment: loss should be > 0.1, got {loss.item():.4f}"

    def test_soft_loss_less_than_hard_on_overlap(self):
        """
        overlap pixel에서 perfect soft w vs hard target:
        soft_loss < hard_loss (soft target이 더 쉽게 달성 가능).
        """
        from models.entity_slot import l_visible_weights

        masks, do = _make_toy_masks_and_orders()
        front_val, back_val = 0.85, 0.05
        w0_target, w1_target = build_visible_targets_soft(
            masks, do, front_val=front_val, back_val=back_val)

        l_soft = l_visible_weights_soft(
            w0_target, w1_target, masks, do,
            front_val=front_val, back_val=back_val)
        l_hard = l_visible_weights(w0_target, w1_target, masks, do)

        assert l_soft.item() <= l_hard.item() + 1e-5, \
            f"soft_loss={l_soft.item():.4f} should be <= hard_loss={l_hard.item():.4f} " \
            f"when using soft targets"

    def test_background_suppression(self):
        """
        w0 = 1 everywhere (even bg) → l_vis_soft includes bg penalty.
        """
        masks, do = _make_toy_masks_and_orders()
        w0_all_one = torch.ones(1, 4)
        w1_zero    = torch.zeros(1, 4)

        loss = l_visible_weights_soft(w0_all_one, w1_zero, masks, do)
        # bg pixel contributes penalty
        assert loss.item() > 0.0, \
            f"w0=1 everywhere: bg penalty should make loss > 0, got {loss.item():.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
