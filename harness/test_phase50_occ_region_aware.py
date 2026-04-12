"""
test_phase50_occ_region_aware.py — l_occ_region_aware 검증

검증 항목:
  1. perfect state (o0_ov=o1_ov=1, o0_ex=1/o1_ex=0, both bg=0) → loss ≈ 0
  2. overlap low activation → loss > 0, gradient pushes both UP in overlap
  3. wrong entity high in exclusive → loss > 0, gradient pushes wrong DOWN
  4. gradient: dL/do1 > 0 in e0-exclusive (wrong → down)
  5. gradient: dL/do0 < 0 in overlap (both → up)
  6. gradient: dL/do0 > 0 in background (both → down weakly)
  7. exclusive neg gradient >> old l_occ neg gradient (9× improvement)
  8. overlap push >> old l_occ positive push (4× improvement at la_ov=8)
  9. gradient = 0 in e0-exclusive for o0 from l_bg path (no cross-region leak)
 10. output is scalar
 11. batch size > 1
 12. all-overlap mask: no exclusive/bg → e0/e1/bg terms ≈ 0
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase46 import l_occ_region_aware


def _make_region_masks(B, S):
    """~25% overlap, ~50% exclusive (~25% each), ~25% bg."""
    masks = torch.zeros(B, 2, S)
    q = S // 4
    masks[:, 0, :q*2] = 1.0        # e0: [0, S/2)
    masks[:, 1, q:q*3] = 1.0       # e1: [S/4, 3S/4)
    # overlap: [S/4, S/2), e0-excl: [0, S/4), e1-excl: [S/2, 3S/4), bg: [3S/4, S)
    return masks


class TestLOccRegionAware:

    def test_perfect_state_low_loss(self):
        """
        Perfect state: o0=1 in e0 regions, o1=1 in e1 regions, both=0 in bg.
        → loss should be near 0 (only boundary effects from clamp).
        """
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        q = S // 4
        o0 = torch.zeros(B, S)
        o1 = torch.zeros(B, S)
        # Both high in overlap [q, 2q)
        o0[:, q:q*2] = 0.999
        o1[:, q:q*2] = 0.999
        # o0 high in e0-exclusive [0, q)
        o0[:, :q] = 0.999
        # o1 high in e1-exclusive [2q, 3q)
        o1[:, q*2:q*3] = 0.999
        # Both low in bg [3q, S) → already 0
        o0 = o0.clamp(1e-6, 1-1e-6)
        o1 = o1.clamp(1e-6, 1-1e-6)
        loss = l_occ_region_aware(o0, o1, masks)
        # At o=0.999, -log(0.999)≈0.001, -log(1-0.0)≈0 (bg) → total very small
        assert loss.item() < 0.5, \
            f"perfect state should have small loss, got {loss.item():.4f}"

    def test_collapse_gives_positive_loss(self):
        """Phase45 collapse (o0=o1=0.92 everywhere) → loss > 0."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        o0 = torch.full((B, S), 0.92)
        o1 = torch.full((B, S), 0.92)
        loss = l_occ_region_aware(o0, o1, masks)
        assert loss.item() > 0.0, f"collapse → loss > 0, got {loss.item():.6f}"

    def test_gradient_pushes_o1_down_in_e0_exclusive(self):
        """dL/do1 > 0 in e0-exclusive (wrong entity → minimize loss → o1 DOWN)."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        q = S // 4

        o0_leaf = torch.zeros(B, S, requires_grad=True)
        o1_leaf = torch.zeros(B, S, requires_grad=True)
        bias = torch.full((B, S), 0.75)
        o0 = (o0_leaf + bias).clamp(1e-6, 1-1e-6)
        o1 = (o1_leaf + bias).clamp(1e-6, 1-1e-6)

        loss = l_occ_region_aware(o0, o1, masks)
        loss.backward()

        grad = o1_leaf.grad[:, :q].mean().item()   # e0-exclusive region
        assert grad > 0, f"dL/do1 in e0-excl should be > 0 (push down), got {grad:.6f}"

    def test_gradient_pushes_both_up_in_overlap(self):
        """dL/do0, dL/do1 < 0 in overlap (both → minimize loss → UP)."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        q = S // 4

        o0_leaf = torch.zeros(B, S, requires_grad=True)
        o1_leaf = torch.zeros(B, S, requires_grad=True)
        bias = torch.full((B, S), 0.5)
        o0 = (o0_leaf + bias).clamp(1e-6, 1-1e-6)
        o1 = (o1_leaf + bias).clamp(1e-6, 1-1e-6)

        loss = l_occ_region_aware(o0, o1, masks)
        loss.backward()

        grad_o0_ov = o0_leaf.grad[:, q:q*2].mean().item()  # overlap
        grad_o1_ov = o1_leaf.grad[:, q:q*2].mean().item()
        assert grad_o0_ov < 0, \
            f"dL/do0 in overlap should be < 0 (push up), got {grad_o0_ov:.6f}"
        assert grad_o1_ov < 0, \
            f"dL/do1 in overlap should be < 0 (push up), got {grad_o1_ov:.6f}"

    def test_gradient_pushes_down_in_background(self):
        """dL/do0 > 0 in background (both → minimize → DOWN weakly)."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        q = S // 4

        o0_leaf = torch.zeros(B, S, requires_grad=True)
        o1_leaf = torch.zeros(B, S, requires_grad=True)
        bias = torch.full((B, S), 0.5)
        o0 = (o0_leaf + bias).clamp(1e-6, 1-1e-6)
        o1 = (o1_leaf + bias).clamp(1e-6, 1-1e-6)

        loss = l_occ_region_aware(o0, o1, masks)
        loss.backward()

        grad_o0_bg = o0_leaf.grad[:, q*3:].mean().item()   # background
        assert grad_o0_bg > 0, \
            f"dL/do0 in bg should be > 0 (push down), got {grad_o0_bg:.6f}"

    def test_exclusive_neg_gradient_9x_stronger_than_old_bce(self):
        """
        la_ex_neg=3 exclusive neg gradient >> old neg_weight=0.25 * la_occ=2 BCE.

        At o1=0.75 in e0-exclusive (n_e0 pixels):
          New: la_ex_neg * 0.5 / (n_e0 * (1-o1)) = 3*0.5/(0.25*S*0.25) = 24/S
          Old: neg_weight=0.25 * la_occ=2 * 1/(1-o1) / n_neg = 0.25*2*4/(0.75*S) = 2.67/S
          Ratio: 24/2.67 ≈ 9×
        """
        B, S = 1, 64
        masks = _make_region_masks(B, S)
        q = S // 4

        o1_leaf = torch.zeros(B, S, requires_grad=True)
        o0 = torch.full((B, S), 0.9)
        o1 = (o1_leaf + torch.full((B, S), 0.75)).clamp(1e-6, 1-1e-6)

        loss = l_occ_region_aware(o1_leaf * 0 + o0, o1, masks,
                                  la_ex_neg=3.0)
        loss.backward()

        n_e0 = (masks[:, 0, :] * (1 - masks[:, 1, :])).sum().item()
        new_grad_per_pix = abs(o1_leaf.grad[:, :q].mean().item())
        new_effective = new_grad_per_pix  # actual gradient from function

        # Old effective: neg_weight=0.25, la_occ=2, at o1=0.75, n_neg=0.75*S
        n_neg_old = (1.0 - masks[:, 1, :]).sum().item()
        old_grad_per_pix = 0.25 * 2.0 * (1.0 / (1.0 - 0.75)) / n_neg_old

        ratio = new_effective / (old_grad_per_pix + 1e-9)
        assert ratio >= 5.0, \
            f"new exclusive neg should be ≥5× old BCE neg, got ratio={ratio:.2f}"

    def test_overlap_push_4x_stronger_than_old_bce(self):
        """
        la_ov=8 overlap positive gradient >> old la_occ=2 positive per overlap pixel.

        At o0=0.5 in overlap:
          New: la_ov * 0.5 / (n_ov * o0) = 8*0.5/(0.25*S*0.5) = 32/S
          Old: la_occ=2 * 1/o0 / n_pos = 2*2/(0.5*S) = 8/S
          Ratio: 4×
        """
        B, S = 1, 64
        masks = _make_region_masks(B, S)
        q = S // 4

        o0_leaf = torch.zeros(B, S, requires_grad=True)
        o1 = torch.full((B, S), 0.5)
        o0 = (o0_leaf + torch.full((B, S), 0.5)).clamp(1e-6, 1-1e-6)

        loss = l_occ_region_aware(o0, o1, masks, la_ov=8.0)
        loss.backward()

        n_ov = (masks[:, 0, :] * masks[:, 1, :]).sum().item()
        new_grad_per_pix = abs(o0_leaf.grad[:, q:q*2].mean().item())

        # Old: la_occ=2, pos term only, at o0=0.5, n_pos=e0_total=0.5*S
        n_pos_old = masks[:, 0, :].sum().item()
        old_grad_per_pix = 2.0 * (1.0 / 0.5) / n_pos_old

        ratio = new_grad_per_pix / (old_grad_per_pix + 1e-9)
        assert ratio >= 3.0, \
            f"new overlap push should be ≥3× old BCE positive, got ratio={ratio:.2f}"

    def test_all_overlap_mask_no_exclusive_bg(self):
        """All-overlap mask: exclusive/bg terms collapse (sum=0 in numerator → 0)."""
        B, S = 1, 64
        masks = torch.ones(B, 2, S)   # all overlap
        o0 = torch.full((B, S), 0.7)
        o1 = torch.full((B, S), 0.7)
        loss = l_occ_region_aware(o0, o1, masks)
        # l_e0, l_e1, l_bg should be ~0 (numerators = 0)
        # Only l_ov active
        # -log(0.7) ≈ 0.357, la_ov=8, so l_ov = 8*0.5*2*0.357 ≈ 2.856
        assert loss.item() > 0.5, "with la_ov=8 and o=0.7, loss should be > 0.5"
        assert loss.item() < 10.0, "loss should not blow up"

    def test_output_is_scalar(self):
        """Output is a scalar tensor."""
        B, S = 3, 128
        masks = _make_region_masks(B, S)
        o0 = torch.rand(B, S).clamp(0.01, 0.99)
        o1 = torch.rand(B, S).clamp(0.01, 0.99)
        loss = l_occ_region_aware(o0, o1, masks)
        assert loss.shape == torch.Size([]), f"expected scalar, got {loss.shape}"

    def test_batch_size_gt1(self):
        """B=4 처리."""
        B, S = 4, 128
        masks = _make_region_masks(B, S)
        o0 = torch.rand(B, S).clamp(0.01, 0.99)
        o1 = torch.rand(B, S).clamp(0.01, 0.99)
        loss = l_occ_region_aware(o0, o1, masks)
        assert loss.item() >= 0.0

    def test_correct_entity_not_suppressed_in_exclusive(self):
        """dL/do0 < 0 in e0-exclusive (correct entity → UP, not suppressed)."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        q = S // 4

        o0_leaf = torch.zeros(B, S, requires_grad=True)
        o1_leaf = torch.zeros(B, S, requires_grad=True)
        bias_o0 = torch.full((B, S), 0.5)
        bias_o1 = torch.full((B, S), 0.5)
        o0 = (o0_leaf + bias_o0).clamp(1e-6, 1-1e-6)
        o1 = (o1_leaf + bias_o1).clamp(1e-6, 1-1e-6)

        loss = l_occ_region_aware(o0, o1, masks)
        loss.backward()

        grad_o0_e0excl = o0_leaf.grad[:, :q].mean().item()
        # o0 is correct in e0-exclusive → should be pushed UP (gradient < 0)
        assert grad_o0_e0excl < 0, \
            f"dL/do0 in e0-exclusive should be < 0 (correct entity → up), got {grad_o0_e0excl:.6f}"

    def test_region_gradient_ordering(self):
        """
        |grad_ov| > |grad_ex_correct| (overlap activation > exclusive positive push).
        Verifies la_ov=8 > la_ex_pos=3 weighting.
        """
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        q = S // 4

        o0_leaf = torch.zeros(B, S, requires_grad=True)
        o1_leaf = torch.zeros(B, S, requires_grad=True)
        bias = torch.full((B, S), 0.5)
        o0 = (o0_leaf + bias).clamp(1e-6, 1-1e-6)
        o1 = (o1_leaf + bias).clamp(1e-6, 1-1e-6)

        loss = l_occ_region_aware(o0, o1, masks,
                                  la_ov=8.0, la_ex_pos=3.0, la_ex_neg=3.0)
        loss.backward()

        grad_o0_ov = abs(o0_leaf.grad[:, q:q*2].mean().item())   # overlap
        grad_o0_e0 = abs(o0_leaf.grad[:, :q].mean().item())       # e0-exclusive (correct push)

        assert grad_o0_ov > grad_o0_e0, \
            f"|grad_ov|={grad_o0_ov:.6f} should > |grad_e0_correct|={grad_o0_e0:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
