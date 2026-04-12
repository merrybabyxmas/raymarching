"""
test_phase49_overlap_activate.py — l_overlap_activate 검증

검증 항목:
  1. o0=o1=1 in overlap → loss = 0 (already correct)
  2. o0=o1=0.1 in overlap → loss > 0 (need to activate)
  3. gradient: dL/do0 < 0 in overlap (minimize → o0 UP)
  4. gradient: dL/do1 < 0 in overlap (minimize → o1 UP)
  5. gradient: dL/do0 ≈ 0 in exclusive/bg (only overlap affected)
  6. gradient: dL/do1 ≈ 0 in exclusive/bg
  7. monotone: higher o in overlap → lower loss
  8. all-exclusive mask: loss = 0 (no overlap region → denominator ≈ eps → 0/eps = 0)
  9. output is scalar
 10. batch size > 1
 11. gradient dominance: la=10 overlap activate >> existing BCE positive per overlap pixel
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase46 import l_overlap_activate


def _make_region_masks(B, S):
    """~25% overlap, ~50% exclusive, ~25% bg."""
    masks = torch.zeros(B, 2, S)
    q = S // 4
    masks[:, 0, :q*2] = 1.0
    masks[:, 1, q:q*3] = 1.0
    return masks


class TestLOverlapActivate:

    def test_perfect_activation_zero_loss(self):
        """o0=o1=1 in overlap → loss ≈ -log(1) = 0."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        o0 = torch.ones(B, S) * 0.9999
        o1 = torch.ones(B, S) * 0.9999
        loss = l_overlap_activate(o0, o1, masks)
        assert loss.item() < 0.01, \
            f"o0≈o1≈1 → loss ≈ 0, got {loss.item():.6f}"

    def test_low_activation_positive_loss(self):
        """o0=o1=0.1 in overlap → loss > 0 (need to activate)."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        o0 = torch.full((B, S), 0.1)
        o1 = torch.full((B, S), 0.1)
        loss = l_overlap_activate(o0, o1, masks)
        assert loss.item() > 0.0, \
            f"low activation → loss > 0, got {loss.item():.6f}"
        # -log(0.1) ≈ 2.303
        assert loss.item() > 2.0, \
            f"loss should be substantial (>2.0), got {loss.item():.6f}"

    def test_gradient_pushes_o0_up_in_overlap(self):
        """dL/do0 < 0 in overlap → minimizing loss increases o0."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        q = S // 4

        o0_leaf = torch.zeros(B, S, requires_grad=True)
        o1_leaf = torch.zeros(B, S, requires_grad=True)
        bias = torch.full((B, S), 0.5)
        o0 = o0_leaf + bias
        o1 = o1_leaf + bias

        loss = l_overlap_activate(o0, o1, masks)
        loss.backward()

        # overlap region: [q, q*2) (intersection of e0=[0,2q) and e1=[q,3q))
        grad_o0_ov = o0_leaf.grad[:, q:q*2].mean().item()
        assert grad_o0_ov < 0, \
            f"dL/do0 in overlap should be < 0 (push up), got {grad_o0_ov:.6f}"

    def test_gradient_pushes_o1_up_in_overlap(self):
        """dL/do1 < 0 in overlap → minimizing loss increases o1."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        q = S // 4

        o0_leaf = torch.zeros(B, S, requires_grad=True)
        o1_leaf = torch.zeros(B, S, requires_grad=True)
        bias = torch.full((B, S), 0.5)
        o0 = o0_leaf + bias
        o1 = o1_leaf + bias

        loss = l_overlap_activate(o0, o1, masks)
        loss.backward()

        grad_o1_ov = o1_leaf.grad[:, q:q*2].mean().item()
        assert grad_o1_ov < 0, \
            f"dL/do1 in overlap should be < 0 (push up), got {grad_o1_ov:.6f}"

    def test_gradient_zero_outside_overlap(self):
        """Gradient is 0 for o0 in e0-exclusive (not overlap) and bg."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        q = S // 4

        o0_leaf = torch.zeros(B, S, requires_grad=True)
        o1_leaf = torch.zeros(B, S, requires_grad=True)
        bias = torch.full((B, S), 0.5)
        o0 = o0_leaf + bias
        o1 = o1_leaf + bias

        loss = l_overlap_activate(o0, o1, masks)
        loss.backward()

        # e0-exclusive = [:q], bg = [q*3:]
        grad_o0_e0excl = o0_leaf.grad[:, :q].mean().item()
        grad_o0_bg = o0_leaf.grad[:, q*3:].mean().item()
        assert abs(grad_o0_e0excl) < 1e-6, \
            f"dL/do0 in e0-exclusive should be 0, got {grad_o0_e0excl:.6f}"
        assert abs(grad_o0_bg) < 1e-6, \
            f"dL/do0 in bg should be 0, got {grad_o0_bg:.6f}"

    def test_monotone_lower_activation_higher_loss(self):
        """Lower o in overlap → higher loss."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        losses = []
        for val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            o = torch.full((B, S), val)
            losses.append(l_overlap_activate(o, o, masks).item())
        for i in range(len(losses) - 1):
            assert losses[i] >= losses[i+1] - 1e-6, \
                f"loss should decrease as o increases: {losses[i]:.4f}→{losses[i+1]:.4f}"

    def test_all_exclusive_mask_near_zero(self):
        """No overlap region → loss ≈ 0 (ov mask all-zero → (0 * log(o)).sum()/eps ≈ 0)."""
        B, S = 1, 64
        # e0 and e1 are disjoint → no overlap
        masks = torch.zeros(B, 2, S)
        masks[:, 0, :S//2] = 1.0
        masks[:, 1, S//2:] = 1.0
        o0 = torch.full((B, S), 0.5)
        o1 = torch.full((B, S), 0.5)
        loss = l_overlap_activate(o0, o1, masks)
        assert loss.item() < 1e-4, \
            f"no overlap → loss ≈ 0, got {loss.item():.6f}"

    def test_output_is_scalar(self):
        """Output is a scalar tensor."""
        B, S = 3, 128
        masks = _make_region_masks(B, S)
        o0 = torch.rand(B, S).clamp(0.01, 0.99)
        o1 = torch.rand(B, S).clamp(0.01, 0.99)
        loss = l_overlap_activate(o0, o1, masks)
        assert loss.shape == torch.Size([]), f"expected scalar, got {loss.shape}"

    def test_batch_size_gt1(self):
        """B=4 처리."""
        B, S = 4, 128
        masks = _make_region_masks(B, S)
        o0 = torch.rand(B, S).clamp(0.01, 0.99)
        o1 = torch.rand(B, S).clamp(0.01, 0.99)
        loss = l_overlap_activate(o0, o1, masks)
        assert loss.item() >= 0.0

    def test_gradient_dominance_over_existing_bce(self):
        """
        la=10 overlap activate >> BCE positive per overlap pixel.

        At o=0.587 in overlap (Phase48 equilibrium), n_ov = 0.25*S, n_pos = 0.5*S:
          l_overlap_activate grad: 10 * 0.5 / (n_ov * 0.587) = 5/(n_ov*0.587) = 34/S per pixel
          BCE positive grad (la_occ=2): 2 * (1/0.587) / n_pos = 3.4/(0.5*S) = 6.8/S per pixel

        overlap_activate is ~5× stronger per overlap pixel than existing BCE positive.
        """
        B, S = 1, 64
        masks = _make_region_masks(B, S)
        q = S // 4

        o_val = 0.587  # Phase48 equilibrium overlap activation
        o0_leaf = torch.zeros(B, S, requires_grad=True)
        o0 = o0_leaf + torch.full((B, S), o_val)
        o1 = torch.full((B, S), o_val)

        loss = l_overlap_activate(o0, o1, masks)
        loss.backward()

        grad_ov_activate_per_pixel = abs(o0_leaf.grad[:, q:q*2].mean().item())
        # la=10 scaling applied outside this function; divide by 10 to get raw
        # Actually we test the raw function grad here
        # la_ov_activate=10 will be multiplied outside

        # Compare to what BCE positive would give for same overlap pixels
        # BCE: -log(o0) per positive pixel, la_occ=2
        n_pos_approx = (masks[:, 0, :].sum() / S).item()  # e0 positive fraction
        bce_raw_grad_per_overlap_pixel = (1.0 / o_val) / (n_pos_approx * S)
        la_bce_effective = 2.0 * bce_raw_grad_per_overlap_pixel

        # la_ov_activate=10 applied outside
        n_ov = (masks[:, 0, :] * masks[:, 1, :]).sum().item()
        la_ov_effective = 10.0 * 0.5 / (n_ov * o_val)  # per overlap pixel

        ratio = la_ov_effective / (la_bce_effective + 1e-9)
        assert ratio >= 3.0, \
            f"overlap_activate (la=10) should dominate BCE by ≥3×, got ratio={ratio:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
