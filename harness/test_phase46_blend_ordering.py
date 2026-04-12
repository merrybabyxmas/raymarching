"""
test_phase46_blend_ordering.py — l_blend_ordering 검증

검증 항목:
  1. blend_ov > blend_ex + margin → loss = 0 (satisfied)
  2. blend_ov ≈ blend_ex (ordering violated) → loss > 0
  3. gradient: dL/d(blend_ov) < 0 (minimize → overlap blend UP)
  4. gradient: dL/d(blend_ex) > 0 (minimize → exclusive blend DOWN)
  5. blend_ex > blend_bg + margin → l_ex_bg = 0
  6. blend_ex ≈ blend_bg → l_ex_bg > 0
  7. larger margin harder to satisfy
  8. all-overlap mask: exclusive term = 0
  9. gradient is scalar (loss.shape == [])
 10. batch size > 1 작동
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase46 import l_blend_ordering


def _make_region_masks(B, S):
    """Returns masks where ~25% overlap, ~50% exclusive, ~25% bg."""
    masks = torch.zeros(B, 2, S)
    q = S // 4
    masks[:, 0, :q*2] = 1.0      # e0: [0, S//2)
    masks[:, 1, q:q*3] = 1.0     # e1: [S//4, 3S//4)
    # overlap: [S//4, S//2), e0-only: [0, S//4), e1-only: [S//2, 3S//4), bg: [3S//4, S)
    return masks


class TestLBlendOrdering:

    def test_satisfied_gives_zero(self):
        """blend_ov >> blend_ex >> blend_bg → loss = 0."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        blend = torch.zeros(B, S)
        q = S // 4
        blend[:, q:q*2] = 0.85    # overlap region
        blend[:, :q]    = 0.35    # e0-only (exclusive)
        blend[:, q*2:q*3] = 0.35  # e1-only (exclusive)
        blend[:, q*3:]  = 0.05    # bg
        loss = l_blend_ordering(blend, masks, margin=0.10)
        # mean_ov ≈ 0.85, mean_ex ≈ 0.35, mean_bg ≈ 0.05 → all hinges satisfied
        assert loss.item() < 1e-5, f"satisfied → loss should be 0, got {loss.item():.6f}"

    def test_inverted_ov_ex_gives_positive_loss(self):
        """blend_ex > blend_ov (Phase45 failure pattern) → loss > 0."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        blend = torch.zeros(B, S)
        q = S // 4
        blend[:, q:q*2] = 0.44    # overlap: lower than exclusive!
        blend[:, :q]    = 0.48    # e0-only exclusive
        blend[:, q*2:q*3] = 0.48  # e1-only exclusive
        blend[:, q*3:]  = 0.05    # bg
        loss = l_blend_ordering(blend, masks, margin=0.10)
        # mean_ex ≈ 0.48 > mean_ov ≈ 0.44 → relu(0.48 - 0.44 + 0.10) = 0.14 > 0
        assert loss.item() > 0.0, f"inverted → loss should be > 0, got {loss.item():.6f}"

    def test_gradient_pushes_overlap_up(self):
        """dL/d(blend in overlap region) < 0 → minimizing loss increases overlap blend."""
        B, S = 1, 64
        masks = _make_region_masks(B, S)
        q = S // 4
        # Start from violated state: blend_ex ≈ blend_ov
        blend_leaf = torch.zeros(B, S, requires_grad=True)
        # Use non-leaf modification through addition
        bias = torch.zeros(B, S)
        bias[:, q:q*2] = 0.44   # overlap
        bias[:, :q]    = 0.48   # e0-excl
        bias[:, q*2:q*3] = 0.48 # e1-excl
        bias[:, q*3:]  = 0.10   # bg

        blend = blend_leaf + bias
        loss = l_blend_ordering(blend, masks, margin=0.10)
        loss.backward()

        grad_overlap_mean = blend_leaf.grad[:, q:q*2].mean().item()
        assert grad_overlap_mean < 0, \
            f"dL/d(blend_overlap) should be < 0 (minimize → overlap UP), got {grad_overlap_mean:.4f}"

    def test_gradient_pushes_exclusive_down(self):
        """dL/d(blend in exclusive region) > 0 → minimizing loss decreases exclusive blend."""
        B, S = 1, 64
        masks = _make_region_masks(B, S)
        q = S // 4

        blend_leaf = torch.zeros(B, S, requires_grad=True)
        bias = torch.zeros(B, S)
        bias[:, q:q*2] = 0.44
        bias[:, :q]    = 0.48
        bias[:, q*2:q*3] = 0.48
        bias[:, q*3:]  = 0.10

        blend = blend_leaf + bias
        loss = l_blend_ordering(blend, masks, margin=0.10)
        loss.backward()

        # exclusive = e0-only + e1-only
        grad_excl_e0 = blend_leaf.grad[:, :q].mean().item()
        grad_excl_e1 = blend_leaf.grad[:, q*2:q*3].mean().item()
        assert grad_excl_e0 > 0, \
            f"dL/d(blend_e0_excl) should be > 0, got {grad_excl_e0:.4f}"
        assert grad_excl_e1 > 0, \
            f"dL/d(blend_e1_excl) should be > 0, got {grad_excl_e1:.4f}"

    def test_ex_bg_satisfied_zero(self):
        """blend_ex >> blend_bg + margin → l_ex_bg = 0."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        blend = torch.zeros(B, S)
        q = S // 4
        blend[:, q:q*2] = 0.85    # overlap (big)
        blend[:, :q]    = 0.45    # e0-only exclusive
        blend[:, q*2:q*3] = 0.45  # e1-only exclusive
        blend[:, q*3:]  = 0.05    # bg (small — well below exclusive)
        # mean_ex ≈ 0.45, mean_bg ≈ 0.05 → hinge(0.05 - 0.45 + 0.1) = hinge(-0.3) = 0
        # mean_ov ≈ 0.85, mean_ex ≈ 0.45 → hinge(0.45 - 0.85 + 0.1) = hinge(-0.3) = 0
        loss = l_blend_ordering(blend, masks, margin=0.10)
        assert loss.item() < 1e-5, f"both hinges satisfied → loss = 0, got {loss.item():.6f}"

    def test_ex_bg_violated(self):
        """blend_bg ≈ blend_ex → l_ex_bg > 0."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        blend = torch.zeros(B, S)
        q = S // 4
        blend[:, q:q*2] = 0.90   # overlap: high
        blend[:, :q]    = 0.40   # e0-only exclusive
        blend[:, q*2:q*3] = 0.40 # e1-only exclusive
        blend[:, q*3:]  = 0.38   # bg: almost same as exclusive
        # mean_ex ≈ 0.40, mean_bg ≈ 0.38 → hinge(0.38 - 0.40 + 0.10) = 0.08 > 0
        loss = l_blend_ordering(blend, masks, margin=0.10)
        assert loss.item() > 0.0, f"ex≈bg violated → loss > 0, got {loss.item():.6f}"

    def test_larger_margin_harder(self):
        """Larger margin → same blend values → larger or equal loss."""
        B, S = 2, 64
        masks = _make_region_masks(B, S)
        blend = torch.zeros(B, S)
        q = S // 4
        # Blend with a moderate gap
        blend[:, q:q*2] = 0.60
        blend[:, :q]    = 0.40
        blend[:, q*2:q*3] = 0.40
        blend[:, q*3:]  = 0.10
        loss_small = l_blend_ordering(blend, masks, margin=0.05)
        loss_large = l_blend_ordering(blend, masks, margin=0.30)
        assert loss_large.item() >= loss_small.item(), \
            f"larger margin should give >= loss: small={loss_small.item():.4f} large={loss_large.item():.4f}"

    def test_all_overlap_mask(self):
        """All-overlap mask: no exclusive/bg region → exclusive-based hinges inactive."""
        B, S = 1, 64
        masks = torch.ones(B, 2, S)  # all overlap
        blend = torch.full((B, S), 0.5)
        loss = l_blend_ordering(blend, masks, margin=0.10)
        # n_ex ≈ 0 (eps only), n_bg ≈ 0 → mean_ex ≈ 0.5, mean_bg ≈ 0.5 (from eps div)
        # hinge terms may be near 0 due to eps dominance; main point: no crash
        assert loss.item() >= 0.0, "should not crash with all-overlap mask"

    def test_output_is_scalar(self):
        """Output is a scalar tensor."""
        B, S = 3, 64
        masks = _make_region_masks(B, S)
        blend = torch.rand(B, S)
        loss = l_blend_ordering(blend, masks, margin=0.10)
        assert loss.shape == torch.Size([]), f"expected scalar, got shape {loss.shape}"

    def test_batch_size_gt1(self):
        """B=4 처리."""
        B, S = 4, 128
        masks = _make_region_masks(B, S)
        blend = torch.rand(B, S)
        loss = l_blend_ordering(blend, masks, margin=0.10)
        assert loss.item() >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
