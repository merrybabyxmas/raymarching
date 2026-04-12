"""
test_phase46_occ_contrastive.py — l_occ_contrastive 검증

검증 항목:
  1. 이미 satisfied (o0 >> o1 in e0-excl) → loss = 0
  2. collapsed (o0=o1=0.7 in e0-excl) → loss > 0 (penalty active)
  3. 방향성: e0-excl에서 gradient가 o1을 낮추고 o0을 올림
  4. e1-excl에서 방향 반전 확인
  5. overlap/bg 구역은 loss에 기여하지 않음
  6. margin 변화가 loss에 단조 영향
  7. 대칭: l_occ_cont(o0, o1, m) ≈ l_occ_cont(o1, o0, swapped_m)
  8. batch size > 1 작동
  9. all-bg (mask=0) → loss = 0 (exclusive 없음)
 10. gradient가 o0, o1 leaf tensor로 흐름
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase46 import l_occ_contrastive


class TestLOccContrastive:

    def _make_masks(self, B, S, e0_fraction=0.5):
        """Simple binary masks: e0 in first half, e1 in second half (no overlap)."""
        masks = torch.zeros(B, 2, S)
        mid = int(S * e0_fraction)
        masks[:, 0, :mid]  = 1.0   # e0-exclusive
        masks[:, 1, mid:]  = 1.0   # e1-exclusive
        return masks

    def test_satisfied_gives_zero_loss(self):
        """o0 >> o1 in e0-excl, o1 >> o0 in e1-excl → loss = 0."""
        B, S = 2, 64
        masks = self._make_masks(B, S)  # e0: [0:32], e1: [32:64]
        margin = 0.5
        # Satisfied: o0=0.9, o1=0.3 in e0-excl → 0.3 - 0.9 + 0.5 = -0.1 → relu=0
        o0 = torch.full((B, S), 0.3)
        o1 = torch.full((B, S), 0.3)
        o0[:, :32] = 0.9  # e0-excl: o0 high
        o1[:, 32:] = 0.9  # e1-excl: o1 high

        loss = l_occ_contrastive(o0, o1, masks, margin=margin)
        assert loss.item() < 1e-6, f"satisfied → loss should be 0, got {loss.item():.6f}"

    def test_collapsed_gives_positive_loss(self):
        """o0=o1=0.7 everywhere (collapse) → loss > 0 in exclusive regions."""
        B, S = 2, 64
        masks = self._make_masks(B, S)
        o0 = torch.full((B, S), 0.7)
        o1 = torch.full((B, S), 0.7)
        loss = l_occ_contrastive(o0, o1, masks, margin=0.5)
        # In e0-excl: relu(0.7 - 0.7 + 0.5) = 0.5 > 0
        assert loss.item() > 0.0, f"collapse → loss should be positive, got {loss.item():.6f}"

    def test_gradient_direction_e0_excl(self):
        """In e0-excl: gradient pushes o1 DOWN (dL/do1 > 0 → step reduces o1)
           and o0 UP (dL/do0 < 0 → step increases o0 when we minimize loss)."""
        B, S = 1, 32
        masks = torch.zeros(B, 2, S)
        masks[:, 0, :16] = 1.0   # e0-exclusive only
        # collapsed: o0=o1=0.7
        o0_leaf = torch.full((B, S), 0.7, requires_grad=True)
        o1_leaf = torch.full((B, S), 0.7, requires_grad=True)

        loss = l_occ_contrastive(o0_leaf, o1_leaf, masks, margin=0.5)
        loss.backward()

        # dL/do1 > 0 in e0-excl → minimizing loss means reducing o1
        assert o1_leaf.grad is not None
        grad_o1_excl = o1_leaf.grad[:, :16].mean().item()
        assert grad_o1_excl > 0, \
            f"dL/do1 in e0-excl should be > 0 (minimize → o1↓), got {grad_o1_excl:.4f}"

        # dL/do0 < 0 in e0-excl → minimizing loss means increasing o0
        assert o0_leaf.grad is not None
        grad_o0_excl = o0_leaf.grad[:, :16].mean().item()
        assert grad_o0_excl < 0, \
            f"dL/do0 in e0-excl should be < 0 (minimize → o0↑), got {grad_o0_excl:.4f}"

    def test_gradient_direction_e1_excl(self):
        """In e1-excl: gradient pushes o0 DOWN and o1 UP (symmetric to e0)."""
        B, S = 1, 32
        masks = torch.zeros(B, 2, S)
        masks[:, 1, 16:] = 1.0   # e1-exclusive only
        o0_leaf = torch.full((B, S), 0.7, requires_grad=True)
        o1_leaf = torch.full((B, S), 0.7, requires_grad=True)

        loss = l_occ_contrastive(o0_leaf, o1_leaf, masks, margin=0.5)
        loss.backward()

        # dL/do0 > 0 in e1-excl → minimizing loss means reducing o0
        grad_o0_excl = o0_leaf.grad[:, 16:].mean().item()
        assert grad_o0_excl > 0, \
            f"dL/do0 in e1-excl should be > 0 (minimize → o0↓), got {grad_o0_excl:.4f}"

        grad_o1_excl = o1_leaf.grad[:, 16:].mean().item()
        assert grad_o1_excl < 0, \
            f"dL/do1 in e1-excl should be < 0 (minimize → o1↑), got {grad_o1_excl:.4f}"

    def test_overlap_bg_no_contribution(self):
        """Overlap and bg pixels don't contribute to loss."""
        B, S = 1, 64
        # All overlap (both masks = 1) — no exclusive
        masks_all_ov = torch.ones(B, 2, S)
        o0 = torch.full((B, S), 0.7)
        o1 = torch.full((B, S), 0.7)
        loss_ov = l_occ_contrastive(o0, o1, masks_all_ov, margin=0.5)
        assert loss_ov.item() == 0.0, \
            f"all-overlap → no exclusive → loss should be 0, got {loss_ov.item():.6f}"

        # All bg (both masks = 0)
        masks_all_bg = torch.zeros(B, 2, S)
        loss_bg = l_occ_contrastive(o0, o1, masks_all_bg, margin=0.5)
        assert loss_bg.item() == 0.0, \
            f"all-bg → no exclusive → loss should be 0, got {loss_bg.item():.6f}"

    def test_larger_margin_larger_loss(self):
        """Larger margin → same collapsed state → larger loss."""
        B, S = 2, 64
        masks = self._make_masks(B, S)
        o0 = torch.full((B, S), 0.7)
        o1 = torch.full((B, S), 0.7)
        loss_small = l_occ_contrastive(o0, o1, masks, margin=0.3)
        loss_large = l_occ_contrastive(o0, o1, masks, margin=0.7)
        assert loss_large.item() > loss_small.item(), \
            f"larger margin should give larger loss: {loss_small.item():.4f} vs {loss_large.item():.4f}"

    def test_symmetry(self):
        """l_occ_cont(o0, o1, m) ≈ l_occ_cont(o1, o0, swapped_m)."""
        B, S = 2, 64
        masks = self._make_masks(B, S)
        o0 = torch.rand(B, S).clamp(0.1, 0.9)
        o1 = torch.rand(B, S).clamp(0.1, 0.9)

        l_01 = l_occ_contrastive(o0, o1, masks, margin=0.5)
        masks_swap = masks[:, [1, 0], :]
        l_10 = l_occ_contrastive(o1, o0, masks_swap, margin=0.5)
        assert abs(l_01.item() - l_10.item()) < 1e-5, \
            f"symmetry failed: {l_01.item():.6f} vs {l_10.item():.6f}"

    def test_batch_size_gt1(self):
        """B>1 batch 처리."""
        B, S = 4, 64
        masks = self._make_masks(B, S)
        o0 = torch.rand(B, S)
        o1 = torch.rand(B, S)
        loss = l_occ_contrastive(o0, o1, masks, margin=0.5)
        assert loss.shape == torch.Size([]), "loss should be scalar"
        assert loss.item() >= 0.0

    def test_all_bg_mask_zero_loss(self):
        """All-zero masks → no exclusive region → loss = 0."""
        B, S = 2, 64
        masks = torch.zeros(B, 2, S)
        o0 = torch.full((B, S), 0.7)
        o1 = torch.full((B, S), 0.7)
        loss = l_occ_contrastive(o0, o1, masks, margin=0.5)
        assert loss.item() == 0.0, f"all-bg → loss=0, got {loss.item():.6f}"

    def test_gradient_flows_to_leaves(self):
        """Gradient flows through to o0_leaf and o1_leaf."""
        B, S = 2, 32
        masks = self._make_masks(B, S)
        o0_leaf = torch.rand(B, S, requires_grad=True)
        o1_leaf = torch.rand(B, S, requires_grad=True)

        loss = l_occ_contrastive(o0_leaf, o1_leaf, masks, margin=0.5)
        loss.backward()

        assert o0_leaf.grad is not None and o0_leaf.grad.abs().max().item() > 0
        assert o1_leaf.grad is not None and o1_leaf.grad.abs().max().item() > 0

    def test_phase45_collapse_penalty(self):
        """Reproduce Phase45 collapse: o0=0.92, o1=0.69 in e0-excl → penalty=0.27."""
        B, S = 1, 64
        masks = torch.zeros(B, 2, S)
        masks[:, 0, :32] = 1.0   # e0-exclusive
        o0 = torch.full((B, S), 0.92)
        o1 = torch.full((B, S), 0.69)
        # In e0-excl: relu(0.69 - 0.92 + 0.5) = relu(0.27) = 0.27
        loss = l_occ_contrastive(o0, o1, masks, margin=0.5)
        expected = 0.27
        assert abs(loss.item() - expected) < 0.01, \
            f"Phase45 collapse penalty: expected ≈{expected:.3f}, got {loss.item():.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
