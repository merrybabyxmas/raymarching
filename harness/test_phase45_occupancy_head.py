"""
test_phase45_occupancy_head.py — OccupancyHead + l_occupancy 검증

검증 항목 (OccupancyHead):
  1. 초기화 후 출력 ≈ 0.5 everywhere (zero-init 보장)
  2. 출력 shape: (B, S)  (squeeze(-1) 확인)
  3. 출력 ∈ [0, 1]  (sigmoid 범위)
  4. 마지막 Linear weight=0이면 output=sigmoid(0)=0.5
  5. gradient가 마지막 Linear weight로 흐름
  6. BCE loss로 0→1 방향 학습 가능 (gradient 방향 확인)
  7. 다양한 hidden 크기 작동

검증 항목 (l_occupancy):
  8. perfect prediction (o=mask) → loss 최소 (접근 0)
  9. worst prediction (o=1-mask) → loss 최대
 10. 대칭: l_occ(o0,o1,m) = l_occ(o1,o0, m_swapped)
 11. gradient가 o0_for_loss로 흐름
 12. mask 전체 0 (bg) → BCE(0.5, 0) 계산
 13. B>1 batch 처리 정상
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase45 import OccupancyHead, l_occupancy


# =============================================================================
# OccupancyHead
# =============================================================================

class TestOccupancyHead:

    def test_zero_init_output_half(self):
        """zero-init 후 output = sigmoid(0) = 0.5 everywhere."""
        head = OccupancyHead(in_dim=64, hidden=64)
        head.eval()
        B, S = 2, 16
        feat = torch.randn(B, S, 64)
        with torch.no_grad():
            out = head(feat)
        assert out.allclose(torch.full_like(out, 0.5), atol=1e-6), \
            f"zero-init: expected 0.5, got max_dev={( out - 0.5).abs().max().item():.2e}"

    def test_output_shape_squeeze(self):
        """출력 shape = (B, S) — squeeze(-1) 확인."""
        head = OccupancyHead(in_dim=32, hidden=32)
        B, S = 3, 64
        feat = torch.randn(B, S, 32)
        out = head(feat)
        assert out.shape == (B, S), f"expected ({B},{S}), got {out.shape}"

    def test_output_in_01_range(self):
        """출력 ∈ [0, 1] — sigmoid 범위."""
        head = OccupancyHead(in_dim=64, hidden=64)
        # Perturb weights so output is not constant
        with torch.no_grad():
            head.net[2].weight.fill_(0.5)
        feat = torch.randn(4, 100, 64)
        out = head(feat)
        assert out.min().item() >= 0.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_zero_last_weight_gives_half(self):
        """마지막 Linear weight=0 → output = sigmoid(0) = 0.5."""
        head = OccupancyHead(in_dim=64, hidden=64)
        # Explicitly set intermediate weights to non-zero, keep last zero
        with torch.no_grad():
            head.net[0].weight.fill_(0.1)
            head.net[2].weight.zero_()
            head.net[2].bias.zero_()
        feat = torch.randn(2, 8, 64)
        with torch.no_grad():
            out = head(feat)
        assert out.allclose(torch.full_like(out, 0.5), atol=1e-5), \
            "zero last weight → output should be 0.5"

    def test_gradient_to_last_layer(self):
        """gradient가 마지막 Linear weight로 흐름."""
        head = OccupancyHead(in_dim=64, hidden=32)
        # Perturb to get non-trivial gradient
        with torch.no_grad():
            head.net[2].weight.fill_(0.01)

        feat = torch.randn(1, 8, 64)
        out = head(feat)
        loss = out.sum()
        loss.backward()
        assert head.net[2].weight.grad is not None
        assert head.net[2].weight.grad.abs().max().item() > 0

    def test_bce_gradient_direction(self):
        """target=1일 때 gradient가 output을 1 방향으로 밀음."""
        head = OccupancyHead(in_dim=32, hidden=32)
        with torch.no_grad():
            head.net[2].weight.fill_(0.0)  # start at 0.5

        feat = torch.randn(1, 4, 32, requires_grad=False)
        # Simulate learning step toward target=1
        opt = torch.optim.SGD(head.parameters(), lr=0.1)
        target = torch.ones(1, 4)

        out_before = head(feat.detach()).mean().item()
        for _ in range(5):
            opt.zero_grad()
            out = head(feat.detach())
            loss = F.binary_cross_entropy(out.clamp(1e-6, 1 - 1e-6), target)
            loss.backward()
            opt.step()
        out_after = head(feat.detach()).mean().item()

        assert out_after > out_before, \
            f"BCE with target=1 should increase output: {out_before:.4f} → {out_after:.4f}"

    def test_various_hidden_sizes(self):
        """다양한 hidden 크기 작동."""
        for h in (16, 32, 64, 128):
            head = OccupancyHead(in_dim=64, hidden=h)
            feat = torch.randn(2, 10, 64)
            out = head(feat)
            assert out.shape == (2, 10), f"hidden={h}: expected (2,10), got {out.shape}"


# =============================================================================
# l_occupancy
# =============================================================================

class TestLOccupancy:

    def test_perfect_prediction_low_loss(self):
        """o = mask → loss 최소."""
        B, S = 2, 64
        # mask 0/1
        masks = torch.zeros(B, 2, S)
        masks[0, 0, :32] = 1.0
        masks[1, 1, 16:48] = 1.0

        # Perfect predictions (close to mask, avoid 0/1 exactly for BCE)
        eps = 1e-4
        o0 = masks[:, 0, :].clamp(eps, 1 - eps)
        o1 = masks[:, 1, :].clamp(eps, 1 - eps)
        loss = l_occupancy(o0, o1, masks)
        # Should be very small
        assert loss.item() < 0.01, f"perfect prediction loss too high: {loss.item():.5f}"

    def test_worst_prediction_high_loss(self):
        """o = 1-mask → loss 최대."""
        B, S = 2, 64
        masks = torch.zeros(B, 2, S)
        masks[:, 0, :32] = 1.0
        masks[:, 1, 32:] = 1.0

        eps = 1e-4
        o0 = (1 - masks[:, 0, :]).clamp(eps, 1 - eps)
        o1 = (1 - masks[:, 1, :]).clamp(eps, 1 - eps)
        loss_worst = l_occupancy(o0, o1, masks)

        o0_good = masks[:, 0, :].clamp(eps, 1 - eps)
        o1_good = masks[:, 1, :].clamp(eps, 1 - eps)
        loss_best = l_occupancy(o0_good, o1_good, masks)

        assert loss_worst.item() > loss_best.item(), \
            f"worst ({loss_worst.item():.4f}) should > best ({loss_best.item():.4f})"

    def test_symmetry(self):
        """l_occ(o0, o1, m) = l_occ(o1, o0, m_swapped)."""
        B, S = 2, 32
        masks = torch.zeros(B, 2, S)
        masks[:, 0, :16] = 1.0
        masks[:, 1, 16:] = 1.0
        o0 = torch.rand(B, S).clamp(1e-4, 1 - 1e-4)
        o1 = torch.rand(B, S).clamp(1e-4, 1 - 1e-4)

        l_01 = l_occupancy(o0, o1, masks)
        # swap o0↔o1 and mask channels
        masks_swapped = masks[:, [1, 0], :]
        l_10 = l_occupancy(o1, o0, masks_swapped)
        assert abs(l_01.item() - l_10.item()) < 1e-5, \
            f"l_occ should be symmetric: {l_01.item():.6f} vs {l_10.item():.6f}"

    def test_gradient_flows_to_o0(self):
        """gradient가 o0_for_loss로 흐름."""
        B, S = 1, 16
        masks = torch.zeros(B, 2, S)
        masks[0, 0, :8] = 1.0

        # Create leaf tensors and retain_grad on the non-leaf clamped versions
        o0_leaf = torch.rand(B, S, requires_grad=True)
        o1_leaf = torch.rand(B, S, requires_grad=True)
        o0 = o0_leaf.clamp(1e-4, 1 - 1e-4)
        o1 = o1_leaf.clamp(1e-4, 1 - 1e-4)
        o0.retain_grad()
        o1.retain_grad()

        loss = l_occupancy(o0, o1, masks)
        loss.backward()
        assert o0_leaf.grad is not None and o0_leaf.grad.abs().max().item() > 0, \
            "gradient should flow to o0_for_loss"
        assert o1_leaf.grad is not None and o1_leaf.grad.abs().max().item() > 0, \
            "gradient should flow to o1_for_loss"

    def test_all_bg_mask(self):
        """mask 전체 0 (bg) → BCE(0.5, 0) = log(2) × 2."""
        B, S = 1, 8
        masks = torch.zeros(B, 2, S)
        o0 = torch.full((B, S), 0.5)
        o1 = torch.full((B, S), 0.5)
        loss = l_occupancy(o0, o1, masks)
        expected = 2.0 * (-torch.log(torch.tensor(0.5))).item()
        assert abs(loss.item() - expected) < 1e-4, \
            f"all-bg BCE: expected {expected:.5f}, got {loss.item():.5f}"

    def test_batch_size_gt1(self):
        """B>1 batch 처리 정상."""
        B, S = 4, 64
        masks = torch.zeros(B, 2, S)
        masks[:, 0, :32] = 1.0
        o0 = torch.rand(B, S).clamp(1e-4, 1 - 1e-4)
        o1 = torch.rand(B, S).clamp(1e-4, 1 - 1e-4)
        loss = l_occupancy(o0, o1, masks)
        assert loss.shape == torch.Size([]), "loss should be scalar"
        assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
