"""
Phase 41 harness: class-balanced l_sigma_spatial + l_visible_weights bg suppression.

Phase 40 root-cause fix:
  l_sigma_spatial(alpha0, alpha1, masks) — 이전: uniform F.mse_loss → entity 4% bg 96%
  → gradient 대부분이 "alpha→0" → IoU 0.082에 고착.

  Phase 41: class-balanced MSE — entity/bg total gradient 균형.
  entity pixel weight=1, bg pixel weight = n_entity/n_bg
  → alpha→0 shortcut 불가.

Tests
-----
27 tests covering:
  1. class-balanced loss correctness (7 tests)
  2. background suppression in l_visible_weights (6 tests)
  3. gradient direction verification (6 tests)
  4. integration with Phase40Processor pipeline (5 tests)
  5. edge cases (3 tests)
"""
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot import (
    l_sigma_spatial,
    l_visible_weights,
    build_visible_targets,
    compute_visible_iou,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_sparse_mask(B: int, S: int, n_entity: int, seed: int = 0):
    """Sparse entity mask: ~n_entity/S coverage per frame, per entity."""
    rng = np.random.RandomState(seed)
    mask = np.zeros((B, 2, S), dtype=bool)
    for b in range(B):
        pos0 = rng.choice(S, size=n_entity, replace=False)
        pos1 = rng.choice(S, size=n_entity, replace=False)
        mask[b, 0, pos0] = True
        mask[b, 1, pos1] = True
    return torch.from_numpy(mask.astype(np.float32))


# =============================================================================
# 1. Class-balanced loss correctness
# =============================================================================

class TestSigmaBalance:

    def test_loss_is_finite(self):
        """l_sigma_spatial는 항상 유한값."""
        B, S = 2, 256
        alpha0 = torch.rand(B, S)
        alpha1 = torch.rand(B, S)
        masks  = _make_sparse_mask(B, S, n_entity=10)
        loss   = l_sigma_spatial(alpha0, alpha1, masks)
        assert torch.isfinite(loss), f"Non-finite loss: {loss}"

    def test_loss_is_nonneg(self):
        """MSE 기반이므로 항상 ≥ 0."""
        B, S = 2, 256
        alpha0 = torch.rand(B, S)
        alpha1 = torch.rand(B, S)
        masks  = _make_sparse_mask(B, S, n_entity=10)
        loss   = l_sigma_spatial(alpha0, alpha1, masks)
        assert loss.item() >= 0.0, f"Negative loss: {loss.item()}"

    def test_perfect_prediction_gives_zero_loss(self):
        """GT mask가 alpha0이면 loss = 0."""
        B, S = 1, 256
        masks  = _make_sparse_mask(B, S, n_entity=15)
        alpha0 = masks[:, 0, :].clone()   # perfect prediction
        alpha1 = masks[:, 1, :].clone()
        loss   = l_sigma_spatial(alpha0, alpha1, masks)
        assert loss.item() < 1e-5, f"Perfect prediction should give 0, got {loss.item()}"

    def test_all_zero_alpha_is_not_optimal(self):
        """
        class-balanced fix: alpha=0 everywhere ≠ optimal.
        (Phase 40 bug: uniform MSE → alpha=0 globally minimized loss)
        """
        B, S = 2, 256
        n_entity = 10
        masks  = _make_sparse_mask(B, S, n_entity=n_entity)

        alpha_zero = torch.zeros(B, S)
        loss_zero  = l_sigma_spatial(alpha_zero, alpha_zero, masks)

        alpha_perfect = masks[:, 0, :].clone()
        loss_perfect  = l_sigma_spatial(alpha_perfect, alpha_perfect, masks)

        # alpha=0 이 optimal이면 안 됨 — perfect가 훨씬 낮아야 함
        assert loss_perfect.item() < loss_zero.item() * 0.5, (
            f"Perfect (loss={loss_perfect.item():.6f}) should be << zero-alpha "
            f"(loss={loss_zero.item():.6f}). Class balance not working.")

    def test_entity_gradient_direction(self):
        """Entity 픽셀에서 alpha를 1로 올리는 방향으로 gradient."""
        B, S = 1, 256
        masks  = _make_sparse_mask(B, S, n_entity=10)
        alpha0 = torch.full((B, S), 0.3, requires_grad=True)
        alpha1 = torch.full((B, S), 0.3, requires_grad=True)

        loss = l_sigma_spatial(alpha0, alpha1, masks)
        loss.backward()

        # Entity0 pixels: alpha should go up → gradient < 0
        entity0_mask = masks[:, 0, :].bool()
        grad_entity  = alpha0.grad[entity0_mask].mean().item()
        assert grad_entity < 0.0, (
            f"Entity0 gradient should be negative (push alpha up), got {grad_entity:.4f}")

        # Background: alpha should go down → gradient > 0
        bg_mask = ~(masks[:, 0, :].bool() | masks[:, 1, :].bool())
        if bg_mask.any():
            grad_bg = alpha0.grad[bg_mask].mean().item()
            assert grad_bg > 0.0, (
                f"Background gradient should be positive (push alpha down), got {grad_bg:.4f}")

    def test_gradient_balance_entity_bg(self):
        """
        class-balanced: entity total gradient ≈ background total gradient.
        (Phase 40 bug: bg gradient ≈ 63× entity gradient → alpha→0 dominates)
        """
        B, S = 1, 256
        n_entity = 10
        masks  = _make_sparse_mask(B, S, n_entity=n_entity)
        alpha0 = torch.full((B, S), 0.5, requires_grad=True)
        alpha1 = torch.full((B, S), 0.5, requires_grad=True)

        loss = l_sigma_spatial(alpha0, alpha1, masks)
        loss.backward()

        entity_mask = masks[:, 0, :].bool()
        bg_mask     = ~entity_mask

        grad_entity_total = alpha0.grad[entity_mask].abs().sum().item()
        grad_bg_total     = alpha0.grad[bg_mask].abs().sum().item()

        ratio = grad_bg_total / (grad_entity_total + 1e-8)
        # Should be close to 1.0 (balanced), not 63 (imbalanced)
        assert ratio < 5.0, (
            f"Gradient imbalance too large: bg/entity = {ratio:.1f} "
            f"(expected ≈ 1.0 for balanced, got {ratio:.1f}). Fix may not work.")

    def test_loss_decreases_with_better_prediction(self):
        """더 나은 예측 → 더 낮은 loss."""
        B, S = 2, 256
        masks = _make_sparse_mask(B, S, n_entity=12)

        alpha_bad  = torch.full((B, S), 0.5)   # uniform, ignores mask
        alpha_good = masks[:, 0, :] * 0.9 + (1 - masks[:, 0, :]) * 0.1  # near-correct

        loss_bad  = l_sigma_spatial(alpha_bad, alpha_bad, masks).item()
        loss_good = l_sigma_spatial(alpha_good, alpha_good, masks).item()

        assert loss_good < loss_bad, (
            f"Better prediction should give lower loss: good={loss_good:.4f} bad={loss_bad:.4f}")


# =============================================================================
# 2. Background suppression in l_visible_weights
# =============================================================================

class TestVisibleWeightsBg:

    def test_bg_suppression_penalizes_high_w0_in_background(self):
        """Background에서 w0이 높으면 penalty."""
        B, S = 1, 256
        masks = _make_sparse_mask(B, S, n_entity=10)
        depth_orders = [(0, 1)] * B

        # w0 = 0.5 everywhere (bad: high in background)
        w0_bad  = torch.full((B, S), 0.5)
        w1_bad  = torch.zeros(B, S)

        # w0 ≈ 0 in background, correct in entity regions
        w0_good = masks[:, 0, :].float() * 0.9
        w1_good = torch.zeros(B, S)

        loss_bad  = l_visible_weights(w0_bad, w1_bad, masks, depth_orders).item()
        loss_good = l_visible_weights(w0_good, w1_good, masks, depth_orders).item()

        assert loss_good < loss_bad, (
            f"Good w0 (entity=0.9, bg=0) should have lower loss than "
            f"bad w0 (everywhere=0.5): good={loss_good:.4f} bad={loss_bad:.4f}")

    def test_bg_gradient_pushes_w0_down(self):
        """Background에서 w0이 양수면 gradient가 아래로 향함."""
        B, S = 1, 256
        masks = _make_sparse_mask(B, S, n_entity=10)
        depth_orders = [(0, 1)] * B

        w0 = torch.full((B, S), 0.3, requires_grad=True)
        w1 = torch.full((B, S), 0.3, requires_grad=True)

        loss = l_visible_weights(w0, w1, masks, depth_orders)
        loss.backward()

        bg_mask = ~(masks[:, 0, :].bool() | masks[:, 1, :].bool())
        if bg_mask.any():
            grad_bg = w0.grad[bg_mask].mean().item()
            assert grad_bg > 0.0, (
                f"Background gradient should be positive (push w0 down), got {grad_bg:.4f}")

    def test_loss_finite_with_empty_entity_mask(self):
        """Empty entity mask에서도 유한값."""
        B, S = 1, 256
        masks = torch.zeros(B, 2, S)  # no entities
        depth_orders = [(0, 1)] * B
        w0 = torch.rand(B, S)
        w1 = torch.rand(B, S)
        loss = l_visible_weights(w0, w1, masks, depth_orders)
        assert torch.isfinite(loss)

    def test_entity_loss_unchanged(self):
        """Entity 픽셀에 대한 loss는 bg suppression 추가 후에도 동일하게 작동."""
        B, S = 1, 256
        masks = _make_sparse_mask(B, S, n_entity=15)
        depth_orders = [(0, 1)] * B

        # Perfect entity prediction, zero background
        w0_tgt, w1_tgt = build_visible_targets(masks, depth_orders)
        w0 = w0_tgt.clone()
        w1 = w1_tgt.clone()  # zero in bg (good)

        loss = l_visible_weights(w0, w1, masks, depth_orders)
        # Entity part = 0 (perfect), bg part = 0 (w0=0 in bg → bg_loss=0)
        assert loss.item() < 0.01, (
            f"Perfect prediction with zero bg should give ~0 loss, got {loss.item():.4f}")

    def test_bg_weight_reduces_with_bg_weight_arg(self):
        """bg_weight=0 → bg suppression 없음."""
        B, S = 1, 256
        masks = _make_sparse_mask(B, S, n_entity=10)
        depth_orders = [(0, 1)] * B
        w0 = torch.full((B, S), 0.5)
        w1 = torch.zeros(B, S)

        loss_with_bg    = l_visible_weights(w0, w1, masks, depth_orders, bg_weight=0.2)
        loss_without_bg = l_visible_weights(w0, w1, masks, depth_orders, bg_weight=0.0)

        assert loss_with_bg.item() > loss_without_bg.item(), (
            f"bg_weight=0.2 should give higher loss when bg is bad: "
            f"with_bg={loss_with_bg:.4f} without_bg={loss_without_bg:.4f}")

    def test_loss_is_nonneg(self):
        B, S = 2, 256
        masks = _make_sparse_mask(B, S, n_entity=12)
        depth_orders = [(0, 1)] * B
        w0 = torch.rand(B, S)
        w1 = torch.rand(B, S)
        loss = l_visible_weights(w0, w1, masks, depth_orders)
        assert loss.item() >= 0.0


# =============================================================================
# 3. Gradient direction verification (key correctness tests)
# =============================================================================

class TestGradientDirections:

    def test_alpha_in_entity_region_goes_up(self):
        """l_sigma_spatial: entity 픽셀 alpha → 1 방향으로 gradient."""
        B, S = 2, 256
        masks = _make_sparse_mask(B, S, n_entity=8)
        alpha0 = torch.full((B, S), 0.1, requires_grad=True)
        alpha1 = torch.full((B, S), 0.1, requires_grad=True)

        loss = l_sigma_spatial(alpha0, alpha1, masks)
        loss.backward()

        ent0_mask = masks[:, 0, :].bool()
        # Gradient < 0 means optimization pushes alpha upward toward 1
        assert alpha0.grad[ent0_mask].mean().item() < 0.0

    def test_alpha_in_bg_region_goes_down(self):
        """l_sigma_spatial: background alpha → 0 방향으로 gradient."""
        B, S = 2, 256
        masks = _make_sparse_mask(B, S, n_entity=8)
        alpha0 = torch.full((B, S), 0.8, requires_grad=True)
        alpha1 = torch.full((B, S), 0.8, requires_grad=True)

        loss = l_sigma_spatial(alpha0, alpha1, masks)
        loss.backward()

        bg_mask = ~(masks[:, 0, :].bool() | masks[:, 1, :].bool())
        if bg_mask.any():
            assert alpha0.grad[bg_mask].mean().item() > 0.0

    def test_w0_in_entity0_exclusive_goes_up(self):
        """l_visible_weights: entity0 exclusive에서 w0 → 1."""
        B, S = 1, 256
        masks = _make_sparse_mask(B, S, n_entity=10, seed=1)
        depth_orders = [(0, 1)] * B
        w0 = torch.full((B, S), 0.3, requires_grad=True)
        w1 = torch.full((B, S), 0.3, requires_grad=True)

        loss = l_visible_weights(w0, w1, masks, depth_orders)
        loss.backward()

        excl0 = (masks[:, 0, :] * (1 - masks[:, 1, :])).bool()
        if excl0.any():
            # w0 should go toward 1 → gradient < 0
            assert w0.grad[excl0].mean().item() < 0.0

    def test_w1_in_entity0_exclusive_goes_down(self):
        """l_visible_weights: entity0 exclusive에서 w1 → 0 (wrong slot)."""
        B, S = 1, 256
        masks = _make_sparse_mask(B, S, n_entity=10, seed=2)
        depth_orders = [(0, 1)] * B
        w0 = torch.full((B, S), 0.5, requires_grad=True)
        w1 = torch.full((B, S), 0.5, requires_grad=True)

        loss = l_visible_weights(w0, w1, masks, depth_orders)
        loss.backward()

        excl0 = (masks[:, 0, :] * (1 - masks[:, 1, :])).bool()
        if excl0.any():
            # w1 in e0-exclusive should go to 0 → gradient > 0
            assert w1.grad[excl0].mean().item() > 0.0

    def test_combined_loss_gradient_is_nonzero(self):
        """vis + sigma 합산 loss에서 VCA params에 gradient 전달됨."""
        B, S = 1, 256
        masks = _make_sparse_mask(B, S, n_entity=10)
        depth_orders = [(0, 1)] * B

        # Simulate alpha and w0 with requires_grad (as from VCA)
        sigma_param = torch.nn.Parameter(torch.randn(B, S, 2, 2))
        alpha0 = sigma_param[:, :, 0, :].max(dim=-1).values.sigmoid()
        alpha1 = sigma_param[:, :, 1, :].max(dim=-1).values.sigmoid()
        w0 = alpha0.detach()
        w1 = alpha1.detach()

        l_sig = l_sigma_spatial(alpha0, alpha1, masks)
        l_vis = l_visible_weights(w0, w1, masks, depth_orders)
        loss  = l_sig + l_vis
        loss.backward()

        assert sigma_param.grad is not None
        assert sigma_param.grad.abs().sum().item() > 0

    def test_sigma_loss_not_zero_with_nonzero_prediction(self):
        """alpha≠GT이면 loss > 0."""
        B, S = 1, 256
        masks = _make_sparse_mask(B, S, n_entity=10)
        alpha0 = torch.full((B, S), 0.5)  # Wrong everywhere
        alpha1 = torch.full((B, S), 0.5)
        loss = l_sigma_spatial(alpha0, alpha1, masks)
        assert loss.item() > 1e-4


# =============================================================================
# 4. Integration with Phase40Processor pipeline
# =============================================================================

class TestPhase41Integration:

    @pytest.fixture(scope="class")
    def dummy_proc(self):
        """Mock Phase40Processor-like state for integration tests."""
        from models.entity_slot_phase40 import Phase40Processor
        proc = Phase40Processor(
            query_dim=640,
            vca_layer=None,
            entity_ctx=None,
            slot_blend_init=0.3,
            inner_dim=640,
            adapter_rank=64,
            lora_rank=4,
        )
        return proc

    def test_l_sigma_handles_alpha_from_porter_duff(self):
        """Porter-Duff alpha (0-1 values, sigmoid output) passes through l_sigma_spatial."""
        B, S = 2, 256
        masks = _make_sparse_mask(B, S, n_entity=10)
        # Porter-Duff alpha: always in [0,1]
        alpha0 = torch.sigmoid(torch.randn(B, S))
        alpha1 = torch.sigmoid(torch.randn(B, S))

        loss = l_sigma_spatial(alpha0, alpha1, masks)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_l_visible_weights_handles_porter_duff_w(self):
        """Porter-Duff w0/w1 (can exceed 1 in theory but clamped) works."""
        B, S = 2, 256
        masks = _make_sparse_mask(B, S, n_entity=12)
        depth_orders = [(0, 1)] * B
        w0 = torch.sigmoid(torch.randn(B, S))  # 0-1 range
        w1 = torch.sigmoid(torch.randn(B, S))

        loss = l_visible_weights(w0, w1, masks, depth_orders)
        assert torch.isfinite(loss)

    def test_multi_frame_batch_consistent(self):
        """8-frame batch (BF=8) gradient accumulation consistent."""
        BF, S = 8, 256
        masks = _make_sparse_mask(BF, S, n_entity=10)
        alpha0 = torch.rand(BF, S, requires_grad=True)
        alpha1 = torch.rand(BF, S, requires_grad=True)

        # Simulate frame-by-frame accumulation (as in training loop)
        l_acc = torch.tensor(0.0)
        for fi in range(BF):
            l_acc = l_acc + l_sigma_spatial(
                alpha0[fi:fi+1], alpha1[fi:fi+1], masks[fi:fi+1])
        l_acc = l_acc / BF
        l_acc.backward()

        assert alpha0.grad is not None
        assert torch.isfinite(alpha0.grad).all()

    def test_batch_single_vs_looped_consistent(self):
        """Batch vs per-frame loop gives consistent results."""
        B, S = 4, 256
        masks  = _make_sparse_mask(B, S, n_entity=10)
        alpha0 = torch.rand(B, S)
        alpha1 = torch.rand(B, S)

        # Batch
        loss_batch = l_sigma_spatial(alpha0, alpha1, masks)

        # Per-frame loop
        loss_loop = torch.tensor(0.0)
        for b in range(B):
            loss_loop = loss_loop + l_sigma_spatial(
                alpha0[b:b+1], alpha1[b:b+1], masks[b:b+1])
        loss_loop = loss_loop / B

        # Should be approximately equal (weighted average differs slightly per batch normalization)
        assert abs(loss_batch.item() - loss_loop.item()) < 0.1

    def test_iou_improves_with_balanced_training(self):
        """
        Simulated 'training step': balanced loss should improve IoU faster
        than uniform loss would.

        We simulate 20 gradient steps with balanced l_sigma and check IoU improvement.
        """
        torch.manual_seed(42)
        B, S = 2, 256
        n_entity = 10
        masks = _make_sparse_mask(B, S, n_entity=n_entity, seed=7)
        depth_orders = [(0, 1)] * B

        # Learnable alpha parameters
        log_alpha0 = torch.nn.Parameter(torch.zeros(B, S))
        log_alpha1 = torch.nn.Parameter(torch.zeros(B, S))
        opt = torch.optim.Adam([log_alpha0, log_alpha1], lr=0.1)

        iou_initial = compute_visible_iou(
            torch.sigmoid(log_alpha0).detach(),
            torch.sigmoid(log_alpha1).detach(),
            masks, depth_orders)

        for _ in range(30):
            opt.zero_grad()
            alpha0 = torch.sigmoid(log_alpha0)
            alpha1 = torch.sigmoid(log_alpha1)
            loss = l_sigma_spatial(alpha0, alpha1, masks)
            # Also add vis loss
            w0 = alpha0
            w1 = alpha1
            loss = loss + l_visible_weights(w0, w1, masks, depth_orders)
            loss.backward()
            opt.step()

        iou_final = compute_visible_iou(
            torch.sigmoid(log_alpha0).detach(),
            torch.sigmoid(log_alpha1).detach(),
            masks, depth_orders)

        assert iou_final > iou_initial + 0.05, (
            f"IoU should improve with balanced training: "
            f"initial={iou_initial:.3f} final={iou_final:.3f}")


# =============================================================================
# 5. Edge cases
# =============================================================================

class TestEdgeCases:

    def test_single_entity_pixel(self):
        """Entity가 1픽셀만 있어도 작동."""
        B, S = 1, 256
        masks = torch.zeros(B, 2, S)
        masks[0, 0, 127] = 1.0   # single entity0 pixel
        alpha0 = torch.rand(B, S)
        alpha1 = torch.rand(B, S)
        loss = l_sigma_spatial(alpha0, alpha1, masks)
        assert torch.isfinite(loss)

    def test_high_entity_coverage(self):
        """Entity가 50%를 커버해도 작동."""
        B, S = 1, 256
        masks = torch.zeros(B, 2, S)
        masks[0, 0, :128] = 1.0   # 50% coverage
        masks[0, 1, 128:] = 1.0
        alpha0 = torch.rand(B, S)
        alpha1 = torch.rand(B, S)
        loss = l_sigma_spatial(alpha0, alpha1, masks)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_all_zero_mask_no_nan(self):
        """Empty mask (no entities) → finite loss."""
        B, S = 1, 256
        masks = torch.zeros(B, 2, S)
        alpha0 = torch.rand(B, S)
        alpha1 = torch.rand(B, S)
        loss = l_sigma_spatial(alpha0, alpha1, masks)
        assert torch.isfinite(loss)


if __name__ == "__main__":
    # Quick smoke test
    print("Running Phase 41 sigma balance tests...")
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=False
    )
    sys.exit(result.returncode)
