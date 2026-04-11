"""
test_phase42_metrics.py — Phase 42 metric / validation score 검증

검증 항목:
  1. compute_visible_iou_e0/e1 toy example 기댓값
  2. val_score_phase42가 id_margin을 포함하지 않는지
  3. val_score_phase42 수식 검증
  4. l_slot_ref: visible mask 적용 + stop-gradient
  5. l_slot_contrast: cosine margin 부호 검증
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase40 import (
    compute_visible_iou_e0,
    compute_visible_iou_e1,
)
from models.entity_slot_phase42 import (
    val_score_phase42,
    l_slot_ref,
    l_slot_contrast,
)


# =============================================================================
# val_score_phase42 검증
# =============================================================================

class TestValScorePhase42:

    def test_formula(self):
        """val_score_phase42 = 0.3*iou0 + 0.3*iou1 + 0.2*ord + 0.2*(1-wrong)."""
        iou_e0 = 0.5
        iou_e1 = 0.4
        ord_   = 0.7
        wrong  = 0.1
        expected = 0.3*0.5 + 0.3*0.4 + 0.2*0.7 + 0.2*(1-0.1)
        result   = val_score_phase42(iou_e0, iou_e1, ord_, wrong)
        assert abs(result - expected) < 1e-6, \
            f"val_score_phase42 수식 오류: expected={expected:.6f}, got={result:.6f}"

    def test_no_id_margin_dependency(self):
        """val_score_phase42는 id_margin 인자를 받지 않음."""
        import inspect
        sig = inspect.signature(val_score_phase42)
        params = list(sig.parameters.keys())
        assert "id_margin" not in params, \
            f"val_score_phase42에 id_margin이 있으면 안 됨: {params}"

    def test_perfect_score(self):
        """iou=1, ord=1, wrong=0 → 1.0."""
        result = val_score_phase42(1.0, 1.0, 1.0, 0.0)
        assert abs(result - 1.0) < 1e-6, f"완벽한 점수가 1.0이 아님: {result}"

    def test_worst_score(self):
        """iou=0, ord=0, wrong=1 → 0.0."""
        result = val_score_phase42(0.0, 0.0, 0.0, 1.0)
        assert abs(result - 0.0) < 1e-6, f"최악 점수가 0.0이 아님: {result}"

    def test_phase42_vs_phase40_different_formula(self):
        """
        val_score_phase42가 id_margin을 빼고 iou 가중치를 올린 것인지 확인.
        phase40: 0.20*iou0 + 0.20*iou1 + 0.20*ord + 0.15*(1-wrong) + 0.15*id + 0.10*rollout
        phase42: 0.30*iou0 + 0.30*iou1 + 0.20*ord + 0.20*(1-wrong)
        iou가 높을 때 phase42 점수가 phase40보다 높아야 함.
        """
        from models.entity_slot_phase40 import val_score_phase40
        iou_e0, iou_e1 = 0.8, 0.8
        ord_, wrong, id_m = 0.7, 0.1, 0.0
        s40 = val_score_phase40(iou_e0, iou_e1, ord_, wrong, id_m, rollout_id=0.0)
        s42 = val_score_phase42(iou_e0, iou_e1, ord_, wrong)
        assert s42 > s40, \
            f"iou 높을 때 phase42 score가 phase40보다 높아야 함: s42={s42:.4f} s40={s40:.4f}"


# =============================================================================
# compute_visible_iou_e0/e1 toy example
# =============================================================================

class TestVisibleIou:

    def _make_toy(self):
        """
        간단한 toy: 4 pixels
          pixel 0: entity0 only (excl_0)
          pixel 1: entity1 only (excl_1)
          pixel 2: both (overlap)
          pixel 3: background
        depth order: entity0 is front
        """
        B, S = 1, 4
        masks = torch.zeros(B, 2, S)
        masks[0, 0, 0] = 1.0  # e0 pixel 0
        masks[0, 0, 2] = 1.0  # e0 pixel 2 (overlap)
        masks[0, 1, 1] = 1.0  # e1 pixel 1
        masks[0, 1, 2] = 1.0  # e1 pixel 2 (overlap)
        depth_orders = [(0, 1)]   # e0 is front
        return masks, depth_orders

    def test_iou_e0_perfect(self):
        """w0=[1,0,1,0] 일 때 iou_e0≈1.0 (e0 front → overlap도 e0에 할당)."""
        masks, do = self._make_toy()
        w0 = torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # (1, 4)
        iou = compute_visible_iou_e0(w0, masks, do)
        assert abs(iou - 1.0) < 0.05, \
            f"perfect w0에서 iou_e0 ≈ 1.0 기대, got {iou:.4f}"

    def test_iou_e0_zero(self):
        """w0=0 everywhere → iou_e0=0."""
        masks, do = self._make_toy()
        w0 = torch.zeros(1, 4)
        iou = compute_visible_iou_e0(w0, masks, do)
        assert iou < 0.05, f"w0=0 이면 iou_e0≈0, got {iou:.4f}"

    def test_iou_e1_perfect(self):
        """w1=[0,1,0,0] 일 때 iou_e1≈1.0 (e0 front → overlap은 e0에 할당 → w1 target은 e1 excl만)."""
        masks, do = self._make_toy()
        # w1_target: e0 front이므로 overlap은 e0. e1 target = excl_1 = pixel 1
        w1 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        iou = compute_visible_iou_e1(w1, masks, do)
        assert abs(iou - 1.0) < 0.05, \
            f"perfect w1에서 iou_e1 ≈ 1.0 기대, got {iou:.4f}"

    def test_iou_symmetric_when_e1_front(self):
        """e1이 front이면 iou_e1 perfect는 w1=[0,1,1,0]."""
        masks, _ = self._make_toy()
        do_e1_front = [(1, 0)]  # e1 is front
        w1 = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        iou = compute_visible_iou_e1(w1, masks, do_e1_front)
        assert abs(iou - 1.0) < 0.05, \
            f"e1 front + perfect w1: iou_e1 ≈ 1.0 기대, got {iou:.4f}"


# =============================================================================
# l_slot_ref 검증
# =============================================================================

class TestLSlotRef:

    def test_zero_loss_when_identical(self):
        """F_slot == F_ref → loss=0."""
        F = torch.randn(1, 16, 64)
        mask = torch.ones(1, 16)
        loss = l_slot_ref(F, F, mask)
        assert loss.item() < 1e-6, f"동일 feature일 때 l_slot_ref ≠ 0: {loss.item()}"

    def test_stop_gradient_on_ref(self):
        """F_ref에 gradient가 흐르지 않는지."""
        F_slot = torch.randn(1, 16, 64, requires_grad=True)
        F_ref  = torch.randn(1, 16, 64, requires_grad=True)
        mask   = torch.ones(1, 16)
        loss   = l_slot_ref(F_slot, F_ref, mask)
        loss.backward()
        assert F_ref.grad is None or F_ref.grad.abs().max().item() == 0.0, \
            "F_ref에 gradient가 흘렀음 — stop-gradient 위반"

    def test_visible_mask_ignores_background(self):
        """mask=0인 pixel은 loss에 기여하지 않음."""
        B, S, D = 1, 16, 32
        F_slot = torch.zeros(B, S, D)
        F_ref  = torch.ones(B, S, D) * 10.0  # 큰 차이
        mask   = torch.zeros(B, S)  # all background
        loss   = l_slot_ref(F_slot, F_ref, mask)
        assert loss.item() < 1e-5, \
            f"mask=0 일 때 l_slot_ref는 0이어야 함: {loss.item()}"

    def test_higher_loss_for_larger_diff(self):
        """F_slot과 F_ref의 차이가 클수록 loss가 큼."""
        B, S, D = 1, 16, 32
        mask = torch.ones(B, S)
        F_ref = torch.zeros(B, S, D)
        F_small = torch.ones(B, S, D) * 0.1
        F_large = torch.ones(B, S, D) * 10.0
        l_small = l_slot_ref(F_small, F_ref, mask)
        l_large = l_slot_ref(F_large, F_ref, mask)
        assert l_large > l_small, \
            f"큰 차이에서 loss가 더 커야 함: l_small={l_small:.4f} l_large={l_large:.4f}"


# =============================================================================
# l_slot_contrast 검증
# =============================================================================

class TestLSlotContrast:

    def test_zero_loss_when_perfectly_aligned(self):
        """
        F0_slot이 F0_ref와 완전히 일치하고 F1_ref와 반대이면 loss=0.
        """
        B, S, D = 1, 16, 32
        F0_ref = F.normalize(torch.ones(B, S, D), dim=-1)
        F1_ref = F.normalize(-torch.ones(B, S, D), dim=-1)
        F0_slot = F0_ref.clone()
        F1_slot = F1_ref.clone()
        mask_e0 = torch.ones(B, S)
        mask_e1 = torch.ones(B, S)
        loss = l_slot_contrast(F0_slot, F1_slot, F0_ref, F1_ref,
                               mask_e0, mask_e1, margin=0.1)
        assert loss.item() < 1e-5, \
            f"완벽히 정렬된 경우 loss=0 기대: {loss.item():.6f}"

    def test_positive_loss_when_confused(self):
        """
        F0_slot이 F1_ref와 가깝고 F0_ref와 멀면 loss > 0.
        """
        B, S, D = 1, 16, 32
        F0_ref  = F.normalize(torch.randn(B, S, D), dim=-1)
        F1_ref  = F.normalize(torch.randn(B, S, D), dim=-1)
        # F0_slot이 F1_ref와 같음 → entity confusion
        F0_slot = F1_ref.clone()
        F1_slot = F0_ref.clone()
        mask_e0 = torch.ones(B, S)
        mask_e1 = torch.ones(B, S)
        loss = l_slot_contrast(F0_slot, F1_slot, F0_ref, F1_ref,
                               mask_e0, mask_e1, margin=0.0)
        assert loss.item() > 0.0, \
            f"entity confusion 상황에서 loss > 0이어야 함: {loss.item():.6f}"

    def test_stop_gradient_on_refs(self):
        """F0_ref, F1_ref에 gradient 없어야 함."""
        B, S, D = 1, 16, 32
        F0_ref  = torch.randn(B, S, D, requires_grad=True)
        F1_ref  = torch.randn(B, S, D, requires_grad=True)
        F0_slot = torch.randn(B, S, D, requires_grad=True)
        F1_slot = torch.randn(B, S, D, requires_grad=True)
        mask_e0 = torch.ones(B, S)
        mask_e1 = torch.ones(B, S)
        loss = l_slot_contrast(F0_slot, F1_slot, F0_ref, F1_ref, mask_e0, mask_e1)
        loss.backward()
        for name, ref in [("F0_ref", F0_ref), ("F1_ref", F1_ref)]:
            assert ref.grad is None or ref.grad.abs().max() == 0.0, \
                f"{name}에 gradient가 흘렀음 — stop-gradient 위반"

    def test_mask_zero_gives_zero_contribution(self):
        """mask=0이면 해당 region은 loss에 기여하지 않음."""
        B, S, D = 1, 16, 32
        F0_ref  = F.normalize(torch.ones(B, S, D), dim=-1)
        F1_ref  = F.normalize(-torch.ones(B, S, D), dim=-1)
        # F0_slot이 F1_ref와 같음 (confusion)
        F0_slot = F1_ref.clone()
        F1_slot = F0_ref.clone()
        mask_zero = torch.zeros(B, S)  # no entities visible
        loss = l_slot_contrast(F0_slot, F1_slot, F0_ref, F1_ref, mask_zero, mask_zero)
        assert loss.item() < 1e-5, \
            f"mask=0일 때 loss=0 기대: {loss.item():.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
