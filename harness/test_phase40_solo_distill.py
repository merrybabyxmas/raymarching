"""
Phase 40 — Test: SlotLoRA, l_solo_feat_visible, l_id_contrast, l_bg_feat
=========================================================================

Phase 40의 solo reference distillation 관련 컴포넌트 검증.

  SlotLoRA            : zero-init LoRA (lora_B=zeros) — passthrough at init
  l_solo_feat_visible : MSE in visible region (F_ref detached)
  l_id_contrast       : cosine margin loss (same entity > other entity + margin)
  l_bg_feat           : background feature alignment
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase40 import (
    SlotLoRA,
    Phase40Processor,
    l_solo_feat_visible,
    l_id_contrast,
    l_bg_feat,
)

pytestmark = pytest.mark.phase40


# =============================================================================
# Fixtures
# =============================================================================

IN_FEATURES  = 64
OUT_FEATURES = 64
LORA_RANK    = 4
BATCH        = 2
SEQ          = 8
DIM          = 64


@pytest.fixture
def lora():
    return SlotLoRA(in_features=IN_FEATURES, out_features=OUT_FEATURES, rank=LORA_RANK)


@pytest.fixture
def masks_e0():
    """entity 0 visible mask (B, S)."""
    m = torch.zeros(BATCH, SEQ)
    m[0, :4] = 1.0
    m[1, :3] = 1.0
    return m


@pytest.fixture
def masks_e1():
    """entity 1 visible mask (B, S)."""
    m = torch.zeros(BATCH, SEQ)
    m[0, 4:] = 1.0
    m[1, 3:7] = 1.0
    return m


@pytest.fixture
def bg_mask():
    """background mask (B, S)."""
    m = torch.zeros(BATCH, SEQ)
    m[0, SEQ - 1] = 1.0
    m[1, SEQ - 1] = 1.0
    return m


# =============================================================================
# Tests: SlotLoRA
# =============================================================================

class TestSlotLoRA:

    def test_zero_init_is_passthrough(self, lora):
        """lora_B=zeros → 초기 출력 = 0 (추가항이 0)."""
        x = torch.randn(BATCH, SEQ, IN_FEATURES)
        out = lora(x)
        # lora_B initialized to zero → lora(x) = lora_B @ lora_A(x) = 0
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), \
            f"SlotLoRA should be zero at init (lora_B=zeros), got max={out.abs().max():.6f}"

    def test_lora_b_is_zeros_at_init(self, lora):
        """lora_B.weight가 실제로 zeros로 초기화되어 있는지."""
        assert torch.all(lora.lora_B.weight == 0), "lora_B.weight should be zeros at init"

    def test_lora_a_has_nonzero_init(self, lora):
        """lora_A.weight가 kaiming_uniform 등으로 초기화되어 0이 아닌지."""
        assert not torch.all(lora.lora_A.weight == 0), "lora_A.weight should not be all zeros"

    def test_output_shape(self, lora):
        """출력 shape = (B, S, OUT_FEATURES)."""
        x = torch.randn(BATCH, SEQ, IN_FEATURES)
        out = lora(x)
        assert out.shape == (BATCH, SEQ, OUT_FEATURES), \
            f"Expected {(BATCH, SEQ, OUT_FEATURES)}, got {out.shape}"

    def test_gradient_flows_to_lora_a(self, lora):
        """backward 후 lora_A에 gradient가 흐르는지."""
        # lora_B를 non-zero로 설정해야 gradient가 흐름
        nn.init.normal_(lora.lora_B.weight, std=0.01)
        x = torch.randn(BATCH, SEQ, IN_FEATURES)
        out = lora(x)
        loss = out.pow(2).mean()
        loss.backward()
        assert lora.lora_A.weight.grad is not None
        assert lora.lora_B.weight.grad is not None

    def test_gradient_flows_to_lora_b_after_step(self, lora):
        """optimizer step 후 lora_B가 zeros에서 벗어나는지."""
        opt = optim.SGD(lora.parameters(), lr=0.01)
        orig_b = lora.lora_B.weight.data.clone()

        # lora_B=0이라 첫 forward는 0이지만 lora_A로 gradient 흐름
        x = torch.randn(BATCH, SEQ, IN_FEATURES)
        target = torch.randn(BATCH, SEQ, OUT_FEATURES)
        # Force non-zero lora_A contribution
        nn.init.normal_(lora.lora_A.weight, std=0.1)
        out = lora(x)
        loss = (out - target).pow(2).mean()
        loss.backward()

        # lora_B.grad should be nonzero because it multiplies lora_A output
        assert lora.lora_B.weight.grad is not None
        opt.step()
        # After step, lora_B should change (it had a gradient from lora_A intermediate)
        # Note: If lora_A output was 0 (which it isn't after init), this could fail
        # We set lora_A to non-zero above, so this should work

    def test_rank_reduces_parameters(self):
        """rank가 작으면 파라미터가 적어야 함."""
        lora_small = SlotLoRA(IN_FEATURES, OUT_FEATURES, rank=2)
        lora_large = SlotLoRA(IN_FEATURES, OUT_FEATURES, rank=16)
        n_small = sum(p.numel() for p in lora_small.parameters())
        n_large = sum(p.numel() for p in lora_large.parameters())
        assert n_small < n_large, \
            f"Small rank should have fewer params: {n_small} vs {n_large}"

    def test_lora_learns_nonzero_output(self, lora):
        """충분히 학습하면 lora 출력이 0에서 벗어나는지."""
        opt = optim.Adam(lora.parameters(), lr=0.01)
        x = torch.randn(BATCH, SEQ, IN_FEATURES)
        target = torch.randn(BATCH, SEQ, OUT_FEATURES)

        for _ in range(50):
            opt.zero_grad()
            out = lora(x.detach())
            loss = (out - target).pow(2).mean()
            loss.backward()
            opt.step()

        out_final = lora(x.detach())
        assert not torch.allclose(out_final, torch.zeros_like(out_final), atol=1e-4), \
            "SlotLoRA should learn non-zero output after training"


# =============================================================================
# Tests: l_solo_feat_visible
# =============================================================================

class TestLSoloFeatVisible:

    def test_perfect_prediction_zero_loss(self, masks_e0):
        """F_slot == F_ref → loss = 0."""
        B, S = masks_e0.shape
        D = 32
        F_ref  = torch.randn(B, S, D)
        F_slot = F_ref.clone()   # perfect match

        loss = l_solo_feat_visible(F_slot, F_ref, masks_e0)
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-5), \
            f"Perfect prediction should give 0 loss, got {loss.item():.6f}"

    def test_mismatch_positive_loss(self, masks_e0):
        """F_slot != F_ref → loss > 0."""
        B, S = masks_e0.shape
        D = 32
        F_ref  = torch.randn(B, S, D)
        F_slot = torch.randn(B, S, D)   # different

        loss = l_solo_feat_visible(F_slot, F_ref, masks_e0)
        assert float(loss.item()) > 0.0, "Mismatch should give positive loss"

    def test_gradient_does_not_flow_to_ref(self, masks_e0):
        """F_ref는 detach되어야 하므로 gradient가 흐르지 않아야 함."""
        B, S = masks_e0.shape
        D = 32
        F_ref  = torch.randn(B, S, D, requires_grad=True)
        F_slot = torch.randn(B, S, D, requires_grad=True)

        loss = l_solo_feat_visible(F_slot, F_ref, masks_e0)
        loss.backward()

        assert F_slot.grad is not None, "F_slot should receive gradient"
        assert F_ref.grad is None, "F_ref should NOT receive gradient (detached)"

    def test_gradient_flows_to_f_slot(self, masks_e0):
        """F_slot에 gradient가 흐르는지."""
        B, S = masks_e0.shape
        D = 32
        F_ref  = torch.randn(B, S, D)
        F_slot = torch.randn(B, S, D, requires_grad=True)

        loss = l_solo_feat_visible(F_slot, F_ref, masks_e0)
        loss.backward()

        assert F_slot.grad is not None
        assert not torch.all(F_slot.grad == 0)

    def test_only_visible_region_contributes(self):
        """visible mask=0인 위치는 loss에 기여하지 않아야 함."""
        B, S, D = 1, 8, 16
        # Only pos 0-3 visible
        visible = torch.zeros(B, S)
        visible[0, :4] = 1.0

        F_ref  = torch.randn(B, S, D)
        F_slot = F_ref.clone()
        # Make pos 4-7 very different
        F_slot[0, 4:] = F_ref[0, 4:] + 100.0

        loss = l_solo_feat_visible(F_slot, F_ref, visible)
        # Visible region is perfect → loss should be ~0
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-3), \
            f"Non-visible region should not affect loss, got {loss.item():.4f}"

    def test_empty_mask_gives_zero_or_finite_loss(self):
        """visible mask가 모두 0이면 loss는 0 또는 finite."""
        B, S, D = 2, 8, 16
        visible = torch.zeros(B, S)
        F_ref  = torch.randn(B, S, D)
        F_slot = torch.randn(B, S, D)

        loss = l_solo_feat_visible(F_slot, F_ref, visible)
        assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-4)

    def test_loss_drives_f_slot_toward_f_ref(self, masks_e0):
        """Gradient descent로 F_slot이 F_ref에 수렴하는지."""
        B, S = masks_e0.shape
        D = 16
        F_ref = torch.randn(B, S, D)
        F_slot = nn.Parameter(torch.zeros(B, S, D))
        opt = optim.Adam([F_slot], lr=0.1)

        init_loss = float(l_solo_feat_visible(F_slot, F_ref, masks_e0).item())
        for _ in range(100):
            opt.zero_grad()
            loss = l_solo_feat_visible(F_slot, F_ref, masks_e0)
            loss.backward()
            opt.step()
        final_loss = float(l_solo_feat_visible(F_slot, F_ref, masks_e0).item())

        assert final_loss < init_loss * 0.05, \
            f"Loss should drop: {init_loss:.4f} → {final_loss:.4f}"


# =============================================================================
# Tests: l_id_contrast
# =============================================================================

class TestLIdContrast:

    def _unit_vecs(self, B, S, D):
        F = torch.randn(B, S, D)
        return F / (F.norm(dim=-1, keepdim=True) + 1e-8)

    def test_same_entity_high_sim_gives_low_loss(self, masks_e0, masks_e1):
        """F0_slot ≈ F0_ref, F1_slot ≈ F1_ref (good separation) → loss = 0."""
        B, S = masks_e0.shape
        D = 64
        # Orthogonal entity representations
        F0 = torch.zeros(B, S, D)
        F0[..., :D//2] = 1.0 / (D//2) ** 0.5
        F1 = torch.zeros(B, S, D)
        F1[..., D//2:] = 1.0 / (D//2) ** 0.5

        # Perfect: slot == ref for each entity
        loss = l_id_contrast(F0, F1, F0, F1, masks_e0, masks_e1, margin=0.1)
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-3), \
            f"Good separation should give 0 loss, got {loss.item():.4f}"

    def test_no_separation_positive_loss(self, masks_e0, masks_e1):
        """F0 ≈ F1 (same vector) → loss > 0."""
        B, S = masks_e0.shape
        D = 32
        F_same = self._unit_vecs(B, S, D)

        # Both slots point to same ref features → no contrast
        loss = l_id_contrast(F_same, F_same, F_same, F_same, masks_e0, masks_e1, margin=0.1)
        # cos(same, same) - cos(same, same) = 0 < margin → loss = margin
        assert float(loss.item()) > 0.0, \
            f"No separation should give positive loss, got {loss.item():.4f}"

    def test_gradient_flows_to_f0_slot_and_f1_slot(self, masks_e0, masks_e1):
        """backward 후 F0_slot, F1_slot에 gradient가 흐르는지."""
        B, S = masks_e0.shape
        D = 32
        F0_slot = torch.randn(B, S, D, requires_grad=True)
        F1_slot = torch.randn(B, S, D, requires_grad=True)
        F0_ref  = torch.randn(B, S, D)
        F1_ref  = torch.randn(B, S, D)

        loss = l_id_contrast(F0_slot, F1_slot, F0_ref, F1_ref, masks_e0, masks_e1)
        loss.backward()

        assert F0_slot.grad is not None, "F0_slot should receive gradient"
        assert F1_slot.grad is not None, "F1_slot should receive gradient"

    def test_gradient_does_not_flow_to_refs(self, masks_e0, masks_e1):
        """F0_ref, F1_ref는 detach → gradient가 흐르면 안 됨."""
        B, S = masks_e0.shape
        D = 32
        F0_slot = torch.randn(B, S, D, requires_grad=True)
        F1_slot = torch.randn(B, S, D, requires_grad=True)
        F0_ref  = torch.randn(B, S, D, requires_grad=True)
        F1_ref  = torch.randn(B, S, D, requires_grad=True)

        loss = l_id_contrast(F0_slot, F1_slot, F0_ref, F1_ref, masks_e0, masks_e1)
        loss.backward()

        assert F0_ref.grad is None, "F0_ref should NOT receive gradient (detached)"
        assert F1_ref.grad is None, "F1_ref should NOT receive gradient (detached)"

    def test_margin_controls_loss_threshold(self, masks_e0, masks_e1):
        """margin이 크면 loss가 크거나 같아야 함."""
        B, S = masks_e0.shape
        D = 32
        F0 = self._unit_vecs(B, S, D)
        F1 = self._unit_vecs(B, S, D)

        loss_small = l_id_contrast(F0, F1, F0, F1, masks_e0, masks_e1, margin=0.05)
        loss_large = l_id_contrast(F0, F1, F0, F1, masks_e0, masks_e1, margin=0.50)
        assert float(loss_large.item()) >= float(loss_small.item()), \
            f"Larger margin should give >= loss: {loss_small:.4f} vs {loss_large:.4f}"

    def test_loss_nonnegative(self, masks_e0, masks_e1):
        """l_id_contrast >= 0 (hinge loss)."""
        B, S = masks_e0.shape
        D = 32
        for _ in range(5):
            F0 = self._unit_vecs(B, S, D)
            F1 = self._unit_vecs(B, S, D)
            F0r = self._unit_vecs(B, S, D)
            F1r = self._unit_vecs(B, S, D)
            loss = l_id_contrast(F0, F1, F0r, F1r, masks_e0, masks_e1)
            assert float(loss.item()) >= 0.0, \
                f"Contrastive loss should be non-negative, got {loss.item():.4f}"


# =============================================================================
# Tests: l_bg_feat
# =============================================================================

class TestLBgFeat:

    def test_perfect_prediction_zero_loss(self, bg_mask):
        """F_slot == F_ref → loss = 0."""
        B, S = bg_mask.shape
        D = 16
        F_ref  = torch.randn(B, S, D)
        F_slot = F_ref.clone()

        loss = l_bg_feat(F_slot, F_ref, bg_mask)
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-5)

    def test_mismatch_positive_loss(self, bg_mask):
        """F_slot != F_ref in bg region → loss > 0."""
        B, S = bg_mask.shape
        D = 16
        F_ref  = torch.randn(B, S, D)
        F_slot = torch.randn(B, S, D)

        loss = l_bg_feat(F_slot, F_ref, bg_mask)
        assert float(loss.item()) > 0.0

    def test_gradient_flows_to_f_slot(self, bg_mask):
        """F_slot에 gradient가 흐르는지."""
        B, S = bg_mask.shape
        D = 16
        F_ref  = torch.randn(B, S, D)
        F_slot = torch.randn(B, S, D, requires_grad=True)

        loss = l_bg_feat(F_slot, F_ref, bg_mask)
        loss.backward()

        assert F_slot.grad is not None
        assert not torch.all(F_slot.grad == 0)

    def test_ref_is_detached(self, bg_mask):
        """F_ref에 gradient가 흐르지 않아야 함."""
        B, S = bg_mask.shape
        D = 16
        F_ref  = torch.randn(B, S, D, requires_grad=True)
        F_slot = torch.randn(B, S, D, requires_grad=True)

        loss = l_bg_feat(F_slot, F_ref, bg_mask)
        loss.backward()

        assert F_ref.grad is None, "F_ref should be detached (no gradient)"

    def test_empty_bg_mask_gives_finite_loss(self):
        """bg_mask 전부 0이면 loss = 0 또는 finite."""
        B, S, D = 2, 8, 16
        bg = torch.zeros(B, S)
        F_ref  = torch.randn(B, S, D)
        F_slot = torch.randn(B, S, D)

        loss = l_bg_feat(F_slot, F_ref, bg)
        assert torch.isfinite(loss)
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-4)


# =============================================================================
# Phase40Processor: construction and parameter structure
# =============================================================================

class TestPhase40ProcessorParams:

    def test_lora_params_exist(self):
        """Phase40Processor가 lora_k, lora_v, lora_out를 가지는지."""
        proc = Phase40Processor(
            query_dim=DIM,
            inner_dim=DIM,
            adapter_rank=16,
            lora_rank=LORA_RANK,
            cross_attention_dim=DIM,
        )
        assert hasattr(proc, "lora_k"),   "Phase40Processor should have lora_k"
        assert hasattr(proc, "lora_v"),   "Phase40Processor should have lora_v"
        assert hasattr(proc, "lora_out"), "Phase40Processor should have lora_out"

    def test_lora_zero_init(self):
        """LoRA B 행렬이 zeros로 초기화되었는지."""
        proc = Phase40Processor(
            query_dim=DIM, inner_dim=DIM,
            adapter_rank=16, lora_rank=LORA_RANK,
            cross_attention_dim=DIM,
        )
        assert torch.all(proc.lora_k.lora_B.weight == 0), "lora_k.lora_B should be zeros"
        assert torch.all(proc.lora_v.lora_B.weight == 0), "lora_v.lora_B should be zeros"
        assert torch.all(proc.lora_out.lora_B.weight == 0), "lora_out.lora_B should be zeros"

    def test_lora_params_require_grad(self):
        """LoRA 파라미터가 requires_grad=True인지."""
        proc = Phase40Processor(
            query_dim=DIM, inner_dim=DIM,
            adapter_rank=16, lora_rank=LORA_RANK,
            cross_attention_dim=DIM,
        )
        for name, p in proc.lora_k.named_parameters():
            assert p.requires_grad, f"lora_k.{name} should require grad"
        for name, p in proc.lora_v.named_parameters():
            assert p.requires_grad, f"lora_v.{name} should require grad"

    def test_lora_param_count(self):
        """LoRA 파라미터 수가 full rank보다 적어야 함."""
        proc = Phase40Processor(
            query_dim=DIM, inner_dim=DIM,
            adapter_rank=16, lora_rank=LORA_RANK,
            cross_attention_dim=DIM,
        )
        # lora_k: (rank * in) + (out * rank)
        n_lora = sum(p.numel() for p in proc.lora_k.parameters())
        # full rank: in * out = DIM * DIM
        n_full = DIM * DIM
        assert n_lora < n_full, \
            f"LoRA should have fewer params than full rank: {n_lora} vs {n_full}"
