"""
test_phase43_projector.py — FeatureProjector gradient flow 검증

검증 항목:
  1. 640 → 1280 projection: gradient flow 확인
  2. 640 → 640 projection: residual → identity at init
  3. 640 → 320 projection: gradient flow 확인
  4. zero-init: 초기 output이 0 (non-residual) 또는 input (residual)
  5. l_slot_ref gradient flow through projector
  6. Phase43Processor에 ref_proj_e0/e1 존재 + 올바른 dim
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase43 import (
    FeatureProjector,
    Phase43Processor,
    PRIMARY_DIM,
    BLOCK_INNER_DIMS,
    DEFAULT_INJECT_KEYS,
)
from models.entity_slot_phase42 import l_slot_ref


# =============================================================================
# FeatureProjector unit tests
# =============================================================================

class TestFeatureProjectorDims:

    def test_640_to_1280_output_shape(self):
        """640 → 1280: output shape 확인."""
        proj = FeatureProjector(640, 1280, hidden=128)
        x = torch.randn(2, 16, 640)
        out = proj(x)
        assert out.shape == (2, 16, 1280), f"expected (2,16,1280) got {out.shape}"

    def test_640_to_640_output_shape(self):
        """640 → 640: residual path, output shape."""
        proj = FeatureProjector(640, 640, hidden=128)
        x = torch.randn(2, 16, 640)
        out = proj(x)
        assert out.shape == (2, 16, 640), f"expected (2,16,640) got {out.shape}"

    def test_640_to_320_output_shape(self):
        """640 → 320: output shape 확인."""
        proj = FeatureProjector(640, 320, hidden=128)
        x = torch.randn(2, 16, 640)
        out = proj(x)
        assert out.shape == (2, 16, 320), f"expected (2,16,320) got {out.shape}"


class TestFeatureProjectorZeroInit:

    def test_non_residual_zero_output_at_init(self):
        """640 → 1280 (non-residual): zero-init last layer → output = 0."""
        proj = FeatureProjector(640, 1280, hidden=128)
        x = torch.randn(2, 16, 640)
        out = proj(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), \
            f"non-residual zero-init: output should be 0, max_abs={out.abs().max():.6f}"

    def test_residual_identity_at_init(self):
        """640 → 640 (residual): zero-init → output ≈ input."""
        proj = FeatureProjector(640, 640, hidden=128)
        x = torch.randn(2, 16, 640)
        out = proj(x)
        assert torch.allclose(out.float(), x.float(), atol=1e-5), \
            f"residual zero-init: output should ≈ input, max_diff={( out-x).abs().max():.6f}"

    def test_non_residual_320_zero_at_init(self):
        """640 → 320 (non-residual): zero-init → output = 0."""
        proj = FeatureProjector(640, 320, hidden=128)
        x = torch.randn(2, 16, 640)
        out = proj(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), \
            f"640→320 zero-init: output should be 0, max_abs={out.abs().max():.6f}"


class TestFeatureProjectorGradientFlow:

    def test_gradient_flows_640_to_1280(self):
        """640 → 1280: gradient가 projector로 흐르는지."""
        proj = FeatureProjector(640, 1280, hidden=64)
        # manually perturb to get non-zero output
        with torch.no_grad():
            proj.net[-1].weight.fill_(0.01)
        x = torch.randn(1, 4, 640)
        out = proj(x)
        loss = out.pow(2).mean()
        loss.backward()
        assert proj.net[-1].weight.grad is not None, "gradient not flowing to projector"
        assert proj.net[-1].weight.grad.abs().max() > 0, "gradient is zero"

    def test_gradient_flows_640_to_320(self):
        """640 → 320: gradient flow 확인."""
        proj = FeatureProjector(640, 320, hidden=64)
        with torch.no_grad():
            proj.net[-1].weight.fill_(0.01)
        x = torch.randn(1, 4, 640)
        out = proj(x)
        loss = out.pow(2).mean()
        loss.backward()
        assert proj.net[-1].weight.grad is not None, "gradient not flowing"
        assert proj.net[-1].weight.grad.abs().max() > 0, "gradient is zero"

    def test_l_slot_ref_gradient_through_projector_640_to_1280(self):
        """
        l_slot_ref gradient가 640 → 1280 projector를 통해 흐르는지.
        slot feature (1280 dim)와 projected ref (640→1280) 사이 MSE.
        """
        proj = FeatureProjector(640, 1280, hidden=64)
        with torch.no_grad():
            proj.net[-1].weight.fill_(0.01)

        F_primary = torch.randn(1, 4, 640)
        F_slot    = torch.randn(1, 4, 1280, requires_grad=True)
        vis_mask  = torch.ones(1, 4)

        F_ref_proj = proj(F_primary)   # (1, 4, 1280)
        loss       = l_slot_ref(F_slot, F_ref_proj.detach(), vis_mask)
        loss.backward()

        assert F_slot.grad is not None, "gradient not flowing to F_slot"
        assert F_slot.grad.abs().max() > 0, "F_slot gradient is zero"


class TestPhase43ProcessorProjectors:

    def test_ref_proj_e0_exists(self):
        """Phase43Processor에 ref_proj_e0 속성 존재."""
        proc = Phase43Processor(query_dim=640, inner_dim=640)
        assert hasattr(proc, 'ref_proj_e0'), "ref_proj_e0 없음"

    def test_ref_proj_e1_exists(self):
        """Phase43Processor에 ref_proj_e1 속성 존재."""
        proc = Phase43Processor(query_dim=640, inner_dim=640)
        assert hasattr(proc, 'ref_proj_e1'), "ref_proj_e1 없음"

    def test_ref_proj_dims_for_primary_block(self):
        """primary block (640→640): ref_proj가 residual."""
        proc = Phase43Processor(query_dim=640, inner_dim=640, primary_dim=640)
        assert proc.ref_proj_e0.residual, "640→640 projector should be residual"
        assert proc.ref_proj_e1.residual, "640→640 projector should be residual"

    def test_ref_proj_dims_for_secondary_block_1280(self):
        """secondary block (1280, primary_dim=640): ref_proj가 non-residual."""
        proc = Phase43Processor(query_dim=1280, inner_dim=1280, primary_dim=640)
        assert not proc.ref_proj_e0.residual, "640→1280 projector should be non-residual"
        assert proc.ref_proj_e0.in_dim  == 640,  f"in_dim={proc.ref_proj_e0.in_dim}"
        assert proc.ref_proj_e0.out_dim == 1280, f"out_dim={proc.ref_proj_e0.out_dim}"

    def test_ref_proj_dims_for_secondary_block_320(self):
        """secondary block (320, primary_dim=640): ref_proj가 non-residual."""
        proc = Phase43Processor(query_dim=320, inner_dim=320, primary_dim=640)
        assert not proc.ref_proj_e0.residual, "640→320 projector should be non-residual"
        assert proc.ref_proj_e0.in_dim  == 640, f"in_dim={proc.ref_proj_e0.in_dim}"
        assert proc.ref_proj_e0.out_dim == 320, f"out_dim={proc.ref_proj_e0.out_dim}"

    def test_projector_params_trainable(self):
        """projector_params()의 모든 파라미터 requires_grad=True."""
        proc = Phase43Processor(query_dim=640, inner_dim=640)
        params = proc.projector_params()
        assert len(params) > 0, "projector_params() is empty"
        for p in params:
            assert p.requires_grad, "projector param requires_grad=False"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
