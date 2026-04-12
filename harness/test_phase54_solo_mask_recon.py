"""Phase 54: masked solo reconstruction loss regression test."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase53 import l_solo_masked_reconstruction


def test_masked_recon_ignores_background():
    pred = torch.zeros(1, 2, 2, 2)
    target = torch.zeros(1, 2, 2, 2)
    target[:, :, 0, 0] = 1.0
    mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

    good = l_solo_masked_reconstruction(target, target, mask)
    bad_bg = l_solo_masked_reconstruction(pred, target, mask)
    assert good.item() < bad_bg.item(), (
        f"masked recon should punish visible error only: good={good.item():.4f} "
        f"bad_bg={bad_bg.item():.4f}"
    )


def test_empty_mask_returns_zero():
    pred = torch.randn(1, 2, 2, 2)
    target = torch.randn(1, 2, 2, 2)
    mask = torch.zeros(1, 2, 2)
    loss = l_solo_masked_reconstruction(pred, target, mask)
    assert torch.isfinite(loss), "loss should stay finite on empty masks"
    assert loss.item() == 0.0, f"empty mask should give zero loss, got {loss.item():.4f}"


if __name__ == "__main__":
    test_masked_recon_ignores_background()
    test_empty_mask_returns_zero()
    print("test_phase54_solo_mask_recon: PASS")
