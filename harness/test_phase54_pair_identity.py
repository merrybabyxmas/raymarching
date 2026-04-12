"""Phase 54: paired identity preservation under collision."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase53 import l_pair_identity_preservation


def _make_masks():
    m0 = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    m1 = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    return m0, m1


def test_correct_pair_beats_collapse():
    m0, m1 = _make_masks()
    f0_ref = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    f1_ref = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]])
    correct = l_pair_identity_preservation(f0_ref, f1_ref, f0_ref, f1_ref, m0, m1)
    collapse = l_pair_identity_preservation(f0_ref, f0_ref, f0_ref, f1_ref, m0, m1)
    assert correct.item() < collapse.item(), (
        f"correct pair should beat collapse: correct={correct.item():.4f} "
        f"collapse={collapse.item():.4f}"
    )


def test_correct_pair_beats_swap():
    m0, m1 = _make_masks()
    f0_ref = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    f1_ref = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]])
    correct = l_pair_identity_preservation(f0_ref, f1_ref, f0_ref, f1_ref, m0, m1)
    swapped = l_pair_identity_preservation(f1_ref, f0_ref, f0_ref, f1_ref, m0, m1)
    assert correct.item() < swapped.item(), (
        f"correct pair should beat swap: correct={correct.item():.4f} "
        f"swap={swapped.item():.4f}"
    )


def test_overlap_strengthens_pair_penalty():
    f0_ref = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    f1_ref = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]])
    m0_sparse = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    m1_sparse = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    m0_overlap = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    m1_overlap = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    collapse_sparse = l_pair_identity_preservation(f0_ref, f0_ref, f0_ref, f1_ref, m0_sparse, m1_sparse)
    collapse_overlap = l_pair_identity_preservation(f0_ref, f0_ref, f0_ref, f1_ref, m0_overlap, m1_overlap)
    assert collapse_overlap.item() > collapse_sparse.item(), (
        f"overlap should strengthen penalty: sparse={collapse_sparse.item():.4f} "
        f"overlap={collapse_overlap.item():.4f}"
    )


if __name__ == "__main__":
    test_correct_pair_beats_collapse()
    test_correct_pair_beats_swap()
    test_overlap_strengthens_pair_penalty()
    print("test_phase54_pair_identity: PASS")
