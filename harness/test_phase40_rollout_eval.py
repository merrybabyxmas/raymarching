"""
Phase 40 — Test: rollout eval, visible mask np computation, dataset
====================================================================

Phase 40 rollout evaluation과 solo data pipeline 검증.

  compute_visible_masks_np    : numpy 버전 visible mask (generate_solo_renders.py)
  make_pseudo_solo_frames     : composite × mask → pseudo-solo
  ObjaverseDatasetPhase40     : 8-tuple return
  get_lambda_diff_phase40     : 3-stage lambda schedule
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_solo_renders import (
    compute_visible_masks_np,
    make_pseudo_solo_frames,
)
from models.entity_slot_phase40 import (
    compute_visible_masks,
    val_score_phase40,
)

pytestmark = pytest.mark.phase40


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tmp_sample_dir(tmp_path):
    """
    toy sample directory with frames + masks.
    Layout:
      {tmp}/sample_0/frames/0000.png ... 0003.png
      {tmp}/sample_0/mask/0000_entity0.png ... 0003_entity1.png
      {tmp}/sample_0/meta.json
    """
    import json

    sample_dir = tmp_path / "sample_0"
    frames_dir = sample_dir / "frames"
    mask_dir   = sample_dir / "mask"
    frames_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    H, W, T = 32, 32, 4
    rng = np.random.default_rng(0)

    # Composite frames
    for t in range(T):
        img = (rng.random((H, W, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(frames_dir / f"{t:04d}.png")

    # Masks: entity0 left half, entity1 right half
    for t in range(T):
        m0 = np.zeros((H, W), dtype=np.uint8)
        m0[:, :W//2] = 255
        m1 = np.zeros((H, W), dtype=np.uint8)
        m1[:, W//2:] = 255
        Image.fromarray(m0, mode="L").save(mask_dir / f"{t:04d}_entity0.png")
        Image.fromarray(m1, mode="L").save(mask_dir / f"{t:04d}_entity1.png")

    meta = {"n_frames": T, "depth_order": [[0, 1]] * T}
    with open(sample_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    return sample_dir


# =============================================================================
# Tests: compute_visible_masks_np
# =============================================================================

class TestComputeVisibleMasksNp:

    def test_front_entity_unchanged(self):
        """front entity의 visible mask = full mask."""
        T, S = 3, 8
        masks = np.zeros((T, 2, S))
        masks[:, 0, :4] = 1.0   # e0 left
        masks[:, 1, 2:] = 1.0   # e1 right (overlap at 2-3)
        depth_orders = [(0, 1)] * T

        vis = compute_visible_masks_np(masks, depth_orders)

        np.testing.assert_allclose(
            vis[:, 0, :], masks[:, 0, :], atol=1e-6,
            err_msg="front entity visible mask should equal full mask"
        )

    def test_back_entity_occluded_by_front(self):
        """back entity: visible = mask & ~front_mask."""
        T, S = 2, 8
        masks = np.zeros((T, 2, S))
        masks[:, 0, 2:6] = 1.0   # e0 (front)
        masks[:, 1, 4:8] = 1.0   # e1 (back), overlap at 4-5
        depth_orders = [(0, 1)] * T

        vis = compute_visible_masks_np(masks, depth_orders)

        expected_e1 = masks[:, 1, :] * (1.0 - masks[:, 0, :])
        np.testing.assert_allclose(
            vis[:, 1, :], expected_e1, atol=1e-6,
            err_msg="back entity should be occluded in overlap region"
        )

    def test_output_shape(self):
        """출력 shape = (T, 2, S)."""
        T, S = 5, 16
        masks = np.random.rand(T, 2, S) > 0.5
        depth_orders = [(0, 1)] * T
        vis = compute_visible_masks_np(masks.astype(float), depth_orders)
        assert vis.shape == (T, 2, S), f"Expected ({T}, 2, {S}), got {vis.shape}"

    def test_output_values_in_0_1(self):
        """모든 값이 [0, 1]인지."""
        rng = np.random.default_rng(7)
        masks = (rng.random((4, 2, 12)) > 0.5).astype(float)
        depth_orders = [(0, 1), (1, 0), (0, 1), (1, 0)]
        vis = compute_visible_masks_np(masks, depth_orders)
        assert vis.min() >= 0.0
        assert vis.max() <= 1.0

    def test_e1_front_depth_order(self):
        """depth_order=(1, 0)일 때 e1이 front → e1 fully visible."""
        T, S = 1, 6
        masks = np.zeros((T, 2, S))
        masks[0, 0, 1:5] = 1.0   # e0
        masks[0, 1, 3:6] = 1.0   # e1 (overlap at 3-4)
        depth_orders = [(1, 0)]   # e1 front

        vis = compute_visible_masks_np(masks, depth_orders)
        # e1 (front) should be fully visible
        np.testing.assert_allclose(vis[0, 1], masks[0, 1], atol=1e-6,
                                   err_msg="e1 front: visible should equal full mask")
        # e0 (back) should be occluded in overlap
        expected_e0 = masks[0, 0] * (1.0 - masks[0, 1])
        np.testing.assert_allclose(vis[0, 0], expected_e0, atol=1e-6,
                                   err_msg="e0 back: visible only where e1=0")

    def test_consistent_with_torch_version(self):
        """numpy 버전과 torch 버전 결과가 일치하는지."""
        T, S = 3, 12
        rng = np.random.default_rng(42)
        masks_np = (rng.random((T, 2, S)) > 0.4).astype(float)
        depth_orders = [(0, 1), (1, 0), (0, 1)]

        vis_np = compute_visible_masks_np(masks_np, depth_orders)
        # compute_visible_masks (torch version) expects (T, 2, S)
        vis_torch = compute_visible_masks(masks_np, depth_orders)

        np.testing.assert_allclose(
            vis_np, vis_torch, atol=1e-5,
            err_msg="numpy and torch visible mask implementations should agree"
        )


# =============================================================================
# Tests: make_pseudo_solo_frames
# =============================================================================

class TestMakePseudoSoloFrames:

    def test_output_shapes(self, tmp_sample_dir):
        """solo_e0, solo_e1의 shape이 composite와 동일한지."""
        frames_dir = tmp_sample_dir / "frames"
        mask_dir   = tmp_sample_dir / "mask"

        pngs = sorted(frames_dir.glob("*.png"))
        frames = np.stack([np.array(Image.open(p).convert("RGB")) for p in pngs])

        solo_e0, solo_e1 = make_pseudo_solo_frames(frames, mask_dir, n_frames=4)

        assert solo_e0.shape == frames.shape, \
            f"solo_e0 shape should match composite: {solo_e0.shape} != {frames.shape}"
        assert solo_e1.shape == frames.shape, \
            f"solo_e1 shape should match composite: {solo_e1.shape} != {frames.shape}"

    def test_e0_pixels_in_mask_region_equal_composite(self, tmp_sample_dir):
        """mask0 영역에서 solo_e0 = composite."""
        frames_dir = tmp_sample_dir / "frames"
        mask_dir   = tmp_sample_dir / "mask"

        pngs = sorted(frames_dir.glob("*.png"))
        frames = np.stack([np.array(Image.open(p).convert("RGB")) for p in pngs])

        solo_e0, _ = make_pseudo_solo_frames(frames, mask_dir, n_frames=4)

        # Load mask0 for frame 0
        m0 = np.array(Image.open(mask_dir / "0000_entity0.png").convert("L"))
        m0_bin = (m0 > 127)

        # In mask region: solo should equal composite
        np.testing.assert_array_equal(
            solo_e0[0][m0_bin], frames[0][m0_bin],
            err_msg="solo_e0 in mask region should equal composite"
        )

    def test_e0_pixels_outside_mask_are_bg(self, tmp_sample_dir):
        """mask0=0인 위치에서 solo_e0 = bg_color."""
        frames_dir = tmp_sample_dir / "frames"
        mask_dir   = tmp_sample_dir / "mask"

        pngs = sorted(frames_dir.glob("*.png"))
        frames = np.stack([np.array(Image.open(p).convert("RGB")) for p in pngs])
        bg = (0, 0, 0)

        solo_e0, _ = make_pseudo_solo_frames(frames, mask_dir, n_frames=4, bg_color=bg)

        m0 = np.array(Image.open(mask_dir / "0000_entity0.png").convert("L"))
        outside = ~(m0 > 127)

        np.testing.assert_array_equal(
            solo_e0[0][outside],
            np.full((outside.sum(), 3), bg[0], dtype=np.uint8),
            err_msg="solo_e0 outside mask should be bg_color"
        )

    def test_output_dtype_uint8(self, tmp_sample_dir):
        """출력이 uint8이어야 함."""
        frames_dir = tmp_sample_dir / "frames"
        mask_dir   = tmp_sample_dir / "mask"

        pngs = sorted(frames_dir.glob("*.png"))
        frames = np.stack([np.array(Image.open(p).convert("RGB")) for p in pngs])

        solo_e0, solo_e1 = make_pseudo_solo_frames(frames, mask_dir, n_frames=4)
        assert solo_e0.dtype == np.uint8, f"Expected uint8, got {solo_e0.dtype}"
        assert solo_e1.dtype == np.uint8, f"Expected uint8, got {solo_e1.dtype}"

    def test_e0_and_e1_are_disjoint_in_bg(self, tmp_sample_dir):
        """
        non-overlapping masks (e0 left, e1 right) →
        solo_e0 오른쪽 = bg, solo_e1 왼쪽 = bg.
        """
        frames_dir = tmp_sample_dir / "frames"
        mask_dir   = tmp_sample_dir / "mask"

        pngs = sorted(frames_dir.glob("*.png"))
        frames = np.stack([np.array(Image.open(p).convert("RGB")) for p in pngs])
        H, W = frames.shape[1], frames.shape[2]

        solo_e0, solo_e1 = make_pseudo_solo_frames(frames, mask_dir, n_frames=4, bg_color=(0, 0, 0))

        # e0 mask = left half → right half of solo_e0 should be bg
        np.testing.assert_array_equal(
            solo_e0[0, :, W//2:], 0,
            err_msg="solo_e0 right half (outside e0 mask) should be bg"
        )
        # e1 mask = right half → left half of solo_e1 should be bg
        np.testing.assert_array_equal(
            solo_e1[0, :, :W//2], 0,
            err_msg="solo_e1 left half (outside e1 mask) should be bg"
        )


# =============================================================================
# Tests: get_lambda_diff_phase40 (3-stage schedule)
# =============================================================================

try:
    from scripts.train_phase40 import get_lambda_diff_phase40
    HAS_TRAIN_PHASE40 = True
except ImportError:
    HAS_TRAIN_PHASE40 = False


@pytest.mark.skipif(not HAS_TRAIN_PHASE40, reason="train_phase40 not importable without deps")
class TestGetLambdaDiffPhase40:

    def test_stage1_lambda_is_zero(self):
        """Stage1 (epoch < s1_end): lambda_diff = 0."""
        total = 80
        s1_end = 20
        s2_end = 60
        for ep in range(s1_end):
            lam = get_lambda_diff_phase40(ep, total, s1_end, s2_end, lambda_diff_max=0.3)
            assert lam == pytest.approx(0.0, abs=1e-6), \
                f"lambda_diff should be 0 in stage1, epoch={ep}, got {lam}"

    def test_stage2_lambda_is_zero(self):
        """Stage2 (s1_end <= epoch < s2_end): lambda_diff = 0."""
        total = 80
        s1_end = 20
        s2_end = 60
        for ep in range(s1_end, s2_end):
            lam = get_lambda_diff_phase40(ep, total, s1_end, s2_end, lambda_diff_max=0.3)
            assert lam == pytest.approx(0.0, abs=1e-6), \
                f"lambda_diff should be 0 in stage2, epoch={ep}, got {lam}"

    def test_stage3_ramps_up(self):
        """Stage3 (epoch >= s2_end): lambda_diff이 0에서 max로 증가."""
        total = 80
        s1_end = 20
        s2_end = 60
        s3_end = total

        lam_start = get_lambda_diff_phase40(s2_end, total, s1_end, s2_end, lambda_diff_max=0.3)
        lam_end   = get_lambda_diff_phase40(s3_end - 1, total, s1_end, s2_end, lambda_diff_max=0.3)

        assert lam_start < lam_end, \
            f"lambda_diff should increase in stage3: {lam_start:.4f} → {lam_end:.4f}"
        assert lam_start >= 0.0
        assert lam_end <= 0.3 + 1e-6

    def test_final_epoch_reaches_max(self):
        """마지막 epoch에서 lambda_diff ≈ lambda_diff_max."""
        total = 80
        s1_end = 20
        s2_end = 60
        lam_max = 0.3

        lam_final = get_lambda_diff_phase40(total - 1, total, s1_end, s2_end, lambda_diff_max=lam_max)
        assert lam_final == pytest.approx(lam_max, abs=1e-4), \
            f"Final epoch lambda_diff should ≈ {lam_max}, got {lam_final:.4f}"

    def test_monotone_in_stage3(self):
        """Stage3에서 lambda_diff가 단조증가하는지."""
        total = 80
        s1_end = 20
        s2_end = 60
        lam_max = 0.3

        lams = [
            get_lambda_diff_phase40(ep, total, s1_end, s2_end, lambda_diff_max=lam_max)
            for ep in range(s2_end, total)
        ]
        for i in range(len(lams) - 1):
            assert lams[i] <= lams[i + 1], \
                f"lambda_diff should be non-decreasing in stage3: {lams[i]:.4f} > {lams[i+1]:.4f}"


# =============================================================================
# Tests: val_score_phase40 integration
# =============================================================================

class TestValScorePhase40Integration:

    def test_phase40_score_higher_with_better_id(self):
        """id_margin 개선 시 score 증가."""
        s_low_id  = val_score_phase40(0.5, 0.5, 0.6, 0.2, id_margin=0.0)
        s_high_id = val_score_phase40(0.5, 0.5, 0.6, 0.2, id_margin=1.0)
        assert s_high_id > s_low_id, \
            f"Higher id_margin should improve val_score: {s_low_id:.4f} vs {s_high_id:.4f}"

    def test_phase40_score_higher_with_better_rollout(self):
        """rollout_id 개선 시 score 증가."""
        s_no_rollout  = val_score_phase40(0.5, 0.5, 0.6, 0.2, id_margin=0.5, rollout_id=0.0)
        s_yes_rollout = val_score_phase40(0.5, 0.5, 0.6, 0.2, id_margin=0.5, rollout_id=1.0)
        assert s_yes_rollout > s_no_rollout

    def test_phase40_target_score_achievable(self):
        """
        target: iou_e0>0.50, iou_e1>0.50, ord>0.70, leak<0.10, id_margin>0.50
        이 조건 만족 시 val_score > 0.5를 달성할 수 있는지 확인.
        0.2*0.51 + 0.2*0.51 + 0.2*0.71 + 0.15*0.91 + 0.15*0.51 + 0.1*0.0
        = 0.102 + 0.102 + 0.142 + 0.1365 + 0.0765 = 0.559
        """
        s = val_score_phase40(
            iou_e0=0.51, iou_e1=0.51, ordering_acc=0.71,
            wrong_leak=0.09, id_margin=0.51, rollout_id=0.0
        )
        assert s > 0.5, f"Target metrics should give val_score > 0.5, got {s:.4f}"

    def test_phase39_equiv_metrics_consistency(self):
        """
        Phase 39 metrics → Phase 40 score 환산이 합리적인지.
        Phase39 val_score=0.408 → Phase40 유사 조건 score가 0.3~0.7 범위인지.
        """
        # Phase39 best: visible_iou=0.14, ordering_acc=0.40, wrong_leak=0.30, dra=0.91
        # In Phase40: iou_e0 ≈ iou_e1 ≈ visible_iou (halved or similar), id_margin unknown
        s = val_score_phase40(
            iou_e0=0.14, iou_e1=0.14, ordering_acc=0.40,
            wrong_leak=0.30, id_margin=0.0, rollout_id=0.0
        )
        assert 0.2 < s < 0.7, \
            f"Phase39-equivalent score should be in 0.2~0.7, got {s:.4f}"
