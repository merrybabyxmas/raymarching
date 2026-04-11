"""
Phase 39 — Test: overlap-heavy held-out validation split
=========================================================

validation set이 실제로 overlap-heavy인지 검증.

Phase 38의 probe_idx=0 하나만 평가하는 방식은 의미가 없음.
Phase 39: overlap score 상위 samples를 val set으로 고정.

검증 항목
---------
1. val set이 train set보다 overlap score가 높은지
2. sampling weight가 overlap에 비례하는지
3. split 비율이 val_frac에 맞는지
4. get_lambda_diff가 2-stage 스케줄을 따르는지
5. val_slot_score가 올바른 가중치로 계산되는지
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot import compute_overlap_score
from scripts.train_phase39 import (
    compute_dataset_overlap_scores,
    split_train_val,
    make_sampling_weights,
    get_lambda_diff,
)

pytestmark = pytest.mark.phase39


# =============================================================================
# Mock Dataset
# =============================================================================

class MockDataset:
    """
    실제 파일 없이 entity_masks를 synthetic하게 생성하는 mock dataset.

    overlap_levels: 각 sample의 target overlap level (0.0 ~ 1.0)
    """

    def __init__(self, n_samples: int = 20, n_frames: int = 4, S: int = 16,
                 overlap_levels: list = None, seed: int = 42):
        self.n_samples = n_samples
        rng = np.random.default_rng(seed)

        if overlap_levels is None:
            # 다양한 overlap level의 샘플
            overlap_levels = np.linspace(0.0, 0.9, n_samples).tolist()

        self._masks = []
        for level in overlap_levels:
            m = np.zeros((n_frames, 2, S), dtype=np.float32)
            if level == 0.0:
                # 겹침 없음: e0 왼쪽 절반, e1 오른쪽 절반
                m[:, 0, :S//2]  = 1.0
                m[:, 1, S//2:]  = 1.0
            else:
                # level에 비례한 겹침
                n_overlap = max(1, int(level * S))
                m[:, 0, :S//2 + n_overlap//2] = 1.0
                m[:, 1, S//2 - n_overlap//2:] = 1.0
            self._masks.append(m)

        # dummy frames, depth_orders, meta
        self._frames = [np.zeros((n_frames, 64, 64, 3), dtype=np.uint8)] * n_samples
        self._depth_orders = [[(0, 1)] * n_frames] * n_samples
        self._metas = [{"keyword0": "cat", "keyword1": "dog",
                         "color0": [0.85, 0.15, 0.1],
                         "color1": [0.1, 0.25, 0.85]}] * n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return (self._frames[i], None, self._depth_orders[i],
                self._metas[i], self._masks[i])


# =============================================================================
# Tests: compute_dataset_overlap_scores
# =============================================================================

class TestComputeDatasetOverlapScores:

    def test_output_length_matches_dataset(self):
        """overlap scores 길이가 dataset 크기와 같아야."""
        ds = MockDataset(n_samples=15)
        scores = compute_dataset_overlap_scores(ds)
        assert len(scores) == 15

    def test_scores_are_in_range(self):
        """모든 score가 [0, 1] 범위인지."""
        ds = MockDataset(n_samples=20)
        scores = compute_dataset_overlap_scores(ds)
        assert np.all(scores >= 0.0), "scores should be >= 0"
        assert np.all(scores <= 1.0), "scores should be <= 1"

    def test_high_overlap_gets_high_score(self):
        """겹침이 많은 샘플이 높은 score를 받는지."""
        # 두 극단적 샘플
        ds = MockDataset(n_samples=2,
                         overlap_levels=[0.0, 0.9])
        scores = compute_dataset_overlap_scores(ds)
        assert scores[1] > scores[0], \
            f"high overlap sample should have higher score: {scores}"

    def test_no_overlap_zero_score(self):
        """겹침 없으면 score = 0."""
        ds = MockDataset(n_samples=5, overlap_levels=[0.0] * 5)
        scores = compute_dataset_overlap_scores(ds)
        assert np.all(scores == pytest.approx(0.0, abs=1e-5))

    def test_monotone_with_overlap_level(self):
        """overlap level이 올라갈수록 score가 단조 증가하는지."""
        levels = [0.0, 0.2, 0.4, 0.6, 0.8]
        ds = MockDataset(n_samples=5, overlap_levels=levels)
        scores = compute_dataset_overlap_scores(ds)
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i+1] + 1e-4, \
                f"scores should be monotone: {scores}"


# =============================================================================
# Tests: split_train_val
# =============================================================================

class TestSplitTrainVal:

    def test_val_is_overlap_heavy(self):
        """val set의 평균 overlap이 train set보다 높아야 함."""
        scores = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        train_idx, val_idx = split_train_val(scores, val_frac=0.3)

        val_scores   = scores[val_idx]
        train_scores = scores[train_idx]
        assert val_scores.mean() > train_scores.mean(), \
            f"val overlap {val_scores.mean():.3f} should > train {train_scores.mean():.3f}"

    def test_val_size_respects_frac(self):
        """val set 크기가 val_frac에 맞는지 (min_val 이상)."""
        N = 20
        scores = np.random.rand(N)
        for frac in [0.1, 0.2, 0.3]:
            train_idx, val_idx = split_train_val(scores, val_frac=frac, min_val=2)
            expected_n_val = max(2, int(np.ceil(N * frac)))
            assert len(val_idx) == min(expected_n_val, N - 1), \
                f"val size mismatch for frac={frac}"

    def test_no_overlap_between_train_and_val(self):
        """train과 val이 겹치지 않아야 함."""
        scores = np.random.rand(30)
        train_idx, val_idx = split_train_val(scores, val_frac=0.2)
        assert len(set(train_idx) & set(val_idx)) == 0, \
            "train and val should not overlap"

    def test_covers_all_samples(self):
        """train + val = 전체 dataset."""
        N = 25
        scores = np.random.rand(N)
        train_idx, val_idx = split_train_val(scores, val_frac=0.2, min_val=3)
        all_idx = sorted(train_idx + val_idx)
        assert all_idx == list(range(N)), "all samples should be covered"

    def test_val_sorted_by_overlap_desc(self):
        """val indices는 overlap score 높은 순으로 정렬되어야 함."""
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        train_idx, val_idx = split_train_val(scores, val_frac=0.5, min_val=2)
        val_scores = scores[val_idx]
        for i in range(len(val_scores) - 1):
            assert val_scores[i] >= val_scores[i + 1] - 1e-6, \
                "val should be sorted desc by overlap score"

    def test_min_val_enforced(self):
        """min_val이 보장되는지."""
        scores = np.random.rand(10)
        _, val_idx = split_train_val(scores, val_frac=0.01, min_val=4)
        assert len(val_idx) >= 4, "min_val should be enforced"

    def test_at_least_one_train(self):
        """val이 너무 커지더라도 train에 최소 1개는 남아야 함."""
        scores = np.random.rand(5)
        train_idx, val_idx = split_train_val(scores, val_frac=0.9, min_val=4)
        assert len(train_idx) >= 1, "at least 1 sample should remain in train"


# =============================================================================
# Tests: make_sampling_weights
# =============================================================================

class TestMakeSamplingWeights:

    def test_weights_sum_to_one(self):
        """sampling weights의 합이 1이어야 함."""
        scores = np.array([0.0, 0.3, 0.6, 0.9])
        train_idx = [0, 1, 2, 3]
        w = make_sampling_weights(train_idx, scores)
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_high_overlap_higher_weight(self):
        """overlap 높은 샘플이 더 높은 sampling weight를 가져야 함."""
        scores = np.array([0.0, 0.1, 0.5, 0.9])
        train_idx = [0, 1, 2, 3]
        w = make_sampling_weights(train_idx, scores)
        assert w[3] > w[0], "higher overlap should have higher weight"
        assert w[3] > w[1]

    def test_weights_positive(self):
        """모든 weight가 양수인지."""
        scores = np.array([0.0, 0.3, 0.7])
        w = make_sampling_weights([0, 1, 2], scores)
        assert np.all(w > 0), "all weights should be positive"

    def test_uniform_overlap_uniform_weights(self):
        """모든 overlap이 동일하면 uniform weight."""
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        w = make_sampling_weights([0, 1, 2, 3], scores)
        expected = 0.25
        for wi in w:
            assert wi == pytest.approx(expected, abs=1e-6)

    def test_weighted_sampling_prefers_hard_samples(self):
        """weighted sampling이 실제로 hard sample을 더 자주 선택하는지."""
        rng = np.random.default_rng(42)
        scores = np.array([0.0, 0.0, 0.0, 0.0, 0.9])   # 마지막만 high overlap
        train_idx = [0, 1, 2, 3, 4]
        w = make_sampling_weights(train_idx, scores)

        chosen = rng.choice(len(train_idx), size=1000, replace=True, p=w)
        # hard sample(index 4)이 uniform(20%)보다 많이 선택되어야 함
        frac_hard = (chosen == 4).mean()
        assert frac_hard > 0.2, \
            f"hard sample fraction {frac_hard:.3f} should > 0.2 (uniform)"


# =============================================================================
# Tests: get_lambda_diff (2-stage ramp-up)
# =============================================================================

class TestGetLambdaDiff:

    def test_stage1_returns_zero(self):
        """stage1 동안 lambda_diff = lambda_diff_s1 (default 0.0)."""
        for epoch in range(0, 60):   # 60% stage1
            lam = get_lambda_diff(epoch, 100, stage1_frac=0.6,
                                  lambda_diff_s1=0.0, lambda_diff_s2=0.3)
            assert lam == pytest.approx(0.0, abs=1e-7), \
                f"stage1 epoch {epoch}: lambda_diff should be 0"

    def test_stage2_ramps_up(self):
        """stage2에서 lambda_diff가 단조 증가하는지."""
        lams = [get_lambda_diff(e, 100, stage1_frac=0.6,
                                lambda_diff_s1=0.0, lambda_diff_s2=0.3)
                for e in range(60, 100)]
        for i in range(len(lams) - 1):
            assert lams[i] <= lams[i+1] + 1e-7, \
                f"lambda_diff should be monotone at stage2 epoch {60+i}"

    def test_final_epoch_reaches_target(self):
        """마지막 epoch에서 lambda_diff_s2에 도달하는지."""
        lam = get_lambda_diff(99, 100, stage1_frac=0.6,
                              lambda_diff_s1=0.0, lambda_diff_s2=0.3)
        assert lam == pytest.approx(0.3, abs=1e-6)

    def test_stage_boundary_is_sharp(self):
        """stage1/2 경계 에서 불연속 없이 이어지는지."""
        lam_end_s1 = get_lambda_diff(59, 100, stage1_frac=0.6,
                                     lambda_diff_s1=0.0, lambda_diff_s2=0.3)
        lam_start_s2 = get_lambda_diff(60, 100, stage1_frac=0.6,
                                       lambda_diff_s1=0.0, lambda_diff_s2=0.3)
        assert lam_end_s1 == pytest.approx(0.0, abs=1e-7)
        assert lam_start_s2 >= 0.0   # stage2 시작: 0 이상

    def test_nonzero_s1_respected(self):
        """lambda_diff_s1 > 0으로 설정하면 stage1에서도 그 값이 유지되는지."""
        for epoch in range(0, 50):
            lam = get_lambda_diff(epoch, 100, stage1_frac=0.5,
                                  lambda_diff_s1=0.05, lambda_diff_s2=0.3)
            assert lam == pytest.approx(0.05, abs=1e-7), \
                f"stage1 lambda_diff should be 0.05 at epoch {epoch}"

    def test_total_epochs_1_no_crash(self):
        """edge case: total_epochs=1."""
        lam = get_lambda_diff(0, 1, stage1_frac=0.6,
                              lambda_diff_s1=0.0, lambda_diff_s2=0.3)
        assert isinstance(lam, float)
        assert 0.0 <= lam <= 0.3
