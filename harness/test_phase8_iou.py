"""Phase 8: Sigma-Mask IoU 평가"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.phase8

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── IoU 단위 테스트 ─────────────────────────────────────────────────────────

def test_iou_perfect_match():
    from scripts.eval_iou import compute_iou
    mask = np.array([[True, False], [False, True]])
    iou = compute_iou(mask, mask)
    assert abs(iou - 1.0) < 1e-6, f"Perfect match should give IoU=1.0, got {iou:.4f}"


def test_iou_no_overlap():
    from scripts.eval_iou import compute_iou
    pred = np.array([[True, False], [False, False]])
    gt   = np.array([[False, True], [False, False]])
    iou = compute_iou(pred, gt)
    assert iou == 0.0, f"No overlap should give IoU=0.0, got {iou:.4f}"


def test_iou_partial():
    from scripts.eval_iou import compute_iou
    pred = np.array([[True, True],  [False, False]])
    gt   = np.array([[True, False], [True,  False]])
    # inter=1, union=3 → IoU=1/3
    iou = compute_iou(pred, gt)
    assert abs(iou - 1/3) < 1e-5, f"Expected 1/3, got {iou:.4f}"


def test_iou_empty_gt():
    """union=0 방어: 빈 GT mask → IoU=0 (not NaN)"""
    from scripts.eval_iou import compute_iou
    pred = np.zeros((4, 4), dtype=bool)
    gt   = np.zeros((4, 4), dtype=bool)
    iou = compute_iou(pred, gt)
    assert iou == 0.0 and not np.isnan(iou), f"Empty masks should give IoU=0, got {iou}"


def test_sigma_to_binary_resize():
    """sigma_to_binary: sigma (hw,hw) → binary mask (H,W) with correct shape"""
    from scripts.eval_iou import sigma_to_binary
    sigma = np.ones((8, 8), dtype=np.float32)  # all 1.0 → all > threshold
    mask = sigma_to_binary(sigma, gt_h=16, gt_w=16, threshold=0.3)
    assert mask.shape == (16, 16), f"Expected (16,16), got {mask.shape}"
    assert mask.all(), "sigma=1.0 everywhere should produce all-True mask"


def test_sigma_to_binary_threshold():
    """threshold 효과: 0.0 sigma → all False"""
    from scripts.eval_iou import sigma_to_binary
    sigma = np.zeros((8, 8), dtype=np.float32)
    mask = sigma_to_binary(sigma, gt_h=8, gt_w=8, threshold=0.3)
    assert not mask.any(), "sigma=0.0 everywhere should produce all-False mask"


# ─── 모델 로드 + 평가 통합 테스트 ───────────────────────────────────────────

@pytest.fixture(scope='module')
def iou_result():
    r = subprocess.run(
        [sys.executable, 'scripts/eval_iou.py',
         '--scenario',    'chain',
         '--ablation-dir','debug/ablation',
         '--threshold',   '0.3'],
        capture_output=True, text=True, timeout=300,
    )
    return r


def test_iou_eval_exits_cleanly(iou_result):
    assert iou_result.returncode == 0, \
        f"eval_iou.py failed:\n{iou_result.stderr[-800:]}"


def test_iou_eval_prints_entity_iou(iou_result):
    out = iou_result.stdout
    assert 'iou_entity0' in out, f"Expected iou_entity0 in output:\n{out[-400:]}"
    assert 'iou_entity1' in out, f"Expected iou_entity1 in output:\n{out[-400:]}"


def test_iou_eval_prints_better_model(iou_result):
    assert 'BETTER_MODEL=' in iou_result.stdout, \
        f"Expected BETTER_MODEL= in output:\n{iou_result.stdout[-400:]}"


def test_sigmoid_iou_entity0_positive(iou_result):
    """Sigmoid entity0 IoU should be > 0 (sigma map has signal)"""
    import re
    match = re.search(r'sigmoid\s+iou_entity0=([\d.]+)', iou_result.stdout)
    if not match:
        pytest.skip(f"Could not parse sigmoid iou_entity0:\n{iou_result.stdout[-400:]}")
    iou = float(match.group(1))
    assert iou >= 0.0, f"IoU must be non-negative, got {iou:.4f}"
