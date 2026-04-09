"""
Phase 17: λ 재조정 재학습 테스트

pytest -m phase17
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.phase17, pytest.mark.slow]

sys.path.insert(0, str(Path(__file__).parent.parent))

STATS_PATH    = Path("debug/dataset_stats/objaverse_stats.json")
P17_CKPT_DIR  = Path("checkpoints/phase17_test")
P17_DEBUG_DIR = Path("debug/train_phase17_test")
P16_CKPT      = Path("checkpoints/objaverse/best.pt")


# ─── GPU 선택 ─────────────────────────────────────────────────────────────────

def _pick_gpu() -> str:
    if not torch.cuda.is_available():
        return ""
    best_gpu, best_free = 0, 0
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free = free
                best_gpu = i
        except Exception:
            continue
    if best_free < 7 * 1024 ** 3:
        pytest.skip("All GPUs < 7 GB free")
    return str(best_gpu)


# ─── 유닛 테스트 (CPU, subprocess 없음) ──────────────────────────────────────

def test_lambda_defaults_reduced():
    """Phase 17 기본값이 Phase 16보다 훨씬 작음"""
    from scripts.train_phase17 import (
        DEFAULT_LAMBDA_DEPTH, DEFAULT_LAMBDA_ORTHO, DEFAULT_LR,
    )
    assert DEFAULT_LAMBDA_DEPTH <= 0.05, \
        f"λ_depth={DEFAULT_LAMBDA_DEPTH} should be ≤ 0.05 (Phase 16 was 1.0)"
    assert DEFAULT_LAMBDA_ORTHO <= 0.01, \
        f"λ_ortho={DEFAULT_LAMBDA_ORTHO} should be ≤ 0.01"
    assert DEFAULT_LR <= 1e-4, \
        f"lr={DEFAULT_LR} should be ≤ 1e-4"


def test_adaptive_lambda_reduces_on_high_ratio():
    """l_depth/l_diff > 0.1이면 lambda 줄어듦"""
    from scripts.train_phase17 import adaptive_lambda_depth
    new_lam = adaptive_lambda_depth(
        l_diff_val=0.40, l_depth_val=0.05,  # ratio=0.125 > 0.1
        current_lambda=0.02,
    )
    assert new_lam < 0.02, f"Lambda should decrease, got {new_lam}"


def test_adaptive_lambda_stable_on_low_ratio():
    """l_depth/l_diff < 0.1이면 lambda 유지"""
    from scripts.train_phase17 import adaptive_lambda_depth
    new_lam = adaptive_lambda_depth(
        l_diff_val=0.40, l_depth_val=0.02,  # ratio=0.05 < 0.1
        current_lambda=0.02,
    )
    assert new_lam == 0.02, f"Lambda should stay at 0.02, got {new_lam}"


def test_adaptive_lambda_min_bound():
    """lambda는 min_lambda 아래로 내려가지 않음"""
    from scripts.train_phase17 import adaptive_lambda_depth
    new_lam = adaptive_lambda_depth(
        l_diff_val=0.01, l_depth_val=0.01,
        current_lambda=1e-4, min_lambda=1e-4,
    )
    assert new_lam >= 1e-4


def test_check_generation_quality_ok():
    """pixel_var 충분, sigma_max 안전 → status=OK"""
    from scripts.train_phase17 import check_generation_quality_mock
    status = check_generation_quality_mock(pixel_var=1200.0, sigma_max=0.70)
    assert status == "OK"


def test_check_generation_quality_degraded_low_var():
    """pixel_var 너무 낮음 → status=DEGRADED"""
    from scripts.train_phase17 import check_generation_quality_mock
    status = check_generation_quality_mock(pixel_var=40.0, sigma_max=0.70)
    assert status == "DEGRADED"


def test_check_generation_quality_degraded_high_sigma():
    """sigma_max 극단값 → status=DEGRADED"""
    from scripts.train_phase17 import check_generation_quality_mock
    status = check_generation_quality_mock(pixel_var=1200.0, sigma_max=0.97)
    assert status == "DEGRADED"


def test_full_dataset_used():
    """max_samples=None → 전체 168 샘플 사용"""
    from scripts.train_objaverse_vca import ObjaverseTrainDataset
    if not Path("toy/data_objaverse").exists():
        pytest.skip("toy/data_objaverse not found")
    ds = ObjaverseTrainDataset(
        data_root="toy/data_objaverse",
        n_frames=4, height=64, width=64,
        max_samples=None,
    )
    assert len(ds) == 168, \
        f"Expected 168 samples (full dataset), got {len(ds)}"


def test_ldiff_dominates_ldepth():
    """DEFAULT_LAMBDA_DEPTH 수준에서 ratio > 10 유지 검증 (mock)"""
    from scripts.train_phase17 import DEFAULT_LAMBDA_DEPTH
    l_diff_expected  = 0.4
    l_depth_expected = DEFAULT_LAMBDA_DEPTH * 0.05  # 전형적인 raw l_depth
    ratio = l_diff_expected / max(l_depth_expected, 1e-9)
    assert ratio >= 10, \
        f"Expected ratio ≥ 10, got {ratio:.1f}x — λ_depth too large"


# ─── subprocess 통합 테스트 ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def train_result():
    """5 epoch, 50 samples — 빠른 검증"""
    if not STATS_PATH.exists():
        pytest.skip("objaverse_stats.json not found")
    gpu_id = _pick_gpu()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id} if gpu_id else None
    r = subprocess.run(
        [sys.executable, "scripts/train_phase17.py",
         "--data-root",    "toy/data_objaverse",
         "--stats-path",   str(STATS_PATH),
         "--epochs",       "5",
         "--max-samples",  "50",
         "--lr",           "5e-5",
         "--lambda-depth", "0.02",
         "--lambda-ortho", "0.005",
         "--t-max",        "200",
         "--save-dir",     str(P17_CKPT_DIR),
         "--debug-dir",    str(P17_DEBUG_DIR)],
        capture_output=True, text=True, timeout=1800, env=env,
    )
    return r


def test_train_exits_cleanly(train_result):
    assert train_result.returncode == 0, \
        f"train_phase17.py failed:\n{train_result.stderr[-1000:]}"


def test_dataset_ok_logged(train_result):
    assert "DATASET_OK" in train_result.stdout, \
        f"DATASET_OK not found:\n{train_result.stdout[:500]}"


def test_ratio_logged_and_sane(train_result):
    """ratio= 가 출력되고 항상 > 5 (생성 품질 보존 최소 기준)"""
    ratios = re.findall(r'ratio=([\d.]+)x', train_result.stdout)
    assert len(ratios) > 0, "ratio= not logged"
    for r in ratios:
        assert float(r) >= 5.0, \
            f"ratio={r} < 5.0 — l_depth dominating l_diff!"


def test_no_degraded_in_early_epochs(train_result):
    """5 epoch 빠른 테스트에서 DEGRADED 없어야 함 (λ 작으니 정상)"""
    assert "status=DEGRADED" not in train_result.stdout, \
        f"Generation quality degraded in early epochs:\n{train_result.stdout}"


def test_loss_components_logged(train_result):
    for key in ["l_diff=", "l_depth=", "l_ortho="]:
        assert key in train_result.stdout, f"{key} not found in stdout"


def test_ldiff_larger_than_ldepth_in_output(train_result):
    """stdout에서 l_diff > l_depth(가중) × 5 확인.
    출력된 l_depth는 lambda_depth × raw — 실제 손실 기여분."""
    diffs  = [float(x) for x in re.findall(r'l_diff=([\d.]+)', train_result.stdout)]
    depths = [float(x) for x in re.findall(r'l_depth=([\d.]+)', train_result.stdout)]
    assert len(diffs) > 0 and len(depths) > 0
    for d, p in zip(diffs, depths):
        if d > 0 and p > 0:
            assert d > p * 5, \
                f"l_diff={d:.4f} should be > l_depth_weighted={p:.4f} × 5 (ratio={d/p:.1f}x)"


def test_checkpoint_saved(train_result):
    assert (P17_CKPT_DIR / "best.pt").exists(), \
        f"best.pt not found in {P17_CKPT_DIR}"


def test_checkpoint_has_required_keys(train_result):
    p = P17_CKPT_DIR / "best.pt"
    if not p.exists():
        pytest.skip("best.pt not found")
    ckpt = torch.load(p, map_location="cpu")
    assert "vca_state_dict"     in ckpt
    assert "lambda_depth_final" in ckpt


def test_sigma_separation_positive(train_result):
    m = re.search(r"FINAL sigma_separation=([\d.]+)", train_result.stdout)
    assert m, f"FINAL sigma_separation not found:\n{train_result.stdout[-400:]}"
    assert float(m.group(1)) > 0.0


def test_learning_ok(train_result):
    assert "LEARNING=OK" in train_result.stdout, \
        f"Expected LEARNING=OK:\n{train_result.stdout[-600:]}"


def test_sigma_gifs_saved(train_result):
    gifs = list(P17_DEBUG_DIR.glob("sigma_epoch*.gif"))
    assert len(gifs) >= 1, f"No sigma GIFs in {P17_DEBUG_DIR}"


# ─── Phase 16 vs Phase 17 시각적 비교 ─────────────────────────────────────────

@pytest.fixture(scope="module")
def visual_compare_result(train_result):
    """
    full train 체크포인트(checkpoints/phase17/best.pt)가 있으면
    cat_dog / fighters 프롬프트로 Phase16 vs Phase17 비교 GIF 생성.
    """
    p17_ckpt = Path("checkpoints/phase17/best.pt")
    p16_ckpt = P16_CKPT
    if not p17_ckpt.exists():
        pytest.skip("Phase 17 full checkpoint not found — run full training first")
    if not p16_ckpt.exists():
        pytest.skip("Phase 16 checkpoint not found")

    gpu_id = _pick_gpu()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id} if gpu_id else None
    r = subprocess.run(
        [sys.executable, "scripts/compare_checkpoints.py",
         "--phase12-ckpt", str(p16_ckpt),
         "--phase16-ckpt", str(p17_ckpt),
         "--prompts",      "cat_dog,fighters",
         "--out-dir",      "debug/comparison_p17",
         "--steps",        "20",
         "--num-frames",   "16",
         "--seed",         "42",
         "--height",       "256",
         "--width",        "256"],
        capture_output=True, text=True, timeout=600, env=env,
    )
    return r


def test_visual_compare_exits_cleanly(visual_compare_result):
    assert visual_compare_result.returncode == 0, \
        visual_compare_result.stderr[-600:]


def test_threeway_gifs_created(visual_compare_result):
    for pid in ["cat_dog", "fighters"]:
        p = Path(f"debug/comparison_p17/{pid}/threeway.gif")
        assert p.exists(), f"threeway.gif not found: {p}"
