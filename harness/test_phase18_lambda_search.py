"""
Phase 18: λ_depth 그리드 탐색 테스트

pytest -m phase18
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.phase18, pytest.mark.slow]

sys.path.insert(0, str(Path(__file__).parent.parent))

STATS_PATH   = Path("debug/dataset_stats/objaverse_stats.json")
GRID_SAVE    = Path("checkpoints/phase18")
GRID_DEBUG   = Path("debug/train_phase18")
LAMBDA_GRID  = [0.1, 0.2, 0.3]


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


# ─── 유닛 테스트 (CPU) ────────────────────────────────────────────────────────

def test_lambda_grid_values():
    """LAMBDA_GRID가 Phase 16(1.0)과 Phase 17(0.02) 사이 값으로 구성됨"""
    from scripts.train_phase18 import LAMBDA_GRID
    assert all(0.02 < lam < 1.0 for lam in LAMBDA_GRID), \
        f"All λ values should be between 0.02 and 1.0, got {LAMBDA_GRID}"
    assert len(LAMBDA_GRID) >= 3, "Need at least 3 λ values to search"


def test_summarize_grid_picks_best():
    """summarize_grid가 sep 기준 최고 λ 선택"""
    import tempfile
    from scripts.train_phase18 import summarize_grid
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for lam, sep in [(0.1, 0.15), (0.2, 0.31), (0.3, 0.22)]:
            d = root / f"lambda_{lam:.1f}"
            d.mkdir()
            torch.save({"sigma_separation": sep, "epoch": 29}, d / "best.pt")
        result = summarize_grid(root)
    assert result["best_lambda"] == 0.2, \
        f"Expected best_lambda=0.2, got {result['best_lambda']}"


def test_summarize_grid_empty():
    """체크포인트 없으면 best_lambda=None"""
    import tempfile
    from scripts.train_phase18 import summarize_grid
    with tempfile.TemporaryDirectory() as tmp:
        result = summarize_grid(Path(tmp))
    assert result["best_lambda"] is None


def test_ratio_invariant_holds_for_grid():
    """λ=0.1~0.3에서도 l_diff > l_depth_weighted × 10 유지 가능한지 이론 검증"""
    from scripts.train_phase18 import LAMBDA_GRID
    l_diff_typical = 0.4      # 전형적인 diffusion loss
    l_depth_raw_typical = 0.1  # 전형적인 raw depth ranking loss
    for lam in LAMBDA_GRID:
        l_depth_w = lam * l_depth_raw_typical
        ratio = l_diff_typical / max(l_depth_w, 1e-9)
        assert ratio >= 10, \
            f"λ={lam}: ratio={ratio:.1f}x < 10 — l_depth dominates even at start"


# ─── subprocess 통합 테스트 (λ=0.1, 5 epoch 빠른 검증) ──────────────────────

@pytest.fixture(scope="module")
def quick_train_01():
    """λ=0.1, 5 epoch, 50 samples — 빠른 검증"""
    if not STATS_PATH.exists():
        pytest.skip("objaverse_stats.json not found")
    gpu_id = _pick_gpu()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id} if gpu_id else None
    save_dir  = GRID_SAVE / "lambda_0.1_test"
    debug_dir = GRID_DEBUG / "lambda_0.1_test"
    r = subprocess.run(
        [sys.executable, "scripts/train_phase18.py",
         "--data-root",    "toy/data_objaverse",
         "--stats-path",   str(STATS_PATH),
         "--lambda-depth", "0.1",
         "--epochs",       "5",
         "--max-samples",  "50",
         "--lr",           "5e-5",
         "--save-dir",     str(save_dir),
         "--debug-dir",    str(debug_dir)],
        capture_output=True, text=True, timeout=1800, env=env,
    )
    return r


def test_quick_train_exits_cleanly(quick_train_01):
    assert quick_train_01.returncode == 0, \
        f"train_phase18.py failed:\n{quick_train_01.stderr[-800:]}"


def test_quick_train_dataset_ok(quick_train_01):
    assert "DATASET_OK" in quick_train_01.stdout


def test_quick_train_ratio_sane(quick_train_01):
    ratios = re.findall(r'ratio=([\d.]+)x', quick_train_01.stdout)
    assert len(ratios) > 0
    for r in ratios:
        assert float(r) >= 5.0, f"ratio={r} < 5.0"


def test_quick_train_ldiff_dominates(quick_train_01):
    diffs  = [float(x) for x in re.findall(r'l_diff=([\d.]+)', quick_train_01.stdout)]
    depths = [float(x) for x in re.findall(r'l_depth=([\d.]+)', quick_train_01.stdout)]
    assert len(diffs) > 0 and len(depths) > 0
    for d, p in zip(diffs, depths):
        if d > 0 and p > 0:
            assert d > p * 5, f"l_diff={d:.4f} should be > l_depth_w={p:.4f} × 5"


def test_quick_train_checkpoint_saved(quick_train_01):
    assert (GRID_SAVE / "lambda_0.1_test" / "best.pt").exists()


def test_quick_train_learning_ok(quick_train_01):
    assert "LEARNING=OK" in quick_train_01.stdout, \
        f"Expected LEARNING=OK:\n{quick_train_01.stdout[-400:]}"


def test_quick_train_gif_saved(quick_train_01):
    gifs = list((GRID_DEBUG / "lambda_0.1_test").glob("sigma_epoch*.gif"))
    assert len(gifs) >= 1


# ─── 그리드 탐색 결과 분석 (full grid 완료 후) ───────────────────────────────

@pytest.fixture(scope="module")
def grid_summary():
    """checkpoints/phase18/ 아래 3개 λ 결과 요약"""
    missing = [lam for lam in LAMBDA_GRID
               if not (GRID_SAVE / f"lambda_{lam:.1f}" / "best.pt").exists()]
    if missing:
        pytest.skip(f"Full grid not complete, missing λ={missing}")
    from scripts.train_phase18 import summarize_grid
    return summarize_grid(GRID_SAVE)


def test_grid_has_three_results(grid_summary):
    assert len(grid_summary["results"]) == 3


def test_grid_all_learning_ok(grid_summary):
    """모든 λ에서 sep > 0.01"""
    for r in grid_summary["results"]:
        assert r["sep"] > 0.01, \
            f"λ={r['lambda']}: sep={r['sep']:.4f} ≤ 0.01 (LEARNING=FAIL)"


def test_grid_best_lambda_beats_phase17_at_30ep(grid_summary):
    """그리드 winner sep > Phase 17 @ 30 epoch (0.208).
    Phase 17 full=0.263은 60 epoch 기준 — 공정 비교는 동일 epoch 수."""
    best = grid_summary["best_lambda"]
    best_sep = next(r["sep"] for r in grid_summary["results"] if r["lambda"] == best)
    assert best_sep > 0.208, \
        f"Best λ={best} sep={best_sep:.4f} should beat Phase 17 @ 30ep (0.208)"


def test_grid_best_lambda_in_range(grid_summary):
    """winner가 LAMBDA_GRID 안에 있음"""
    from scripts.train_phase18 import LAMBDA_GRID
    assert grid_summary["best_lambda"] in LAMBDA_GRID


def test_grid_report_logged(grid_summary):
    """summary JSON 저장 확인"""
    report_path = GRID_SAVE / "grid_summary.json"
    if not report_path.exists():
        pytest.skip("grid_summary.json not written yet")
    with open(report_path) as f:
        data = json.load(f)
    assert "best_lambda" in data
    assert "results" in data
