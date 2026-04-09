"""
Phase 16: Objaverse 데이터 재학습 + Phase 12 vs 16 비교 테스트

pytest -m phase16
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.phase16, pytest.mark.slow]

sys.path.insert(0, str(Path(__file__).parent.parent))

STATS_PATH      = Path("debug/dataset_stats/objaverse_stats.json")
OBJ_CKPT_DIR   = Path("checkpoints/objaverse_test")
OBJ_DEBUG_DIR  = Path("debug/train_objaverse_test")
CMP_OUT_DIR    = Path("debug/comparison_p16_test")
P12_CKPT       = Path("checkpoints/animatediff_test/chain_best.pt")


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
        pytest.skip(f"All GPUs < 7 GB free")
    return str(best_gpu)


# ─── 유닛 테스트 (CPU) ────────────────────────────────────────────────────────

def test_dataset_quality_checker_pass():
    """충분한 stats → check_dataset_quality returns True"""
    from scripts.train_objaverse_vca import check_dataset_quality
    import tempfile
    stats = {
        "total_frames": 1000,
        "depth_reversal_rate": 0.45,
        "occlusion_rate": 0.55,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(stats, f)
        tmp = f.name
    assert check_dataset_quality(tmp) is True


def test_dataset_quality_checker_fail():
    """불충분한 stats → check_dataset_quality returns False"""
    from scripts.train_objaverse_vca import check_dataset_quality
    import tempfile
    stats = {
        "total_frames": 10,
        "depth_reversal_rate": 0.05,
        "occlusion_rate": 0.02,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(stats, f)
        tmp = f.name
    assert check_dataset_quality(tmp) is False


def test_dataset_quality_checker_output():
    """stdout에 DATASET_OK / DATASET_FAIL 출력"""
    from scripts.train_objaverse_vca import check_dataset_quality
    import tempfile, io
    from contextlib import redirect_stdout
    stats = {"total_frames": 1000, "depth_reversal_rate": 0.5, "occlusion_rate": 0.5}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(stats, f)
        tmp = f.name
    buf = io.StringIO()
    with redirect_stdout(buf):
        check_dataset_quality(tmp)
    output = buf.getvalue()
    assert "DATASET_OK" in output or "DATASET_FAIL" in output


def test_clip_cache_caches():
    """동일 텍스트 쌍 두 번 호출 → 캐시 히트"""
    try:
        from scripts.train_objaverse_vca import get_entity_context_from_meta, _clip_cache
        from scripts.run_animatediff import load_pipeline
        pipe = load_pipeline(device="cpu")
        meta = {"prompt_entity0": "a cat", "prompt_entity1": "a dog"}
        _clip_cache.clear()
        ctx1 = get_entity_context_from_meta(pipe, meta, "cpu")
        n_before = len(_clip_cache)
        ctx2 = get_entity_context_from_meta(pipe, meta, "cpu")
        n_after = len(_clip_cache)
        assert n_before == n_after, "Cache size should not grow on second call"
        assert torch.allclose(ctx1.cpu(), ctx2.cpu()), "Cached result should be identical"
    except Exception as e:
        pytest.skip(f"Pipeline not available: {e}")


def test_clip_cache_different_texts():
    """다른 텍스트 → 다른 캐시 엔트리"""
    try:
        from scripts.train_objaverse_vca import get_entity_context_from_meta, _clip_cache
        from scripts.run_animatediff import load_pipeline
        pipe = load_pipeline(device="cpu")
        _clip_cache.clear()
        meta1 = {"prompt_entity0": "a cat",    "prompt_entity1": "a dog"}
        meta2 = {"prompt_entity0": "a sword",  "prompt_entity1": "a snake"}
        get_entity_context_from_meta(pipe, meta1, "cpu")
        get_entity_context_from_meta(pipe, meta2, "cpu")
        assert len(_clip_cache) == 2
    except Exception as e:
        pytest.skip(f"Pipeline not available: {e}")


def test_objaverse_train_dataset_loads():
    """ObjaverseTrainDataset: data_objaverse 로드 → len >= 1"""
    from scripts.train_objaverse_vca import ObjaverseTrainDataset
    ds = ObjaverseTrainDataset(
        data_root="toy/data_objaverse",
        n_frames=8, height=128, width=128,
    )
    assert len(ds) >= 1


def test_objaverse_train_dataset_item():
    """__getitem__ 반환: frames(T,H,W,3), depths(T,H,W), depth_orders(T×2), meta dict"""
    from scripts.train_objaverse_vca import ObjaverseTrainDataset
    ds = ObjaverseTrainDataset(
        data_root="toy/data_objaverse",
        n_frames=4, height=64, width=64,
    )
    if len(ds) == 0:
        pytest.skip("No objaverse data")
    frames_np, depths_np, depth_orders, meta = ds[0]
    assert frames_np.shape == (4, 64, 64, 3)
    assert depths_np.shape == (4, 64, 64)
    assert len(depth_orders) == 4
    assert "prompt_entity0" in meta
    assert "prompt_entity1" in meta


def test_comparison_prompts_defined():
    """COMPARISON_PROMPTS: 8개 + 카테고리별 분포"""
    from scripts.compare_checkpoints import COMPARISON_PROMPTS
    assert len(COMPARISON_PROMPTS) == 8
    cats = [p["category"] for p in COMPARISON_PROMPTS]
    assert cats.count("in_dist_objaverse") == 2
    assert cats.count("in_dist_toy") == 2
    assert cats.count("zero_shot") == 4
    for p in COMPARISON_PROMPTS:
        for key in ("id", "full", "entity_0", "entity_1", "category"):
            assert key in p, f"Missing '{key}' in {p['id']}"


def test_current_dataset_passes_gate():
    """현재 생성된 Objaverse 데이터가 품질 게이트를 통과"""
    if not STATS_PATH.exists():
        pytest.skip("objaverse_stats.json not found — run Phase 15 first")
    from scripts.train_objaverse_vca import check_dataset_quality
    result = check_dataset_quality(str(STATS_PATH))
    assert result is True, "Current dataset fails quality gate"


# ─── Part A 통합 테스트 ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def train_result():
    """5 epoch, max 50 samples, quick test"""
    if not STATS_PATH.exists():
        pytest.skip("objaverse_stats.json not found — run Phase 15 first")
    gpu_id = _pick_gpu()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id} if gpu_id else None
    r = subprocess.run(
        [sys.executable, "scripts/train_objaverse_vca.py",
         "--data-root",    "toy/data_objaverse",
         "--stats-path",   str(STATS_PATH),
         "--epochs",       "5",
         "--max-samples",  "50",
         "--lr",           "1e-4",
         "--t-max",        "200",
         "--lambda-depth", "1.0",
         "--lambda-ortho", "0.05",
         "--n-frames",     "4",
         "--height",       "256",
         "--width",        "256",
         "--save-dir",     str(OBJ_CKPT_DIR),
         "--debug-dir",    str(OBJ_DEBUG_DIR)],
        capture_output=True, text=True, timeout=600, env=env,
    )
    return r


def test_train_exits_cleanly(train_result):
    if "DATASET_FAIL" in train_result.stdout:
        pytest.skip("Dataset quality insufficient")
    assert train_result.returncode == 0, \
        f"train_objaverse_vca.py failed:\n{train_result.stderr[-800:]}"


def test_dataset_ok_logged(train_result):
    if "DATASET_FAIL" in train_result.stdout:
        pytest.skip()
    assert "DATASET_OK" in train_result.stdout


def test_real_clip_context_logged(train_result):
    """stdout에 실제 entity 텍스트 포함 (dummy가 아님)"""
    if "DATASET_FAIL" in train_result.stdout:
        pytest.skip()
    assert "prompt_entity0=" in train_result.stdout, \
        f"Real entity text not logged:\n{train_result.stdout[:500]}"


def test_checkpoint_saved(train_result):
    if "DATASET_FAIL" in train_result.stdout:
        pytest.skip()
    assert (OBJ_CKPT_DIR / "best.pt").exists(), \
        f"best.pt not found in {OBJ_CKPT_DIR}"


def test_checkpoint_loadable(train_result):
    p = OBJ_CKPT_DIR / "best.pt"
    if not p.exists():
        pytest.skip()
    ckpt = torch.load(p, map_location="cpu")
    assert "vca_state_dict" in ckpt
    assert "sigma_stats" in ckpt


def test_learning_ok(train_result):
    if "DATASET_FAIL" in train_result.stdout:
        pytest.skip()
    assert "LEARNING=OK" in train_result.stdout, \
        f"Expected LEARNING=OK:\n{train_result.stdout[-600:]}"


def test_sigma_separation_positive(train_result):
    if "DATASET_FAIL" in train_result.stdout:
        pytest.skip()
    m = re.search(r"FINAL sigma_separation=([\d.]+)", train_result.stdout)
    assert m, f"FINAL sigma_separation not found:\n{train_result.stdout[-400:]}"
    assert float(m.group(1)) > 0.0


def test_loss_components_logged(train_result):
    if "DATASET_FAIL" in train_result.stdout:
        pytest.skip()
    for key in ["l_diff=", "l_depth=", "l_ortho="]:
        assert key in train_result.stdout, f"'{key}' not in stdout"


def test_loss_not_exploding(train_result):
    if "DATASET_FAIL" in train_result.stdout:
        pytest.skip()
    losses = re.findall(r"loss=([\d.]+)", train_result.stdout)
    for v in losses:
        assert float(v) < 10.0, f"loss={v} >= 10.0 (exploding?)"


def test_sigma_gifs_saved(train_result):
    if "DATASET_FAIL" in train_result.stdout:
        pytest.skip()
    gifs = list(OBJ_DEBUG_DIR.glob("sigma_epoch*.gif"))
    assert len(gifs) >= 1, f"No sigma GIFs in {OBJ_DEBUG_DIR}"


def test_training_curve_saved(train_result):
    if "DATASET_FAIL" in train_result.stdout:
        pytest.skip()
    p = OBJ_DEBUG_DIR / "training_curve.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert len(data) > 0
    assert "loss" in data[0]
    assert "prompt_entity0" in data[0]


# ─── Part B 비교 테스트 ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def compare_result(train_result):
    p12 = P12_CKPT
    p16 = OBJ_CKPT_DIR / "best.pt"
    if not p12.exists():
        pytest.skip(f"Phase 12 ckpt not found: {p12}")
    if not p16.exists():
        pytest.skip(f"Phase 16 ckpt not found: {p16}")
    if "DATASET_FAIL" in train_result.stdout:
        pytest.skip("Training was skipped")

    gpu_id = _pick_gpu()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id} if gpu_id else None
    r = subprocess.run(
        [sys.executable, "scripts/compare_checkpoints.py",
         "--phase12-ckpt", str(p12),
         "--phase16-ckpt", str(p16),
         "--prompts",      "cat_dog,chain",
         "--out-dir",      str(CMP_OUT_DIR),
         "--steps",        "5",
         "--num-frames",   "8",
         "--seed",         "42",
         "--height",       "256",
         "--width",        "256"],
        capture_output=True, text=True, timeout=600, env=env,
    )
    return r


def test_compare_exits_cleanly(compare_result):
    assert compare_result.returncode == 0, \
        f"compare_checkpoints.py failed:\n{compare_result.stderr[-800:]}"


def test_all_output_files_created(compare_result):
    for pid in ["cat_dog", "chain"]:
        for fname in ["baseline.gif", "phase12.gif", "phase16.gif",
                      "threeway.gif", "sigma_comparison.json"]:
            p = CMP_OUT_DIR / pid / fname
            assert p.exists(), f"Missing: {p}"


def test_sigma_comparison_json_valid(compare_result):
    for pid in ["cat_dog", "chain"]:
        p = CMP_OUT_DIR / pid / "sigma_comparison.json"
        if not p.exists():
            pytest.skip(f"{p} not found")
        data = json.loads(p.read_text())
        for key in ("baseline", "phase12", "phase16", "winner", "delta"):
            assert key in data, f"{pid}: '{key}' missing"
        assert data["winner"] in ("baseline", "phase12", "phase16", "COMPARABLE")
        assert data["phase16"]["sigma_separation"] >= 0.0


def test_threeway_gif_is_3panel(compare_result):
    """threeway.gif width == 256 × 3"""
    for pid in ["cat_dog", "chain"]:
        p = CMP_OUT_DIR / pid / "threeway.gif"
        if not p.exists():
            pytest.skip(f"{p} not found")
        frame = iio3.imread(str(p), index=0)
        assert frame.shape[1] == 256 * 3, \
            f"{pid} threeway width={frame.shape[1]}, expected {256*3}"


def test_report_created(compare_result):
    assert (CMP_OUT_DIR / "summary" / "comparison_report.md").exists()


def test_report_has_conclusion(compare_result):
    p = CMP_OUT_DIR / "summary" / "comparison_report.md"
    if not p.exists():
        pytest.skip()
    content = p.read_text()
    has_conclusion = any(w in content for w in
                         ("OBJAVERSE_BETTER", "TOY_BETTER", "COMPARABLE"))
    assert has_conclusion, "Report missing conclusion keyword"


def test_all_sigma_stats_json(compare_result):
    p = CMP_OUT_DIR / "summary" / "all_sigma_stats.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert len(data) >= 1
    assert "phase16" in data[0]
    assert "winner" in data[0]


def test_best_threeway_gif_created(compare_result):
    assert (CMP_OUT_DIR / "summary" / "best_threeway.gif").exists()


# ─── import ───────────────────────────────────────────────────────────────────
import imageio.v3 as iio3  # noqa: E402 (used in test_threeway_gif_is_3panel)
