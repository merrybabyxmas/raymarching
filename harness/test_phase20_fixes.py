"""
Phase 20: Fix 1+2+3 (단일 mid_block 주입) 테스트

pytest -m phase20
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.phase20, pytest.mark.slow]

sys.path.insert(0, str(Path(__file__).parent.parent))

STATS_PATH    = Path("debug/dataset_stats/objaverse_stats.json")
P20_CKPT_DIR  = Path("checkpoints/phase20_test")
P20_DEBUG_DIR = Path("debug/train_phase20_test")


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


# ─── Fix 1: per-frame depth loss (Phase 19에서 재사용) ──────────────────────

def test_perframe_loss_differs_from_majority_vote():
    """Fix 1 유지: per-frame loss ≠ majority vote loss."""
    from scripts.train_phase19 import l_depth_ranking_perframe
    from models.losses import l_depth_ranking

    torch.manual_seed(0)
    sigma = torch.rand(8, 64, 2, 2, requires_grad=True)
    depth_orders = [[0, 1]] * 4 + [[1, 0]] * 4

    perframe = l_depth_ranking_perframe([sigma], depth_orders)
    majority = l_depth_ranking(sigma, [0, 1])

    assert not torch.isclose(perframe, majority, atol=1e-4), \
        "Per-frame loss should differ from majority-vote when orders are mixed"


def test_perframe_loss_gradient_flows():
    from scripts.train_phase19 import l_depth_ranking_perframe

    sigma = torch.rand(4, 16, 2, 2, requires_grad=True)
    loss = l_depth_ranking_perframe([sigma], [[0,1],[1,0],[0,1],[1,0]])
    loss.backward()
    assert sigma.grad is not None
    assert not torch.all(sigma.grad == 0)


# ─── Fix 2: 고정 probe ───────────────────────────────────────────────────────

def test_probe_t_values_reused():
    """Fix 2 유지: PROBE_T_VALUES가 Phase 19와 동일."""
    from scripts.train_phase19 import PROBE_T_VALUES
    assert len(PROBE_T_VALUES) >= 3
    assert min(PROBE_T_VALUES) >= 0
    assert max(PROBE_T_VALUES) <= 200


# ─── Fix 3: depth_pe init scale ──────────────────────────────────────────────

def test_depth_pe_init_scale_p20():
    """Fix 3 유지: Phase 20 depth_pe_init_scale=0.3."""
    from scripts.train_phase20 import DEPTH_PE_INIT_SCALE
    assert DEPTH_PE_INIT_SCALE >= 0.1, \
        f"depth_pe_init_scale={DEPTH_PE_INIT_SCALE} too small"


def test_vcalayer_p20_init_scale():
    """VCALayer depth_pe std가 0.3 scale로 충분히 큼."""
    from models.vca_attention import VCALayer
    vca = VCALayer(query_dim=64, context_dim=128, depth_pe_init_scale=0.3)
    std = float(vca.depth_pe.std())
    # std ≈ 0.3 * std(randn) ≈ 0.3 × 1 = 0.3, check > 5× Phase 1~18's 0.02
    assert std > 0.05, f"depth_pe std={std:.4f} unexpectedly small"


# ─── Fix 4 제거: mid_block 단일 주입 확인 ────────────────────────────────────

def test_inject_key_is_mid_block():
    """Phase 20은 mid_block 단일 주입."""
    from scripts.train_phase20 import INJECT_KEY_P20
    assert 'mid_block' in INJECT_KEY_P20
    assert 'attn2' in INJECT_KEY_P20


def test_inject_key_is_single():
    """INJECT_KEY_P20은 str (단일 키)."""
    from scripts.train_phase20 import INJECT_KEY_P20
    assert isinstance(INJECT_KEY_P20, str), "Phase 20 uses single injection key (str)"


def test_sigma_acc_single_entry():
    """단일 주입 시 sigma_acc는 forward 후 1개 원소."""
    from models.vca_attention import VCALayer

    vca = VCALayer(query_dim=64, context_dim=128, n_heads=4, n_entities=2, z_bins=2)
    x   = torch.randn(2, 16, 64)
    ctx = torch.randn(2, 2, 128)

    assert len(vca.sigma_acc) == 0
    vca(x, ctx)
    assert len(vca.sigma_acc) == 1   # single-layer injection: 1 entry
    vca.reset_sigma_acc()
    assert len(vca.sigma_acc) == 0


# ─── subprocess 통합 테스트 ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def train_result():
    if not STATS_PATH.exists():
        pytest.skip("objaverse_stats.json not found")
    gpu_id = _pick_gpu()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id} if gpu_id else None
    r = subprocess.run(
        [sys.executable, "scripts/train_phase20.py",
         "--data-root",    "toy/data_objaverse",
         "--stats-path",   str(STATS_PATH),
         "--epochs",       "5",
         "--max-samples",  "50",
         "--lr",           "5e-5",
         "--lambda-depth", "0.3",
         "--save-dir",     str(P20_CKPT_DIR),
         "--debug-dir",    str(P20_DEBUG_DIR)],
        capture_output=True, text=True, timeout=1800, env=env,
    )
    return r


def test_train_exits_cleanly(train_result):
    assert train_result.returncode == 0, \
        f"train_phase20.py failed:\n{train_result.stderr[-1000:]}"


def test_dataset_ok(train_result):
    assert "DATASET_OK" in train_result.stdout


def test_inject_1_layer(train_result):
    """단일 레이어 주입 확인."""
    m = re.search(r'inject_p20\] (\d+) layer', train_result.stdout)
    assert m, "inject_p20 log not found"
    assert int(m.group(1)) == 1, f"Expected 1 layer, got {m.group(1)}"


def test_probe_sep_logged(train_result):
    """probe_sep= 가 매 epoch 출력됨."""
    seps = re.findall(r'probe_sep=([\d.]+)', train_result.stdout)
    assert len(seps) >= 5, f"Expected 5 probe_sep logs, got {len(seps)}"


def test_probe_sep_stable(train_result):
    """Fix 2: probe_sep 분산이 Phase 16~18 대비 작아야 함."""
    seps = [float(x) for x in re.findall(r'probe_sep=([\d.]+)', train_result.stdout)]
    if len(seps) < 3:
        pytest.skip("Not enough probe_sep values")
    sep_range = max(seps) - min(seps)
    assert sep_range < 0.4, \
        f"probe_sep range={sep_range:.3f} too high (unstable measurement)"


def test_ratio_sane(train_result):
    """ratio > 2.0: per-frame loss가 작동 중."""
    ratios = re.findall(r'ratio=([\d.]+)x', train_result.stdout)
    assert len(ratios) > 0
    for r in ratios:
        assert float(r) >= 2.0, f"ratio={r} < 2.0 — l_depth dominating l_diff"


def test_loss_components_logged(train_result):
    for key in ["l_diff=", "l_depth=", "l_ortho="]:
        assert key in train_result.stdout


def test_checkpoint_saved(train_result):
    assert (P20_CKPT_DIR / "best.pt").exists()


def test_checkpoint_single_layer_flag(train_result):
    """체크포인트에 multi_layer=False 기록됨."""
    p = P20_CKPT_DIR / "best.pt"
    if not p.exists():
        pytest.skip()
    ckpt = torch.load(p, map_location="cpu")
    assert ckpt.get("multi_layer") is False
    assert "inject_key" in ckpt
    assert 'mid_block' in ckpt["inject_key"]


def test_learning_ok(train_result):
    assert "LEARNING=OK" in train_result.stdout, \
        f"Expected LEARNING=OK:\n{train_result.stdout[-500:]}"


def test_gif_saved(train_result):
    gifs = list(P20_DEBUG_DIR.glob("sigma_epoch*.gif"))
    assert len(gifs) >= 1
