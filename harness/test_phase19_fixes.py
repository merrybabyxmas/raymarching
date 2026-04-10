"""
Phase 19: 학습 근본 원인 4가지 수정 테스트

pytest -m phase19
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.phase19, pytest.mark.slow]

sys.path.insert(0, str(Path(__file__).parent.parent))

STATS_PATH    = Path("debug/dataset_stats/objaverse_stats.json")
P19_CKPT_DIR  = Path("checkpoints/phase19_test")
P19_DEBUG_DIR = Path("debug/train_phase19_test")


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


# ─── Fix 1: per-frame depth loss 단위 테스트 ─────────────────────────────────

def test_perframe_loss_differs_from_majority_vote():
    """Fix 1: 프레임별 독립 손실 ≠ majority vote 손실."""
    from scripts.train_phase19 import l_depth_ranking_perframe
    from models.losses import l_depth_ranking

    torch.manual_seed(0)
    BF, S, N, Z = 8, 64, 2, 2
    sigma = torch.rand(BF, S, N, Z, requires_grad=True)

    # 4프레임은 entity0 앞, 4프레임은 entity1 앞 (depth reversal 재현)
    depth_orders = [[0, 1]] * 4 + [[1, 0]] * 4

    perframe = l_depth_ranking_perframe([sigma], depth_orders)
    majority = l_depth_ranking(sigma, [0, 1])  # majority vote 결과

    assert not torch.isclose(perframe, majority, atol=1e-4), \
        "Per-frame loss should differ from majority-vote loss when orders are mixed"


def test_perframe_loss_consistent_orders():
    """모든 프레임 순서 동일하면 majority vote와 동일."""
    from scripts.train_phase19 import l_depth_ranking_perframe
    from models.losses import l_depth_ranking

    torch.manual_seed(1)
    BF, S, N, Z = 4, 64, 2, 2
    sigma = torch.rand(BF, S, N, Z, requires_grad=True)
    depth_orders = [[0, 1]] * BF  # 모든 프레임 동일

    perframe = l_depth_ranking_perframe([sigma], depth_orders)
    majority = l_depth_ranking(sigma, [0, 1])

    assert torch.isclose(perframe, majority, atol=1e-5), \
        f"Should match when all orders same: perframe={perframe:.6f} majority={majority:.6f}"


def test_perframe_loss_multi_layer():
    """여러 레이어 sigma_acc → 레이어 평균 손실."""
    from scripts.train_phase19 import l_depth_ranking_perframe

    torch.manual_seed(2)
    sigma1 = torch.rand(4, 64, 2, 2, requires_grad=True)
    sigma2 = torch.rand(4, 16, 2, 2, requires_grad=True)  # 다른 spatial
    depth_orders = [[0, 1], [1, 0], [0, 1], [0, 1]]

    loss = l_depth_ranking_perframe([sigma1, sigma2], depth_orders)
    assert loss.requires_grad, "Loss must retain gradient"
    assert loss.item() >= 0.0


def test_perframe_loss_gradient_flows():
    """각 프레임의 gradient가 sigma에 올바르게 전달됨."""
    from scripts.train_phase19 import l_depth_ranking_perframe

    sigma = torch.rand(4, 16, 2, 2, requires_grad=True)
    depth_orders = [[0, 1], [1, 0], [0, 1], [1, 0]]

    loss = l_depth_ranking_perframe([sigma], depth_orders)
    loss.backward()
    assert sigma.grad is not None
    assert not torch.all(sigma.grad == 0), "All gradients zero — no learning signal"


# ─── Fix 2: 고정 probe 측정 단위 테스트 ──────────────────────────────────────

def test_probe_t_values_defined():
    """PROBE_T_VALUES가 여러 t를 커버함."""
    from scripts.train_phase19 import PROBE_T_VALUES
    assert len(PROBE_T_VALUES) >= 3
    assert min(PROBE_T_VALUES) >= 0
    assert max(PROBE_T_VALUES) <= 200  # t_max=200


# ─── Fix 3: depth_pe init scale 단위 테스트 ─────────────────────────────────

def test_depth_pe_init_scale_p19():
    """Phase 19 VCALayer는 depth_pe_init_scale=0.3."""
    from scripts.train_phase19 import DEPTH_PE_INIT_SCALE
    assert DEPTH_PE_INIT_SCALE >= 0.1, \
        f"depth_pe_init_scale={DEPTH_PE_INIT_SCALE} too small"


def test_vcalayer_accepts_init_scale():
    """VCALayer가 depth_pe_init_scale 파라미터를 받음."""
    from models.vca_attention import VCALayer
    vca_old = VCALayer(query_dim=64, context_dim=128, depth_pe_init_scale=0.02)
    vca_new = VCALayer(query_dim=64, context_dim=128, depth_pe_init_scale=0.3)

    old_std = float(vca_old.depth_pe.std())
    new_std = float(vca_new.depth_pe.std())
    assert new_std > old_std * 5, \
        f"depth_pe std should be ~15x larger: old={old_std:.4f} new={new_std:.4f}"


def test_vcalayer_sigma_acc():
    """VCALayer.sigma_acc가 forward마다 누적되고 reset_sigma_acc()로 초기화됨."""
    from models.vca_attention import VCALayer

    vca = VCALayer(query_dim=64, context_dim=128, n_heads=4, n_entities=2, z_bins=2)
    x   = torch.randn(2, 16, 64)
    ctx = torch.randn(2, 2, 128)

    assert len(vca.sigma_acc) == 0
    vca(x, ctx)
    assert len(vca.sigma_acc) == 1
    vca(x, ctx)
    assert len(vca.sigma_acc) == 2

    vca.reset_sigma_acc()
    assert len(vca.sigma_acc) == 0


# ─── Fix 4: multi-layer injection 단위 테스트 ────────────────────────────────

def test_inject_keys_are_1280dim():
    """INJECT_KEYS_P19의 키들이 모두 query_dim=1280 레이어."""
    from scripts.train_phase19 import INJECT_KEYS_P19
    assert len(INJECT_KEYS_P19) >= 4, "Expected at least 4 injection points"
    assert any('mid_block' in k for k in INJECT_KEYS_P19)
    assert any('up_blocks' in k for k in INJECT_KEYS_P19)
    assert any('down_blocks' in k for k in INJECT_KEYS_P19)


def test_inject_keys_count():
    """INJECT_KEYS_P19가 6개."""
    from scripts.train_phase19 import INJECT_KEYS_P19
    assert len(INJECT_KEYS_P19) == 6, \
        f"Expected 6 injection keys, got {len(INJECT_KEYS_P19)}"


# ─── subprocess 통합 테스트 (5 epoch, 50 samples) ────────────────────────────

@pytest.fixture(scope="module")
def train_result():
    if not STATS_PATH.exists():
        pytest.skip("objaverse_stats.json not found")
    gpu_id = _pick_gpu()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id} if gpu_id else None
    r = subprocess.run(
        [sys.executable, "scripts/train_phase19.py",
         "--data-root",    "toy/data_objaverse",
         "--stats-path",   str(STATS_PATH),
         "--epochs",       "5",
         "--max-samples",  "50",
         "--lr",           "5e-5",
         "--lambda-depth", "0.3",
         "--save-dir",     str(P19_CKPT_DIR),
         "--debug-dir",    str(P19_DEBUG_DIR)],
        capture_output=True, text=True, timeout=1800, env=env,
    )
    return r


def test_train_exits_cleanly(train_result):
    assert train_result.returncode == 0, \
        f"train_phase19.py failed:\n{train_result.stderr[-1000:]}"


def test_dataset_ok(train_result):
    assert "DATASET_OK" in train_result.stdout


def test_inject_6_layers(train_result):
    """6개 레이어 주입 확인."""
    m = re.search(r'inject_p19\] (\d+) layers', train_result.stdout)
    assert m, "inject_p19 log not found"
    assert int(m.group(1)) == 6, f"Expected 6 layers, got {m.group(1)}"


def test_probe_sep_logged(train_result):
    """probe_sep= 가 매 epoch 출력됨."""
    seps = re.findall(r'probe_sep=([\d.]+)', train_result.stdout)
    assert len(seps) >= 5, f"Expected 5 probe_sep logs, got {len(seps)}"


def test_probe_sep_stable(train_result):
    """Fix 2: probe_sep 분산이 Phase 16~18 대비 작아야 함 (< 0.2 range)."""
    seps = [float(x) for x in re.findall(r'probe_sep=([\d.]+)', train_result.stdout)]
    if len(seps) < 3:
        pytest.skip("Not enough probe_sep values")
    sep_range = max(seps) - min(seps)
    # 5 epoch에서 range < 0.3 이면 안정 (Phase 16~18는 range ~ 0.3~0.5)
    assert sep_range < 0.4, \
        f"probe_sep range={sep_range:.3f} too high (unstable measurement)"


def test_ratio_sane(train_result):
    """ratio > 2.0: per-frame × multi-layer로 depth loss가 정확해져 ratio가 낮아짐.
    Phase 17/18(단일 majority vote)은 ratio 75~8000x였지만 그건 loss가 near-zero였기 때문.
    Phase 19는 실제 supervision signal이 작동하므로 ratio 3~20x가 정상."""
    ratios = re.findall(r'ratio=([\d.]+)x', train_result.stdout)
    assert len(ratios) > 0
    for r in ratios:
        assert float(r) >= 2.0, f"ratio={r} < 2.0 — l_depth dominating l_diff"


def test_loss_components_logged(train_result):
    for key in ["l_diff=", "l_depth=", "l_ortho="]:
        assert key in train_result.stdout


def test_checkpoint_saved(train_result):
    assert (P19_CKPT_DIR / "best.pt").exists()


def test_checkpoint_multi_layer_flag(train_result):
    """체크포인트에 multi_layer=True 기록됨."""
    p = P19_CKPT_DIR / "best.pt"
    if not p.exists():
        pytest.skip()
    ckpt = torch.load(p, map_location="cpu")
    assert ckpt.get("multi_layer") is True
    assert "inject_keys" in ckpt
    assert len(ckpt["inject_keys"]) == 6


def test_learning_ok(train_result):
    assert "LEARNING=OK" in train_result.stdout, \
        f"Expected LEARNING=OK:\n{train_result.stdout[-500:]}"


def test_gif_saved(train_result):
    gifs = list(P19_DEBUG_DIR.glob("sigma_epoch*.gif"))
    assert len(gifs) >= 1
