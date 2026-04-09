"""
Phase 10-B: 학습 유효성 검증 (check_learning.py)

세 가지 지표 측정 후 stdout 출력:
  1. sigma_consistency: 학습 후 front entity sigma_z0 > back entity sigma_z0 비율
  2. sigma_separation : |sigma_e0 - sigma_e1| 데이터셋 평균
  3. loss_curve       : epoch별 l_depth 값

출력 형식:
  ENCODING x_std=X ctx_diff=Y
  BEFORE sigma_consistency=X sigma_separation=Y
  AFTER  sigma_consistency=X sigma_separation=Y
  LOSS_CURVE v0 v1 v2 ...
  LEARNING=OK  또는  LEARNING=FAIL
"""
import sys
import argparse
from pathlib import Path

import torch
torch.manual_seed(42)

import numpy as np
np.random.seed(42)

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.losses import l_depth_ranking, l_ortho
from scripts.train_vca import ToyVCADataset

LEARNING_OK_THRESHOLD = 0.60   # sigma_consistency > 이 값이면 LEARNING=OK


# ─── 지표 계산 ────────────────────────────────────────────────────────────────

def compute_encoding_stats(dataset: ToyVCADataset) -> dict:
    """x_std, ctx_diff를 전체 데이터셋 평균으로 측정"""
    x_stds, ctx_diffs = [], []
    for idx in range(len(dataset)):
        x, ctx, _, _ = dataset[idx]
        x_np  = x.squeeze(0).numpy()    # (S, D)
        c_np  = ctx.squeeze(0).numpy()  # (N, CD)
        x_stds.append(float(x_np.std()))
        ctx_diffs.append(float(np.abs(c_np[0] - c_np[1]).mean()))
    return {
        'x_std':    float(np.mean(x_stds)),
        'ctx_diff': float(np.mean(ctx_diffs)),
    }


def compute_sigma_metrics(vca: VCALayer, dataset: ToyVCADataset) -> dict:
    """sigma_consistency 와 sigma_separation 측정"""
    vca.eval()
    consistencies, separations = [], []

    with torch.no_grad():
        for idx in range(len(dataset)):
            x, ctx, depth_order, _ = dataset[idx]
            x_in   = x.squeeze(0).unsqueeze(0)    # (1, S, D)
            ctx_in = ctx.squeeze(0).unsqueeze(0)  # (1, N, CD)
            vca(x_in, ctx_in)
            sigma = vca.last_sigma   # (1, S, N, Z)

            front, back = int(depth_order[0]), int(depth_order[1])
            s_front = float(sigma[0, :, front, 0].mean())
            s_back  = float(sigma[0, :, back,  0].mean())

            consistencies.append(1.0 if s_front > s_back else 0.0)
            # separation: mean |sigma_e0 - sigma_e1| over S and Z
            s0 = float(sigma[0, :, 0, :].mean())
            s1 = float(sigma[0, :, 1, :].mean())
            separations.append(abs(s0 - s1))

    return {
        'sigma_consistency': float(np.mean(consistencies)),
        'sigma_separation':  float(np.mean(separations)),
    }


# ─── 학습 ─────────────────────────────────────────────────────────────────────

def train(vca: VCALayer, dataset: ToyVCADataset,
          epochs: int, lr: float, use_ortho: bool = True) -> list:
    """학습 실행, epoch별 l_depth 반환"""
    params = [p for p in vca.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False)
    ldepth_curve = []

    for ep in range(epochs):
        vca.train()
        ep_ldepth = 0.0

        for x, ctx, depth_order, _ in loader:
            x   = x.squeeze(1)
            ctx = ctx.squeeze(1)
            opt.zero_grad()

            vca(x, ctx)
            sigma_raw = vca.last_sigma_raw   # with grad
            order = [int(depth_order[0][0]), int(depth_order[1][0])]
            ld = l_depth_ranking(sigma_raw, order)
            loss = ld
            if use_ortho:
                loss = loss + 0.1 * l_ortho(vca.depth_pe)
            loss.backward()
            opt.step()
            ep_ldepth += ld.item()

        ldepth_curve.append(ep_ldepth / len(loader))

    return ldepth_curve


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario',    type=str,   default='chain')
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--query-dim',   type=int,   default=64)
    parser.add_argument('--context-dim', type=int,   default=128)
    args = parser.parse_args()

    dataset = ToyVCADataset(
        scenario=args.scenario,
        query_dim=args.query_dim,
        context_dim=args.context_dim,
    )
    if len(dataset) == 0:
        print("ERROR: No toy data.", flush=True)
        sys.exit(1)

    # ── 인코딩 통계 ──────────────────────────────────────────────────────────
    enc = compute_encoding_stats(dataset)
    print(f"ENCODING x_std={enc['x_std']:.3f} ctx_diff={enc['ctx_diff']:.3f}",
          flush=True)

    # ── 공통 VCA 구조 ─────────────────────────────────────────────────────────
    def make_vca():
        return VCALayer(
            query_dim=args.query_dim,
            context_dim=args.context_dim,
            n_heads=4, n_entities=2, z_bins=2, lora_rank=4,
            use_softmax=False,
        )

    # ── BEFORE 측정 (untrained) ──────────────────────────────────────────────
    vca_fresh = make_vca()
    before = compute_sigma_metrics(vca_fresh, dataset)
    print(f"BEFORE sigma_consistency={before['sigma_consistency']:.4f} "
          f"sigma_separation={before['sigma_separation']:.4f}",
          flush=True)

    # ── 학습 ─────────────────────────────────────────────────────────────────
    vca_trained = make_vca()
    ldepth_curve = train(vca_trained, dataset, epochs=args.epochs, lr=args.lr)

    # ── AFTER 측정 ───────────────────────────────────────────────────────────
    after = compute_sigma_metrics(vca_trained, dataset)
    print(f"AFTER  sigma_consistency={after['sigma_consistency']:.4f} "
          f"sigma_separation={after['sigma_separation']:.4f}",
          flush=True)

    # ── Loss curve ───────────────────────────────────────────────────────────
    curve_str = ' '.join(f'{v:.4f}' for v in ldepth_curve)
    print(f"LOSS_CURVE {curve_str}", flush=True)

    # ── 판정 ─────────────────────────────────────────────────────────────────
    ok = after['sigma_consistency'] > LEARNING_OK_THRESHOLD
    print(f"LEARNING={'OK' if ok else 'FAIL'}", flush=True)

    if not ok:
        print(
            f"DIAGNOSIS: sigma_consistency={after['sigma_consistency']:.4f} "
            f"< threshold={LEARNING_OK_THRESHOLD}. "
            f"Check encoding (x_std={enc['x_std']:.3f}, ctx_diff={enc['ctx_diff']:.3f}) "
            f"and depth discrimination in dataset.",
            flush=True,
        )


if __name__ == '__main__':
    main()
