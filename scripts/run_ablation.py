"""
Phase 7 Ablation: VCA-Sigmoid vs VCA-Softmax

동일한 toy 데이터로 두 모델을 학습 후 Disappearance 현상을 정량 비교.

핵심 지표:
  visibility_min  : min(σ_entity0, σ_entity1) → 높을수록 두 entity 모두 살아있음
  disappearance_rate : 한 entity의 σ가 threshold 미만인 샘플 비율
  both_high           : σ_entity0 > 0.5 AND σ_entity1 > 0.5인 spatial position 비율
"""
import argparse, sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.losses import l_depth_ranking, l_ortho
from scripts.train_vca import ToyVCADataset
from scripts.debug_gif import make_comparison_gif


DISAPPEARANCE_THRESHOLD = 0.15   # entity σ가 이 이하면 "사라진 것"으로 간주


# ─── 지표 계산 ────────────────────────────────────────────────────────────────
def batch_metrics(sigma: torch.Tensor) -> dict:
    """sigma: (BF, S, N, Z) → 배치 평균 지표"""
    # entity별 mean sigma (over S, Z)
    s0 = sigma[:, :, 0, :].mean(dim=[1, 2])  # (BF,)
    s1 = sigma[:, :, 1, :].mean(dim=[1, 2])  # (BF,)

    visibility_min  = torch.min(s0, s1).mean().item()
    disappearance   = ((s0 < DISAPPEARANCE_THRESHOLD) | (s1 < DISAPPEARANCE_THRESHOLD)).float().mean().item()
    both_high       = ((sigma[:, :, 0, :] > 0.5) & (sigma[:, :, 1, :] > 0.5)).float().mean().item()
    return {
        'visibility_min':    visibility_min,
        'disappearance_rate': disappearance,
        'both_high':          both_high,
    }


# ─── 학습 + 평가 ─────────────────────────────────────────────────────────────
def train_and_eval(use_softmax: bool, dataset, epochs: int, lr: float = 1e-3) -> dict:
    name = 'softmax' if use_softmax else 'sigmoid'

    vca = VCALayer(
        query_dim=64, context_dim=128, n_heads=4,
        n_entities=2, z_bins=2, lora_rank=4,
        use_softmax=use_softmax,
    )
    params = [p for p in vca.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False)

    all_sigmas, all_rgbs = [], []

    for ep in range(epochs):
        vca.train()
        ep_loss, ep_sigmas = 0.0, []

        for x, ctx, depth_order, rgb in loader:
            x   = x.squeeze(1)
            ctx = ctx.squeeze(1)
            opt.zero_grad()

            _ = vca(x, ctx)
            sigma_raw = vca.last_sigma_raw  # (B, S, N, Z) with grad — for loss
            sigma     = vca.last_sigma      # (B, S, N, Z) detached — for metrics

            order = [int(depth_order[0][0]), int(depth_order[1][0])]
            ld = l_depth_ranking(sigma_raw, order)
            # L_ortho는 Sigmoid에만 적용 (Softmax variant는 구조적 비교 목적)
            lo = l_ortho(vca.depth_pe) if not use_softmax else torch.tensor(0.0)
            loss = ld + 0.1 * lo

            loss.backward()
            opt.step()
            ep_loss += loss.item()
            ep_sigmas.append(sigma.detach())
            all_rgbs.append(rgb[0].numpy())

        ep_sigma_all = torch.cat(ep_sigmas, dim=0)
        m = batch_metrics(ep_sigma_all)
        print(
            f"[{name}] epoch={ep}  loss={ep_loss/len(loader):.4f}  "
            f"visibility_min={m['visibility_min']:.4f}  "
            f"disappearance_rate={m['disappearance_rate']:.4f}  "
            f"both_high={m['both_high']:.4f}",
            flush=True,
        )
        all_sigmas.append(ep_sigma_all)

    final_sigma = all_sigmas[-1]
    final_m = batch_metrics(final_sigma)
    print(
        f"FINAL {name} "
        f"visibility_min={final_m['visibility_min']:.4f} "
        f"disappearance_rate={final_m['disappearance_rate']:.4f} "
        f"both_high={final_m['both_high']:.4f}",
        flush=True,
    )

    return {
        'vca': vca,
        'final_sigma': final_sigma,
        'final_metrics': final_m,
        'sigma_history': all_sigmas,
        'rgb_history': all_rgbs,
    }


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',   type=int, default=5)
    parser.add_argument('--scenario', type=str, default='chain')
    parser.add_argument('--out-dir',  type=str, default='debug/ablation')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ToyVCADataset(scenario=args.scenario, query_dim=64, context_dim=128)
    if len(dataset) == 0:
        print("ERROR: No toy data. Run: python toy/generate_toy_data.py", flush=True)
        sys.exit(1)

    print("=== Sigmoid Model ===", flush=True)
    sig_result  = train_and_eval(use_softmax=False, dataset=dataset, epochs=args.epochs)

    print("=== Softmax Model ===", flush=True)
    sof_result  = train_and_eval(use_softmax=True,  dataset=dataset, epochs=args.epochs)

    # ── 모델 체크포인트 저장 ────────────────────────────────────────────────
    torch.save({
        'state_dict':  sig_result['vca'].state_dict(),
        'use_softmax': False,
        'query_dim': 64, 'context_dim': 128, 'n_heads': 4,
        'n_entities': 2, 'z_bins': 2, 'lora_rank': 4,
    }, out_dir / 'sigmoid_final.pt')

    torch.save({
        'state_dict':  sof_result['vca'].state_dict(),
        'use_softmax': True,
        'query_dim': 64, 'context_dim': 128, 'n_heads': 4,
        'n_entities': 2, 'z_bins': 2, 'lora_rank': 4,
    }, out_dir / 'softmax_final.pt')

    # ── Comparison GIF ──────────────────────────────────────────────────────
    sig_final = sig_result['final_sigma']   # (BF, S, N, Z)
    sof_final = sof_result['final_sigma']

    S = sig_final.shape[1]
    hw = int(S ** 0.5)

    if hw * hw == S:
        def to_sigma_hw(sigma, batch_idx=0):
            return sigma[batch_idx, :, :, 0].T.reshape(2, hw, hw).numpy()  # (N, hw, hw)

        rgb_list  = sig_result['rgb_history'][-len(dataset):]
        sig_list  = [to_sigma_hw(sig_result['sigma_history'][-1][i:i+1]) for i in range(min(8, len(rgb_list)))]
        sof_list  = [to_sigma_hw(sof_result['sigma_history'][-1][i:i+1]) for i in range(min(8, len(rgb_list)))]
        rgb_np    = [r.numpy().astype(np.uint8) if hasattr(r, 'numpy') else r for r in rgb_list[:len(sig_list)]]

        make_comparison_gif(rgb_np, sig_list, sof_list, out_dir / 'comparison.gif', panel_size=64)
        print(f"Comparison GIF saved: {out_dir / 'comparison.gif'}", flush=True)

    print("Done.", flush=True)


if __name__ == '__main__':
    main()
