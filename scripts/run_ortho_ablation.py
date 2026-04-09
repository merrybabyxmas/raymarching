"""
Phase 9: L_ortho Ablation — z-collapse 방지 효과 검증

핵심 지표: z0_dominance = sigma[:,:,:,0].mean() / sigma.mean()
  L_ortho 없음: depth_pe 모든 z가 동일 방향 → 모든 attention이 z=0으로 쏠림 → z0_dominance > 0.6
  L_ortho 있음: 각 z를 서로 다른 방향으로 강제 → z 분산 → z0_dominance ≈ 0.5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.losses import l_depth_ranking, l_ortho
from scripts.train_vca import ToyVCADataset


ORTHO_WEIGHT = 0.5  # Phase 9: 강한 L_ortho로 효과를 명확히 보여줌


def z0_dominance(sigma: torch.Tensor) -> float:
    """sigma (BF, S, N, Z) → fraction of total mass at z=0 (uniform → 1/Z)"""
    z0_sum    = sigma[:, :, :, 0].sum().item()
    total_sum = sigma.sum().item()
    return z0_sum / max(total_sum, 1e-8)


def train_and_eval(use_ortho: bool, dataset, epochs: int, lr: float = 1e-3) -> dict:
    name = 'with_ortho' if use_ortho else 'no_ortho'

    vca = VCALayer(
        query_dim=64, context_dim=128, n_heads=4,
        n_entities=2, z_bins=2, lora_rank=4,
        use_softmax=False,  # Phase 9은 Sigmoid만 — Disappearance 무관하게 z-collapse만 비교
    )
    params = [p for p in vca.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False)

    all_sigmas = []

    for ep in range(epochs):
        vca.train()
        ep_loss = 0.0
        ep_sigmas = []

        for x, ctx, depth_order, rgb in loader:
            x   = x.squeeze(1)
            ctx = ctx.squeeze(1)
            opt.zero_grad()

            _ = vca(x, ctx)
            sigma_raw = vca.last_sigma_raw  # (B, S, N, Z) with grad — for loss
            sigma     = vca.last_sigma      # (B, S, N, Z) detached — for metrics

            order = [int(depth_order[0][0]), int(depth_order[1][0])]
            loss  = l_depth_ranking(sigma_raw, order)
            if use_ortho:
                loss = loss + ORTHO_WEIGHT * l_ortho(vca.depth_pe)

            loss.backward()
            opt.step()
            ep_loss += loss.item()
            ep_sigmas.append(sigma.detach())

        ep_sigma_all = torch.cat(ep_sigmas, dim=0)
        dom = z0_dominance(ep_sigma_all)
        print(
            f"[{name}] epoch={ep}  loss={ep_loss/len(loader):.4f}  z0_dominance={dom:.4f}",
            flush=True,
        )
        all_sigmas.append(ep_sigma_all)

    final_sigma = all_sigmas[-1]
    final_dom   = z0_dominance(final_sigma)
    print(
        f"FINAL {name} z0_dominance={final_dom:.4f}",
        flush=True,
    )

    return {
        'vca':            vca,
        'final_sigma':    final_sigma,
        'final_z0_dom':   final_dom,
        'sigma_history':  all_sigmas,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',   type=int, default=5)
    parser.add_argument('--scenario', type=str, default='chain')
    parser.add_argument('--out-dir',  type=str, default='debug/ortho_ablation')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ToyVCADataset(scenario=args.scenario, query_dim=64, context_dim=128)
    if len(dataset) == 0:
        print("ERROR: No toy data. Run: python toy/generate_toy_data.py", flush=True)
        sys.exit(1)

    print("=== No L_ortho ===", flush=True)
    no_ortho_result = train_and_eval(use_ortho=False, dataset=dataset, epochs=args.epochs)

    print("=== With L_ortho ===", flush=True)
    ortho_result = train_and_eval(use_ortho=True, dataset=dataset, epochs=args.epochs)

    # 체크포인트 저장
    torch.save({
        'state_dict':  no_ortho_result['vca'].state_dict(),
        'use_ortho':   False,
        'query_dim': 64, 'context_dim': 128, 'n_heads': 4,
        'n_entities': 2, 'z_bins': 2, 'lora_rank': 4,
    }, out_dir / 'no_ortho_final.pt')

    torch.save({
        'state_dict':  ortho_result['vca'].state_dict(),
        'use_ortho':   True,
        'query_dim': 64, 'context_dim': 128, 'n_heads': 4,
        'n_entities': 2, 'z_bins': 2, 'lora_rank': 4,
    }, out_dir / 'with_ortho_final.pt')

    # 요약
    dom_no  = no_ortho_result['final_z0_dom']
    dom_yes = ortho_result['final_z0_dom']
    print(f"\n=== Phase 9 Summary ===", flush=True)
    print(f"no_ortho   z0_dominance={dom_no:.4f}", flush=True)
    print(f"with_ortho z0_dominance={dom_yes:.4f}", flush=True)

    if dom_no > dom_yes:
        print("RESULT: L_ortho reduces z0_dominance (z-collapse prevented)", flush=True)
    else:
        print("RESULT: L_ortho effect not observed — check ORTHO_WEIGHT or epochs", flush=True)

    print("Done.", flush=True)


if __name__ == '__main__':
    main()
