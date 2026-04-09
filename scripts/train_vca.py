"""
train_vca.py — VCALayer 단독 PoC 학습 루프

UNet 없이 VCALayer만 학습. toy/data에서 depth + mask를 supervision으로 사용.
매 epoch sigma GIF를 debug/로 저장해 학습 진행을 시각적으로 추적한다.

손실: L = w_depth * l_depth_ranking + w_ortho * l_ortho
"""
import argparse
import sys
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import imageio.v3 as iio

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.vca_attention import VCALayer
from models.losses import l_ortho, l_depth_ranking
from scripts.debug_gif import make_debug_gif


# ─── Dataset ────────────────────────────────────────────────────────────────
class ToyVCADataset(Dataset):
    """
    Phase 2 toy 데이터 → VCALayer 학습용 배치.

    입력:
      x   = (1, S, query_dim)   RGB 패치를 query_dim으로 zero-padding
      ctx = (1, N, context_dim) entity mask 영역 mean depth를 context_dim으로 인코딩

    감독:
      depth_order = [front_entity_idx, back_entity_idx]
    """
    def __init__(
        self,
        scenario: str = 'chain',
        cam: str = 'front_right',
        query_dim: int = 64,
        context_dim: int = 128,
        patch_size: int = 16,
        n_entities: int = 2,
    ):
        base = Path(f'toy/data/{scenario}/{cam}')
        self.frames = sorted((base / 'frames').glob('*.png'))
        self.depths = sorted((base / 'depth').glob('*.npy'))
        self.base   = base
        self.query_dim    = query_dim
        self.context_dim  = context_dim
        self.patch_size   = patch_size
        self.n_entities   = n_entities

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        rgb = iio.imread(str(self.frames[idx]))[..., :3].astype(np.float32) / 255.0
        H, W = rgb.shape[:2]
        depth = np.load(str(self.depths[idx]))  # (H, W) float32

        # ── x: RGB 패치 → (1, S, query_dim) ─────────────────────────────────
        p = self.patch_size
        rgb_p = rgb[:H//p*p, :W//p*p]           # crop to multiple of p
        patches = rgb_p.reshape(H//p, p, W//p, p, 3).mean((1, 3))  # (ph, pw, 3)
        ph, pw = patches.shape[:2]
        S = ph * pw
        x_raw = patches.reshape(S, 3)            # (S, 3)
        x = np.zeros((S, self.query_dim), dtype=np.float32)
        x[:, :3] = x_raw

        # ── ctx: entity별 depth 정보 → (1, N, context_dim) ──────────────────
        ctx = np.zeros((self.n_entities, self.context_dim), dtype=np.float32)
        depths_mean = []
        for ei in range(self.n_entities):
            mask_path = self.base / 'mask' / f'{idx:04d}_entity{ei}.png'
            mask = iio.imread(str(mask_path)) > 128  # (H, W) bool
            # entity mean color
            if mask.sum() > 0:
                mean_color = rgb[mask].mean(axis=0)      # (3,)
                mean_depth = float(depth[mask].mean())
            else:
                mean_color = np.zeros(3, dtype=np.float32)
                mean_depth = 0.0
            ctx[ei, :3]  = mean_color
            ctx[ei,  3]  = mean_depth / 10.0            # normalize
            depths_mean.append(mean_depth)

        # depth_order: 낮은 depth(가까운) entity가 앞
        front = int(np.argmin(depths_mean))
        back  = 1 - front

        return (
            torch.from_numpy(x).unsqueeze(0),            # (1, S, D)
            torch.from_numpy(ctx).unsqueeze(0),           # (1, N, CD)
            [front, back],
            torch.from_numpy(rgb_p.reshape(ph*p, pw*p, 3)),  # RGB for GIF
        )


# ─── Trainer ────────────────────────────────────────────────────────────────
class VCATrainer:
    def __init__(
        self,
        vca_layer: VCALayer,
        dataset: ToyVCADataset,
        out_dir: str,
        lr: float = 1e-3,
        w_depth: float = 1.0,
        w_ortho: float = 0.1,
        batch_size: int = 4,
    ):
        self.vca     = vca_layer
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.w_depth = w_depth
        self.w_ortho = w_ortho

        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # LoRA + depth_pe만 학습 (base weight frozen)
        params = [p for p in vca_layer.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=lr)

    def train_epoch(self, epoch_idx: int) -> dict:
        self.vca.train()
        total_loss = total_ldepth = total_lortho = 0.0
        sigma_list, rgb_list = [], []

        for x, ctx, depth_order, rgb in self.loader:
            # x: (B,1,S,D) → (B,S,D)  ctx: (B,1,N,CD) → (B,N,CD)
            x   = x.squeeze(1)
            ctx = ctx.squeeze(1)

            self.opt.zero_grad()
            _ = self.vca(x, ctx)
            sigma_raw = self.vca.last_sigma_raw  # (B, S, N, Z) with grad
            sigma     = self.vca.last_sigma      # (B, S, N, Z) detached (for GIF)

            # depth_order: list of [front, back] per sample — use first sample's order
            order = [int(depth_order[0][0]), int(depth_order[1][0])]

            ld = l_depth_ranking(sigma_raw, order)
            lo = l_ortho(self.vca.depth_pe)
            loss = self.w_depth * ld + self.w_ortho * lo

            loss.backward()
            self.opt.step()

            total_loss   += loss.item()
            total_ldepth += ld.item()
            total_lortho += lo.item()

            # sigma를 GIF용으로 수집 (첫 샘플만)
            with torch.no_grad():
                S  = sigma.shape[1]
                hw = int(S ** 0.5)
                if hw * hw == S:
                    s_np = sigma[0].detach().cpu().numpy()  # (S, N, Z)
                    # z=0 mean, shape (N, hw, hw)
                    s_hw = s_np[:, :, 0].T.reshape(self.vca.n_entities, hw, hw)
                    sigma_list.append(s_hw)
                    rgb_list.append(rgb[0].numpy().astype(np.uint8))

        n = len(self.loader)
        metrics = {
            'loss':    total_loss   / n,
            'l_depth': total_ldepth / n,
            'l_ortho': total_lortho / n,
        }

        if sigma_list:
            self._save_sigma_gif(epoch_idx, sigma_list, rgb_list)

        return metrics

    def _save_sigma_gif(self, epoch_idx: int, sigma_list, rgb_list):
        out = self.out_dir / f'sigma_epoch{epoch_idx:03d}.gif'
        # sigma_list: list of (N, H, W) float32
        # rgb_list:   list of (H, W, 3) uint8 (or float → convert)
        rgbs = []
        for r in rgb_list:
            if r.dtype != np.uint8:
                r = (r * 255).clip(0, 255).astype(np.uint8)
            rgbs.append(r)
        make_debug_gif(rgbs, sigma_list, out, panel_size=64)


# ─── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=5)
    parser.add_argument('--scenario',   type=str,   default='chain')
    parser.add_argument('--cam',        type=str,   default='front_right')
    parser.add_argument('--batch-size', type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--query-dim',  type=int,   default=64)
    parser.add_argument('--context-dim',type=int,   default=128)
    parser.add_argument('--out-dir',    type=str,   default='debug/train')
    args = parser.parse_args()

    dataset = ToyVCADataset(
        scenario=args.scenario, cam=args.cam,
        query_dim=args.query_dim, context_dim=args.context_dim,
    )
    if len(dataset) == 0:
        print("ERROR: no toy data found. Run: python toy/generate_toy_data.py", flush=True)
        sys.exit(1)

    vca = VCALayer(
        query_dim=args.query_dim,
        context_dim=args.context_dim,
        n_heads=4,
        n_entities=2,
        z_bins=2,
        lora_rank=4,
    )

    trainer = VCATrainer(vca, dataset, args.out_dir,
                         lr=args.lr, batch_size=args.batch_size)

    for ep in range(args.epochs):
        m = trainer.train_epoch(ep)
        print(
            f"epoch={ep}  loss={m['loss']:.6f}  "
            f"l_depth={m['l_depth']:.6f}  l_ortho={m['l_ortho']:.6f}",
            flush=True,
        )


if __name__ == '__main__':
    main()
