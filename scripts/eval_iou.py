"""
eval_iou.py — Phase 8: Sigma-Mask IoU 평가

Phase 7에서 저장한 sigmoid/softmax 모델을 로드해
ground truth entity mask와 sigma map의 IoU를 비교한다.

핵심 가설:
  Sigmoid 모델: 두 entity 모두 sigma가 있음 → IoU (entity 0, 1 모두)
  Softmax 모델: entity 1 (back)이 사라짐   → entity 1 IoU ≈ 0
"""
import sys, argparse
from pathlib import Path

import numpy as np
import torch
import imageio.v3 as iio
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.vca_attention import VCALayer
from scripts.train_vca import ToyVCADataset


# ─── IoU 계산 ────────────────────────────────────────────────────────────────
def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """binary mask IoU"""
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    return inter / max(union, 1)


def sigma_to_binary(sigma_hw: np.ndarray, gt_h: int, gt_w: int, threshold: float) -> np.ndarray:
    """sigma (hw, hw) float → binary mask (gt_h, gt_w)"""
    img = Image.fromarray((sigma_hw * 255).clip(0, 255).astype(np.uint8))
    img = img.resize((gt_w, gt_h), Image.BILINEAR)
    return np.array(img) > int(threshold * 255)


# ─── 모델 로드 ───────────────────────────────────────────────────────────────
def load_vca(ckpt_path: Path) -> VCALayer:
    ck = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    vca = VCALayer(
        query_dim=ck['query_dim'], context_dim=ck['context_dim'],
        n_heads=ck['n_heads'], n_entities=ck['n_entities'],
        z_bins=ck['z_bins'], lora_rank=ck['lora_rank'],
        use_softmax=ck['use_softmax'],
    )
    vca.load_state_dict(ck['state_dict'])
    vca.eval()
    return vca


def _train_fresh(use_softmax: bool, dataset, epochs: int = 5) -> VCALayer:
    """체크포인트 없을 때 즉석 학습"""
    from models.losses import l_depth_ranking, l_ortho
    vca = VCALayer(
        query_dim=64, context_dim=128, n_heads=4,
        n_entities=2, z_bins=2, lora_rank=4,
        use_softmax=use_softmax,
    )
    opt = torch.optim.Adam([p for p in vca.parameters() if p.requires_grad], lr=1e-3)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False)
    for _ in range(epochs):
        for x, ctx, depth_order, rgb in loader:
            x, ctx = x.squeeze(1), ctx.squeeze(1)
            opt.zero_grad()
            _ = vca(x, ctx)
            sigma_raw = vca.last_sigma_raw  # with grad — for loss
            order = [int(depth_order[0][0]), int(depth_order[1][0])]
            loss = l_depth_ranking(sigma_raw, order)
            if not use_softmax:
                loss = loss + 0.1 * l_ortho(vca.depth_pe)
            loss.backward()
            opt.step()
    vca.eval()
    return vca


# ─── IoU 평가 ────────────────────────────────────────────────────────────────
def evaluate_iou(vca: VCALayer, dataset: ToyVCADataset, threshold: float = 0.3) -> dict:
    """전체 dataset에 대한 entity별 평균 IoU"""
    patch_size = dataset.patch_size
    ious = {0: [], 1: []}

    for idx in range(len(dataset)):
        x, ctx, _, rgb = dataset[idx]
        x   = x.unsqueeze(0)    # (1,1,S,D) wait — dataset returns (1,S,D) for x
        ctx = ctx.unsqueeze(0)  # (1,1,N,CD)

        # dataset[idx] returns (1,S,D) and (1,N,CD) - already has batch dim 1
        x_in   = x.squeeze(0)    # (1, S, D) → use as (BF=1, S, D)
        ctx_in = ctx.squeeze(0)  # (1, N, CD)

        with torch.no_grad():
            vca(x_in, ctx_in)

        sigma = vca.last_sigma  # (1, S, N, Z)
        S = sigma.shape[1]
        hw = int(S ** 0.5)
        if hw * hw != S:
            continue

        for ei in range(2):
            # sigma z=0 slice → (hw, hw)
            sigma_hw = sigma[0, :, ei, 0].reshape(hw, hw).numpy()

            # ground truth mask
            fname = f'{idx:04d}_entity{ei}.png'
            mask_path = dataset.base / 'mask' / fname
            if not mask_path.exists():
                continue
            gt_mask = iio.imread(str(mask_path)) > 128  # (H, W) bool
            H, W = gt_mask.shape

            pred_mask = sigma_to_binary(sigma_hw, H, W, threshold)
            iou = compute_iou(pred_mask, gt_mask)
            ious[ei].append(iou)

    return {
        f'iou_entity{ei}': float(np.mean(v)) if v else 0.0
        for ei, v in ious.items()
    }


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario',   type=str, default='chain')
    parser.add_argument('--ablation-dir', type=str, default='debug/ablation')
    parser.add_argument('--threshold',  type=float, default=0.3)
    args = parser.parse_args()

    abl_dir = Path(args.ablation_dir)
    dataset = ToyVCADataset(scenario=args.scenario, query_dim=64, context_dim=128)

    if len(dataset) == 0:
        print("ERROR: No toy data.", flush=True); sys.exit(1)

    # 모델 로드 (없으면 즉석 학습)
    sig_ckpt = abl_dir / 'sigmoid_final.pt'
    sof_ckpt = abl_dir / 'softmax_final.pt'

    if sig_ckpt.exists():
        sig_vca = load_vca(sig_ckpt)
        print(f"Loaded sigmoid model from {sig_ckpt}", flush=True)
    else:
        print("No checkpoint found, training sigmoid model...", flush=True)
        sig_vca = _train_fresh(use_softmax=False, dataset=dataset)

    if sof_ckpt.exists():
        sof_vca = load_vca(sof_ckpt)
        print(f"Loaded softmax model from {sof_ckpt}", flush=True)
    else:
        print("No checkpoint found, training softmax model...", flush=True)
        sof_vca = _train_fresh(use_softmax=True, dataset=dataset)

    # IoU 평가
    print(f"\n=== IoU Evaluation (threshold={args.threshold}) ===", flush=True)
    sig_iou = evaluate_iou(sig_vca, dataset, threshold=args.threshold)
    sof_iou = evaluate_iou(sof_vca, dataset, threshold=args.threshold)

    for ei in range(2):
        k = f'iou_entity{ei}'
        print(f"sigmoid  {k}={sig_iou[k]:.4f}", flush=True)
        print(f"softmax  {k}={sof_iou[k]:.4f}", flush=True)

    sig_mean = np.mean(list(sig_iou.values()))
    sof_mean = np.mean(list(sof_iou.values()))
    print(f"\nsigmoid  mean_iou={sig_mean:.4f}", flush=True)
    print(f"softmax  mean_iou={sof_mean:.4f}", flush=True)

    winner = 'sigmoid' if sig_mean > sof_mean else 'softmax'
    print(f"\nBETTER_MODEL={winner}", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
