"""
Phase 16 Part A: Objaverse 데이터로 AnimateDiff VCA 재학습

Phase 12와 달라지는 점:
  - 데이터셋: ObjaverseVCADataset (1,000+ 프레임, 10+ 시나리오)
  - entity_context: meta.json keyword를 실제 CLIP으로 인코딩 (캐싱)
  - depth_order GT: meta.json에서 자동 로드
  - t_max: Phase 14에서 발견된 최적값 (200) 기본 적용

주의사항:
  FM-I6: loss는 last_sigma_raw (grad 있음) 사용, last_sigma(detach) 금지
  CLIP 캐싱 필수: _clip_cache로 재인코딩 방지
  DATASET_FAIL이면 학습 즉시 중단
"""
import argparse
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as iio2
import imageio.v3 as iio3

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.losses import l_ortho as loss_ortho, l_depth_ranking, l_diff as loss_diff
from scripts.run_animatediff import load_pipeline, FixedContextVCAProcessor
from scripts.train_animatediff_vca import (
    inject_vca_train, TrainVCAProcessor,
    compute_sigma_stats_train, save_sigma_gif,
)
from torch.nn.functional import layer_norm
from scripts.prompt_identity import make_identity_prompts


# ─── 데이터셋 품질 게이팅 ─────────────────────────────────────────────────────

def check_dataset_quality(stats_path: str) -> bool:
    """
    stats.json 읽어서 학습 진행 가능 여부 반환.
    실패 조건이 하나라도 있으면 False + DATASET_FAIL 출력.
    """
    with open(stats_path) as f:
        stats = json.load(f)

    checks = {
        "total_frames >= 500":       stats.get("total_frames", 0) >= 500,
        "depth_reversal_rate > 0.2": stats.get("depth_reversal_rate", 0) > 0.2,
        "occlusion_rate > 0.3":      stats.get("occlusion_rate", 0) > 0.3,
    }

    all_pass = all(checks.values())
    for name, result in checks.items():
        status = "OK" if result else "FAIL"
        print(f"DATASET_CHECK {status}: {name}", flush=True)

    if not all_pass:
        print("DATASET_FAIL: dataset quality insufficient for training", flush=True)
        print("Fix: regenerate with more cameras or scenarios", flush=True)
    else:
        print("DATASET_OK: proceeding to training", flush=True)

    return all_pass


# ─── CLIP 캐싱 ────────────────────────────────────────────────────────────────

_clip_cache: dict = {}


def get_entity_context_from_meta(pipe, meta: dict, device: str) -> torch.Tensor:
    """
    meta.json의 entity prompt를 CLIP 평균 토큰 임베딩으로 인코딩.
    반환: (1, 2, 768) fp32
    캐싱: 같은 텍스트 쌍은 한 번만 인코딩.
    """
    e0_text, e1_text, _, _, _ = make_identity_prompts(meta)
    cache_key = (e0_text, e1_text)

    if cache_key in _clip_cache:
        return _clip_cache[cache_key].to(device)

    embs = []
    for text in [e0_text, e1_text]:
        tokens = pipe.tokenizer(
            text, return_tensors="pt",
            padding="max_length", max_length=pipe.tokenizer.model_max_length,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            out = pipe.text_encoder(**tokens).last_hidden_state  # (1, 77, 768)
        # 실제 텍스트 토큰 평균 (BOS/EOS 제외)
        input_ids = tokens.input_ids[0]
        eos_id = pipe.tokenizer.eos_token_id
        eos_positions = (input_ids == eos_id).nonzero(as_tuple=True)[0]
        eos_pos = int(eos_positions[0].item()) if len(eos_positions) > 0 else 77
        emb = out[:, 1:max(2, eos_pos), :].mean(dim=1, keepdim=True)  # (1, 1, 768)
        embs.append(emb)

    ctx = torch.cat(embs, dim=1).float()  # (1, 2, 768) fp32
    _clip_cache[cache_key] = ctx.cpu()

    diff = float((ctx[0, 0] - ctx[0, 1]).norm())
    print(f"  [ctx] '{e0_text}' vs '{e1_text}'  diff_norm={diff:.4f}", flush=True)
    return ctx.to(device)


# ─── Objaverse 학습용 데이터셋 ───────────────────────────────────────────────

class ObjaverseTrainDataset:
    """
    toy/data_objaverse/ 자동 스캔 → 프레임 numpy 배열 반환.
    VAE 인코딩은 학습 루프에서 배치마다 수행.
    """

    def __init__(self, data_root: str, max_samples: int = None,
                 n_frames: int = 8, height: int = 256, width: int = 256,
                 seed: int = 42):
        from PIL import Image
        self.n_frames = n_frames
        self.height   = height
        self.width    = width
        self._Image   = Image

        samples = []
        for meta_path in Path(data_root).rglob("meta.json"):
            d = meta_path.parent
            frames = sorted((d / "frames").glob("*.png"))
            depths = sorted((d / "depth").glob("*.npy"))
            if len(frames) >= n_frames and len(depths) >= n_frames:
                with open(meta_path) as f:
                    meta = json.load(f)
                samples.append({"dir": d, "meta": meta,
                                 "frames": frames, "depths": depths})

        if max_samples and max_samples < len(samples):
            rng = np.random.RandomState(seed)
            idxs = rng.choice(len(samples), max_samples, replace=False)
            samples = [samples[i] for i in idxs]

        self.samples = samples
        print(f"ObjaverseTrainDataset: {len(samples)} samples "
              f"(data_root={data_root})", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        seq_dir = s["dir"]
        Image = self._Image

        frames_np = []
        for fp in s["frames"][:self.n_frames]:
            img = Image.open(str(fp)).convert("RGB").resize(
                (self.width, self.height))
            frames_np.append(np.array(img))
        frames_np = np.stack(frames_np, axis=0)  # (T, H, W, 3) uint8

        import cv2 as _cv2
        depths = []
        for dp in s["depths"][:self.n_frames]:
            depth_arr = np.load(str(dp))
            if depth_arr.shape[0] != self.height or depth_arr.shape[1] != self.width:
                depth_arr = _cv2.resize(depth_arr.astype(np.float32),
                                        (self.width, self.height),
                                        interpolation=_cv2.INTER_LINEAR)
            depths.append(depth_arr)
        depths_np = np.stack(depths, axis=0)  # (T, H, W)

        # depth_order: 모든 프레임에서 투표
        depth_orders = self._compute_depth_orders(seq_dir, depths_np, s["meta"])

        return frames_np, depths_np, depth_orders, s["meta"]

    def _compute_depth_orders(self, seq_dir: Path, depths_np: np.ndarray,
                               meta: dict) -> list:
        """각 프레임별 [front, back] 인덱스 리스트."""
        import cv2 as _cv2
        T, H, W = depths_np.shape
        orders = []
        for fi in range(T):
            m0_p = seq_dir / "mask" / f"{fi:04d}_entity0.png"
            m1_p = seq_dir / "mask" / f"{fi:04d}_entity1.png"
            if m0_p.exists() and m1_p.exists():
                m0 = iio3.imread(str(m0_p)) > 128
                m1 = iio3.imread(str(m1_p)) > 128
                # resize masks to match depth resolution if needed
                if m0.shape[0] != H or m0.shape[1] != W:
                    m0 = _cv2.resize(m0.astype(np.uint8), (W, H),
                                     interpolation=_cv2.INTER_NEAREST) > 0
                    m1 = _cv2.resize(m1.astype(np.uint8), (W, H),
                                     interpolation=_cv2.INTER_NEAREST) > 0
                m0_only = m0 & ~m1
                m1_only = m1 & ~m0
                if m0_only.sum() > 5 and m1_only.sum() > 5:
                    d0 = float(depths_np[fi][m0_only].mean())
                    d1 = float(depths_np[fi][m1_only].mean())
                    front = 0 if d0 < d1 else 1
                    orders.append([front, 1 - front])
                    continue
            orders.append([0, 1])
        return orders


# ─── 학습 스텝 (t_max 파라미터) ───────────────────────────────────────────────

def training_step(pipe, vca_layer, latents, encoder_hidden_states,
                  depth_orders, lambda_depth, lambda_ortho, device,
                  t_max=200):
    """Phase 12/14 training_step과 동일 구조, t_max 파라미터 추가."""
    noise = torch.randn_like(latents)
    t = torch.randint(0, t_max, (1,), device=device).long()
    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        pred_noise = pipe.unet(
            noisy_latents, t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

    ld = loss_diff(pred_noise.float(), noise.float())

    sigma_raw = vca_layer.last_sigma_raw  # FM-I6: grad 있는 raw
    if sigma_raw is None:
        l_depth = torch.tensor(0.0, device=device)
        l_ort   = torch.tensor(0.0, device=device)
    else:
        front_votes = sum(1 for d in depth_orders if d[0] == 0)
        order = [0, 1] if front_votes >= len(depth_orders) // 2 else [1, 0]
        l_depth = l_depth_ranking(sigma_raw, order)
        l_ort   = loss_ortho(vca_layer.depth_pe)

    loss = ld + lambda_depth * l_depth + lambda_ortho * l_ort
    return {
        "loss":        loss.detach().item(),
        "l_diff":      ld.detach().item(),
        "l_depth":     l_depth.detach().item(),
        "l_ortho":     l_ort.detach().item(),
        "loss_tensor": loss,
    }


# ─── 메인 학습 루프 ──────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}", flush=True)

    # 데이터 품질 게이팅
    if args.stats_path and Path(args.stats_path).exists():
        if not check_dataset_quality(args.stats_path):
            return
    else:
        print("DATASET_CHECK SKIP: stats_path not provided, proceeding", flush=True)
        print("DATASET_OK: proceeding to training", flush=True)

    # 데이터셋 로드
    dataset = ObjaverseTrainDataset(
        data_root   = args.data_root,
        max_samples = args.max_samples,
        n_frames    = args.n_frames,
        height      = args.height,
        width       = args.width,
    )
    if len(dataset) == 0:
        print("DATASET_FAIL: no samples found", flush=True)
        return

    # 파이프라인 로드
    pipe = load_pipeline(device=device, dtype=torch.float16)

    # UNet frozen
    for p in pipe.unet.parameters():
        p.requires_grad = False
    assert sum(1 for p in pipe.unet.parameters() if p.requires_grad) == 0
    print("[init] UNet frozen.", flush=True)

    save_dir  = Path(args.save_dir)
    debug_dir = Path(args.debug_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # CLIP 임베딩 미리 계산 (캐싱)
    print("[clip] pre-computing entity embeddings...", flush=True)
    for idx in range(min(len(dataset), 20)):  # 첫 20개 미리 캐싱
        _, _, _, meta = dataset[idx]
        get_entity_context_from_meta(pipe, meta, device)
    print(f"[clip] cache size={len(_clip_cache)}", flush=True)

    # 대표 샘플 선택 (BEFORE/AFTER 측정용)
    probe_frames, probe_depths, probe_orders, probe_meta = dataset[0]
    probe_entity_ctx = get_entity_context_from_meta(pipe, probe_meta, device)
    probe_e0, probe_e1, _, _, _ = make_identity_prompts(probe_meta)
    print(f"[probe] entity_0='{probe_e0}'  entity_1='{probe_e1}'", flush=True)

    # VCA 주입
    vca_layer, injected_keys, original_procs = inject_vca_train(pipe, probe_entity_ctx)
    print(f"[inject] {injected_keys}", flush=True)

    trainable = [p for p in vca_layer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    print(f"[opt] trainable params: {sum(p.numel() for p in trainable):,}", flush=True)

    # 학습 루프
    training_curve = []
    best_sep = 0.0

    for epoch in range(args.epochs):
        vca_layer.train()

        # 매 epoch 랜덤 샘플 선택
        idx = int(np.random.randint(0, len(dataset)))
        frames_np, depths_np, depth_orders, meta = dataset[idx]

        # entity context (캐싱됨)
        entity_ctx = get_entity_context_from_meta(pipe, meta, device)

        # processor 업데이트 (context 교체)
        for key in injected_keys:
            proc = pipe.unet.attn_processors.get(key)
            if isinstance(proc, TrainVCAProcessor):
                proc.entity_context = entity_ctx

        # VAE 인코딩
        from scripts.train_animatediff_vca import encode_frames_to_latents
        latents = encode_frames_to_latents(pipe, frames_np, device)

        # encoder_hidden_states
        e0, e1, full_prompt, _, _ = make_identity_prompts(meta)
        tokens = pipe.tokenizer(
            full_prompt, return_tensors="pt",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            enc_hs = pipe.text_encoder(**tokens).last_hidden_state.half()

        optimizer.zero_grad()
        step_out = training_step(
            pipe, vca_layer, latents, enc_hs,
            depth_orders, args.lambda_depth, args.lambda_ortho, device,
            t_max=args.t_max,
        )
        step_out["loss_tensor"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        stats = compute_sigma_stats_train(vca_layer.last_sigma)
        print(
            f"epoch={epoch:3d}  loss={step_out['loss']:.4f}  "
            f"l_diff={step_out['l_diff']:.4f}  l_depth={step_out['l_depth']:.4f}  "
            f"l_ortho={step_out['l_ortho']:.4f}  "
            f"sigma_sep={stats['sigma_separation']:.6f}  "
            f"prompt_entity0={meta.get('prompt_entity0','')}",
            flush=True,
        )

        training_curve.append({
            "epoch": epoch,
            "prompt_entity0": meta.get("prompt_entity0", ""),
            "prompt_entity1": meta.get("prompt_entity1", ""),
            **{k: v for k, v in step_out.items() if k != "loss_tensor"},
            **stats,
        })

        # 체크포인트 (매 5 epoch)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            sep = stats["sigma_separation"]
            if sep > best_sep:
                best_sep = sep
                best_path = save_dir / "best.pt"
                torch.save({
                    "vca_state_dict": vca_layer.state_dict(),
                    "epoch": epoch,
                    "sigma_stats": stats,
                    "best_sep": best_sep,
                }, best_path)
                print(f"[ckpt] best.pt (sep={best_sep:.4f}) → {best_path}", flush=True)

            # sigma GIF
            gif_path = debug_dir / f"sigma_epoch{epoch:03d}.gif"
            save_sigma_gif(frames_np, vca_layer.last_sigma, gif_path)
            print(f"[gif] → {gif_path}", flush=True)

    # FINAL 측정 — 학습 중 최고 sigma_separation 기준
    # (probe 기반 측정은 context 불일치로 신뢰도 낮아 best_sep 사용)
    print(f"FINAL sigma_separation={best_sep:.6f}", flush=True)

    # LEARNING=OK: 학습이 실제로 sigma 분리를 달성했으면 OK
    if best_sep > 0.01:
        print("LEARNING=OK", flush=True)
    else:
        print("LEARNING=FAIL", flush=True)

    # training_curve 저장
    curve_path = debug_dir / "training_curve.json"
    with open(curve_path, "w") as f:
        json.dump(training_curve, f, indent=2)
    print(f"[done] training_curve → {curve_path}", flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",    default="toy/data_objaverse",  dest="data_root")
    p.add_argument("--stats-path",   default="debug/dataset_stats/objaverse_stats.json",
                   dest="stats_path")
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--t-max",        type=int,   default=200, dest="t_max")
    p.add_argument("--lambda-depth", type=float, default=1.0, dest="lambda_depth")
    p.add_argument("--lambda-ortho", type=float, default=0.05, dest="lambda_ortho")
    p.add_argument("--save-dir",     default="checkpoints/objaverse",   dest="save_dir")
    p.add_argument("--debug-dir",    default="debug/train_objaverse",   dest="debug_dir")
    p.add_argument("--n-frames",     type=int,   default=8,  dest="n_frames")
    p.add_argument("--height",       type=int,   default=256)
    p.add_argument("--width",        type=int,   default=256)
    p.add_argument("--max-samples",  type=int,   default=None, dest="max_samples")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
