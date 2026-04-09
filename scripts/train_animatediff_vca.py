"""
Phase 12: AnimateDiff VCA 실제 파인튜닝

AnimateDiff UNet frozen, VCALayer(LoRA + depth_pe)만 학습.
toy/data/{chain,robot_arm} 영상에서 실제 mid_block hidden_states를 supervision으로 사용.

주의:
  FM-A1 : hook 금지 → unet.set_attn_processor()
  FM-A2 : VCALayer fp32, UNet fp16, dtype 변환 명시
  FM-I6 : loss 계산은 last_sigma_raw (grad 있음). last_sigma(detach)로는 gradient 0.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio
from torch.nn.functional import layer_norm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.losses import l_ortho as loss_ortho, l_depth_ranking, l_diff as loss_diff
from scripts.debug_gif import make_debug_gif
from scripts.run_animatediff import load_pipeline, FixedContextVCAProcessor


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_frames(scenario: str, n_frames: int = 8, height: int = 256, width: int = 256):
    """toy/data/{scenario}/front_right/frames/*.png → (T,H,W,3) uint8"""
    from PIL import Image
    frame_dir = Path(f'toy/data/{scenario}/front_right/frames')
    paths = sorted(frame_dir.glob('*.png'))[:n_frames]
    frames = []
    for p in paths:
        img = Image.open(p).convert('RGB').resize((width, height))
        frames.append(np.array(img))
    return np.stack(frames, axis=0)  # (T, H, W, 3)


def load_depths_and_masks(scenario: str, n_frames: int = 8,
                          height: int = 256, width: int = 256, n_entities: int = 2):
    """depth npy + entity mask → depth_order per frame"""
    from PIL import Image
    base = Path(f'toy/data/{scenario}/front_right')
    depth_paths = sorted((base / 'depth').glob('*.npy'))[:n_frames]
    depth_orders = []
    for fidx, dp in enumerate(depth_paths):
        depth = np.load(dp)  # (H_orig, W_orig)
        depths_mean = []
        for ei in range(n_entities):
            mask_path = base / 'mask' / f'{fidx:04d}_entity{ei}.png'
            mask_arr = np.array(Image.open(mask_path).resize((width, height))) > 128
            if mask_arr.sum() > 0:
                # resize depth to match
                import cv2
                depth_r = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
                depths_mean.append(float(depth_r[mask_arr].mean()))
            else:
                depths_mean.append(float('inf'))
        # lower depth = closer = front
        front = int(np.argmin(depths_mean))
        back  = 1 - front
        depth_orders.append([front, back])
    return depth_orders  # list of [front_idx, back_idx] per frame


# ─── VAE 인코딩 ──────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_frames_to_latents(pipe, frames_np: np.ndarray, device: str) -> torch.Tensor:
    """
    frames_np: (T, H, W, 3) uint8
    → latents: (1, 4, T, H//8, W//8)
    """
    frames_tensor = torch.from_numpy(frames_np).float() / 127.5 - 1.0  # [-1,1]
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)   # (T, 3, H, W)
    frames_tensor = frames_tensor.to(device, dtype=torch.float16)

    # VAE는 프레임별로 처리
    latent_list = []
    for i in range(frames_tensor.shape[0]):
        lat = pipe.vae.encode(frames_tensor[i:i+1]).latent_dist.sample()
        lat = lat * pipe.vae.config.scaling_factor
        latent_list.append(lat)
    latents = torch.cat(latent_list, dim=0)  # (T, 4, H//8, W//8)
    # AnimateDiff 형태: (1, 4, T, H//8, W//8)
    latents = latents.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 4, T, H//8, W//8)
    return latents


# ─── VCA 주입 (학습용 — last_sigma_raw 수집) ─────────────────────────────────

class TrainVCAProcessor:
    """
    학습용 VCAProcessor. FixedContextVCAProcessor와 달리
    last_sigma_raw (with grad)를 vca_layer에 유지한다.
    """
    def __init__(self, vca_layer: VCALayer, entity_context: torch.Tensor):
        self.vca = vca_layer               # fp32
        self.entity_context = entity_context  # (1, N, 768) fp32

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        BF = hidden_states.shape[0]
        ctx = self.entity_context.expand(BF, -1, -1)
        # FM-A2: fp16 → fp32, layer_norm으로 score 포화 방지
        # 고노이즈 timestep에서 hidden_states가 매우 크면 sigmoid 포화 → gradient=0
        x = layer_norm(hidden_states.float(), hidden_states.shape[-1:])
        vca_out = self.vca(x, ctx)        # (BF, S, D), fp32; sets last_sigma_raw
        attn_out = vca_out - x            # 잔차 제거 (이중 잔차 방지)
        return attn_out.to(hidden_states.dtype)


def inject_vca_train(pipe, entity_context: torch.Tensor):
    """
    mid_block attn2에 TrainVCAProcessor 주입.
    Returns: vca_layer, injected_keys, original_procs
    """
    unet = pipe.unet
    vca_layer = VCALayer(
        query_dim=1280, context_dim=768,
        n_heads=8, n_entities=2, z_bins=2, lora_rank=8,
        use_softmax=False,
    ).to(pipe.device)  # fp32

    processor = TrainVCAProcessor(vca_layer, entity_context)
    original_procs = copy.copy(dict(unet.attn_processors))
    new_procs = {}
    injected_keys = []
    for key, proc in original_procs.items():
        if 'mid_block' in key and 'attn2' in key:
            new_procs[key] = processor
            injected_keys.append(key)
        else:
            new_procs[key] = proc
    unet.set_attn_processor(new_procs)
    print(f"[inject] VCA → {injected_keys}", flush=True)
    return vca_layer, injected_keys, original_procs


# ─── entity_context 빌드 ─────────────────────────────────────────────────────

def get_entity_embedding_mean(pipe, text: str) -> torch.Tensor:
    """텍스트 → CLIP 텍스트 토큰 평균 임베딩 (1, 1, 768)

    BOS(pos 0) 대신 실제 텍스트 토큰들의 평균 사용.
    'red chain link' vs 'blue chain link' 처럼 비슷한 텍스트도 명확히 구분.

    BOS 토큰은 Transformer의 self-attention을 거친 후에도
    고도로 유사한 텍스트에서 거의 같은 값을 냄 → entity_ctx[0] ≈ entity_ctx[1] → sigma 동일.
    """
    tokens = pipe.tokenizer(
        text, return_tensors='pt', padding='max_length',
        max_length=pipe.tokenizer.model_max_length, truncation=True,
    ).to(pipe.device)
    with torch.no_grad():
        emb = pipe.text_encoder(**tokens).last_hidden_state  # (1, 77, 768)
    # EOS 이전 텍스트 토큰들 (BOS 제외, EOS/PAD 제외) 평균
    input_ids = tokens.input_ids[0]
    eos_id = pipe.tokenizer.eos_token_id
    eos_positions = (input_ids == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        eos_pos = int(eos_positions[0].item())
        # 텍스트 토큰: BOS(0) 제외, EOS(eos_pos) 제외 → [1, eos_pos)
        text_emb = emb[:, 1:max(2, eos_pos), :].mean(dim=1, keepdim=True)
    else:
        text_emb = emb[:, 1:, :].mean(dim=1, keepdim=True)
    return text_emb  # (1, 1, 768)


def build_entity_context(pipe, scenario: str, device: str) -> torch.Tensor:
    """prompts.json → (1, 2, 768) fp32"""
    with open('toy/data/prompts.json') as f:
        prompts = json.load(f)
    info = prompts[scenario]
    e0 = get_entity_embedding_mean(pipe, info['entity_0'])  # (1,1,768)
    e1 = get_entity_embedding_mean(pipe, info['entity_1'])  # (1,1,768)
    ctx = torch.cat([e0, e1], dim=1).float()                # (1,2,768) fp32
    diff_norm = float((ctx[0, 0] - ctx[0, 1]).norm())
    print(f"[ctx] entity embedding diff_norm={diff_norm:.4f}  "
          f"(should be >> 0 for good entity separation)", flush=True)
    return ctx.to(device)


# ─── 학습 단계 ───────────────────────────────────────────────────────────────

def training_step(
    pipe,
    vca_layer: VCALayer,
    latents: torch.Tensor,       # (1, 4, T, H//8, W//8)
    encoder_hidden_states: torch.Tensor,  # (1, 77, 768) fp16
    depth_orders: list,          # list of [front, back] per frame
    lambda_depth: float,
    lambda_ortho: float,
    device: str,
) -> dict:
    """한 배치 (= 전체 T 프레임) forward + loss 계산."""
    T = latents.shape[2]

    # ── noise 추가 ─────────────────────────────────────────────────────────
    noise = torch.randn_like(latents)
    t = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,), device=device).long()
    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

    # ── UNet forward (fp16, autocast) ──────────────────────────────────────
    # UNet은 frozen이므로 gradient 필요 없음. VCA processor가 grad를 수집함.
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        pred_noise = pipe.unet(
            noisy_latents,
            t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

    # ── L_diff ─────────────────────────────────────────────────────────────
    # fp32로 계산 (autocast 밖)
    ld = loss_diff(pred_noise.float(), noise.float())

    # ── sigma_raw 수집 (FM-I6: last_sigma_raw, grad 있음) ──────────────────
    sigma_raw = vca_layer.last_sigma_raw   # (BF, S, N, Z) or None
    if sigma_raw is None:
        l_depth = torch.tensor(0.0, device=device)
        l_ort   = torch.tensor(0.0, device=device)
    else:
        # depth_order: T 프레임 평균 (front가 0인 비율로 결정)
        front_votes = sum(1 for d in depth_orders if d[0] == 0)
        if front_votes >= len(depth_orders) // 2:
            order = [0, 1]
        else:
            order = [1, 0]
        l_depth = l_depth_ranking(sigma_raw, order)
        l_ort   = loss_ortho(vca_layer.depth_pe)

    loss = ld + lambda_depth * l_depth + lambda_ortho * l_ort

    return {
        'loss':    loss.detach().item(),
        'l_diff':  ld.detach().item(),
        'l_depth': l_depth.detach().item(),
        'l_ortho': l_ort.detach().item(),
        'loss_tensor': loss,
        'sigma_raw': sigma_raw,
    }


# ─── sigma 지표 계산 ──────────────────────────────────────────────────────────

def compute_sigma_stats_train(sigma: torch.Tensor) -> dict:
    """(BF, S, N, Z) → sigma_separation, sigma_consistency

    z=0 슬라이스만 사용: l_depth_ranking이 z=0에서 front>back을 직접 학습하므로
    z=1은 미학습 상태이며 평균에 포함하면 신호 희석.
    """
    if sigma is None:
        return {'sigma_separation': 0.0, 'sigma_consistency': 0.0}
    with torch.no_grad():
        s = sigma if not sigma.requires_grad else sigma.detach()
        e0 = float(s[:, :, 0, 0].mean())   # z=0, entity 0
        e1 = float(s[:, :, 1, 0].mean())   # z=0, entity 1
        sep  = abs(e0 - e1)
        cons = float(e0 > e1)
    return {'sigma_separation': sep, 'sigma_consistency': cons, 'e0_z0': e0, 'e1_z0': e1}


# ─── debug sigma GIF (T 프레임 × [RGB | E0σ | E1σ]) ─────────────────────────

def save_sigma_gif(frames_np: np.ndarray, sigma: torch.Tensor,
                   out_path: Path, panel_size: int = 256) -> None:
    """sigma (BF, S, N, Z) → 3-panel GIF"""
    if sigma is None:
        return
    T = frames_np.shape[0]
    hw = int(sigma.shape[1] ** 0.5)
    s = sigma[:T].detach().cpu()   # (T, S, N, Z)
    # z=0 slice
    e0_map = s[:, :, 0, 0].reshape(T, hw, hw)  # (T, hw, hw)
    e1_map = s[:, :, 1, 0].reshape(T, hw, hw)

    from PIL import Image
    panels = []
    for fi in range(T):
        rgb = Image.fromarray(frames_np[fi]).resize((panel_size, panel_size))

        def to_heatmap(m):
            arr = (m.numpy() * 255).clip(0, 255).astype(np.uint8)
            arr = np.array(Image.fromarray(arr).resize((panel_size, panel_size)))
            heat = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
            heat[:, :, 0] = arr  # red channel
            return heat

        e0_img = to_heatmap(e0_map[fi])
        e1_img = to_heatmap(e1_map[fi])
        panel = np.concatenate([np.array(rgb), e0_img, e1_img], axis=1)
        panels.append(panel)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(out_path), panels, fps=8, loop=0)


# ─── 최종 영상 생성 (학습된 VCA 주입 후) ─────────────────────────────────────

def generate_final_videos(pipe, vca_layer: VCALayer, entity_context: torch.Tensor,
                          scenario: str, debug_dir: Path,
                          num_frames: int = 16, steps: int = 20,
                          height: int = 256, width: int = 256, seed: int = 42):
    """학습된 VCA를 주입하고 baseline vs generated GIF 저장."""
    from scripts.run_animatediff import generate_video, make_sigma_debug_gif

    with open('toy/data/prompts.json') as f:
        prompts_dict = json.load(f)
    info = prompts_dict.get(scenario, {})
    prompt = f"{info.get('entity_0','entity0')} and {info.get('entity_1','entity1')}"

    # baseline (VCA 없이)
    print("[final] Generating baseline...", flush=True)
    baseline_frames = generate_video(pipe, prompt, num_frames=num_frames,
                                     steps=steps, height=height, width=width, seed=seed)

    # 학습된 VCA 주입
    proc = FixedContextVCAProcessor(vca_layer, entity_context.half())
    original_procs = copy.copy(dict(pipe.unet.attn_processors))
    new_procs = {}
    for key, p in original_procs.items():
        if 'mid_block' in key and 'attn2' in key:
            new_procs[key] = proc
        else:
            new_procs[key] = p
    pipe.unet.set_attn_processor(new_procs)

    print("[final] Generating with trained VCA...", flush=True)
    gen_frames = generate_video(pipe, prompt, num_frames=num_frames,
                                steps=steps, height=height, width=width, seed=seed)

    # 복구
    pipe.unet.set_attn_processor(original_procs)

    # 저장
    debug_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(debug_dir / 'final_baseline.gif'), baseline_frames, fps=8, loop=0)
    iio.imwrite(str(debug_dir / 'final_generated.gif'), gen_frames, fps=8, loop=0)
    print(f"[final] Saved baseline + generated GIFs → {debug_dir}", flush=True)


# ─── 메인 학습 루프 ──────────────────────────────────────────────────────────

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[init] device={device}", flush=True)

    # 시나리오 목록
    scenarios = ['chain', 'robot_arm'] if args.scenario == 'all' else [args.scenario]

    # 파이프라인 로드
    pipe = load_pipeline(device=device, dtype=torch.float16)

    # UNet frozen
    for param in pipe.unet.parameters():
        param.requires_grad = False

    frozen_count = sum(1 for p in pipe.unet.parameters() if p.requires_grad)
    assert frozen_count == 0, f"UNet has {frozen_count} trainable params — must be 0!"
    print(f"[init] UNet frozen. Trainable params: 0", flush=True)
    # gradient checkpointing 비활성화: 4프레임 256×256은 충분히 가벼움,
    # 커스텀 processor + checkpointing 조합의 grad_fn 순서 문제 방지

    save_dir  = Path(args.save_dir)
    debug_dir = Path(args.debug_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    training_curve = []

    for scenario in scenarios:
        print(f"\n{'='*60}", flush=True)
        print(f"[scenario] {scenario}", flush=True)
        print(f"{'='*60}", flush=True)

        # ── 데이터 로드 ────────────────────────────────────────────────────
        frames_np = load_frames(scenario, n_frames=args.n_frames,
                                height=args.height, width=args.width)
        depth_orders = load_depths_and_masks(scenario, n_frames=args.n_frames,
                                             height=args.height, width=args.width)
        print(f"[data] frames={frames_np.shape}, depth_orders={depth_orders[:3]}", flush=True)

        # ── latent 인코딩 (once) ───────────────────────────────────────────
        latents = encode_frames_to_latents(pipe, frames_np, device)
        print(f"[vae] latents.shape={latents.shape}", flush=True)

        # ── entity_context ─────────────────────────────────────────────────
        entity_ctx = build_entity_context(pipe, scenario, device)  # (1,2,768) fp32
        print(f"[ctx] entity_context.shape={entity_ctx.shape}", flush=True)

        # ── encoder_hidden_states (프롬프트 임베딩) ────────────────────────
        with open('toy/data/prompts.json') as f:
            prompts_dict = json.load(f)
        info = prompts_dict[scenario]
        full_prompt = f"{info['entity_0']} and {info['entity_1']}"
        tokens = pipe.tokenizer(full_prompt, return_tensors='pt',
                                padding='max_length',
                                max_length=pipe.tokenizer.model_max_length,
                                truncation=True).to(device)
        with torch.no_grad():
            enc_hs = pipe.text_encoder(**tokens).last_hidden_state.half()  # (1,77,768)

        # ── VCA 주입 ───────────────────────────────────────────────────────
        vca_layer, injected_keys, original_procs = inject_vca_train(pipe, entity_ctx)
        print(f"[inject] keys={injected_keys}", flush=True)

        # ── optimizer (LoRA + depth_pe만) ──────────────────────────────────
        trainable = [p for p in vca_layer.parameters() if p.requires_grad]
        print(f"[opt] trainable params: {sum(p.numel() for p in trainable):,}", flush=True)
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

        # BEFORE 지표: 실제 latents로 UNet forward → 진짜 hidden_states에서 sigma 측정
        # (랜덤 프로브는 E[sigmoid(Q@K)] ≈ 0.5 by symmetry → sep=0 항상, 의미 없음)
        vca_layer.eval()
        with torch.no_grad():
            noise_b = torch.randn_like(latents)
            t_b = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,),
                                device=device).long()
            noisy_b = pipe.scheduler.add_noise(latents, noise_b, t_b)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _ = pipe.unet(noisy_b, t_b, encoder_hidden_states=enc_hs).sample
        before_stats = compute_sigma_stats_train(vca_layer.last_sigma)
        vca_layer.train()
        print(f"[BEFORE] sigma_separation={before_stats['sigma_separation']:.6f}  "
              f"sigma_consistency={before_stats['sigma_consistency']:.4f}  "
              f"e0_z0={before_stats.get('e0_z0', 0):.4f}  "
              f"e1_z0={before_stats.get('e1_z0', 0):.4f}", flush=True)

        # ── 에폭 루프 ──────────────────────────────────────────────────────
        loss_history = []
        for epoch in range(args.epochs):
            vca_layer.train()
            optimizer.zero_grad()

            step_out = training_step(
                pipe, vca_layer, latents, enc_hs,
                depth_orders, args.lambda_depth, args.lambda_ortho, device,
            )

            step_out['loss_tensor'].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            # sigma 지표 (detached last_sigma)
            sigma_det = vca_layer.last_sigma
            stats = compute_sigma_stats_train(sigma_det)
            loss_history.append(step_out['loss'])

            print(
                f"epoch={epoch:3d}  "
                f"loss={step_out['loss']:.4f}  "
                f"l_diff={step_out['l_diff']:.4f}  "
                f"l_depth={step_out['l_depth']:.4f}  "
                f"l_ortho={step_out['l_ortho']:.4f}  "
                f"sigma_sep={stats['sigma_separation']:.6f}  "
                f"e0={stats.get('e0_z0', 0):.4f}  e1={stats.get('e1_z0', 0):.4f}",
                flush=True,
            )

            training_curve.append({
                'epoch': epoch, 'scenario': scenario,
                **{k: v for k, v in step_out.items()
                   if k not in ('loss_tensor', 'sigma_raw')},
                **stats,
            })

            # 체크포인트 & sigma GIF (매 5 에폭)
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                ckpt = {
                    'vca_state_dict': vca_layer.state_dict(),
                    'epoch': epoch,
                    'loss': step_out['loss'],
                    'scenario': scenario,
                }
                ckpt_path = save_dir / f'{scenario}_epoch{epoch:03d}.pt'
                torch.save(ckpt, ckpt_path)
                print(f"[ckpt] saved → {ckpt_path}", flush=True)

                gif_path = debug_dir / f'{scenario}_sigma_epoch{epoch:03d}.gif'
                save_sigma_gif(frames_np, vca_layer.last_sigma, gif_path)
                print(f"[gif] sigma saved → {gif_path}", flush=True)

        # ── AFTER 지표: 실제 latents로 UNet forward ────────────────────────
        vca_layer.eval()
        with torch.no_grad():
            noise_a = torch.randn_like(latents)
            t_a = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,),
                                device=device).long()
            noisy_a = pipe.scheduler.add_noise(latents, noise_a, t_a)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _ = pipe.unet(noisy_a, t_a, encoder_hidden_states=enc_hs).sample
        after_sigma = vca_layer.last_sigma
        after_stats = compute_sigma_stats_train(after_sigma)
        vca_layer.train()

        print(f"[AFTER]  sigma_separation={after_stats['sigma_separation']:.6f}  "
              f"sigma_consistency={after_stats['sigma_consistency']:.4f}  "
              f"e0_z0={after_stats.get('e0_z0', 0):.4f}  "
              f"e1_z0={after_stats.get('e1_z0', 0):.4f}", flush=True)

        # LEARNING 판정
        l_depth_first = next((c['l_depth'] for c in training_curve
                              if c['scenario'] == scenario), 1.0)
        l_depth_last  = next((c['l_depth'] for c in reversed(training_curve)
                              if c['scenario'] == scenario), 1.0)
        depth_decreased = l_depth_last < l_depth_first

        ok = (
            after_stats['sigma_separation'] > before_stats['sigma_separation']
            and after_stats['sigma_consistency'] > 0.55
            and depth_decreased
        )

        # best.pt 저장
        best_path = save_dir / f'{scenario}_best.pt'
        torch.save({'vca_state_dict': vca_layer.state_dict(),
                    'epoch': args.epochs - 1, 'sigma_stats': after_stats,
                    'scenario': scenario}, best_path)
        print(f"[ckpt] best saved → {best_path}", flush=True)

        # 최종 영상 생성
        pipe.unet.set_attn_processor(original_procs)  # 학습용 processor 제거
        generate_final_videos(pipe, vca_layer, entity_ctx, scenario, debug_dir,
                              num_frames=16, steps=20,
                              height=args.height, width=args.width, seed=42)

        # 최종 출력
        print(f"FINAL sigma_separation={after_stats['sigma_separation']:.6f}", flush=True)
        print(f"FINAL sigma_consistency={after_stats['sigma_consistency']:.4f}", flush=True)
        if ok:
            print("LEARNING=OK", flush=True)
        else:
            print("LEARNING=FAIL", flush=True)
            reason = []
            if after_stats['sigma_separation'] <= before_stats['sigma_separation']:
                reason.append(f"sigma_separation did not increase "
                              f"({before_stats['sigma_separation']:.6f} → "
                              f"{after_stats['sigma_separation']:.6f})")
            if after_stats['sigma_consistency'] <= 0.55:
                reason.append(f"sigma_consistency={after_stats['sigma_consistency']:.4f} ≤ 0.55")
            if not depth_decreased:
                reason.append(f"l_depth not decreasing ({l_depth_first:.4f} → {l_depth_last:.4f})")
            print(f"FAIL_REASON: {'; '.join(reason)}", flush=True)

        # 재주입 (다음 scenario를 위해)
        _, injected_keys, original_procs = inject_vca_train(pipe, entity_ctx)
        pipe.unet.set_attn_processor(original_procs)

    # training_curve 저장
    curve_path = debug_dir / 'training_curve.json'
    with open(curve_path, 'w') as f:
        json.dump(training_curve, f, indent=2)
    print(f"[done] training_curve → {curve_path}", flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--scenario',      default='chain',
                   help="chain | robot_arm | all")
    p.add_argument('--epochs',        type=int,   default=20)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--lambda-depth',  type=float, default=0.1, dest='lambda_depth')
    p.add_argument('--lambda-ortho',  type=float, default=0.05, dest='lambda_ortho')
    p.add_argument('--save-dir',      default='checkpoints/animatediff')
    p.add_argument('--debug-dir',     default='debug/train_animatediff')
    p.add_argument('--n-frames',      type=int,   default=8, dest='n_frames')
    p.add_argument('--height',        type=int,   default=256)
    p.add_argument('--width',         type=int,   default=256)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
