"""
Phase 11: AnimateDiff + VCA 주입 → real video generation

검증된 toy 학습(sigma_consistency=0.875)을 실제 AnimateDiff 파이프라인에서 시각화.

FM-A1: hook 금지 → unet.set_attn_processor() 사용
FM-A2: pipe fp16이면 entity_context도 .half()로 맞춤
"""
import sys
import json
import argparse
import copy
from pathlib import Path
from typing import Optional

import torch
torch.manual_seed(42)
import numpy as np
np.random.seed(42)
import imageio.v2 as iio2

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from scripts.debug_gif import make_debug_gif


# ─── 프롬프트 목록 ────────────────────────────────────────────────────────────
PROMPTS = [
    {
        "dir": "chain",
        "full": "Two interlocked chain links, one red and one blue, slowly rotating",
        "entity_0": "a red chain link",
        "entity_1": "a blue chain link",
    },
    {
        "dir": "robot_arm",
        "full": "Two robotic arms, one red and one blue, crossing each other in motion",
        "entity_0": "a red robotic arm",
        "entity_1": "a blue robotic arm",
    },
    {
        "dir": "wrestling",
        "full": "A white cat and a black dog wrestling playfully on the floor",
        "entity_0": "a white cat",
        "entity_1": "a black dog",
    },
]


# ─── FixedContextVCAProcessor ────────────────────────────────────────────────
class FixedContextVCAProcessor:
    """
    entity_context를 미리 고정한 VCA cross-attention processor.

    중요:
      - VCALayer.forward()는 x + attn_out을 반환 (residual 포함)
      - diffusers transformer block은 별도로 residual을 더함
      - 따라서 여기서는 vca_out - x = attn_out 만 반환해야 이중 잔차 방지
    """
    def __init__(self, vca_layer: VCALayer, entity_context: torch.Tensor):
        self.vca = vca_layer
        self.entity_context = entity_context  # (1, N, CD)
        self.call_count = 0

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,                           # (BF, S, D)
        encoder_hidden_states: Optional[torch.Tensor] = None,  # ignored
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        self.call_count += 1
        BF = hidden_states.shape[0]
        ctx = self.entity_context.expand(BF, -1, -1)  # (BF, N, CD)

        # dtype 맞춤 (FM-A2)
        x = hidden_states.float()
        vca_out = self.vca(x, ctx.float())   # (BF, S, D)  = x + attn_out
        attn_out = vca_out - x               # 잔차 제거: transformer block이 처리

        return attn_out.to(hidden_states.dtype)


# ─── 파이프라인 로드 ──────────────────────────────────────────────────────────
def load_pipeline(device: str = 'cuda', dtype=torch.float16):
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

    print("Loading MotionAdapter...", flush=True)
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3",
        torch_dtype=dtype,
    )
    print("Loading base model (emilianJR/epiCRealism)...", flush=True)
    pipe = AnimateDiffPipeline.from_pretrained(
        "emilianJR/epiCRealism",
        motion_adapter=adapter,
        torch_dtype=dtype,
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.enable_vae_slicing()
    pipe = pipe.to(device)
    print(f"Pipeline on {device}, dtype={dtype}", flush=True)
    return pipe


# ─── entity 임베딩 ────────────────────────────────────────────────────────────
def get_entity_embedding(pipe, text: str) -> torch.Tensor:
    """텍스트 → CLIP CLS 임베딩 (1, 1, 768)"""
    tokens = pipe.tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
    ).to(pipe.device)
    with torch.no_grad():
        emb = pipe.text_encoder(**tokens).last_hidden_state  # (1, 77, 768)
    return emb[:, :1, :].clone()  # CLS 토큰: (1, 1, 768)


# ─── VCA 주입 (mid_block attn2만) ────────────────────────────────────────────
def inject_vca_midblock(
    pipe,
    entity_context: torch.Tensor,  # (1, N, CD)
    context_dim: int = 768,
) -> tuple[VCALayer, list, dict]:
    """mid_block의 attn2에만 FixedContextVCAProcessor 주입.

    Returns
    -------
    vca_layer       : 주입된 VCALayer (last_sigma 수집용)
    injected_keys   : 교체된 processor key 목록
    original_procs  : 원래 processor dict (복구용)
    """
    unet = pipe.unet

    # SD 1.5 mid_block hidden_dim = 1280
    mid_block_dim = 1280

    # VCA는 fp32로 유지 (FM-A2: 입출력만 dtype 변환, 내부 계산은 fp32 안정)
    vca_layer = VCALayer(
        query_dim=mid_block_dim,
        context_dim=context_dim,
        n_heads=8,
        n_entities=2,
        z_bins=2,
        lora_rank=8,
        use_softmax=False,
    ).to(pipe.device)

    processor = FixedContextVCAProcessor(vca_layer, entity_context)

    original_procs = copy.copy(dict(unet.attn_processors))

    new_processors = {}
    injected_keys = []
    for key, proc in original_procs.items():
        if 'mid_block' in key and 'attn2' in key:
            new_processors[key] = processor
            injected_keys.append(key)
        else:
            new_processors[key] = proc

    unet.set_attn_processor(new_processors)
    print(f"VCA injected: {injected_keys}", flush=True)
    return vca_layer, injected_keys, original_procs


# ─── 비디오 생성 ──────────────────────────────────────────────────────────────
def generate_video(
    pipe,
    prompt: str,
    num_frames: int = 16,
    steps: int = 20,
    guidance_scale: float = 7.5,
    height: int = 256,
    width: int = 256,
    seed: int = 42,
) -> list:
    """list of np.ndarray (H, W, 3) uint8"""
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    output = pipe(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
        output_type='pil',
    )
    return [np.array(f) for f in output.frames[0]]


# ─── sigma 통계 ───────────────────────────────────────────────────────────────
def compute_sigma_stats(vca_layer: VCALayer, num_frames: int = 16) -> dict:
    """last_sigma로부터 entity 분리 지표 계산.

    AnimateDiff CFG: UNet은 [uncond_frames, cond_frames] 순서로 배치 처리.
    → last_sigma 후반 num_frames가 conditional sigma.
    """
    sigma = vca_layer.last_sigma  # (BF_total, S, N, Z)
    if sigma is None:
        return {k: 0.0 for k in
                ["sigma_separation", "entity0_mean_sigma", "entity1_mean_sigma",
                 "both_high_ratio", "sigma_consistency"]}

    # CFG: 뒷절반이 conditional
    F = min(num_frames, sigma.shape[0] // 2)
    sigma_cond = sigma[-F:]  # (F, S, N, Z)

    e0 = float(sigma_cond[:, :, 0, :].mean())
    e1 = float(sigma_cond[:, :, 1, :].mean())
    both_high = float(
        ((sigma_cond[:, :, 0, :] > 0.5) &
         (sigma_cond[:, :, 1, :] > 0.5)).float().mean()
    )
    return {
        "sigma_separation":    float(abs(e0 - e1)),   # raw precision (no round)
        "entity0_mean_sigma":  float(e0),
        "entity1_mean_sigma":  float(e1),
        "both_high_ratio":     float(both_high),
        "sigma_consistency":   float(e0 > e1),
    }


# ─── debug GIF 생성 ────────────────────────────────────────────────────────────
def make_sigma_debug_gif(
    frames_rgb: list,
    vca_layer: VCALayer,
    num_frames: int,
    out_path: Path,
    panel_size: int = 256,
) -> None:
    """[RGB | E0 sigma | E1 sigma] 3-panel debug GIF.

    last_sigma: (BF_total, S, N, Z) → 후반 conditional F 프레임 사용.
    spatial: S = hw * hw → reshape → (N, hw, hw) per frame.
    """
    sigma = vca_layer.last_sigma
    if sigma is None:
        print("WARNING: last_sigma is None, skipping debug GIF", flush=True)
        return

    F = min(num_frames, sigma.shape[0] // 2)
    sigma_cond = sigma[-F:]        # (F, S, N, Z)
    S = sigma_cond.shape[1]
    hw = int(S ** 0.5)
    if hw * hw != S:
        print(f"WARNING: S={S} is not a perfect square, skipping debug GIF", flush=True)
        return

    # (F, N, hw, hw) — z=0 slice
    sigma_np = (sigma_cond[:, :, :, 0]
                .permute(0, 2, 1)          # (F, N, S)
                .reshape(F, 2, hw, hw)
                .cpu().numpy())

    n = min(F, len(frames_rgb))
    sigma_maps = [sigma_np[f] for f in range(n)]

    make_debug_gif(frames_rgb[:n], sigma_maps, out_path, panel_size=panel_size)
    print(f"debug.gif saved (hw={hw}, {n} frames)", flush=True)


# ─── 단일 프롬프트 실험 ───────────────────────────────────────────────────────
def run_one(
    pipe,
    prompt_cfg: dict,
    out_dir: Path,
    context_dim: int = 768,
    steps: int = 20,
    seed: int = 42,
    num_frames: int = 16,
    height: int = 256,
    width: int = 256,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}", flush=True)
    print(f"[{prompt_cfg['dir']}]  {prompt_cfg['full']}", flush=True)

    # ── 1. Baseline (VCA 없이) ────────────────────────────────────────────────
    print("Generating baseline (no VCA)...", flush=True)
    baseline_frames = generate_video(
        pipe, prompt_cfg['full'],
        num_frames=num_frames, steps=steps,
        height=height, width=width, seed=seed,
    )
    iio2.mimsave(str(out_dir / 'baseline.gif'), baseline_frames, duration=100)
    print(f"  baseline.gif saved ({len(baseline_frames)} frames)", flush=True)

    # ── 2. entity CLIP 임베딩 ─────────────────────────────────────────────────
    e0 = get_entity_embedding(pipe, prompt_cfg['entity_0'])
    e1 = get_entity_embedding(pipe, prompt_cfg['entity_1'])
    entity_context = torch.cat([e0, e1], dim=1)           # (1, 2, 768)
    dtype = next(pipe.unet.parameters()).dtype
    entity_context = entity_context.to(dtype)              # FM-A2: dtype 맞춤

    # ── 3. VCA 주입 ───────────────────────────────────────────────────────────
    vca_layer, injected_keys, original_procs = inject_vca_midblock(
        pipe, entity_context, context_dim,
    )
    assert len(injected_keys) > 0, "No mid_block attn2 found — check UNet architecture"

    # ── 4. VCA 생성 ───────────────────────────────────────────────────────────
    print("Generating with VCA...", flush=True)
    vca_frames = generate_video(
        pipe, prompt_cfg['full'],
        num_frames=num_frames, steps=steps,
        height=height, width=width, seed=seed,
    )
    iio2.mimsave(str(out_dir / 'generated.gif'), vca_frames, duration=100)
    # 주입된 processor의 호출 횟수 확인 (FM-A1 검증)
    proc_key = injected_keys[0]
    call_cnt = pipe.unet.attn_processors[proc_key].call_count
    print(f"  generated.gif saved  |  processor call_count={call_cnt}", flush=True)
    assert call_cnt > 0, "FM-A1 FAIL: FixedContextVCAProcessor was never called!"

    # ── 5. sigma 통계 ─────────────────────────────────────────────────────────
    stats = compute_sigma_stats(vca_layer, num_frames)
    with open(out_dir / 'sigma_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  sigma_stats: {stats}", flush=True)

    # ── 6. debug GIF ──────────────────────────────────────────────────────────
    make_sigma_debug_gif(vca_frames, vca_layer, num_frames,
                         out_dir / 'debug.gif', panel_size=256)

    # ── 7. 원래 processor 복구 (다음 실험 위해) ───────────────────────────────
    pipe.unet.set_attn_processor(original_procs)
    print("  Processors restored.", flush=True)

    return stats


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt',      type=str, default=None)
    parser.add_argument('--entity0',     type=str, default=None)
    parser.add_argument('--entity1',     type=str, default=None)
    parser.add_argument('--out-dir',     type=str, default='debug/animatediff')
    parser.add_argument('--context-dim', type=int, default=768)
    parser.add_argument('--steps',       type=int, default=20)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--num-frames',  type=int, default=16)
    parser.add_argument('--height',      type=int, default=256)
    parser.add_argument('--width',       type=int, default=256)
    parser.add_argument('--run-all',     action='store_true',
                        help='3개 PROMPTS 전부 실행')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = load_pipeline(device=device)

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    if args.run_all:
        all_stats = {}
        for p in PROMPTS:
            stats = run_one(
                pipe, p, out_base / p['dir'],
                context_dim=args.context_dim,
                steps=args.steps, seed=args.seed,
                num_frames=args.num_frames,
                height=args.height, width=args.width,
            )
            all_stats[p['dir']] = stats

        print("\n=== Phase 11 Summary ===", flush=True)
        for k, v in all_stats.items():
            print(f"{k}: separation={v['sigma_separation']:.4f} "
                  f"both_high={v['both_high_ratio']:.4f}",
                  flush=True)

    else:
        if not (args.prompt and args.entity0 and args.entity1):
            parser.error("--prompt, --entity0, --entity1 필요 (또는 --run-all 사용)")
        prompt_cfg = {
            'dir': out_base.name,
            'full': args.prompt,
            'entity_0': args.entity0,
            'entity_1': args.entity1,
        }
        run_one(
            pipe, prompt_cfg, out_base,
            context_dim=args.context_dim,
            steps=args.steps, seed=args.seed,
            num_frames=args.num_frames,
            height=args.height, width=args.width,
        )

    print("Done.", flush=True)


if __name__ == '__main__':
    main()
