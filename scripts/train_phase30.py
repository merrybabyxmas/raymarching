"""
Phase 25: Text attention alignment — cross-attention 직접 supervision

Phase 23/24 문제:
  - text_attn_chart: cross-attention이 frozen이라 gradient 없음 → chart 불변
  - VCA 학습만으로는 text attention alignment 개선 불가

Phase 25 수정:
  1. ATTN_CAPTURE_KEY (up_blocks.3 attn2) to_q/to_k unfreeze
     + TrainableAttnProcessor: weight tensor with grad 보존
  2. l_attn_mask_loss:
     entity 토큰(color+entity) attention mass가 GT entity mask 안에 집중되도록
     → maximize  Σ(attn_weight[entity_tok, in_mask]) / Σ(attn_weight[entity_tok, :])
     프레임별·entity별·GT mask 기준 (spatial token 수준 supervision)
  3. Phase 24 depth fix 포함:
     lambda_depth=5.0, lambda_diff=0.05, ratio abs fix
  4. trainable params: VCA layer + attn2.to_q/to_k (alignment fine-tune만)

목표:
  - text_attn_chart overlap이 random baseline 이상으로 상승
  - depth_rank_accuracy > 0.75
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import layer_norm
from torch.utils.data import DataLoader
from diffusers.models.attention_processor import AttnProcessor2_0

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.losses import l_ortho as loss_ortho, l_depth_ranking, l_diff as loss_diff
from scripts.run_animatediff import load_pipeline, FixedContextVCAProcessor
from scripts.train_animatediff_vca import (
    compute_sigma_stats_train, save_sigma_gif, encode_frames_to_latents,
)
from scripts.train_objaverse_vca import (
    ObjaverseTrainDataset, check_dataset_quality, get_entity_context_from_meta,
)
from scripts.train_phase17 import adaptive_lambda_depth, RATIO_WARNING_THRESH
from scripts.train_phase19 import l_depth_ranking_perframe, PROBE_T_VALUES

# ─── 기본값 ───────────────────────────────────────────────────────────────────
DEFAULT_LAMBDA_DEPTH  = 5.0    # Phase 24/25: depth 집중
DEFAULT_LAMBDA_DIFF   = 0.05   # Phase 24/25: diff loss 감소
DEFAULT_LAMBDA_ORTHO  = 0.005
DEFAULT_LAMBDA_ATTN   = 3.0    # Phase 28: LoRA + 10x lambda (Phase 25 gradient too weak)
DEFAULT_LR            = 5e-5
DEFAULT_EPOCHS        = 60
DEFAULT_T_MAX         = 200
DEPTH_PE_INIT_SCALE   = 0.3
VCA_ALPHA             = 0.3   # VCA 기여 강도: text_attn + alpha*vca_delta

# Phase 22/23 공통: up_blocks.2 주입, up_blocks.3 캡처
INJECT_KEY        = 'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor'
ATTN_CAPTURE_KEY  = 'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor'
INJECT_QUERY_DIM  = 640
CAPTURE_N_HEADS   = 8


# ─── Phase 23: Color-qualified prompts ───────────────────────────────────────

# RGB → 색이름 (meta.json color0/color1 변환용)
_COLOR_TABLE = [
    ((0.6, 0.2, 0.2), "red"),
    ((0.2, 0.2, 0.6), "blue"),
    ((0.2, 0.6, 0.2), "green"),
    ((0.6, 0.6, 0.1), "yellow"),
    ((0.5, 0.1, 0.5), "purple"),
    ((0.6, 0.3, 0.1), "orange"),
    ((0.5, 0.5, 0.5), "gray"),
]

def rgb_to_color_name(rgb: list) -> str:
    """RGB float 리스트 → 가장 가까운 색이름."""
    r, g, b = rgb
    best, best_dist = "colored", 1e9
    for (cr, cg, cb), name in _COLOR_TABLE:
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_dist:
            best_dist, best = d, name
    return best


def make_color_prompts(meta: dict) -> tuple:
    """
    meta.json → color-qualified entity prompts.
    반환: (e0_prompt, e1_prompt, full_prompt, color0_name, color1_name)
    예) "a red cat", "a blue dog", "a red cat and a blue dog"
    """
    c0 = rgb_to_color_name(meta.get("color0", [0.85, 0.15, 0.1]))
    c1 = rgb_to_color_name(meta.get("color1", [0.1,  0.25, 0.85]))
    kw0 = meta.get("keyword0", meta.get("prompt_entity0", "entity0"))
    kw1 = meta.get("keyword1", meta.get("prompt_entity1", "entity1"))
    # keyword가 "a cat" 형식이면 그냥 사용, 아니면 "a {color} {keyword}" 생성
    if kw0.startswith("a "):
        e0 = f"a {c0} {kw0[2:]}"
    else:
        e0 = f"a {c0} {kw0}"
    if kw1.startswith("a "):
        e1 = f"a {c1} {kw1[2:]}"
    else:
        e1 = f"a {c1} {kw1}"
    full = f"{e0} and {e1}"
    return e0, e1, full, c0, c1


def get_color_entity_context(pipe, meta: dict, device: str) -> torch.Tensor:
    """
    Color-qualified 텍스트로 entity CLIP 임베딩 생성.
    반환: (1, 2, 768) fp32
    """
    e0_text, e1_text, _, _, _ = make_color_prompts(meta)
    embs = []
    for text in [e0_text, e1_text]:
        tokens = pipe.tokenizer(
            text, return_tensors="pt",
            padding="max_length", max_length=pipe.tokenizer.model_max_length,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            out = pipe.text_encoder(**tokens).last_hidden_state  # (1, 77, 768)
        input_ids = tokens.input_ids[0]
        eos_id = pipe.tokenizer.eos_token_id
        mask = (input_ids != pipe.tokenizer.pad_token_id) & (input_ids != eos_id)
        mask[0] = False  # BOS 제외
        emb = out[0][mask].mean(0)  # (768,)
        embs.append(emb)
    ctx = torch.stack(embs, dim=0).unsqueeze(0).float()  # (1, 2, 768)
    return ctx.to(device)


# ─── Phase 23: Dataset with entity masks ─────────────────────────────────────

class ObjaverseDatasetWithMasks(ObjaverseTrainDataset):
    """
    ObjaverseTrainDataset 확장: 프레임별 entity spatial mask 추가 반환.

    mask/{fi:04d}_entity{n}.png (256×256 binary) → 16×16 다운샘플 → (T, 2, S) bool
    S = VCA_HW² = 256  (up_blocks.2 spatial 해상도 = 16×16)

    l_zorder_direct에서 entity별 spatial token만 선택적으로 loss에 반영:
      - 배경 픽셀 제외 → 신호 희석 방지
      - 프레임마다 entity 위치가 다를 수 있으므로 per-frame mask 필수
    """
    VCA_HW = 16  # up_blocks.2 attention spatial dim

    def __getitem__(self, idx):
        frames_np, depths_np, depth_orders, meta = super().__getitem__(idx)
        s     = self.samples[idx]
        seq_dir = s["dir"]
        T = frames_np.shape[0]

        from PIL import Image as _PIL_
        masks_per_frame = []
        for fi in range(T):
            m0_p = seq_dir / "mask" / f"{fi:04d}_entity0.png"
            m1_p = seq_dir / "mask" / f"{fi:04d}_entity1.png"
            if m0_p.exists() and m1_p.exists():
                m0 = np.array(_PIL_.open(m0_p).convert('L').resize(
                    (self.VCA_HW, self.VCA_HW), _PIL_.NEAREST)) > 128
                m1 = np.array(_PIL_.open(m1_p).convert('L').resize(
                    (self.VCA_HW, self.VCA_HW), _PIL_.NEAREST)) > 128
            else:
                # 마스크 없으면 전체 spatial 사용 (fallback)
                m0 = np.ones((self.VCA_HW, self.VCA_HW), dtype=bool)
                m1 = np.ones((self.VCA_HW, self.VCA_HW), dtype=bool)
            # (2, S=VCA_HW²)
            masks_per_frame.append(np.stack([m0.ravel(), m1.ravel()], axis=0))

        entity_masks = np.stack(masks_per_frame, axis=0)  # (T, 2, S)
        return frames_np, depths_np, depth_orders, meta, entity_masks


# ─── Phase 23: Direct z-order loss ───────────────────────────────────────────

def l_zorder_direct(sigma_acc: list, depth_orders: list,
                    entity_masks: np.ndarray | None = None) -> torch.Tensor:
    """
    GT depth order를 z-bin 할당으로 직접 학습.

    σ(front entity, z=0)  최대화  → 앞 entity는 near bin에 집중
    σ(back  entity, z=-1) 최대화  → 뒤 entity는 far  bin에 집중
    σ(front entity, z=-1) 최소화  (반대 방향 억제)
    σ(back  entity, z=0)  최소화

    entity_masks: (BF, 2, S) bool numpy — entity별 spatial mask (16×16 flatten)
      - None이면 전체 spatial 평균 (기존 동작)
      - 마스크 사용 시 entity가 실제 존재하는 spatial token만 loss에 반영
        → 배경 픽셀 제외, 프레임/entity별 위치 차이 반영

    l_depth_ranking_perframe 대체:
    ranking loss는 front>back 비교만 → z-bin 자체 할당 미학습
    z-order loss는 각 z-bin이 올바른 entity를 직접 활성화하도록 강제
    """
    loss = torch.tensor(0.0)
    count = 0
    for sigma in sigma_acc:                         # (BF, S, N, Z), requires_grad
        if loss.device != sigma.device:
            loss = loss.to(sigma.device)
        BF = sigma.shape[0]
        Z  = sigma.shape[3]
        n  = min(BF, len(depth_orders))
        for fi in range(n):
            front = int(depth_orders[fi][0])
            back  = int(depth_orders[fi][1])

            if entity_masks is not None and fi < entity_masks.shape[0]:
                # entity별 spatial mask → 실제 entity 위치의 token만 선택
                mf = torch.from_numpy(entity_masks[fi, front]).to(sigma.device)
                mb = torch.from_numpy(entity_masks[fi, back]).to(sigma.device)
                if mf.sum() == 0: mf = torch.ones_like(mf)
                if mb.sum() == 0: mb = torch.ones_like(mb)
                s_fn = sigma[fi][mf, front, 0].mean()     # front entity near ↑
                s_bf = sigma[fi][mb, back,  Z-1].mean()   # back  entity far  ↑
                s_ff = sigma[fi][mf, front, Z-1].mean()   # front entity far  ↓
                s_bn = sigma[fi][mb, back,  0].mean()     # back  entity near ↓
            else:
                s_fn = sigma[fi, :, front, 0].mean()
                s_bf = sigma[fi, :, back,  Z-1].mean()
                s_ff = sigma[fi, :, front, Z-1].mean()
                s_bn = sigma[fi, :, back,  0].mean()

            loss = loss + (-s_fn - s_bf + s_ff + s_bn)
            count += 1
    return loss / max(count, 1)


# ─── Phase 27: LoRA-based text attention alignment ───────────────────────────

def _get_attn_module(unet, processor_key: str):
    """
    'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor'
    → up_blocks.3의 attn2 nn.Module 반환 (processor 앞 경로 navigate)
    """
    path = processor_key.replace('.processor', '').split('.')
    mod = unet
    for part in path:
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


import torch.nn as nn

class LoRALayer(nn.Module):
    """
    Phase 27: Low-Rank Adaptation for attn2 to_q / to_k.

    원본 fp16 weights는 완전 frozen.
    fp32 저랭크 행렬 (A×B)만 학습 → NaN 문제 근본 해결.

    delta_q = B_q @ A_q @ hidden_states   (rank=4 bottleneck)
    delta_k = B_k @ A_k @ ctx

    초기화: B=0 → 학습 시작 시 delta=0 (모델 동작 무변화)
    """
    def __init__(self, q_dim: int, context_dim: int, rank: int = 4):
        super().__init__()
        self.lora_A_q = nn.Linear(q_dim,       rank, bias=False)
        self.lora_B_q = nn.Linear(rank,         q_dim, bias=False)
        self.lora_A_k = nn.Linear(context_dim,  rank, bias=False)
        self.lora_B_k = nn.Linear(rank,         q_dim, bias=False)
        # B=0 init: 시작 시 delta=0 (모델 동작 보존)
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.zeros_(self.lora_B_k.weight)
        # A ~ N(0, 1/rank): 작은 초기값
        nn.init.normal_(self.lora_A_q.weight, std=1.0 / rank)
        nn.init.normal_(self.lora_A_k.weight, std=1.0 / rank)

    def delta_q(self, x: torch.Tensor) -> torch.Tensor:
        """(BF, S, q_dim) fp16 → LoRA delta fp16"""
        with torch.amp.autocast("cuda", enabled=False):
            return self.lora_B_q(self.lora_A_q(x.float())).to(x.dtype)

    def delta_k(self, x: torch.Tensor) -> torch.Tensor:
        """(BF, L, context_dim) fp16 → LoRA delta fp16"""
        with torch.amp.autocast("cuda", enabled=False):
            return self.lora_B_k(self.lora_A_k(x.float())).to(x.dtype)


class LoRAAttnProcessor:
    """
    Phase 27: frozen to_q/to_k + LoRA delta (fp32, 안정적).

    Phase 25/26 문제: fp16 to_q/to_k 직접 학습 → NaN (optimizer first step overflow)
    Phase 27 해결: 원본 fp16 weights 완전 frozen + 별도 fp32 LoRA layer만 학습
    """
    def __init__(self, lora_layer: LoRALayer):
        self.lora            = lora_layer
        self.last_weights        = None   # numpy, no grad (debug용)
        self.last_weights_tensor = None   # float32 WITH grad (l_attn_mask_loss 전용)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        residual = hidden_states  # original (for residual connection gradient path)
        ctx = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        # Q/K 입력을 detach: l_attn gradient가 UNet hidden states로 역전파 안 됨
        # → VCA parameter에 l_attn gradient 오염 없음 (Phase 26/27 NaN 근본 원인 해결)
        # l_diff gradient는 v와 residual 경로로 정상 흐름 (VCA 학습 유지)
        hs_det  = hidden_states.detach()
        ctx_det = ctx.detach()

        q = attn.to_q(hs_det)             # fp16, gradient: q → hs_det (stops)
        k = attn.to_k(ctx_det)            # fp16
        v = attn.head_to_batch_dim(attn.to_v(ctx))   # fp16, ctx not detached (v path)

        # LoRA delta: fp32 → fp16, gradient flows to LoRA.A,B ONLY
        q = q + self.lora.delta_q(hs_det)    # gradient: q → delta_q → LoRA
        k = k + self.lora.delta_k(ctx_det)   # gradient: k → delta_k → LoRA

        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)

        # float32 scores — beta=0 필수: torch.empty는 미초기화값 포함, beta=1이면 NaN 발생
        scores = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1],
                        dtype=torch.float32, device=q.device),
            q.float(), k.float().transpose(-2, -1), beta=0, alpha=float(attn.scale),
        )
        if attention_mask is not None:
            scores = scores + attention_mask.float()

        weights_f32 = scores.softmax(dim=-1)   # float32, has grad → LoRA
        self.last_weights        = weights_f32.detach().cpu().numpy()
        self.last_weights_tensor = weights_f32

        # forward output detached: l_diff/l_depth이 LoRA로 역전파 안 됨
        out = torch.bmm(weights_f32.detach().to(v.dtype), v)
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if attn.residual_connection:
            out = out + residual.to(out.dtype)
        out = out / attn.rescale_output_factor
        return out


def l_attn_mask_loss(weights_tensor, entity_masks: np.ndarray,
                     tok_idx_e0: list, tok_idx_e1: list, n_heads: int) -> torch.Tensor:
    """
    text attention이 GT entity mask 내 spatial token에 집중되도록 최대화.

    weights_tensor : (BF*heads, S_attn, L) with grad
    entity_masks   : (BF, 2, S_mask) bool numpy  — 16×16 flatten
    tok_idx_e0/e1  : [color_tok..., entity_tok...] per entity
    n_heads        : attention head 수

    반환: scalar loss (음수 = maximize overlap)
    """
    weights_tensor = weights_tensor.float()   # ensure float32 for numerical stability
    BFh, S_attn, L = weights_tensor.shape
    BF = BFh // n_heads

    # (BF*h, S, L) → (BF, h, S, L) → mean over heads → (BF, S, L)
    W_4d  = weights_tensor.reshape(BF, n_heads, S_attn, L)
    W_mean = W_4d.mean(dim=1)   # (BF, S_attn, L), grad preserved

    loss  = torch.zeros(1, device=weights_tensor.device)
    count = 0

    for fi in range(min(BF, entity_masks.shape[0])):
        for ent_idx, tok_idxs in enumerate([tok_idx_e0, tok_idx_e1]):
            valid_toks = [t for t in tok_idxs if t < L]
            if not valid_toks:
                continue

            raw_mask = entity_masks[fi, ent_idx]            # (S_mask,) bool numpy
            if raw_mask.sum() == 0:
                continue

            # mask를 attention 공간 해상도로 resize
            if len(raw_mask) != S_attn:
                from PIL import Image as _pi
                hw_m = max(1, int(len(raw_mask) ** 0.5))
                hw_a = max(1, int(S_attn ** 0.5))
                m_img = (raw_mask.reshape(hw_m, hw_m).astype(np.uint8) * 255)
                m_rs  = np.array(
                    _pi.fromarray(m_img).resize((hw_a, hw_a), _pi.NEAREST)
                ) > 128
                mask_flat = m_rs.ravel()
            else:
                mask_flat = raw_mask

            mask_t = torch.from_numpy(mask_flat).to(weights_tensor.device)  # (S_attn,)
            # attention weights for this entity's tokens → average over tokens → (S_attn,)
            attn_ent = W_mean[fi, :, valid_toks].mean(dim=-1)   # (S_attn,)

            total  = attn_ent.sum() + 1e-8
            inside = (attn_ent * mask_t.float()).sum()
            # maximize overlap fraction → minimize negative
            loss  = loss - inside / total
            count += 1

    return loss / max(count, 1)


# ─── 핵심 수정: Additive VCA Processor ──────────────────────────────────────

class AdditiveVCAProcessor:
    """
    Phase 21: text cross-attention 유지 + VCA를 depth bias로 추가.

    기존 (Phase 1~20):
      return VCA_output   ← text attn 완전 대체, 이미지 품질 파괴

    Phase 21:
      text_out = original_AttnProcessor2_0(attn, hidden_states, encoder_hidden_states)
      vca_delta = VCA(layer_norm(hidden_states)) - layer_norm(hidden_states)
      return text_out + alpha * vca_delta

    gradient는 vca_delta(→VCALayer)에만 흐름. text_out은 detach.
    """
    def __init__(self, vca_layer: VCALayer, entity_ctx: torch.Tensor,
                 orig_processor: AttnProcessor2_0, alpha: float = VCA_ALPHA):
        self.vca   = vca_layer
        self.ctx   = entity_ctx      # (1, N, 768) fp32
        self.orig  = orig_processor  # 원본 AttnProcessor2_0
        self.alpha = alpha           # backward compat (used for AdditiveVCAInferProcessor)
        # Learnable gamma: initialized to alpha (0.3), replaces fixed alpha during training
        # gamma↓: VCA contribution decreases (harmful) → diagnostic signal
        # gamma↑: VCA contribution increases (beneficial) → model embraces VCA
        self.gamma = torch.tensor(float(alpha), dtype=torch.float32, requires_grad=True)
        self.last_delta_ratio: float = float(alpha)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        BF = hidden_states.shape[0]

        # 1. 원본 text cross-attention (frozen, no grad)
        with torch.no_grad():
            text_out = self.orig(attn, hidden_states, encoder_hidden_states,
                                 attention_mask, temb, *args, **kwargs)

        # 2. VCA depth delta (query_dim=INJECT_QUERY_DIM at up_blocks.2)
        ctx = self.ctx.expand(BF, -1, -1).float()
        x   = layer_norm(hidden_states.float(), [INJECT_QUERY_DIM])
        vca_out   = self.vca(x, ctx)          # (BF, S, D), sets last_sigma_raw/acc
        delta_raw = vca_out - x               # LN-space delta, O(1) scale

        # FM-I13 수정: delta를 text_out magnitude에 비례 정규화
        # text_out은 fp16 attn projection (scale ≈ 0.05~0.1),
        # delta_raw는 LN-space (scale ≈ 1) → 초기 ratio ≈ 4~10x → manifold escape
        # 정규화: |vca_delta| = alpha * |text_out|  (alpha = 상대 강도)
        with torch.no_grad():
            text_mag  = text_out.float().abs().mean() + 1e-8
            delta_mag = delta_raw.abs().mean() + 1e-8
        vca_delta = delta_raw * (text_mag / delta_mag) * self.gamma

        # diagnostic: actual |vca_delta| / |text_out| ratio (should stay ≈ gamma by normalization)
        with torch.no_grad():
            self.last_delta_ratio = float(
                (vca_delta.float().abs().mean() / (text_out.float().abs().mean() + 1e-8)).item()
            )

        # 3. text quality 유지 + depth bias 추가 (이제 비율이 정확히 gamma)
        return text_out + vca_delta.to(text_out.dtype)


class AdditiveVCAInferProcessor:
    """추론용: grad 불필요, text_out.detach() 생략."""
    def __init__(self, vca_layer: VCALayer, entity_ctx: torch.Tensor,
                 orig_processor: AttnProcessor2_0, alpha: float = VCA_ALPHA):
        self.vca   = vca_layer
        self.ctx   = entity_ctx
        self.orig  = orig_processor
        self.alpha = alpha
        # 진단용 (debug_vca_internals에서 읽음)
        self.last_text_out  = None   # (BF, S, D) fp32
        self.last_vca_delta = None   # (BF, S, D) fp32

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        BF = hidden_states.shape[0]
        text_out = self.orig(attn, hidden_states, encoder_hidden_states,
                             attention_mask, temb, *args, **kwargs)
        ctx = self.ctx.expand(BF, -1, -1).float()
        x   = layer_norm(hidden_states.float(), [INJECT_QUERY_DIM])
        vca_out   = self.vca(x, ctx)
        delta_raw = vca_out - x
        # FM-I13: text_out magnitude 기준 정규화 (ratio ≈ alpha 로 고정)
        text_mag  = text_out.float().abs().mean() + 1e-8
        delta_mag = delta_raw.abs().mean() + 1e-8
        vca_delta = delta_raw * (text_mag / delta_mag) * self.alpha
        # 마지막 호출 저장 (진단용)
        self.last_text_out  = text_out.detach().float()
        self.last_vca_delta = vca_delta.detach()
        return text_out + vca_delta.to(text_out.dtype)


# ─── VCA 주입 ────────────────────────────────────────────────────────────────

def inject_vca_p21(pipe, entity_ctx: torch.Tensor):
    """학습용 Additive VCA 주입."""
    unet = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))
    orig_proc  = orig_procs[INJECT_KEY]       # AttnProcessor2_0

    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM, context_dim=768,
        n_heads=8, n_entities=2, z_bins=2, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=DEPTH_PE_INIT_SCALE,
    ).to(pipe.device)

    new_proc = AdditiveVCAProcessor(vca_layer, entity_ctx, orig_proc, alpha=VCA_ALPHA)
    # .to() creates non-leaf → optimizer error; detach+requires_grad rebuilds leaf on device
    new_proc.gamma = new_proc.gamma.detach().to(pipe.device).requires_grad_(True)
    new_procs = dict(orig_procs)
    new_procs[INJECT_KEY] = new_proc
    unet.set_attn_processor(new_procs)

    print(f"[inject_p23] additive VCA (alpha={VCA_ALPHA}) → {INJECT_KEY}", flush=True)
    return vca_layer, orig_procs


def inject_vca_p21_infer(pipe, vca_layer: VCALayer, entity_ctx: torch.Tensor):
    """추론용 Additive VCA 주입."""
    unet = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))
    orig_proc  = orig_procs[INJECT_KEY]
    new_proc   = AdditiveVCAInferProcessor(vca_layer, entity_ctx, orig_proc, alpha=VCA_ALPHA)
    new_procs  = dict(orig_procs)
    new_procs[INJECT_KEY] = new_proc
    unet.set_attn_processor(new_procs)
    return orig_procs


def restore_procs(pipe, orig_procs):
    pipe.unet.set_attn_processor(dict(orig_procs))  # copy: set_attn_processor pops from dict


# ─── Fix 2: 고정 probe 측정 ──────────────────────────────────────────────────

def measure_probe_sep(pipe, vca_layer, probe_latents, probe_enc_hs, device):
    """고정 probe × 5 t → 안정적 sigma_separation."""
    vca_layer.eval()
    noise = torch.randn_like(probe_latents)
    seps  = []
    with torch.no_grad():
        for t_val in PROBE_T_VALUES:
            t = torch.tensor([t_val], device=device)
            noisy = pipe.scheduler.add_noise(probe_latents, noise, t)
            vca_layer.reset_sigma_acc()
            pipe.unet(noisy, t, encoder_hidden_states=probe_enc_hs)
            if vca_layer.last_sigma is not None:
                s = compute_sigma_stats_train(vca_layer.last_sigma)
                seps.append(s['sigma_separation'])
    vca_layer.train()
    return float(sum(seps) / max(len(seps), 1))


# ─── 핵심 지표: depth_rank_accuracy ─────────────────────────────────────────

@torch.no_grad()
def measure_depth_rank_accuracy(pipe, vca_layer, dataset, device,
                                 n_samples=20, t_val=100):
    """
    아이디어 유효성 직접 측정.

    학습 데이터 샘플에 대해:
      sigma[front_entity, z=0] > sigma[back_entity, z=0] 이면 정답

    Returns:
      accuracy: float (0~1)
      n_correct, n_total: int
    """
    vca_layer.eval()
    n_correct = 0
    n_total   = 0
    t = torch.tensor([t_val], device=device)

    indices = list(range(min(n_samples, len(dataset))))
    for idx in indices:
        frames_np, _, depth_orders, meta, entity_masks = dataset[idx]
        entity_ctx = get_color_entity_context(pipe, meta, device)

        # 현재 주입된 processor entity_ctx 업데이트
        proc = pipe.unet.attn_processors.get(INJECT_KEY)
        if isinstance(proc, (AdditiveVCAProcessor, AdditiveVCAInferProcessor)):
            proc.ctx = entity_ctx.float()

        latents = encode_frames_to_latents(pipe, frames_np, device)
        noise   = torch.randn_like(latents)
        noisy   = pipe.scheduler.add_noise(latents, noise, t)

        # text encoding (color-qualified prompt)
        _, _, full_prompt, _, _ = make_color_prompts(meta)
        tok = pipe.tokenizer(full_prompt, return_tensors="pt", padding="max_length",
                             max_length=pipe.tokenizer.model_max_length,
                             truncation=True).to(device)
        enc_hs = pipe.text_encoder(**tok).last_hidden_state.half()

        vca_layer.reset_sigma_acc()
        pipe.unet(noisy, t, encoder_hidden_states=enc_hs)

        if vca_layer.last_sigma is None:
            continue

        # (BF, S, N, Z) → per-frame accuracy
        sigma_np = vca_layer.last_sigma.detach().cpu().float().numpy()
        BF = sigma_np.shape[0]
        T  = min(BF, len(depth_orders))

        for fi in range(T):
            order = depth_orders[fi]  # [front_idx, back_idx]
            front, back = int(order[0]), int(order[1])
            # entity 위치 mask 적용 (있으면): entity가 있는 spatial token만 평균
            if fi < entity_masks.shape[0]:
                mf = entity_masks[fi, front].astype(bool)
                mb = entity_masks[fi, back].astype(bool)
                if mf.sum() == 0: mf = np.ones(sigma_np.shape[1], bool)
                if mb.sum() == 0: mb = np.ones(sigma_np.shape[1], bool)
                e_front = float(sigma_np[fi, mf, front, 0].mean())
                e_back  = float(sigma_np[fi, mb, back,  0].mean())
            else:
                e_front = float(sigma_np[fi, :, front, 0].mean())
                e_back  = float(sigma_np[fi, :, back,  0].mean())
            if e_front > e_back:
                n_correct += 1
            n_total += 1

    vca_layer.train()
    accuracy = n_correct / max(n_total, 1)
    return accuracy, n_correct, n_total


# ─── 학습 중간 recon 디버그 ──────────────────────────────────────────────────

@torch.no_grad()
def measure_ldiff_ablation(pipe, vca_layer, dataset, device,
                           n_samples: int = 5, t_max: int = 200):
    """
    VCA on (alpha=0.3) vs off (alpha=0) 때 l_diff 비교.

    Q1 (backbone damage): l_diff_on >> l_diff_off → VCA가 백본 손상
    Q2 (backbone dominance): l_diff_on ≈ l_diff_off → VCA가 생성에 영향 없음
                              → 백본이 VCA delta를 무시하고 혼자 denoise

    Returns: (l_diff_on, l_diff_off, ratio_on_off)
    """
    vca_layer.eval()
    proc = pipe.unet.attn_processors.get(INJECT_KEY)
    if not isinstance(proc, AdditiveVCAProcessor):
        return None, None, None

    orig_alpha = proc.alpha
    # Save original gamma value, temporarily zero it for VCA-off measurement
    orig_gamma = None
    if hasattr(proc, 'gamma'):
        orig_gamma = proc.gamma.data.clone()
    ldiff_on_list, ldiff_off_list = [], []

    indices = list(range(min(n_samples, len(dataset))))
    for idx in indices:
        sample = dataset[idx]
        frames_np, _, depth_orders, meta, *_ = sample
        entity_ctx = get_color_entity_context(pipe, meta, device)
        proc.ctx = entity_ctx.float()

        latents = encode_frames_to_latents(pipe, frames_np, device)
        noise   = torch.randn_like(latents)
        t       = torch.randint(0, t_max, (1,), device=device).long()
        noisy   = pipe.scheduler.add_noise(latents, noise, t)

        _, _, full_prompt, _, _ = make_color_prompts(meta)
        tok = pipe.tokenizer(
            full_prompt, return_tensors="pt", padding="max_length",
            max_length=pipe.tokenizer.model_max_length, truncation=True,
        ).to(device)
        enc_hs = pipe.text_encoder(**tok).last_hidden_state.half()

        # VCA on (restore original gamma)
        proc.alpha = VCA_ALPHA
        if orig_gamma is not None:
            proc.gamma.data.copy_(orig_gamma)
        vca_layer.reset_sigma_acc()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred_on = pipe.unet(noisy, t, encoder_hidden_states=enc_hs).sample
        ldiff_on_list.append(loss_diff(pred_on.float(), noise.float()).item())

        # VCA off (gamma=0 → delta=0, pure backbone)
        proc.alpha = 0.0
        if orig_gamma is not None:
            proc.gamma.data.zero_()  # gamma=0 → VCA off
        vca_layer.reset_sigma_acc()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred_off = pipe.unet(noisy, t, encoder_hidden_states=enc_hs).sample
        ldiff_off_list.append(loss_diff(pred_off.float(), noise.float()).item())

    proc.alpha = orig_alpha
    # Restore gamma
    if orig_gamma is not None:
        proc.gamma.data.copy_(orig_gamma)
    vca_layer.train()

    on  = float(np.mean(ldiff_on_list))
    off = float(np.mean(ldiff_off_list))
    ratio = on / (off + 1e-8)
    return on, off, ratio


def debug_generation(pipe, vca_layer, orig_procs, train_procs,
                     probe_frames_np, probe_meta, probe_entity_ctx,
                     debug_dir: Path, epoch: int, height=256, width=256):
    """
    매 N epoch마다 호출: [GT | Baseline | VCA] 3-panel GIF + sigma overlay 저장.
    학습 중간에 reconstruction 품질과 sigma 분리를 동시에 확인.

    orig_procs:  AttnProcessor2_0 (VCA 없음)
    train_procs: AdditiveVCAProcessor (학습용)
    """
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as iio2

    vca_layer.eval()
    # Save orig_proc reference BEFORE any set_attn_processor call depletes orig_procs.
    # diffusers' set_attn_processor() pops from the dict it receives — we must pass
    # dict(orig_procs) copies everywhere and never pass orig_procs directly.
    orig_proc_ref = orig_procs.get(INJECT_KEY)
    if orig_proc_ref is None:
        orig_proc_ref = AttnProcessor2_0()  # fallback (should not happen)

    prompt = (f"{probe_meta.get('prompt_entity0','entity0')} and "
              f"{probe_meta.get('prompt_entity1','entity1')}")
    kw = dict(num_frames=8, steps=20, height=height, width=width, seed=42)

    def _lbl(arr, text):
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        except Exception:
            font = ImageFont.load_default()
        draw.text((3, 3), text, fill=(255, 255, 255), font=font)
        return np.array(img)

    # 1. baseline: VCA 완전 제거 (copy: set_attn_processor pops from dict)
    pipe.unet.set_attn_processor(dict(orig_procs))
    gen = torch.Generator(device=pipe.device).manual_seed(42)
    out = pipe(prompt=prompt, num_frames=8, num_inference_steps=20,
               guidance_scale=7.5, height=height, width=width,
               generator=gen, output_type='pil')
    baseline_frames = [np.array(f) for f in out.frames[0]]

    # 2. VCA 생성: AdditiveVCAInferProcessor 임시 주입
    infer_proc = AdditiveVCAInferProcessor(
        vca_layer, probe_entity_ctx,
        orig_proc_ref, alpha=VCA_ALPHA,  # use pre-saved reference
    )
    infer_procs = dict(orig_procs)
    infer_procs[INJECT_KEY] = infer_proc
    pipe.unet.set_attn_processor(infer_procs)  # infer_procs is already a fresh dict
    gen2 = torch.Generator(device=pipe.device).manual_seed(42)
    out2 = pipe(prompt=prompt, num_frames=8, num_inference_steps=20,
                guidance_scale=7.5, height=height, width=width,
                generator=gen2, output_type='pil')
    vca_frames = [np.array(f) for f in out2.frames[0]]

    # 3. VCA with SWAPPED entity context (entity0↔1) → depth ordering should flip
    #    If generations look different → VCA IS controlling depth
    #    If identical → backbone ignores VCA signal
    ctx_swapped = torch.flip(probe_entity_ctx.clone(), dims=[1])  # (1,2,768): swap e0↔e1
    infer_proc_sw = AdditiveVCAInferProcessor(
        vca_layer, ctx_swapped,
        orig_proc_ref, alpha=VCA_ALPHA,
    )
    infer_procs_sw = dict(orig_procs)
    infer_procs_sw[INJECT_KEY] = infer_proc_sw
    pipe.unet.set_attn_processor(infer_procs_sw)
    gen3 = torch.Generator(device=pipe.device).manual_seed(42)
    out3 = pipe(prompt=prompt, num_frames=8, num_inference_steps=20,
                guidance_scale=7.5, height=height, width=width,
                generator=gen3, output_type='pil')
    swapped_frames = [np.array(f) for f in out3.frames[0]]

    # 학습용 processor 복원 (copy to avoid depleting train_procs)
    pipe.unet.set_attn_processor(dict(train_procs))
    vca_layer.train()

    P = height

    # ── ① reconstruction.gif: [GT | Baseline | VCA | VCA swapped] ────────
    recon_gif = []
    for fi in range(len(baseline_frames)):
        gt_arr = np.array(Image.fromarray(probe_frames_np[fi]).resize(
            (P, P), Image.BILINEAR)) if fi < len(probe_frames_np) else np.zeros((P,P,3),dtype=np.uint8)
        b_arr  = np.array(Image.fromarray(baseline_frames[fi]).resize((P, P), Image.BILINEAR))
        v_arr  = np.array(Image.fromarray(vca_frames[fi]).resize((P, P), Image.BILINEAR))
        row = np.concatenate([
            _lbl(gt_arr, f"GT  e{epoch:02d}"),
            _lbl(b_arr,  "Baseline (no VCA)"),
            _lbl(v_arr,  f"VCA ctx=normal"),
            _lbl(np.array(Image.fromarray(swapped_frames[fi] if fi < len(swapped_frames) else swapped_frames[-1]).resize((P,P), Image.BILINEAR)),
                 "VCA ctx=swapped"),
        ], axis=1)
        recon_gif.append(row)
    recon_path = debug_dir / f"recon_epoch{epoch:03d}.gif"
    iio2.mimsave(str(recon_path), recon_gif, duration=200)

    # ── ② sigma_overlay.gif: VCA 프레임 위에 E0(파)/E1(빨) 반투명 오버레이 ─
    if vca_layer.last_sigma is not None:
        sig_np = vca_layer.last_sigma.detach().cpu().float().numpy()
        BF, S, N, Z = sig_np.shape
        hw = max(1, int(S ** 0.5))
        overlay_gif = []
        for fi in range(min(len(vca_frames), BF)):
            s = sig_np[fi]                  # (S, N, Z)
            e0 = s[:, 0, 0].reshape(hw, hw)
            e1 = s[:, 1, 0].reshape(hw, hw)

            def _heat(m):
                lo, hi = m.min(), m.max()
                n = (m - lo) / (hi - lo + 1e-6)
                r = np.clip(n*3-2, 0, 1); g = np.clip(n*3-1, 0, 1); b = np.clip(n*3, 0, 1)
                return (np.stack([b, g, r], -1) * 255).astype(np.uint8)

            e0_heat = np.array(Image.fromarray(_heat(e0)).resize((P, P), Image.NEAREST))
            e1_heat = np.array(Image.fromarray(_heat(e1)).resize((P, P), Image.NEAREST))

            rgb = np.array(Image.fromarray(vca_frames[fi]).resize((P, P), Image.BILINEAR)).astype(float)
            e0n = np.array(Image.fromarray(((e0 - e0.min()) / (e0.max()-e0.min()+1e-6)*255).astype(np.uint8)).resize((P,P),Image.BILINEAR)) / 255.
            e1n = np.array(Image.fromarray(((e1 - e1.min()) / (e1.max()-e1.min()+1e-6)*255).astype(np.uint8)).resize((P,P),Image.BILINEAR)) / 255.
            overlay = np.clip(
                rgb/255 + 0.4*np.stack([np.zeros_like(e0n), np.zeros_like(e0n), e0n], -1)
                        + 0.4*np.stack([e1n, np.zeros_like(e1n), np.zeros_like(e1n)], -1), 0, 1)
            overlay_u8 = (overlay * 255).astype(np.uint8)

            row = np.concatenate([
                _lbl(np.array(Image.fromarray(vca_frames[fi]).resize((P,P),Image.BILINEAR)),
                     f"VCA e{epoch:02d}"),
                _lbl(e0_heat, f"E0σ sep={float(np.mean(e0)):.3f}"),
                _lbl(e1_heat, f"E1σ sep={float(np.mean(e1)):.3f}"),
                _lbl(overlay_u8, "Overlay"),
            ], axis=1)
            overlay_gif.append(row)
        sig_path = debug_dir / f"sigma_overlay_epoch{epoch:03d}.gif"
        iio2.mimsave(str(sig_path), overlay_gif, duration=200)

    e0_mean = float(sig_np[:, :, 0, 0].mean()) if vca_layer.last_sigma is not None else 0.
    e1_mean = float(sig_np[:, :, 1, 0].mean()) if vca_layer.last_sigma is not None else 0.
    print(f"  [debug] recon → {recon_path.name}  "
          f"E0σ={e0_mean:.3f} E1σ={e1_mean:.3f} sep={abs(e0_mean-e1_mean):.3f}",
          flush=True)


# ─── 다양한 카메라 뷰 시각화 ─────────────────────────────────────────────────

def debug_multiview(pipe, vca_layer, orig_procs, train_procs,
                    probe_meta, data_root: str, debug_dir: Path, epoch: int,
                    height=256, width=256, max_views=4):
    """
    같은 entity pair를 다양한 카메라 뷰(orbit/rotate × front/top/front_left 등)로 학습하는지 확인.

    probe_meta의 keyword0/keyword1로 같은 entity pair 시퀀스를 찾아서
    각 뷰(mode+camera 조합)의 GT 첫 프레임 + VCA 생성 결과를 나란히 시각화.

    출력: debug_dir/multiview_epoch{N}.gif
      행: 각 카메라 뷰
      열: [GT 첫 프레임 | VCA 생성 1프레임]
    """
    import imageio.v2 as iio2
    from PIL import Image, ImageDraw, ImageFont
    import glob, json as _json

    vca_layer.eval()
    # Save orig_proc reference before any set_attn_processor call depletes orig_procs
    orig_proc_ref_mv = orig_procs.get(INJECT_KEY)
    if orig_proc_ref_mv is None:
        orig_proc_ref_mv = AttnProcessor2_0()

    k0 = probe_meta.get("keyword0", "")
    k1 = probe_meta.get("keyword1", "")

    # 같은 keyword pair를 가진 시퀀스 찾기 (다른 카메라/모션)
    import os
    all_dirs = sorted(os.listdir(data_root))
    same_pair_dirs = []
    for d in all_dirs:
        meta_path = f"{data_root}/{d}/meta.json"
        if not os.path.exists(meta_path):
            continue
        m = _json.load(open(meta_path))
        if m.get("keyword0") == k0 and m.get("keyword1") == k1:
            same_pair_dirs.append((d, m))
        if len(same_pair_dirs) >= max_views:
            break

    if not same_pair_dirs:
        return

    def _lbl(arr, text):
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        except Exception:
            font = ImageFont.load_default()
        draw.text((2, 2), text, fill=(255,255,255), font=font)
        return np.array(img)

    P = height
    rows = []
    prompt = (f"{probe_meta.get('prompt_entity0','entity0')} and "
              f"{probe_meta.get('prompt_entity1','entity1')}")

    # VCA 생성 (probe entity ctx 사용, 여러 뷰 같은 프롬프트)
    infer_proc = AdditiveVCAInferProcessor(
        vca_layer, get_color_entity_context(pipe, probe_meta, str(pipe.device)),
        orig_proc_ref_mv, alpha=VCA_ALPHA,  # use pre-saved reference
    )
    infer_procs = dict(orig_procs)
    infer_procs[INJECT_KEY] = infer_proc
    pipe.unet.set_attn_processor(infer_procs)  # fresh dict, OK

    gen = torch.Generator(device=pipe.device).manual_seed(42)
    out = pipe(prompt=prompt, num_frames=8, num_inference_steps=20,
               guidance_scale=7.5, height=height, width=width,
               generator=gen, output_type='pil')
    vca_frames_shared = [np.array(f) for f in out.frames[0]]

    pipe.unet.set_attn_processor(dict(train_procs))  # copy to avoid depleting train_procs
    vca_layer.train()

    for dir_name, meta in same_pair_dirs:
        # GT 첫 프레임
        frame_paths = sorted(glob.glob(f"{data_root}/{dir_name}/frames/*.png"))
        if not frame_paths:
            continue
        gt_frame = np.array(Image.open(frame_paths[0]).convert("RGB").resize((P, P), Image.BILINEAR))
        vca_frame = np.array(Image.fromarray(vca_frames_shared[0]).resize((P, P), Image.BILINEAR))

        mode   = meta.get("mode", "?")
        camera = meta.get("camera", "?")
        view_label = f"{mode}/{camera}"

        # sigma: E0 vs E1 mean
        if vca_layer.last_sigma is not None:
            sig = vca_layer.last_sigma.detach().cpu().float().numpy()
            e0m = float(sig[:, :, 0, 0].mean())
            e1m = float(sig[:, :, 1, 0].mean())
            sig_str = f"E0={e0m:.2f} E1={e1m:.2f}"
        else:
            sig_str = ""

        row = np.concatenate([
            _lbl(gt_frame,   f"GT {view_label}"),
            _lbl(vca_frame,  f"VCA {sig_str}"),
        ], axis=1)
        rows.append(row)

    if rows:
        # 모든 뷰를 세로로 쌓음
        mosaic = np.concatenate(rows, axis=0)
        out_path = debug_dir / f"multiview_epoch{epoch:03d}.gif"
        iio2.mimsave(str(out_path), [mosaic], duration=2000)
        print(f"  [debug] multiview ({len(rows)} views) → {out_path.name}", flush=True)


# ─── 심층 VCA 진단 시각화 ────────────────────────────────────────────────────

def _heat_map(m, lo=None, hi=None):
    """(H, W) float → (H, W, 3) uint8 컬러 히트맵 (파랑→초록→빨강)."""
    lo = float(m.min()) if lo is None else lo
    hi = float(m.max()) if hi is None else hi
    n = (m - lo) / (hi - lo + 1e-8)
    r = np.clip(n * 3 - 2,       0, 1)
    g = np.clip(1 - np.abs(n * 3 - 1.5), 0, 1)
    b = np.clip(1 - n * 3,       0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def _decode_latents_safe(pipe, latents):
    """
    AnimateDiff latent (1, C, T, H, W) or (T, C, H, W) → (T, H, W, 3) uint8.
    VAE decode는 한 번에 한 frame씩 처리해 OOM 방지.
    """
    try:
        if latents.dim() == 5:
            lat = latents[0].permute(1, 0, 2, 3)  # (T, C, H, W)
        elif latents.dim() == 4:
            lat = latents                           # (T, C, H, W)
        else:
            return None
        lat = lat.float() / pipe.vae.config.scaling_factor
        frames = []
        for i in range(lat.shape[0]):
            dec = pipe.vae.decode(lat[i:i+1]).sample  # (1, 3, H, W)
            frames.append(dec[0].permute(1, 2, 0).cpu().numpy())
        arr = np.stack(frames)                          # (T, H, W, 3)
        return np.clip((arr + 1) / 2 * 255, 0, 255).astype(np.uint8)
    except Exception:
        return None


def debug_vca_internals(pipe, vca_layer, orig_procs, train_procs,
                        probe_frames_np, probe_meta, probe_entity_ctx,
                        debug_dir: Path, epoch: int, height=256, width=256):
    """
    Phase 21 심층 진단. 세 가지 GIF 저장:

    1. denoise_traj_epoch{N}.gif   — Baseline vs VCA 디노이징 경과
       step=0,4,9,14,19 × 8 frames.  어느 step에서 artifact가 발생하는지 포착.

    2. vca_delta_epoch{N}.gif      — VCA delta 강도 + weight collapse 진단
       패널: |Δ|/|text_out| 비율 / ΣWeight map / σ(E0) / σ(E1)
       mean ratio > 0.5 시 경고.  배경 collapse (ΣW→0) 시각화.

    3. ray3d_epoch{N}.gif          — ray marching 3D internals
       σ(E0,E1) × z=0,1 / Transmittance T × z=0,1 / ΣWeight / sep map
       3D module이 실제로 동작하는지 확인.
    """
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as iio2

    vca_layer.eval()
    orig_proc_ref = orig_procs.get(INJECT_KEY)
    if orig_proc_ref is None:
        orig_proc_ref = AttnProcessor2_0()

    prompt = (f"{probe_meta.get('prompt_entity0','entity0')} and "
              f"{probe_meta.get('prompt_entity1','entity1')}")
    P = height

    def _lbl(arr, text, fs=11):
        img = Image.fromarray(arr.astype(np.uint8))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
        except Exception:
            font = ImageFont.load_default()
        draw.text((2, 2), text, fill=(255, 255, 255), font=font)
        return np.array(img)

    def _to_panel(arr_hw, title, colormap=True):
        """1D spatial 배열 (S,) or 2D (H, W) → (P, P, 3) labeled panel."""
        m = arr_hw.reshape(int(len(arr_hw.flat) ** 0.5 + 0.5),
                           -1) if arr_hw.ndim == 1 else arr_hw
        hw_h, hw_w = m.shape
        if colormap:
            img = _heat_map(m)
        else:
            n = (m - m.min()) / (m.max() - m.min() + 1e-8)
            img = (np.stack([n, n, n], -1) * 255).astype(np.uint8)
        img_p = np.array(Image.fromarray(img).resize((P, P), Image.NEAREST))
        return _lbl(img_p, title)

    # ── Instrumented processor ────────────────────────────────────────────────
    instr_proc = AdditiveVCAInferProcessor(
        vca_layer, probe_entity_ctx, orig_proc_ref, alpha=VCA_ALPHA)

    # ── 1. Denoising trajectory ───────────────────────────────────────────────
    traj_baseline: dict = {}
    traj_vca:      dict = {}
    capture_steps  = {0, 4, 9, 14, 19}

    # Denoising trajectory: step count 변화로 근사 (callback 불필요)
    # step_counts = [3, 6, 10, 15, 20] → 점점 더 많이 denoising한 결과
    step_counts = [3, 6, 10, 15, 20]

    def _run_steps(procs, n_steps, gen_seed):
        pipe.unet.set_attn_processor(procs)
        g = torch.Generator(device=pipe.device).manual_seed(gen_seed)
        try:
            out = pipe(prompt=prompt, num_frames=8,
                       num_inference_steps=n_steps,
                       guidance_scale=7.5, height=height, width=width,
                       generator=g, output_type='pil')
            return np.stack([np.array(f) for f in out.frames[0]])
        except Exception:
            return None

    for n_steps in step_counts:
        arr_b = _run_steps(dict(orig_procs), n_steps, 99)
        if arr_b is not None:
            traj_baseline[n_steps] = arr_b

    infer_procs = dict(orig_procs)
    infer_procs[INJECT_KEY] = instr_proc
    for n_steps in step_counts:
        infer_procs_n = dict(orig_procs)
        infer_procs_n[INJECT_KEY] = instr_proc
        arr_v = _run_steps(infer_procs_n, n_steps, 99)
        if arr_v is not None:
            traj_vca[n_steps] = arr_v

    # 학습용 복원
    pipe.unet.set_attn_processor(dict(train_procs))
    vca_layer.train()

    # Build denoise_traj GIF: columns = step counts, rows = [Base | VCA]
    # GIF frame = (frame_index, step_count) — 가로: Base vs VCA, 세로: step 진행
    step_keys = sorted(traj_baseline.keys() | traj_vca.keys())
    if step_keys:
        traj_gif = []
        for fi in range(8):
            panels = []
            for n_steps in step_keys:
                b_arr = traj_baseline.get(n_steps)
                v_arr = traj_vca.get(n_steps)
                def _get(arr, i=fi):
                    if arr is None or i >= len(arr):
                        return np.zeros((P, P, 3), np.uint8)
                    return np.array(Image.fromarray(arr[i]).resize((P, P), Image.BILINEAR))
                col = np.concatenate([
                    _lbl(_get(b_arr), f"Base {n_steps}steps"),
                    _lbl(_get(v_arr), f"VCA  {n_steps}steps"),
                ], axis=0)  # [Base위 VCA아래]
                panels.append(col)
            traj_gif.append(np.concatenate(panels, axis=1))
        if traj_gif:
            iio2.mimsave(str(debug_dir / f"denoise_traj_epoch{epoch:03d}.gif"),
                         traj_gif, duration=250)
            print(f"  [debug] denoise_traj → denoise_traj_epoch{epoch:03d}.gif", flush=True)

    # ── 2. VCA delta magnitude + weight collapse ──────────────────────────────
    delta_gif = []
    mean_ratio = 0.0
    if (instr_proc.last_vca_delta is not None
            and instr_proc.last_text_out is not None):
        delta  = instr_proc.last_vca_delta.float()     # (BF, S, D)
        ttext  = instr_proc.last_text_out.float()
        BF, S, D = delta.shape
        hw = max(1, int(S ** 0.5))

        # |Δ|/|text| per token (manifold escape indicator)
        ratio    = delta.abs().mean(-1) / (ttext.abs().mean(-1) + 1e-8)   # (BF, S)
        mean_ratio = float(ratio.mean())
        max_ratio  = float(ratio.max())

        # weight sum from vca_layer
        weight_sum_np = None
        if (vca_layer.last_sigma is not None
                and vca_layer.last_transmittance is not None):
            sig_t = vca_layer.last_sigma.float()       # (BF, S, N, Z)
            T_t   = vca_layer.last_transmittance.float()  # (BF, S, Z)
            # w = T[z] * sigma[n,z], sum over N,Z
            ws = (T_t.unsqueeze(2) * sig_t).sum(dim=(2, 3))  # (BF, S)
            weight_sum_np = ws.cpu().numpy()

        for fi in range(BF):
            panels = []
            ratio_map = ratio[fi].cpu().numpy().reshape(hw, hw)
            panels.append(_to_panel(ratio_map,
                                    f"|Δ|/|txt| e{epoch:02d}f{fi} m={mean_ratio:.3f}"))

            if weight_sum_np is not None:
                ws_map = weight_sum_np[fi].reshape(hw, hw)
                panels.append(_to_panel(ws_map, f"ΣWeight (0=collapse) f{fi}"))

            if vca_layer.last_sigma is not None:
                sig_np = vca_layer.last_sigma[fi].cpu().numpy()  # (S, N, Z)
                panels.append(_to_panel(sig_np[:, 0, 0].reshape(hw, hw), "σ E0 z=0"))
                panels.append(_to_panel(sig_np[:, 1, 0].reshape(hw, hw), "σ E1 z=0"))

            delta_gif.append(np.concatenate(panels, axis=1))

        if delta_gif:
            iio2.mimsave(str(debug_dir / f"vca_delta_epoch{epoch:03d}.gif"),
                         delta_gif, duration=200)
            warn = "  ⚠ LARGE — manifold escape 위험" if mean_ratio > 0.5 else ""
            print(f"  [debug] vca_delta |Δ|/|txt| mean={mean_ratio:.3f} "
                  f"max={max_ratio:.3f}{warn}", flush=True)

    # ── 3. Ray marching 3D internals ─────────────────────────────────────────
    if vca_layer.last_sigma is not None:
        sig_np = vca_layer.last_sigma.float().cpu().numpy()        # (BF, S, N, Z)
        T_np   = (vca_layer.last_transmittance.float().cpu().numpy()
                  if vca_layer.last_transmittance is not None else None)
        BF, S, N, Z = sig_np.shape
        hw = max(1, int(S ** 0.5))
        n_per_row = 4
        ray_gif = []

        for fi in range(BF):
            panels = []
            # σ per entity × z-bin
            for n_e in range(N):
                for z in range(Z):
                    m = sig_np[fi, :, n_e, z].reshape(hw, hw)
                    panels.append(_to_panel(m, f"σ E{n_e} z={z}"))
            # Transmittance
            if T_np is not None:
                for z in range(Z):
                    m = T_np[fi, :, z].reshape(hw, hw)
                    panels.append(_to_panel(m, f"T z={z} (1=통과)"))
            # ΣWeight per token
            if T_np is not None:
                ws = (T_np[fi, :, :, None] * sig_np[fi]).sum(axis=(1, 2)).reshape(hw, hw)
                panels.append(_to_panel(ws, "ΣWeight (ray hit)"))
            # Separation map: σ(E0,z=0) - σ(E1,z=0) → 양수=E0 앞
            sep = (sig_np[fi, :, 0, 0] - sig_np[fi, :, 1, 0]).reshape(hw, hw)
            sep_col = _heat_map(sep, lo=-1.0, hi=1.0)
            sep_img  = np.array(Image.fromarray(sep_col).resize((P, P), Image.NEAREST))
            panels.append(_lbl(sep_img, "σ(E0)-σ(E1) [R=E0앞/B=E1앞]"))

            # 4-per-row 격자
            rows_out = []
            for i in range(0, len(panels), n_per_row):
                chunk = panels[i:i+n_per_row]
                while len(chunk) < n_per_row:
                    chunk.append(np.zeros((P, P, 3), np.uint8))
                rows_out.append(np.concatenate(chunk, axis=1))
            ray_gif.append(np.concatenate(rows_out, axis=0))

        if ray_gif:
            iio2.mimsave(str(debug_dir / f"ray3d_epoch{epoch:03d}.gif"),
                         ray_gif, duration=200)
            print(f"  [debug] ray3d internals → ray3d_epoch{epoch:03d}.gif", flush=True)

    return mean_ratio   # 호출부에서 경고 조건 판단용


# ─── 학습 진단 GIF (train_denoise 스타일 / row=종류, col=조건, anim=frame) ──

# ── Cross-Attention 캡처 프로세서 ─────────────────────────────────────────────

class CaptureAttnProcessor:
    """
    mid_block attn2를 일시적으로 대체하여 cross-attention weights를 캡처.
    explicit softmax 사용 → attention weights 접근 가능.
    사용 후 반드시 원본 프로세서로 복원해야 함.
    """
    def __init__(self):
        self.last_weights = None  # CPU numpy (BF*heads, S, L_text)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        residual = hidden_states
        ctx = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        q = attn.head_to_batch_dim(attn.to_q(hidden_states))  # (BF*h, S, d)
        k = attn.head_to_batch_dim(attn.to_k(ctx))            # (BF*h, L, d)
        v = attn.head_to_batch_dim(attn.to_v(ctx))            # (BF*h, L, d)

        scores = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1],
                        dtype=q.dtype, device=q.device),
            q, k.transpose(-2, -1), beta=0, alpha=attn.scale,
        )
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = scores.softmax(dim=-1).to(v.dtype)           # (BF*h, S, L)
        self.last_weights = weights.detach().float().cpu().numpy()

        out = torch.bmm(weights, v)
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if attn.residual_connection:
            out = out + residual
        out = out / attn.rescale_output_factor
        return out


# ── 공통 유틸 ──────────────────────────────────────────────────────────────────

def _debug_utils(pipe, height):
    """공통 유틸: _lbl, _decode_frame 반환."""
    from PIL import Image, ImageDraw, ImageFont

    P = height
    vae_dtype = next(pipe.vae.parameters()).dtype

    def _lbl(arr, text, fs=10):
        img = Image.fromarray(arr.astype(np.uint8))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
        except Exception:
            font = ImageFont.load_default()
        draw.text((2, 2), text, fill=(255, 255, 255), font=font)
        return np.array(img)

    def _decode_frame(lat5d, fi):
        """(1,4,T,lH,lW) or (1,4,lH,lW) → (H,W,3) uint8."""
        try:
            if lat5d.dim() == 5:
                lat = lat5d[0, :, fi, :, :]
            else:
                lat = lat5d[0]
            inp = lat.unsqueeze(0).to(vae_dtype) / pipe.vae.config.scaling_factor
            dec = pipe.vae.decode(inp).sample[0]   # (3,H,W)
            arr = dec.float().permute(1, 2, 0).cpu().numpy()
            return np.clip((arr + 1) / 2 * 255, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"  [decode err fi={fi}] {e}", flush=True)
            return np.zeros((P, P, 3), np.uint8)

    def _resize(arr, p=P):
        from PIL import Image
        return np.array(Image.fromarray(arr).resize((p, p), Image.BILINEAR))

    def _pred_x0(pipe, noisy, t, enc_hs, alphas_cumprod):
        """UNet forward → x̂₀. empty_cache로 메모리 단편화 방지."""
        t_val = int(t.item())
        torch.cuda.empty_cache()
        pred_eps = pipe.unet(noisy, t, encoder_hidden_states=enc_hs).sample
        ab = alphas_cumprod[t_val]
        x0 = (noisy - (1 - ab).sqrt() * pred_eps) / ab.sqrt()
        del pred_eps
        torch.cuda.empty_cache()
        return x0

    return _lbl, _decode_frame, _resize, _pred_x0


@torch.no_grad()
def debug_train_denoising(pipe, vca_layer, probe_latents, probe_enc_hs,
                          probe_frames_np, out_dir: Path,
                          height=256, width=256,
                          t_values=(10, 50, 100, 150, 190)):
    """
    GIF: train_denoise.gif
    Row 1 — GT: 원본 학습 영상 (첫 컬럼만, 나머지 blank)
    Row 2 — Noised x_t: t별 노이즈 추가 후 VAE decode
    Row 3 — Model pred x̂₀: UNet으로 복원한 x̂₀ VAE decode
    Columns: t values (10→190), Animation: video frames
    """
    import imageio.v2 as iio2
    from PIL import Image

    vca_layer.eval()
    P = height
    _lbl, _decode_frame, _resize, _pred_x0 = _debug_utils(pipe, P)
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(probe_latents.device)
    noise = torch.randn_like(probe_latents)
    T_frames = probe_latents.shape[2]
    N_cols   = len(t_values)

    gif_frames = []
    for fi in range(min(T_frames, len(probe_frames_np))):
        gt_raw  = _resize(probe_frames_np[fi])
        gt_row  = [_lbl(gt_raw, f"GT f={fi}")] + \
                  [np.zeros((P, P, 3), np.uint8)] * (N_cols - 1)
        noised_row   = []
        denoised_row = []
        for t_val in t_values:
            t     = torch.tensor([t_val], device=probe_latents.device)
            noisy = pipe.scheduler.add_noise(probe_latents, noise, t)
            noised_row.append(_lbl(_resize(_decode_frame(noisy, fi)),
                                   f"t={t_val}"))
            x0 = _pred_x0(pipe, noisy, t, probe_enc_hs, alphas_cumprod).clamp(-4, 4)
            denoised_row.append(_lbl(_resize(_decode_frame(x0, fi)),
                                     f"Denoised t={t_val}"))
        gif_frames.append(np.concatenate([
            np.concatenate(gt_row,       axis=1),
            np.concatenate(noised_row,   axis=1),
            np.concatenate(denoised_row, axis=1),
        ], axis=0))

    if gif_frames:
        p = out_dir / "train_denoise.gif"
        iio2.mimsave(str(p), gif_frames, duration=300)
        print(f"  [debug] {p.parent.name}/train_denoise.gif", flush=True)
    vca_layer.train()


@torch.no_grad()
def debug_text_cond(pipe, vca_layer, probe_latents, probe_meta,
                    probe_frames_np, out_dir: Path,
                    height=256, width=256,
                    t_values=(10, 100, 190)):
    """
    GIF: text_cond.gif — 텍스트 컨디셔닝이 제대로 들어가는지 검증.

    Columns: t values (10, 100, 190)
    Rows (per column):
      Row 0 — GT (첫 column만, 나머지 blank)
      Row 1 — Full prompt  "entity0 and entity1"
      Row 2 — Null prompt  (빈 문자열 → uncond)
      Row 3 — Entity0 only "entity0"
      Row 4 — Entity1 only "entity1"
    Animation: video frames

    Row 1~4가 서로 다르게 보이면 text conditioning 정상.
    Row 1≈Row 2 → text guidance 죽어있음.
    """
    import imageio.v2 as iio2
    from PIL import Image

    vca_layer.eval()
    P = height
    _lbl, _decode_frame, _resize, _pred_x0 = _debug_utils(pipe, P)
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(probe_latents.device)
    device = probe_latents.device

    e0 = probe_meta.get('prompt_entity0', 'entity0')
    e1 = probe_meta.get('prompt_entity1', 'entity1')

    def _encode_prompt(text):
        tok = pipe.tokenizer(
            text, return_tensors="pt", padding="max_length",
            max_length=pipe.tokenizer.model_max_length, truncation=True,
        ).to(device)
        return pipe.text_encoder(**tok).last_hidden_state.half()

    prompts = [
        (f"{e0} and {e1}", "Full"),
        ("",               "Null"),
        (e0,               e0[:12]),
        (e1,               e1[:12]),
    ]
    enc_list = [(label, _encode_prompt(txt)) for txt, label in prompts]

    noise = torch.randn_like(probe_latents)
    T_frames = probe_latents.shape[2]
    N_cols   = len(t_values)

    gif_frames = []
    for fi in range(min(T_frames, len(probe_frames_np))):
        gt_raw = _resize(probe_frames_np[fi])
        rows = []
        for col_i, t_val in enumerate(t_values):
            t     = torch.tensor([t_val], device=device)
            noisy = pipe.scheduler.add_noise(probe_latents, noise, t)
            col_panels = []
            # GT (첫 column만)
            gt_panel = _lbl(gt_raw, f"GT f={fi}") if col_i == 0 \
                       else np.zeros((P, P, 3), np.uint8)
            col_panels.append(gt_panel)
            # 각 text condition
            for label, enc_hs in enc_list:
                x0 = _pred_x0(pipe, noisy, t, enc_hs,
                              alphas_cumprod).clamp(-4, 4)
                col_panels.append(_lbl(_resize(_decode_frame(x0, fi)),
                                       f"{label} t={t_val}"))
            rows.append(np.concatenate(col_panels, axis=0))  # 세로 쌓기
        gif_frames.append(np.concatenate(rows, axis=1))      # 가로 쌓기

    if gif_frames:
        p = out_dir / "text_cond.gif"
        iio2.mimsave(str(p), gif_frames, duration=300)
        print(f"  [debug] {p.parent.name}/text_cond.gif", flush=True)
    vca_layer.train()


@torch.no_grad()
def debug_depth_effect(pipe, vca_layer, orig_procs, train_procs,
                       probe_latents, probe_enc_hs,
                       probe_frames_np, probe_entity_ctx,
                       out_dir: Path, height=256, width=256,
                       t_values=(10, 100, 190)):
    """
    GIF: depth_effect.gif — VCA depth conditioning 효과 + sigma map.

    Columns: t values
    Rows:
      Row 0 — GT (첫 column만)
      Row 1 — Noised x_t
      Row 2 — Baseline pred x̂₀  (orig_procs, VCA 없음)
      Row 3 — VCA pred x̂₀       (train_procs, depth bias 있음)
      Row 4 — σ(E0) z=0 heatmap  (depth 예측: 앞에 있는 entity)
      Row 5 — σ(E1) z=0 heatmap
      Row 6 — sep map: σ(E0)-σ(E1) (빨강=E0앞, 파랑=E1앞)
    Animation: video frames

    Row 2≈Row 3 → VCA 효과 없음 (alpha 너무 작거나 학습 안됨).
    Row 4/5 분리 명확 → depth 학습 성공.
    """
    import imageio.v2 as iio2
    from PIL import Image

    vca_layer.eval()
    P = height
    _lbl, _decode_frame, _resize, _pred_x0 = _debug_utils(pipe, P)
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(probe_latents.device)

    def _heat(m, lo=None, hi=None):
        lo = float(m.min()) if lo is None else lo
        hi = float(m.max()) if hi is None else hi
        n  = (m - lo) / (hi - lo + 1e-8)
        r  = np.clip(n * 3 - 2,       0, 1)
        g  = np.clip(1 - np.abs(n*3 - 1.5), 0, 1)
        b  = np.clip(1 - n * 3,       0, 1)
        return (np.stack([r, g, b], -1) * 255).astype(np.uint8)

    def _sigma_panel(sig_np_fi, entity, title):
        # sig_np_fi: (S, N, Z)
        hw = max(1, int(sig_np_fi.shape[0] ** 0.5))
        m  = sig_np_fi[:, entity, 0].reshape(hw, hw)
        h  = np.array(Image.fromarray(_heat(m)).resize((P, P), Image.NEAREST))
        return _lbl(h, title)

    def _sep_panel(sig_np_fi, title):
        hw = max(1, int(sig_np_fi.shape[0] ** 0.5))
        m  = (sig_np_fi[:, 0, 0] - sig_np_fi[:, 1, 0]).reshape(hw, hw)
        h  = np.array(Image.fromarray(_heat(m, lo=-1., hi=1.)).resize((P, P), Image.NEAREST))
        return _lbl(h, title)

    noise    = torch.randn_like(probe_latents)
    T_frames = probe_latents.shape[2]

    orig_proc_ref = orig_procs.get(INJECT_KEY)
    if orig_proc_ref is None:
        orig_proc_ref = AttnProcessor2_0()

    gif_frames = []
    for fi in range(min(T_frames, len(probe_frames_np))):
        gt_raw = _resize(probe_frames_np[fi])
        rows = []
        for col_i, t_val in enumerate(t_values):
            t     = torch.tensor([t_val], device=probe_latents.device)
            noisy = pipe.scheduler.add_noise(probe_latents, noise, t)

            # Baseline (no VCA)
            pipe.unet.set_attn_processor(dict(orig_procs))
            x0_base = _pred_x0(pipe, noisy, t, probe_enc_hs,
                               alphas_cumprod).clamp(-4, 4)

            # VCA (train_procs)
            pipe.unet.set_attn_processor(dict(train_procs))
            vca_layer.reset_sigma_acc()
            x0_vca = _pred_x0(pipe, noisy, t, probe_enc_hs,
                              alphas_cumprod).clamp(-4, 4)
            sigma_now = vca_layer.last_sigma   # (BF, S, N, Z)

            col_panels = [
                _lbl(gt_raw, f"GT f={fi}") if col_i == 0
                    else np.zeros((P, P, 3), np.uint8),
                _lbl(_resize(_decode_frame(noisy, fi)),   f"Noised  t={t_val}"),
                _lbl(_resize(_decode_frame(x0_base, fi)), f"Base    t={t_val}"),
                _lbl(_resize(_decode_frame(x0_vca, fi)),  f"VCA     t={t_val}"),
            ]
            if sigma_now is not None:
                sig_np = sigma_now.float().cpu().numpy()
                fi_s   = min(fi, sig_np.shape[0] - 1)
                col_panels += [
                    _sigma_panel(sig_np[fi_s], 0, f"σ(E0) t={t_val}"),
                    _sigma_panel(sig_np[fi_s], 1, f"σ(E1) t={t_val}"),
                    _sep_panel(sig_np[fi_s],      "σ(E0)-σ(E1) R=E0앞"),
                ]
            else:
                col_panels += [np.zeros((P, P, 3), np.uint8)] * 3

            rows.append(np.concatenate(col_panels, axis=0))

        gif_frames.append(np.concatenate(rows, axis=1))

    # 학습 프로세서 복원
    pipe.unet.set_attn_processor(dict(train_procs))
    vca_layer.train()

    if gif_frames:
        p = out_dir / "depth_effect.gif"
        iio2.mimsave(str(p), gif_frames, duration=300)
        print(f"  [debug] {p.parent.name}/depth_effect.gif", flush=True)


@torch.no_grad()
def debug_text_attn(pipe, probe_latents, probe_enc_hs, probe_meta,
                    probe_frames_np, out_dir: Path,
                    height=256, width=256, t_values=(50, 150),
                    entity_masks: np.ndarray | None = None):
    """
    GIF: text_attn.gif  +  PNG: text_attn_chart.png

    GIF columns: t values
    GIF rows:
      Row 0 — GT frame
      Row 1 — color0 token attention overlay  (e.g. "red")
      Row 2 — entity0 token attention overlay (e.g. "cat")
      Row 3 — color1 token attention overlay  (e.g. "blue")
      Row 4 — entity1 token attention overlay (e.g. "dog")

    Chart: entity_masks (GT spatial location) 와 text attention 의 overlap 비교.
      X=frame, Y=attention_mass_in_gt_mask / total_attention (0→1, 1=perfect)
      Subplots: one per t_value
      Lines: [c0_tok→E0mask, e0_tok→E0mask, c1_tok→E1mask, e1_tok→E1mask]
      GT = 1.0 dashed (perfect alignment baseline)
      → 학습이 진행될수록 선들이 1.0에 가까워지는지 확인

    entity_masks: (T, 2, S) bool numpy (S=VCA_HW²=256)  from ObjaverseDatasetWithMasks
      None이면 chart 생략
    """
    import imageio.v2 as iio2
    from PIL import Image as PILImg

    device = probe_latents.device
    P = min(height, width)
    _lbl, _decode_frame, _resize, _ = _debug_utils(pipe, P)

    # ── 토큰 인덱스 찾기 (Phase 23: color0, entity0, color1, entity1 4개) ──────
    tokenizer = pipe.tokenizer
    e0_kw = probe_meta.get('keyword0', probe_meta.get('prompt_entity0', 'entity0'))
    e1_kw = probe_meta.get('keyword1', probe_meta.get('prompt_entity1', 'entity1'))

    # color-qualified 프롬프트 생성: "a red cat and a blue dog"
    e0_text, e1_text, full_prompt, c0_name, c1_name = make_color_prompts(probe_meta)

    def find_tok_idx(keyword, full_ids):
        """full_prompt 내에서 keyword 토큰 위치를 찾는다."""
        kw_ids = tokenizer(keyword, add_special_tokens=False)['input_ids']
        for i in range(len(full_ids) - len(kw_ids) + 1):
            if full_ids[i:i + len(kw_ids)] == kw_ids:
                return list(range(i, i + len(kw_ids)))
        return [1]  # fallback: 첫 번째 non-BOS 토큰

    full_ids = tokenizer(full_prompt, add_special_tokens=True)['input_ids']
    # 4 token groups: color0, entity0, color1, entity1
    c0_tok_idx  = find_tok_idx(c0_name, full_ids)   # e.g. "red"
    e0_tok_idx  = find_tok_idx(e0_kw,   full_ids)   # e.g. "cat"
    c1_tok_idx  = find_tok_idx(c1_name, full_ids)   # e.g. "blue"
    e1_tok_idx  = find_tok_idx(e1_kw,   full_ids)   # e.g. "dog"

    # re-encode with color-qualified full prompt
    probe_enc_hs_color = pipe.tokenizer(
        full_prompt, return_tensors="pt", padding="max_length",
        max_length=pipe.tokenizer.model_max_length, truncation=True,
    ).to(device)
    with torch.no_grad():
        probe_enc_hs_color = pipe.text_encoder(**probe_enc_hs_color).last_hidden_state.half()

    # ── Use existing processor (LoRAAttnProcessor) — don't replace with CaptureAttnProcessor ─
    # Fix 2: read last_weights from the EXISTING processor after forward pass
    # This shows LoRA-modified attention (not base model attention)
    attn_proc_for_debug = dict(pipe.unet.attn_processors).get(ATTN_CAPTURE_KEY)
    pipe.unet.eval()

    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
    n_frames = probe_latents.shape[2]
    # up_blocks.3 spatial: 32×32 = 1024 (라텐트와 동일 해상도)
    n_heads = CAPTURE_N_HEADS

    def _attn_to_overlay(attn_map_hw, gt_frame, color_rgb, alpha=0.65):
        """
        Normalized attention map → colored overlay on GT frame.
        color_rgb: (R, G, B) 0~1 각 채널 가중치 (entity 색)
        """
        mn, mx = attn_map_hw.min(), attn_map_hw.max()
        norm = (attn_map_hw - mn) / (mx - mn + 1e-8)
        norm = np.nan_to_num(norm, nan=0.0)
        # Hs×Ws → P×P 업스케일
        norm_img = np.array(
            PILImg.fromarray((norm * 255).clip(0, 255).astype(np.uint8))
            .resize((P, P), PILImg.BILINEAR)
        ).astype(np.float32) / 255.0

        colored = np.stack([norm_img * color_rgb[0],
                            norm_img * color_rgb[1],
                            norm_img * color_rgb[2]], axis=-1)
        gt_f = gt_frame.astype(np.float32) / 255.0
        overlay = np.clip(gt_f * (1 - alpha) + colored * alpha, 0, 1)
        return (overlay * 255).astype(np.uint8)

    # chart 누적용: { t_val: { 'c0_e0': [...], 'e0_e0': [...], 'c1_e1': [...], 'e1_e1': [...] } }
    # 각 리스트에 per-frame overlap score 저장
    attn_records = {tv: {'c0_e0': [], 'e0_e0': [], 'c1_e1': [], 'e1_e1': []} for tv in t_values}

    def _attn_overlap(attn_hw, mask_flat, attn_S):
        """
        attention map (Hs, Ws) vs GT entity mask (S,) bool
        → fraction of attention mass inside the GT mask region.
        attn_hw reshape to S → weight by mask → sum / total
        """
        Hs_a, Ws_a = attn_hw.shape
        S_a = Hs_a * Ws_a
        # resize mask to match attention spatial resolution
        if S_a != len(mask_flat):
            from PIL import Image as _pi
            hw_m = max(1, int(len(mask_flat) ** 0.5))
            m_img = (mask_flat.reshape(hw_m, hw_m).astype(np.uint8) * 255)
            m_resized = np.array(
                _pi.fromarray(m_img).resize((Ws_a, Hs_a), _pi.NEAREST)
            ) > 128
        else:
            hw_m = Hs_a
            m_resized = mask_flat.reshape(Hs_a, Ws_a)
        attn_flat = attn_hw.ravel()
        mask_flat_r = m_resized.ravel()
        total = float(attn_flat.sum()) + 1e-8
        inside = float(attn_flat[mask_flat_r].sum())
        return inside / total

    gif_frames = []
    for fi in range(n_frames):
        col_panels = []
        for t_val in t_values:
            t     = torch.tensor([t_val], device=device)
            noise = torch.randn_like(probe_latents)
            ab    = alphas_cumprod[t_val]
            noisy = ab.sqrt() * probe_latents + (1 - ab).sqrt() * noise

            torch.cuda.empty_cache()
            pipe.unet(noisy, t, encoder_hidden_states=probe_enc_hs_color)
            torch.cuda.empty_cache()

            # Fix 2: read last_weights from EXISTING processor (LoRAAttnProcessor)
            # after the forward pass — shows LoRA-modified attention
            attn_weights_np = None
            if hasattr(attn_proc_for_debug, 'last_weights') and attn_proc_for_debug.last_weights is not None:
                attn_weights_np = attn_proc_for_debug.last_weights  # (BF*heads, S_attn, L)
            W = attn_weights_np  # (BF*h, S, L) numpy float32
            if W is None:
                col_panels.append(np.zeros((5 * P, P, 3), np.uint8))
                continue

            BFh, S, L = W.shape
            BF = BFh // n_heads
            # (BF*h, S, L) → (BF, h, S, L) → mean heads → (BF, S, L)
            W_4d   = W.reshape(BF, n_heads, S, L)
            W_mean = W_4d.mean(axis=1)  # (BF, S, L)

            fi_clamped = min(fi, BF - 1)
            w_fi = W_mean[fi_clamped]  # (S, L)

            # spatial dim: S = Hs × Ws
            Hs = Ws = max(1, int(S ** 0.5))
            if Hs * Ws != S:
                Hs, Ws = S, 1

            # Phase 23: 4 token groups — color0, entity0, color1, entity1
            def _safe_tok(idx_list):
                idx_list = [i for i in idx_list if i < L]
                return idx_list if idx_list else [min(1, L - 1)]

            attn_c0 = w_fi[:, _safe_tok(c0_tok_idx)].mean(axis=-1).reshape(Hs, Ws)
            attn_e0 = w_fi[:, _safe_tok(e0_tok_idx)].mean(axis=-1).reshape(Hs, Ws)
            attn_c1 = w_fi[:, _safe_tok(c1_tok_idx)].mean(axis=-1).reshape(Hs, Ws)
            attn_e1 = w_fi[:, _safe_tok(e1_tok_idx)].mean(axis=-1).reshape(Hs, Ws)

            # chart overlap 누적
            if entity_masks is not None and fi < entity_masks.shape[0]:
                mask_e0 = entity_masks[fi, 0]  # (S,) bool, S=256
                mask_e1 = entity_masks[fi, 1]
                attn_records[t_val]['c0_e0'].append(_attn_overlap(attn_c0, mask_e0, S))
                attn_records[t_val]['e0_e0'].append(_attn_overlap(attn_e0, mask_e0, S))
                attn_records[t_val]['c1_e1'].append(_attn_overlap(attn_c1, mask_e1, S))
                attn_records[t_val]['e1_e1'].append(_attn_overlap(attn_e1, mask_e1, S))

            gt_frame = _resize(probe_frames_np[fi])

            # Row overlays: color = warm/cool tone pair per entity
            ov_c0 = _attn_to_overlay(attn_c0, gt_frame, (1.0, 0.50, 0.05))  # orange-red
            ov_e0 = _attn_to_overlay(attn_e0, gt_frame, (1.0, 0.10, 0.05))  # deep red
            ov_c1 = _attn_to_overlay(attn_c1, gt_frame, (0.05, 0.60, 1.0))  # light blue
            ov_e1 = _attn_to_overlay(attn_e1, gt_frame, (0.05, 0.15, 1.0))  # deep blue

            # Dual overlay: R channel = E0 combined, B channel = E1 combined
            def _norm_attn(m):
                mn, mx = m.min(), m.max()
                n = np.nan_to_num((m - mn) / (mx - mn + 1e-8))
                return np.array(PILImg.fromarray(
                    (n * 255).clip(0, 255).astype(np.uint8)
                ).resize((P, P), PILImg.BILINEAR)).astype(np.float32) / 255.0

            n_e0 = np.clip(_norm_attn(attn_c0) * 0.4 + _norm_attn(attn_e0) * 0.6, 0, 1)
            n_e1 = np.clip(_norm_attn(attn_c1) * 0.4 + _norm_attn(attn_e1) * 0.6, 0, 1)
            dual_col = np.stack([n_e0, np.zeros_like(n_e0), n_e1], axis=-1)
            gt_f     = gt_frame.astype(np.float32) / 255.0
            dual_ov  = (np.clip(gt_f * 0.25 + dual_col * 0.75, 0, 1) * 255).astype(np.uint8)

            # 5 rows: GT, color0, entity0, color1, entity1
            col = np.vstack([
                _lbl(gt_frame, f"GT  t={t_val}  '{full_prompt[:30]}'"),
                _lbl(ov_c0,    f"[RED-ORA] '{c0_name}'  tok={','.join(str(i) for i in c0_tok_idx)}"),
                _lbl(ov_e0,    f"[RED    ] '{e0_kw}'   tok={','.join(str(i) for i in e0_tok_idx)}"),
                _lbl(ov_c1,    f"[BLU-LIT] '{c1_name}'  tok={','.join(str(i) for i in c1_tok_idx)}"),
                _lbl(ov_e1,    f"[BLU    ] '{e1_kw}'   tok={','.join(str(i) for i in e1_tok_idx)}"),
            ])
            col_panels.append(col)

        if col_panels:
            gif_frames.append(np.hstack(col_panels))

    # Fix 2: No restore needed — we never replaced the processor with CaptureAttnProcessor.
    # The existing LoRAAttnProcessor remains in place.

    if gif_frames:
        p = out_dir / "text_attn.gif"
        iio2.mimsave(str(p), gif_frames, duration=300)
        print(f"  [debug] {p.parent.name}/text_attn.gif", flush=True)

    # ── text_attn_chart.png: GT mask vs text attention overlap per frame ────────
    if entity_masks is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 엔티티 mask coverage = 얼마나 많은 spatial token이 entity를 포함하는지
        # random baseline = mask_coverage (attention이 uniform하면 이 값)
        mask_coverage_e0 = float(entity_masks[:n_frames, 0].mean()) if n_frames > 0 else 0.5
        mask_coverage_e1 = float(entity_masks[:n_frames, 1].mean()) if n_frames > 0 else 0.5

        n_t = len(t_values)
        fig, axes = plt.subplots(1, n_t, figsize=(5 * n_t, 4),
                                 facecolor='#0d0d1a', sharey=True)
        if n_t == 1:
            axes = [axes]

        E0_COLOR_DARK  = '#e84545'   # entity0 entity token (deep red)
        E0_COLOR_LIGHT = '#ff9966'   # entity0 color token (orange-red)
        E1_COLOR_DARK  = '#3a7fff'   # entity1 entity token (deep blue)
        E1_COLOR_LIGHT = '#66ccff'   # entity1 color token (light blue)

        for ax, t_val in zip(axes, t_values):
            rec = attn_records[t_val]
            frames_idx = list(range(len(rec['c0_e0'])))

            ax.set_facecolor('#0d0d1a')

            # 실제 attention overlap 선
            if frames_idx:
                ax.plot(frames_idx, rec['c0_e0'], color=E0_COLOR_LIGHT, linewidth=2.0,
                        marker='o', markersize=3,
                        label=f"'{c0_name}' tok → E0 mask")
                ax.plot(frames_idx, rec['e0_e0'], color=E0_COLOR_DARK,  linewidth=2.0,
                        marker='s', markersize=3,
                        label=f"'{e0_kw}' tok → E0 mask")
                ax.plot(frames_idx, rec['c1_e1'], color=E1_COLOR_LIGHT, linewidth=2.0,
                        marker='o', markersize=3,
                        label=f"'{c1_name}' tok → E1 mask")
                ax.plot(frames_idx, rec['e1_e1'], color=E1_COLOR_DARK,  linewidth=2.0,
                        marker='s', markersize=3,
                        label=f"'{e1_kw}' tok → E1 mask")

            # GT 1.0 (완벽한 alignment)
            ax.axhline(1.0, color='white', linewidth=1.0, linestyle='--',
                       alpha=0.5, label='GT (perfect=1.0)')
            # Random baseline = mask coverage (attention이 uniform이면 이 값)
            ax.axhline(mask_coverage_e0, color=E0_COLOR_DARK, linewidth=0.8,
                       linestyle=':', alpha=0.4, label=f'random E0={mask_coverage_e0:.2f}')
            ax.axhline(mask_coverage_e1, color=E1_COLOR_DARK, linewidth=0.8,
                       linestyle=':', alpha=0.4, label=f'random E1={mask_coverage_e1:.2f}')

            ax.set_title(f"t={t_val}", color='white', fontsize=9, pad=4)
            ax.set_xlabel('frame', color='#888899', fontsize=7)
            if ax == axes[0]:
                ax.set_ylabel('attention mass in GT mask\n(higher = better alignment)',
                              color='#888899', fontsize=7)
            ax.set_ylim(0.0, 1.05)
            ax.tick_params(colors='#888899', labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333355')
            ax.legend(fontsize=5.5, facecolor='#1a1a2e', edgecolor='none',
                      labelcolor='white', framealpha=0.8, loc='upper right')
            ax.grid(True, color='#222244', linewidth=0.5, alpha=0.6)

        fig.suptitle(
            f"Text Attention Alignment:  RED='{e0_kw}'({c0_name})   BLUE='{e1_kw}'({c1_name})\n"
            f"solid=entity_tok  light=color_tok  dashed=GT(1.0)  dotted=random_baseline",
            color='white', fontsize=8, y=1.02,
        )
        fig.tight_layout()
        chart_path = out_dir / 'text_attn_chart.png'
        fig.savefig(str(chart_path), dpi=120, bbox_inches='tight',
                    facecolor='#0d0d1a')
        plt.close(fig)
        print(f"  [debug] {out_dir.name}/text_attn_chart.png", flush=True)


@torch.no_grad()
def debug_multiangle_depth(pipe, vca_layer, dataset, train_procs,
                           probe_entity_ctx,
                           out_dir: Path, height=256, width=256,
                           t_val=100):
    """
    GIF: multiangle_depth.gif — 같은 motion, 다른 카메라 angle에서
    VCA depth ordering이 올바르게 달라지는지 검증.

    Columns: camera angles (front / front_left / front_right / top)
    Rows (per column):
      Row 0 — 실제 프레임 (GT)
      Row 1 — Depth diagram: 카메라 → 깊이 축에 entity 위치 표시
               (E0=빨강, E1=파랑, filled=VCA 예측, outline=GT)
    Animation: video frames

    depth diagram 읽는 법:
      세로축 = 깊이 (위=camera쪽, 아래=멀어짐)
      빨강/파랑 원 = VCA가 예측한 E0/E1 깊이
      빨강/파랑 X  = GT 깊이
      위에 있는 entity가 카메라에 더 가깝다 (앞쪽)
    """
    import imageio.v2 as iio2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import io as _io
    from PIL import Image as PILImg

    device = next(pipe.unet.parameters()).device
    P = min(height, width)
    _lbl, _decode_frame, _resize, _ = _debug_utils(pipe, P)

    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)

    # ── 같은 (entities, motion) 다른 angle 그룹 찾기 ────────────────────────
    from collections import defaultdict
    groups = defaultdict(list)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        frames_np, latents_or_none, depth_orders, meta, *extra = sample
        entity_masks = extra[0] if extra else None  # (T, 2, S) bool numpy
        key = f"{meta.get('keyword0','')}_{meta.get('keyword1','')}_{meta.get('mode','')}"
        groups[key].append((frames_np, depth_orders, meta, entity_masks))

    # angle이 여럿인 그룹 선택 (4각도 우선)
    best_group = None
    best_n = 0
    for key, samples in groups.items():
        cameras = [s[2].get('camera', '') for s in samples]
        if len(cameras) >= 3 and len(best_group or []) < len(cameras):
            best_group = samples
            best_n     = len(cameras)
        if best_n >= 4:
            break

    if not best_group:
        print("  [debug] multiangle_depth: no multi-angle group found", flush=True)
        return

    # 최대 4각도 사용
    angle_samples = best_group[:4]

    # ── 각 angle별 VCA sigma → depth score 계산 ─────────────────────────────
    vca_layer.eval()
    pipe.unet.set_attn_processor(dict(train_procs))

    def _sigma_depth_scores(frames_np, entity_masks=None, meta=None):
        """frames_np → (n_frames, 2) VCA depth score (E0, E1) per frame."""
        from scripts.train_animatediff_vca import encode_frames_to_latents
        lat = encode_frames_to_latents(pipe, frames_np, device)
        proc = dict(train_procs).get(INJECT_KEY)
        if isinstance(proc, AdditiveVCAProcessor):
            proc.ctx = probe_entity_ctx.float()

        t = torch.tensor([t_val], device=device)
        noise = torch.randn_like(lat)
        ab    = alphas_cumprod[t_val]
        noisy = ab.sqrt() * lat + (1 - ab).sqrt() * noise

        # Fix 3d: use current sample's own meta for encoder_hidden_states
        enc_hs_meta = meta if meta is not None else angle_samples[0][2]

        vca_layer.reset_sigma_acc()
        torch.cuda.empty_cache()
        pipe.unet(noisy, t,
                  encoder_hidden_states=_enc_hs_for_meta(pipe, enc_hs_meta, device))
        torch.cuda.empty_cache()

        if vca_layer.last_sigma is None:
            return None
        # sigma: (BF, S, N, Z)
        sig = vca_layer.last_sigma.float().cpu().numpy()
        # depth score per entity: weighted sum over z-bins
        Z = sig.shape[3]
        z_weights = np.arange(Z, dtype=np.float32)  # z=0 가까움, z=Z-1 멂
        scores = []
        for fi in range(sig.shape[0]):
            # Fix 3c: use entity-specific spatial tokens if masks available (matches DRA metric)
            if entity_masks is not None and fi < entity_masks.shape[0]:
                m0 = entity_masks[fi, 0].astype(bool)  # entity0 mask (S,)
                m1 = entity_masks[fi, 1].astype(bool)  # entity1 mask (S,)
                if m0.sum() == 0: m0 = np.ones(sig.shape[1], dtype=bool)
                if m1.sum() == 0: m1 = np.ones(sig.shape[1], dtype=bool)
                e0_sig = sig[fi, m0, 0, :]  # entity0 spatial tokens, entity0 channel
                e1_sig = sig[fi, m1, 1, :]  # entity1 spatial tokens, entity1 channel
            else:
                e0_sig = sig[fi, :, 0, :]
                e1_sig = sig[fi, :, 1, :]
            # expected z-index (높을수록 멀다)
            d0 = float((e0_sig.mean(0) * z_weights).sum() / (e0_sig.mean(0).sum() + 1e-8))
            d1 = float((e1_sig.mean(0) * z_weights).sum() / (e1_sig.mean(0).sum() + 1e-8))
            scores.append((d0, d1))
        return scores  # [(d0, d1), ...]

    def _gt_depth_scores(frames_np, depth_orders, meta):
        """GT depth 파일에서 mask별 평균 depth를 읽는다."""
        import json, os
        # dataset root 추론 (meta에서)
        # depth npy와 mask는 이미 dataset에 포함되어 있음
        # depth_orders: [(front_idx, back_idx), ...] — 이미 순서 정보 있음
        # GT depth score는 depth_orders에서 직접 읽는다:
        # front_idx entity가 앞 → GT d_front < d_back
        scores = []
        for fi, order in enumerate(depth_orders[:len(frames_np)]):
            front, back = int(order[0]), int(order[1])
            # front entity = 가까움 (depth_score 낮음) → 0.3, back = 0.7 (정규화 상대값)
            if front == 0:
                scores.append((0.3, 0.7))
            else:
                scores.append((0.7, 0.3))
        return scores

    def _enc_hs_for_meta(pipe, meta, device):
        e0 = meta.get('prompt_entity0', f"a {meta.get('keyword0','entity0')}")
        e1 = meta.get('prompt_entity1', f"a {meta.get('keyword1','entity1')}")
        full_prompt = meta.get('prompt_full', f"{e0} and {e1}")
        tokens = pipe.tokenizer(
            full_prompt, return_tensors='pt', padding='max_length',
            max_length=pipe.tokenizer.model_max_length, truncation=True,
        ).to(device)
        return pipe.text_encoder(**tokens).last_hidden_state.half()

    angle_data = []
    for frames_np, depth_orders, meta, entity_masks_sample in angle_samples:
        vca_scores = _sigma_depth_scores(frames_np, entity_masks=entity_masks_sample, meta=meta)
        gt_scores  = _gt_depth_scores(frames_np, depth_orders, meta)
        angle      = meta.get('camera', 'unknown')
        e0_kw      = meta.get('keyword0', 'E0')
        e1_kw      = meta.get('keyword1', 'E1')
        angle_data.append({
            'frames': frames_np,
            'vca': vca_scores,
            'gt':  gt_scores,
            'angle': angle,
            'e0': e0_kw,
            'e1': e1_kw,
        })

    vca_layer.train()

    # ── Depth diagram 렌더링 함수 ────────────────────────────────────────────
    E0_COLOR = '#e84545'   # 빨강 (entity0)
    E1_COLOR = '#3a7fff'   # 파랑 (entity1)
    DIAGRAM_W = P
    DIAGRAM_H = P

    # 전체 VCA 스코어 범위를 미리 계산 → 동적 Y축 스케일
    all_vca_scores = []
    for ad in angle_data:
        if ad['vca']:
            for d0, d1 in ad['vca']:
                all_vca_scores += [d0, d1]
    if len(all_vca_scores) >= 2:
        vca_min = min(all_vca_scores)
        vca_max = max(all_vca_scores)
        margin  = max((vca_max - vca_min) * 0.3, 0.05)
        y_lo    = max(0.0, vca_min - margin)
        y_hi    = min(1.0, vca_max + margin)
        # 범위가 너무 좁으면 강제 확장 (차이 < 0.1)
        if y_hi - y_lo < 0.1:
            mid = (y_hi + y_lo) / 2
            y_lo, y_hi = max(0.0, mid - 0.1), min(1.0, mid + 0.1)
    else:
        y_lo, y_hi = 0.0, 1.0

    def _render_depth_diagram(d0_vca, d1_vca, d0_gt, d1_gt, e0_name, e1_name,
                               y_lo=y_lo, y_hi=y_hi):
        """
        세로 depth 다이어그램 (동적 Y축).
          상단 = 카메라 (FRONT)  하단 = 멀어짐 (BACK)
          filled circle = VCA 예측  /  X = GT
          빨강=E0, 파랑=E1

        y_lo/y_hi: 전체 frame에서 계산된 실제 스코어 범위 → 변화가 작아도 잘 보임
        """
        fig, ax = plt.subplots(figsize=(DIAGRAM_W / 72, DIAGRAM_H / 72),
                               dpi=72, facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(y_lo, y_hi)
        ax.invert_yaxis()  # 위=낮은값(가까움), 아래=높은값(멀음)

        # 깊이 축 + 레이블
        ax.axvline(0, color='#555577', linewidth=1.5, linestyle='--', alpha=0.5)
        ax.text(0, y_lo, '[CAM]', ha='center', va='bottom',
                fontsize=7, color='#aaaacc', fontweight='bold')
        ax.text(0, y_hi, 'BACK', ha='center', va='top',
                fontsize=6, color='#666688')

        # Y축 눈금선 (5개)
        for yv in np.linspace(y_lo, y_hi, 5):
            ax.axhline(yv, color='#333355', linewidth=0.5, alpha=0.4)
            ax.text(1.25, yv, f'{yv:.2f}', ha='left', va='center',
                    fontsize=5, color='#888899')

        # GT markers (큰 X, 테두리 강조)
        ax.scatter([-0.3], [d0_gt], marker='x', s=200, color=E0_COLOR,
                   linewidths=3.0, zorder=4)
        ax.scatter([0.3],  [d1_gt], marker='x', s=200, color=E1_COLOR,
                   linewidths=3.0, zorder=4)

        # VCA prediction (filled circle, 좌/우로 분리)
        ax.scatter([-0.3], [d0_vca], s=300, color=E0_COLOR, zorder=5, alpha=0.95,
                   edgecolors='white', linewidths=1.5)
        ax.scatter([0.3],  [d1_vca], s=300, color=E1_COLOR, zorder=5, alpha=0.95,
                   edgecolors='white', linewidths=1.5)

        # Entity 이름 라벨 (좌/우)
        ax.text(-0.65, d0_vca, f"{e0_name[:7]}", ha='right', va='center',
                fontsize=9, color=E0_COLOR, fontweight='bold')
        ax.text(0.65,  d1_vca, f"{e1_name[:7]}", ha='left', va='center',
                fontsize=9, color=E1_COLOR, fontweight='bold')

        # depth 값 (소수점 3자리)
        ax.text(-0.65, d0_vca + (y_hi - y_lo) * 0.08,
                f'{d0_vca:.3f}', ha='right', va='top',
                fontsize=6, color='#ccccdd')
        ax.text(0.65,  d1_vca + (y_hi - y_lo) * 0.08,
                f'{d1_vca:.3f}', ha='left', va='top',
                fontsize=6, color='#ccccdd')

        # 연결선 + FRONT/BACK 표시 (두 원 사이)
        mid_y = (d0_vca + d1_vca) / 2
        ax.plot([-0.3, 0.3], [d0_vca, d1_vca], color='#666688',
                linewidth=1.0, linestyle=':', alpha=0.6, zorder=3)
        if abs(d0_vca - d1_vca) > (y_hi - y_lo) * 0.05:
            winner    = e0_name[:5] if d0_vca < d1_vca else e1_name[:5]
            w_color   = E0_COLOR if d0_vca < d1_vca else E1_COLOR
            w_side    = -0.3    if d0_vca < d1_vca else 0.3
            ax.text(w_side, mid_y, 'FRONT', ha='center', va='center',
                    fontsize=6, color=w_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='#1a1a2e',
                              ec=w_color, alpha=0.7))

        # 범례
        legend_handles = [
            mpatches.Patch(color=E0_COLOR, label=f'RED={e0_name[:8]}'),
            mpatches.Patch(color=E1_COLOR, label=f'BLU={e1_name[:8]}'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='w',
                       markersize=5, label='VCA pred', linewidth=0),
            plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='w',
                       markersize=5, label='GT order', linewidth=0),
        ]
        ax.legend(handles=legend_handles, loc='lower right', fontsize=5,
                  facecolor='#2a2a3e', edgecolor='none', labelcolor='white',
                  framealpha=0.8)
        ax.axis('off')

        buf = _io.BytesIO()
        fig.tight_layout(pad=0.1)
        fig.savefig(buf, format='png', dpi=72,
                    bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)
        buf.seek(0)
        arr = np.array(PILImg.open(buf).resize((DIAGRAM_W, DIAGRAM_H),
                                                PILImg.BILINEAR))[:, :, :3]
        return arr

    # ── Line-chart PNG: 각 angle별 VCA depth score vs frame ─────────────────
    def _save_depth_linechart(angle_data, out_dir):
        """
        PNG: multiangle_depth_chart.png
        각 angle에 대해 subplot — X=frame, Y=depth_score
        E0(red) / E1(blue) VCA 예측선 + GT order (dashed)
        서로 다른 angle에서 depth ordering이 달라지는 패턴 확인용
        """
        n_ang = len(angle_data)
        fig, axes = plt.subplots(1, n_ang,
                                 figsize=(4 * n_ang, 4),
                                 facecolor='#0d0d1a', sharey=True)
        if n_ang == 1:
            axes = [axes]

        for ax, ad in zip(axes, angle_data):
            ax.set_facecolor('#0d0d1a')
            vca = ad['vca'] or []
            frames_idx = list(range(len(vca)))

            if vca:
                d0s = [s[0] for s in vca]
                d1s = [s[1] for s in vca]
                ax.plot(frames_idx, d0s, color=E0_COLOR, linewidth=2.0,
                        label=f"VCA {ad['e0'][:6]}", marker='o', markersize=3)
                ax.plot(frames_idx, d1s, color=E1_COLOR, linewidth=2.0,
                        label=f"VCA {ad['e1'][:6]}", marker='o', markersize=3)

                # Fill between — 앞에 있는 entity 강조
                for fi2, (d0, d1) in enumerate(vca):
                    color = E0_COLOR if d0 < d1 else E1_COLOR
                    ax.axvspan(fi2 - 0.4, fi2 + 0.4, alpha=0.08,
                               color=color)

            # GT order bands (horizontal guide: 0.3=front, 0.7=back)
            gt = ad['gt'] or []
            if gt:
                gt_d0 = [s[0] for s in gt]
                gt_d1 = [s[1] for s in gt]
                ax.plot(range(len(gt)), gt_d0, color=E0_COLOR,
                        linewidth=1.0, linestyle='--', alpha=0.5)
                ax.plot(range(len(gt)), gt_d1, color=E1_COLOR,
                        linewidth=1.0, linestyle='--', alpha=0.5)

            ax.set_title(ad['angle'], color='white', fontsize=9, pad=4)
            ax.set_xlabel('frame', color='#888899', fontsize=7)
            if ax == axes[0]:
                ax.set_ylabel('depth score\n(lower=closer)', color='#888899', fontsize=7)
            ax.tick_params(colors='#888899', labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333355')
            ax.legend(fontsize=6, facecolor='#1a1a2e', edgecolor='none',
                      labelcolor='white', framealpha=0.8)
            ax.invert_yaxis()  # 위=낮은값(가까움)
            ax.set_ylim(y_hi + 0.05, y_lo - 0.05)  # 동일 동적 스케일
            ax.grid(True, color='#222244', linewidth=0.5, alpha=0.6)

        # 상단 타이틀
        e0n = angle_data[0]['e0'] if angle_data else 'E0'
        e1n = angle_data[0]['e1'] if angle_data else 'E1'
        fig.suptitle(
            f"Multi-angle depth: RED={e0n}  BLUE={e1n}\n"
            f"solid=VCA pred  dashed=GT order  (lower score = closer to cam)",
            color='white', fontsize=8, y=1.02,
        )
        fig.tight_layout()
        chart_path = out_dir / 'multiangle_depth_chart.png'
        fig.savefig(str(chart_path), dpi=120, bbox_inches='tight',
                    facecolor='#0d0d1a')
        plt.close(fig)
        print(f"  [debug] {out_dir.name}/multiangle_depth_chart.png", flush=True)

    _save_depth_linechart(angle_data, out_dir)

    # ── GIF 생성 ─────────────────────────────────────────────────────────────
    n_frames = min(len(ad['frames']) for ad in angle_data)
    gif_frames = []

    for fi in range(n_frames):
        col_panels = []
        for ad in angle_data:
            frames_np = ad['frames']
            vca_scores = ad['vca']
            gt_scores  = ad['gt']
            angle      = ad['angle']

            gt_frame = _resize(frames_np[fi] if fi < len(frames_np) else frames_np[-1])

            if vca_scores and fi < len(vca_scores):
                d0_v, d1_v = vca_scores[fi]
            else:
                d0_v, d1_v = (y_lo + y_hi) / 2, (y_lo + y_hi) / 2

            d0_g, d1_g = gt_scores[fi] if fi < len(gt_scores) else (y_lo, y_hi)

            diagram = _render_depth_diagram(
                d0_v, d1_v, d0_g, d1_g,
                ad['e0'], ad['e1'],
            )

            e0n, e1n = ad['e0'][:5], ad['e1'][:5]
            col = np.vstack([
                _lbl(gt_frame, f"{angle}  |  RED={e0n}  BLUE={e1n}"),
                _lbl(diagram,  f"●=VCA pred  X=GT order  (up=near cam)"),
            ])
            col_panels.append(col)

        gif_frames.append(np.hstack(col_panels))

    if gif_frames:
        p = out_dir / "multiangle_depth.gif"
        iio2.mimsave(str(p), gif_frames, duration=300)
        print(f"  [debug] {p.parent.name}/multiangle_depth.gif", flush=True)


# ─── training_step ───────────────────────────────────────────────────────────

def training_step_p21(pipe, vca_layer, latents, encoder_hidden_states,
                      depth_orders, lambda_depth, lambda_ortho, device,
                      t_max=200, entity_masks=None,
                      lambda_diff=DEFAULT_LAMBDA_DIFF,
                      lambda_attn=DEFAULT_LAMBDA_ATTN,
                      trainable_attn_proc=None,
                      tok_idx_e0=None, tok_idx_e1=None):
    """
    Phase 25: depth fix (lambda_diff<<lambda_depth) + text attention supervision.

    trainable_attn_proc: TrainableAttnProcessor — last_weights_tensor with grad.
    tok_idx_e0/e1: token index lists for entity0/1 (color+entity tokens).
    """
    noise = torch.randn_like(latents)
    t = torch.randint(0, t_max, (1,), device=device).long()
    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

    vca_layer.reset_sigma_acc()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        pred_noise = pipe.unet(
            noisy_latents, t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

    ld = loss_diff(pred_noise.float(), noise.float())

    sigma_acc = vca_layer.sigma_acc
    if sigma_acc:
        l_depth = l_zorder_direct(sigma_acc, depth_orders, entity_masks=entity_masks)
        l_ort   = loss_ortho(vca_layer.depth_pe)
    else:
        l_depth = torch.tensor(0.0, device=device)
        l_ort   = torch.tensor(0.0, device=device)

    # text attention mask loss (Phase 25 핵심)
    l_attn = torch.tensor(0.0, device=device)
    if (trainable_attn_proc is not None
            and trainable_attn_proc.last_weights_tensor is not None
            and entity_masks is not None
            and tok_idx_e0 is not None and tok_idx_e1 is not None):
        l_attn = l_attn_mask_loss(
            trainable_attn_proc.last_weights_tensor,
            entity_masks, tok_idx_e0, tok_idx_e1, CAPTURE_N_HEADS,
        )

    # NaN guard: skip attn loss if numerical issue
    if torch.isnan(l_attn):
        l_attn = torch.tensor(0.0, device=device)

    loss = (lambda_diff * ld
            + lambda_depth * l_depth
            + lambda_ortho * l_ort
            + lambda_attn * l_attn)
    return {
        "loss":        loss.detach().item(),
        "l_diff":      ld.detach().item(),
        "l_depth":     l_depth.detach().item(),
        "l_ortho":     l_ort.detach().item(),
        "l_attn":      l_attn.detach().item(),
        "loss_tensor": loss,
    }


# ─── 메인 학습 루프 ───────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Phase 30] LoRA-based text attn (rank=4, fp32), lambda_attn={args.lambda_attn}",
          flush=True)
    print(f"[init] device={device}  lambda_depth={args.lambda_depth}  "
          f"lambda_diff={args.lambda_diff}  lambda_attn={args.lambda_attn}  alpha={VCA_ALPHA}",
          flush=True)

    if args.stats_path and Path(args.stats_path).exists():
        if not check_dataset_quality(args.stats_path):
            return
    else:
        print("DATASET_OK: proceeding", flush=True)

    dataset = ObjaverseDatasetWithMasks(
        data_root=args.data_root, max_samples=args.max_samples,
        n_frames=args.n_frames, height=args.height, width=args.width,
    )
    if len(dataset) == 0:
        print("DATASET_FAIL: no samples", flush=True)
        return

    n_samples = len(dataset)
    print(f"DATASET_OK: {n_samples} samples", flush=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=lambda x: x[0])

    pipe = load_pipeline(device=device, dtype=torch.float16)
    for p in pipe.unet.parameters():
        p.requires_grad = False

    save_dir  = Path(args.save_dir);  save_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.debug_dir); debug_dir.mkdir(parents=True, exist_ok=True)

    # probe 세트
    probe_frames, _, probe_orders, probe_meta, probe_masks = dataset[0]
    probe_entity_ctx = get_color_entity_context(pipe, probe_meta, device)
    e0_p, e1_p, full_probe_color, c0_p, c1_p = make_color_prompts(probe_meta)
    print(f"[probe] '{e0_p}' vs '{e1_p}'  (color-qualified)", flush=True)

    # Additive VCA 주입
    vca_layer, orig_procs = inject_vca_p21(pipe, probe_entity_ctx)

    # ── Phase 27: LoRALayer 주입 (fp32, 안정적) ──────────────────────────────
    # Phase 25/26 문제: fp16 to_q/to_k 직접 학습 → NaN (AdamW first step overflow)
    # Phase 27 해결: 원본 fp16 weights 완전 frozen + fp32 LoRA matrices만 학습
    attn2_module = _get_attn_module(pipe.unet, ATTN_CAPTURE_KEY)
    q_dim       = attn2_module.to_q.weight.shape[0]         # 320
    context_dim = attn2_module.to_k.weight.shape[1]         # 768

    lora_layer = LoRALayer(q_dim, context_dim, rank=4).to(device).float()
    lora_attn_proc = LoRAAttnProcessor(lora_layer)

    attn_procs_now = dict(pipe.unet.attn_processors)
    attn_procs_now[ATTN_CAPTURE_KEY] = lora_attn_proc
    pipe.unet.set_attn_processor(attn_procs_now)

    n_lora_params = sum(p.numel() for p in lora_layer.parameters())
    print(f"[p30] LoRA attn2 (q_dim={q_dim}, ctx={context_dim}, rank=4): "
          f"{n_lora_params:,} fp32 params @ {ATTN_CAPTURE_KEY}", flush=True)

    # 학습용 processor dict 저장 (trainable_attn_proc API 호환)
    trainable_attn_proc = lora_attn_proc
    train_procs = copy.copy(dict(pipe.unet.attn_processors))

    probe_latents = encode_frames_to_latents(pipe, probe_frames, device)
    # Phase 23: color-qualified probe prompt
    probe_tokens = pipe.tokenizer(
        full_probe_color, return_tensors="pt", padding="max_length",
        max_length=pipe.tokenizer.model_max_length, truncation=True,
    ).to(device)
    with torch.no_grad():
        probe_enc_hs = pipe.text_encoder(**probe_tokens).last_hidden_state.half()
    print(f"[probe_prompt] '{full_probe_color}'", flush=True)

    trainable_vca  = [p for p in vca_layer.parameters() if p.requires_grad]
    trainable_attn = list(lora_layer.parameters())   # fp32, 작음 (4 matrices, rank=4)

    # Also train the learnable gamma in AdditiveVCAProcessor
    vca_proc = pipe.unet.attn_processors.get(INJECT_KEY)
    trainable_gamma = [vca_proc.gamma] if isinstance(vca_proc, AdditiveVCAProcessor) else []

    trainable = trainable_vca + trainable_gamma + trainable_attn
    optimizer = torch.optim.AdamW(
        [{'params': trainable_vca,   'lr': args.lr, 'eps': 1e-8},
         {'params': trainable_gamma, 'lr': args.lr * 0.1, 'eps': 1e-8},  # conservative lr for gamma
         {'params': trainable_attn,  'lr': args.lr, 'eps': 1e-8}],
        # LoRA는 fp32 → eps 트릭 불필요, 동일 lr 사용
        weight_decay=1e-4,
    )
    print(f"[opt] VCA: {sum(p.numel() for p in trainable_vca):,}  "
          f"gamma: {len(trainable_gamma)} scalar  "
          f"LoRA: {sum(p.numel() for p in trainable_attn):,}", flush=True)

    lambda_depth = args.lambda_depth
    lambda_ortho = args.lambda_ortho
    best_sep     = 0.0
    best_dra     = 0.0   # depth_rank_accuracy
    training_curve = []

    for epoch in range(args.epochs):
        vca_layer.train()
        epoch_losses = {"loss": 0., "l_diff": 0., "l_depth": 0., "l_ortho": 0., "l_attn": 0.}
        epoch_steps  = 0
        last_frames_np = probe_frames
        epoch_delta_ratio = VCA_ALPHA  # per-epoch initialization

        for batch in loader:
            frames_np, depths_np, depth_orders, meta, entity_masks = batch
            last_frames_np = frames_np

            entity_ctx = get_color_entity_context(pipe, meta, device)
            proc = pipe.unet.attn_processors.get(INJECT_KEY)
            if isinstance(proc, AdditiveVCAProcessor):
                proc.ctx = entity_ctx.float()

            latents = encode_frames_to_latents(pipe, frames_np, device)
            # color-qualified prompt
            _, _, full_prompt, c0_name, c1_name = make_color_prompts(meta)
            e0_kw = meta.get('keyword0', meta.get('prompt_entity0', 'entity0'))
            e1_kw = meta.get('keyword1', meta.get('prompt_entity1', 'entity1'))
            tokens = pipe.tokenizer(
                full_prompt, return_tensors="pt", padding="max_length",
                max_length=pipe.tokenizer.model_max_length, truncation=True,
            ).to(device)
            with torch.no_grad():
                enc_hs = pipe.text_encoder(**tokens).last_hidden_state.half()

            # Condition Dropout (args.cond_dropout): args.cond_dropout 확률로 null text embedding 사용
            # 목적: backbone이 text conditioning 없이 denoise할 때 VCA entity ctx만 남음
            #        → VCA delta를 무시하면 l_diff 증가 → VCA로 강한 gradient
            if torch.rand(1).item() < args.cond_dropout:
                with torch.no_grad():
                    null_tokens = pipe.tokenizer(
                        "", return_tensors="pt", padding="max_length",
                        max_length=pipe.tokenizer.model_max_length, truncation=True,
                    ).to(device)
                    enc_hs = pipe.text_encoder(**null_tokens).last_hidden_state.half()

            # token indices for attn mask loss (per sample)
            full_ids = pipe.tokenizer(full_prompt, add_special_tokens=True)['input_ids']
            def _find_tok(kw):
                kw_ids = pipe.tokenizer(kw, add_special_tokens=False)['input_ids']
                for i in range(len(full_ids) - len(kw_ids) + 1):
                    if full_ids[i:i+len(kw_ids)] == kw_ids:
                        return list(range(i, i+len(kw_ids)))
                return [1]
            sample_tok_e0 = _find_tok(c0_name) + _find_tok(e0_kw)
            sample_tok_e1 = _find_tok(c1_name) + _find_tok(e1_kw)

            optimizer.zero_grad()
            step_out = training_step_p21(
                pipe, vca_layer, latents, enc_hs,
                depth_orders, lambda_depth, lambda_ortho, device, t_max=args.t_max,
                entity_masks=entity_masks,
                lambda_diff=args.lambda_diff, lambda_attn=args.lambda_attn,
                trainable_attn_proc=trainable_attn_proc,
                tok_idx_e0=sample_tok_e0, tok_idx_e1=sample_tok_e1,
            )
            step_out["loss_tensor"].backward()
            # Phase 27: LoRA fp32 → NaN 위험 없음, 적당한 clip
            torch.nn.utils.clip_grad_norm_(trainable_vca,  1.0)
            torch.nn.utils.clip_grad_norm_(trainable_attn, 0.5)
            # 안전을 위해 NaN grad 체크 유지
            for p in trainable_attn:
                if p.grad is not None and torch.isnan(p.grad).any():
                    p.grad.zero_()
            optimizer.step()

            # Track VCA delta ratio (diagnostic)
            proc_vca = train_procs.get(INJECT_KEY)
            if isinstance(proc_vca, AdditiveVCAProcessor):
                epoch_delta_ratio = getattr(proc_vca, 'last_delta_ratio', VCA_ALPHA)

            for k in epoch_losses:
                epoch_losses[k] += step_out[k]
            epoch_steps += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(epoch_steps, 1)

        probe_sep = measure_probe_sep(
            pipe, vca_layer, probe_latents, probe_enc_hs, device
        )

        # 핵심 지표: depth_rank_accuracy
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
            dra, n_correct, n_total = measure_depth_rank_accuracy(
                pipe, vca_layer, dataset, device,
                n_samples=min(20, n_samples), t_val=100,
            )
            print(f"  [dra] depth_rank_accuracy={dra:.3f} ({n_correct}/{n_total})",
                  flush=True)
        else:
            dra = best_dra

        if (epoch + 1) % 10 == 0:
            ld_on, ld_off, ld_ratio = measure_ldiff_ablation(
                pipe, vca_layer, dataset, device, n_samples=5,
            )
            if ld_on is not None:
                print(f"  [ablation] l_diff: VCA_on={ld_on:.4f}  VCA_off={ld_off:.4f}  "
                      f"ratio={ld_ratio:.3f}x  "
                      f"({'BACKBONE_DOMINANT' if ld_ratio < 1.05 else 'VCA_AFFECTS_GENERATION' if ld_ratio < 1.3 else 'VCA_HURTS_BACKBONE'})",
                      flush=True)

        l_diff_v  = epoch_losses["l_diff"]
        l_depth_w = epoch_losses["l_depth"] * lambda_depth
        ratio     = l_diff_v / max(abs(l_depth_w), 1e-9)

        # Phase 25: depth-focused → adaptive lambda disabled

        l_attn_v = epoch_losses['l_attn'] * args.lambda_attn
        delta_ratio_str = f"  vca_ratio={epoch_delta_ratio:.3f}"
        gamma_val = float(vca_proc.gamma.item()) if trainable_gamma else VCA_ALPHA
        gamma_str = f"  gamma={gamma_val:.3f}"
        print(
            f"epoch={epoch:3d} step={epoch_steps} "
            f"loss={epoch_losses['loss']:.4f} "
            f"l_diff={l_diff_v:.4f} "
            f"l_depth={l_depth_w:.4f} "
            f"l_attn={l_attn_v:.4f} "
            f"l_ortho={epoch_losses['l_ortho']:.4f} "
            f"ratio={ratio:.1f}x "
            f"probe_sep={probe_sep:.4f} "
            f"dra={dra:.3f}"
            f"{delta_ratio_str}"
            f"{gamma_str}",
            flush=True,
        )

        training_curve.append({
            "epoch": epoch, "lambda_depth": lambda_depth,
            **epoch_losses, "l_depth_weighted": l_depth_w,
            "probe_sep": probe_sep, "depth_rank_accuracy": dra,
        })

        if probe_sep > best_sep:
            best_sep = probe_sep
            best_dra = dra
            torch.save({
                "vca_state_dict":      vca_layer.state_dict(),
                "epoch":               epoch,
                "probe_sep":           best_sep,
                "depth_rank_accuracy": best_dra,
                "lambda_depth_final":  lambda_depth,
                "inject_key":          INJECT_KEY,
                "depth_pe_init_scale": DEPTH_PE_INIT_SCALE,
                "vca_alpha":           VCA_ALPHA,
                "multi_layer":         False,
                "additive":            True,   # 핵심 플래그
            }, save_dir / "best.pt")
            print(f"[ckpt] best.pt (probe_sep={best_sep:.4f} dra={best_dra:.3f})",
                  flush=True)

        # 디버그: debug_every epoch마다 + 첫/마지막 epoch
        if (epoch + 1) % args.debug_every == 0 or epoch == 0 or epoch == args.epochs - 1:
            epoch_dir = debug_dir / f"epoch_{epoch:03d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)

            # probe_entity_ctx 기준으로 train_procs 컨텍스트 업데이트
            proc = train_procs.get(INJECT_KEY)
            if isinstance(proc, AdditiveVCAProcessor):
                proc.ctx = probe_entity_ctx.float()

            # ① 학습 denoising 품질: GT / Noised / pred_x0
            debug_train_denoising(
                pipe, vca_layer,
                probe_latents, probe_enc_hs,
                probe_frames, epoch_dir,
                height=args.height, width=args.width,
            )

            # ② 텍스트 컨디셔닝 검증: full / null / e0 only / e1 only
            debug_text_cond(
                pipe, vca_layer,
                probe_latents, probe_meta,
                probe_frames, epoch_dir,
                height=args.height, width=args.width,
                t_values=(50, 150),
            )

            # ③ VCA depth 효과: Baseline vs VCA + σ map
            debug_depth_effect(
                pipe, vca_layer, orig_procs, train_procs,
                probe_latents, probe_enc_hs,
                probe_frames, probe_entity_ctx,
                epoch_dir, height=args.height, width=args.width,
                t_values=(50, 150),
            )

            # ④ text attention map + chart: entity 토큰이 올바른 공간에 attend하는지
            debug_text_attn(
                pipe, probe_latents, probe_enc_hs,
                probe_meta, probe_frames,
                epoch_dir, height=args.height, width=args.width,
                t_values=(50, 150),
                entity_masks=probe_masks,
            )

            # ⑤ 동일 motion 다각도: angle별 depth ordering 변화
            debug_multiangle_depth(
                pipe, vca_layer, dataset, train_procs,
                probe_entity_ctx,
                epoch_dir, height=args.height, width=args.width,
                t_val=100,
            )

            # ⑥ 실제 생성 비교: Baseline / VCA(normal ctx) / VCA(swapped ctx)
            #    normal vs swapped이 다르면 VCA가 실제로 depth ordering 제어 중
            debug_generation(
                pipe, vca_layer, orig_procs, train_procs,
                probe_frames, probe_meta, probe_entity_ctx,
                debug_dir, epoch,
                height=args.height, width=args.width,
            )

    # ─── 최종 평가 ────────────────────────────────────────────────────────────
    final_dra, fc, ft = measure_depth_rank_accuracy(
        pipe, vca_layer, dataset, device,
        n_samples=min(50, n_samples), t_val=100,
    )
    print(f"\nFINAL probe_sep={best_sep:.6f}", flush=True)
    print(f"FINAL depth_rank_accuracy={final_dra:.4f} ({fc}/{ft})", flush=True)

    if final_dra >= 0.65:
        print("IDEA=WORKS", flush=True)
    elif final_dra >= 0.55:
        print("IDEA=PARTIAL", flush=True)
    else:
        print("IDEA=FAIL", flush=True)

    if best_sep > 0.01:
        print("LEARNING=OK", flush=True)
    else:
        print("LEARNING=FAIL", flush=True)

    with open(debug_dir / "training_curve.json", "w") as f:
        json.dump(training_curve, f, indent=2)
    print(f"[done]", flush=True)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",    default="toy/data_objaverse",  dest="data_root")
    p.add_argument("--stats-path",   default="debug/dataset_stats/objaverse_stats.json",
                   dest="stats_path")
    p.add_argument("--lambda-depth", type=float, default=DEFAULT_LAMBDA_DEPTH, dest="lambda_depth")
    p.add_argument("--lambda-ortho", type=float, default=DEFAULT_LAMBDA_ORTHO, dest="lambda_ortho")
    p.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--lr",           type=float, default=DEFAULT_LR)
    p.add_argument("--t-max",        type=int,   default=DEFAULT_T_MAX, dest="t_max")
    p.add_argument("--lambda-diff",  type=float, default=DEFAULT_LAMBDA_DIFF,  dest="lambda_diff")
    p.add_argument("--lambda-attn",  type=float, default=DEFAULT_LAMBDA_ATTN,  dest="lambda_attn")
    p.add_argument("--save-dir",     default="checkpoints/phase30",    dest="save_dir")
    p.add_argument("--debug-dir",    default="debug/train_phase30",    dest="debug_dir")
    p.add_argument("--n-frames",     type=int,   default=8,  dest="n_frames")
    p.add_argument("--height",       type=int,   default=256)
    p.add_argument("--width",        type=int,   default=256)
    p.add_argument("--max-samples",  type=int,   default=None, dest="max_samples")
    p.add_argument("--debug-every",  type=int,   default=5,    dest="debug_every",
                   help="몇 epoch마다 recon/multiview 디버그 GIF 저장 (기본 5)")
    p.add_argument("--cond-dropout", type=float, default=0.15, dest="cond_dropout",
                   help="Probability of replacing text prompt with null embedding during training")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
