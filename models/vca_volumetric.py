"""
Phase 35 — Volumetric Text Cross-Attention (VTA)

핵심 아이디어
----------
기존 text cross-attention (attn2)는 2D spatial (S) 공간에서 attention을 계산한다.
두 entity가 같은 (x,y) 위치에 겹칠 때, 두 entity의 text token이 동시에 같은 spatial
token에 attend → chimera mixing.

Fix: spatial Q 토큰을 depth-bin (z) 으로 확장:
    Q_vol[s*Z + z] = Q_std[s] + z_pe[z]     z_pe ∈ R^{Z×D} (학습)

Text token은 이제 (S×Z) 볼류메트릭 공간에서 attend한다.
GT supervision 후:
  - z=0 (front bin): 앞 entity text token에 attend를 배움
  - z=1 (back  bin): 뒤 entity text token에 attend를 배움

결과: overlap 위치에서도 z-bin 이 다르므로 attention이 분리 → chimera 구조적 불가능.

아키텍처 안전성
--------------
초기화: z_pe ≈ 0 → 학습 초기에 volumetric ≈ standard attention (훈련 안정)
blend:  output = text_out + gamma × (vol_agg - text_out)
        gamma=0 → 순수 standard, gamma=1 → 순수 volumetric
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn


# =============================================================================
# VolumetricTextCrossAttentionProcessor
# =============================================================================

class VolumetricTextCrossAttentionProcessor(nn.Module):
    """
    attn2 (text cross-attention) 를 volumetric 버전으로 교체.

    Standard attn2: Q=(B,S,D),   K=V=(B,T,D)  → (B,S,D)
    Volumetric:     Q=(B,S*Z,D), K=V=(B,T,D)  → (B,S*Z,D) → aggregate → (B,S,D)

    Parameters
    ----------
    query_dim   : D (= INJECT_QUERY_DIM, up_blocks.2 에서는 640)
    z_bins      : depth bin 수 (기본 2: front / back)
    vca_layer   : VCALayer — sigma 계산용 (z-bin 가중 aggregation에 사용)
    entity_ctx  : (1, N, 768) entity CLIP 임베딩 (VCA sigma 입력)
    gamma_init  : volumetric contribution 초기 비율 (0=off, 1=full)
    """

    def __init__(
        self,
        query_dim: int,
        z_bins: int = 2,
        vca_layer=None,
        entity_ctx: Optional[torch.Tensor] = None,
        gamma_init: float = 0.5,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.z_bins = z_bins
        self.vca = vca_layer
        self.entity_ctx = entity_ctx

        # z-positional encoding — initialized near-zero.
        # 초기에는 모든 z-version이 동일 → volumetric ≈ standard.
        # 훈련하면서 z-bin별로 차별화됨.
        self.z_pe = nn.Parameter(torch.zeros(z_bins, query_dim))
        nn.init.normal_(self.z_pe, std=0.02)

        # gamma: standard vs volumetric 혼합 비율
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        # 훈련 루프에서 loss 계산을 위해 보관 (grad 유지)
        self.last_attn_weights: Optional[torch.Tensor] = None  # (B, S*Z, T) grad 포함
        self.last_sigma: Optional[torch.Tensor] = None          # (B, S, N, Z)
        self.sigma_acc: list = []   # l_zorder_direct 호환

    # ------------------------------------------------------------------
    def reset_sigma_acc(self):
        self.sigma_acc = []
        self.last_attn_weights = None

    def set_entity_ctx(self, ctx: torch.Tensor):
        self.entity_ctx = ctx

    # ------------------------------------------------------------------
    def __call__(
        self,
        attn,                                           # diffusers Attention 모듈
        hidden_states: torch.Tensor,                   # (B, S, D)
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ) -> torch.Tensor:
        B, S, D = hidden_states.shape
        Z = self.z_bins
        dtype = hidden_states.dtype

        enc_hs = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        T_seq = enc_hs.shape[1]

        # inner_dim may differ from D when group-norm is used; get from weight
        inner_dim = attn.to_q.weight.shape[0]
        n_heads   = attn.heads
        head_dim  = inner_dim // n_heads
        scale     = head_dim ** -0.5

        # ── helper: reshape to multi-head ─────────────────────────────────
        def _mh(x: torch.Tensor, seq_len: int) -> torch.Tensor:
            return x.reshape(B, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)

        # ── 1. Standard K, V (text) — shared for both paths ───────────────
        k = attn.to_k(enc_hs)                           # (B, T, inner_dim)
        v = attn.to_v(enc_hs)
        k_mh = _mh(k, T_seq)                            # (B, H, T, Dh)
        v_mh = _mh(v, T_seq)

        # ── 2. Standard Q (spatial) — baseline text cross-attention ───────
        q_std    = attn.to_q(hidden_states)             # (B, S, inner_dim)
        q_std_mh = _mh(q_std, S)                        # (B, H, S, Dh)

        scores_std = torch.matmul(q_std_mh, k_mh.transpose(-1, -2)) * scale  # (B,H,S,T)
        w_std      = scores_std.softmax(dim=-1)
        out_std    = (torch.matmul(w_std, v_mh)
                      .permute(0, 2, 1, 3)
                      .reshape(B, S, inner_dim))                # (B, S, inner_dim)

        # ── 3. VCA sigma — z-bin 가중치 계산용 ───────────────────────────
        sigma: Optional[torch.Tensor] = None
        if self.vca is not None and self.entity_ctx is not None:
            ctx = self.entity_ctx.expand(B, -1, -1).to(dtype)
            # VCA는 fp32로 계산하고 sigma 저장
            vca_hidden = hidden_states.float()
            vca_ctx    = ctx.float()
            if self.training:
                _ = self.vca(vca_hidden, vca_ctx)
            else:
                with torch.no_grad():
                    _ = self.vca(vca_hidden, vca_ctx)
            sigma = getattr(self.vca, 'last_sigma', None)
            if sigma is not None:
                self.last_sigma = sigma.detach().float()
                if self.training:
                    self.sigma_acc.append(self.vca.last_sigma_raw
                                          if hasattr(self.vca, 'last_sigma_raw')
                                          else sigma.float())

        # ── 4. Volumetric Q: (B, S*Z, D) ──────────────────────────────────
        z_pe = self.z_pe.to(dtype)                       # (Z, D)
        # broadcast: (B,S,1,D) + (1,1,Z,D) → (B,S,Z,D) → (B,S*Z,D)
        h_vol = (hidden_states.unsqueeze(2) + z_pe[None, None, :, :]).reshape(B, S * Z, D)

        q_vol    = attn.to_q(h_vol)                      # (B, S*Z, inner_dim)
        q_vol_mh = _mh(q_vol, S * Z)                     # (B, H, S*Z, Dh)

        scores_vol = torch.matmul(q_vol_mh, k_mh.transpose(-1, -2)) * scale  # (B,H,S*Z,T)
        w_vol      = scores_vol.softmax(dim=-1)           # (B, H, S*Z, T)

        # 훈련 시 grad 유지, 추론 시 detach
        self.last_attn_weights = w_vol.mean(dim=1)       # (B, S*Z, T)

        out_vol_flat = (torch.matmul(w_vol, v_mh)
                        .permute(0, 2, 1, 3)
                        .reshape(B, S * Z, inner_dim))    # (B, S*Z, inner_dim)

        # ── 5. Z-bin aggregation → (B, S, inner_dim) ──────────────────────
        out_3d = out_vol_flat.reshape(B, S, Z, inner_dim)

        if sigma is not None and sigma.shape[:2] == (B, S):
            # sigma: (B,S,N,Z) → max over N → (B,S,Z) softmax → 가중합
            sig = sigma.to(device=out_3d.device, dtype=out_3d.dtype)
            sig_w = sig.max(dim=2).values.softmax(dim=2).unsqueeze(-1)  # (B,S,Z,1)
            agg   = (out_3d * sig_w).sum(dim=2)           # (B, S, inner_dim)
        else:
            agg = out_3d.mean(dim=2)                       # (B, S, inner_dim)

        # ── 6. Blend: text_out + gamma × (vol_agg - text_out) ─────────────
        gamma   = self.gamma.clamp(-3.0, 3.0)
        blended = out_std + gamma * (agg - out_std)       # (B, S, inner_dim)

        # ── 7. Output projection ──────────────────────────────────────────
        out = attn.to_out[0](blended)
        out = attn.to_out[1](out)                         # dropout

        return out.to(dtype)


# =============================================================================
# Helper: entity token index lookup
# =============================================================================

def find_entity_token_positions(tokenizer, full_prompt: str, entity_text: str) -> List[int]:
    """
    full_prompt 를 tokenize한 결과에서 entity_text 에 해당하는 토큰 인덱스 반환.

    Returns empty list if entity_text is not found as a contiguous sub-sequence.
    """
    full_ids   = tokenizer(
        full_prompt, add_special_tokens=True,
        max_length=tokenizer.model_max_length, truncation=True,
    ).input_ids
    entity_ids = tokenizer(entity_text, add_special_tokens=False).input_ids
    n = len(entity_ids)

    positions: List[int] = []
    for i in range(len(full_ids) - n + 1):
        if full_ids[i:i + n] == entity_ids:
            positions.extend(range(i, i + n))
    return positions


# =============================================================================
# Volumetric attention supervision loss
# =============================================================================

def l_pe_antisep(z_pe: torch.Tensor) -> torch.Tensor:
    """
    z_pe[0]과 z_pe[1] 을 antiparallel (cos → -1) 로 적극 유도.

    l_pe_antisep = (1 + cos(z_pe[0], z_pe[1]))^2  ∈ [0, 4]
      cos = -1  → loss = 0   (antiparallel, ideal)
      cos =  0  → loss = 1   (orthogonal)
      cos = +1  → loss = 4   (co-alignment, worst)

    Phase 36 의 l_pe_sep (co-alignment 패널티) 보다 강한 버전:
    anti-alignment 를 적극 유도한다.
    """
    v0 = z_pe[0].float()
    v1 = z_pe[1].float()
    cos = torch.dot(v0, v1) / (v0.norm() * v1.norm() + 1e-8)
    return (1.0 + cos).pow(2)


def l_pe_sep(z_pe: torch.Tensor) -> torch.Tensor:
    """
    z_pe[0]과 z_pe[1] 간 코사인 유사도 패널티.

    두 depth bin 이 같은 방향으로 collapse되는 걸 방지한다.
    co-alignment (cos > 0) 만 패널티 — anti-alignment (cos < 0, 반대 방향)는 좋음.

    l_pe_sep = relu(cos(z_pe[0], z_pe[1]))^2  ∈ [0, 1]
    """
    v0 = z_pe[0].float()
    v1 = z_pe[1].float()
    cos = torch.dot(v0, v1) / (v0.norm() * v1.norm() + 1e-8)
    return torch.nn.functional.relu(cos).pow(2)


def l_pe_norm(z_pe: torch.Tensor, target_norm: float = 25.0) -> torch.Tensor:
    """
    z_pe 각 bin 의 norm 이 target_norm 아래로 내려가지 않도록 패널티.

    target_norm = sqrt(D) ≈ sqrt(640) ≈ 25:
    이 크기에서 z_pe 가 hidden_states 와 동등한 크기를 가져
    attention score 를 의미있게 조종할 수 있다.

    l_pe_norm = mean_z max(0, target - ||z_pe[z]||)^2
    """
    per_bin_norm = z_pe.float().norm(dim=-1)          # (Z,)
    deficit = (target_norm - per_bin_norm).clamp(min=0.0)
    return deficit.pow(2).mean()


def l_vol_attn_loss(
    attn_weights: torch.Tensor,     # (B, S*Z, T_seq)  mean over heads
    entity_tok_e0: List[int],       # token indices of E0 in full_prompt
    entity_tok_e1: List[int],       # token indices of E1 in full_prompt
    entity_masks: torch.Tensor,     # (B, 2, S) float32 in [0, 1]
    depth_orders: list,             # [(front_entity_idx, back_entity_idx)] len >= B
    z_bins: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    GT-supervised volumetric attention loss.

    각 entity가 점유하는 spatial 위치 s 에서:
      - front entity → z=0 (near bin) 가 해당 entity text tokens에 attend해야 함
      - back  entity → z=Z-1 (far bin) 가 해당 entity text tokens에 attend해야 함

    Loss = −mean log( Σ attn_weight[vol_idx, entity_tok] )

    이 loss를 통해 z_pe 는 depth-stratified attention을 학습한다.
    """
    B, SZ, T_seq = attn_weights.shape
    S = SZ // z_bins
    device = attn_weights.device

    losses: list = []
    for b in range(B):
        if b >= len(depth_orders):
            break
        front_e = int(depth_orders[b][0])
        back_e  = int(depth_orders[b][1])

        for ei in range(2):
            tok_list = entity_tok_e0 if ei == 0 else entity_tok_e1
            if not tok_list:
                continue

            tok_t = torch.tensor(
                [t for t in tok_list if t < T_seq],
                device=device, dtype=torch.long,
            )
            if tok_t.numel() == 0:
                continue

            mask = entity_masks[b, ei]               # (S,) float
            pos  = (mask > 0.3).nonzero(as_tuple=False).squeeze(-1)  # (n_pos,)
            if pos.numel() == 0:
                continue

            # front entity → z=0; back entity → z=Z-1
            z = 0 if ei == front_e else (z_bins - 1)
            vol_idx = (pos * z_bins + z).clamp(0, SZ - 1)  # (n_pos,)

            # attention mass on correct entity tokens at the correct z-bin
            a = attn_weights[b][vol_idx][:, tok_t]          # (n_pos, n_tok)
            a = a.sum(dim=-1).clamp(min=eps)                 # (n_pos,)
            losses.append(-a.log().mean())

    if not losses:
        # 유효 샘플 없음 — grad graph 유지 위해 attn_weights 사용
        return attn_weights.sum() * 0.0

    return torch.stack(losses).mean()
