import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


class LoRALinear(nn.Module):
    """base weight frozen, LoRA branch만 학습.
    초기화: lora_B=zeros → 초기 LoRA 기여 0 (원 논문 방식, 분산 안정)
    lora_A=randn*0.01은 의도적. kaiming으로 바꾸지 마라 — lora_B=zeros라 초기 출력은 어차피 0.
    """
    def __init__(self, in_f: int, out_f: int, rank: int = 8):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight)
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))

    def forward(self, x):
        return F.linear(x, self.weight) + F.linear(F.linear(x, self.lora_A), self.lora_B)


class VCALayer(nn.Module):
    """
    Volumetric Cross-Attention.
    핵심: Sigmoid (독립 밀도) + Transmittance (물리적 occlusion)
    절대로 Softmax로 바꾸지 마라 — zero-sum이 되어 Disappearance 발생

    context_dim 선택:
      SD 1.5  기반 AnimateDiff → context_dim=768
      SDXL    기반 (Dual CLIP) → context_dim=2048
    파이프라인을 확정하기 전까지 하드코딩하지 말고 반드시 인자로 받아라.
    """
    def __init__(self, query_dim, context_dim: int, n_heads=8,
                 n_entities=2, z_bins=2, lora_rank=8, use_softmax=False):
        # context_dim에 기본값 없음 — 호출부에서 명시적으로 지정 강제
        # use_softmax=True: Softmax variant (ablation용, 실제 VCA는 항상 False)
        super().__init__()
        assert query_dim % n_heads == 0
        self.n_heads, self.n_entities, self.z_bins = n_heads, n_entities, z_bins
        self.use_softmax = use_softmax
        self.head_dim = query_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = LoRALinear(context_dim, query_dim, rank=lora_rank)
        self.to_v = LoRALinear(context_dim, query_dim, rank=lora_rank)
        self.depth_pe = nn.Parameter(torch.randn(z_bins, context_dim) * 0.02)
        self.to_out = nn.Linear(query_dim, query_dim)

        self.last_sigma: Optional[torch.Tensor] = None         # (BF, S, N, Z) detached
        self.last_sigma_raw: Optional[torch.Tensor] = None    # (BF, S, N, Z) with grad (for loss)
        self.last_transmittance: Optional[torch.Tensor] = None # (BF, S, Z)

    def _expand_context(self, ctx):
        # ctx: (BF, N, CD) → (BF, N*Z, CD)
        pe = self.depth_pe.unsqueeze(0).unsqueeze(0)           # (1,1,Z,CD)
        ctx3d = ctx.unsqueeze(2) + pe                          # (BF,N,Z,CD)
        return rearrange(ctx3d, 'bf n z cd -> bf (n z) cd')

    def _transmittance(self, sigma):
        # sigma: (BF,S,N,Z) → T: (BF,S,Z)
        # T[z=0]=1, T[z=k]=prod_{z'<k}(1 - clamp(sum_n sigma[z'], max=1))
        opacity = sigma.sum(dim=2).clamp(max=1.0)
        ones = torch.ones(*opacity.shape[:2], 1, device=sigma.device, dtype=sigma.dtype)
        return torch.cumprod(torch.cat([ones, 1.0 - opacity], dim=-1), dim=-1)[:, :, :-1]

    def forward(self, x, context):
        # x: (BF,S,D)  context: (BF,N,CD)
        BF, S, _ = x.shape
        N, Z, h = self.n_entities, self.z_bins, self.n_heads

        Q = rearrange(self.to_q(x), 'bf s (h d) -> bf h s d', h=h)
        ctx3d = self._expand_context(context)
        K = rearrange(self.to_k(ctx3d), 'bf (n z) (h d) -> bf h (n z) d', h=h, n=N, z=Z)
        V = rearrange(self.to_v(ctx3d), 'bf (n z) (h d) -> bf h (n z) d', h=h, n=N, z=Z)

        scores = torch.einsum('bhsd,bhtd->bhst', Q, K) * self.scale
        if self.use_softmax:
            # zero-sum: N*Z key 전체에 Softmax → entity 간 경쟁 → Disappearance
            sigma_flat = torch.softmax(scores, dim=-1)           # (BF,h,S,N*Z)
        else:
            # Sigmoid: entity 독립 [0,1] 밀도 → 동시에 high 가능
            sigma_flat = torch.sigmoid(scores)
        sigma = rearrange(sigma_flat, 'bf h s (n z) -> bf h s n z', n=N, z=Z)

        sigma_mean = sigma.mean(dim=1)                         # (BF,S,N,Z)
        T = self._transmittance(sigma_mean)                    # (BF,S,Z)

        weights = T.unsqueeze(1).unsqueeze(3) * sigma          # (BF,h,S,N,Z)
        weights = rearrange(weights, 'bf h s n z -> bf h s (n z)')
        out = rearrange(torch.einsum('bhst,bhtd->bhsd', weights, V), 'bf h s d -> bf s (h d)')

        self.last_sigma_raw = sigma_mean                   # with grad — for loss computation
        self.last_sigma = sigma_mean.detach()              # detached — for metrics / GIF
        self.last_transmittance = T.detach()

        return x + self.to_out(out)
