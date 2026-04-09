"""
VCAAttnProcessor2_0 — VCALayer을 diffusers attn_processor 인터페이스로 감싸는 어댑터.

사용법:
    vca = VCALayer(query_dim=320, context_dim=768, ...)
    proc = VCAAttnProcessor2_0(vca)
    processors = {k: proc for k in unet.attn_processors if 'attn2' in k}
    unet.set_attn_processor(processors)

주의:
    encoder_hidden_states는 (BF, N, CD) 형태여야 한다.
    표준 CLIP 77 토큰 시퀀스가 아닌 N entity 임베딩을 별도로 구성해서 넘겨라.
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Optional

from models.vca_attention import VCALayer


class VCAAttnProcessor2_0:
    def __init__(self, vca_layer: VCALayer):
        self.vca_layer = vca_layer

    def __call__(
        self,
        attn,                                                    # diffusers Attention module (or None in tests)
        hidden_states: torch.Tensor,                             # (BF, S, D)
        encoder_hidden_states: Optional[torch.Tensor] = None,   # (BF, N, CD)
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            # self-attention fallback: scaled dot-product
            # attn 모듈의 to_q/k/v 사용
            BF, S, D = hidden_states.shape
            h = attn.heads
            d = D // h
            scale = d ** -0.5
            q = rearrange(attn.to_q(hidden_states), 'b s (h d) -> b h s d', h=h)
            k = rearrange(attn.to_k(hidden_states), 'b s (h d) -> b h s d', h=h)
            v = rearrange(attn.to_v(hidden_states), 'b s (h d) -> b h s d', h=h)
            out = F.scaled_dot_product_attention(q, k, v, scale=scale)
            out = rearrange(out, 'b h s d -> b s (h d)')
            return attn.to_out[0](out)

        # cross-attention: VCALayer 사용
        # encoder_hidden_states: (BF, N, CD)
        return self.vca_layer(hidden_states, encoder_hidden_states)
