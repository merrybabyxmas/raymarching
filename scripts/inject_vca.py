"""
inject_vca.py — AnimateDiff UNet에 VCALayer를 cross-attention processor로 주입.

FM-A1 반영: hook 방식 금지. unet.set_attn_processor() 공식 API 사용.
"""
from __future__ import annotations
from models.vca_attention import VCALayer
from models.vca_processor import VCAAttnProcessor2_0


def inject_vca(
    unet,
    query_dim: int,
    context_dim: int,
    n_entities: int = 2,
    z_bins: int = 2,
    lora_rank: int = 8,
    n_heads: int = 8,
) -> dict:
    """
    UNet의 모든 cross-attention processor를 VCAAttnProcessor2_0으로 교체.

    Parameters
    ----------
    unet        : AnimateDiff / SD UNet (diffusers UNet2DConditionModel)
    query_dim   : SD 1.5 → 320, SDXL → 64
    context_dim : CLIP 임베딩 차원 (SD 1.5 → 768, SDXL → 2048)

    Returns
    -------
    vca_layers : {layer_key: VCALayer}  sigma 수집·학습에 사용
    """
    vca_layers: dict[str, VCALayer] = {}
    new_processors: dict = {}

    for key, _ in unet.attn_processors.items():
        if 'attn2' in key:  # cross-attention만 교체
            layer = VCALayer(
                query_dim=query_dim,
                context_dim=context_dim,
                n_heads=n_heads,
                n_entities=n_entities,
                z_bins=z_bins,
                lora_rank=lora_rank,
            )
            vca_layers[key] = layer
            new_processors[key] = VCAAttnProcessor2_0(layer)

    unet.set_attn_processor(new_processors)
    return vca_layers
