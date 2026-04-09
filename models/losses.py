"""
losses.py — VCA-Diffusion 손실 함수

세 가지 독립적인 손실:
  l_ortho         : depth_pe 직교성 (FM-T4 방지)
  l_depth_ranking : sigma depth ordering hinge loss
  l_diff          : diffusion 노이즈 복원 MSE
"""
import torch
import torch.nn.functional as F


def l_ortho(depth_pe: torch.Tensor) -> torch.Tensor:
    """
    depth_pe (Z, CD): z_bins 벡터들이 서로 직교하도록 강제.

    FM-T4 방지: L_ortho 없으면 모든 entity가 z=0으로 몰린다.

    손실 = ||G - I||_F² 여기서 G = normalized(depth_pe) @ normalized(depth_pe).T
    완벽한 직교 기저일 때 G = I → 손실 = 0
    """
    pe_norm = F.normalize(depth_pe, dim=-1)          # (Z, CD)
    gram    = pe_norm @ pe_norm.T                     # (Z, Z)
    eye     = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return (gram - eye).pow(2).mean()


def l_depth_ranking(
    sigma: torch.Tensor,
    depth_order: list,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    앞 entity(z=0)의 sigma가 뒤 entity보다 margin 이상 커야 한다.

    Parameters
    ----------
    sigma       : (BF, S, N, Z)
    depth_order : [front_entity_idx, back_entity_idx]
                  depth_order=[0,1] → entity 0이 앞, entity 1이 뒤.
    margin      : hinge margin (기본 0.05)

    손실 = mean(max(0, sigma_back[z=0] - sigma_front[z=0] + margin))
    올바른 순서(sigma_front > sigma_back + margin)이면 손실 = 0
    """
    front, back = depth_order
    s_front = sigma[:, :, front, 0]   # (BF, S) — z=0 slice of front entity
    s_back  = sigma[:, :, back,  0]   # (BF, S) — z=0 slice of back entity
    return F.relu(s_back - s_front + margin).mean()


def l_diff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """기본 diffusion 노이즈 복원 손실 (MSE)."""
    return F.mse_loss(pred, target)
