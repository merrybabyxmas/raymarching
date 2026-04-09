"""
debug_gif.py — VCA sigma map 3-panel GIF 생성기

패널 구조: [생성 프레임 | E0 sigma heatmap | E1 sigma heatmap]
각 패널 panel_size px → total width = panel_size * 3

FM-I2 반영: GIF 쓰기는 imageio.v2.mimsave() 사용 (v3에는 없음)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import imageio.v2 as iio2
from PIL import Image, ImageDraw, ImageFont


def _sigma_to_heatmap(sigma_hw: np.ndarray, panel_size: int) -> np.ndarray:
    """(H, W) float32 → (panel_size, panel_size, 3) uint8 hot colormap"""
    s = sigma_hw.copy()
    s_min, s_max = s.min(), s.max()
    s = (s - s_min) / (s_max - s_min + 1e-6)

    # hot colormap: low=black, high=white via red→yellow→white
    r = np.clip(s * 3.0,        0, 1)
    g = np.clip(s * 3.0 - 1.0, 0, 1)
    b = np.clip(s * 3.0 - 2.0, 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)  # (H,W,3)

    img = Image.fromarray(rgb).resize((panel_size, panel_size), Image.NEAREST)
    return np.array(img)


def _add_label(panel: np.ndarray, text: str) -> np.ndarray:
    """패널 우상단에 흰색 텍스트 오버레이"""
    img = Image.fromarray(panel)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 4), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def make_debug_gif(
    frames_rgb: list,        # list[np.ndarray] (H,W,3) uint8
    sigma_maps: list,        # list[np.ndarray] (N,H,W) float32
    out_path,                # str | Path
    panel_size: int = 256,
) -> None:
    """
    3-panel GIF 생성:
      [ RGB frame | E0 sigma | E1 sigma ]
      total width = panel_size * 3, height = panel_size

    Parameters
    ----------
    frames_rgb : 프레임당 RGB 이미지 (H, W, 3) uint8
    sigma_maps : 프레임당 entity sigma (N, H, W) float32
    out_path   : 출력 GIF 경로
    panel_size : 각 패널의 픽셀 크기
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gif_frames = []
    for rgb, sigma in zip(frames_rgb, sigma_maps):
        # Panel 0: RGB frame
        p0 = np.array(Image.fromarray(rgb).resize((panel_size, panel_size), Image.BILINEAR))

        # Panel 1 & 2: per-entity sigma heatmap
        panels = [p0]
        n_entities = sigma.shape[0]
        for ei in range(n_entities):
            hmap = _sigma_to_heatmap(sigma[ei], panel_size)
            t_mean = float(sigma[ei].mean())
            label  = f"E{ei} σ={t_mean:.3f}"
            hmap   = _add_label(hmap, label)
            panels.append(hmap)

        # 부족한 패널 채우기 (N < 2인 경우 방어)
        while len(panels) < 3:
            panels.append(np.zeros((panel_size, panel_size, 3), dtype=np.uint8))

        # 가로로 concat: (panel_size, panel_size*3, 3)
        frame = np.concatenate(panels[:3], axis=1)
        gif_frames.append(frame)

    iio2.mimsave(str(out_path), gif_frames, duration=250)  # 250ms = 4fps


def make_comparison_gif(
    frames_rgb: list,
    sigmoid_sigma_maps: list,   # list[np.ndarray] (N, H, W) float32
    softmax_sigma_maps: list,   # list[np.ndarray] (N, H, W) float32
    out_path,
    panel_size: int = 64,
) -> None:
    """
    4-panel Sigmoid vs Softmax 비교 GIF:
    [Sigmoid-E0σ | Sigmoid-E1σ | Softmax-E0σ | Softmax-E1σ]
    total width = panel_size * 4

    Sigmoid: 두 entity가 동시에 high → 양쪽 패널 모두 밝음
    Softmax: 한 entity가 zero-sum으로 지배 → E1 패널이 어두움 (Disappearance)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gif_frames = []
    for rgb, sig_sigma, sof_sigma in zip(frames_rgb, sigmoid_sigma_maps, softmax_sigma_maps):
        panels = []
        for sigma, label_prefix in [(sig_sigma, 'Sig'), (sof_sigma, 'Sof')]:
            n_e = sigma.shape[0]
            for ei in range(min(n_e, 2)):
                hmap  = _sigma_to_heatmap(sigma[ei], panel_size)
                t_mean = float(sigma[ei].mean())
                hmap  = _add_label(hmap, f"{label_prefix}-E{ei} σ={t_mean:.2f}")
                panels.append(hmap)

        while len(panels) < 4:
            panels.append(np.zeros((panel_size, panel_size, 3), dtype=np.uint8))

        frame = np.concatenate(panels[:4], axis=1)
        gif_frames.append(frame)

    iio2.mimsave(str(out_path), gif_frames, duration=250)
