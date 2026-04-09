import importlib
import pytest

PACKAGES = ["torch", "diffusers", "transformers", "einops", "pyvista", "imageio", "numpy"]

@pytest.mark.phase0
@pytest.mark.parametrize("pkg", PACKAGES)
def test_import(pkg):
    assert importlib.import_module(pkg) is not None

@pytest.mark.phase0
def test_torch_version():
    import torch
    assert int(torch.__version__.split(".")[0]) >= 2

@pytest.mark.phase0
def test_pyvista_offscreen():
    import pyvista as pv
    try:
        pv.start_xvfb()      # headless 환경 자동 처리 (Xvfb 없으면 skip)
    except (OSError, Exception):
        pass                 # EGL/OSMesa fallback으로 off_screen 렌더링
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(pv.Sphere())
    pl.render()
    img = pl.screenshot(return_img=True)
    pl.close()
    assert img.shape[2] == 3  # RGB
