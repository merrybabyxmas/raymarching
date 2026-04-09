import pytest
import torch
import numpy as np

_phase_failed: set[int] = set()


@pytest.fixture(autouse=True)
def _fix_random_seed():
    """
    재현성 보장: 매 테스트 전 torch/numpy 시드를 42로 고정.

    FM-I4 교훈: test_entities_independent는 시드 고정으로 해결했지만
    근본 원인(anti-symmetric ctx → sigmoid 항등식)을 제거해 테스트를 재설계했다.
    이 fixture는 나머지 테스트의 일반적 재현성을 위해 유지한다.
    """
    torch.manual_seed(42)
    np.random.seed(42)

def pytest_runtest_makereport(item, call):
    if call.when == "call" and call.excinfo is not None:
        phase = _get_phase(item)
        if phase is not None:
            _phase_failed.add(phase)

def pytest_runtest_setup(item):
    phase = _get_phase(item)
    if phase is None or phase == 0:
        return
    for prev in range(phase):
        if prev in _phase_failed:
            pytest.skip(f"Phase {prev} failed — skipping Phase {phase}")

def _get_phase(item):
    for m in item.iter_markers():
        if m.name.startswith("phase"):
            try:
                return int(m.name.replace("phase", ""))
            except ValueError:
                pass
    return None
