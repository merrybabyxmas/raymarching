from pathlib import Path
import pytest

pytestmark = pytest.mark.phase4

REQUIRED = ["FM-T1","FM-T2","FM-T3","FM-T4","FM-A1","FM-A2","FM-I"]

@pytest.fixture(scope="module")
def content():
    p = Path("docs/failures.md")
    assert p.exists()
    return p.read_text(encoding="utf-8")

@pytest.mark.parametrize("tag", REQUIRED)
def test_section_present(content, tag):
    assert tag in content, f"{tag} section missing"

def test_solutions_present(content):
    for tag in ["FM-T1","FM-T2","FM-T3"]:
        idx = content.find(tag)
        section = content[idx:idx+600]
        has_solution = any(w in section for w in ["해결","fix","solution","수정"])
        assert has_solution, f"{tag} has no solution documented"
