from pathlib import Path
import pytest

pytestmark = pytest.mark.phase3

PAPERS = [
    ("Attend-and-Excite", "Chefer"), ("PEEKABOO", "Jain"),
    ("TokenFlow", "Geyer"),          ("CONFORM", "Meral"),
    ("NeRF", "Mildenhall"),          ("LoRA", "Hu"),
    ("AnimateDiff", "Guo"),          ("VideoComposer", "Wang"),
    ("MotionDirector", "Zhao"),      ("FLATTEN", "Cong"),
]

@pytest.fixture(scope="module")
def content():
    p = Path("docs/related_papers.md")
    assert p.exists(), "docs/related_papers.md not found"
    return p.read_text(encoding="utf-8")

@pytest.mark.parametrize("paper,author", PAPERS)
def test_paper_present(content, paper, author):
    assert paper in content and author in content

def test_vca_diff_present(content):
    assert "VCA" in content and ("차이" in content or "differs" in content.lower())

def test_min_length(content):
    assert len(content.split()) >= 800, f"Too short: {len(content.split())} words"
