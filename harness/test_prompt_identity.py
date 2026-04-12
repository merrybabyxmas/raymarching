"""
test_prompt_identity.py — collision prompt identity preservation

Verifies that shared prompt construction keeps real nouns like cat/dog,
preserves collision wording, and keeps entity context noun-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prompt_identity import make_color_prompts, make_identity_prompts


def test_colorized_collision_prompt_preserves_nouns_and_collision_phrase():
    meta = {
        "keyword0": "cat",
        "keyword1": "dog",
        "prompt_entity0": "a cat",
        "prompt_entity1": "a dog",
        "prompt_full": "a cat and a dog tangled together",
        "color0": [0.85, 0.15, 0.10],
        "color1": [0.10, 0.25, 0.85],
    }

    e0, e1, full, c0, c1 = make_color_prompts(meta)

    assert c0 == "red"
    assert c1 == "blue"
    assert "cat" in e0 and "red" in e0
    assert "dog" in e1 and "blue" in e1
    assert full.startswith("a cat and a dog tangled together")
    assert "cat" in full and "dog" in full
    assert "tangled together" in full
    assert "object 0 is" not in full
    assert "object 1 is" not in full


def test_identity_prompt_is_noun_only():
    meta = {
        "keyword0": "cat",
        "keyword1": "dog",
        "prompt_entity0": "a cat",
        "prompt_entity1": "a dog",
        "prompt_full": "a cat and a dog tangled together",
        "color0": [0.85, 0.15, 0.10],
        "color1": [0.10, 0.25, 0.85],
    }

    e0, e1, full, c0, c1 = make_identity_prompts(meta)

    assert c0 == "red"
    assert c1 == "blue"
    assert e0 == "a cat"
    assert e1 == "a dog"
    assert full == "a cat and a dog tangled together"


def test_fallback_prompt_keeps_collision_semantics():
    meta = {
        "keyword0": "cat",
        "keyword1": "dog",
        "color0": [0.85, 0.15, 0.10],
        "color1": [0.10, 0.25, 0.85],
    }

    e0, e1, full, _, _ = make_color_prompts(meta)

    assert "cat" in e0 and "dog" in e1
    assert "tangled together" in full
    assert "object 0 is" not in full and "object 1 is" not in full
    assert "cat" in full and "dog" in full


if __name__ == "__main__":
    test_colorized_collision_prompt_preserves_nouns_and_collision_phrase()
    test_identity_prompt_is_noun_only()
    test_fallback_prompt_keeps_collision_semantics()
    print("test_prompt_identity: PASS")
