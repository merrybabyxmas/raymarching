"""Shared helpers for identity-preserving collision prompts."""

from __future__ import annotations


_COLOR_TABLE = [
    ((0.6, 0.2, 0.2), "red"),
    ((0.2, 0.2, 0.6), "blue"),
    ((0.2, 0.6, 0.2), "green"),
    ((0.6, 0.6, 0.1), "yellow"),
    ((0.5, 0.1, 0.5), "purple"),
    ((0.6, 0.3, 0.1), "orange"),
    ((0.5, 0.5, 0.5), "gray"),
]


def rgb_to_color_name(rgb: list) -> str:
    """Map an RGB triplet to the nearest named color."""
    r, g, b = rgb
    best, best_dist = "colored", 1e9
    for (cr, cg, cb), name in _COLOR_TABLE:
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_dist:
            best_dist, best = d, name
    return best


def _entity_core(text: str) -> str:
    text = str(text).strip()
    for prefix in ("a ", "an ", "the "):
        if text.lower().startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _colorize_entity(color_name: str, entity_text: str) -> str:
    text = _entity_core(entity_text)
    if text:
        return f"a {color_name} {text}"
    return f"a {color_name}"


def _noun_entity_prompt(entity_text: str) -> str:
    text = _entity_core(entity_text)
    if text:
        if text.lower().startswith(("a ", "an ", "the ")):
            return text
        return f"a {text}"
    return "an entity"


def _build_full_prompt(base_prompt: str, noun0: str, noun1: str, suffix: str) -> str:
    full = str(base_prompt or "").strip()
    if full:
        return full
    return f"{noun0} and {noun1} {suffix}".strip()


def make_identity_prompts(meta: dict) -> tuple[str, str, str, str, str]:
    """
    Build noun-only entity prompts and the collision prompt.

    Returns (e0_prompt, e1_prompt, full_prompt, color0_name, color1_name).
    """
    c0 = rgb_to_color_name(meta.get("color0", [0.85, 0.15, 0.1]))
    c1 = rgb_to_color_name(meta.get("color1", [0.1, 0.25, 0.85]))
    kw0 = str(meta.get("keyword0", meta.get("prompt_entity0", "entity0"))).strip()
    kw1 = str(meta.get("keyword1", meta.get("prompt_entity1", "entity1"))).strip()
    e0 = _noun_entity_prompt(kw0)
    e1 = _noun_entity_prompt(kw1)
    collision_suffix = str(meta.get("collision_phrase", "tangled together")).strip() or "tangled together"
    full = _build_full_prompt(meta.get("prompt_full", ""), e0, e1, collision_suffix)
    return e0, e1, full, c0, c1


def make_color_prompts(meta: dict) -> tuple[str, str, str, str, str]:
    """
    Build color-qualified object prompts while preserving collision semantics.

    Returns (e0_prompt, e1_prompt, full_prompt, color0_name, color1_name).
    """
    c0 = rgb_to_color_name(meta.get("color0", [0.85, 0.15, 0.1]))
    c1 = rgb_to_color_name(meta.get("color1", [0.1, 0.25, 0.85]))
    kw0 = str(meta.get("keyword0", meta.get("prompt_entity0", "entity0"))).strip()
    kw1 = str(meta.get("keyword1", meta.get("prompt_entity1", "entity1"))).strip()
    e0 = _colorize_entity(c0, kw0)
    e1 = _colorize_entity(c1, kw1)

    collision_suffix = str(meta.get("collision_phrase", "tangled together")).strip() or "tangled together"
    noun0 = _noun_entity_prompt(kw0)
    noun1 = _noun_entity_prompt(kw1)
    full = _build_full_prompt(meta.get("prompt_full", ""), noun0, noun1, collision_suffix)
    return e0, e1, full, c0, c1
