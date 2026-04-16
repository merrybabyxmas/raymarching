"""scene_prior/entity_parser.py
================================
Lightweight text parser for two-entity scene prompts.

Usage
-----
>>> from scene_prior.entity_parser import parse_prompt
>>> sp = parse_prompt("a red cat and a blue dog")
>>> sp.entities[0].name, sp.entities[0].color
('cat', [0.85, 0.15, 0.1])
>>> sp.entities[1].name, sp.entities[1].color
('dog', [0.1, 0.2, 0.85])
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Known vocabulary
# ---------------------------------------------------------------------------

KNOWN_ENTITIES = [
    "cat", "dog", "wolf", "snake", "alligator",
    "person", "sword", "unknown",
]

# Named-color table:  name → approximate (R, G, B) ∈ [0, 1]
_COLOR_MAP: dict[str, list[float]] = {
    "red":    [0.85, 0.15, 0.10],
    "green":  [0.10, 0.75, 0.15],
    "blue":   [0.10, 0.20, 0.85],
    "yellow": [0.90, 0.85, 0.10],
    "orange": [0.90, 0.50, 0.05],
    "purple": [0.55, 0.05, 0.80],
    "pink":   [0.95, 0.45, 0.65],
    "white":  [0.95, 0.95, 0.95],
    "black":  [0.05, 0.05, 0.05],
    "grey":   [0.50, 0.50, 0.50],
    "gray":   [0.50, 0.50, 0.50],
    "brown":  [0.55, 0.30, 0.05],
    "cyan":   [0.05, 0.85, 0.85],
}

# Default per-slot colors when not specified
_DEFAULT_COLORS: list[list[float]] = [
    _COLOR_MAP["red"],   # entity 0 default
    _COLOR_MAP["blue"],  # entity 1 default
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EntitySpec:
    """Specification for a single entity in the scene."""

    name: str
    attributes: List[str] = field(default_factory=list)
    color: Optional[List[float]] = None  # (R, G, B) ∈ [0, 1]


@dataclass
class ScenePrompt:
    """Parsed scene description with up to 2 entities."""

    entities: List[EntitySpec]
    interaction: str
    background: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_name(token: str) -> str:
    """Return the canonical entity name or 'unknown'."""
    t = token.lower().strip()
    # exact match
    if t in KNOWN_ENTITIES:
        return t
    # prefix match (e.g. "cats" → "cat")
    for k in KNOWN_ENTITIES:
        if t.startswith(k) or k.startswith(t):
            return k
    return "unknown"


def _extract_color(tokens: list[str]) -> tuple[Optional[list[float]], list[str]]:
    """Scan *tokens* for a color word; return (color, remaining_tokens)."""
    remaining: list[str] = []
    found: Optional[list[float]] = None
    for tok in tokens:
        t = tok.lower().strip().rstrip("s")  # strip trailing plural s
        if t in _COLOR_MAP and found is None:
            found = _COLOR_MAP[t]
        else:
            remaining.append(tok)
    return found, remaining


def _parse_entity_phrase(phrase: str, slot_idx: int) -> EntitySpec:
    """Parse a single entity phrase like 'red cat' or 'large blue dog'."""
    # drop articles
    tokens = [
        w for w in phrase.strip().split()
        if w.lower() not in {"a", "an", "the"}
    ]
    color, remaining = _extract_color(tokens)

    # find entity name among remaining tokens
    name = "unknown"
    attributes: list[str] = []
    for tok in remaining:
        candidate = _normalise_name(tok)
        if candidate != "unknown":
            name = candidate
        else:
            attributes.append(tok)

    # fall back to default color if not specified
    if color is None:
        color = list(_DEFAULT_COLORS[slot_idx % len(_DEFAULT_COLORS)])

    return EntitySpec(name=name, attributes=attributes, color=color)


def _split_entities(prompt: str) -> list[str]:
    """Split prompt on conjunctions to isolate entity phrases."""
    # Remove background clause (everything after 'on a', 'in a', 'against')
    bg_pattern = re.compile(r"\s+(on|in|against|with background|background:)\s+", re.I)
    bg_match = bg_pattern.search(prompt)
    scene_part = prompt[:bg_match.start()] if bg_match else prompt

    # Split on 'and', 'with', comma
    parts = re.split(r"\s+and\s+|\s+with\s+|,", scene_part, flags=re.I)
    return [p.strip() for p in parts if p.strip()]


def _extract_interaction(prompt: str, entity_names: list[str]) -> str:
    """Best-effort extraction of the interaction verb phrase."""
    verbs = re.findall(
        r"\b(fight|fight(?:s|ing)?|chasing|chase|play(?:s|ing)?|collide|"
        r"approach(?:es|ing)?|walk(?:s|ing)?|run(?:s|ning)?|stalk(?:s|ing)?)\b",
        prompt, re.I,
    )
    return verbs[0] if verbs else "interact"


def _extract_background(prompt: str) -> Optional[str]:
    """Extract background description if present."""
    bg_pattern = re.compile(
        r"\b(?:on a|in a|against a|background:?)\s+([a-zA-Z ]+?)(?:\.|,|$)", re.I
    )
    m = bg_pattern.search(prompt)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_prompt(prompt: str) -> ScenePrompt:
    """Parse a natural-language scene prompt into a ScenePrompt.

    Supports prompts like:
      - "a red cat and a blue dog fighting"
      - "a cat and a dog"
      - "a green snake and an orange alligator on a sandy ground"

    When colors are absent, defaults are red (entity 0) and blue (entity 1).
    When fewer than 2 entity phrases are found, the second slot defaults to
    EntitySpec(name='unknown', color=blue).

    Parameters
    ----------
    prompt : str
        Free-form scene description.

    Returns
    -------
    ScenePrompt
        Parsed scene with up to 2 entities.
    """
    phrases = _split_entities(prompt)

    entities: list[EntitySpec] = []
    for i, phrase in enumerate(phrases[:2]):
        entities.append(_parse_entity_phrase(phrase, slot_idx=i))

    # Pad to exactly 2 entities
    while len(entities) < 2:
        slot_idx = len(entities)
        entities.append(
            EntitySpec(
                name="unknown",
                attributes=[],
                color=list(_DEFAULT_COLORS[slot_idx]),
            )
        )

    interaction = _extract_interaction(prompt, [e.name for e in entities])
    background = _extract_background(prompt)

    return ScenePrompt(
        entities=entities,
        interaction=interaction,
        background=background,
    )
