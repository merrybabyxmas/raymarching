"""
Config loading utilities for Phase 62.

load_config('config/phase62/base.yaml')
load_config('config/phase62/base.yaml', overrides=['training.epochs=5', 'data.val_frac=0.3'])
load_config('config/phase62/base.yaml', overlay='config/phase62/train_smoke.yaml')
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import yaml


def _coerce_scalar(v):
    """Best-effort coercion for YAML strings like '5e-4'."""
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() == "none":
        return None
    try:
        if any(ch in s.lower() for ch in [".", "e"]):
            return float(s)
        return int(s)
    except ValueError:
        return v


def _coerce_tree(obj):
    """Recursively coerce config scalars."""
    if isinstance(obj, dict):
        return {k: _coerce_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_tree(v) for v in obj]
    return _coerce_scalar(obj)


def _deep_update(base: dict, patch: dict) -> dict:
    """Recursively merge patch into base dict (in-place)."""
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _apply_overrides(cfg: dict, overrides: List[str]) -> dict:
    """Apply dotted key=value overrides.

    Examples:
        training.epochs=5  ->  cfg['training']['epochs'] = 5
        la_diff=0.5        ->  cfg['la_diff'] = 0.5
    """
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov}")
        key, val_str = ov.split("=", 1)
        keys = key.strip().split(".")

        # Parse value: try int, then float, then bool, else str
        val_str = val_str.strip()
        if val_str.lower() == "true":
            val = True
        elif val_str.lower() == "false":
            val = False
        elif val_str.lower() == "none":
            val = None
        else:
            try:
                val = int(val_str)
            except ValueError:
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str

        # Walk nested dict
        d = cfg
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = val

    return cfg


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert dict to SimpleNamespace for attribute access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def load_config(
    base_path: str,
    overrides: Optional[List[str]] = None,
    overlay: Optional[str] = None,
) -> SimpleNamespace:
    """
    Load YAML config with optional overlay and CLI overrides.

    Args:
        base_path: Path to base YAML config.
        overrides: List of 'dotted.key=value' strings.
        overlay:   Path to overlay YAML (merged on top of base).

    Returns:
        SimpleNamespace with nested attribute access.
    """
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Config not found: {base_path}")

    with open(base_path) as f:
        cfg = _coerce_tree(yaml.safe_load(f) or {})

    if overlay is not None:
        overlay_path = Path(overlay)
        if overlay_path.exists():
            with open(overlay_path) as f:
                patch = _coerce_tree(yaml.safe_load(f) or {})
            _deep_update(cfg, patch)

    if overrides:
        _apply_overrides(cfg, overrides)

    return _dict_to_namespace(cfg)
