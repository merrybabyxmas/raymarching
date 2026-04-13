"""
Phase 62 main entry.

This script delegates to the OOP/config-driven Phase62 v2 path so the
repo's default phase62 entry no longer runs the deprecated monolithic
trainer that still carried older design assumptions.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_phase62_v2 import main


if __name__ == "__main__":
    main()
