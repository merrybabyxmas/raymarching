from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified main entrypoint for the repository mainline (Phase 65)")
    parser.add_argument("--stage", choices=["1", "2"], required=True, help="Training stage to run")
    parser.add_argument("--config", type=str, default="", help="Optional explicit config path")
    parser.add_argument("--stage1_ckpt", type=str, default="", help="Required for stage 2 if not using default path")
    args, unknown = parser.parse_known_args()

    root = Path(__file__).resolve().parent.parent
    if args.stage == "1":
        config = args.config or str(root / "config/phase65_min3d/stage1.yaml")
        cmd = [sys.executable, str(root / "scripts/train_phase65_stage1.py"), "--config", config] + unknown
    else:
        config = args.config or str(root / "config/phase65_min3d/stage2.yaml")
        cmd = [sys.executable, str(root / "scripts/train_phase65_stage2.py"), "--config", config]
        if args.stage1_ckpt:
            cmd += ["--stage1_ckpt", args.stage1_ckpt]
        cmd += unknown

    print("[train_main] exec:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
