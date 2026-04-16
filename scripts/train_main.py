from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> int:
    print("[train_main] exec:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified main entrypoint for the repository mainline (Phase 65)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Run stage 1 or stage 2 training")
    p_train.add_argument("--stage", choices=["1", "2"], required=True)
    p_train.add_argument("--config", type=str, default="")
    p_train.add_argument("--stage1_ckpt", type=str, default="")

    p_check = sub.add_parser("check-data", help="Run Phase 65 dataset sanity check")
    p_check.add_argument("--config", type=str, default="")

    p_viz = sub.add_parser("viz", help="Run Phase 65 visualization")
    p_viz.add_argument("--config", type=str, default="")
    p_viz.add_argument("--ckpt", type=str, default="")
    p_viz.add_argument("--out", type=str, default="")
    p_viz.add_argument("--n_samples", type=int, default=8)

    p_pipeline = sub.add_parser("pipeline", help="Run the full Phase 65 pipeline shell script")

    p_test = sub.add_parser("test", help="Run Phase 65 smoke tests")

    args, unknown = parser.parse_known_args()
    root = Path(__file__).resolve().parent.parent

    if args.command == "train":
        if args.stage == "1":
            config = args.config or str(root / "config/phase65_min3d/stage1.yaml")
            cmd = [sys.executable, str(root / "scripts/train_phase65_stage1.py"), "--config", config] + unknown
            return _run(cmd)
        config = args.config or str(root / "config/phase65_min3d/stage2.yaml")
        cmd = [sys.executable, str(root / "scripts/train_phase65_stage2.py"), "--config", config]
        if args.stage1_ckpt:
            cmd += ["--stage1_ckpt", args.stage1_ckpt]
        cmd += unknown
        return _run(cmd)

    if args.command == "check-data":
        config = args.config or str(root / "config/phase65_min3d/main.yaml")
        return _run([sys.executable, str(root / "scripts/check_phase65_dataset.py"), "--config", config] + unknown)

    if args.command == "viz":
        config = args.config or str(root / "config/phase65_min3d/stage1.yaml")
        ckpt = args.ckpt or str(root / "outputs/phase65/stage1/best_scene_module.pt")
        out = args.out or str(root / "outputs/phase65/viz")
        return _run([
            sys.executable,
            str(root / "scripts/viz_phase65.py"),
            "--config", config,
            "--ckpt", ckpt,
            "--out", out,
            "--n_samples", str(args.n_samples),
        ] + unknown)

    if args.command == "pipeline":
        return _run(["bash", str(root / "scripts/run_phase65_pipeline.sh")] + unknown)

    if args.command == "test":
        return _run([sys.executable, "-m", "pytest", str(root / "tests/test_phase65_smoke.py")] + unknown)

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
