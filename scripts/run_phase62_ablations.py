"""
Phase 62 — Ablation Runner
============================

Runs one or more ablation configs sequentially.

Usage:
    # Run single ablation
    python scripts/run_phase62_ablations.py --config config/phase62/ablations/b0_fgid_volume_only.yaml

    # Run priority list (B0, B1, B2, A0, C0)
    python scripts/run_phase62_ablations.py --priority

    # Run all ablations
    python scripts/run_phase62_ablations.py --all
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from models.phase62.system import Phase62System
from models.phase62.backbone_adapter import (
    inject_backbone_extractors,
    BackboneManager,
    DEFAULT_INJECT_KEYS,
)
from training.phase62.ablation_trainer import AblationTrainer
from data.phase62.dataset_adapter import Phase62DatasetAdapter
from scripts.run_animatediff import load_pipeline

PRIORITY_ORDER = [
    "config/phase62/ablations/b0_fgid_volume_only.yaml",
    "config/phase62/ablations/b1_fgid_freeze_bind_front.yaml",
    "config/phase62/ablations/b2_fgid_freeze_bind_fourstream.yaml",
    "config/phase62/ablations/a0_indep_bce_volume_only.yaml",
    "config/phase62/ablations/c0_amodal_only.yaml",
]

ALL_ABLATIONS = PRIORITY_ORDER + [
    "config/phase62/ablations/a1_indep_bce_freeze_bind_front.yaml",
    "config/phase62/ablations/b3_fgid_low_lr_bind_fourstream.yaml",
    "config/phase62/ablations/c1_visible_only.yaml",
    "config/phase62/ablations/d0_center_offset.yaml",
]


def run_ablation(config_path: str, pipe, dataset, device: str):
    """Run one ablation from config."""
    config = load_config(config_path)
    run_name = Path(config_path).stem

    config.save_dir = f"checkpoints/phase62_ablations/{run_name}"
    config.debug_dir = f"outputs/phase62_ablations/{run_name}"

    print(f"\n{'='*60}", flush=True)
    print(f"[Ablation] {run_name}", flush=True)
    print(f"  objective={config.objective}  schedule={config.schedule}  "
          f"guide={config.guide_family}  repr={config.representation}", flush=True)
    print(f"{'='*60}", flush=True)

    # Re-inject backbone extractors (fresh weights per ablation)
    extractors, _ = inject_backbone_extractors(
        pipe,
        adapter_rank=config.model.adapter_rank,
        lora_rank=config.model.lora_rank,
        inject_keys=DEFAULT_INJECT_KEYS,
    )
    for ext in extractors:
        ext.to(device)
    backbone_mgr = BackboneManager(extractors, DEFAULT_INJECT_KEYS, primary_idx=1)

    system = Phase62System(config).to(device)
    system.injection_mgr.register_hooks(pipe.unet)

    trainer = AblationTrainer(
        config=config,
        pipe=pipe,
        system=system,
        backbone_mgr=backbone_mgr,
        dataset=dataset,
        device=device,
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"[Ablation] {run_name} FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {"name": run_name, "status": "failed", "error": str(e)}

    result = {
        "name": run_name,
        "status": "done",
        "best_epoch": trainer.best_epoch,
        "best_val_score": trainer.best_val_score,
        "history": trainer.history,
    }

    system.injection_mgr.remove_hooks()
    del system, trainer, backbone_mgr
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 62 Ablation Runner")
    parser.add_argument("--config", type=str, default=None, help="Single config to run")
    parser.add_argument("--priority", action="store_true", help="Run priority list (B0,B1,B2,A0,C0)")
    parser.add_argument("--all", action="store_true", help="Run all 9 ablations")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("[Ablation] Loading pipeline...", flush=True)
    pipe = load_pipeline(device=device)
    pipe.scheduler.set_timesteps(50, device=device)
    pipe.unet.enable_gradient_checkpointing()

    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    print("[Ablation] Loading dataset...", flush=True)
    dataset = Phase62DatasetAdapter("toy/data_objaverse", n_frames=8)
    print(f"  dataset size: {len(dataset)}", flush=True)

    if args.config:
        configs = [args.config]
    elif args.priority:
        configs = PRIORITY_ORDER
    elif args.all:
        configs = ALL_ABLATIONS
    else:
        configs = PRIORITY_ORDER
        print("[Ablation] No flag specified, running priority list.", flush=True)

    results = []
    for config_path in configs:
        if not Path(config_path).exists():
            print(f"[warn] Config not found: {config_path}", flush=True)
            continue
        r = run_ablation(config_path, pipe, dataset, device)
        results.append(r)

    # Summary
    summary_path = Path("outputs/phase62_ablations/summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}", flush=True)
    print("[Ablation Summary]", flush=True)
    for r in results:
        status = r["status"]
        name = r["name"]
        if status == "done":
            print(f"  {name}: best_epoch={r['best_epoch']} "
                  f"val_score={r['best_val_score']:.4f}", flush=True)
        else:
            print(f"  {name}: FAILED — {r.get('error', 'unknown')}", flush=True)
    print(f"  Summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
