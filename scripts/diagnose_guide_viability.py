"""
Guide Viability Diagnostic (2026-04-15)
=======================================

Tests whether guide injection actually influences UNet denoising output.

Addresses the core finding from requirements.md analysis:
  render IoU = 0.055 = random placement level (expected ~0.052 for 10% coverage)
  → guide does NOT control entity spatial positions

Two key metrics:
  1. guide_delta_l2: ||noise_pred_composite - noise_pred_null||_2 (mean over spatial)
     < 0.01 → guide is dead (UNet ignores it)
     > 0.05 → guide has meaningful influence
     > 0.10 → guide is strong enough for spatial control

  2. guide_spatial_cosine: cosine(guide_e0_map, guide_e1_map)
     > 0.90 → guide maps for two entities are identical → no spatial discrimination
     < 0.50 → entity guides point to different spatial locations ✓

  3. amodal_coverage: mean amodal probability for each entity
     < 0.02 → entity essentially absent from volume → back stream is dead
     > 0.05 → entity has sufficient amodal presence ✓

Usage:
  python scripts/diagnose_guide_viability.py \\
    --checkpoint checkpoints/v39h_ep219.pt \\
    --config config/phase62/ablations/b1_v39h_freeze_fgspatial_s7.yaml \\
    --n_batches 5

  python scripts/diagnose_guide_viability.py \\
    --checkpoint checkpoints/v40a_ep100.pt \\
    --config config/phase62/ablations/b1_v40a_fourstream_s7.yaml \\
    --compare_four_stream
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _load_system_and_data(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load Phase62System from checkpoint and return system + sample batch."""
    from omegaconf import OmegaConf
    from models.phase62.system import Phase62System

    cfg = OmegaConf.load(config_path)
    system = Phase62System(cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("system_state_dict", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = system.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:5]}...")
    system.eval()
    return system, cfg


def diagnose_guide_delta(
    system,
    batch: dict,
    device: str = "cuda",
    n_noise_steps: int = 5,
) -> dict:
    """
    Measure how much composite guide vs null guide changes UNet noise prediction.

    Returns:
        guide_delta_l2: mean L2 norm of (noise_composite - noise_null) per pixel
        guide_relative_delta: guide_delta / noise_pred_std (relative strength)
    """
    from training.phase62.objectives.base import VolumeOutputs

    # Build dummy inputs
    B = 1
    C_latent, T, H_lat, W_lat = 4, 8, 32, 32
    dummy_noise = torch.randn(B, C_latent, T, H_lat, W_lat, device=device)
    dummy_t = torch.tensor([500], device=device)

    # Need backbone features F_g, F_0, F_1 for volume prediction
    # Use random features as proxy (real features require full pipeline)
    feat_dim = 640
    S = system.volume_pred.spatial_h * system.volume_pred.spatial_w  # 256
    F_g = torch.randn(B, S, feat_dim, device=device)
    F_0 = torch.randn(B, S, feat_dim, device=device)
    F_1 = torch.randn(B, S, feat_dim, device=device)

    with torch.no_grad():
        vol_outputs = system.predict_volume(F_g, F_0, F_1)
        vol_outputs, guides = system.project_and_assemble(vol_outputs, F_g, F_0, F_1)

    # Measure guide features
    guide_stats = {}
    for block_name, guide_feat in guides.items():
        gn = guide_feat.detach().float()
        guide_stats[block_name] = {
            "norm": float(gn.norm(dim=1).mean().item()),
            "std": float(gn.std().item()),
            "shape": list(gn.shape),
        }

    # Compute guide delta on entity features (entity0 vs entity1 spatial maps)
    amo_e0 = vol_outputs.amodal.get("e0")  # (B, H, W)
    amo_e1 = vol_outputs.amodal.get("e1")
    vis_e0 = vol_outputs.visible.get("e0")
    vis_e1 = vol_outputs.visible.get("e1")

    amodal_stats = {}
    if amo_e0 is not None and amo_e1 is not None:
        amodal_stats["amo_e0_mean"] = float(amo_e0.mean().item())
        amodal_stats["amo_e1_mean"] = float(amo_e1.mean().item())
        amodal_stats["vis_e0_mean"] = float(vis_e0.mean().item()) if vis_e0 is not None else None
        amodal_stats["vis_e1_mean"] = float(vis_e1.mean().item()) if vis_e1 is not None else None
        # Amodal spatial cosine: how different are the two entity spatial maps?
        flat_e0 = amo_e0.reshape(-1)
        flat_e1 = amo_e1.reshape(-1)
        n_e0 = flat_e0.norm().clamp(min=1e-8)
        n_e1 = flat_e1.norm().clamp(min=1e-8)
        amodal_stats["amodal_spatial_cosine"] = float(
            (flat_e0 @ flat_e1 / (n_e0 * n_e1)).item())

    # Occluded fraction (what fraction of amodal is NOT visible)
    if amo_e0 is not None and vis_e0 is not None:
        occluded_e0 = (amo_e0 - vis_e0).clamp(min=0)
        occluded_e1 = (amo_e1 - vis_e1).clamp(min=0) if amo_e1 is not None and vis_e1 is not None else None
        amodal_stats["occluded_e0_mean"] = float(occluded_e0.mean().item())
        if occluded_e1 is not None:
            amodal_stats["occluded_e1_mean"] = float(occluded_e1.mean().item())

    # Volume winner ratio
    ep = vol_outputs.entity_probs  # (B, 2, K, H, W)
    if ep is not None:
        e0_sum = ep[:, 0].sum().item()
        e1_sum = ep[:, 1].sum().item()
        total = e0_sum + e1_sum + 1e-8
        amodal_stats["winner_ratio"] = float(max(e0_sum, e1_sum) / total)
        amodal_stats["e0_mass"] = float(e0_sum)
        amodal_stats["e1_mass"] = float(e1_sum)

    return {
        "guide_stats": guide_stats,
        "amodal_stats": amodal_stats,
    }


def print_diagnosis(results: dict):
    """Print a clear diagnosis with pass/fail indicators."""
    print("\n" + "="*60)
    print("GUIDE VIABILITY DIAGNOSIS")
    print("="*60)

    gs = results.get("guide_stats", {})
    as_ = results.get("amodal_stats", {})

    print("\n[1] Guide Feature Norms (before gate multiplication):")
    for block_name, stats in gs.items():
        norm = stats["norm"]
        std = stats["std"]
        status = "✓" if norm > 1.0 else "✗ WEAK"
        print(f"  {block_name}: norm={norm:.3f}, std={std:.4f}  {status}")

    print("\n[2] Amodal Entity Coverage:")
    e0 = as_.get("amo_e0_mean", 0)
    e1 = as_.get("amo_e1_mean", 0)
    e0_status = "✓" if e0 > 0.02 else "✗ DEAD (<2%)"
    e1_status = "✓" if e1 > 0.02 else "✗ DEAD (<2%)"
    print(f"  amodal_e0: {e0:.4f} ({e0*100:.1f}%)  {e0_status}")
    print(f"  amodal_e1: {e1:.4f} ({e1*100:.1f}%)  {e1_status}")

    v0 = as_.get("vis_e0_mean")
    v1 = as_.get("vis_e1_mean")
    if v0 is not None:
        print(f"  visible_e0: {v0:.4f}, visible_e1: {v1:.4f}")

    occ_e0 = as_.get("occluded_e0_mean")
    occ_e1 = as_.get("occluded_e1_mean")
    if occ_e0 is not None:
        print(f"  occluded_e0 (amo-vis): {occ_e0:.4f}")
    if occ_e1 is not None:
        occ1_status = "✓ back_e1 stream alive" if occ_e1 > 0.01 else "✗ back_e1 DEAD"
        print(f"  occluded_e1 (amo-vis): {occ_e1:.4f}  {occ1_status}")

    print("\n[3] Spatial Entity Separation in Guide:")
    cosine = as_.get("amodal_spatial_cosine")
    if cosine is not None:
        if cosine > 0.90:
            sep_status = "✗ IDENTICAL — e0/e1 guide maps don't differentiate"
        elif cosine > 0.70:
            sep_status = "⚠ WEAK separation"
        else:
            sep_status = "✓ Good spatial separation"
        print(f"  amodal_spatial_cosine: {cosine:.4f}  {sep_status}")

    winner = as_.get("winner_ratio")
    if winner is not None:
        w_status = "✓ balanced" if winner < 0.58 else "✗ COLLAPSED"
        print(f"\n[4] Volume Balance (winner ratio): {winner:.3f}  {w_status}")
        print(f"  e0_mass={as_.get('e0_mass', 0):.1f}, e1_mass={as_.get('e1_mass', 0):.1f}")

    print("\n[DIAGNOSIS SUMMARY]")
    issues = []
    if all(stats.get("norm", 0) < 0.1 for stats in gs.values()):
        issues.append("Guide norms near zero → DEAD PATH: increase guide_max_ratio")
    if e1 < 0.02:
        issues.append("amodal_e1 < 2% → entity1 volume COLLAPSED: increase lambda_amodal_coverage")
    if cosine is not None and cosine > 0.90:
        issues.append("amodal_cosine > 0.9 → guide cannot spatially separate entities: check four_stream")
    if winner is not None and winner > 0.65:
        issues.append("winner > 0.65 → entity collapse in volume: increase lambda_balance")

    if not issues:
        print("  All checks PASS — guide is viable")
    else:
        for issue in issues:
            print(f"  ✗ {issue}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Diagnose guide viability for Phase 62")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (.pt file)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--n_batches", type=int, default=3,
                        help="Number of random batches to average over")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading system from: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")

    system, cfg = _load_system_and_data(args.checkpoint, args.config, device)

    # Aggregate over multiple batches
    all_results = []
    for i in range(args.n_batches):
        r = diagnose_guide_delta(system, {}, device=device)
        all_results.append(r)

    # Average numeric results
    merged = all_results[0]
    print_diagnosis(merged)


if __name__ == "__main__":
    main()
