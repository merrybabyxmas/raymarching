"""Analyze separation loss experiment results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results():
    """Load all separation experiment results."""
    results = {}

    # A3 baseline (no separation)
    a3_hist = Path('outputs/phase65/ablation_A3/history.json')
    if a3_hist.exists():
        h = json.loads(a3_hist.read_text())
        results['A3 (no sep)'] = h

    # exp_sep_only
    sep_hist = Path('outputs/phase65/exp_sep_only/history.json')
    if sep_hist.exists():
        h = json.loads(sep_hist.read_text())
        results['exp_sep_only'] = h

    # Sweep results
    sweep_values = [0.0, 0.1, 0.3, 0.5, 1.0]
    for val in sweep_values:
        name = f'exp_sep_{val}'.replace('.', '_')
        hist_path = Path(f'outputs/phase65/{name}/history.json')
        if hist_path.exists():
            h = json.loads(hist_path.read_text())
            results[f'λ_sep={val}'] = h

    return results


def load_multiview_results():
    """Load multi-view consistency results."""
    results = {}

    # A2 baseline
    a2_mv = Path('outputs/phase65/multiview_consistency_A2.json')
    if a2_mv.exists():
        results['A2 (no sep)'] = json.loads(a2_mv.read_text())

    # exp_sep_only
    sep_mv = Path('outputs/phase65/exp_sep_only_multiview.json')
    if sep_mv.exists():
        results['exp_sep_only'] = json.loads(sep_mv.read_text())

    return results


def load_catdog_results():
    """Load cat/dog pilot results."""
    results = {}

    # A2 baseline
    a2_cd = Path('outputs/phase65/catdog_pilot_A2.json')
    if a2_cd.exists():
        results['A2 (no sep)'] = json.loads(a2_cd.read_text())

    # exp_sep_only
    sep_cd = Path('outputs/phase65/exp_sep_only_catdog.json')
    if sep_cd.exists():
        results['exp_sep_only'] = json.loads(sep_cd.read_text())

    return results


def plot_training_comparison(results: dict, output_path: str = 'outputs/phase65/separation_analysis.png'):
    """Create comprehensive analysis plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Separation Loss Analysis', fontsize=18, fontweight='bold')

    # Plot 1: Entity Balance
    ax = axes[0, 0]
    ax.set_title('Entity Balance Over Training', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Entity Balance')
    ax.grid(True, alpha=0.3)

    for name, hist in results.items():
        epochs = [e['epoch'] for e in hist]
        eb = [e.get('val_entity_balance', 0) for e in hist]
        ax.plot(epochs, eb, label=name, linewidth=2)

    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    # Plot 2: Cross-slot Cosine (from l_sep if available)
    ax = axes[0, 1]
    ax.set_title('Separation Loss (l_sep)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Separation Loss')
    ax.grid(True, alpha=0.3)

    for name, hist in results.items():
        if any('train_l_sep' in e for e in hist):
            epochs = [e['epoch'] for e in hist]
            l_sep = [e.get('train_l_sep', 0) for e in hist]
            ax.plot(epochs, l_sep, label=name, linewidth=2)

    ax.legend()

    # Plot 3: Amodal IoU
    ax = axes[1, 0]
    ax.set_title('Amodal IoU Min', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Amodal IoU Min')
    ax.grid(True, alpha=0.3)

    for name, hist in results.items():
        epochs = [e['epoch'] for e in hist]
        iou = [e.get('val_amodal_iou_min', 0) for e in hist]
        ax.plot(epochs, iou, label=name, linewidth=2)

    ax.legend()

    # Plot 4: Final metrics comparison table
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title('Final Metrics Comparison', fontsize=14, fontweight='bold', pad=15)

    table_data = [['Experiment', 'Entity\nBalance', 'Amodal\nIoU', 'Survival\nMin']]

    for name, hist in results.items():
        if hist:
            last = hist[-1]
            eb = last.get('val_entity_balance', 0)
            iou = last.get('val_amodal_iou_min', 0)
            surv = last.get('val_visible_survival_min', 0)
            table_data.append([
                name,
                f'{eb:.4f}',
                f'{iou:.4f}',
                f'{surv:.4f}'
            ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.35, 0.22, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved analysis to {output_path}")
    plt.close()


def print_summary():
    """Print text summary of all results."""
    print("\n" + "="*80)
    print("SEPARATION LOSS EXPERIMENT SUMMARY")
    print("="*80 + "\n")

    # Training results
    results = load_results()
    if results:
        print("Training Results:")
        print("-" * 80)
        for name, hist in results.items():
            if hist:
                last = hist[-1]
                print(f"{name:30} | "
                      f"Entity Balance: {last.get('val_entity_balance', 0):.4f} | "
                      f"Amodal IoU: {last.get('val_amodal_iou_min', 0):.4f} | "
                      f"Survival: {last.get('val_visible_survival_min', 0):.4f}")
        print()

    # Multi-view results
    mv_results = load_multiview_results()
    if mv_results:
        print("Multi-view Consistency:")
        print("-" * 80)
        for name, data in mv_results.items():
            print(f"{name:30} | "
                  f"Same-slot cosine: {data.get('same_slot_cosine_mean', 0):.4f} | "
                  f"Cross-slot cosine: {data.get('cross_slot_cosine_mean', 0):.4f} | "
                  f"Separation ratio: {data.get('slot_separation_mean', 0):.4f}")
        print()

    # Cat/Dog results
    cd_results = load_catdog_results()
    if cd_results:
        print("Cat/Dog Pilot:")
        print("-" * 80)
        for name, data in cd_results.items():
            summary = data.get('summary', {})
            print(f"{name:30} | "
                  f"Success rate: {summary.get('success_rate', 0):.2%} | "
                  f"Visible fail: {summary.get('visible_fail_rate', 0):.2%} | "
                  f"Both fail: {summary.get('both_fail_rate', 0):.2%}")
        print()

    print("="*80 + "\n")


if __name__ == '__main__':
    results = load_results()
    if results:
        plot_training_comparison(results)
    print_summary()
