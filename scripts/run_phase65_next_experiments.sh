#!/bin/bash
# Master script for Phase65 next-stage experiments
# Runs separation loss experiments and evaluations

set -e

echo "==============================================="
echo "Phase65 Next-Stage Experiments"
echo "==============================================="
echo ""

# Step 1: Single separation experiment (exp_sep_only)
echo "[1/4] Running exp_sep_only (lambda_sep=0.5, 50 epochs)"
echo "----------------------------------------------"
PYTHONPATH=.:$PYTHONPATH python scripts/train_ablation.py \
    --config config/phase65_min3d/exp_sep_only.yaml \
    --output_dir outputs/phase65/exp_sep_only \
    2>&1 | tee outputs/phase65/exp_sep_only_train.log

echo ""
echo "[1/4] Complete: exp_sep_only"
echo ""

# Step 2: Multi-view consistency test on exp_sep_only
echo "[2/4] Multi-view consistency test (exp_sep_only)"
echo "----------------------------------------------"
PYTHONPATH=.:$PYTHONPATH python scripts/eval_multiview_consistency.py \
    --checkpoint outputs/phase65/exp_sep_only/best_scene_module.pt \
    --config config/phase65_min3d/exp_sep_only.yaml \
    --output_json outputs/phase65/exp_sep_only_multiview.json \
    --num_samples 50 \
    2>&1 | tee outputs/phase65/exp_sep_only_multiview.log

echo ""
echo "[2/4] Complete: Multi-view consistency"
echo ""

# Step 3: Cat/Dog pilot on exp_sep_only
echo "[3/4] Cat/Dog pilot (exp_sep_only, 30 samples)"
echo "----------------------------------------------"
PYTHONPATH=.:$PYTHONPATH python scripts/eval_catdog_pilot.py \
    --checkpoint outputs/phase65/exp_sep_only/best_scene_module.pt \
    --config config/phase65_min3d/exp_sep_only.yaml \
    --output_json outputs/phase65/exp_sep_only_catdog.json \
    --num_samples 30 \
    2>&1 | tee outputs/phase65/exp_sep_only_catdog.log

echo ""
echo "[3/4] Complete: Cat/Dog pilot"
echo ""

# Step 4: Separation loss sweep (optional - takes longer)
read -p "[4/4] Run separation loss sweep? (0.0, 0.1, 0.3, 0.5, 1.0) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "[4/4] Running separation loss sweep"
    echo "----------------------------------------------"
    bash scripts/run_separation_sweep.sh
    echo ""
    echo "[4/4] Complete: Separation sweep"
else
    echo "[4/4] Skipped: Separation sweep"
fi

echo ""
echo "==============================================="
echo "All Phase65 Next-Stage Experiments Complete!"
echo "==============================================="
echo ""
echo "Results:"
echo "  - exp_sep_only: outputs/phase65/exp_sep_only/"
echo "  - Multi-view: outputs/phase65/exp_sep_only_multiview.json"
echo "  - Cat/Dog: outputs/phase65/exp_sep_only_catdog.json"
echo "  - Sweep (if run): outputs/phase65/exp_sep_*/"
echo ""
