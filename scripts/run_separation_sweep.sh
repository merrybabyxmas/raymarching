#!/bin/bash
# Run separation loss sweep experiments

set -e

SWEEP_VALUES=(0.0 0.1 0.3 0.5 1.0)

echo "=== Starting Separation Loss Sweep ==="
echo "Testing lambda_sep values: ${SWEEP_VALUES[@]}"
echo ""

for lambda_sep in "${SWEEP_VALUES[@]}"; do
    exp_name="exp_sep_${lambda_sep//./_}"
    echo "[$(date)] Running $exp_name with lambda_sep=$lambda_sep"

    # Create output directory
    mkdir -p "outputs/phase65/$exp_name"

    # Run training with modified lambda_sep
    PYTHONPATH=.:$PYTHONPATH python scripts/train_ablation.py \
        --config config/phase65_min3d/exp_sep_sweep.yaml \
        --output_dir "outputs/phase65/$exp_name" \
        --lambda_sep "$lambda_sep" \
        2>&1 | tee "outputs/phase65/$exp_name/train.log"

    echo "[$(date)] Completed $exp_name"
    echo ""
done

echo "=== Separation Sweep Complete ==="
echo "Results in: outputs/phase65/exp_sep_*"
