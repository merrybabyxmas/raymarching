#!/bin/bash
set -e

cd /home/dongwoo38/papers/raymarching
source .venv/bin/activate
export PYTHONPATH=/home/dongwoo38/papers/raymarching

echo "=== Running Ablation Study ==="

echo "[1/4] Training A0: No camera cond, No hard data..."
python scripts/train_ablation.py --config config/phase65_min3d/ablation_A0.yaml
echo "A0 done!"

echo "[2/4] Training A1: Camera cond only..."
python scripts/train_ablation.py --config config/phase65_min3d/ablation_A1.yaml
echo "A1 done!"

echo "[3/4] Training A2: Hard data only..."
python scripts/train_ablation.py --config config/phase65_min3d/ablation_A2.yaml
echo "A2 done!"

echo "[4/4] Training A3: Camera cond + Hard data..."
python scripts/train_ablation.py --config config/phase65_min3d/ablation_A3.yaml
echo "A3 done!"

echo "=== All ablation training complete ==="
