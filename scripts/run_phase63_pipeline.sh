#!/bin/bash
# Phase 63 — Full Training Pipeline
# Runs Stage 2 → Stage 3 → Eval sequentially on GPU 3.
# Stage 1 must already be complete (checkpoints/phase63/p63_stage1/best.pt).
#
# Usage:
#   chmod +x scripts/run_phase63_pipeline.sh
#   ./scripts/run_phase63_pipeline.sh

set -e
export CUDA_VISIBLE_DEVICES=3

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ENV="paper_env"
PY="conda run -n $ENV python"

echo "============================================================"
echo "[Phase63 Pipeline] Starting from Stage 2"
echo "  Stage1 ckpt: checkpoints/phase63/p63_stage1/best.pt"
echo "============================================================"

# ── Stage 2: Guide encoder pretrain ─────────────────────────────────────────
echo ""
echo "[Stage 2] Starting at $(date)"
$PY scripts/run_phase63.py \
    --config config/phase63/stage2.yaml \
    --device cuda
echo "[Stage 2] Done at $(date)"

# ── Stage 3: Full joint training ─────────────────────────────────────────────
echo ""
echo "[Stage 3] Starting at $(date)"
$PY scripts/run_phase63.py \
    --config config/phase63/stage3.yaml \
    --device cuda
echo "[Stage 3] Done at $(date)"

# ── Eval ──────────────────────────────────────────────────────────────────────
echo ""
echo "[Eval] Starting at $(date)"
$PY scripts/eval_phase63.py \
    --config config/phase63/stage1.yaml \
    --out outputs/phase63/eval_final \
    --max_samples 50 \
    --device cuda
echo "[Eval] Done at $(date)"

echo ""
echo "============================================================"
echo "[Phase63 Pipeline] COMPLETE"
echo "  Stage2 ckpt: checkpoints/phase63/p63_stage2/best.pt"
echo "  Stage3 ckpt: checkpoints/phase63/p63_stage3/best.pt"
echo "  Eval output: outputs/phase63/eval_final/"
echo "============================================================"
