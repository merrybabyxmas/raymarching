#!/usr/bin/env bash
# Phase 64 — Full Pipeline: Stage 0+1 → 2 → 3 → 4 → eval
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase64_full_pipeline.sh
# Note: set -e removed so individual stage errors surface but pipeline continues

CONDA_ENV=paper_env
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

LOG_DIR="logs/phase64"
mkdir -p "$LOG_DIR"

S1_CKPT="checkpoints/phase64/p64_stage1/best_scene_prior.pt"
S2_CKPT="checkpoints/phase64/p64_stage2/best_decoder.pt"
S3_CKPT="checkpoints/phase64/p64_stage3_animatediff/stage3_epoch0029.pt"
S4_CKPT="checkpoints/phase64/p64_stage4_sdxl/sdxl_adapter.pt"

run() {
    local label="$1"
    local log="$LOG_DIR/${label}.log"
    shift
    echo ""
    echo "============================================================"
    echo "  [$label] START  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  CMD: $*"
    echo "============================================================"
    conda run -n "$CONDA_ENV" "$@" 2>&1 | tee "$log"
    local ec=${PIPESTATUS[0]}
    if [ $ec -ne 0 ]; then
        echo "  [$label] WARNING: exit code=$ec  $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo "  [$label] DONE  $(date '+%Y-%m-%d %H:%M:%S')"
    fi
    return $ec
}

echo "=== Phase 64 Full Pipeline  GPU=${CUDA_VISIBLE_DEVICES:-all}  START $(date) ==="

# ── Stage 0 + 1: Dataset validation + scene prior ─────────────────────────
run "stage0_1" python scripts/train_phase64_scene.py \
    --config config/phase64/stage1.yaml
if [ ! -f "$S1_CKPT" ]; then
    echo "[ABORT] Stage 1 checkpoint not found: $S1_CKPT"
    exit 1
fi
echo "Stage 1 checkpoint confirmed: $S1_CKPT"

# ── Stage 2: Structured decoder ─────────────────────────────────────────
run "stage2" python scripts/train_phase64_decoder.py \
    --config config/phase64/stage2.yaml \
    --stage1_ckpt "$S1_CKPT"
if [ ! -f "$S2_CKPT" ]; then
    echo "[ABORT] Stage 2 checkpoint not found: $S2_CKPT"
    exit 1
fi
echo "Stage 2 checkpoint confirmed: $S2_CKPT"

# ── Stage 3: AnimateDiff adapter ─────────────────────────────────────────
run "stage3" python scripts/train_phase64_backbone.py \
    --config config/phase64/stage3.yaml \
    --stage 3 \
    --stage1_ckpt "$S1_CKPT" \
    --stage2_ckpt "$S2_CKPT"

# ── Stage 4: SDXL transfer adapter ───────────────────────────────────────
run "stage4" python scripts/train_phase64_backbone.py \
    --config config/phase64/stage4.yaml \
    --stage 4 \
    --stage1_ckpt "$S1_CKPT"

# ── Transfer eval ─────────────────────────────────────────────────────────
run "eval_transfer" python scripts/eval_phase64_transfer.py \
    --config config/phase64/stage3.yaml \
    --stage1_ckpt "$S1_CKPT" \
    --stage3_ckpt "$S3_CKPT" \
    --stage4_ckpt "$S4_CKPT" \
    --out outputs/phase64/eval_transfer_final

echo ""
echo "=== Phase 64 Full Pipeline COMPLETE  $(date) ==="
echo "=== Logs: $LOG_DIR ==="
echo "=== Results: outputs/phase64/eval_transfer_final ==="
