#!/usr/bin/env bash
# Phase 64 Scaled — Full Pipeline
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase64_scaled_pipeline.sh
CONDA_ENV=paper_env
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="logs/phase64_scaled"
mkdir -p "$LOG_DIR"

S1_CKPT="checkpoints/p64s_stage1/best_scene_prior.pt"
S2_CKPT="checkpoints/p64s_stage2/best_decoder.pt"
S3_CKPT="checkpoints/p64s_stage3_animatediff/stage3_epoch0060.pt"
S4_CKPT="checkpoints/p64s_stage4_sdxl/sdxl_adapter.pt"

run() {
    local label="$1"; local log="$LOG_DIR/${label}.log"; shift
    echo ""
    echo "============================================================"
    echo "  [$label] START  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  CMD: $*"
    echo "============================================================"
    conda run -n "$CONDA_ENV" "$@" 2>&1 | tee "$log"
    local ec=${PIPESTATUS[0]}
    [ $ec -ne 0 ] && echo "  [$label] WARNING exit=$ec" || echo "  [$label] DONE  $(date '+%Y-%m-%d %H:%M:%S')"
    return $ec
}

echo "=== Phase 64 SCALED Pipeline  GPU=${CUDA_VISIBLE_DEVICES:-all}  START $(date) ==="

# ── Stage 0+1 ────────────────────────────────────────────────────────────
run "stage0_1" python scripts/train_phase64_scene.py \
    --config config/phase64_scaled/stage1.yaml
[ ! -f "$S1_CKPT" ] && echo "[ABORT] $S1_CKPT not found" && exit 1
echo "Stage 1 OK: $S1_CKPT"

# ── Stage 2 ──────────────────────────────────────────────────────────────
run "stage2" python scripts/train_phase64_decoder.py \
    --config config/phase64_scaled/stage2.yaml \
    --stage1_ckpt "$S1_CKPT"
[ ! -f "$S2_CKPT" ] && echo "[ABORT] $S2_CKPT not found" && exit 1
echo "Stage 2 OK: $S2_CKPT"

# ── Stage 3 ──────────────────────────────────────────────────────────────
run "stage3" python scripts/train_phase64_backbone.py \
    --config config/phase64_scaled/stage3.yaml \
    --stage 3 \
    --stage1_ckpt "$S1_CKPT" \
    --stage2_ckpt "$S2_CKPT"

# ── Stage 4 ──────────────────────────────────────────────────────────────
run "stage4" python scripts/train_phase64_backbone.py \
    --config config/phase64_scaled/stage4.yaml \
    --stage 4 \
    --stage1_ckpt "$S1_CKPT"

# ── Viz ──────────────────────────────────────────────────────────────────
run "viz" python scripts/viz_phase64.py \
    --config config/phase64_scaled/stage1.yaml \
    --stage1_ckpt "$S1_CKPT" \
    --out outputs/phase64_scaled/viz \
    --n_samples 35

echo ""
echo "=== Phase 64 SCALED Pipeline COMPLETE  $(date) ==="
echo "=== Logs   : $LOG_DIR ==="
echo "=== Results: outputs/phase64_scaled/viz/ ==="
