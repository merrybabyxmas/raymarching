#!/usr/bin/env bash
# Phase 65 Minimal 3D — mainline pipeline
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase65_pipeline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="logs/phase65"
mkdir -p "$LOG_DIR"

MAIN_CFG="config/phase65_min3d/main.yaml"
S1_CFG="config/phase65_min3d/stage1.yaml"
S2_CFG="config/phase65_min3d/stage2.yaml"
S1_CKPT="outputs/phase65/stage1/best_scene_module.pt"
S2_CKPT="outputs/phase65/stage2/best_stage2.pt"
VIZ_OUT="outputs/phase65/viz"

run() {
    local label="$1"; local log="$LOG_DIR/${label}.log"; shift
    echo ""
    echo "============================================================"
    echo "  [$label] START  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  CMD: $*"
    echo "============================================================"
    "$@" 2>&1 | tee "$log"
    local ec=${PIPESTATUS[0]}
    [ $ec -ne 0 ] && echo "  [$label] WARNING exit=$ec" || echo "  [$label] DONE  $(date '+%Y-%m-%d %H:%M:%S')"
    return $ec
}

echo "=== Phase 65 Minimal 3D Pipeline START $(date) ==="
run "check_data" python scripts/check_phase65_dataset.py --config "$MAIN_CFG"
run "stage1" python scripts/train_phase65_stage1.py --config "$S1_CFG"
[ ! -f "$S1_CKPT" ] && echo "[ABORT] $S1_CKPT not found" && exit 1
run "viz_stage1" python scripts/viz_phase65.py --config "$S1_CFG" --ckpt "$S1_CKPT" --out "$VIZ_OUT"
run "stage2" python scripts/train_phase65_stage2.py --config "$S2_CFG" --stage1_ckpt "$S1_CKPT"
[ ! -f "$S2_CKPT" ] && echo "[WARN] $S2_CKPT not found yet" || echo "Stage 2 OK: $S2_CKPT"
echo "=== Phase 65 Minimal 3D Pipeline COMPLETE $(date) ==="
