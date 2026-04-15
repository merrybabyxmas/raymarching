#!/bin/bash
# Auto-launch v39e on GPU 0 when v39a (PID 2974375) finishes.
# Usage: nohup bash scripts/launch_v39e_when_free.sh > logs/launch_v39e.log 2>&1 &

TARGET_PID=2974375  # v39a
GPU=0

echo "[launcher] Waiting for v39a (PID $TARGET_PID) to finish..."
while kill -0 $TARGET_PID 2>/dev/null; do
    sleep 60
done
echo "[launcher] v39a finished. Launching v39e on GPU $GPU..."

CUDA_VISIBLE_DEVICES=$GPU nohup python scripts/run_phase62_ablations.py \
    --config config/phase62/ablations/b1_v39e_p4_lcc_fix_s7.yaml \
    > logs/v39e_p4_lcc_fix_s7.log 2>&1 &

echo "[launcher] v39e launched (PID: $!)"
