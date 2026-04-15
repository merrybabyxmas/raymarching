#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
source /home/dongwoo44/miniconda3/etc/profile.d/conda.sh
conda activate paper_env
cd /home/dongwoo44/papers/raymarching
python -u scripts/run_phase63.py --config config/phase63/stage1.yaml
