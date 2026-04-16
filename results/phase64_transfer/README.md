# Phase 64 — Stage 4: SDXL Transfer Eval Results

**Date**: 2026-04-16  
**Stage**: Stage 4 — Backbone-agnostic transfer (scene prior frozen, new SDXL adapter)  
**Backbone**: stabilityai/sdxl-turbo  
**n_samples**: 35 (val split)

## Key Results

| Metric | Baseline (no guide) | Guided | Delta |
|--------|-------------------|--------|-------|
| vis_iou_min | 0.023 | 0.374 | **+0.350** |
| amodal_iou_min | 0.025 | 0.344 | **+0.320** |

## Interpretation

The same Stage 1 scene prior (trained without any backbone) transfers to SDXL-Turbo
with a thin adapter, achieving vis_iou_min 0.023 → 0.374 (+0.350). This confirms
backbone-agnostic portability: the scene prior's entity fields generalize across
backbone architectures without retraining.

## Stage Summary

| Stage | Metric | Value |
|-------|--------|-------|
| Stage 0 | Valid samples | 171/180 (95%) |
| Stage 1 | val vis_iou_min | 0.3735 |
| Stage 2 | Decoder PSNR | 29.38 dB (gate: 18 dB) |
| Stage 3 | AnimateDiff diff_loss | 0.079 |
| Stage 4 | SDXL vis_iou_min (guided) | 0.374 |
