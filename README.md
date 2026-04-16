# raymarching

The repository mainline is now **Phase 65 Minimal 3D (`phase65_min3d`)**.

This refactor replaces older experimental directions with a cleaner architecture focused on reducing **chimera** in contact-heavy multi-entity generation.

## Mainline Goal

Generate two-entity scenes such as:

- cat + dog rolling together,
- contact and occlusion,
- reappearance after hiding,
- stable identity across time,
- no hybrid single-entity collapse.

## Mainline Principle

The repository now follows this rule:

> **Scene structure must be represented explicitly before the final backbone generates appearance.**

Instead of relying on class-volume competition or winner-take-all projection, the new mainline uses an **entity-separated layered 2.5D scene representation**.

---

## Mainline Architecture

```text
Entity names / prompt / optional previous state
        │
        ▼
SceneModule
  ├─ EntitySlotEncoder
  ├─ TemporalSlotMemory
  ├─ MotionRollout
  ├─ LayeredEntityDecoder (e0)
  ├─ LayeredEntityDecoder (e1)
  └─ OcclusionComposer
        │
        ▼
SceneState
  ├─ visible_e0 / visible_e1
  ├─ hidden_e0  / hidden_e1
  ├─ amodal_e0  / amodal_e1
  ├─ depth_e0   / depth_e1
  ├─ feat_e0    / feat_e1
  └─ mem_e0     / mem_e1
        │
        ▼
Backbone-specific Adapter
        │
        ▼
Backbone / Decoder / Refiner
```

---

## Current Mainline Directory

```text
phase65_min3d/
├─ scene_outputs.py
├─ slot_encoder.py
├─ temporal_slots.py
├─ motion_rollout.py
├─ layered_decoder.py
├─ occlusion_composer.py
├─ scene_module.py
├─ losses.py
├─ evaluator.py
├─ trainer_stage1.py
├─ trainer_stage2.py
├─ data.py
├─ adapters/
└─ backbones/
```

---

## Unified Main Entrypoint

The preferred main entrypoint is now `scripts/train_main.py`.

### Stage 1
```bash
python scripts/train_main.py train --stage 1
```

### Stage 2
```bash
python scripts/train_main.py train --stage 2 \
  --stage1_ckpt outputs/phase65/stage1/best_scene_module.pt
```

### Dataset sanity check
```bash
python scripts/train_main.py check-data
```

### Visualization
```bash
python scripts/train_main.py viz
```

### Smoke tests
```bash
python scripts/train_main.py test
```

### Full pipeline
```bash
python scripts/train_main.py pipeline
```

---

## Direct Entrypoints

### Stage 1
```bash
python scripts/train_phase65_stage1.py --config config/phase65_min3d/stage1.yaml
```

### Stage 2
```bash
python scripts/train_phase65_stage2.py \
  --config config/phase65_min3d/stage2.yaml \
  --stage1_ckpt outputs/phase65/stage1/best_scene_module.pt
```

### Full shell pipeline
```bash
bash scripts/run_phase65_pipeline.sh
```

---

## Dataset Sanity Check

Run this before training on a new Phase 65 dataset.

```bash
python scripts/check_phase65_dataset.py --config config/phase65_min3d/main.yaml
```

This checks:
- visible area statistics for both entities,
- amodal area statistics,
- overlap ratio frequency,
- clip type distribution.

The script exits with an error if the dataset is too weak for contact-heavy training.

---

## Visualization

Visualize SceneState predictions from a trained stage-1 checkpoint:

```bash
python scripts/viz_phase65.py \
  --config config/phase65_min3d/stage1.yaml \
  --ckpt outputs/phase65/stage1/best_scene_module.pt \
  --out outputs/phase65/viz
```

The visualization now includes:
- input frame,
- GT overlay,
- predicted overlay,
- per-entity visible maps,
- per-entity hidden maps,
- depth visualization,
- frame-level metric text.

---

## Smoke Tests and CI

Run the basic smoke tests locally:

```bash
pytest tests/test_phase65_smoke.py
```

These verify:
- SceneModule forward pass,
- adapter wiring,
- decoder baseline output shape,
- Stage 1 trainer step,
- Stage 2 trainer step.

A GitHub Actions workflow is also included:

- `.github/workflows/phase65_smoke.yml`

---

## Configs

Main configs:

- `config/phase65_min3d/main.yaml`
- `config/phase65_min3d/stage1.yaml`
- `config/phase65_min3d/stage2.yaml`

---

## Checkpoint Policy

Phase 65 uses stricter evaluator gating.

Checkpoints are rejected when they show signs of one-entity collapse, including:
- low `visible_survival_min`,
- low `amodal_survival_min`,
- very low `visible_iou_min`,
- extreme entity imbalance.

This is intended to prevent numerically smooth but visually collapsed models from being selected as best.

---

## Implementation Spec

Full design and implementation details:

- `docs/phase65_min3d_implementation_spec.md`

This document should be treated as the authoritative design reference.

---

## Legacy Code Policy

Older Phase 62 / Phase 64 files are being removed as the repository is refactored.

Mainline development should:
- build on `phase65_min3d/`
- use Phase 65 configs and scripts
- avoid reviving class-volume / first-hit / guide-family branching as the primary design

If a design choice conflicts with legacy assumptions, follow **Phase 65**.
