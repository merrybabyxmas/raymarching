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

## Training Stages

### Stage 1 — Scene-only pretraining
Train the Scene Module using:
- visible loss
- amodal / hidden loss
- temporal identity consistency
- optional weak depth ordering

### Stage 2 — Backbone alignment
Attach adapter + backbone and continue training while keeping scene losses active.

---

## Entrypoints

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

### Full pipeline
```bash
bash scripts/run_phase65_pipeline.sh
```

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
