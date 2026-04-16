# Phase 65 Mainline Architecture

## Goal

Reduce chimera in contact-heavy multi-entity generation by enforcing an **entity-separated layered 2.5D scene representation** before final image/video generation.

The mainline design principle is:

- **representation first**,
- **backbone alignment second**,
- **no shared class-volume competition**,
- **no first-hit winner-take-all scene core**.

---

## Current Mainline

The repository mainline is now **Phase 65 Minimal 3D (`phase65_min3d/`)**.

Historical Phase 62 / Phase 64 code may still exist elsewhere in the repository for reference, but it is **not** the architectural direction for new experiments.

The implementation spec for the new mainline lives at:

- `docs/phase65_min3d_implementation_spec.md`

---

## Core Idea

A scene with two entities is represented explicitly as:

- visible support for each entity,
- hidden / occluded support for each entity,
- per-entity relative depth,
- per-entity latent identity-aware features,
- persistent temporal slot memory.

This converts overlap from a **mixing problem** into an **occlusion reasoning problem**.

---

## Mainline File Map

```text
phase65_min3d/
├─ scene_outputs.py          # SceneMaps / SceneFeatures / SceneState dataclasses
├─ slot_encoder.py           # stable per-entity identity slots
├─ temporal_slots.py         # per-entity memory updates
├─ motion_rollout.py         # coarse 2.5D layout rollout
├─ layered_decoder.py        # per-entity visible / hidden / depth decoders
├─ occlusion_composer.py     # compose entity layers into coherent scene state
├─ scene_module.py           # top-level Scene Module
├─ losses.py                 # minimal core losses
├─ evaluator.py              # per-entity survival / IoU evaluation
├─ trainer_stage1.py         # scene-only pretraining
├─ trainer_stage2.py         # backbone alignment
├─ data.py                   # dataset wrapper / collate
├─ adapters/
│  ├─ base.py                # adapter interface
│  ├─ decoder_adapter.py     # reconstruction baseline adapter
│  ├─ animatediff.py         # AnimateDiff adapter stub/mainline path
│  └─ sdxl.py                # SDXL adapter stub/mainline path
└─ backbones/
   └─ reconstruction_decoder.py
```

Entrypoints:

```text
scripts/train_phase65_stage1.py
scripts/train_phase65_stage2.py
config/phase65_min3d/stage1.yaml
config/phase65_min3d/stage2.yaml
```

---

## Scene Representation Contract

For each entity `i` at frame `t`, the Scene Module predicts:

- `visible_i_t`
- `hidden_i_t`
- `amodal_i_t`
- `depth_i_t`
- `feat_i_t`

These are packaged into a `SceneState`:

```text
SceneState
├─ maps
│  ├─ visible_e0 / visible_e1
│  ├─ hidden_e0 / hidden_e1
│  ├─ amodal_e0 / amodal_e1
│  ├─ depth_e0 / depth_e1
│  └─ contact (optional)
├─ features
│  ├─ feat_e0 / feat_e1
│  └─ global_feat (optional)
└─ mem_e0 / mem_e1
```

### Invariants

- `amodal_ei >= visible_ei`
- `hidden_ei = amodal_ei - visible_ei` after clipping
- entity0 and entity1 remain separate throughout the scene core

---

## End-to-End Mainline Flow

```text
(entity names, prompt, optional prev state / prev frame)
        │
        ▼
EntitySlotEncoder
        │
        ▼
TemporalSlotMemory
        │
        ▼
MotionRollout
        │
        ▼
LayeredEntityDecoder (e0)
LayeredEntityDecoder (e1)
        │
        ▼
OcclusionComposer
        │
        ▼
SceneState
        │
        ▼
Backbone-specific Adapter
        │
        ▼
Backbone / Refiner / Decoder
        │
        ▼
Final RGB or backbone-native output
```

---

## Training Philosophy

We use a **2-stage** training schedule.

### Stage 1 — Scene-only pretraining

Train the Scene Module using only scene-aligned losses:

- visible loss,
- amodal / hidden loss,
- temporal identity consistency,
- optional weak depth ordering.

Goal:
- stable entity-separated layered scene representation,
- no single-entity collapse in SceneState.

### Stage 2 — Backbone alignment

Attach adapter + backbone.
Keep scene losses active while adding backbone generation loss.

Goal:
- backbone learns to obey the structured scene representation,
- scene representation does not collapse into a backbone-preferred one-object shortcut.

---

## Mainline Losses

The design target is **4 core losses**, plus optional weak depth ordering.

### Core

- `L_vis`
- `L_amo`
- `L_temp`
- `L_backbone`

### Optional weak term

- `L_depth`

### Explicit non-goals

The mainline architecture should not depend on a large collection of heuristic losses such as:

- balance loss,
- compactness loss,
- guide-family-specific losses,
- gate-push losses,
- class-volume exclusivity losses,
- many overlapping contrastive auxiliaries.

---

## Why This Replaces Earlier Approaches

Earlier repository phases centered on ideas such as:

- class-volume prediction with bg/entity competition,
- first-hit projection as the scene core,
- multi-family guide assembly,
- frame-conditioned scene parsing.

Those approaches can still be useful for historical comparison, but they are not the recommended mainline for solving the chimera problem.

Phase 65 replaces them with:

- per-entity explicit layered scene modeling,
- separate visible and hidden support,
- explicit temporal entity memory,
- backbone-agnostic scene interface.

---

## Checkpoint Selection Rule

A checkpoint is invalid if it visibly collapses to one entity.

Mainline best-checkpoint logic should enforce hard constraints such as:

- `visible_survival_min > threshold`
- `visible_iou_min > threshold`

A numerically smooth but single-entity solution must never be selected as best.

---

## Current Refactor Direction

The repository is being actively refactored toward:

- keeping reusable infrastructure,
- deleting misleading legacy entrypoints and docs,
- centralizing new work under `phase65_min3d/`.

If a design decision conflicts with legacy Phase 62 / 64 assumptions, prefer the Phase 65 design.
