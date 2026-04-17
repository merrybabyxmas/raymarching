# Phase 65 — Toy-Generalization Experiment Plan

## Goal
Train the Stage-1 scene module to learn an **entity-centric layered scene prior** rather than memorizing the toy dataset.

The key question is not just whether the model can separate two synthetic objects, but whether it learns a prior that survives:
- unseen shapes
- unseen collision dynamics
- unseen cameras
- randomized colors / appearance shortcuts
- future backbone swaps after the scene prior is frozen

---

## What success should mean
A successful Phase-1 scene prior should satisfy all of the following:

1. **Visible survival**: both entities stay alive under overlap.
2. **Amodal survival**: occluded entities remain represented.
3. **Generalization**: separation quality holds on withheld shape/camera/collision splits.
4. **Backbone independence**: the scene prior can be frozen and reused as a conditioning source for a later image/video backbone.

---

## Recommended experiment matrix

| Exp ID | Train Split | Eval Split | Purpose | Must-pass metrics |
|---|---|---|---|---|
| E1 | random | random | sanity baseline | visible_iou_min, entity_balance |
| E2 | shapes: sphere/box only | unseen cylinder-heavy | test shape shortcut failure | visible_iou_min, amodal_iou_min |
| E3 | colors randomized | colors randomized | remove red/blue shortcut | visible_survival_min |
| E4 | cameras: front/front_left | unseen top/front_right | test view generalization | visible_iou_min, temporal drift |
| E5 | collisions: head_on/orbit_tight | unseen diagonal/vertical_pass | test motion prior | visible_iou_min, entity_balance |
| E6 | all toy data | synthetic_overlap only | stress high-overlap regime | visible_survival_min, amodal_survival_min |
| E7 | scene prior frozen | backbone adapter A/B | test backbone portability | downstream two-object survival |

---

## Minimal acceptance criteria

### Stage-1 (scene module only)
- `visible_iou_min >= 0.18` on random held-out validation
- `visible_survival_min >= 0.80`
- `amodal_survival_min >= 0.95`
- `entity_balance >= 0.50`
- `temporal_identity_drift <= 0.10`

### Generalization
- no more than **25% relative drop** from random-split validation when evaluated on:
  - unseen shape split
  - unseen collision split
  - unseen camera split

### Backbone-transfer readiness
After freezing the scene module, a lightweight adapter should still recover two-entity conditioning maps with no collapse. This is a downstream Phase-2 requirement, but Stage-1 should already be measured with transfer in mind.

---

## Suggested training schedule

### Run A — Stable baseline
- config: `config/phase65_min3d/stage1_generalization.yaml`
- data split: `random`
- purpose: obtain a reproducible reference checkpoint

### Run B — Shape holdout
- split mode: `shape_holdout`
- held-out tags: `cylinder`
- purpose: test whether the scene prior memorizes shape templates

### Run C — Collision holdout
- split mode: `collision_holdout`
- held-out tags: `diagonal,vertical_pass`
- purpose: test motion / overlap generalization

### Run D — Camera holdout
- split mode: `camera_holdout`
- held-out tags: `top,front_right`
- purpose: test view invariance

---

## Recommended logging
Every checkpoint should save:
- train metrics
- validation metrics
- split metadata
- per-clip visualization GIFs
- failure cases with the lowest `visible_iou_min`

Every benchmark run should save a JSON report containing:
- aggregate metrics
- per-group metrics by shape / collision / camera
- number of clips
- rejected-checkpoint flag

---

## Reading failures correctly

### Failure type A — one entity dies only on held-out split
Likely memorization / shortcut.

### Failure type B — both entities survive but IoU drops
Scene prior is partially correct but poorly calibrated spatially.

### Failure type C — temporal drift spikes on held-out motions
Temporal memory is overfitting to training trajectories.

### Failure type D — random split looks good but held-out shape/camera fails badly
The model is not yet learning a portable structural prior.

---

## Immediate next step after Stage-1
Once a checkpoint passes E2/E4/E5 reasonably well:
1. freeze the scene module
2. export scene maps (`visible`, `amodal`, optional depth/separation)
3. attach a small adapter to a downstream backbone
4. compare transfer quality without retraining the scene prior

That is the first real test of whether the model learned a backbone-agnostic structural prior instead of a toy-specific predictor.
