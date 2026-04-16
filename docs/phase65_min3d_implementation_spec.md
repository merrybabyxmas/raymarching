# Phase 65 Minimal 3D (p65m) — End-to-End Implementation Specification

> Status: design spec for a clean reimplementation
> Goal: enable direct implementation of a **minimal layered 2.5D scene representation** for multi-entity video generation that reduces chimera during contact / occlusion.
>
> This document is intentionally prescriptive. The target reader should be able to implement the system by following this file alone.

---

# 0. Problem Statement

We want to generate a video of two entities (for example, a cat and a dog) that may:

- move independently,
- contact each other,
- occlude each other,
- reappear after being occluded,
- remain identity-consistent across time,
- **avoid chimera**, meaning the final rendered output should not collapse two entities into a hybrid object.

The central claim of Phase 65 is:

> Chimera can be reduced if the model is forced to represent scene structure using an **entity-separated layered 2.5D representation** before the final video backbone generates appearance.

This phase deliberately avoids:

- full NeRF / full radiance field complexity,
- class-volume competition with bg / entity0 / entity1 as a single softmax universe,
- overly complex multi-family guide logic,
- many auxiliary losses,
- frame-conditioned parsing that merely reconstructs the already-visible RGB input.

The design target is a **minimal, trainable, backbone-agnostic scene interface**.

---

# 1. Core Design Principles

## 1.1 Representation first, rendering second

We split the system into two conceptual pieces:

1. **Scene Module**
   - produces a structured scene representation that is entity-separated and occlusion-aware.
2. **Backbone Adapter + Backbone**
   - consumes the scene representation and generates the final image/video.

The Scene Module should solve:

- who exists,
- who is visible,
- who is hidden but still present,
- who is in front / behind,
- how identity persists across time.

The backbone should solve:

- appearance realism,
- texture refinement,
- motion realism,
- photorealistic final rendering.

## 1.2 Minimal 3D-aware means layered 2.5D, not full 3D fields

We do **not** start with full 3D voxel radiance fields.

We use the minimum representation that still converts overlap into occlusion reasoning:

For entity `i` at time `t`, predict:

- `V_i_t`: visible occupancy map
- `H_i_t`: hidden / occluded occupancy map
- `D_i_t`: relative depth map or frontness score
- `E_i_t`: identity-aware latent feature map

Amodal occupancy is:

- `A_i_t = clip(V_i_t + H_i_t, 0, 1)`

This is sufficient to express:

- existence,
- visibility,
- occlusion,
- relative ordering,
- entity identity persistence.

## 1.3 Backbone-agnostic scene interface

The scene representation must not depend on a specific diffusion hidden size.

The scene interface should be a combination of:

- low-bandwidth interpretable 2D maps,
- high-bandwidth per-entity latent feature maps.

This lets us keep the **Scene Module** fixed while swapping adapters for:

- AnimateDiff,
- SDXL,
- image decoder baselines,
- other video backbones.

## 1.4 Fewer losses, stronger structure

We prefer a structure that works with **4 core losses** rather than many heuristic losses.

Target training losses:

- visible loss,
- amodal / hidden loss,
- temporal identity consistency loss,
- backbone generation loss,
- optional weak depth ordering loss.

Everything else should be treated as diagnostic or optional ablation.

---

# 2. System Overview

The full system is:

```text
(text prompt, entity ids, optional prev state)
        │
        ▼
 Scene Module
   ├─ Entity Slot Encoder
   ├─ Motion / Interaction Rollout
   ├─ Layered 2.5D Decoder
   ├─ Temporal Slot Memory
   └─ Occlusion Composer
        │
        ├─ SceneMaps  (interpretable maps)
        └─ SceneFeatures (entity-aware latent features)
        │
        ▼
 Backbone-specific Adapter
        │
        ▼
 Video Backbone / Image Backbone
        │
        ▼
 Final RGB frames or noise predictions
```

---

# 3. Exact Input / Output Contracts

## 3.1 Dataset sample contract

Each training sample is a clip with length `T`.

Required fields:

```python
sample = {
    "frames":           FloatTensor[T, 3, H_img, W_img],
    "visible_masks":    FloatTensor[T, 2, H_mask, W_mask],
    "amodal_masks":     FloatTensor[T, 2, H_mask, W_mask],
    "depth_maps":       FloatTensor[T, 1, H_depth, W_depth],
    "depth_order":      LongTensor[T, 2],   # [front_idx, back_idx] when overlap exists
    "entity_names":     [str, str],         # e.g. ["cat", "dog"]
    "clip_type":        str,                # one of {O, C, R, X} or similar categories
}
```

Optional fields:

```python
sample.update({
    "solo_frames_e0":   FloatTensor[T, 3, H_img, W_img],
    "solo_frames_e1":   FloatTensor[T, 3, H_img, W_img],
    "camera":           Dict[str, Any],
    "contact_mask":     FloatTensor[T, 1, H_mask, W_mask],
    "metadata":         Dict[str, Any],
})
```

### Notes

- `visible_masks` are the actually visible per-entity masks.
- `amodal_masks` include occluded support.
- `depth_order` is only used where overlap exists; for non-overlap frames it can still store a deterministic ordering or `[-1, -1]`.
- `solo_frames_e0/e1` are highly recommended for diagnostics and optional isolation training.

---

## 3.2 Scene Module input contract

The Scene Module receives **semantic identity**, **optional previous memory**, and optionally **weak observation context**.

The minimal mainline implementation should use:

```python
scene_input = {
    "entity_names": [str, str],
    "text_prompt": str,
    "prev_state": Optional[SceneState],
    "t_index": int,
}
```

Optional observation-augmented version for training only:

```python
scene_input.update({
    "prev_frame": Optional[FloatTensor[3, H_img, W_img]],
})
```

### Important restriction

The Scene Module **must not** take the current target RGB frame as its main input in the mainline architecture.

Reason:
- if the module sees the target frame directly, it can become a frame-conditioned parser rather than a predictive structured scene model.

Teacher-forced training with previous frame is allowed, but current-frame direct supervision path should not become the main inference route.

---

## 3.3 Scene Module output contract

We define two outputs:

### A. Interpretable SceneMaps

```python
SceneMaps = {
    "visible_e0": FloatTensor[T, 1, Hs, Ws],
    "visible_e1": FloatTensor[T, 1, Hs, Ws],
    "hidden_e0":  FloatTensor[T, 1, Hs, Ws],
    "hidden_e1":  FloatTensor[T, 1, Hs, Ws],
    "amodal_e0":  FloatTensor[T, 1, Hs, Ws],
    "amodal_e1":  FloatTensor[T, 1, Hs, Ws],
    "depth_e0":   FloatTensor[T, 1, Hs, Ws],
    "depth_e1":   FloatTensor[T, 1, Hs, Ws],
    "contact":    FloatTensor[T, 1, Hs, Ws],   # optional
}
```

### B. High-bandwidth SceneFeatures

```python
SceneFeatures = {
    "feat_e0": FloatTensor[T, Ce, Hf, Wf],
    "feat_e1": FloatTensor[T, Ce, Hf, Wf],
    "global":  FloatTensor[T, Cg, Hf, Wf],     # optional
}
```

### Canonical merged SceneState

```python
SceneState = {
    "maps": SceneMaps,
    "features": SceneFeatures,
    "slot_memory": {
        "mem_e0": FloatTensor[Ds],
        "mem_e1": FloatTensor[Ds],
    }
}
```

---

## 3.4 Adapter input contract

The adapter receives the SceneState.

Backbone-agnostic interface:

```python
adapter_input = {
    "scene_maps": SceneMaps,
    "scene_features": SceneFeatures,
}
```

Backbone-specific adapters convert this to the correct format:

- multiscale feature injections for UNet backbones,
- token projections for transformer backbones,
- concatenative conditioning for decoder baselines.

---

# 4. Exact Architecture Specification

## 4.1 Module list

The new implementation should create a new package, for example:

```text
phase65_min3d/
├─ __init__.py
├─ scene_outputs.py
├─ slot_encoder.py
├─ motion_rollout.py
├─ temporal_slots.py
├─ layered_decoder.py
├─ occlusion_composer.py
├─ scene_module.py
├─ adapters/
│  ├─ base.py
│  ├─ animatediff.py
│  ├─ sdxl.py
│  └─ decoder_adapter.py
├─ backbones/
│  ├─ reconstruction_decoder.py
│  └─ animatediff_refiner.py
├─ losses.py
├─ evaluator.py
├─ trainer_stage1.py
├─ trainer_stage2.py
└─ utils.py
```

---

## 4.2 `scene_outputs.py`

Define typed containers.

### Required classes

```python
from dataclasses import dataclass
from typing import Dict, Optional
import torch

@dataclass
class SceneMaps:
    visible_e0: torch.Tensor
    visible_e1: torch.Tensor
    hidden_e0: torch.Tensor
    hidden_e1: torch.Tensor
    amodal_e0: torch.Tensor
    amodal_e1: torch.Tensor
    depth_e0: torch.Tensor
    depth_e1: torch.Tensor
    contact: Optional[torch.Tensor] = None

@dataclass
class SceneFeatures:
    feat_e0: torch.Tensor
    feat_e1: torch.Tensor
    global_feat: Optional[torch.Tensor] = None

@dataclass
class SceneState:
    maps: SceneMaps
    features: SceneFeatures
    mem_e0: Optional[torch.Tensor] = None
    mem_e1: Optional[torch.Tensor] = None
```

### Constraints

- all maps are float tensors in `[0, 1]` after sigmoid unless documented otherwise,
- `Hs, Ws` can differ from `Hf, Wf`,
- `amodal_ei` must be numerically consistent with `visible_ei + hidden_ei`.

---

## 4.3 `slot_encoder.py`

Purpose:
- encode entity identity independently of current frame pixels.

### Input

```python
entity_names: list[str]  # length 2
text_prompt: str
```

### Output

```python
slot_e0: Tensor[B, Ds]
slot_e1: Tensor[B, Ds]
```

### Recommended implementation

Minimal version:
- learned embedding table for frequent entity names,
- fallback text encoder projection for open vocabulary.

Pseudo-API:

```python
class EntitySlotEncoder(nn.Module):
    def __init__(self, slot_dim: int = 256, text_dim: int = 768): ...
    def forward(self, entity_names: list[str], text_context: Optional[Tensor] = None):
        return slot_e0, slot_e1
```

### Notes

- do not over-engineer this first,
- slot vectors must remain stable across time.

---

## 4.4 `temporal_slots.py`

Purpose:
- maintain per-entity persistent memory across frames.

### Input

```python
prev_mem_e0: Tensor[B, Ds]
prev_mem_e1: Tensor[B, Ds]
slot_e0: Tensor[B, Ds]
slot_e1: Tensor[B, Ds]
obs_e0: Tensor[B, Do]
obs_e1: Tensor[B, Do]
```

### Output

```python
new_mem_e0: Tensor[B, Ds]
new_mem_e1: Tensor[B, Ds]
```

### Recommended implementation

Use a small GRU or gated MLP memory cell.

Pseudo-API:

```python
class TemporalSlotMemory(nn.Module):
    def __init__(self, slot_dim: int = 256, obs_dim: int = 256): ...
    def forward(self, prev_mem_e0, prev_mem_e1, slot_e0, slot_e1, obs_e0, obs_e1):
        return mem_e0, mem_e1
```

### Rules

- memory updates must be **per entity**, never through an early shared fusion,
- memory is allowed to interact later through a contact module, but not at the first update step.

---

## 4.5 `motion_rollout.py`

Purpose:
- predict coarse layout state for each entity over time.

### Output per entity per frame

At minimum:

```python
layout_i_t = {
    "center": Tensor[B, 2],   # normalized x,y
    "scale": Tensor[B, 1],
    "frontness": Tensor[B, 1],
    "orientation": Tensor[B, 1],  # optional
}
```

### Recommended API

```python
class MotionRollout(nn.Module):
    def __init__(self, slot_dim: int = 256, hidden_dim: int = 256): ...
    def forward(self, slot_e0, slot_e1, mem_e0, mem_e1, t_index, global_context=None):
        return layout_e0, layout_e1
```

### Important note

This is not a physically perfect 3D pose model.
It is a **minimum 2.5D layout prior**.

Do not attempt full SE(3) dynamics in v1.

---

## 4.6 `layered_decoder.py`

Purpose:
- produce per-entity visible / hidden / depth maps and latent features.

### Inputs

```python
slot_i: Tensor[B, Ds]
mem_i: Tensor[B, Ds]
layout_i: Dict[str, Tensor]
```

### Outputs

```python
raw_visible_i: Tensor[B, 1, Hs, Ws]
raw_hidden_i:  Tensor[B, 1, Hs, Ws]
raw_depth_i:   Tensor[B, 1, Hs, Ws]
feat_i:        Tensor[B, Ce, Hf, Wf]
```

### Minimal implementation

Use a coordinate-aware decoder:
- broadcast slot + memory to spatial grid,
- concatenate positional channels,
- add layout channels (center heatmap, scale, frontness),
- small conv decoder.

Pseudo-API:

```python
class LayeredEntityDecoder(nn.Module):
    def __init__(self, slot_dim=256, feat_dim=64, Hs=64, Ws=64, Hf=32, Wf=32): ...
    def forward(self, slot_i, mem_i, layout_i):
        return raw_visible_i, raw_hidden_i, raw_depth_i, feat_i
```

### Important rules

- do not fuse entity0 and entity1 before producing raw per-entity outputs,
- visible and hidden should be emitted independently,
- do not derive hidden solely as a post-hoc subtraction from a single map,
- depth is per entity, not just global.

---

## 4.7 `occlusion_composer.py`

Purpose:
- resolve entity interaction into a coherent scene state.

### Inputs

```python
raw_visible_e0, raw_hidden_e0, raw_depth_e0, feat_e0
raw_visible_e1, raw_hidden_e1, raw_depth_e1, feat_e1
```

### Output

`SceneState`

### Core logic

1. Convert raw maps with sigmoid / bounded transforms.
2. Compute front/back relation from depth.
3. Suppress simultaneous double-visible support where overlap exists.
4. Reassign suppressed visible mass into hidden mass.
5. Construct amodal maps.
6. Preserve entity-specific features.

### Recommended equations

Let:

```text
v0 = sigmoid(raw_visible_e0)
v1 = sigmoid(raw_visible_e1)
h0 = sigmoid(raw_hidden_e0)
h1 = sigmoid(raw_hidden_e1)
d0 = sigmoid(raw_depth_e0)
d1 = sigmoid(raw_depth_e1)
```

Define frontness gates:

```text
f0 = sigmoid(alpha * (d1 - d0))
f1 = sigmoid(alpha * (d0 - d1))
```

where `alpha` controls sharpness.

Then visible allocation:

```text
v0_final = v0 * f0
v1_final = v1 * f1
```

Overlap leakage can be reduced by normalization:

```text
norm = (v0_final + v1_final).clamp(min=1.0)
v0_final = v0_final / norm
v1_final = v1_final / norm
```

Then hidden completion:

```text
a0 = clip(v0 + h0, 0, 1)
a1 = clip(v1 + h1, 0, 1)
h0_final = clip(a0 - v0_final, 0, 1)
h1_final = clip(a1 - v1_final, 0, 1)
```

### API

```python
class OcclusionComposer(nn.Module):
    def __init__(self, sharpness: float = 8.0): ...
    def forward(self, raw_visible_e0, raw_hidden_e0, raw_depth_e0, feat_e0,
                      raw_visible_e1, raw_hidden_e1, raw_depth_e1, feat_e1):
        return SceneState(...)
```

### Notes

- do not use a bg/entity softmax universe,
- do not collapse both entities into a single class map,
- preserve both entities as first-class citizens throughout composition.

---

## 4.8 `scene_module.py`

Purpose:
- orchestrate slot encoding, motion rollout, memory update, decoding, and composition.

### API

```python
class SceneModule(nn.Module):
    def __init__(self, ...): ...
    def forward(self,
                entity_names: list[str],
                text_prompt: str,
                prev_state: Optional[SceneState] = None,
                prev_frame: Optional[torch.Tensor] = None,
                t_index: Optional[int] = None) -> SceneState:
        ...
```

### Execution order

1. encode entity slots,
2. summarize previous state into observation tokens,
3. update temporal memory,
4. rollout motion/layout,
5. decode per-entity layered maps and features,
6. compose occlusion-aware SceneState.

### Output invariants

- `amodal_ei >= visible_ei` elementwise up to numerical tolerance,
- `hidden_ei = amodal_ei - visible_ei` after clipping,
- entity0 and entity1 are not merged into a single map anywhere inside SceneState.

---

# 5. Adapter Architecture

## 5.1 Philosophy

Adapters are allowed to be backbone-specific.
This is acceptable because the **scene representation** is the reusable part.

We define a common adapter API:

```python
class BaseSceneAdapter(nn.Module):
    def forward(self, scene_state: SceneState) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
```

---

## 5.2 Common adapter internals

Adapters should process scene input through two branches:

### Map branch

Input:

```python
maps = concat([
    visible_e0, visible_e1,
    hidden_e0, hidden_e1,
    amodal_e0, amodal_e1,
    depth_e0, depth_e1,
], dim=1)
```

### Feature branch

Input:

```python
feat = concat([feat_e0, feat_e1], dim=1)
```

### Rule

The adapter **must not** early-mix entity0 and entity1 into a single undifferentiated hidden state without preserving per-entity streams.

A safe pattern is:

- entity0 branch encoder,
- entity1 branch encoder,
- depth/global branch encoder,
- late fusion.

---

## 5.3 AnimateDiff / UNet adapter

### Output

A dict of multiscale tensors keyed by injection block:

```python
{
    "mid": Tensor[B, Cmid, Hmid, Wmid],
    "up1": Tensor[B, Cup1, Hup1, Wup1],
    "up2": Tensor[B, Cup2, Hup2, Wup2],
    "up3": Tensor[B, Cup3, Hup3, Wup3],
}
```

### Implementation outline

1. build a shared low-resolution scene trunk,
2. produce block-specific projections,
3. use post-normalization gating similar to current good practice,
4. keep entity-preserving structure until the last projector stage.

### Important rule

Do not use guide families (`front_only`, `dual`, `four_stream`) in the new system.
The scene representation already encodes visible/hidden/depth explicitly.

The adapter should be deterministic and fixed in semantics.

---

## 5.4 Decoder baseline adapter

For a structured decoder baseline (non-diffusion), simply concatenate scene maps and projected features into a small decoder.

This baseline is essential for checking whether SceneState itself is informative.

---

# 6. Loss Design

We aim for **4 core losses**, plus optional weak depth ordering.

## 6.1 Core loss 1 — visible loss

Supervise actual visibility.

```python
L_vis = dice(pred_visible_e0, gt_visible_e0) + dice(pred_visible_e1, gt_visible_e1)
```

Use Dice or BCE+Dice.

Recommendation:
- Dice + BCE combined if masks are thin / sparse.

---

## 6.2 Core loss 2 — amodal / hidden loss

Supervise existence even under occlusion.

```python
pred_amodal_e0 = pred_visible_e0 + pred_hidden_e0
pred_amodal_e1 = pred_visible_e1 + pred_hidden_e1

L_amo = dice(pred_amodal_e0, gt_amodal_e0) + dice(pred_amodal_e1, gt_amodal_e1)
```

Additionally enforce consistency:

```python
L_occ = mean(relu(pred_visible_e0 - pred_amodal_e0)) + mean(relu(pred_visible_e1 - pred_amodal_e1))
```

You may absorb `L_occ` into `L_amo` if you want to stay at exactly four named terms.

---

## 6.3 Core loss 3 — temporal identity consistency

Purpose:
- keep entity identity stable across contact and reappearance.

Define pooled entity features:

```python
z_e0_t = masked_avg_pool(feat_e0_t, amodal_e0_t)
z_e1_t = masked_avg_pool(feat_e1_t, amodal_e1_t)
```

Then:

```python
L_temp = mse(z_e0_t, stopgrad(z_e0_t_prev)) + mse(z_e1_t, stopgrad(z_e1_t_prev))
```

Optional stronger form:
- contrastive temporal loss with negative entity across slot mismatch.

### Important rule

Do not start with a complex contrastive zoo.
A simple same-slot consistency loss is enough in v1.

---

## 6.4 Core loss 4 — backbone generation loss

For reconstruction decoder baseline:

```python
L_backbone = l1(pred_rgb, gt_rgb) + 0.1 * l2(pred_rgb, gt_rgb)
```

For diffusion backbone:

```python
L_backbone = mse(pred_noise, true_noise)
```

### Important rule

This loss must not be the only loss during stage 2.
Keep scene losses active to prevent collapse into a backbone-preferred one-object solution.

---

## 6.5 Optional weak loss — depth ordering

If depth GT or front/back GT is available:

```python
L_depth = ranking_loss(pred_depth_e0, pred_depth_e1, gt_front_idx)
```

This should be weak.
If you do not have clean depth supervision, make it optional and small.

---

## 6.6 Total losses by stage

### Stage 1

```python
L_stage1 =
    lambda_vis  * L_vis +
    lambda_amo  * L_amo +
    lambda_temp * L_temp +
    lambda_depth * L_depth   # optional weak term
```

Recommended initial weights:

```python
lambda_vis  = 1.0
lambda_amo  = 1.0
lambda_temp = 0.25
lambda_depth = 0.1
```

### Stage 2

```python
L_stage2 =
    lambda_backbone * L_backbone +
    lambda_vis      * L_vis +
    lambda_amo      * L_amo +
    lambda_temp     * L_temp +
    lambda_depth    * L_depth
```

Recommended initial weights:

```python
lambda_backbone = 1.0
lambda_vis      = 0.5
lambda_amo      = 0.5
lambda_temp     = 0.1
lambda_depth    = 0.05
```

### Explicit non-goals

Do **not** add the following to v1 mainline unless a very specific failure mode remains:

- balance loss,
- compactness loss,
- gate push loss,
- overlay preserve loss,
- many separate contrast losses,
- many family-specific guide losses.

Use them only as temporary ablations if absolutely necessary.

---

# 7. Training Schedule

We use a **2-stage** pipeline.

## 7.1 Stage 1 — Scene-only pretraining

Train only:
- Scene Module

Do not attach the full video backbone yet.

### Goal

Learn a stable entity-separated layered scene representation.

### Inputs

- entity names,
- prompt,
- previous state,
- optional previous frame.

### Outputs supervised

- visible maps,
- hidden/amodal maps,
- optional depth,
- temporal consistency.

### Frozen components

- all adapters,
- all video backbones.

### Recommended optimizer settings

```python
AdamW(
    scene_module.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
)
```

### Suggested duration

Enough to reach stable:
- visible IoU,
- amodal IoU,
- temporal consistency.

Do not overtrain to the point where the module memorizes training layouts.

---

## 7.2 Stage 2 — Backbone alignment

Attach:
- adapter,
- backbone.

### Trainable components

Option A (recommended to start):
- adapter: trainable
- backbone: partially trainable or LoRA only
- scene module: low learning rate or partially frozen

### Optimizer groups example

```python
optimizer = AdamW([
    {"params": scene_module.parameters(), "lr": 5e-5},
    {"params": adapter.parameters(),      "lr": 2e-4},
    {"params": backbone_lora.parameters(),"lr": 1e-4},
], weight_decay=1e-4)
```

### Goal

Teach the backbone to obey the scene representation without destroying it.

### Key rule

Keep stage-1 scene losses alive.
Do not train only with backbone loss.

---

## 7.3 Optional short joint finetune

Only if needed.

- short,
- low learning rate,
- stop immediately if scene metrics degrade while visual quality superficially improves.

This stage is not required for v1.

---

# 8. Dataset Construction Plan

## 8.1 Required supervision

For v1, build a synthetic dataset that provides:

- RGB clip,
- visible masks per entity,
- amodal masks per entity,
- depth maps or front/back ordering,
- entity IDs across time.

### Recommended directory format

```text
data/phase65/
  sample_000001/
    meta.json
    frames/
      0000.png
      0001.png
      ...
    visible_masks/
      0000_e0.png
      0000_e1.png
      ...
    amodal_masks/
      0000_e0.png
      0000_e1.png
      ...
    depth/
      0000.npy
      0001.npy
      ...
    solo/
      0000_e0.png
      0000_e1.png
      ...
```

### `meta.json` recommended fields

```json
{
  "entity_names": ["cat", "dog"],
  "clip_type": "contact",
  "camera_id": 3,
  "num_frames": 16,
  "overlap_ratio_mean": 0.21,
  "front_switches": 2,
  "has_reappearance": true
}
```

---

## 8.2 Dataset split policy

Create stratified splits by clip type.

Suggested groups:
- `O`: clean occlusion
- `C`: contact / same-depth difficult interaction
- `R`: reappearance after occlusion
- `X`: stress / unusual poses / hard negatives

Recommendation:
- train 80%
- val 20%
- preserve clip-type ratios in both sets.

---

## 8.3 Dataset sanity checks

Before stage 1, compute and save:

- visible area histogram per entity,
- hidden area histogram per entity,
- overlap ratio histogram,
- front/back switch counts,
- fraction of frames where both entities survive visibly,
- clip counts per category.

Fail early if:
- one entity is systematically tiny,
- amodal masks are inconsistent,
- visible > amodal frequently,
- depth ordering is missing in overlap-heavy clips.

---

# 9. Evaluation Plan

We evaluate Scene Module and final generation separately.

## 9.1 Stage 1 scene metrics

### Required metrics

- `visible_iou_e0`, `visible_iou_e1`
- `visible_iou_min`
- `amodal_iou_e0`, `amodal_iou_e1`
- `amodal_iou_min`
- `temporal_identity_drift`
- `visible_survival_min`
- `hidden_reappearance_consistency`

### Definitions

#### visible survival min

For each entity `i`, compute fraction of frames where predicted visible mass exceeds threshold.
Take the minimum across entities.

#### temporal identity drift

Cosine or L2 distance between pooled entity features across adjacent frames.

---

## 9.2 Stage 2 generation metrics

### Required metrics

- generated visible IoU per entity,
- generated visible survival min,
- isolated/composite consistency,
- two-object detector score (optional external evaluator),
- final human visual inspection on contact clips.

### Important selection rule

A checkpoint cannot be considered best if either entity is consistently absent.

Enforce hard constraints such as:

```text
pred_visible_area_e0 > tau
pred_visible_area_e1 > tau
visible_survival_min > tau_survival
```

This prevents the old failure mode of choosing a numerically good single-entity checkpoint.

---

# 10. Implementation Steps in Order

## Step 1
Create `phase65_min3d/scene_outputs.py`.

## Step 2
Implement `EntitySlotEncoder` in `slot_encoder.py`.

## Step 3
Implement `TemporalSlotMemory` in `temporal_slots.py`.

## Step 4
Implement `MotionRollout` in `motion_rollout.py`.

## Step 5
Implement `LayeredEntityDecoder` in `layered_decoder.py`.

## Step 6
Implement `OcclusionComposer` in `occlusion_composer.py`.

## Step 7
Implement `SceneModule` in `scene_module.py`.

## Step 8
Implement `losses.py` with the 4+1 losses only.

## Step 9
Implement `trainer_stage1.py` and verify that scene maps learn without any backbone.

## Step 10
Implement a tiny reconstruction decoder baseline and confirm SceneState is informative.

## Step 11
Implement `adapters/base.py` and one real adapter, preferably AnimateDiff first.

## Step 12
Implement `trainer_stage2.py` with low-LR Scene Module + adapter + backbone alignment.

## Step 13
Implement `evaluator.py` with hard best-checkpoint criteria based on per-entity survival.

---

# 11. Explicit Deletions / Non-Reuse Guidance

When implementing Phase 65, do **not** carry over the following as mainline concepts:

- bg / entity0 / entity1 class competition as a single voxel softmax universe,
- first-hit visible class as the primary scene interface,
- guide-family branching (`front_only`, `dual`, `four_stream`) as a permanent architectural knob,
- contract-heavy best model selection where a single-entity solution can still pass,
- current-frame RGB parsing as the main route to scene decomposition,
- large collections of heuristic losses.

These may remain in the repository for historical reference, but should not shape the new mainline implementation.

---

# 12. What Can Be Reused from the Existing Repository

The following are safe to reuse or adapt:

- dataset loading infrastructure,
- visualization utilities,
- experiment config patterns,
- checkpoint IO helpers,
- multiscale hook injection mechanics,
- evaluation plotting utilities,
- synthetic data generation scripts and mask/depth export pipelines.

The new design should reuse **infrastructure**, not the old conceptual core.

---

# 13. Minimal Config Specification

A first-pass YAML config should contain only what is needed.

```yaml
model:
  slot_dim: 256
  feat_dim: 64
  hidden_dim: 128
  Hs: 64
  Ws: 64
  Hf: 32
  Wf: 32

train:
  stage1_epochs: 100
  stage2_epochs: 80
  batch_size: 4
  lr_scene_stage1: 3.0e-4
  lr_scene_stage2: 5.0e-5
  lr_adapter: 2.0e-4
  lr_backbone: 1.0e-4
  grad_clip: 1.0

loss:
  lambda_vis: 1.0
  lambda_amo: 1.0
  lambda_temp: 0.25
  lambda_depth: 0.1
  lambda_backbone: 1.0

data:
  root: data/phase65
  image_size: 256
  mask_size: 64
  feat_size: 32
  num_frames: 16
  use_prev_frame: true

adapter:
  type: animatediff
  inject_blocks: [mid, up1, up2, up3]
  guide_max_ratio: 0.15
```

---

# 14. Sanity Tests to Implement Before Full Training

## Unit tests

### Test 1 — scene map shape test
- verify all SceneMaps and SceneFeatures have expected shapes.

### Test 2 — amodal consistency test
- verify `amodal ~= visible + hidden` within tolerance.

### Test 3 — front/back suppression test
- overlapping entities with depth separation should not both be fully visible at same pixel.

### Test 4 — temporal memory stability test
- repeated same input should not randomly swap entity memories.

### Test 5 — adapter output test
- adapter outputs correct multiscale channels and spatial sizes for the chosen backbone.

---

# 15. Failure Modes and Debugging Checklist

## Failure A — one entity disappears in stage 1
Check:
- visible mask imbalance in dataset,
- slot encoder collapse,
- decoder early shared fusion,
- loss weight imbalance (`lambda_vis` too small),
- composer suppressing one entity too hard.

## Failure B — scene maps look good but final output is still chimera
Check:
- adapter early-fusing both entities,
- backbone loss overwhelming scene losses,
- scene module learning rate too high in stage 2,
- no hard checkpoint rule for per-entity survival,
- final backbone not receiving high-bandwidth `feat_e0/e1` information.

## Failure C — temporal identity drifts during contact
Check:
- memory update too weak,
- no masked pooled feature consistency,
- entity slot vectors not stable,
- motion rollout entangling the two entities too early.

## Failure D — hidden maps become trivial zeros
Check:
- dataset amodal masks quality,
- no frames with real occlusion in train split,
- hidden channel supervised only indirectly,
- composer making visible dominate too strongly.

---

# 16. Final Recommendation

The correct implementation strategy is:

1. build a **new clean package** for Phase 65,
2. keep the Scene Module **small, interpretable, and entity-separated**, 
3. use only the **4 core losses** plus weak optional depth ordering,
4. train in **2 stages**,
5. force checkpoint selection to reject single-entity collapse,
6. treat adapters as replaceable backbone-specific translators rather than as the core source of scene understanding.

The essential design choice is this:

> The backbone is not responsible for discovering entity separation from scratch.
> The Scene Module must provide an explicit, structured, occlusion-aware representation first.

If the system follows this rule, it has a realistic chance of reducing chimera in contact-heavy multi-entity video generation.

---

# 17. Immediate Next Step

Implement Stage 1 only.

Do not start with AnimateDiff.

Success criterion for the first milestone:
- entity-separated visible maps,
- non-trivial hidden maps on occlusion frames,
- stable temporal identity,
- no single-entity collapse in SceneState.

Only after that should Stage 2 backbone alignment begin.
