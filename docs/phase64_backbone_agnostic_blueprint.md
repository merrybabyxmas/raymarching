# Phase 64 Blueprint
## Backbone-agnostic Entity-centric 3D Scene Prior for Chimera-free Multi-entity Video Generation

Version: draft v1  
Target repository: `merrybabyxmas/raymarching`  
Primary objective: learn a **backbone-independent structural prior** that decomposes a multi-entity interaction scene into **entity-centric 3D fields** and uses that decomposition as conditioning so that **chimera is reduced even when the downstream generative backbone changes**.

---

# 0. Executive summary

This document proposes a full redesign of the current system so that the most important part of the method is **not tied to a specific UNet / AnimateDiff / SDEdit pipeline**, but instead becomes a standalone **Scene Prior Module** that:

1. parses the prompt into entities and interaction,
2. predicts a temporally consistent **entity-centric scene representation**,
3. renders **visible / amodal / depth / separation** maps from that scene,
4. exports those maps to a thin **Backbone Adapter Layer**, and
5. allows many backbones to consume the same structured representation.

The main thesis is:

> The thing that should generalize across backbones is **not** the guide tensor itself. The thing that should generalize is the **entity-centric scene decomposition**. The adapter to a specific backbone should be shallow, replaceable, and cheap.

This blueprint is intentionally opinionated:

- delete components whose main function is legacy ablation or contract engineering,
- keep the genuinely valuable ideas already present in the repository,
- add only modules that directly improve identifiability, transfer, and final rendering quality.

The proposal is designed to be **implemented incrementally inside the current repository**, not as a complete restart.

---

# 1. What success means

The final target is not:

- good loss curves,
- a better contract score,
- a nicer separation heatmap,
- or a better single-backbone result.

The final target is:

> For a prompt like "a cat and a dog rolling together", the system should build a scene representation in which cat and dog remain separate entities through occlusion, contact, and reappearance, and that same scene representation should be consumable by multiple downstream generators with reduced chimera.

A result counts as a success only if **all four** are true:

1. **Entity persistence**: both entities remain represented through time, even when one becomes partially or mostly occluded.
2. **Amodal correctness**: the hidden entity remains latent, not erased.
3. **Backbone transfer**: the same scene representation improves at least two distinct downstream generators.
4. **Render usefulness**: the downstream generator actually turns the representation into visually correct images, not just better internal masks.

---

# 2. Diagnosis of the current repository

The repository already contains several strong ingredients that should be preserved.

## 2.1 Valuable parts already present

### Transmittance-based projector
`models/phase62/projection.py` computes occupancy, transmittance, front probabilities, back probabilities, visible projections, and amodal projections from entity probabilities. This is the correct geometric direction for the project because it explicitly models hidden vs visible entity mass. fileciteturn15file0

### Structured guide assembly and injection
`models/phase62/conditioning.py` already has a reasonably good injection infrastructure:
- block-specific projectors,
- guide gates,
- multiscale injection,
- gate-after-normalization fix,
- and a `four_stream` family that already uses visible and amodal streams. fileciteturn14file0

### Rendered GT builder
`scripts/build_volume_gt.py` uses depth, entity masks, and depth order to build volumetric supervision. This is important because it gives the method something stronger than 2D silhouette fitting. fileciteturn16file0

## 2.2 What is still fundamentally wrong

### The representation and the adapter are entangled
Right now the code learns a representation that is already shaped by the needs of a specific backbone adapter. That means the learned scene structure is not cleanly separable from the choice of generator.

### Entity competition is still too strong
Even with transmittance and four-stream conditioning, the learned fields can still converge to one-dominant-entity solutions because a stronger visible stream often produces a stronger guide, which can further amplify the same entity.

### The system is still optimized mainly through surrogate objectives
The trainer contains many useful losses, but the main optimization process is still more aligned with internal structure metrics than with the final user goal: two visibly distinct entities in the final video. fileciteturn8file0

### Current guide representations are too close to “backbone-conditioned features” and not yet “backbone-agnostic scene outputs”
`GuideFeatureAssembler` currently receives backbone features `F_g, F_0, F_1` and immediately mixes them with scene-derived projections. That is useful for current performance, but it is exactly what prevents clean transfer. fileciteturn14file0

---

# 3. Core design principle for Phase 64

We split the system into **three clean layers**.

## Layer A — Scene Prior Module (backbone-agnostic)
Input:
- prompt,
- entity identities,
- motion seed / trajectory seed,
- optional coarse image context.

Output:
- per-entity visible maps,
- per-entity amodal maps,
- per-entity depth occupancy,
- global depth / ordering maps,
- optional low-resolution composite render,
- temporal slot memory.

This layer must not know or care whether the downstream generator is:
- AnimateDiff,
- SDXL,
- DiT video model,
- an image UNet,
- or a future backbone.

## Layer B — Backbone Adapter (thin, replaceable)
Input:
- structured scene outputs from Layer A.

Output:
- conditioning tensors matched to a specific backbone’s interface.

This layer is allowed to be backbone-specific. It should be shallow and small.

## Layer C — Renderer / Refiner Backbone
This is a standard generator that consumes the scene prior through the adapter and refines or renders the final image/video.

---

# 4. Formal problem statement

We model a video as a sequence of frames indexed by time `t = 1..T`.

Assume two entities for now, with identities:

- entity 1: `e1`
- entity 2: `e2`

For each entity `i` and time `t`, we define a 3D field over point `p ∈ R^3`:

```math
f_{i,t}(p) = (\sigma_{i,t}(p), a_{i,t}(p), z_i)
```

where:

- `σ_{i,t}(p) >= 0` is density / occupancy,
- `a_{i,t}(p)` is appearance feature,
- `z_i` is a time-invariant identity embedding.

The scene at time `t` is:

```math
S_t = {f_{1,t}, f_{2,t}}
```

The camera ray for pixel `u` is `r(u, λ)` with depth parameter `λ`.

Entity-specific density along the ray:

```math
\sigma_{i,t}(u, \lambda) = \sigma_{i,t}(r(u, \lambda))
```

Total density:

```math
\Sigma_t(u, \lambda) = \sum_i \sigma_{i,t}(u, \lambda)
```

Transmittance:

```math
T_t(u, \lambda) = \exp\left(-\int_0^\lambda \Sigma_t(u, s) ds\right)
```

Visible contribution of entity `i`:

```math
V_{i,t}(u) = \int T_t(u, \lambda)\,\sigma_{i,t}(u, \lambda)\,d\lambda
```

Amodal presence of entity `i`:

```math
A_{i,t}(u) = 1 - \exp\left(-\int \sigma_{i,t}(u, \lambda)\,d\lambda\right)
```

Separation map:

```math
S^{sep}_t(u) = V_{1,t}(u) - V_{2,t}(u)
```

Depth cue:

```math
D_t(u) = \frac{\int \lambda \sum_i T_t(u,\lambda)\sigma_{i,t}(u,\lambda)d\lambda}
{\int \sum_i T_t(u,\lambda)\sigma_{i,t}(u,\lambda)d\lambda + \epsilon}
```

These quantities are the *portable scene outputs*.

---

# 5. New repository architecture

## 5.1 Proposed directory layout

```text
raymarching/
├── scene_prior/
│   ├── entity_parser.py
│   ├── motion_model.py
│   ├── entity_field.py
│   ├── renderer.py
│   ├── temporal_memory.py
│   ├── scene_outputs.py
│   └── losses.py
│
├── adapters/
│   ├── base_adapter.py
│   ├── animatediff_adapter.py
│   ├── sdxl_adapter.py
│   ├── dit_video_adapter.py
│   └── guide_encoders.py
│
├── backbones/
│   ├── animatediff_refiner.py
│   ├── sdedit_refiner.py
│   ├── reconstruction_decoder.py
│   └── interface.py
│
├── data/
│   ├── phase64_dataset.py
│   ├── phase64_transforms.py
│   ├── phase64_splits.py
│   └── build_scene_gt.py
│
├── training/
│   ├── stage0_validate_dataset.py
│   ├── stage1_train_scene_prior.py
│   ├── stage2_train_decoder.py
│   ├── stage3_train_adapter_backbone.py
│   ├── stage4_transfer_eval.py
│   └── evaluator_phase64.py
│
├── scripts/
│   ├── train_phase64_scene.py
│   ├── train_phase64_decoder.py
│   ├── train_phase64_backbone.py
│   ├── eval_phase64_transfer.py
│   └── export_scene_outputs.py
│
└── docs/
    └── phase64_blueprint.md
```

---

# 6. What to keep, remove, and refactor from the current codebase

## 6.1 Keep

### Keep the geometric rendering idea from `models/phase62/projection.py`
The current file already computes transmittance, visible, and amodal from entity probabilities. That logic should be migrated into a generalized entity-field renderer. fileciteturn15file0

### Keep multiscale guide injection infrastructure from `models/phase62/conditioning.py`
The hook manager, block registration, and gate-after-normalization design are worth preserving. fileciteturn14file0

### Keep rendered-GT builder philosophy from `scripts/build_volume_gt.py`
The idea of using rendered depth, masks, and explicit depth order is correct and should be generalized into scene-level GT. fileciteturn16file0

## 6.2 Remove or deprecate

### Remove guide family branching from mainline
Deprecate:
- `none`
- `front_only`
- `dual`

Mainline should use a single structured guide format.

### Remove contract-heavy checkpoint selection
Current contract-driven checkpoint selection can remain in archived experiments, but Phase 64 should select checkpoints based on final-goal aligned metrics.

### Remove backbone feature dependence from the scene prior
The scene prior should not take backbone features as primary inputs. That is the central decoupling move.

## 6.3 Refactor

### Refactor `GuideFeatureAssembler` into two modules
1. `SceneGuideEncoder`: backbone-agnostic structured-to-guide encoder
2. `BackboneAdapter`: backbone-specific reshaper / projector

---

# 7. Scene Prior Module

## 7.1 Entity parser
Input text prompt should be parsed into:

- entity labels
- interaction label
- optional trajectory hints

Example:

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EntitySpec:
    name: str
    attributes: List[str]

@dataclass
class ScenePrompt:
    entities: List[EntitySpec]
    interaction: str
    background: Optional[str] = None


def parse_prompt(prompt: str) -> ScenePrompt:
    # placeholder parser; replace with LLM / grammar parser later
    if "cat" in prompt and "dog" in prompt:
        return ScenePrompt(
            entities=[EntitySpec("cat", []), EntitySpec("dog", [])],
            interaction="rolling together",
            background="generic"
        )
    raise ValueError(f"Unsupported prompt: {prompt}")
```

This parser must eventually become more robust, but it should stay outside the backbone.

---

## 7.2 Motion model
The motion model predicts pose / trajectory tokens for each entity.

```python
import torch
import torch.nn as nn


class MotionModel(nn.Module):
    def __init__(self, id_dim: int = 128, hidden_dim: int = 256, pose_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(id_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pose_dim),
        )

    def forward(self, entity_id: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # entity_id: (B, id_dim)
        # timesteps: (B, 1)
        x = torch.cat([entity_id, timesteps], dim=-1)
        return self.net(x)
```

This is intentionally simple. In the first pass, it only needs to provide a controllable latent trajectory.

---

## 7.3 Entity field module
Each entity gets its own field decoder.

```python
import torch
import torch.nn as nn


class EntityField(nn.Module):
    def __init__(self, id_dim: int = 128, pose_dim: int = 16, hidden_dim: int = 256, app_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 + id_dim + pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.to_sigma = nn.Linear(hidden_dim, 1)
        self.to_app = nn.Linear(hidden_dim, app_dim)

    def forward(self, xyz: torch.Tensor, entity_id: torch.Tensor, pose_code: torch.Tensor):
        # xyz: (..., 3)
        # entity_id: (..., id_dim)
        # pose_code: (..., pose_dim)
        h = torch.cat([xyz, entity_id, pose_code], dim=-1)
        h = self.mlp(h)
        sigma = torch.relu(self.to_sigma(h))
        app = self.to_app(h)
        return sigma, app
```

Two important rules:

1. one decoder per entity slot or a shared decoder conditioned on identity,
2. no competition between entity slots inside the field decoder.

---

## 7.4 Temporal slot memory
A hidden entity must survive occlusion. For that, each slot gets a memory state.

```python
import torch
import torch.nn as nn


class TemporalSlotMemory(nn.Module):
    def __init__(self, slot_dim: int = 128):
        super().__init__()
        self.gru = nn.GRUCell(slot_dim, slot_dim)

    def forward(self, prev_state: torch.Tensor, slot_obs: torch.Tensor) -> torch.Tensor:
        return self.gru(slot_obs, prev_state)
```

This should update per entity:

- visible evidence,
- hidden evidence,
- motion continuity,
- reappearance recovery.

---

# 8. Renderer

The renderer is the heart of the method.

## 8.1 Interface

```python
from dataclasses import dataclass
import torch


@dataclass
class SceneOutputs:
    visible_e0: torch.Tensor
    visible_e1: torch.Tensor
    amodal_e0: torch.Tensor
    amodal_e1: torch.Tensor
    depth_map: torch.Tensor
    sep_map: torch.Tensor
    visible_rgb_coarse: torch.Tensor | None = None
```

## 8.2 Differentiable entity renderer

```python
import torch
import torch.nn as nn


class EntityRenderer(nn.Module):
    def __init__(self, n_samples: int = 16):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, sigma_e0: torch.Tensor, sigma_e1: torch.Tensor, rgb_e0=None, rgb_e1=None):
        # sigma_e0, sigma_e1: (B, K, H, W)
        occ = 1.0 - (1.0 - sigma_e0) * (1.0 - sigma_e1)

        trans = []
        running = torch.ones_like(occ[:, 0])
        for k in range(occ.shape[1]):
            trans.append(running)
            running = running * (1.0 - occ[:, k]).clamp(min=1e-6)
        trans = torch.stack(trans, dim=1)

        visible_e0 = (trans * sigma_e0).sum(dim=1)
        visible_e1 = (trans * sigma_e1).sum(dim=1)

        amodal_e0 = 1.0 - (1.0 - sigma_e0).prod(dim=1)
        amodal_e1 = 1.0 - (1.0 - sigma_e1).prod(dim=1)

        depth_axis = torch.linspace(0, 1, sigma_e0.shape[1], device=sigma_e0.device)[None, :, None, None]
        vis_total = visible_e0 + visible_e1
        depth_map = ((trans * (sigma_e0 + sigma_e1) * depth_axis).sum(dim=1)
                     / vis_total.clamp(min=1e-6))

        sep_map = visible_e0 - visible_e1

        return SceneOutputs(
            visible_e0=visible_e0,
            visible_e1=visible_e1,
            amodal_e0=amodal_e0,
            amodal_e1=amodal_e1,
            depth_map=depth_map,
            sep_map=sep_map,
            visible_rgb_coarse=None,
        )
```

This renderer intentionally mirrors the logic that already exists in the repository, but is now explicit and backbone-independent. Compare this to the current `FirstHitProjector` design that uses transmittance and computes visible/amodal from entity probabilities. fileciteturn15file0

---

# 9. Scene guide format

This is the key abstraction that must remain portable.

## 9.1 Canonical guide tensor format
For each frame, build:

```text
channels = [
  visible_e0,
  visible_e1,
  amodal_e0,
  amodal_e1,
  depth_map,
  sep_map,
  hidden_e0 = relu(amodal_e0 - visible_e0),
  hidden_e1 = relu(amodal_e1 - visible_e1),
]
```

This 8-channel tensor is the canonical scene representation for adapters.

## 9.2 SceneGuideEncoder

```python
import torch
import torch.nn as nn


class SceneGuideEncoder(nn.Module):
    def __init__(self, in_ch: int = 8, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, scene_outputs: SceneOutputs) -> torch.Tensor:
        hidden_e0 = (scene_outputs.amodal_e0 - scene_outputs.visible_e0).clamp(min=0)
        hidden_e1 = (scene_outputs.amodal_e1 - scene_outputs.visible_e1).clamp(min=0)
        x = torch.stack([
            scene_outputs.visible_e0,
            scene_outputs.visible_e1,
            scene_outputs.amodal_e0,
            scene_outputs.amodal_e1,
            scene_outputs.depth_map,
            scene_outputs.sep_map,
            hidden_e0,
            hidden_e1,
        ], dim=1)
        return self.net(x)
```

This module replaces the need for multiple guide families.

---

# 10. Backbone adapters

Adapters are intentionally shallow.

## 10.1 Base adapter API

```python
from abc import ABC, abstractmethod
import torch


class BaseBackboneAdapter(ABC):
    @abstractmethod
    def build_guides(self, scene_features: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError
```

## 10.2 AnimateDiff-style adapter

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


BLOCK_DIMS = {
    "mid": 1280,
    "up1": 1280,
    "up2": 640,
    "up3": 320,
}

BLOCK_SPATIAL = {
    "mid": (4, 4),
    "up1": (8, 8),
    "up2": (16, 16),
    "up3": (32, 32),
}


class AnimateDiffAdapter(nn.Module):
    def __init__(self, in_ch: int = 64, inject_blocks=("mid", "up1", "up2", "up3")):
        super().__init__()
        self.inject_blocks = inject_blocks
        self.projectors = nn.ModuleDict()
        for bn in inject_blocks:
            self.projectors[bn] = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(in_ch, BLOCK_DIMS[bn], 1),
            )
        self.gates = nn.ParameterDict({bn: nn.Parameter(torch.zeros(1)) for bn in inject_blocks})

    def build_guides(self, scene_features: torch.Tensor):
        guides = {}
        for bn in self.inject_blocks:
            h, w = BLOCK_SPATIAL[bn]
            x = F.interpolate(scene_features, size=(h, w), mode="bilinear", align_corners=False)
            proj = self.projectors[bn](x)
            gate = torch.tanh(self.gates[bn])
            guides[bn] = proj * gate
        return guides
```

This is the adapter. It is allowed to be specific.

---

# 11. Structured decoder before diffusion

One of the major lessons from Phase 63 is that the scene decomposition can improve substantially before the final image generator knows how to use it. Therefore a supervised decoder should sit between the scene prior and the final diffusion refiner.

## 11.1 Why it is needed

If structured maps are good but the final image is still noisy or single-entity, the problem is not the scene prior anymore. The problem is that the downstream generator does not know how to turn the structured representation into pixels.

Therefore we introduce:

```text
Scene Prior -> Structured Decoder -> Coarse RGB -> Diffusion Refiner
```

## 11.2 Decoder module

```python
import torch
import torch.nn as nn


class StructuredDecoder(nn.Module):
    def __init__(self, in_ch: int = 64, out_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, scene_features: torch.Tensor) -> torch.Tensor:
        return self.net(scene_features)
```

Train this decoder first before trusting a backbone-specific refiner.

---

# 12. Data design

## 12.1 Required annotations
For each clip and each frame:

- RGB frame
- depth map
- visible masks per entity
- solo frame per entity if possible
- camera parameters if available
- entity identity labels
- overlap ratio
- front/back switches through time

## 12.2 Dataset splits

### Split O — layered occlusion
Strong front/back depth gaps.

### Split C — contact / near-depth collision
This is the hardest case and must be overrepresented.

### Split R — reappearance
An entity is hidden for multiple frames and then returns.

### Split X — transfer stress
New categories, new shapes, new camera motions.

## 12.3 Sampling schedule
Recommended train mix:

- 35% contact / near-depth collision
- 25% layered occlusion
- 20% reappearance
- 20% transfer stress

---

# 13. Training plan

## Stage 0 — dataset validation
Before training anything, compute oracle stats.

### Must compute
- visible mask coverage distribution
- amodal-visible hidden fraction distribution
- overlap histogram
- depth-gap histogram
- GT object count per frame
- same-depth vs layered breakdown

This is mandatory because many previous problems came from metrics that did not match the data assumptions.

---

## Stage 1 — train scene prior only
Freeze all backbones.

### Inputs
- prompt entities
- time index
- optional trajectory seed

### Outputs
- visible e0/e1
- amodal e0/e1
- sep map
- depth map

### Losses

```math
L_{scene} = \lambda_{vis}L_{vis} + \lambda_{amo}L_{amo} + \lambda_{id}L_{id} + \lambda_{temp}L_{temp} + \lambda_{occ}L_{occ}
```

### Goal
The scene prior alone should become accurate and stable.

---

## Stage 2 — train structured decoder
Freeze scene prior.

### Input
Scene outputs / scene features.

### Output
- coarse composite RGB
- optional isolated RGB e0/e1

### Losses
- reconstruction loss
- perceptual loss
- isolated consistency loss
- object count preservation loss

### Goal
Prove that the structured representation contains enough information to reconstruct plausible images.

If this stage fails, do **not** move on to diffusion. Fix the scene prior first.

---

## Stage 3 — adapter + backbone training
Freeze most of scene prior, optionally fine-tune lightly.

### Losses
- standard diffusion denoising loss
- image reconstruction against decoder output
- object count / visible survival penalties
- isolated/composite consistency

### Goal
Teach the backbone to use the structured scene representation.

---

## Stage 4 — transfer training / evaluation
Take the same frozen scene prior and train a second adapter for another backbone.

This stage is the real test of whether the structural prior generalized.

---

# 14. Loss functions in detail

## 14.1 Visible loss

```python
import torch


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(-2, -1))
    denom = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()
```

Use for each entity visible map.

## 14.2 Amodal loss
Same as visible, but against amodal GT or pseudo amodal targets.

## 14.3 Identity contrastive loss

```python
import torch
import torch.nn.functional as F


def identity_contrastive(h1: torch.Tensor, h2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor, margin: float = 0.5):
    pos = F.mse_loss(h1, z1) + F.mse_loss(h2, z2)
    neg = torch.relu(margin - (h1 - h2).norm(dim=-1)).mean()
    return pos + neg
```

## 14.4 Reappearance consistency
A hidden entity must come back as the same entity.

```python
def reappearance_loss(hidden_hist: torch.Tensor, reappeared: torch.Tensor):
    return ((hidden_hist - reappeared) ** 2).mean()
```

## 14.5 Object count survival loss
If both entities exist in GT, both must keep nonzero visible or amodal evidence.

```python
def visible_survival_loss(v0: torch.Tensor, v1: torch.Tensor, tau: float = 0.02):
    l0 = torch.relu(tau - v0.mean(dim=(-2, -1))).mean()
    l1 = torch.relu(tau - v1.mean(dim=(-2, -1))).mean()
    return l0 + l1
```

---

# 15. Inference pipeline

## 15.1 High-level steps

1. parse prompt into entities and interaction,
2. sample trajectory seed,
3. generate scene outputs for all frames,
4. decode coarse RGB,
5. pass scene features and coarse RGB into a backbone-specific refiner,
6. optionally run verification / repair.

## 15.2 Pseudocode

```python
@torch.no_grad()
def infer_video(prompt: str, scene_prior, scene_encoder, decoder, adapter, backbone, T: int = 8):
    parsed = parse_prompt(prompt)
    # user-level simplification: exactly two entities
    e0, e1 = parsed.entities[0], parsed.entities[1]

    frames = []
    memory0 = None
    memory1 = None

    for t in range(T):
        t_scalar = torch.tensor([[t / max(T - 1, 1)]], device="cuda")
        # placeholder identity embeddings
        z0 = torch.randn(1, 128, device="cuda")
        z1 = torch.randn(1, 128, device="cuda")

        scene_outputs = scene_prior(z0, z1, t_scalar, memory0, memory1)
        scene_features = scene_encoder(scene_outputs)
        coarse_rgb = decoder(scene_features)
        guides = adapter.build_guides(scene_features)
        refined = backbone.refine(coarse_rgb, prompt, guides)
        frames.append(refined)

    return frames
```

---

# 16. Evaluation protocol

## 16.1 Scene-prior metrics
These must be measured before involving the backbone.

- visible IoU e0/e1
- amodal IoU e0/e1
- hidden fraction accuracy
- reappearance consistency
- slot swap rate
- contact-frame separation accuracy

## 16.2 Decoder metrics
- composite RGB PSNR / LPIPS
- isolated RGB LPIPS
- crop-level semantic similarity
- object count recognition

## 16.3 Backbone-transfer metrics
For each backbone:
- two-object detection rate
- chimera rate
- isolated/composite consistency
- visible survival min
- reappearance identity accuracy

A scene prior is only considered portable if it improves at least two different backbones.

---

# 17. Checkpoint selection rules

Do **not** select checkpoints by a mixed contract score.

## New checkpoint rule
Keep `best_scene.pt` using:
1. `min(vis_iou_e0, vis_iou_e1)`
2. `min(amo_iou_e0, amo_iou_e1)`
3. `visible_survival_min`
4. `slot_swap_rate`

Keep `best_decoder.pt` using:
1. composite LPIPS
2. isolated LPIPS
3. object count preservation

Keep `best_backbone_{name}.pt` using:
1. chimera rate
2. two-object detection rate
3. isolated/composite consistency
4. visible survival min

---

# 18. Minimum viable implementation plan

## Week 1: scene prior extraction layer
- create `scene_prior/renderer.py`
- create `scene_prior/scene_outputs.py`
- refactor current projector logic into renderer
- remove guide-family dependency from mainline

## Week 2: structured decoder
- create `backbones/reconstruction_decoder.py`
- train on current synthetic dataset
- confirm scene outputs can reconstruct meaningful coarse RGB

## Week 3: adapter split
- move current injection code into `adapters/animatediff_adapter.py`
- keep hook/gate logic from current conditioning implementation fileciteturn14file0

## Week 4: transfer benchmark
- freeze scene prior
- train second adapter for a second backbone
- compare to baseline without scene prior

---

# 19. Specific repository changes

## 19.1 Deprecation list

Mark the following as deprecated for mainline Phase 64:

- guide-family branching experiments in main path
- contract-based best checkpoint selection
- any path where the scene representation is built directly from backbone feature maps rather than scene modules

## 19.2 Migration list

### Migrate from `models/phase62/projection.py`
- transmittance logic
- visible/amodal computation
- temperature sharpening only if still needed for current density parameterization fileciteturn15file0

### Migrate from `models/phase62/conditioning.py`
- gate-after-normalization idea
- multiscale hook manager
- block registry and projector infrastructure fileciteturn14file0

### Migrate from `scripts/build_volume_gt.py`
- depth/mask/order-derived scene supervision construction fileciteturn16file0

---

# 20. Risks and mitigations

## Risk 1: the scene prior still overfits the toy dataset
### Mitigation
- hold out categories,
- hold out contact geometries,
- hold out camera motions,
- hold out appearance styles,
- measure scene-prior metrics separately from generator metrics.

## Risk 2: structured decoder works, backbone still ignores the scene prior
### Mitigation
- make the adapter stronger,
- add reconstruction-conditioned diffusion,
- force isolated/composite consistency,
- reduce backbone freedom by starting from coarse RGB.

## Risk 3: the scene prior is too weak to reconstruct coarse RGB
### Mitigation
- improve field resolution,
- add per-entity appearance field,
- add trajectory memory,
- increase temporal supervision.

## Risk 4: cross-backbone transfer is poor
### Mitigation
- confirm scene outputs are good first,
- keep adapter shallow,
- compare zero-shot adapter, low-shot adapter, and full adapter.

---

# 21. Why this is realistically implementable

This plan is not asking the repository to become a full NeRF system or a full 3D simulator. It is realistic because:

1. the repository already contains transmittance-based rendering logic, fileciteturn15file0
2. it already contains multiscale conditioned injection logic, fileciteturn14file0
3. it already uses rendered synthetic depth/masks as supervision, fileciteturn16file0
4. the main missing step is *modularization*, not invention from scratch.

The transition path is incremental:

- first isolate the scene prior,
- then prove the representation is image-reconstructable,
- then prove it transfers across backbones.

---

# 22. Final recommendation

The most important architectural decision is this:

> **Stop thinking of the method as “a better guide for a specific diffusion backbone.” Start thinking of it as “a portable scene prior that can supervise or condition many renderers.”**

That means the project should now be optimized in the following order:

1. **Scene decomposition quality**
2. **Structured-to-image decodability**
3. **Backbone transferability**
4. **Final perceptual quality**

Not the reverse.

If this ordering is respected, then the system can eventually reach the actual goal:

> a backbone-independent structural prior that first builds an entity-centric 3D scene and then conditions any downstream generator with reduced chimera.

---

# 23. Immediate actionable checklist

## Keep
- transmittance renderer logic fileciteturn15file0
- gate-after-normalization multiscale injection fileciteturn14file0
- rendered depth/mask/order GT philosophy fileciteturn16file0

## Delete or deprecate
- guide-family branching in mainline
- contract-heavy checkpointing
- scene representation that depends directly on backbone features

## Add now
- `scene_prior/renderer.py`
- `scene_prior/entity_field.py`
- `scene_prior/temporal_memory.py`
- `adapters/base_adapter.py`
- `backbones/reconstruction_decoder.py`
- `training/stage1_train_scene_prior.py`
- `training/stage2_train_decoder.py`
- `training/stage4_transfer_eval.py`

## Prove before claiming success
- scene prior works without backbone,
- structured decoder reconstructs images from scene outputs,
- same scene prior transfers to a second backbone,
- chimera decreases on both.

