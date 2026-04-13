# Phase 61: Depth-Layered Volume Diffusion

## Goal
Real object identity preservation under collision.
Two entities (cat, dog) stay visually distinct even when overlapping.

---

## File Map

```
models/phase61_layered_volume.py   # Model: DepthVolumeHead, VolumeCompositor, Phase61Processor
models/phase61_losses.py           # Losses: composite, alpha_volume, visible_ownership, depth_expected
scripts/train_phase61.py           # Training loop: 3-stage, collision augmentation, CFG rollout eval
```

---

## Architecture Overview

```
                    AnimateDiff UNet (shared trunk)
                              │
                    ┌─────────┼─────────┐
                    │         │         │
              up_blocks.1  up_blocks.2  up_blocks.3
              (1280-dim)   (640-dim)    (320-dim)
              non-primary  PRIMARY      non-primary
                    │         │         │
                    │    ┌────┴────┐    │
                    │    │         │    │
                    │  F_0(cat)  F_1(dog)  ← masked attention + slot adapters
                    │    │         │    │
                    │    ▼         ▼    │
                    │  DepthVolumeHead  │   ← PRIMARY ONLY
                    │    │         │    │
                    │  alpha0_bins(B,S,K)  alpha1_bins(B,S,K)
                    │  feat0_bins(B,S,K,D) feat1_bins(B,S,K,D)
                    │    │         │    │
                    │    └────┬────┘    │
                    │         │         │
                    │  VolumeCompositor  │
                    │         │         │
                    │   composed(B,S,D) │
                    │   w0_bins, w1_bins │  ← rendering weights
                    │   w_bg            │
                    │         │         │
                    └────┬────┴────┬────┘
                         │         │
                    non-primary blocks
                    reuse primary routing
                    (interpolated to their resolution)
```

---

## Key Classes

### `DepthVolumeHead` (models/phase61_layered_volume.py:53)
Predicts per-depth-bin occupancy + feature delta from entity features.

**Input**: `feat (B, S, D)` — entity-specific cross-attention features
**Output**:
- `alpha_logits (B, S, K)` — raw logits (NOT sigmoid), zero-init
- `feat_bins (B, S, K, D)` — base feat + learned delta per bin

```python
# Zero-init → alpha starts at logit=0, feat_delta starts at 0
# At init: sigmoid(0) = 0.5 everywhere, softmax(0,0,0) = 1/3 each
```

### `VolumeCompositor` (models/phase61_layered_volume.py:117)
NeRF-style front-to-back rendering with per-bin softmax competition.

**Key innovation**: At each depth bin z, entities COMPETE via softmax:
```
logits = stack([logit_e0(z), logit_e1(z), 0_bg])  # (B, S, K, 3)
p0(z), p1(z), p_bg(z) = softmax(logits, dim=-1)

alpha_total(z) = p0(z) + p1(z)  = 1 - p_bg(z)
T(z) = prod_{k<z} p_bg(k)       # transmittance
w0(z) = T(z) * p0(z)            # entity 0 rendering weight
w1(z) = T(z) * p1(z)            # entity 1 rendering weight
w_bg = prod_all p_bg(z)         # background weight

composed = sum_z(w0(z)*feat0(z) + w1(z)*feat1(z)) + w_bg * F_g
```

**Guarantees**: w0.sum() + w1.sum() + w_bg = 1.0 exactly (partition of unity).

### `Phase61Processor` (models/phase61_layered_volume.py:204)
Cross-attention processor injected into UNet.

**All blocks get**:
- Shared LoRA on K, V, Out (`SlotLoRA`)
- Slot adapters for entity-masked attention (`SlotAdapter`)

**Primary block (up_blocks.2, inner_dim=640) additionally gets**:
- `DepthVolumeHead` for e0 and e1
- `VolumeCompositor`
- Stores `alpha_bins, feat_bins, w_bins, w_bg` for loss computation

**Non-primary blocks**: Reuse primary's visible routing weights (interpolated to their spatial resolution).

### `Phase61Manager` (models/phase61_layered_volume.py:435)
Manages all processors. Key methods:
- `set_entity_tokens(toks_e0, toks_e1)` — tell all blocks which text tokens belong to which entity
- `reset()` — clear stored predictions
- `propagate_routing()` — copy primary's routing to non-primary blocks
- `volume_predictions` — property returning all primary block outputs

---

## Loss Functions (models/phase61_losses.py)

| Loss | Formula | Purpose |
|------|---------|---------|
| `loss_composite` | MSE(noise_pred, noise_gt) | Overall scene quality |
| `loss_alpha_volume` | BCE_with_logits(alpha_logits, bin_targets) × valid_mask | Per-bin occupancy |
| `loss_visible_ownership` | BCE(w_bins.sum(K), visible_mask) | Visible region correctness |
| `loss_depth_expected` | relu(d_front - d_back + margin) in overlap | Front/back ordering |

### GT Target Construction: `build_depth_bin_targets()`
For K=2 bins (bin 0 = front, bin 1 = back):
- **Exclusive pixel** (only entity 0): `alpha0_target[bin=0] = 1`
- **Overlap pixel** (both present, entity 0 in front): `alpha0_target[bin=0] = 1, alpha1_target[bin=1] = 1`
- Returns valid masks indicating which bin positions have supervision

---

## Training (scripts/train_phase61.py)

### 3 Stages
| Stage | Epochs | Trainable | Focus |
|-------|--------|-----------|-------|
| A | 0-7 | Volume heads + adapters | Ownership bootstrap |
| B | 8-15 | + Shared LoRA K/V/Out | Feature separation |
| C | 16-24 | All params | Joint fine-tuning |

### Per-Step Flow
```python
# 1. Get entity token positions (with keyword fallback)
toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)

# 2. Encode paired frames → latents → add noise
latents = encode_frames_to_latents(pipe, frames_np, device)
noise = torch.randn_like(latents)
noisy = pipe.scheduler.add_noise(latents, noise, t)

# 3. Single UNet forward (Phase61Processors run inside)
noise_pred = pipe.unet(noisy, t, encoder_hidden_states=enc_full).sample

# 4. Propagate primary routing to non-primary blocks
manager.propagate_routing()

# 5. Read volume predictions
alpha0_bins, alpha1_bins, ..., w0_bins, w1_bins, w_bg = manager.volume_predictions

# 6. Build per-bin GT targets from entity_masks + depth_orders
tgt0, tgt1, valid0, valid1 = build_depth_bin_targets(masks_feat, depth_orders, K)

# 7. Compute 4 losses
l_comp  = loss_composite(noise_pred, noise)
l_alpha = loss_alpha_volume(alpha0_bins, alpha1_bins, tgt0, tgt1, valid0, valid1)
l_own   = loss_visible_ownership(w0_bins, w1_bins, visible_masks)
l_depth = loss_depth_expected(alpha0_bins, alpha1_bins, depth_orders, masks)

# 8. Total loss (weights vary by stage)
loss = la_comp*l_comp + la_alpha*l_alpha + la_own*l_own + la_depth*l_depth
```

### Validation / Eval
- **val_score**: 0.45*IoU + 0.25*comp + 0.15*alpha + 0.15*own (IoU-weighted)
- **Composite rollout**: CFG-enabled (guidance=7.5), 20 steps, decoded to GIF
- **Overlay GIF**: ownership maps on generated frames
- **Collision augmentation**: shifts entity masks to create overlap during training

---

## Inference / Rollout

```python
# CFG-enabled generation loop
pipe.scheduler.set_timesteps(n_steps, device=device)
for step_t in pipe.scheduler.timesteps:
    manager.reset()
    # CFG: run UNet twice (uncond + cond), blend with guidance_scale
    lat2 = cat([latents, latents])
    enc2 = cat([enc_uncond, enc_cond])
    pred = pipe.unet(lat2, step_t, encoder_hidden_states=enc2).sample
    uncond_p, cond_p = pred.chunk(2)
    noise_pred = uncond_p + guidance_scale * (cond_p - uncond_p)
    latents = pipe.scheduler.step(noise_pred, step_t, latents)

# Decode to frames
for fi in range(n_frames):
    decoded = pipe.vae.decode((latents_4d[fi] / scale).half()).sample
    frame = ((decoded / 2 + 0.5).clamp(0, 1) * 255).uint8
```

---

## Dataset

`ObjaverseDatasetPhase40('toy/data_objaverse', n_frames=8)` returns:
| Index | Name | Shape | Description |
|-------|------|-------|-------------|
| 0 | frames | (T, 256, 256, 3) uint8 | RGB video frames |
| 1 | depth | (T, 256, 256) float32 | Per-frame depth maps |
| 2 | depth_orders | list[(front, back)] | Per-frame front/back entity index |
| 3 | meta | dict | keyword0, keyword1, prompt_full, colors, etc. |
| 4 | entity_masks | (T, 2, 256) bool | Flattened 16x16 binary entity masks |
| 5 | visible_masks | (T, 2, 256) float32 | Visibility-weighted masks |
| 6 | solo_e0 | (T, 256, 256, 3) uint8 | Solo render of entity 0 |
| 7 | solo_e1 | (T, 256, 256, 3) uint8 | Solo render of entity 1 |

---

## Key Dependencies

```python
from models.entity_slot import SlotAdapter              # Adapter for entity features
from models.entity_slot_phase40 import SlotLoRA         # LoRA for K/V/Out
from scripts.train_animatediff_vca import encode_frames_to_latents
from scripts.train_phase35 import get_entity_token_positions
from scripts.generate_solo_renders import ObjaverseDatasetPhase40
from scripts.run_animatediff import load_pipeline
```

## Checkpoint Format
```python
{
    "epoch": int,
    "stage": "A" | "B" | "C",
    "val_score": float,
    "inject_keys": List[str],
    "procs_state": [
        {
            "lora_k": state_dict,
            "lora_v": state_dict,
            "lora_out": state_dict,
            "slot0_adapter": state_dict,
            "slot1_adapter": state_dict,
            # PRIMARY only:
            "e0_volume": state_dict,
            "e1_volume": state_dict,
        }
        for each injected block
    ],
}
```
