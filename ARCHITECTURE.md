# Phase 62: 3D Entity Volume Generator + First-Hit Projection

## Goal
Real object identity preservation under collision.
No soft blending. No transparency. No alpha compositing.
One pixel = one entity. Period.

---

## What Changed from Phase 60/61

| Phase 60/61 (REMOVED) | Phase 62 (NEW) |
|------------------------|----------------|
| Scalar alpha per entity | 3D volume logits (B, N+1, K, H, W) |
| Porter-Duff / transmittance | First-hit projection (argmax scan) |
| Soft ownership BCE | Volume cross-entropy (one loss) |
| 8+ auxiliary losses | 2 losses only: L_diffusion + L_volume_ce |
| Primary-block-only control | Multi-scale injection (mid/up2/multiscale) |
| Heuristic depth ordering | Learned 3D topology |

---

## File Map

```
models/phase62_entity_volume.py    # EntityVolumePredictor: 3D conv volume head
models/phase62_projection.py       # FirstHitProjector: argmax scan over depth
models/phase62_conditioning.py     # VolumeGuidedProcessor: UNet injection
models/phase62_losses.py           # 2 losses: diffusion MSE + volume CE
scripts/train_phase62.py           # Training loop
scripts/build_volume_gt.py         # V_gt from 3D rendered data
harness/test_phase62_volume.py     # Volume shape/softmax tests
harness/test_phase62_projection.py # First-hit tests
harness/test_phase62_forward.py    # End-to-end smoke test
```

---

## Architecture

```
Dataset (3D rendered) ──► build_volume_gt.py ──► V_gt (B, K, H, W) class indices
                                                   │
AnimateDiff UNet                                   │
     │                                             │
     ├─► Entity features F_0, F_1, F_g             │
     │        │                                    │
     │   EntityVolumePredictor                     │
     │        │                                    │
     │   V_logits (B, N+1, K, H, W)               │
     │        │                         L_volume_ce(V_logits, V_gt)
     │   softmax over class dim ──────────────────►│
     │        │                                    │
     │   FirstHitProjector                         │
     │        │                                    │
     │   visible_class_map (B, H, W)               │
     │   visible_features (B, D, H, W)             │
     │        │                                    │
     │   VolumeGuidedProcessor                     │
     │        │ (inject at mid / up2 / multiscale) │
     │        ▼                                    │
     ├─► noise_pred ───────────────────► L_diffusion(noise_pred, noise_gt)
     │
     ▼
  output
```

---

## Key Design Decisions

### 1. Volume Representation
- `V_logits`: (B, N+1, K, H, W) where N=2 entities, K=8 depth bins
- **Voxel-wise softmax** over class dimension (dim=1): bg vs cat vs dog compete
- **NOT** independent sigmoid — mutual exclusion enforced
- 3D conv layers for depth-axis continuity

### 2. V_gt Construction
- Built from **actual 3D rendered data**: per-entity depth maps + binary masks
- Each voxel gets a class label: 0=bg, 1=entity0, 2=entity1
- Depth bins derived from actual rendered depth values, not heuristic
- Fallback (2D mask + depth_order) exists but is NOT the default path

### 3. First-Hit Projection
- Scan depth axis k=0..K-1 (front to back)
- First non-background class wins the pixel
- **No max pooling, no mean, no weighted average, no transparency**
- Training: Gumbel-softmax or straight-through for differentiability
- Inference: hard argmax

### 4. UNet Injection
- Projected 2D guide conditions the UNet via spatial addition
- Configurable injection points:
  - `mid_only`: mid_block only
  - `mid_up2`: mid_block + up_blocks.2
  - `multiscale`: mid + up_blocks.1 + up_blocks.2 + up_blocks.3
- Each injection point gets a lightweight projection layer

### 5. Losses (exactly 2)
- `L_diffusion`: MSE(noise_pred, noise_gt)
- `L_volume_ce`: CrossEntropy(V_logits, V_gt) with class weighting

### 6. Topology Update Schedule
- `hybrid`: compute volume at step 0, refine at steps T//3 and 2T//3, fix rest
- `fixed_once`: compute once at step 0, fix for all steps
- `every_step`: recompute every step (expensive, for ablation only)

---

## Forbidden (from Phase 60/61)
- scalar alpha/depth heads
- Porter-Duff ownership
- transmittance/cumprod rendering
- ownership BCE / depth expected / leak / temporal / solo / divergence losses
- stage2 inpainting
- soft blending at any level

---

## Evaluation
- **Primary**: composite GIF visual inspection (cat ≠ dog, no chimera)
- **Quantitative**: volume accuracy, projected class IoU vs visible mask
- **Collision**: overlap frames evaluated separately
- **Best checkpoint**: selected by projected_class_iou, NOT by soft CE alone
