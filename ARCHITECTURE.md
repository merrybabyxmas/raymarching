# Phase 62: 3D Entity Volume Generator + First-Hit Projection

## Goal
Real object identity preservation under collision.
No soft blending. No transparency. No alpha compositing.
One pixel = one entity. Period.

---

## What Changed from Phase 60/61

| Phase 60/61 (REMOVED) | Phase 62 (NEW) |
|------------------------|----------------|
| Scalar alpha per entity | 3D volume logits / factorized fg-id volume |
| Porter-Duff / transmittance | First-hit projection (argmax scan) |
| Soft ownership BCE | Mainline volume BCE + ablation-only structural losses |
| Primary-block-only control | Multi-scale injection (mid/up2/multiscale) |
| Heuristic depth ordering | Learned 3D topology |

---

## File Map

```
models/phase62_entity_volume.py    # EntityVolumePredictor: independent / factorized / center-offset heads
models/phase62_projection.py       # FirstHitProjector: depth scan + amodal/visible projections
models/phase62_conditioning.py     # GuideFeatureAssembler + UNet injection hooks
training/phase62/objectives/       # Objective families: independent_bce, factorized_fg_id, projected_* , center_offset
training/phase62/ablation_trainer.py# Config-driven stage schedule + ablation runner
scripts/train_phase62.py           # Main entry (delegates to v2 path)
scripts/run_phase62_ablations.py   # Ablation matrix runner
scripts/build_volume_gt.py         # V_gt from rendered depth / masks
harness/test_phase62_volume.py     # Volume shape / projection smoke tests
harness/test_phase62_projection.py  # First-hit tests
harness/test_phase62_forward.py    # End-to-end smoke test
```

---

## Architecture

```
Dataset (rendered depth + masks) ─► build_volume_gt.py ─► V_gt / gt_visible / gt_amodal
                                              │
AnimateDiff UNet ─► BackboneFeatureExtractor ─► F_g, F_0, F_1
                                              │
                                              ├─► EntityVolumePredictor
                                              │      └─► V_logits / entity_probs
                                              │
                                              ├─► FirstHitProjector
                                              │      └─► visible_class / front_probs / back_probs
                                              │
                                              ├─► GuideFeatureAssembler
                                              │      └─► guide tensors per injection block
                                              │
                                              └─► GuideInjectionManager -> UNet hooks
                                                      └─► noise_pred
                                                            └─► L_diffusion
```

---

## Key Design Decisions

### 1. Volume Representation
- `V_logits`: (B, N+1, K, H, W) where N=2 entities, K=8 depth bins
- `independent_bce` is the current stable baseline; `factorized_fg_id` is the main ablation family
- `factorized_fg_id` uses `p_fg * q_id` factorization to reduce all-bg collapse
- 3D conv layers provide depth-axis continuity bias

### 2. V_gt Construction
- Built from rendered depth + masks in `data.phase62`
- Each voxel gets a class label: 0=bg, 1=entity0, 2=entity1
- `VolumeGTBuilder` caches rendered voxel targets per sample
- Fallback (2D mask + depth_order) exists but is not the default path

### 3. First-Hit Projection
- Scan depth axis k=0..K-1 (front to back)
- First non-background class wins the pixel
- Produces `visible_class`, `front_probs`, `back_probs`, plus `amodal` / `visible` maps for ablations
- **No max pooling, no mean, no weighted average, no transparency**
- Training uses straight-through style gradients where needed; inference is hard argmax

### 4. UNet Injection
- Projected 2D guide conditions the UNet via spatial addition
- Configurable injection points:
  - `mid_only`: mid_block only
  - `mid_up2`: mid_block + up_blocks.2
  - `multiscale`: mid + up_blocks.1 + up_blocks.2 + up_blocks.3
- Each injection point gets a lightweight projection layer

### 5. Loss Families
- Mainline baseline: `L_diffusion` + `L_volume_ce`
- `L_volume_ce` currently runs as the stable baseline in `independent_bce`
- Ablation-only structural losses are tracked separately in `training/phase62/objectives/`
- `loss_visible_dice`, `loss_amodal_dice`, `loss_voxel_exclusive` are experimental, not default

### 6. Topology Update Schedule
- `S0`: volume-only
- `S1`: stage1 volume, stage2 diffusion-only binding
- `S2`: stage1 volume, stage2 low-LR volume + diffusion
- `S3`: short joint fine-tune after stage2

---

## Forbidden (from Phase 60/61)
- scalar alpha/depth heads
- Porter-Duff ownership
- transmittance/cumprod rendering
- ownership BCE / depth expected / leak / temporal / solo / divergence losses
- stage2 inpainting
- soft blending at any level

---

## Current Results Snapshot
- `outputs/phase62_ablations/summary.json` records the latest ablation pass.
- Best current factorized run: `b2_fgid_freeze_bind_fourstream`
- It improves entity voxels relative to the dead-all-bg regime, but still shows asymmetric drift and `iou_min` collapse in late epochs.
- Practical takeaway: `independent_bce` remains the conservative baseline; `factorized_fg_id` is the current main ablation to retest after the id-branch fix.

## Evaluation
- **Primary**: composite GIF visual inspection (cat ≠ dog, no chimera)
- **Quantitative**: volume accuracy, projected class IoU vs visible mask
- **Collision**: overlap frames evaluated separately
- **Best checkpoint**: selected by projected_class_iou and `iou_min`, not by soft CE alone
