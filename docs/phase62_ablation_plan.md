# Phase 62 — Ablation Plan

## Core Principle
- Stage 1: **one structural loss family only**
- Stage 2: **diffusion binding only**

## Objective Families

### 1. `independent_bce` (current baseline)
$$\mathcal{L} = \sum_{n=0}^{1} \text{BCE}(z_n, Y_n)$$

### 2. `factorized_fg_id` (recommended)
$$p_{\text{fg}}(x) = \sigma(z_{\text{fg}}(x))$$
$$q_n(x) = \text{softmax}(z_{\text{id}}(x))_n$$
$$p_n(x) = p_{\text{fg}}(x) \cdot q_n(x)$$
$$\mathcal{L} = \mathcal{L}_{\text{fg}} + \lambda_{\text{id}} \mathcal{L}_{\text{id}}$$

### 3. `projected_visible_only`
$$V_n(h,w) = \sum_k T_k(h,w) \cdot p_n(k,h,w)$$
$$\mathcal{L} = \sum_n (1 - \text{Dice}(V_n, V_n^{gt}))$$

### 4. `projected_amodal_only`
$$A_n(h,w) = 1 - \prod_k (1 - p_n(k,h,w))$$
$$\mathcal{L} = \sum_n (1 - \text{Dice}(A_n, A_n^{gt}))$$

### 5. `center_offset`
$$\mathcal{L} = \sum_{x \in \Omega_n} \|x + o_n(x) - c_n\|_1$$

## Guide Families
- `none`: G = 0
- `front_only`: G = [V_0⊙F_0, V_1⊙F_1]
- `dual`: G = [F_front, F_back] (mixed)
- `four_stream`: G = [V_0⊙F_0, V_1⊙F_1, (A_0-V_0)⊙F_0, (A_1-V_1)⊙F_1]

## Training Schedules
- **S0**: volume_only (no diffusion ever)
- **S1**: stage1=volume, stage2=diffusion only (volume frozen)
- **S2**: stage1=volume, stage2=volume low LR + diffusion
- **S3**: stage1 → stage2 → stage3 short joint

## Ablation Matrix

| ID  | Objective           | Schedule | Guide        | Priority |
|-----|---------------------|----------|--------------|----------|
| B0  | factorized_fg_id    | S0       | none         | 1        |
| B1  | factorized_fg_id    | S1       | front_only   | 2        |
| B2  | factorized_fg_id    | S1       | four_stream  | 3        |
| A0  | independent_bce     | S0       | none         | 4        |
| C0  | projected_amodal    | S0       | none         | 5        |
| A1  | independent_bce     | S1       | front_only   | 6        |
| B3  | factorized_fg_id    | S2       | four_stream  | 7        |
| C1  | projected_visible   | S0       | none         | 8        |
| D0  | center_offset       | S0       | none         | 9        |

## Evaluation Criteria
- `min_visible_dice = min(D_0^vis, D_1^vis)` — both entities visible
- `min_amodal_dice = min(D_0^amo, D_1^amo)` — full body coverage
- `iou_e0, iou_e1, iou_min` — projected class IoU
- `acc_entity` — voxel-level entity accuracy

## Running
```bash
# Priority list (B0, B1, B2, A0, C0)
python scripts/run_phase62_ablations.py --priority

# Single ablation
python scripts/run_phase62_ablations.py --config config/phase62/ablations/b0_fgid_volume_only.yaml

# All 9
python scripts/run_phase62_ablations.py --all
```

## Current Empirical Takeaway
- The current summary at `outputs/phase62_ablations/summary.json` shows `b2_fgid_freeze_bind_fourstream` as the best factorized run so far.
- It is better than the old all-background collapse regime, but it still exhibits asymmetric drift in later epochs.
- The practical mainline baseline remains `independent_bce` until the factorized id branch is re-tested with the corrected dual-entity input path.
- Structural losses that directly optimize projected visible masks remain ablation-only; they are not the current mainline because they tend to suppress entity mass early.
