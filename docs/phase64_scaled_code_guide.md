# Phase 64 Scaled — 학습 및 추론 코드 가이드

> 작성일: 2026-04-16  
> 현재 활성 파이프라인: **Phase 64 Scaled (p64s)**  
> GPU: CUDA_VISIBLE_DEVICES=0

---

## 1. 전체 구조 한눈에 보기

```
Phase64 파이프라인 = 4단계 순차 학습

Stage 0  → 데이터셋 검증 (1회)
Stage 1  → Scene Prior 학습          → checkpoints/p64s_stage1/best_scene_prior.pt
Stage 2  → Structured Decoder 학습   → checkpoints/p64s_stage2/best_decoder.pt
Stage 3  → AnimateDiff Adapter 학습  → checkpoints/p64s_stage3_animatediff/stage3_epoch0060.pt
Stage 4  → SDXL Transfer 검증        → checkpoints/p64s_stage4_sdxl/sdxl_adapter.pt
```

**핵심 아이디어**: Scene Prior(Stage 1)는 백본에 무관한 표현을 학습.  
동일한 8채널 SceneOutputs를 AnimateDiff(Stage 3)와 SDXL(Stage 4)에 각각 주입해  
backbone-agnostic 분리가 실제로 동작함을 증명한다.

---

## 2. 전체 파이프라인 실행 (원커맨드)

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase64_scaled_pipeline.sh
```

로그는 `logs/phase64_scaled/` 아래 스테이지별로 분리 저장됨.

---

## 3. 스테이지별 학습 코드

### Stage 0: 데이터셋 검증

```bash
# Stage 1 스크립트가 내부적으로 Stage 0을 먼저 실행하므로 별도 실행 불필요
# 단독 실행 시:
python training/phase64/stage0_validate_dataset.py \
    --config config/phase64_scaled/stage1.yaml
```

- **코드**: `training/phase64/stage0_validate_dataset.py`
- **역할**: `compute_and_save_stats()` — 가시성/은닉 분율, 겹침 분포, O/C/R/X 분할 수 통계
- **출력**: `outputs/phase64_scaled/stage0/dataset_stats.json`, `histograms.png`

---

### Stage 1: Scene Prior 학습 (핵심 단계)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_phase64_scene.py \
    --config config/phase64_scaled/stage1.yaml
```

- **엔트리**: `scripts/train_phase64_scene.py`
- **트레이너**: `training/phase64/stage1_train_scene_prior.py` → `Stage1Trainer`
- **모델**: `scene_prior/entity_field.py` → `ScenePriorModule`
- **설정**: `config/phase64_scaled/stage1.yaml`
- **체크포인트**: `checkpoints/p64s_stage1/best_scene_prior.pt`

#### 모델 구조 (`ScenePriorModule`)

```
RGB frame (B, 3, H, W)
    │
    ▼
ImageContextEncoder        ← stride-2 conv 3단계, 백본 없음
(B, ctx_dim=128, H/8, W/8)
    │
    ├──────────────────────────────────────────┐
    │                                          │
EntityEmbedding(name)      MotionModel         │
(B, id_dim=256)            (B, pose_dim=64)    │
    │                          │               │
    └──────────┬───────────────┘               │
               ▼                               │
   BackboneAgnosticFieldDecoder (entity 0)     │
   BackboneAgnosticFieldDecoder (entity 1)     │  routing_hint (color affinity map)
               │                               │
               └──────── density fields ───────┘
                         (B, depth_bins=16, spatial_h=64, spatial_w=64)
                                  │
                                  ▼
                           EntityRenderer         ← transmittance 기반
                                  │
                                  ▼
                          SceneOutputs (B, 8, 64, 64)
```

#### SceneOutputs 8채널 구성

| 채널 | 이름 | 의미 |
|------|------|------|
| 0 | `visible_e0` | entity 0의 first-hit 가시 기여 |
| 1 | `visible_e1` | entity 1의 first-hit 가시 기여 |
| 2 | `amodal_e0` | entity 0의 전체 점유 (폐색 무관) |
| 3 | `amodal_e1` | entity 1의 전체 점유 (폐색 무관) |
| 4 | `depth_map` | 기대 깊이 [0, 1] |
| 5 | `sep_map` | 분리 신호 (visible_e0 − visible_e1) |
| 6 | `hidden_e0` | entity 0의 폐색된 부분 (amodal − visible) |
| 7 | `hidden_e1` | entity 1의 폐색된 부분 |

#### 손실 함수

```python
total_loss = (
    lambda_vis   * (loss_vis_e0 + loss_vis_e1)   # 2.0  — IoU: 예측 vs GT 가시 마스크
  + lambda_amo   * (loss_amo_e0 + loss_amo_e1)   # 1.0  — BCE: amodal vs GT amodal
  + lambda_occ   * loss_occ                      # 0.5  — amodal ≥ visible 일관성
  + lambda_surv  * loss_surv                     # 10.0 — 두 entity 모두 생존 보장
  + lambda_color * loss_color_routing            # 3.0  — sep_map ↔ 색상 affinity 정합
  + lambda_sep   * loss_sep                      # 0.2  — sep_map 부호 일관성
)
```

#### 옵티마이저 (3개 파라미터 그룹)

| 그룹 | 학습률 | 대상 |
|------|--------|------|
| `field_params()` | 5e-4 | entity decoder (BackboneAgnosticFieldDecoder) |
| `encoder_params()` | 2e-4 | context encoder, entity embedding, motion model |
| `memory_params()` | 1e-4 | temporal slot memory (GRU) |

#### 스케일업 설정 (toy → scaled)

| 파라미터 | toy | scaled |
|----------|-----|--------|
| hidden_dim | 64 | 128 |
| depth_bins | 8 | 16 |
| spatial_h/w | 32 | 64 |
| id_dim | 128 | 256 |
| pose_dim | 32 | 64 |
| slot_dim | 128 | 256 |
| epochs | 60 | 200 |
| steps/epoch | 30 | 100 |

---

### Stage 2: Structured Decoder 학습

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_phase64_decoder.py \
    --config config/phase64_scaled/stage2.yaml \
    --stage1_ckpt checkpoints/p64s_stage1/best_scene_prior.pt
```

- **코드**: `training/phase64/stage2_train_decoder.py` → `Stage2Trainer`, `StructuredDecoder`
- **역할**: SceneOutputs(8ch) → RGB(3ch) 재건. 확산 모델 없이 씬 표현이 충분한지 검증
- **모델**: `conv → GELU → conv → GELU → conv → GELU → conv(3ch) → Sigmoid`
- **손실**: L1 + L2 재건 손실 + isolation 손실 (두 entity 모두 기여 강제)
- **게이트**: PSNR 기준 미달 시 Stage 3 진행 중단

---

### Stage 3: AnimateDiff Adapter 학습

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_phase64_backbone.py \
    --config config/phase64_scaled/stage3.yaml \
    --stage 3 \
    --stage1_ckpt checkpoints/p64s_stage1/best_scene_prior.pt \
    --stage2_ckpt checkpoints/p64s_stage2/best_decoder.pt
```

- **코드**: `training/phase64/stage3_train_adapter_backbone.py`
- **어댑터**: `adapters/animatediff_adapter.py` → `AnimateDiffAdapter`
- **가이드 인코더**: `adapters/guide_encoders.py` → `SceneGuideEncoder`
- **백본**: `emilianJR/epiCRealism` + `guoyww/animatediff-motion-adapter-v1-5-3`

#### 주입 방식 (forward hook)

```
SceneOutputs (8ch)
    │
    ▼
SceneGuideEncoder          ← 8ch → hidden=128
    │
    ▼
AnimateDiffAdapter
    ├─ up1_proj (1280d) ─── hook → UNet up1 블록 출력에 더하기
    ├─ up2_proj (640d)  ─── hook → UNet up2 블록 출력에 더하기
    └─ up3_proj (320d)  ─── hook → UNet up3 블록 출력에 더하기
```

- Gate 초기화: zeros (학습 초기 백본 동작 보존)
- amplitude normalization: `guide_max_ratio=0.15` 클리핑

#### 손실

```python
total_loss = (
    lambda_diff     * diffusion_mse_loss    # 1.0 — SDEdit 스타일 노이즈 예측 MSE
  + lambda_surv     * survival_loss         # 2.0 — entity 생존
  + lambda_contrast * contrast_loss         # 0.3 — entity 분리 대조
)
```

---

### Stage 4: SDXL Transfer 검증

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_phase64_backbone.py \
    --config config/phase64_scaled/stage4.yaml \
    --stage 4 \
    --stage1_ckpt checkpoints/p64s_stage1/best_scene_prior.pt
```

- **코드**: `training/phase64/stage4_transfer_eval.py`
- **어댑터**: `adapters/sdxl_adapter.py` → `SDXLAdapter` (Stage 3과 동일 구조, 새로 초기화)
- **백본**: `stabilityai/sdxl-turbo`
- **핵심**: Stage 1 체크포인트 그대로 재사용. 어댑터만 새로 학습 → backbone-agnostic 증명

---

## 4. 추론 / 시각화 코드

### Scene Prior 추론 (Stage 1 결과 시각화)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/viz_phase64.py \
    --stage1_ckpt checkpoints/p64s_stage1/best_scene_prior.pt \
    --out outputs/phase64_scaled/viz \
    --n_samples 35
```

- **코드**: `scripts/viz_phase64.py`
- **출력 구조** (`outputs/phase64_scaled/viz/sample_XXXX_<entity0>_<entity1>/`):
  - `scene_maps.png` — 가로 스트립: `[input | vis_e0 | vis_e1 | amo_e0 | amo_e1 | depth | sep]`
  - `visible_e0.png`, `visible_e1.png` — entity별 가시 히트맵
  - `amodal_e0.png`, `amodal_e1.png` — entity별 amodal 히트맵
  - `overlay.png` — GT 마스크를 입력 프레임에 오버레이
  - `overview.gif` — 모든 샘플 순회 GIF

### Transfer 평가

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_phase64_transfer.py \
    --stage1_ckpt checkpoints/p64s_stage1/best_scene_prior.pt \
    --adapter_ckpt checkpoints/p64s_stage4_sdxl/sdxl_adapter.pt
```

- **출력**: `results/phase64_transfer/metrics.json` — O/C/R/X 분할별 메트릭

### SceneOutputs raw 내보내기

```bash
python scripts/export_scene_outputs.py \
    --stage1_ckpt checkpoints/p64s_stage1/best_scene_prior.pt \
    --out outputs/phase64_scaled/exports
```

---

## 5. 평가 메트릭

**코드**: `training/phase64/evaluator_phase64.py` → `Phase64Evaluator`

| 메트릭 | 의미 | 목표 |
|--------|------|------|
| `val_visible_iou_e0/e1` | entity별 가시 마스크 IoU | ↑ |
| `val_visible_iou_min` | 두 entity 중 낮은 쪽 IoU | ↑ (bottleneck 지표) |
| `val_amodal_iou_min` | 두 entity 중 낮은 amodal IoU | ↑ |
| `val_slot_swap_rate` | entity 0/1 라벨 뒤집힘 빈도 | ↓ (0 목표) |
| `val_visible_survival_min` | 두 entity 모두 프레임에 생존하는 비율 | ↑ |
| `val_contact_separation_accuracy` | 접촉/분리 장면 분류 정확도 | ↑ |
| `val_entity_balance` | entity 0/1 예측 면적 균형 | → 0.5 |

**현재 결과** (p64s_stage1, ep=200):
```
vis_iou_e0 = 0.435
vis_iou_e1 = 0.348
vis_iou_min = 0.348   ← bottleneck
amo_iou_min = 0.349
slot_swap_rate = 0.0  ✓ (entity 혼동 없음)
```

---

## 6. 데이터 파이프라인

```
toy/data_objaverse_full/
    <entity0>_<entity1>_<motion>_<view>/
        frames/     0000.png … 0015.png
        depth/      0000.npy … 0015.npy
        mask/       0000_entity0.png, 0000_entity1.png …
        video.mp4
```

- **Dataset 클래스**: `data/phase64/phase64_dataset.py` → `Phase64Dataset`
- **GT 빌드**: `data/phase64/build_scene_gt.py` → `SceneGT` (vis/amodal 마스크, depth, 분할 타입)
- **Routing map**: entity 색상(빨강/파랑) 기반 pixel-wise affinity → entity decoder의 공간 힌트
- **분할**: train 80% / val 20%, 추가로 O/C/R/X 타입별 stratified split

---

## 7. 핵심 파일 목록

### Scene Prior
| 파일 | 역할 |
|------|------|
| `scene_prior/entity_field.py` | ScenePriorModule 전체 아키텍처 |
| `scene_prior/scene_outputs.py` | SceneOutputs 8채널 dataclass |
| `scene_prior/renderer.py` | transmittance 렌더러 |
| `scene_prior/temporal_memory.py` | TemporalSlotMemory (GRU) |
| `scene_prior/losses.py` | 전체 손실 라이브러리 |
| `scene_prior/motion_model.py` | MotionModel (pose 예측) |

### 학습
| 파일 | 역할 |
|------|------|
| `training/phase64/stage1_train_scene_prior.py` | Stage 1 Trainer |
| `training/phase64/stage2_train_decoder.py` | Stage 2 Decoder + Trainer |
| `training/phase64/stage3_train_adapter_backbone.py` | Stage 3 AnimateDiff Trainer |
| `training/phase64/stage4_transfer_eval.py` | Stage 4 SDXL Transfer |
| `training/phase64/evaluator_phase64.py` | 종합 평가 메트릭 |

### 어댑터 / 백본
| 파일 | 역할 |
|------|------|
| `adapters/guide_encoders.py` | SceneGuideEncoder (8ch → guide feat) |
| `adapters/animatediff_adapter.py` | Stage 3 UNet hook 주입 |
| `adapters/sdxl_adapter.py` | Stage 4 SDXL hook 주입 |
| `backbones/animatediff_refiner.py` | AnimateDiff 래퍼 |
| `backbones/reconstruction_decoder.py` | StructuredDecoder |

### 스크립트
| 파일 | 역할 |
|------|------|
| `scripts/run_phase64_scaled_pipeline.sh` | 전체 파이프라인 오케스트레이터 |
| `scripts/train_phase64_scene.py` | Stage 0+1 엔트리 |
| `scripts/train_phase64_decoder.py` | Stage 2 엔트리 |
| `scripts/train_phase64_backbone.py` | Stage 3/4 엔트리 |
| `scripts/viz_phase64.py` | 시각화 |
| `scripts/eval_phase64_transfer.py` | Transfer 평가 |

### 설정
| 파일 | 역할 |
|------|------|
| `config/phase64_scaled/stage1.yaml` | Scene Prior 하이퍼파라미터 |
| `config/phase64_scaled/stage2.yaml` | Decoder 하이퍼파라미터 |
| `config/phase64_scaled/stage3.yaml` | AnimateDiff 하이퍼파라미터 |
| `config/phase64_scaled/stage4.yaml` | SDXL 하이퍼파라미터 |
