# VCA Experiment Log (Phase 22+)

Phase 1~21 기록은 `docs/failures.md` 참조.

---

## Phase 22 — VCA 주입 레이어 up_blocks.2 이동 + text attention 고해상도 캡처

**날짜**: 2026-04-10  
**스크립트**: `scripts/train_phase22.py`  
**데이터**: `toy/data_objaverse` (180 samples, bad asset 제거 완료)

### 배경 (Phase 21 문제점)

| 항목 | Phase 21 | Phase 22 |
|------|----------|----------|
| VCA 주입 레이어 | `mid_block` (4×4=16 spatial) | `up_blocks.2` (16×16=256 spatial) |
| query_dim | 1280 | 640 |
| text_attn 캡처 레이어 | `mid_block` (4×4) | `up_blocks.3` (32×32=1024 spatial) |

**Phase 21 문제**: `mid_block`은 4×4=16 spatial token. 두 entity가 같은 quadrant에
있으면 같은 token에 묶임 → "cat" 어텐션이 dog 위치로 가는 binding problem 발생.
sigma map이 entity를 공간적으로 분리하지 못함.

**Phase 22 수정**:
- VCA를 `up_blocks.2`(16×16)에 주입 → sigma map 해상도 16배 향상
- text_attn 캡처를 `up_blocks.3`(32×32)에서 → 라텐트 동일 해상도, 픽셀 수준 매핑

### 하이퍼파라미터

```
INJECT_KEY       = up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor
ATTN_CAPTURE_KEY = up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor
INJECT_QUERY_DIM = 640
VCA_ALPHA        = 0.3
lambda_depth     = 0.3 (adaptive)
lr               = 5e-5
epochs           = 60
depth_pe_init_scale = 0.3
```

### 결과

| Epoch | DRA | probe_sep | 비고 |
|-------|-----|-----------|------|
| 0 | - | - | |
| 5 | - | - | |
| ... | | | |

### 주요 관찰

- [ ] text_attn.gif: cat/dog 어텐션이 각자 올바른 entity 위치로 가는지
- [ ] depth_effect.gif: VCA가 실제로 depth 변화를 만드는지
- [ ] multiangle_depth_chart.png: angle별 depth ordering 정확도
- [ ] DRA > 80% 달성 여부

### 결론

> (학습 완료 후 작성)

---

## Phase 23 — Color-qualified 프롬프트 + Direct z-order loss

**날짜**: 2026-04-10  
**스크립트**: `scripts/train_phase23.py`  
**데이터**: `toy/data_objaverse` (180 samples)

### 배경 (Phase 22 문제점)

| 항목 | Phase 22 | Phase 23 |
|------|----------|----------|
| 텍스트 프롬프트 | "a cat", "a dog" (색상 없음) | "a red cat", "a blue dog" (색+객체) |
| Depth loss | `l_depth_ranking_perframe` (ranking, 간접) | `l_zorder_direct` (z-bin 직접 할당) |
| text_attn 시각화 | 2 토큰 (entity0, entity1) | 4 토큰 (color0, entity0, color1, entity1) |

**Phase 22 문제**: text_attn이 cat/dog를 구분 못함 → 색 없는 텍스트로는 CLIP 임베딩이 비슷함.  
ranking loss는 front>back 비교만 → z-bin 자체 할당 안 배움.

**Phase 23 수정**:
1. `make_color_prompts(meta)`: meta.json의 color0/color1 RGB → "a red cat and a blue dog"
2. `get_color_entity_context`: color+entity 결합 CLIP 임베딩 (1, 2, 768)
3. `l_zorder_direct`: σ(front,z=0)↑ + σ(back,z=-1)↑ + σ(front,z=-1)↓ + σ(back,z=0)↓
4. `debug_text_attn` 5행: GT + color0 + entity0 + color1 + entity1 overlay

### 하이퍼파라미터

```
INJECT_KEY       = up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor
ATTN_CAPTURE_KEY = up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor
INJECT_QUERY_DIM = 640
VCA_ALPHA        = 0.3
lambda_depth     = 0.3 (adaptive)
lr               = 5e-5
epochs           = 60
depth_pe_init_scale = 0.3
```

### 결과

| Epoch | DRA | probe_sep | 비고 |
|-------|-----|-----------|------|
| 0 | - | - | |
| 5 | - | - | |
| ... | | | |

### 주요 관찰

- [ ] text_attn.gif: color+entity 4토큰이 각자 올바른 entity 위치로 가는지
- [ ] depth_effect.gif: l_zorder_direct가 실제로 z-bin 분리를 만드는지
- [ ] multiangle_depth_chart.png: angle별 depth ordering 정확도
- [ ] DRA > 85% 달성 여부

### 결론

> (학습 완료 후 작성)

---

## Phase 24 — Depth-focused: lambda_depth 증가 + ratio fix (GPU 0)

**날짜**: 2026-04-10  
**스크립트**: `scripts/train_phase24.py`

### 배경 (Phase 23 문제)
- `ratio=73M`: l_depth가 negative → `max(l_depth_w, 1e-9)=1e-9` → ratio 폭발, adaptive lambda 오작동
- `lambda_depth=0.3`이 `l_diff=0.07`에 완전히 묻혀 depth 신호 없음

### 수정
| 항목 | Phase 23 | Phase 24 |
|------|----------|----------|
| lambda_depth | 0.3 | 5.0 (17x) |
| lambda_diff | 1.0 | 0.05 (20x 감소) |
| ratio 계산 | max(l_depth_w, 1e-9) | max(abs(l_depth_w), 1e-9) |
| adaptive 조건 | l_depth_w > 0 | abs(l_depth_w) > 1e-8 |

### 결과

| Epoch | DRA | probe_sep | ratio | 비고 |
|-------|-----|-----------|-------|------|
| ... | | | | |

---

## Phase 25 — Text attention supervision: TrainableAttnProcessor (GPU 1)

**날짜**: 2026-04-10  
**스크립트**: `scripts/train_phase25.py`

### 배경
- text_attn_chart: cross-attention frozen → gradient 없음 → chart 개선 불가
- Phase 24 depth fix도 포함

### 수정
1. `TrainableAttnProcessor` @ ATTN_CAPTURE_KEY: weight tensor with grad
2. `l_attn_mask_loss`: entity 토큰 attention mass → GT mask 내 최대화
3. unfreeze attn2.to_q + to_k @ up_blocks.3 (lr * 0.1로 보수적 fine-tune)
4. trainable: VCA 843K + attn2 to_q/to_k

### 결과

| Epoch | DRA | text_attn_overlap | probe_sep | 비고 |
|-------|-----|-------------------|-----------|------|
| ... | | | | |

---

## Phase 26 — Text attention 강한 supervision: lambda_attn=30.0 + clip=0.5 (GPU 0)

**날짜**: 2026-04-10  
**스크립트**: `scripts/train_phase26.py`

### 배경 (Phase 25 문제)

| 항목 | Phase 25 | Phase 26 |
|------|----------|----------|
| lambda_attn | 0.3 | 3.0 (10x) |
| attn grad clip | 0.01 | 0.1 (10x 완화) |
| attn lr | args.lr × 0.1 | args.lr (full) |

**Phase 25 문제**: `l_attn`이 14 epoch 동안 -0.010~-0.011에 완전 고착.
- entity mask coverage ≈ 1% of 1024 spatial tokens → weighted l_attn = -0.003 (depth의 5000배 약함)
- gradient too weak → to_q/to_k 실질적으로 학습 안됨

**시도 실패 (lambda_attn=30.0)**:
- epoch 1에서 l_attn=-11.0, l_depth=-0.68 → attn이 16배 dominant
- probe_sep: 0.036 → 0.010 → 0.001 (depth 파괴)
- DRA: 0.412 (random 이하) → 즉시 중단

**Phase 26 수정 (lambda_attn=3.0)**:
- l_attn 기여: 3.0 × 0.034 = 0.10 vs l_depth 기여: 5.0 × 0.6 = 3.0 → depth 30배 dominant, 균형 유지
- clip=0.1 (10x), full lr: 충분한 attn 학습 허용
- 목표: DRA >80% 유지 + l_attn 개선

### 결과

| Epoch | DRA | l_attn | text_attn_overlap | 비고 |
|-------|-----|--------|-------------------|------|
| ... | | | | |

---

## Phase 27 — Text attention LoRA: fp32 rank-4 (GPU 0)

**날짜**: 2026-04-10  
**스크립트**: `scripts/train_phase27.py`

### 배경 (Phase 25/26 문제)

| 항목 | Phase 25 | Phase 26 | Phase 27 |
|------|----------|----------|----------|
| 방법 | fp16 to_q/to_k 직접 학습 | 동일 | fp32 LoRA rank-4 |
| lambda_attn | 0.3 | 3.0 | 3.0 |
| 결과 | l_attn 고착 (-0.010) | epoch 1 NaN | ? |

**Phase 25/26 근본 문제**: fp16 파라미터를 AdamW로 학습하면 gradient overflow → NaN
- Phase 25: gradient too weak (lambda too small)
- Phase 26: NaN from epoch 1 (fp16 weight update 불안정)

**Phase 27 LoRA 해결**:
- `LoRALayer`: fp32 A×B matrices (rank=4, 6,912 params total)
- 원본 to_q/to_k 완전 frozen → dtype 이슈 없음
- `LoRAAttnProcessor`: q += delta_q, k += delta_k (fp32→fp16 cast)
- `torch.amp.autocast(enabled=False)` 사용 → autocast가 fp32 연산 방해 안 함

### 설정
- lambda_depth = 5.0, lambda_attn = 3.0
- LoRA rank = 4, q_dim=320, context_dim=768
- clip_grad_norm_(trainable_attn, 0.5), full lr

### 핵심 버그 수정

**NaN 원인 1 — `torch.baddbmm` beta=1 uninitialized**:
```python
# 잘못됨: beta=1 (default) → torch.empty 미초기화값 + score → NaN
scores = torch.baddbmm(torch.empty(...), q.float(), k.float().T, alpha=scale)

# 수정됨: beta=0
scores = torch.baddbmm(torch.empty(...), q.float(), k.float().T, beta=0, alpha=scale)
```

**NaN 원인 2 — l_attn gradient UNet 역전파**:
```python
# 수정됨: hs_det / ctx_det detach → LoRA에만 gradient
hs_det  = hidden_states.detach()
ctx_det = ctx.detach()
q = attn.to_q(hs_det) + lora.delta_q(hs_det)
k = attn.to_k(ctx_det) + lora.delta_k(ctx_det)
```

### 결과

| Epoch | DRA | l_attn | probe_sep | 비고 |
|-------|-----|--------|-----------|------|
| 9 | 0.637 | -1.75 | 0.19 | l_attn 학습 시작 |
| 24 | 0.794 | -2.49 | 0.18 | DRA 80% 근접 |
| 29 | **0.806** | -2.54 | 0.16 | **DRA >80% 달성** (Phase 25보다 30 epoch 빠름) |
| 54 | 0.800 | -2.67 | 0.25 | 안정 유지 |
| 59 | **0.806** | -2.66 | 0.26 | **FINAL** |

**FINAL: probe_sep=0.285, DRA=0.8350 (334/400)**

### 결론

- **depth chart**: DRA=83.5% > 80% 목표 달성. Phase 25(83.0%)와 동등하나 30 epoch 빠름.
- **text attention chart**: l_attn Phase 25 고착(-0.010) → Phase 27 지속 학습(-2.66). LoRA fp32 gradient isolation으로 NaN 완전 해결.
- **두 가지 버그 수정**: (1) `beta=0` in baddbmm, (2) `hidden_states.detach()` for Q,K input.
- IDEA=WORKS, LEARNING=OK

---

## Phase 28 — Visualization fixes: LoRA attn + masked depth chart

**날짜**: 2026-04-10  
**스크립트**: `scripts/train_phase28.py`

### 배경 (Phase 27 시각화 버그 2가지)

| 버그 | 원인 | 현상 | 수정 |
|------|------|------|------|
| text attn 완전 파괴 | `CaptureAttnProcessor.baddbmm` beta=1 + `torch.empty` → fp16 쓰레기값 | attention map 완전 noise | `beta=0` 추가 |
| text attn base model 표시 | CaptureAttnProcessor가 LoRA delta 없는 base attention 표시 | LoRA 효과 시각화 불가 | LoRAAttnProcessor.last_weights 직접 사용 |
| depth chart 평행이동 | `_sigma_depth_scores`가 ALL 256 spatial token 평균 (배경 포함) | entity signal 희석 → 두 entity 동일하게 이동 | entity_masks로 masking (DRA 측정방식과 동일) |
| depth encoder_hs 버그 | 모든 angle이 angle[0]의 text prompt 사용 | hidden states 불일치 | 각 angle 자체 meta 사용 |

### 설정
Phase 27과 동일 (lambda_depth=5.0, lambda_diff=0.05, lambda_attn=3.0, LoRA rank=4)

### 결과

| Epoch | DRA | l_attn | probe_sep | 비고 |
|-------|-----|--------|-----------|------|
| 9 | 0.637 | -1.75 | 0.19 | |
| 24 | 0.794 | -2.49 | 0.18 | |
| 29 | 0.806 | -2.54 | 0.16 | DRA >80% |
| 59 | **0.806** | -2.66 | 0.26 | **FINAL** |

**FINAL: probe_sep=0.285, DRA=0.8350 (334/400) — Phase 27과 동일 (학습 로직 변경 없음)**

### 결론

- 학습 결과: Phase 27과 완전히 동일 (visualization fix만)
- **text_attn_chart.png**: LoRA-modified attention 표시 → entity별 분리 확인 필요
- **multiangle_depth_chart.png**: entity_masks masking → crossing 패턴 확인 필요
- debug 파일: `debug/train_phase28/epoch_059/` 참조

---

## 실험 설계 가이드

### 새 Phase 추가 시 체크리스트

1. `scripts/train_phase{N}.py` 생성 (이전 phase 복사 후 수정)
2. `pytest.ini`에 phase{N} marker 추가
3. `docs/experiment_log.md` 에 Phase {N} 섹션 추가
4. 학습 실행: `CUDA_VISIBLE_DEVICES=1 python scripts/train_phase{N}.py ...`
5. debug GIF 확인 후 결과 기록

### 핵심 지표 의미

| 지표 | 의미 | 목표 |
|------|------|------|
| `depth_rank_accuracy` | sigma가 실제 depth 순서를 맞추는 비율 | > 80% |
| `probe_sep` | 고정 probe에서 두 entity sigma 분리도 | > 0.1 |
| `ratio` | l_diff / l_depth_weighted | 1~30x (adaptive) |
| text_attn | cat/dog 어텐션이 각자 영역으로 분리 | 육안 확인 |
