# Raymarching 프로젝트 전면 재설계 제안서

## 목표: **두 개의 독립 entity가 뒹굴고 가려져도 최종 2D 비디오에서 chimera 없이 identity가 유지되는 생성 시스템**

이 문서는 **현재 코드에서 과감히 삭제할 부분**, **반드시 유지할 부분**, **본질적으로 새로 추가해야 할 부분**, 그리고 **데이터셋 / 학습 / 추론 / 평가 / 실험 세팅**까지 한 번에 재설계한 제안서다.
목표는 하나다.

> **“cat + dog rolling together” 같은 동적 shot에서, 각 객체가 접촉·가림·재등장을 반복해도 최종 composite video가 두 개의 독립 객체로 보이게 만들기.**

---

# 0. 최종 결론 먼저

현재 코드베이스는 “3D-aware 구조”를 일부 갖추고 있지만, 아직은 **representation을 잘 배우는 시스템**에 더 가깝고, **two-entity dynamic rendering을 강제하는 시스템**은 아니다.
특히 현재 구현은 이미

* transmittance 기반 visible/amodal projection
* four-stream guide
* guide gate gradient fix
* V_gt from rendered depth/masks

를 갖고 있으므로, 아이디어의 핵심은 살아 있다. `GuideFeatureAssembler`는 `front_only`, `dual`, `four_stream` guide family를 지원하고, `four_stream`은 visible/amodal 분리를 이미 사용한다. 
또 `FirstHitProjector`는 현재 실제로는 transmittance를 사용해 `visible`, `amodal`, `front_probs`, `back_probs`를 계산하고 있다. 
그리고 GT volume은 rendered depth + entity masks + depth order로부터 구성된다. 

하지만 지금 시스템의 핵심 병목은 다음이다.

1. **entity 표현이 여전히 경쟁적이다**
   현재 projector는 entity probability를 기반으로 visible winner를 만들고, guide는 그 visible/amodal 신호를 backbone feature와 곱해 UNet에 주입한다. 이 과정에서 **dominant entity가 guide scale까지 더 크게 가져가는 양성 피드백**이 생기기 쉽다. `conditioning.py`의 `four_stream`이 바로 그 핵심 경로다. 

2. **학습 목표가 최종 사용자 목표를 직접 강제하지 않는다**
   현재 trainer는 diffusion loss, structure loss, feature separation, spatial coherence, fg prior, permutation consistency, temporal centroid consistency 등 다양한 surrogate를 최적화한다. 그러나 이 조합은 **“최종 composite에서 두 entity가 동시에 독립적으로 보이는가”**를 가장 강하게 강제하지 않는다. 

3. **현재 volume은 진짜 object-centric field보다 “projection에 유리한 sparse evidence”로 학습되기 쉽다**
   지금 구조는 최종적으로 `visible_class` 및 guide 품질을 좋게 만들려는 방향으로 수렴할 수 있고, 이 경우 실제 3D objectness보다 **가시성에 유리한 점상/희박한 구조**로도 높은 점수를 얻는다.

따라서 이 문서의 제안은 다음 한 줄로 요약된다.

> **현재의 “class volume + guide injection” 시스템을, “entity-centric dynamic field + amodal/visible dual rendering + structured video refinement” 시스템으로 재구성해야 한다.**

---

# 1. 무엇을 유지하고, 무엇을 삭제하고, 무엇을 추가할 것인가

## 1.1 유지할 것

### A. **Rendered depth/mask 기반 GT 생성 파이프라인**

이 부분은 매우 중요하다.
`scripts/build_volume_gt.py`는 depth map, entity masks, depth order로부터 V_gt를 만든다. overlap에서는 front entity를 더 가까운 bin, back entity를 더 먼 bin에 배치한다. 이건 **학습 가능한 3D supervision**의 출발점으로 매우 좋다. 

**유지 이유**

* synthetic / rendered scene의 기하 정보를 잘 활용함
* 2D segmentation만 쓰는 것보다 훨씬 낫다
* collision/occlusion에 대한 supervision이 가능하다

---

### B. **Transmittance 기반 projector의 큰 철학**

현재 `models/phase62/projection.py`는 temperature-scaled entity probabilities로 occupancy를 만들고, transmittance를 누적하여 `front_probs`, `back_probs`, `visible`, `amodal`을 계산한다. 

이건 유지해야 한다.
다만 “current class-volume version”을 그대로 유지하는 게 아니라, **entity-centric density field renderer**로 일반화해서 유지해야 한다.

**유지 이유**

* 네 원래 아이디어와 가장 잘 맞음
* visible/amodal 분리를 자연스럽게 제공
* contact/occlusion shot에서 필수적임

---

### C. **UNet에 structured condition을 주입하는 방식**

현재 `GuideFeatureAssembler` + `GuideInjectionManager` 구조는 매우 유용하다. block별 projector, multiscale injection, gate after normalization 등은 그대로 살릴 가치가 크다. 

**유지 이유**

* diffusion backbone을 직접 갈아엎지 않아도 됨
* structured scene signal을 비디오 생성에 주입할 수 있음
* engineering cost 대비 효과가 큼

---

## 1.2 과감히 삭제할 것

### A. **legacy / 미사용 파일**

업로드된 교정 메모 기준으로, 실제 실행 경로가 아닌 옛 파일은 정리해야 한다.
특히 `models/phase62_conditioning.py` 같은 구 버전/비실행 파일은 제거 대상이다. 

**삭제 이유**

* 혼동 유발
* 분석/디버깅 효율 저하
* 잘못된 patch가 들어갈 위험 증가

---

### B. **guide family 다중 분기 (`none`, `front_only`, `dual`)**

현재 `GuideFeatureAssembler`는 `none`, `front_only`, `dual`, `four_stream`을 지원한다. 
이건 연구 실험용으로는 좋았지만, 이제는 아키텍처를 고정해야 할 시점이다.

**삭제/비활성화 권장**

* `none`: baseline용만 남기고 mainline에서 제거
* `front_only`: 명백히 정보 부족
* `dual`: entity 분리가 약하고 두 객체가 섞여 들어가기 쉬움

**남길 것**

* `four_stream` 철학을 일반화한 **entity-centric multi-stream guide**

  * visible_e0
  * visible_e1
  * hidden/amodal_e0
  * hidden/amodal_e1
  * optional depth/order cue

즉 **guide family를 하나로 고정**해라.

---

### C. **bg-class 중심 voxel softmax 세계관**

초기 분석에서 보였듯, 예전 `EntityVolumePredictor`는 `(bg, e0, e1)` class softmax 기반이었고 bg bias도 강했다. 
현재 실실행 아키텍처는 이보다 진화했지만, 여전히 설계의 뿌리가 “class assignment”에 가깝다.

이걸 근본적으로 버려야 한다.

**삭제 이유**

* bg vs entity 경쟁, entity0 vs entity1 경쟁이 동시에 생김
* same-depth/contact에서 한쪽 승자독식이 쉬움
* “존재”와 “가시성”을 같은 축에서 다루게 됨

**대체**

* 각 entity는 **독립 density field**
* background는 entity absence로 정의
* visible은 ray compositing으로 나중에 계산

---

### D. **현재 contract-driven selection 로직**

현재 best checkpoint selection은 contract score와 val score의 혼합이다. `_save_checkpoint()`가 그 흐름을 쓴다. 
이건 이제 버려야 한다.

**삭제 이유**

* 현재 contract는 최종 사용자 목표와 완전히 일치하지 않음
* visible e0/e1가 한쪽 죽어도 통과하는 구조가 있었음
* “잘 최적화된 잘못된 해법”을 best로 뽑을 수 있음

---

## 1.3 본질적으로 추가할 것

### A. **Entity-centric dynamic field**

핵심 추가다.

현재는 “scene → class volume → projection”에 가깝다.
앞으로는 다음이어야 한다.

[
\text{entity } i \text{ at time } t
;\rightarrow;
f_{i,t}(\mathbf{p}) = (\sigma_{i,t}(\mathbf{p}), a_{i,t}(\mathbf{p}), e_i)
]

여기서

* (\sigma_{i,t}): density / occupancy
* (a_{i,t}): appearance feature
* (e_i): time-invariant identity embedding

즉 object 2개를 **각각 독립된 3D field**로 유지해야 한다.

---

### B. **Amodal identity memory**

contact 중 가려져도 entity가 사라지지 않게 하려면,
각 entity에 대한 latent memory가 있어야 한다.

[
m_i \leftarrow \text{persistent slot for entity } i
]

이 memory가

* frame (t)의 field 생성
* frame (t+1)의 continuation
* occlusion 뒤 재등장

을 연결한다.

---

### C. **Temporal slot consistency**

현재 trainer에는 permutation consistency와 centroid consistency가 있지만, rolling/contact shot에는 약하다. 

추가해야 할 것은:

* slot-level contrastive temporal consistency
* overlap 전/중/후에서 동일 entity matching
* hidden entity가 reappear했을 때 same slot 유지

---

### D. **Render-level direct supervision**

surrogate metric이 아니라, 진짜 최종 목표를 directly 잡는 손실이 필요하다.

예:

* composite frame에서 object count 유지
* per-entity visible survival
* isolated-composite consistency
* reappearance consistency

---

# 2. 새 아키텍처 제안

## 2.1 전체 구조

다음 5개 모듈로 나눈다.

1. **Entity Motion / Pose Module**
2. **Entity Field Module**
3. **Differentiable Renderer**
4. **Structured Guide Encoder**
5. **Video Refiner (Diffusion UNet)**

---

## 2.2 수식으로 본 전체 파이프라인

### 입력

* text prompt (y)
* entity identity tokens (e_1, e_2)
* time indices (t = 1,\dots,T)

### Step 1. motion / pose rollout

각 entity의 time-varying pose를 생성:

[
g_{i,t} = \text{MotionNet}(e_i, y, t)
]

---

### Step 2. per-entity field 생성

각 entity의 3D field:

[
f_{i,t}(\mathbf{p}) = \text{FieldDecoder}(e_i, g_{i,t}, \mathbf{p})
]

출력은
[
(\sigma_{i,t}(\mathbf{p}), a_{i,t}(\mathbf{p}))
]

---

### Step 3. differentiable rendering

카메라 ray (r(u,\lambda))에 대해

[
\sigma_{i,t}(u,\lambda) = \sigma_{i,t}(r(u,\lambda))
]

전체 occlusion은 entity별 density의 합으로 정의:

[
\Sigma_t(u,\lambda)=\sum_i \sigma_{i,t}(u,\lambda)
]

transmittance:

[
T_t(u,\lambda)=\exp\left(-\int_0^\lambda \Sigma_t(u,s)ds\right)
]

entity (i)의 visible contribution:

[
V_{i,t}(u)=\int T_t(u,\lambda)\sigma_{i,t}(u,\lambda)d\lambda
]

entity (i)의 amodal presence:

[
A_{i,t}(u)=1-\exp\left(-\int \sigma_{i,t}(u,\lambda)d\lambda\right)
]

depth cue:

[
D_t(u)=\frac{\int \lambda \sum_i T_t(u,\lambda)\sigma_{i,t}(u,\lambda)d\lambda}
{\int \sum_i T_t(u,\lambda)\sigma_{i,t}(u,\lambda)d\lambda+\epsilon}
]

---

### Step 4. guide encoding

Structured guide:

[
G_t = \text{GuideEncoder}(V_{1,t},V_{2,t},A_{1,t},A_{2,t},D_t)
]

---

### Step 5. diffusion refinement

최종 frame 생성:

[
\hat{x}_t = \text{DiffusionUNet}(z_t, y, G_t)
]

여기서 diffusion은 **scene를 만들지 않고 appearance를 refine**하는 역할만 한다.

---

# 3. 새 loss 설계

## 3.1 Rendering loss

최종 composite reconstruction:

[
\mathcal{L}_{render}
====================

\sum_t |\hat{x}_t - x_t|_1
]

또는 perceptual loss 포함.

---

## 3.2 Visible supervision

GT visible masks (M^{vis}_{i,t}):

[
\mathcal{L}_{vis}
=================

\sum_{i,t}\mathrm{Dice}(V_{i,t}, M^{vis}_{i,t})
]

---

## 3.3 Amodal supervision

GT or pseudo amodal masks (M^{amo}_{i,t}):

[
\mathcal{L}_{amo}
=================

\sum_{i,t}\mathrm{Dice}(A_{i,t}, M^{amo}_{i,t})
]

이게 있어야 가려진 entity가 latent에서 사라지지 않는다.

---

## 3.4 Identity separation

entity pooled feature (h_{i,t}):

[
h_{i,t} = \mathrm{Pool}(a_{i,t}, \sigma_{i,t})
]

contrastive / triplet loss:

[
\mathcal{L}_{id}
================

\sum_t
|h_{1,t} - e_1|^2
+
|h_{2,t} - e_2|^2
+
\max(0, m - |h_{1,t} - h_{2,t}|)
]

---

## 3.5 Temporal slot consistency

같은 entity는 time에서 같은 slot 유지:

[
\mathcal{L}_{temp}
==================

\sum_{i,t}
| h_{i,t+1} - \mathcal{W}(h_{i,t}) |
]

혹은 correspondence-based contrastive loss.

---

## 3.6 Occlusion consistency

visible과 amodal 관계 강제:

[
V_{i,t}(u) \le A_{i,t}(u)
]

[
\mathcal{L}_{occ}
=================

\sum_{i,t,u}
\max(0, V_{i,t}(u)-A_{i,t}(u))
]

또는 hidden fraction regularizer.

---

## 3.7 Isolation consistency

entity (i)만 남긴 isolated render (\hat{x}^{iso}*{i,t})를 만들고,
solo frame (x^{solo}*{i,t}) 또는 pseudo-label과 비교:

[
\mathcal{L}_{iso}
=================

\sum_{i,t}
|\hat{x}^{iso}*{i,t} - x^{solo}*{i,t}|
]

현재 trainer에도 isolated diffusion loss가 있지만, structured renderer 기준으로 더 직접적으로 재설계해야 한다. 

---

## 3.8 Total loss

[
\mathcal{L}
===========

\lambda_{render}\mathcal{L}*{render}
+
\lambda*{vis}\mathcal{L}*{vis}
+
\lambda*{amo}\mathcal{L}*{amo}
+
\lambda*{id}\mathcal{L}*{id}
+
\lambda*{temp}\mathcal{L}*{temp}
+
\lambda*{occ}\mathcal{L}*{occ}
+
\lambda*{iso}\mathcal{L}_{iso}
]

---

# 4. 데이터셋 설계

## 4.1 반드시 필요한 GT

각 frame (t)마다 최소한 다음이 필요하다.

* RGB frame (x_t)
* entity visible mask (M^{vis}*{1,t}, M^{vis}*{2,t})
* depth map (d_t)
* camera parameters
* entity identity labels
* 가능하면 solo render of each entity
* 가능하면 amodal mask

현재 repo는 depth map, entity masks, depth order로부터 V_gt를 만들 수 있다. 이건 그대로 살린다. 

---

## 4.2 데이터 split

### Split A: clean layered occlusion

* 앞/뒤 depth 분리 명확
* overlap 있음
* identity 유지가 쉬운 케이스

### Split B: same-depth / near-depth contact

* rolling/contact 핵심 데이터
* hardest regime

### Split C: temporal reappearance

* 한 entity가 거의 사라졌다가 다시 등장
* memory/temporal consistency 테스트

### Split D: style variation

* cat, dog, similar silhouettes, texture confound

---

## 4.3 dataset generation 원칙

1. **solo frames를 반드시 저장**

   * isolated loss와 evaluation에 필수

2. **camera pose와 object pose를 저장**

   * temporal warp / correspondence에 필요

3. **collision difficulty를 메타데이터로 저장**

   * overlap ratio
   * depth gap
   * front/back switches
   * contact duration

---

## 4.4 추천 데이터 구성 비율

* 30% clean layered
* 30% near-depth contact
* 25% hard same-depth rolling
* 15% reappearance / partial occlusion stress

---

# 5. 학습 단계 설계

## Stage 0. Dataset validation

먼저 oracle metric부터 계산한다.

* GT visible IoU upper bound
* GT amodal consistency
* overlap statistics
* same-depth difficulty histogram

이 단계 없이 threshold를 세우면 또 같은 문제가 반복된다.

---

## Stage 1. Entity field pretraining

목표:

* diffusion 없이 field + renderer만 먼저 학습

학습:
[
\mathcal{L}*{vis} + \mathcal{L}*{amo} + \mathcal{L}*{id} + \mathcal{L}*{temp}
]

출력:

* stable (V_{i,t}), (A_{i,t})

이 단계가 지금 코드에서 가장 부족하다.
현재는 structure와 diffusion이 너무 빨리 얽혀 있다. 

---

## Stage 2. Structured guide pretraining

목표:

* guide encoder가 실제로 scene 정보를 잘 담게 만들기

입력:

* (V_1,V_2,A_1,A_2,D)

출력:

* block-level guide tensor

학습:

* guide reconstruction auxiliary loss
* zero-guide vs full-guide distinguishability
* isolated/composite separation

---

## Stage 3. Diffusion refinement coupling

이제 diffusion과 결합한다.

학습:
[
\mathcal{L}*{render} + \lambda*{iso}\mathcal{L}_{iso}
]

이때 backbone은 low-LR로만 조정하고,
scene module은 쉽게 collapse하지 않게 보호해야 한다.

---

## Stage 4. Temporal fine-tuning

video-level temporal coherence를 강화한다.

추가:

* slot matching
* reappearance consistency
* long-range identity consistency

---

# 6. 추론 단계 설계

## 입력

* text prompt: `"a cat and a dog rolling together on the grass"`
* optional trajectory prior / motion seed

## Step 1. entity parse

prompt에서 entity 2개를 추출한다.

[
(e_1=\text{cat},; e_2=\text{dog})
]

## Step 2. motion rollout

[
g_{1,1:T}, g_{2,1:T}
]

## Step 3. field generation

[
f_{1,1:T}, f_{2,1:T}
]

## Step 4. visible/amodal rendering

[
V_{1,t},V_{2,t},A_{1,t},A_{2,t},D_t
]

## Step 5. diffusion refinement

[
\hat{x}_t = \text{UNet}(z_t, y, G_t)
]

## Step 6. optional consistency correction

* object count check
* identity drift detector
* regenerate local window if violated

---

# 7. 평가 지표 재설계

현재처럼 contract pass/fail 중심이 아니라, 최종 목표 aligned metric으로 간다.

## 7.1 반드시 써야 할 metric

### Per-entity visible survival

[
S_i = \frac{1}{T}\sum_t \mathbf{1}[V_{i,t} \text{ above threshold}]
]

### Visible IoU min

[
\min(\mathrm{IoU}*{vis,e0}, \mathrm{IoU}*{vis,e1})
]

### Amodal IoU min

[
\min(\mathrm{IoU}*{amo,e0}, \mathrm{IoU}*{amo,e1})
]

### Reappearance accuracy

가려졌다가 다시 나온 entity가 같은 identity인지

### Composite-isolated consistency

isolated render와 composite 내 해당 entity crop이 같은지

### Two-object recognition score

외부 detector / embedding 모델로 composite에서 두 object가 감지되는지

---

## 7.2 버려야 할 metric

* `winner`류 balance 단독 metric
* gate 자체가 높다는 이유로 pass
* render IoU를 참고치로만 두는 방식
* contract score 혼합으로 best model 선택

---

# 8. 현재 코드 기준 구체적 리팩토링 계획

## 8.1 유지 파일

* `scripts/build_volume_gt.py` 또는 그 실사용 버전
  → depth/mask/order 기반 GT builder 유지. 

* `models/phase62/projection.py`
  → transmittance renderer 철학 유지, class-world 대신 entity-field world로 일반화. 

* `models/phase62/conditioning.py`
  → block projector, gate-after-normalization, injection manager 유지. guide family 다중 분기 제거 후 단일 structured guide encoder로 개편. 

---

## 8.2 삭제 / 통합 파일

* 구버전 `models/phase62_conditioning.py`
  → 삭제. 

* `guide_family` switch
  → `front_only`, `dual`, `none` mainline 제거

* 복잡한 contract / score selection 로직
  → evaluator 중심으로 단순화

---

## 8.3 새로 만들 파일

### `models/entity_field.py`

역할:

* entity identity + pose → density/appearance field

### `models/renderer.py`

역할:

* per-entity ray-marching
* visible/amodal/depth rendering

### `models/guide_encoder.py`

역할:

* ([V_1,V_2,A_1,A_2,D]) → multiscale guide

### `training/losses_entity.py`

역할:

* visibility / amodal / identity / temporal / isolation losses

### `training/evaluator_entity.py`

역할:

* per-entity visible survival
* reappearance consistency
* isolated/composite consistency
* two-object detection score

---

# 9. 추천 코드 수정 순서

## Phase 1 — 구조 분리

1. guide family 제거
2. projector 출력 포맷을 structured maps로 고정
3. evaluator를 final-goal aligned metric으로 교체

## Phase 2 — entity field 도입

1. class volume 대신 independent entity density heads
2. bg class 제거
3. renderer generalization

## Phase 3 — temporal memory 도입

1. slot memory
2. temporal contrastive
3. reappearance supervision

## Phase 4 — full diffusion coupling

1. guide encoder 안정화
2. low-LR backbone adaptation
3. isolation consistency 강화

---

# 10. 왜 현재 로직은 안 되는가

이제 위 설계를 기준으로 현재 시스템을 다시 보면 실패 이유가 분명해진다.

## 10.1 scene representation이 충분히 entity-centric하지 않다

현재 구조는 projector와 guide가 visible/amodal을 계산하긴 하지만, 최종적으로는 여전히 **guide injection이 dominant entity scale에 끌려간다**. `four_stream`이 구조적으로 정보는 갖고 있어도, stream magnitude가 불균형하면 one-entity 강화 루프가 생긴다.  

---

## 10.2 trainer가 최종 목표보다 surrogate에 더 충실하다

현재 trainer는 다수의 손실을 조합하지만, 최종 composite에서 두 entity가 동시에 살아 있는지를 가장 직접적으로 보지는 않는다. 
그래서 **잘 최적화된 잘못된 해법**이 best로 뽑힐 수 있다.

---

## 10.3 class/threshold 관점이 아직 남아 있다

현재 projector도 `p_fg_k > 0.3` 같은 threshold 기반 visible class 결정을 일부 사용한다. 
이건 practical fix였지만, 근본적으로는 “entity 존재”와 “visible class assignment”를 분리하지 못한다.

---

## 10.4 temporal identity가 약하다

현재 permutation consistency, centroid consistency는 rolling/contact/reappearance 수준의 identity preservation엔 부족하다. 

---

# 11. 최종 추천 실험 세팅

## 실험 이름

**Phase 63 — Dynamic Entity Fields**

## 핵심 설정

* 2 entities fixed
* independent density field per entity
* transmittance renderer
* structured guide = `[V1,V2,A1,A2,D]`
* diffusion = refiner only
* no guide-family branching
* no contract-score checkpointing

## 학습 스케줄

* 50 epochs: field + renderer pretrain
* 30 epochs: guide encoder pretrain
* 80 epochs: diffusion coupling
* 40 epochs: temporal fine-tune

## batch 구성

* 50% collision/contact clips
* 30% layered occlusion clips
* 20% reappearance clips

## best checkpoint 기준

1. visible IoU min
2. visible survival min
3. isolated/composite consistency
4. reappearance consistency
5. two-object recognition score

---

# 12. 아주 짧은 실행 요약

## 삭제

* legacy conditioning file
* front_only / dual / none mainline
* contract-heavy checkpoint logic
* bg-class 경쟁 중심 사고

## 유지

* rendered GT builder
* transmittance rendering 철학
* multiscale guide injection
* diffusion refiner backbone

## 추가

* entity-centric dynamic field
* amodal identity memory
* temporal slot consistency
* final-goal aligned evaluator
* structured guide encoder on `[V1,V2,A1,A2,D]`

---

# 13. 최종 권고안

가장 현실적인 최종 아키텍처는 이것이다.

> **“Independent entity density fields + transmittance-based visible/amodal rendering + structured multiscale guide injection + diffusion refinement + temporal slot consistency”**

이게 좋은 이유는 네 원래 아이디어를 가장 잘 살리기 때문이다.

* 3D 정보를 진짜 활용한다
* visible과 amodal을 분리한다
* 가려져도 entity identity를 유지한다
* diffusion이 scene을 새로 상상하지 않고 structured condition 위에서 refinement한다
* rolling/contact shot에서도 two-entity identity preservation이 가능해진다

현재 코드베이스에서 가장 큰 가치가 있는 부분은 **renderer 철학과 guide injection 인프라**다.
가장 과감히 버려야 할 부분은 **guide family branching, contract-driven best selection, class-competition 중심 구조**다.
그 사이에 **entity field / temporal slot / render-level supervision**을 넣는 것이 정답에 가장 가깝다.   

원하면 다음 답변에서 이걸 이어서
**“파일별 patch plan”** 형태로,
어느 파일을 삭제하고 어느 파일을 새로 만들고 어느 함수 시그니처를 어떻게 바꿀지까지
실제 코드 작업 단위로 쪼개서 적어줄게.
