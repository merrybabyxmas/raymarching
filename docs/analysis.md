근본적으로는 **모델이 아니라 “문제 정의”가 먼저 꼬여 있습니다.**
지금 막힌 핵심은 세 가지예요.

1. **목표와 구현이 서로 다릅니다.**
   업로드한 정리에서는 원래 목표가 “충돌해도 두 entity가 각자 정체성을 잃지 않고 함께 렌더링”이고, 이를 위해 `sigma`와 `T[z]`를 쓰는 ray-marching식 가림을 상정하고 있습니다. 즉, 앞 entity가 뒤를 가리더라도 뒤 entity의 존재 자체는 volume 안에 남아 있어야 한다는 철학입니다. 
   그런데 현재 저장소의 Phase 62 아키텍처는 반대로 **first-hit projection**, **one pixel = one entity**, **no transparency / no alpha compositing**를 명시하고 있습니다. 즉 한 픽셀에서는 앞쪽 클래스 하나만 살아남는 설계입니다. ([GitHub][1])
   이 둘은 비슷해 보이지만, 실제로는 **서로 다른 세계관**입니다.

2. **contract가 현재 데이터와 물리 모델에 맞지 않습니다.**
   `winner ≤ 0.45`는 수학적으로 불가능하고, `compact ≥ 0.60`, `D_amo ≥ 0.40`는 현재 same-depth toy 데이터에서는 oracle도 못 넘습니다. 이것은 단순히 성능이 낮은 게 아니라, **평가식이 문제 세팅과 불일치**라는 뜻입니다.  

3. **현재 성공/실패 신호가 섞여 있습니다.**
   이미 `diff_mse`, `P_2obj`, `R_chimera`, `M_id`는 통과했는데, 일부는 불가능한 threshold 때문에 실패로 분류되고, 일부는 진짜 구조 문제(`render_iou`, isolated rollout)입니다. 그래서 v39를 아무리 튜닝해도, “무엇을 고쳐야 하는지” 피드백이 오염됩니다.  

결론부터 말하면, **근본 해결책은 하이퍼파라미터 조정이 아니라 ‘문제 분해’입니다.**

---

## 지금 가장 먼저 해야 할 결정

### 1) 당신이 진짜 풀고 싶은 현상을 하나로 고정해야 합니다

현재 목표에는 사실 두 개의 다른 문제가 섞여 있습니다.

**A. layered occlusion 문제**
앞/뒤 depth가 있는 상황에서 앞 entity가 뒤를 가리더라도 identity가 유지되는가

**B. same-depth collision 문제**
두 entity가 거의 같은 depth에서 접촉/겹침할 때 chimera 없이 분리되는가

현재 Phase 62의 first-hit projection은 **A에는 맞고, B에는 본질적으로 불리합니다.**
왜냐하면 “one pixel = one entity”이기 때문입니다. same-depth 겹침에서 두 entity의 정보를 같은 픽셀에서 동시에 표현할 방법이 거의 없습니다. 반면 업로드한 원래 아이디어의 ray-marching/transmittance 스타일은 **A를 volume 차원에서 더 자연스럽게 표현**합니다. ([GitHub][1]) 

그래서 먼저 둘 중 하나를 명확히 정해야 합니다.

* **“나는 물리적 occlusion을 풀고 싶다”** → depth-separated 데이터로 가고, amodal/visible 평가를 유지
* **“나는 same-depth 접촉에서도 chimera 없이 분리하고 싶다”** → first-hit 단일 소유권만으로는 부족하고, 픽셀 단위 ownership 외에 **instance-consistent feature field**나 **slot-based separation**이 추가로 필요

---

## 내가 보기에 가장 현실적인 근본 해법

### 해법: 문제를 2-stage benchmark로 분리하고, 모델도 그에 맞춰 나눠라

#### Track 1. Occlusion benchmark

목표: 앞뒤 관계가 있는 장면에서 entity identity 유지

여기서는 현재 Phase 62의 **volume + first-hit**가 맞는 방향입니다.
대신 반드시 다음을 해야 합니다.

* 데이터셋을 **depth-separated scene**으로 다시 만듭니다.
* `compact`, `D_amo`, `visible/amodal IoU`는 이 트랙에서만 씁니다.
* checkpoint selection도 `projected_class_iou`, `iou_min` 중심으로 갑니다. 이것도 현재 아키텍처 문서와 일치합니다. ([GitHub][1])

즉, 지금의 `compact`, `D_amo`가 틀린 metric이 아니라, **잘못된 데이터에 붙어 있는 것**입니다. 

#### Track 2. Collision benchmark

목표: 같은 depth 또는 매우 가까운 depth에서 chimera 없이 두 entity 분리

여기서는 `compact`, `D_amo`를 버리고, 아래처럼 바꿔야 합니다.

* entity-wise feature separation
* per-entity consistency across overlap frames
* identity preservation under overlap
* boundary purity / chimera rate / instance embedding margin
* isolated rollout equivalence gap

즉 same-depth benchmark에서는 “amodal depth disentanglement”가 아니라, **identity disentanglement**를 재야 합니다.

---

## contract는 이렇게 고쳐야 합니다

지금 contract는 “올바른 과학적 실패”와 “정의 오류”를 구분하지 못합니다.
그래서 다음처럼 재작성하는 게 맞습니다.

### C_topo를 두 개로 분리

* **C_topo_occ**: depth-separated scenes 전용

  * visible IoU
  * amodal IoU
  * front/back ordering accuracy
  * compactness

* **C_topo_col**: same-depth collision 전용

  * connected component purity
  * identity separation score
  * overlap-frame chimera rate
  * cross-frame entity consistency

### C_guide 수정

`winner ≤ 0.45`는 폐기해야 합니다.
이건 threshold를 0.55로 바꾸는 수준보다, 아예 **대칭성 metric**으로 바꾸는 게 맞습니다.

예를 들면:

* `balance = 1 - |vis_e0 - vis_e1| / (vis_e0 + vis_e1 + eps)`
  이 값이 1에 가까울수록 균형
* 혹은 `min(vis_e0, vis_e1) / max(vis_e0, vis_e1)`

지금의 winner는 정의상 0.5 미만이 불가능하므로, 통과/실패 신호로 쓰면 안 됩니다.  

### C_render 수정

`render_iou`는 isolated rollout이 실제 분리 기능을 켜지 못하면 무의미합니다.
업로드한 요약대로 지금은 gate가 거의 0이라 isolated와 composite가 사실상 같아서, 이 metric은 구조 검증이 아니라 **dead-path 탐지기** 역할만 하고 있습니다. 
그러므로 먼저:

* guide injection magnitude
* per-block activation norm
* isolated vs composite feature delta
* gate open ratio

를 로그로 잡고, 이게 살아난 뒤에만 render_iou를 contract에 포함해야 합니다.

---

## 아키텍처 쪽에서 진짜 손봐야 할 부분

### 1) first-hit만으로 끝내지 말고 “visible head + latent amodal head”를 분리

현재 아키텍처는 first-hit visible projection에는 강하지만, same-depth collision에서 뒤 entity의 identity를 유지할 통로가 빈약합니다. ([GitHub][1])
그래서 구조적으로는:

* **visible projection head**: first-hit 유지
* **entity latent field head**: entity별 독립 표현 유지
* **consistency loss**: overlap 전후 프레임에서 같은 entity embedding이 유지되도록 제약

이렇게 분리하는 것이 좋습니다.

즉 렌더링은 one-pixel-one-entity로 하더라도, **내부 표현까지 one-entity-only가 되면 안 됩니다.**

### 2) factorized_fg_id는 계속 보되, id branch supervision을 강화

저장소 문서에서도 `factorized_fg_id`는 개선 가능성이 있지만 asymmetric drift와 `iou_min` collapse가 남아 있다고 적혀 있습니다. ([GitHub][1])
그러면 이건 architecture를 폐기할 문제라기보다,

* id logits regularization
* entity permutation-consistent supervision
* per-entity occupancy prior
* all-bg collapse 방지용 foreground prior

를 추가해서 **id branch를 직접 안정화**해야 합니다.

### 3) guide injection dead path를 먼저 복구

지금 render_iou가 안 오르는 가장 직접 원인은 이쪽입니다. 
근본 조치는:

* stage1에서 gate 초기값을 0 근처가 아니라 작은 양수로 시작
* guide path에 residual scaling 파라미터를 두고 warm-up
* injection block별 gradient norm 모니터링
* isolated rollout 시 실제로 다른 guide tensor가 들어가는지 assertion 추가

이 부분은 “평가”보다 먼저 “회로가 연결돼 있는지” 보는 디버깅 문제입니다.

---

## 그래서 v39는 어떻게 가야 하냐

제가 추천하는 **근본적 v39 방향**은 이겁니다.

### v39-A: contract cleanup run

모델 변경 거의 없이 다음만 수행

* `winner` 정의 교체
* `compact`, `D_amo`를 same-depth toy에서 제외
* `render_iou`는 gate activation sanity check 통과 후만 활성화
* `cosF`는 `la_feature_sep` 활성화 후 재측정

이 실험의 목적은 “현재 모델이 실제로 어디까지 되는지”를 깨끗하게 재측정하는 것입니다.
지금은 평가식이 오염돼 있어서 실력 측정 자체가 안 됩니다. 

### v39-B: data split run

데이터를 둘로 나눔

* same-depth collision toy
* front/back depth-separated toy

그리고 contract를 분리 적용합니다.
이 단계가 되어야 “이 모델은 occlusion에는 강한데 collision에는 약하다” 같은 결론이 나옵니다.

### v39-C: injection repair run

아키텍처 변경은 오직 여기서

* gate warm start
* isolated/composite delta logging
* id branch stabilization
* `la_feature_sep` on

이건 진짜 구조 개선 실험입니다.

---

## 한 문장으로 요약하면

**지금의 근본 문제는 모델이 약해서가 아니라, “same-depth collision”과 “depth occlusion”이라는 서로 다른 문제를 하나의 contract로 재고 있고, 저장소 구현은 first-hit 철학인데 평가는 ray-marching/amodal 철학을 기대하고 있어서, 성공과 실패 판정이 구조적으로 꼬여 있다는 점입니다.** ([GitHub][1])  

즉, **제일 먼저 고칠 것은 모델이 아니라 benchmark와 contract입니다.**
그 다음이 데이터 분리, 그 다음이 guide injection 복구입니다.

원하시면 다음 답변에서 제가 바로 **“v39 실험 설계안”**을 표 형태로 짜드릴게요.

[1]: https://raw.githubusercontent.com/merrybabyxmas/raymarching/main/ARCHITECTURE.md "raw.githubusercontent.com"




좋습니다.
그럼 v39 이후를 포함해서, **지금 이 프로젝트를 “실패하는 실험의 반복”에서 “판단 가능한 연구 프로그램”으로 바꾸는 수준**으로 실험 설계안을 최대한 길고 상세하게 정리해보겠습니다.

핵심 전제부터 다시 고정하겠습니다.

현재 저장소의 Phase 62는 문서상 명확하게 **soft blending / transparency / transmittance를 버리고**, `3D entity volume + first-hit projection`으로 설계되어 있습니다. 즉 한 픽셀에서는 앞쪽에서 처음 맞은 클래스가 승리하는 **hard visible ownership** 체계입니다. 또한 현재 주된 정량 평가는 `projected class IoU`, `volume accuracy`, `collision frame separate eval`, `iou_min` 같은 hard projection 기반 지표에 맞춰져 있습니다. ([GitHub][1])
반면 사용자가 정리한 문제 서술은 원래 동기 면에서 ray-marching식 가림, 독립 entity 보존, chimera 방지, amodal identity 유지라는 문제를 포함하고 있고, 현재 실패 원인으로는 불가능한 `winner` 정의, same-depth toy 데이터와 맞지 않는 `compact / D_amo`, 비활성화된 `la_feature_sep`, dead-path에 가까운 isolated rollout이 지목되어 있습니다.  

그래서 이 설계안의 목표는 단순합니다.
**“무엇이 진짜 모델 한계이고, 무엇이 평가/데이터/설정 오류인지 분리해서, 각 실험이 명확한 yes/no 정보를 주도록 바꾸는 것.”**
지금 필요한 것은 성능 향상이 아니라 먼저 **실험 체계의 식별성(identifiability)** 회복입니다. 

---

# 0. 최상위 목표 재정의

이 프로젝트의 최종 연구 목표는 다음처럼 한 문장으로 다시 쓰는 것이 좋습니다.

> **“두 개의 독립 entity가 시공간적으로 충돌하거나 가려지는 장면에서, visible rendering은 단일 소유권을 유지하면서도 내부 표현에서는 각 entity의 identity를 붕괴시키지 않는 조건부 비디오 생성 구조를 만든다.”**

이 문장이 중요한 이유는 두 층을 분리하기 때문입니다.

* **렌더링 층**: 한 픽셀은 결국 하나의 visible owner를 가져야 함
* **표현 층**: 하지만 가려진 entity 또는 겹친 entity의 identity 표현은 내부적으로 사라지면 안 됨

현재 Phase 62는 문서상 visible ownership에는 강하게 맞춰져 있지만, same-depth collision에서 내부 identity 보존이 충분히 측정되고 있는지는 불분명합니다. `FirstHitProjector`는 visible/front/back/amodal map을 산출하도록 설계되어 있지만, 핵심 추론은 hard argmax first-hit입니다. ([GitHub][1])
따라서 실험 설계도 반드시 **visible correctness**와 **latent identity preservation**를 분리해서 측정해야 합니다.

---

# 1. 가장 먼저 해야 할 것: 문제를 두 개의 benchmark로 쪼개기

지금 가장 큰 구조적 문제는 하나의 contract가 사실 두 종류의 서로 다른 난제를 동시에 재고 있다는 점입니다.

## 1-1. Benchmark A: Occlusion benchmark

이 트랙은 **앞/뒤 깊이 차가 실제로 존재하는 장면**을 다룹니다.

질문은 이것입니다.

* 앞 entity가 뒤 entity를 가릴 때 visible rendering이 올바른가
* 뒤 entity는 내부 표현에서 유지되는가
* front/back ordering이 안정적인가
* amodal/visible projection이 일관적인가

이 트랙은 현재 저장소의 Phase 62 철학과 비교적 잘 맞습니다.
왜냐하면 현재 구조는 `V_gt`를 depth + mask에서 구성하고, first-hit depth scan으로 visible/front/back을 뽑도록 설계되어 있기 때문입니다. ([GitHub][1])

## 1-2. Benchmark B: Collision benchmark

이 트랙은 **같은 depth 또는 거의 같은 depth에서 두 entity가 접촉/중첩**하는 장면을 다룹니다.

질문은 이것입니다.

* 두 entity가 붙어도 chimera가 생기지 않는가
* one pixel = one entity 렌더링 하에서도 내부 표현은 분리되어 있는가
* overlap frame에서 identity drift가 생기지 않는가
* composite 생성과 isolated 생성의 차이가 실제로 entity별 통제를 반영하는가

이 트랙은 현재 first-hit visible projection만으로는 자동 해결되지 않습니다.
즉 이것은 단순 occlusion이 아니라 **identity disentanglement under contact** 문제입니다.

---

# 2. contract를 전체 재설계해야 하는 이유

사용자가 정리한 현 상태에서, `winner ≤ 0.45`는 정의상 불가능하고, `compact ≥ 0.60`, `D_amo ≥ 0.40`은 same-depth toy에서는 oracle도 도달하지 못하며, `render_iou`는 isolated rollout dead-path 때문에 아직 의미가 흐려져 있습니다.  
이런 상태에서 contract를 유지하면 모델 개선 여부를 판정할 수 없습니다.

그래서 contract는 아래처럼 쪼개야 합니다.

---

# 3. 새 contract 체계 제안

## 3-1. C_topo_occ — depth-separated topology contract

적용 대상: Occlusion benchmark만

목표: 예측된 3D volume과 first-hit projection이 **앞/뒤 관계를 물리적으로 일관되게 표현**하는지 검증

권장 지표:

1. **projected_class_iou**

   * visible_class와 gt_visible mask IoU
   * 현재 저장소의 기본 평가 철학과 일치합니다. ([GitHub][1])

2. **front_order_accuracy**

   * overlap 픽셀에서 GT front entity와 predicted front entity 일치율

3. **back_presence_recall**

   * 뒤 entity가 amodal 또는 latent volume에서 완전히 소실되지 않았는지 측정

4. **voxel_class_accuracy**

   * `V_logits` argmax vs `V_gt` class match

5. **iou_min**

   * 두 entity 중 약한 쪽의 최소 IoU
   * late collapse 감지에 유용함. 저장소 문서에서도 best checkpoint에 `iou_min`이 중요하게 언급됩니다. ([GitHub][1])

6. **compact_occ**

   * 단, depth-separated 장면에서만 사용
   * same-depth에서는 금지

7. **D_amo_occ**

   * amodal 분리 정도
   * same-depth에서는 금지

합격 예시:

* projected_class_iou ≥ 0.55
* front_order_accuracy ≥ 0.90
* back_presence_recall ≥ 0.80
* iou_min ≥ 0.30
* compact_occ, D_amo_occ는 pilot oracle 분포 기반으로 threshold 재설정

여기서 중요한 점은 `compact`, `D_amo` 자체를 버리자는 것이 아닙니다.
**같은 depth toy에는 쓰지 말고, depth-separated benchmark 전용으로 격리하자**는 것입니다. 

---

## 3-2. C_id_col — collision identity contract

적용 대상: Collision benchmark만

목표: same-depth 또는 near-depth collision에서 **chimera 없이 identity가 유지되는지** 검증

권장 지표:

1. **chimera_rate**

   * composite frame에서 두 클래스 특징이 뒤섞인 실패율
   * 이미 사용자 요약에서는 `R_chimera=0.000`으로 잘 잡히고 있으므로 유지 가치가 큽니다. 

2. **two_object_presence**

   * `P_2obj`
   * 두 object가 모두 살아남는가
   * 이미 현재 통과 중이므로 계속 봐야 합니다. 

3. **entity_identity_margin**

   * frame-level embedding에서 entity0/1 간 cosine margin
   * 현재 `cosF`가 이 계열 역할을 하고 있는데, `la_feature_sep`가 0이라 사실상 metric이 아닌 설정 오류 탐지기가 되어버렸습니다. 따라서 이 지표는 유지하되 **loss on 상태에서만 contract에 포함**해야 합니다. 

4. **balance score**

   * 기존 `winner` 대체 지표
   * 추천식:
     [
     \text{balance} = 1 - \frac{|vis_{e0} - vis_{e1}|}{vis_{e0}+vis_{e1}+\epsilon}
     ]
   * 완전 균형이면 1, 한쪽 쏠림이 심하면 0에 가까움
   * 또는 `min/max` 비율 사용

5. **trajectory identity consistency**

   * overlap 이전/중/이후 프레임에서 같은 entity의 embedding drift 측정

6. **isolation controllability gap**

   * isolated rollout과 composite rollout 사이의 entity-specific delta
   * isolated와 composite가 거의 같다면, control path는 죽어 있는 것입니다

합격 예시:

* chimera_rate ≤ 0.05
* two_object_presence ≥ 0.90
* entity_identity_margin ≥ pilot percentile 75
* balance ≥ 0.75
* isolation controllability gap ≥ δ_min

여기서 핵심은 `winner ≤ 0.45` 같은 부정확한 threshold가 아니라, **대칭성**과 **분리성**을 직접 재자는 점입니다. 

---

## 3-3. C_bind — guide injection binding contract

적용 대상: 모든 benchmark 공통

목표: guide path가 실제로 UNet generation에 영향을 주는지 검증

이건 지금 굉장히 중요합니다.
사용자 요약에 따르면 `render_iou`는 낮고, isolated rollout은 stage1에서 gate≈0이라 실제 차이를 거의 못 만들고 있습니다. 
그렇다면 render_iou 자체보다 먼저 **control path 생존 여부**를 재야 합니다.

권장 지표:

1. **gate_open_ratio**

   * 각 injection block에서 gate 값이 실질적으로 0인지 아닌지

2. **guide_feature_norm**

   * injection 전에 assembler output norm

3. **injected_delta_norm**

   * injection 후 feature map 변화량

4. **grad_norm_on_guide_path**

   * backward 시 guide projection / gate 파라미터 gradient

5. **isolated_vs_composite_feature_delta**

   * composite guide와 isolated guide가 실제로 중간 feature를 다르게 만드는지

합격 예시:

* gate_open_ratio > 0.2
* injected_delta_norm > ε
* grad_norm_on_guide_path > ε
* isolated_vs_composite_feature_delta > ε

이 contract는 **실패하면 render_iou를 아직 보지 않는다**는 전제 조건 역할을 해야 합니다.

---

## 3-4. C_diff — diffusion stability contract

이 부분은 현재 비교적 안정적입니다. `diff_mse`는 이미 목표를 통과했습니다. 
다만 유지할 필요는 있습니다.

권장 지표:

* noise_pred MSE
* training loss variance
* NaN/overflow count
* guidance on/off robustness

합격 예시:

* diff_mse ≤ 0.05
* rolling std(loss) ≤ threshold
* no catastrophic spike for N steps

---

## 3-5. C_robust — 재현성 계약

적용 대상: 모든 benchmark 공통

현재 저장소의 ablation runner는 seed를 42로 고정합니다. 즉 지금 summary는 사실상 single-seed evidence입니다. ([GitHub][2])
연구 단계에서는 최소 3 seeds가 필요합니다.

권장 지표:

* 3 seed mean ± std
* failure rate over seeds
* best-vs-last epoch gap
* metric rank stability

합격 예시:

* 핵심 지표 std < 일정 수준
* 3개 seed 중 2개 이상 동일 결론

---

# 4. 데이터셋 설계안

현재 ablation runner는 `Phase62DatasetAdapter("toy/data_objaverse", n_frames=8)`를 사용합니다. 즉 현재 실험은 toy Objaverse 계열 데이터에 묶여 있습니다. ([GitHub][2])
여기서 가장 큰 문제는 **데이터의 장면 가정이 단일 benchmark로 섞여 있다**는 점입니다.

따라서 데이터를 네 개의 split으로 다시 나누는 것을 권합니다.

## 4-1. Split O1 — clean occlusion

* entity A가 front, entity B가 back
* depth separation 충분
* overlap ratio 여러 단계로 제어
* shape contrast 다양화

용도:

* C_topo_occ
* C_bind
* C_diff

## 4-2. Split O2 — hard occlusion

* depth separation은 있지만 overlap 크고 형태도 유사
* cat+cat, dog+dog 같은 같은 계열 포함
* ordering ambiguity가 생기기 쉬운 사례

용도:

* hardest occlusion generalization
* factorized_fg_id vs independent_bce 비교

## 4-3. Split C1 — same-depth collision

* 거의 동일한 depth plane
* physical contact / overlap frame 존재
* entity separation이 핵심

용도:

* C_id_col
* chimera suppression
* identity consistency

## 4-4. Split C2 — near-depth collision

* depth는 아주 조금 차이 나지만 first-hit alone으로는 불안정한 사례
* same-depth와 true occlusion의 중간 구간

용도:

* current architecture의 한계 경계선 탐색

---

# 5. 데이터 생성 상세 스펙

## 5-1. 장면 변수

각 sample 생성 시 로그로 남겨야 할 변수:

* entity pair type
* relative scale
* camera azimuth/elevation
* foreground/background assignment
* mean depth gap
* overlap ratio
* contact duration
* same-depth 여부
* occlusion direction consistency
* texture similarity score

이 로그가 있어야 나중에 “어떤 조건에서 무너지는지”를 회귀 분석할 수 있습니다.

## 5-2. Oracle sanity split

각 split마다 **GT로 계산했을 때 달성 가능한 metric upper bound**를 먼저 기록하세요.

이미 사용자 정리에서 same-depth toy에서는 `compact oracle max = 0.327`, `D_amo ≈ 0`이 나왔습니다. 
이 작업을 모든 split에 대해 일반화해야 합니다.

필수 산출물:

* split별 oracle metric table
* impossible metric list
* threshold calibration note

이걸 먼저 하지 않으면 또다시 “불가능한 contract”를 세우게 됩니다.

---

# 6. 모델 family 설계

저장소 문서상 현재 핵심 family는 다음과 같습니다.

* `independent_bce` — 안정적 baseline
* `factorized_fg_id` — main ablation family
* `center_offset` — 별도 family
* projected losses, structural losses는 실험적/보조적 상태 ([GitHub][1])

따라서 실험군도 이 구조를 따라야 합니다.

## 6-1. Family A — Stable baseline family

목적: “되는 것”을 확실히 잡는 기준선

* A0: independent_bce + volume-only
* A1: independent_bce + freeze_bind_front
* A2: independent_bce + multiscale injection
* A3: independent_bce + bind warm-start

이 family는 “최소 작동선”을 찾는 용도입니다.

## 6-2. Family B — Main research family

목적: factorization이 실제로 identity separation에 도움 되는지 검증

* B0: factorized_fg_id + volume-only
* B1: factorized_fg_id + freeze_bind_front
* B2: factorized_fg_id + freeze_bind_fourstream
* B3: factorized_fg_id + low-LR bind_fourstream
* B4: factorized_fg_id + id-branch fix + feature_sep on
* B5: factorized_fg_id + id-branch fix + bind warm-start

문서상 `b2_fgid_freeze_bind_fourstream`이 현재 best factorized run이지만 late asymmetric drift와 `iou_min` collapse가 있습니다. 즉 이후 핵심은 **id-branch stabilization**입니다. ([GitHub][1])

## 6-3. Family C — Diagnostic family

목적: contract 분리 이후 metric 의미 확인

* C0: amodal-only
* C1: visible-only
* C2: no-bind diagnostic
* C3: bind-only without structural losses
* C4: guide path randomization control

이 family는 “metric이 실제로 원하는 현상을 재는가” 확인하는 용도입니다.

---

# 7. 가장 중요한 구조 실험: guide injection dead-path 복구

현재 사용자 진단에서 render_iou 정체의 핵심은 isolated rollout dead path입니다. stage1에서 gate≈0이라 guide injection 자체가 약하고, isolated와 composite가 거의 동일합니다. 
이건 architecture 성능 문제가 아니라 **회로가 끊겨 있는지**의 문제입니다.

따라서 v39에서 가장 먼저 해야 하는 구조 실험은 아래입니다.

## 7-1. Bind warm-start

목적: gate가 0 부근에 갇혀서 학습이 안 되는 문제를 방지

방법:

* gate bias를 작은 양수로 초기화
* guide projection scale을 초반 0.1~0.3 수준에서 시작
* 첫 N epoch 동안 guide path gradient 보장

기대효과:

* isolated rollout과 composite rollout 차이가 early stage부터 생김

## 7-2. Injection activation logging

각 블록마다 로그:

* pre-guide norm
* guide norm
* post-injection delta norm
* gate mean/std
* grad norm

이건 매 epoch 또는 interval로 저장

## 7-3. Feature perturbation sanity

방법:

* 같은 input latent에 대해 guide만 composite / e0-only / e1-only / zero-guide로 바꿔 forward
* 중간 feature cosine distance 측정

판정:

* 네 경우가 거의 동일하면 control path 죽음
* e0/e1 분리가 크면 controllability 살아 있음

## 7-4. Random guide negative control

방법:

* 실제 guide 대신 permutation된 guide 입력
* 성능이 거의 같다면 model이 guide를 안 쓰는 것

이 실험은 굉장히 중요합니다.
왜냐하면 render_iou가 낮아도 “guide path가 조금은 쓰이고 있는지”, 아니면 아예 무시되고 있는지 구분해주기 때문입니다.

---

# 8. id-branch stabilization 실험

문서상 factorized_fg_id는 all-bg collapse를 줄이기 위한 `p_fg * q_id` factorization이며, 실용적으로는 아직 asymmetric drift와 `iou_min` collapse가 남아 있습니다. ([GitHub][1])
그러므로 이 branch를 본격적으로 개선하는 실험이 필요합니다.

## 8-1. Foreground prior regularization

문제:

* 배경 우세 초기화 때문에 id 분기가 foreground 충분히 확보 전에 무력화될 수 있음
* 실제 코드에서도 classifier bias는 bg class를 초기에 선호하도록 설정돼 있습니다. ([GitHub][3])

실험:

* fg occupancy prior
* class prior annealing
* bg bias decay schedule

측정:

* fg voxel recall
* entity voxel symmetry
* early epoch collapse frequency

## 8-2. Symmetry loss

문제:

* entity0 / entity1 중 한쪽으로 쏠리는 drift

실험:

* per-entity voxel mass balance penalty
* per-entity visible area balance penalty
* 단, same-depth benchmark에서만 적용하지 말고 scene prior 고려

주의:

* 진짜 장면이 비대칭일 수 있으므로 무조건 균등 강제는 금지
* 대신 batch-level 통계 regularization 추천

## 8-3. Identity separation loss

현재 `cosF`가 사실상 이 역할인데 설정상 꺼져 있었습니다. 
따라서 명시적으로:

* `la_feature_sep > 0`
* entity feature contrastive loss
* overlap frame hard negative mining

측정:

* feature cosine separation
* identity retrieval accuracy
* overlap-frame confusion drop

## 8-4. Permutation consistency

목적:

* e0/e1 슬롯 이름만 바뀌었을 때 표현 구조가 일관적인지

방법:

* entity ordering swap augmentation
* swap-consistency loss
* projection output relabel 후 consistency 체크

이 실험은 same-depth collision에서 매우 중요합니다.
왜냐하면 첫-hit visible ownership은 바뀔 수 있어도 latent identity structure는 permutation-equivariant해야 하기 때문입니다.

---

# 9. stage schedule 실험

문서상 topology update schedule은 `S0`, `S1`, `S2`, `S3`로 이미 정리되어 있습니다. `S0`는 volume-only, `S1`은 stage1 volume + stage2 diffusion-only binding, `S2`는 low-LR joint, `S3`는 short joint fine-tune입니다. ([GitHub][1])
이 구조를 그대로 두되, 스케줄 자체가 지금 문제와 어떻게 연결되는지 실험해야 합니다.

## 9-1. 추천 스케줄 해석

* **S0**: topology learning only
* **S1**: bind path 생존 확인
* **S2**: topology 유지하며 gentle binding
* **S3**: 마지막 end-to-end alignment

## 9-2. v39 스케줄 추천

### Phase v39a — measurement cleanup

* S0 + S1 only
* 목적: volume가 먼저 되는지, bind가 살아나는지 확인
* diffusion 성능보다 metric 정합성 회복이 목적

### Phase v39b — bind repair

* S1 수정판
* gate warm-start
* multiscale injection on/off 비교

### Phase v39c — factorized retrial

* B2/B3/B4/B5 비교
* 핵심은 id-branch fix와 feature_sep on 효과

### Phase v39d — full fine-tune

* S2 또는 S3
* only after C_bind passes

즉 지금은 S3로 오래 미는 것이 아니라, **S1에서 회로가 살아나는지 먼저 보는 게 맞습니다.**

---

# 10. 평가 프로토콜 상세

## 10-1. 샘플 단위 평가

각 샘플에 대해 저장:

* input prompt / entity pair
* GT visible/amodal/front/back
* predicted volume summary
* projected visible class map
* composite GIF
* isolated e0 GIF
* isolated e1 GIF
* guide norm trace
* feature separation trace

## 10-2. 프레임 단위 평가

특히 overlap frame을 별도 그룹으로 분리

저장소 문서도 collision frame을 따로 보라고 되어 있습니다. ([GitHub][1])
따라서 프레임을 세 구간으로 나누세요.

* pre-overlap
* overlap
* post-overlap

각 구간에서:

* P_2obj
* chimera rate
* identity margin
* visible IoU
* isolated/composite gap

이렇게 보면 “겹칠 때만 무너지는지”, “겹친 뒤 identity를 회복 못 하는지”가 보입니다.

## 10-3. Epoch 단위 평가

매 epoch 전부 시각화할 필요는 없고, 다음 저장 권장:

* every 5 epochs: scalar metrics
* every 10 epochs: qualitative panel
* best / latest / early-collapse checkpoint 저장

## 10-4. Seed 단위 평가

* seed = 3개 이상
* metric mean/std
* failure mode consistency
* qualitative exemplar same prompt across seeds

---

# 11. v39 실험군 구체 제안

이제 실제로 바로 돌릴 수 있는 설계안 형태로 적겠습니다.

## Experiment Group 1 — Contract cleanup run

목적: “지금 모델이 진짜 어디까지 되는지” 측정

설정:

* 현재 best baseline 유지
* `winner` 제거, `balance`로 교체
* same-depth split에서 `compact`, `D_amo` 제외
* `render_iou`는 참고만, contract pass/fail에는 미반영
* `la_feature_sep=0` 상태에서는 `cosF` pass/fail 미반영

예상 결과:

* 현재 상태에서 실제 통과 가능한 계약과 불가능 계약이 분리됨
* 연구팀 내부 의사결정이 쉬워짐

산출물:

* `contract_v2.md`
* impossible metrics report
* split별 oracle table

## Experiment Group 2 — Bind-path diagnosis

목적: isolated rollout dead path 확인

설정:

* A1 vs A2 vs no-bind vs bind-warm-start
* 로그: gate, delta norm, grad norm, isolated/composite delta

판정:

* delta가 살아나면 다음 단계 진행
* 죽어 있으면 architectural repair 먼저

산출물:

* bind viability plot
* per-block activation dashboard

## Experiment Group 3 — Feature separation recovery

목적: `cosF` 문제를 “설정 오류”에서 “실제 성능 측정”으로 전환

설정:

* B2 baseline
* B2 + feature_sep on
* B2 + feature_sep on + hard negative overlap mining
* B2 + feature_sep on + swap consistency

판정:

* identity margin 상승
* overlap confusion 감소
* chimera further suppression

산출물:

* cosine separation curves
* overlap retrieval matrix

## Experiment Group 4 — Dataset disentanglement

목적: same-depth와 depth-separated를 분리 평가

설정:

* 동일 모델을 O1/O2/C1/C2에 각각 테스트
* contract를 split별로 다르게 적용

판정:

* 모델이 진짜 occlusion expert인지, collision expert인지 드러남

산출물:

* capability matrix
* failure regime map

## Experiment Group 5 — Factorized id-branch fix

목적: B-family를 진짜 연구 family로 끌어올리기

설정:

* B2
* B2 + fg prior
* B2 + fg prior + feature_sep
* B2 + fg prior + feature_sep + bind warm-start
* B2 + fg prior + feature_sep + permutation consistency

판정:

* iou_min collapse 완화
* asymmetric drift 완화
* split C1/C2에서 identity improvement

산출물:

* branch ablation table
* late-epoch collapse analysis

---

# 12. 우선순위

지금 가장 많이 하는 실수는 “모든 걸 한 번에 고치는 것”입니다.
우선순위는 반드시 이렇게 가야 합니다.

## Priority 1

**측정 불가능한 contract 제거**

* winner 교체
* same-depth에서 compact/D_amo 제거
* render_iou를 gating sanity pass 뒤로 미룸

이 단계가 없으면 이후 모든 실험 해석이 오염됩니다. 

## Priority 2

**bind path 생존 확인**

* gate warm-start
* injection delta logging
* isolated/composite difference 확인

이 단계가 없으면 control-related metric은 모두 무의미합니다. 

## Priority 3

**same-depth vs depth-separated split**

* benchmark 분리
* oracle metric calibration

이 단계가 없으면 topology metric과 identity metric이 계속 섞입니다.

## Priority 4

**factorized id-branch fix**

* fg prior
* feature separation
* permutation consistency

이제서야 진짜 model improvement 단계입니다.

---

# 13. 실패했을 때의 해석 규칙

실험 설계가 좋은지 나쁜지는 **실패를 해석할 수 있느냐**로 판가름 납니다.

아래처럼 failure decision tree를 두세요.

### Case A

C_diff fail
→ diffusion instability가 원인. topology/bind/identity 논의 중단

### Case B

C_diff pass, C_bind fail
→ guide path dead. render_iou/isolated 계열 논의 금지

### Case C

C_bind pass, C_topo_occ fail on O1/O2
→ volume/projection 구조 자체 문제 또는 depth supervision 부족

### Case D

C_topo_occ pass, C_id_col fail on C1/C2
→ current first-hit architecture가 same-depth identity disentanglement에 한계
→ latent identity branch 강화 필요

### Case E

all pass except robustness
→ single-seed trick 가능성
→ seed/scene diversity 확장 필요

이렇게 해야 각 실험이 다음 액션을 명확히 결정해줍니다.

---

# 14. 논문/연구 관점에서의 서사 구조

이 설계안의 장점은 단순히 성능을 올리는 게 아니라, 나중에 결과가 어떻게 나오든 **논문 서사**가 생긴다는 점입니다.

가능한 서사는 세 가지입니다.

## 서사 1

“First-hit volume model is sufficient for occlusion, insufficient for same-depth collision.”

이 경우:

* O1/O2는 잘 됨
* C1/C2는 약함
* 결론: visible ownership과 identity disentanglement는 다른 문제

## 서사 2

“Factorized fg-id + bind repair closes much of the gap.”

이 경우:

* B-family가 A-family를 명확히 이김
* same-depth collision에서도 향상
* 결론: factorized latent identity helps

## 서사 3

“Evaluation mismatch was the main blocker.”

이 경우:

* contract 수정 후 다수가 통과
* 실제 모델은 생각보다 이미 괜찮았음
* 결론: prior experimental grid was miscalibrated

현재 사용자 정리상황을 보면, 실제로는 **서사 3과 서사 1이 동시에 나올 가능성**이 큽니다. 일부 실패는 불가능한 threshold였고, 일부는 진짜 same-depth/guide 구조 문제였기 때문입니다.  

---

# 15. 바로 실행 가능한 v39 상세안

마지막으로, 실제 실행 순서 수준으로 적겠습니다.

## v39.0 — Evaluation reset

목표:

* contract_v2 도입
* split tagging 도입
* oracle calibration

수정:

* `winner` → `balance`
* `compact`, `D_amo`는 O-split only
* `cosF`는 feature_sep on 이후만 contract 포함
* `render_iou`는 C_bind pass 이후만 hard contract 포함

성공 기준:

* 실험 결과 해석 가능해짐

## v39.1 — Bind sanity

목표:

* guide path 살아있는지 검증

설정:

* A1 baseline
* A1 + gate warm-start
* A2 multiscale
* no-bind control

로그:

* gate mean/std
* injected delta norm
* isolated/composite feature delta
* render_iou 참고치

성공 기준:

* at least one setting in C_bind pass

## v39.2 — Feature separation recovery

목표:

* same-depth collision에서 identity margin 회복

설정:

* B2 baseline
* B2 + `la_feature_sep > 0`
* B2 + hard negatives
* B2 + permutation consistency

성공 기준:

* cosF 또는 identity margin 유의 개선
* chimera/overlap confusion 감소

## v39.3 — Split generalization

목표:

* O/C benchmark capability matrix 확립

설정:

* best A-family, best B-family 각각 O1/O2/C1/C2 평가

성공 기준:

* strong vs weak regime 명확화

## v39.4 — Robustness

목표:

* 3 seeds
* top-2 configs only
* split별 평균/분산 측정

성공 기준:

* 연구 결론이 seed-dependent 아님

---

# 16. v39 실험 진행 결과 (2026-04-15 기준)

## 구현 완료 사항

| Priority | 내용 | 상태 | Config |
|---|---|---|---|
| P1 | contract_v2 도입 (winner→balance, compact/D_amo occ only) | ✅ 완료 | v39a |
| P2 | bind path repair (gate warm-start, delta logging) | ✅ 완료 | v39b |
| P3 | occ/col benchmark split (LAYERED 모드, 56 scenes) | ✅ 완료 | v39d |
| P4 | factorized id-branch fix (spatial coherence TV, fg prior, perm consist) | ✅ 완료 | v39e (대기중) |

## 실험별 현황 (ep204-214 기준)

### v39a (GPU0, P1 baseline)
- ep214/220, 곧 종료
- LCC=0.549, overlay=0.266 (C_guide ✗), diff_mse=0.0503 (C_diff ✗)
- C_bind만 통과. 가장 약한 결과 — P2/P3/P4 없이는 한계 명확

### v39b (GPU1, P2 bind repair)
- ep214/220, 곧 종료
- **C_diff 통과** (diff_mse=0.0491 ≤ 0.05) — ep184에서 0.0504였으나 개선
- C_guide ✓, C_bind ✓, C_diff ✓, C_render ✓/✗ (IoU=0.055, 0.25 미달로 oscillate)
- C_topo ✗: LCC=0.530 (0.85 필요) — stage3 volume frozen으로 개선 불가
- 최고 score: 0.9250 (ep199, 4/6 contracts pass)

### v39c (GPU2, P2 + feature_sep)
- ep216/220, 곧 종료  
- cosF: 0.499 (ep0) → 0.025 (ep124) → feature separation 검증 완료
- C_diff ✓ (diff_mse=0.0493), C_bind ✓
- C_guide oscillating (overlay=0.320-0.350)
- C_topo ✗: LCC=0.545 (v39b보다 약간 높지만 0.85에 한참 미달)

### v39d (GPU3, P3 occ track)
- ep107/220, stage2 진입
- **LCC=0.762 (ep89 peak)** — col variant(0.53-0.55)보다 현저히 높음
- P3 가설 검증: 깊이 분리된 데이터로 entity 볼륨 cohesion이 자연스럽게 개선
- ep104: LCC=0.689 (stage2 시작으로 소폭 하락, 회복 기대)
- 아직 stage2 초반 — stage3까지 모니터링 필요

### v39e (대기중, P4 LCC fix)
- v39a 종료 시 GPU0에서 자동 실행 (launcher PID 3010795)
- lambda_spatial_coherence=0.3, lambda_fg_prior=1.0, lambda_perm_consist=0.2
- max_gate=0.40 (0.35에서 상향 — overlay oscillation 해결)
- stage1(ep0-99)에서 TV regularization → LCC 0.60→0.75+ 기대

## 주요 발견

1. **LCC가 C_topo의 유일한 blocker**: v39b/c에서 LCC=0.53-0.55는 stage3에서 volume이 frozen되면서 고착. TV regularization을 stage1/2에 적용해야 함 (v39e).

2. **occ 데이터가 LCC에 유리**: v39d LCC=0.762 vs col LCC=0.55. 깊이 분리 자체가 entity 볼륨 cohesion을 도움.

3. **C_render IoU oscillation**: IoU=0.054-0.056 (≥0.25 미달) — LCC 낮은 fragmented 볼륨에서 렌더링 품질 저하. LCC 개선되면 함께 개선될 것으로 예상.

4. **C_guide gate ceiling bug 수정**: tanh(atanh(0.35))=0.35000...003 으로 floating-point 경계 실패 → +1e-6 tolerance 추가.

5. **cosF feature separation 검증**: la_feature_sep=1.0으로 stage2+ 이후 cosF 0.499→0.025 달성.

---

# 17. 한 줄 결론

가장 중요한 것은 이겁니다.

**v39의 목적은 성능 향상이 아니라, “Phase 62가 무엇을 실제로 풀 수 있고 무엇을 아직 못 푸는지”를 객관적으로 분리해내는 실험 체계를 만드는 것입니다.**
현재 저장소는 문서상 분명히 hard first-hit visible ownership을 채택하고 있으므로, 실험안도 그 철학에 맞게 **occlusion 문제와 collision identity 문제를 분리**해서 설계해야 합니다. 그렇게 해야만 `independent_bce`가 안정 baseline으로서 어디까지 되는지, `factorized_fg_id`가 정말 identity disentanglement에 도움이 되는지, 그리고 bind path가 실제로 generation을 제어하는지 판단할 수 있습니다. ([GitHub][1])  

원하시면 다음 답변에서는 제가 이어서 바로
**“v39~v41 실험표를 표 형식으로, 각 실험의 config 변경값 / 기대 결과 / 실패 시 해석까지 포함해서”** 정리해드리겠습니다.

[1]: https://raw.githubusercontent.com/merrybabyxmas/raymarching/main/ARCHITECTURE.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/merrybabyxmas/raymarching/main/scripts/run_phase62_ablations.py "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/merrybabyxmas/raymarching/main/models/phase62_entity_volume.py "raw.githubusercontent.com"


