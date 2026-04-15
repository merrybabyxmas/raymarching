너는 비디오 생성/3D-aware conditioning/occlusion reasoning/instance identity preservation를 연구하는 수석 연구 엔지니어다.
내가 원하는 것은 단순히 “두 object가 나오게 하는 것”이 아니라, 고양이와 강아지처럼 서로 다른 두 entity가 영상 안에서 뒹굴고 접촉하고 서로 가리더라도, 최종 2D 렌더링 결과에서 chimera 없이 각자의 정체성을 유지하는 shot 생성 구조를 만드는 것이다.

내 핵심 아이디어는 다음과 같다.

장면 안에는 실제로 두 개의 독립된 entity가 있다.
이 두 entity는 시간에 따라 3D 위치, 깊이, 자세, overlap 관계가 계속 바뀐다.
카메라는 결국 2D 이미지를 보지만, 생성 과정에서는 3D entity 정보를 먼저 추론한 뒤 그것을 2D로 projection하면 chimera 문제를 줄일 수 있을 것이라고 생각했다.
즉, “2D에서 바로 섞여버린 특징”을 처리하는 대신, 3D entity volume / depth ordering / occlusion structure를 먼저 표현한 후, 이를 camera projection에 반영하면 두 entity의 identity가 더 잘 보존될 것이라는 가설이다.

현재 코드 저장소는 이 아이디어를 구현하려고 하지만, 실제 결과는 아직 single entity처럼 보이는 collapse가 자주 나타난다. 예를 들어 고양이+강아지를 원했는데 출력은 둘이 아니라 하나의 합쳐진 동물처럼 보이거나 한 entity만 남는 결과가 나온다.

내가 원하는 것은 다음 두 가지다.

첫째, 지금 왜 single-entity collapse가 생기는지 근본 원인을 구조적으로 파악
둘째, 내 최종 목표(시간에 따라 바뀌는 3D entity 관계를 이용해 chimera 없는 2D shot 생성)에 맞게 어떤 방향으로 고쳐야 하는지 가장 현실적이고 구체적인 해결책 제시

현재 저장소와 실험 상태에 대해 알고 있어야 할 핵심 사실은 다음과 같다.

현재 구현 철학
현재 Phase 62는 3D Entity Volume + First-Hit Projection 구조다.
철학은 매우 명확하다:
No soft blending
No transparency
No alpha compositing
One pixel = one entity
즉, projection 시 depth 축을 앞에서 뒤로 스캔해서 처음 맞은 클래스가 그 픽셀의 주인이 되는 first-hit 방식이다.
이 구조는 layered occlusion에는 맞을 수 있지만, same-depth contact/collision에서는 본질적으로 한 entity가 다른 entity를 이겨버리기 쉽다.
현재 volume head 구조
EntityVolumePredictor는 F_g, F_0, F_1에서 3D volume logits를 예측한다.
출력은 (bg, entity0, entity1) 클래스에 대한 voxel-wise softmax logits이다.
classifier bias가 초기부터 background를 선호하도록 설정되어 있다.
이 때문에 초기에 foreground/entity 표현이 충분히 생기기 전에 bg collapse 혹은 한쪽 entity collapse가 발생할 가능성이 높다.
현재까지 파악한 문제
나는 원래 “3D 정보를 써서 chimera를 없애자”가 목표였지만, 현재 구조는 사실상 hard visible ownership에 가깝다.
즉, 내부적으로 3D volume을 쓴다고 해도 최종 visible projection에서 한 픽셀은 하나의 entity만 가질 수 있으므로, collision 시 둘 다 안정적으로 남기기 어렵다.
latest analysis 기준으로:
일부 contract는 잘못 정의되어 있었고 재보정되었다.
feature separation은 어느 정도 개선 가능성이 보였다.
그러나 render IoU는 여전히 매우 낮아서, guide injection이 실제 공간 위치를 제대로 제어하지 못하고 있다.
identity signal은 어느 정도 있는데, 그 identity를 “둘 다 다른 위치/깊이로 유지하며 렌더링하는 능력”은 부족하다.
지금 single entity 결과는 단순히 “학습이 덜 됐다”가 아니라,
volume 단계에서 한쪽 entity가 죽거나
projection 단계에서 one-pixel-one-entity first-hit 때문에 한쪽이 소실되거나
guide injection이 위치/분리 제어를 거의 못 해서 diffusion이 하나의 coherent object로 수렴하거나
same-depth collision 문제를 layered occlusion용 구조로 풀려고 해서
생기는 것일 수 있다.
내가 너에게 원하는 구체적인 작업

다음 질문들에 대해 가설 → 근거 → 확인 방법 → 수정 방향의 형식으로 아주 깊게 분석해줘.

질문 1.

내 최종 목표는 “고양이와 강아지가 뒹구는 shot”처럼, 시간에 따라 3D 관계가 계속 바뀌는 두 entity를 2D로 projection해도 chimera가 생기지 않게 하는 것이다.
이 목표에 비추어 볼 때, 현재의 first-hit projection 철학이 맞는 방향인지, 아니면 내 원래 아이디어(ray-marching/transmittance 계열 혹은 visible/amodal 분리 구조)에 더 가까운 구조가 필요한지 판단해줘.

여기서 단순 찬반이 아니라 다음을 구분해서 분석해줘.

layered occlusion에서는 무엇이 맞는가
same-depth collision에서는 무엇이 맞는가
dynamic rolling/contact shot에서는 어떤 표현이 더 자연스러운가
one-pixel-one-entity 제약을 유지해도 내부 latent level에서 두 entity를 동시에 보존할 수 있는가
“visible ownership”과 “identity preservation”를 같은 head로 처리하면 왜 무너질 수 있는가
질문 2.

현재 single-entity collapse의 원인을 가능한 한 구체적으로 분해해줘.
특히 아래 후보 원인들을 각각 평가해줘.

Volume collapse
V_logits 단계에서 bg나 entity0/1 한쪽으로 이미 collapse되는가
factorized/id branch/fg prior 문제인가
class imbalance나 bg-biased init 때문인가
Projection collapse
volume 내부엔 둘 다 있는데 first-hit projection에서 한쪽만 visible winner가 되는가
same-depth overlap에서 이 현상이 구조적으로 unavoidable한가
front/back assignment가 unstable한가
Guide injection dead-path
isolated/composite/zero-guide가 실제로 거의 차이가 없는가
guide가 identity는 조금 전달하지만 위치 제어는 못 하는가
그래서 diffusion이 두 object가 아니라 하나의 coherent animal-like blob로 수렴하는가
Representation mismatch
3D topology learning과 2D generative rendering이 아직 서로 느슨하게 연결되어 있는가
volume에서 배운 entity separation이 UNet의 actual denoising path에 충분히 binding되지 않는가
Benchmark mismatch
occlusion 문제와 collision 문제를 같은 metric/contract로 재서 잘못된 방향으로 최적화했는가

각 원인에 대해
“얼마나 가능성이 높은지”,
“어떤 로그나 시각화를 보면 바로 판별되는지”,
“현재 코드베이스에서 어디를 보면 되는지”
를 구체적으로 적어줘.

질문 3.

내 목표는 시간에 따라 자세와 깊이가 계속 바뀌는 동적 shot이다.
그런데 현재 구조는 frame-by-frame visible ownership만 잘하게 되는 방향으로 갈 위험이 있다.
이럴 때 시간축을 포함한 entity identity consistency를 어떻게 넣어야 할지 제안해줘.

다음 항목을 반드시 포함해줘.

overlap 전 / overlap 중 / overlap 후에서 같은 entity가 어떻게 유지되어야 하는가
“고양이였던 것”과 “강아지였던 것”이 접촉했다가 분리된 뒤에도 동일 identity로 추적되도록 하려면 어떤 latent consistency가 필요한가
temporal consistency를 soft blending이 아니라 entity-level consistency로 넣는 방법
slot consistency, permutation consistency, cross-frame entity matching, trajectory-level embedding consistency 같은 방법 중 어떤 게 가장 현실적인가
질문 4.

내가 정말 해야 할 실험들을 우선순위대로 정리해줘.
단, 막연한 얘기가 아니라 내가 다음 실험 5개만 돌린다면 무엇을 돌려야 하는지 형태로 써줘.

각 실험에 대해 다음을 포함해줘.

목적
바꿀 config / 모듈
측정할 로그
기대되는 패턴
실패 시 해석
다음 액션

실험은 반드시 아래 범주를 포함해야 한다.

volume collapse 진단
projection collapse 진단
guide injection viability 진단
same-depth vs depth-separated 데이터 분리 실험
identity consistency 강화 실험
질문 5.

구조 수정안을 제안해줘.
단, “전부 갈아엎자”가 아니라 현재 코드베이스를 최대한 살리면서 갈 수 있는 수정과,
정말로 철학을 바꿔야 하는 수정으로 나눠서 제안해줘.

A. 현재 코드베이스를 유지한 채 가능한 수정

예:

visible head와 latent amodal/identity head 분리
bg bias 완화
per-entity occupancy prior
feature separation loss
permutation consistency
guide injection warm-start
isolated/composite delta regularization
front/back supervision 강화
overlap-frame hard negative mining
temporal identity consistency loss
B. 철학을 바꾸는 수정

예:

first-hit only 구조를 완화
visible rendering은 hard ownership으로 유지하되, 내부는 transmittance-aware latent 유지
entity-specific latent field를 두고 projection은 visible만 hard하게 하도록 분리
2.5D layered representation 또는 amodal field 추가
same-depth contact를 위해 slot-based object field 추가

각 수정안에 대해

왜 필요한지
무엇을 해결하는지
어떤 부작용이 있는지
구현 난이도는 어떤지
내 목표와 얼마나 잘 맞는지
를 평가해줘.
질문 6.

내 최종 목표는 “cat and dog rolling together” 같은 shot이다.
이 목표를 기준으로 봤을 때, 아래 두 방향 중 어느 쪽이 더 맞는지 비판적으로 판단해줘.

방향 A:

3D volume을 배우고
first-hit visible projection으로 2D를 condition
visible ownership을 강하게 유지
chimera를 막는다

방향 B:

3D entity-aware latent field를 유지하되
visible projection과 identity preservation을 분리
최종 visible은 hard ownership일 수 있어도 내부 latent는 amodal하게 유지
overlap/contact 동안에도 두 entity identity가 동시에 살아있게 한다

각 방향의 장단점, failure mode, 내 목표와의 정합성을 비교해줘.
그리고 가능하면 “최종적으로는 A에서 시작하되 B의 요소를 반드시 넣어야 한다” 같은 hybrid 제안도 해줘.

분석 형식 요구사항

응답은 반드시 아래 순서를 따라줘.

문제의 본질 한 문단 요약
현재 구조와 목표의 불일치 진단
single-entity collapse의 가능한 원인 트리
가장 가능성 높은 원인 Top 3
즉시 해야 할 진단 로그/시각화
다음 5개 실험 우선순위
단기 수정안
중기 구조 수정안
장기적으로 연구 방향 자체를 어떻게 정리해야 하는지
최종 권고안: 지금 당장 무엇부터 고칠지
매우 중요한 제약
나는 단순한 general explanation이 아니라 내 현재 코드베이스와 실험 상태를 고려한 현실적인 디버깅/연구 전략을 원한다.
“더 많은 데이터”, “더 오래 학습” 같은 일반론으로 흐리지 말고, 왜 single entity가 나오는지 구조적으로 파악해줘.
특히 3D 정보를 2D projection에 활용하면 chimera를 줄일 수 있다는 원래 아이디어가 어디까지 맞고, 어디서 current implementation이 그 아이디어를 배반하고 있는지를 가장 중요하게 다뤄줘.
내 목표는 layered occlusion만이 아니라 rolling/contact/collision처럼 시간에 따라 depth 관계가 바뀌는 실제 shot이다.
따라서 정적인 visible mask metric만 보지 말고, 동적인 entity identity preservation 관점에서 판단해줘.
가능하면 코드 수준으로도 어디를 의심해야 할지 언급해줘:
volume predictor
projection module
conditioning/injection
trainer/stage schedule
objective/contract design

마지막에는 반드시
**“내 목표를 가장 잘 만족하는 현실적인 추천 아키텍처”**를 하나 골라서 제안해줘.
그리고 왜 그것이 지금 코드베이스에서 가장 실현 가능하면서도 chimera 문제를 줄일 수 있는지 설명해줘.

같이 참고해야 할 현재 상태 요약
최종 목표: 두 entity가 뒹굴고 접촉하는 dynamic shot에서 chimera 없는 2D 생성
현재 구조: 3D entity volume + first-hit projection
현재 철학: no transparency, no soft blending, one pixel = one entity
현재 문제: single entity collapse / one object dominance / low render IoU
현재 관찰:
일부 identity separation 신호는 있음
하지만 공간 위치 제어와 render binding이 약함
occlusion과 collision 문제가 섞여 있었음
guide injection이 실제 generation path를 충분히 바꾸지 못했을 가능성이 큼
same-depth contact를 current hard visible ownership 구조만으로 풀기 어려울 수 있음

이 모든 조건을 고려해서, 내 연구 목표를 살리면서도 현실적으로 고칠 수 있는 가장 정교한 분석과 제안을 해줘.