# Related Papers — VCA-Diffusion

10편의 관련 논문을 VCA와의 차이점 중심으로 정리한다.

---

## Attend-and-Excite (Chefer et al., SIGGRAPH 2023)

- **핵심 아이디어**: Diffusion 추론 시 attention map의 최솟값을 최대화하는 gradient-based 조작으로 텍스트 프롬프트의 모든 entity가 생성 이미지에 반드시 등장하도록 강제한다.
- **해결 못한 문제**: 추론 시간에만 작용하므로 학습된 가중치는 변경되지 않는다. 두 entity가 공간적으로 겹쳐야 하는 경우(occlusion)의 depth ordering을 제어할 수 없다.
- **VCA와의 차이점**: Attend-and-Excite는 기존 cross-attention의 Softmax를 그대로 사용하고, attention을 사후적으로 조작한다. VCA는 Sigmoid 기반 독립 밀도와 Transmittance로 occlusion을 물리적으로 모델링하며, 학습 중 LoRA를 통해 가중치를 수정한다. 또한 Attend-and-Excite는 단일 프레임에 적용되는 반면 VCA는 비디오(AnimateDiff) 프레임 전체에 걸쳐 일관된 entity를 유지한다.
- **참고할 기술 요소**: attention map에서 token별 최대값을 추적하는 `aggregate_attention()` 함수 구현 방식; 수렴 기준으로 사용하는 smooth max 손실 공식.

---

## PEEKABOO (Jain et al., CVPR 2024)

- **핵심 아이디어**: 공간적으로 한정된 bounding box 조건부 생성을 위해, 각 entity가 지정된 영역 내에서만 생성되도록 attention 분포를 bounding box 마스크로 억제한다.
- **해결 못한 문제**: bounding box는 2D 평면 정보만 담고 있어 entity 간의 3D depth 관계를 표현하지 못한다. 두 bounding box가 겹치면 어느 entity가 앞에 있는지 결정할 수 없다.
- **VCA와의 차이점**: PEEKABOO는 2D 공간 마스크를 외부 조건으로 사용하지만 VCA는 z_bins 차원을 추가해 depth 축을 내재적으로 모델링한다. VCA에서는 bounding box 없이 Transmittance만으로 occlusion이 자연히 결정된다. 또한 PEEKABOO는 추론 시 guidance scale 조정에 의존하지만 VCA는 학습 시 LoRA branch를 통해 depth 표현을 직접 학습한다.
- **참고할 기술 요소**: bounding box를 attention mask로 변환하는 방법; spatial token 집합에 대한 conditional masking 구현.

---

## TokenFlow (Geyer et al., ICLR 2024)

- **핵심 아이디어**: 비디오 편집 시 인접 프레임 간 self-attention feature를 공유해 시간적 일관성을 유지한다. 키프레임의 feature를 다른 프레임에 전파함으로써 flickering 없이 편집 효과를 전파한다.
- **해결 못한 문제**: TokenFlow는 시간 일관성에는 강하지만 entity 수가 늘어날 때 각 entity를 독립적으로 추적하지 못한다. 두 entity가 교차하는 순간의 occlusion 처리가 없다.
- **VCA와의 차이점**: TokenFlow는 self-attention 공유로 프레임 간 일관성을 달성하지만 VCA는 cross-attention의 volumetric 구조로 entity별 depth 일관성을 달성한다. VCA의 sigma와 Transmittance는 프레임마다 독립적으로 계산되지만 entity 임베딩이 일정하므로 시간 일관성은 CLIP 임베딩 수준에서 보장된다. TokenFlow는 single entity를 가정하는 반면 VCA는 N entities를 명시적으로 처리한다.
- **참고할 기술 요소**: feature 공유를 위한 `extended_attention()` 패턴; 배치 내 프레임 간 key/value 재사용 방식.

---

## CONFORM (Meral et al., 2024)

- **핵심 아이디어**: 여러 entity를 생성할 때 각 token의 attention이 다른 entity와 겹치지 않도록, attention map 간 상호 정보를 최소화하는 contrastive 손실을 추가한다.
- **해결 못한 문제**: entity 간 attention 독립성을 손실 함수로 강제하지만, 이는 두 entity가 같은 픽셀을 점유하는 상황(occlusion)에서는 실패한다. depth 관계 모델링이 없어 겹침 순서를 제어할 수 없다.
- **VCA와의 차이점**: CONFORM은 Softmax cross-attention 위에 외부 contrastive 손실을 추가하는 방식이고, VCA는 attention 구조 자체를 Sigmoid + Transmittance로 교체해 entity 독립성과 occlusion을 동시에 달성한다. CONFORM의 contrastive 손실은 entity들이 겹치는 것을 막으려 하지만 VCA는 겹침을 허용하면서 depth ordering을 물리적으로 해결한다.
- **참고할 기술 요소**: pairwise attention map KL divergence 계산; 역전파 가능한 attention 독립성 지표 구현.

---

## NeRF (Mildenhall et al., ECCV 2020)

- **핵심 아이디어**: 임의의 시점에서 연속적인 volumetric scene representation을 MLP로 학습하고, volume rendering 적분으로 픽셀 색상을 합성한다. 핵심 공식은 ray를 따라 샘플링한 점들의 밀도(σ)와 색상(c)을 transmittance로 가중합하는 것이다.
- **해결 못한 문제**: NeRF는 단일 정적 씬을 암시적으로 표현하며, 언어 조건부 생성이나 entity별 제어가 없다. 학습에 수백 장의 다시점 이미지가 필요하다.
- **VCA와의 차이점**: VCA는 NeRF의 transmittance 공식을 직접 차용하지만, 연속적인 ray sampling 대신 이산적인 z_bins를 사용한다. NeRF의 σ는 MLP에서 나오지만 VCA의 σ(sigma)는 sigmoid(Q·K/√D)로 attention score에서 직접 계산된다. VCA는 NeRF의 3D geometry 학습이 아니라 entity-aware attention weighting에 transmittance를 적용한다.
- **참고할 기술 요소**: exclusive cumprod를 이용한 transmittance 계산 (T[z=0]=1 보장); opacity 누적 공식 `T[k] = ∏_{i<k}(1-σᵢ)`.

---

## LoRA (Hu et al., ICLR 2022)

- **핵심 아이디어**: 대형 언어/비전 모델의 파인튜닝 시 전체 가중치 행렬 대신 저랭크 행렬 두 개(A, B)의 곱만 학습한다. W_new = W_frozen + B·A로 파라미터 효율적 적응을 달성한다.
- **해결 못한 문제**: LoRA 자체는 entity-aware attention이나 depth 모델링을 다루지 않는다. 어떤 가중치를 학습해야 하는지의 선택 문제는 남아 있다.
- **VCA와의 차이점**: VCA는 LoRA를 K, V projection에 적용해 entity-specific CLIP 임베딩 적응만 학습한다. Q projection과 기반 모델의 나머지 가중치는 frozen 상태를 유지해 원래 생성 능력을 보존한다. LoRA 원 논문의 초기화 방식(lora_B=zeros, lora_A=small randn)을 그대로 채택해 학습 초기 LoRA 기여가 0이 되도록 한다.
- **참고할 기술 요소**: `LoRALinear` 구현에서 `requires_grad=False`로 base weight 동결; `F.linear(F.linear(x, lora_A), lora_B)` 체이닝.

---

## AnimateDiff (Guo et al., ICLR 2024)

- **핵심 아이디어**: 기존 Stable Diffusion에 temporal attention 모듈을 삽입해 비디오를 생성한다. 시간 축 self-attention으로 프레임 간 motion consistency를 학습하고, 기존 spatial 모듈은 frozen 유지한다.
- **해결 못한 문제**: AnimateDiff는 단일 텍스트 프롬프트로 전체 씬을 제어하며, 여러 entity의 독립적인 motion이나 상호 occlusion을 다루지 않는다. 두 entity가 교차할 때 depth ordering이 붕괴한다.
- **VCA와의 차이점**: VCA는 AnimateDiff를 백본으로 사용하며, 기존 temporal attention은 유지하고 cross-attention만 VCALayer로 교체한다. AnimateDiff의 프레임 배치 처리 방식(BF × S × D)을 그대로 따르며, VCA의 context_dim=768은 SD 1.5 기반 AnimateDiff에 맞춘 것이다(SDXL 기반이면 2048).
- **참고할 기술 요소**: temporal attention 삽입 위치(UNet의 ResBlock 이후); BF(batch×frames) 텐서 형태로 처리하는 배치 관례.

---

## VideoComposer (Wang et al., NeurIPS 2023)

- **핵심 아이디어**: 비디오 생성 시 다양한 조건(sketch, depth, motion, style)을 동시에 주입할 수 있는 composable 프레임워크를 제안한다. 각 조건을 별도 encoder로 처리하고 STC-encoder로 통합한다.
- **해결 못한 문제**: 각 조건은 전체 씬에 적용되며 entity별 독립 제어가 없다. depth 조건은 외부 depth map을 입력으로 받아야 하며, 모델 스스로 entity depth를 학습하지 않는다.
- **VCA와의 차이점**: VideoComposer는 depth를 외부 입력 조건으로 취급하지만 VCA는 depth를 z_bins 차원으로 모델 내부에 내재화한다. VCA는 depth map 없이도 sigma의 Transmittance 가중치로 depth ordering을 자동 결정한다. 또한 VideoComposer는 multi-condition fusion을 다루지만 VCA는 특히 multi-entity cross-attention에 집중한다.
- **참고할 기술 요소**: 다중 조건 embedding을 하나의 context vector로 합치는 concatenation 방식; 각 조건 encoder의 temporal alignment 처리.

---

## MotionDirector (Zhao et al., 2024)

- **핵심 아이디어**: 참조 비디오로부터 특정 object의 motion pattern을 추출해, 새로운 텍스트 프롬프트에 그 motion을 적용한다. spatial LoRA로 appearance, temporal LoRA로 motion을 분리 학습한다.
- **해결 못한 문제**: single object의 motion 학습에 특화되어 있어 두 entity의 상호작용(교차, occlusion)을 다루지 못한다. 두 object의 motion이 충돌하는 경우 처리 방법이 없다.
- **VCA와의 차이점**: MotionDirector는 motion 학습을 위한 LoRA이고 VCA는 entity-aware attention을 위한 LoRA다. MotionDirector의 temporal LoRA는 시간 축 attention에 적용되지만 VCA의 LoRA는 cross-attention의 K, V에 적용되어 entity 임베딩 적응을 담당한다. VCA는 entity 간 motion 충돌을 Transmittance로 물리적으로 해결한다.
- **참고할 기술 요소**: spatial/temporal LoRA를 분리 학습하는 스케줄; 참조 비디오에서 motion prior를 추출하는 방식.

---

## FLATTEN (Cong et al., ICLR 2024)

- **핵심 아이디어**: 비디오 편집 시 optical flow를 이용해 프레임 간 대응점을 찾고, 대응하는 token들 사이에서만 attention을 계산해 시간 일관성을 보장한다. Flow-guided attention으로 moving object의 일관된 편집을 달성한다.
- **해결 못한 문제**: optical flow 기반이므로 두 object가 교차해 flow가 불명확해지는 상황에서 attention이 혼란스러워진다. entity별 독립적인 attention이 아니라 flow 대응점 기반이므로 occlusion 경계에서 artifact가 발생한다.
- **VCA와의 차이점**: FLATTEN은 flow correspondences를 이용한 spatial token matching으로 시간 일관성을 달성하지만 VCA는 entity CLIP 임베딩 + depth_pe로 entity identity를 명시적으로 인코딩한다. VCA는 optical flow 없이도 entity 추적이 가능하며, occlusion이 발생해도 Transmittance가 자연스러운 depth weighting을 제공한다. FLATTEN이 VCA보다 dense correspondence에 강하지만 entity semantics를 다루지 않는다.
- **참고할 기술 요소**: flow field에서 token correspondence를 추출하는 warping 방식; attention mask를 flow-guided로 구성하는 구현.

---

## 요약 비교표

| 논문 | Attention 방식 | Depth 처리 | Entity 독립성 | VCA 차이점 핵심 |
|------|--------------|-----------|-------------|----------------|
| Attend-and-Excite | Softmax + 추론 조작 | 없음 | 최대화 최솟값 | VCA는 학습 시 Sigmoid 구조로 해결 |
| PEEKABOO | Softmax + bbox 마스크 | 2D only | 없음 | VCA는 z_bins로 3D depth 내재화 |
| TokenFlow | Self-attention 공유 | 없음 | 없음 | VCA는 cross-attention volumetric |
| CONFORM | Softmax + contrastive 손실 | 없음 | 손실로 강제 | VCA는 구조 자체가 독립성 보장 |
| NeRF | — | Ray marching | — | VCA가 transmittance 수식 직접 차용 |
| LoRA | — | — | — | VCA가 K,V에 entity 적응용 LoRA 적용 |
| AnimateDiff | Temporal self-attn | 없음 | 없음 | VCA의 백본, cross-attn만 교체 |
| VideoComposer | Multi-condition | 외부 입력 | 없음 | VCA는 depth를 내재화 |
| MotionDirector | Spatial/temporal LoRA | 없음 | 없음 | VCA는 entity interaction에 집중 |
| FLATTEN | Flow-guided attn | 없음 | 없음 | VCA는 flow 없이 entity 추적 |
