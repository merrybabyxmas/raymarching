# Chimera Experiment Plan: Depth-Ordered Occlusion for Video Generation

## 1. Problem Definition

**2-Entity Collision Chimera Problem**: AnimateDiff 등 video diffusion model에서 두 entity가 영상 내에서 충돌/겹칠 때, overlap 영역에서 두 entity의 feature가 섞인 "키메라(chimera)" 아티팩트가 발생한다. 이는 cross-attention이 overlap 영역에서 두 entity 토큰에 동시에 attend하기 때문이다.

**목표**: VCA(Volumetric Cross-Attention)가 학습한 depth ordering 정보를 이용해 chimera를 제거한다.
- VCA sigma값: 각 spatial token에서 어느 entity가 "앞"에 있는지 이미 알고 있음
- 이 정보를 활용해 overlap 영역에서 "뒤" entity의 cross-attention 기여를 억제한다

---

## 2. Experimental Setup

### 2.1 Backbone 및 체크포인트

- **Video Model**: AnimateDiff (guoyww/animatediff-motion-adapter-v1-5-2) + epiCRealism
- **Trained VCA**: `checkpoints/phase31/best.pt` (Phase 31, DRA=86%, gen_diff=48px)
- **Injection point**: `up_blocks.2.attentions.0.transformer_blocks.0.attn2` (16×16 spatial)
- **Gamma trained**: ≈ 1.077

### 2.2 Collision Prompt 설계

Chimera가 확실히 나타나는 프롬프트 조건:
- 두 entity가 **서로를 향해 움직이는** 동작 기술
- **색상 구분** (red vs blue) → 픽셀 수준 chimera 검출 가능
- Entity는 단순하고 명확한 형태 (구체, 동물, 로봇)

```python
COLLISION_PROMPTS = [
    {
        "prompt": "a red ball and a blue ball rolling toward each other "
                  "on a wooden table, they collide in the center, "
                  "cinematic lighting, 4k, photorealistic",
        "negative": "low quality, blurry, deformed, watermark, text, chimera, "
                    "merged, fused, artifact",
        "entity0": "red ball",
        "entity1": "blue ball",
        "color0": "red",
        "color1": "blue",
    },
    {
        "prompt": "a red cat and a blue cat running toward each other "
                  "on a grassy field, they meet in the middle, "
                  "cinematic, high detail, photorealistic",
        "negative": "low quality, blurry, deformed, watermark, text, chimera, "
                    "merged, fused, artifact, cartoon",
        "entity0": "red cat",
        "entity1": "blue cat",
        "color0": "red",
        "color1": "blue",
    },
    {
        "prompt": "a shiny red sphere and a shiny blue sphere floating "
                  "toward each other in space, collision course, "
                  "dramatic lighting, studio background, 8k",
        "negative": "low quality, blurry, text, watermark, chimera, merged",
        "entity0": "red sphere",
        "entity1": "blue sphere",
        "color0": "red",
        "color1": "blue",
    },
]
```

### 2.3 Chimera Detection Metric

두 entity가 색상(red vs blue)으로 구분되므로, RGB 분석으로 chimera 픽셀을 검출한다.

```
chimera_pixel: R_channel > 80 AND B_channel > 80 (at same pixel)
overlap_region: sigma_E0 > threshold AND sigma_E1 > threshold

chimera_score = sum(chimera_pixels in overlap_region) / sum(overlap_region)
  → 0.0 = no chimera (perfect occlusion)
  → 1.0 = complete chimera (full blending)
```

### 2.4 GIF 출력 형식

각 method당 생성되는 GIF:
1. **`{method}_frames.gif`**: 8프레임 생성 영상 (256×256)
2. **`{method}_overlay.gif`**: 프레임 + sigma overlay (E0=빨강, E1=파랑 반투명)
3. **`{method}_chimera_mask.gif`**: chimera 검출 마스크 (노랑=키메라, 흰=정상)

최종 비교 GIF:
- **`chimera_comparison.gif`**: [Baseline | Occlusion | Guidance | Compositing] 4열 side-by-side
- 각 열: 생성 프레임 + 하단에 chimera_score 표시

---

## 3. Method 0: Baseline (VCA without occlusion correction)

**목적**: Chimera 문제가 실제로 존재함을 보인다.

**설정**:
- Phase 31 VCA 로드, `AdditiveVCAInferProcessor` (gamma_trained=1.077)
- 표준 AnimateDiff inference (20 steps, guidance_scale=7.5)
- N=5 랜덤 seed로 chimera가 가장 명확히 보이는 케이스 선택

**기대 결과**: overlap 영역에서 red+blue 혼합 픽셀 발생, chimera_score ≈ 0.3–0.6

---

## 4. Method 1: Attention Occlusion Masking (Phase 32)

**핵심 아이디어**: VCA sigma가 "앞 entity"를 알고 있으므로, overlap spatial token에서 "뒤 entity"의 text cross-attention weight를 0으로 억제한다.

**메커니즘**:
```
For each spatial token s at frame fi:
  if sigma[fi, s, E0, z=0] > sigma[fi, s, E1, z=0]:  ← E0 앞
    attn_weights[fi, s, tok_E1_indices] *= suppression_factor  ← E1 억제
  else:  ← E1 앞
    attn_weights[fi, s, tok_E0_indices] *= suppression_factor  ← E0 억제
```

**구현 세부사항**:
- `OcclusionVCAProcessor`: up_blocks.2 attn2에서 직접 attention weight 수정
- `suppression_factor = 0.0` (hard suppression, ablation: 0.1, 0.3도 테스트)
- Sigma는 해당 타임스텝에서 VCA forward 결과 사용 (within same call)
- 추가로 `OcclusionPropagator`: mid_block + up_blocks.0/1/3의 cross-attn에도 sigma-derived mask 전파
  - sigma는 16×16 → 각 레이어 해상도로 bilinear upscale

**핵심 주의사항**:
- `to_q`/`to_k`/`to_v`를 명시적으로 재계산해 attention weight에 접근 필요
- fp16 overflow 방지: weight 계산 시 float32 사용 후 변환
- `beta=0` for baddbmm (NaN 방지)

**설정**:
- suppression_factor ∈ {0.0, 0.1, 0.3} — 3가지 강도 비교
- 최적 값으로 최종 GIF 생성

**기대 결과**: chimera_score ≈ 0.05–0.15 (베이스라인 대비 큰 감소)

---

## 5. Method 2: Score/Gradient Guidance (Phase 33)

**핵심 아이디어**: 학습된 VCA를 depth ordering oracle로 사용해, 각 denoising step에서 depth rank loss의 gradient를 latent에 추가한다. Training-free, 어떤 video model에도 적용 가능.

**수식**:
```
ε̃_θ(x_t) = ε_θ(x_t) - λ_guide · ∇_{x_t} L_depth(VCA(x_t))
```

여기서 `L_depth = l_zorder_direct(sigma_acc, depth_orders, entity_masks)`

**구현 세부사항**:
```python
with torch.enable_grad():
    latents_grad = latents.detach().requires_grad_(True)
    noise_pred = unet(latents_grad, t, encoder_hidden_states=enc_hs).sample
    
    # VCA sigma는 UNet forward 중 자동 계산됨
    depth_loss = l_zorder_direct(vca_layer.sigma_acc, depth_orders, entity_masks)
    grad = torch.autograd.grad(depth_loss, latents_grad)[0]

noise_pred_guided = noise_pred.detach() - guidance_scale * grad.detach()
latents = scheduler.step(noise_pred_guided, t, latents).prev_sample
```

**설정**:
- `guidance_scale` ∈ {0.1, 0.5, 1.0, 2.0} — 4가지 강도 비교
- 매 denoising step마다 적용 (20 steps × gradient computation)
- `depth_orders`: entity pair 충돌 예상 순서 (E0이 앞으로 설정)
- Gradient clipping: 0.1 norm 이하로 클리핑 (stability)

**기대 결과**: chimera_score ≈ 0.10–0.20, 약간 blur 증가 가능

---

## 6. Method 3: Entity-Specific Latent Compositing (Phase 34)

**핵심 아이디어**: 각 entity를 별도로 denoising하고, VCA sigma로 생성된 depth mask에 따라 frame-level compositing.

**파이프라인**:
```
1. Generate video_E0: prompt_E0만으로 생성 (entity0 only)
2. Generate video_E1: prompt_E1만으로 생성 (entity1 only)
3. Generate video_base: full prompt (baseline, VCA 포함)
4. Extract sigma mask from video_base generation
5. Composite:
   frame_final[fi, y, x] = video_E0[fi, y, x]  if sigma_E0 > sigma_E1 at (y,x)
                          = video_E1[fi, y, x]  otherwise
6. Temporal smoothing: Gaussian blur on mask (σ=2) to prevent hard edges
```

**구현 세부사항**:
- Entity-only prompts: "a red ball rolling to the left" (entity0만 있는 자연스러운 씬)
- Sigma mask: 16×16 → 256×256 bilinear upscale + Gaussian smooth (sigma=2)
- Temporal consistency: mask를 시간축으로도 smooth (1D Gaussian over frames)
- Edge blending: mask soft boundary로 Poisson blending 효과

**기대 결과**: chimera_score ≈ 0.0–0.05 (거의 완벽한 분리)
단점: entity 간 occlusion 경계에서 incoherence 발생 가능

---

## 7. Evaluation & Comparison

### 정량 지표
| Method | Chimera Score ↓ | Frame Quality (PSNR) | Temporal Consistency |
|--------|----------------|---------------------|---------------------|
| Baseline | ~ | ~ | ~ |
| Occlusion (M1) | ~ | ~ | ~ |
| Guidance (M2) | ~ | ~ | ~ |
| Compositing (M3) | ~ | ~ | ~ |

### 정성 비교 GIF
`debug/chimera/chimera_comparison.gif`:
- 4열: Baseline | Occlusion | Guidance | Compositing
- 2행: 생성 프레임 | chimera mask (노랑=키메라)
- 프레임 하단: chimera_score 텍스트

---

## 8. 실행 스크립트

```bash
# Phase 32: Occlusion Masking
python scripts/chimera_phase32_occlusion.py \
  --ckpt checkpoints/phase31/best.pt \
  --debug-dir debug/chimera \
  --n-seeds 5 --suppression 0.0

# Phase 33: Score Guidance  
python scripts/chimera_phase33_guidance.py \
  --ckpt checkpoints/phase31/best.pt \
  --debug-dir debug/chimera \
  --guidance-scale 1.0

# Phase 34: Entity Compositing
python scripts/chimera_phase34_compositing.py \
  --ckpt checkpoints/phase31/best.pt \
  --debug-dir debug/chimera

# Comparison
python scripts/chimera_compare.py \
  --debug-dir debug/chimera
```

---

## 9. 예상 실험 결과 및 논문 기여

**Claim**: "VCA의 depth ordering 예측을 통해 video diffusion model의 2-entity collision chimera 문제를 해결한다."

**Evidence**:
1. Baseline: chimera_score 높음 → 문제 실재 확인
2. Method 1 (Occlusion): 가장 이론적으로 깔끔, attention level에서 chimera 제거
3. Method 2 (Guidance): training-free, model-agnostic, 범용성
4. Method 3 (Compositing): 가장 강력하지만 temporal incoherence 단점

**핵심 메시지**: VCA는 단순한 depth estimator가 아니라, video generation의 occlusion ordering을 직접 제어하는 모듈이다.
