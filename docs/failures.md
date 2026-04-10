# Failures.md — VCA-Diffusion 실패 기록

Phase 1~4 진행 중 발생한 실패 모드를 이론적/실제적으로 문서화한다.

---

## 이론적 실패 모드

### FM-T1: Softmax 사용 시 Disappearance

**현상**: cross-attention에서 Softmax를 사용하면 두 entity가 겹치는 픽셀에서 한 entity의 attention weight가 올라갈수록 다른 entity의 weight는 강제로 0에 수렴한다. 모델은 한 entity를 아예 렌더링하지 않는 방향으로 수렴해 "객체 소실(Disappearance)" 현상이 발생한다.

**원인**:
```
w[n] = exp(Q·K[n]) / sum_n exp(Q·K[n])   # zero-sum: 모든 w[n]의 합 = 1
```
두 entity가 경쟁 관계이므로 entity 0의 weight가 높아지면 entity 1의 weight는 1 - w[0]이 된다.

**해결(fix)**: Softmax를 Sigmoid로 교체. 각 entity는 독립적으로 [0,1] 밀도를 가진다.
```python
sigma = torch.sigmoid(Q @ K.T / sqrt(D))   # 각 entity 독립적 → 합이 1 아니어도 됨
```
`test_entities_independent` 테스트가 이를 강제 검증한다.

---

### FM-T2: Transmittance off-by-one (T[z=0] ≠ 1.0)

**현상**: T[z=0]이 1.0이 아니면 "아무것도 가리지 않은 상태"에서도 빛이 감쇠한다. z=0에 있는 entity가 자신이 가려진 것처럼 처리되어 weight가 비정상적으로 낮아진다.

**원인**:
```python
# WRONG: z=0이 (1 - opacity[0])를 받아버림 → T[z=0] < 1
T = torch.cumprod(1.0 - opacity, dim=-1)
```

**해결(fix)**: exclusive cumprod 사용. z=0 앞에 1.0을 prepend하고 마지막을 제거한다.
```python
# CORRECT: T[z=0]=1, T[z=1]=1-opacity[0], T[z=2]=(1-opacity[0])*(1-opacity[1])
ones = torch.ones(*opacity.shape[:2], 1, device=opacity.device)
T = torch.cumprod(torch.cat([ones, 1.0 - opacity], dim=-1), dim=-1)[:, :, :-1]
```
`test_T_z0_is_one` 테스트가 이를 검증한다.

---

### FM-T3: opacity clamp 누락 (T가 음수)

**현상**: 여러 entity의 sigma를 합산할 때 합이 1을 초과하면 `1 - opacity`가 음수가 된다. cumprod에서 음수가 들어가면 T가 음수 또는 NaN이 될 수 있어 weight가 비물리적이 된다.

**원인**:
```python
opacity = sigma.sum(dim=2)   # clamp 없음: 여러 entity의 합 > 1 가능
T = cumprod(1.0 - opacity)   # 1.0 - 1.5 = -0.5 → 음수 transmittance
```

**해결(fix)**: sum 후 max=1.0으로 클램핑.
```python
opacity = sigma.sum(dim=2).clamp(max=1.0)   # 물리적 범위 [0,1] 보장
```

---

### FM-T4: L_ortho 없을 때 z=0 몰림

**현상**: depth_pe가 무작위로 초기화되고 정규화 손실이 없으면, 모든 entity가 z=0 슬라이스로 몰리는 경향이 있다. 이 경우 depth ordering이 학습되지 않고 sigma[z=0] ≫ sigma[z=1]로 고정된다.

**원인**: depth positional encoding이 학습 중 서로 비슷해지면 z_bins 간 구별이 사라진다. entity가 항상 같은 depth slice를 선호하면 transmittance의 occlusion 효과가 무의미해진다.

**해결(fix)**: orthogonality loss를 depth_pe에 추가한다.
```python
L_ortho = (depth_pe @ depth_pe.T - eye).pow(2).mean()
loss = L_recon + lambda_ortho * L_ortho
```
혹은 depth_pe를 초기에 직교 행렬로 초기화한다.

---

## 통합 실패 모드

### FM-A1: diffusers hook 미호출

**현상**: AnimateDiff UNet에 VCALayer를 삽입할 때 기존 cross-attention hook을 올바르게 등록하지 않으면 VCALayer가 순전파에서 아예 호출되지 않는다. 손실은 계산되지만 기존 cross-attention이 그대로 동작해 VCA 효과가 없다.

**원인**: diffusers의 `attn_processor` 등록 없이 모듈만 교체하거나, `register_forward_hook()` 위치가 잘못된 경우.

**해결(fix)**: `unet.set_attn_processor()` 또는 `attn_processor` dict를 명시적으로 지정한다.
```python
from diffusers.models.attention_processor import AttnProcessor2_0
processors = {k: VCAProcessor(...) for k in unet.attn_processors.keys() if 'cross' in k}
unet.set_attn_processor(processors)
```

---

### FM-A2: float16/float32 dtype 불일치

**현상**: AnimateDiff 파이프라인은 보통 fp16으로 실행되는데, VCALayer의 `depth_pe`가 fp32이면 Q·K 내적에서 dtype mismatch 에러가 발생한다. 특히 `torch.einsum`에서 fp16 × fp32 연산 시 RuntimeError.

**원인**: `nn.Parameter(torch.randn(...))` 기본값이 fp32인데, 파이프라인이 `.half()`를 호출하면 `nn.Parameter`는 변환되지만 일부 연산 중간값이 캐시된 경우 mismatch.

**해결(fix)**: 명시적 dtype 캐스팅 또는 autocast 적용.
```python
# forward 내부에서 input dtype으로 캐스팅
depth_pe = self.depth_pe.to(x.dtype)
ctx3d = ctx.unsqueeze(2) + depth_pe.unsqueeze(0).unsqueeze(0)
```

---

## 실제 발생한 실패 (FM-I)

### FM-I1: pyvista Camera.view_up AttributeError (Phase 2)

**현상**: `pl.camera.view_up = (0, 0, 1)` 실행 시 `PyVistaAttributeError: Attribute 'view_up' does not exist and cannot be added to class 'Camera'` 발생. generate_toy_data.py가 returncode 1로 종료.

**원인**: pyvista 0.47.x에서 Camera 클래스에 `__setattr__` 보호가 추가되어 직접 속성 할당이 불가해졌다. `view_up`은 getter는 있지만 setter가 없는 읽기 전용 property로 변경되었다.

**해결(fix)**: `pl.camera_position = [camera_pos, focal_point, view_up]` 튜플 형태로 position, focal_point, view_up을 한 번에 지정.
```python
# WRONG
pl.camera.position = camera_pos
pl.camera.focal_point = focal_point
pl.camera.view_up = view_up

# CORRECT (pyvista 0.47+)
pl.camera_position = [camera_pos, focal_point, view_up]
```

---

### FM-I2: imageio.v3.get_writer AttributeError (Phase 2)

**현상**: `iio.get_writer(str(video_path), fps=fps, codec="libx264", quality=7)` 실행 시 `AttributeError: module 'imageio.v3' has no attribute 'get_writer'` 발생.

**원인**: imageio v3 API는 `imwrite`/`imread`/`imopen` 등 함수형 인터페이스만 제공하며, `get_writer` 같은 스트리밍 writer 인터페이스는 imageio.v2 (레거시 API)에만 존재한다.

**해결(fix)**: 비디오 스트리밍 writer는 `imageio.v2.get_writer()` 사용.
```python
import imageio.v2 as iio2
writer = iio2.get_writer(str(video_path), fps=fps, codec="libx264", quality=7)
for p in pngs:
    writer.append_data(iio.imread(str(p)))  # 프레임 읽기는 v3 사용
writer.close()
```

---

### FM-I3: pyvista z-buffer 전체 NaN (Phase 2)

**현상**: `pl.get_image_depth()` 반환값이 전부 NaN (64183/65536 픽셀). `depth[mask].mean()` 계산 시 NaN 전파로 `test_depth_ordering_in_overlap` 실패.

**원인**: DISPLAY 없는 headless 환경에서 OpenGL XWindow 렌더윈도우(`vtkXOpenGLRenderWindow`)가 X 서버 연결 경고를 발생시키며 z-buffer를 정상적으로 읽지 못한다. EGL fallback 렌더링은 color buffer는 정상이지만 depth buffer 접근이 실패한다.

**해결(fix)**: z-buffer 대신 메시의 per-vertex 유클리드 거리를 scalar로 렌더링해 depth map을 구성하는 `_render_depth()` 메서드 구현. color channel에 depth를 인코딩([d_min, d_max] → [0, 255])하여 RGB screenshot에서 읽어낸다.
```python
m2['depth_scalar'] = np.linalg.norm(mesh.points - cam_pos, axis=1)
pl.add_mesh(m2, scalars='depth_scalar', cmap='gray', clim=[d_min, d_max])
encoded = pl.screenshot(return_img=True)[..., 0]  # R channel
depth = d_min + (encoded / 255.0) * d_range
depth[encoded == 0] = 0.0  # 배경 제거
```

---

### FM-I4: anti-symmetric context가 Sigmoid 항등식을 건드린 test_entities_independent (Phase 1)

**현상**: `test_entities_independent`가 전체 pytest 실행 시 간헐적으로 실패한다. `diff=0.0021` 같이 0에 가까운 값이 나오며 `assert diff > 0.01` 실패. 개별 `-m phase1` 실행에서는 통과한다.

**원인**: 수학적 항등식 충돌. 테스트 코드가 다음 설정을 사용:
```python
ctx[:, 0, :] = +5.0   # entity 0
ctx[:, 1, :] = -5.0   # entity 1
```

LoRALinear의 초기 `lora_B = zeros` 상태에서:
```
K_entity0 = W @ (+5.0) = +5 * Σw
K_entity1 = W @ (-5.0) = -5 * Σw = -K_entity0   ← 정확히 반대 방향
```

Sigmoid 항등식: `sigmoid(x) + sigmoid(-x) = 1` 이므로:
```
σ_entity0[s] + σ_entity1[s] = 1   (모든 spatial position에서 성립)
mean(σ_entity0) + mean(σ_entity1) = 1   (항상)
diff = |2 * mean(σ_entity0) - 1|   ← random walk around 0
```

이는 flaky test가 아니라 **테스트 설계 자체가 Sigmoid 항등식을 역이용해 diff ≈ 0을 만드는 구조**다. `diff > 0.01`이 통과할 확률은 이론적으로 약 50~87%로 결정론적이지 않다.

**해결(fix)**: anti-symmetric context 대신 동일 방향 context를 사용해 Sigmoid의 진짜 특성(두 entity가 동시에 높은 밀도를 가질 수 있음)을 검증한다.

```python
# WRONG: ±5로 anti-symmetric → sigmoid 항등식 → diff ≈ 0 (flaky)
ctx[:, 0, :] = +5.0
ctx[:, 1, :] = -5.0
diff = abs(mean(σ_entity0) - mean(σ_entity1))
assert diff > 0.01   # ← 항상 0에 가까움

# CORRECT: 동일 방향 context → 두 entity가 동시에 high sigma 가능 여부 검증
ctx[:, 0, :] = 5.0
ctx[:, 1, :] = 5.0
# K_entity0 = K_entity1 → 같은 점수 → 둘 다 높거나 둘 다 낮음
# σ_entity0 = σ_entity1 (동시에 high) → both_high ≈ 50%
# Softmax였다면: zero-sum → 한 entity가 high이면 다른 entity는 low → both_high ≈ 0%
both_high = ((σ_entity0 > 0.5) & (σ_entity1 > 0.5)).float().mean()
assert both_high > 0.1   # ← Sigmoid: ~50%, Softmax: ~0%
```

**논문 시사점**: 이 발견은 VCA가 Softmax 대비 갖는 핵심 강점을 테스트 레벨에서 직접 증명한다. Sigmoid에서는 두 entity가 같은 픽셀을 동시에 높은 밀도로 점유할 수 있고(transmittance가 depth ordering을 결정), Softmax에서는 구조적으로 불가능하다.

---

### FM-I5: Phase 16 λ_depth=1.0 → UNet feature 파괴 (kaleidoscope 현상)

**현상**: Phase 16 full training (λ_depth=1.0, 30 epoch, 1 sample/epoch) 후 비교 inference 시:
- `cat_dog` 프롬프트 → 고양이 4마리 kaleidoscope 타일링
- `fighters` 프롬프트 → 검은 X자 패턴

sigma_separation=0.31로 수치 지표는 좋았지만 생성 품질이 완전히 파괴됨.

**원인**:
1. λ_depth=1.0이 l_diff(≈0.4)와 동일 수준 → depth loss가 UNet feature를 압도
2. sigma가 극단값(0 또는 1)으로 수렴 → Transmittance weight 비정상 → 출력 폭발
3. 1 sample/epoch × 30 epoch → dog+sword 4회, red person 3회 반복 노출 → 과적합

**교훈**: `l_diff > l_depth_weighted × 10` 비율이 VCA의 불변 원칙 (FM-I7 참조).
체크포인트: `checkpoints/objaverse/best.pt`

---

### FM-I6: last_sigma (detach) 사용 시 gradient 미전달

**현상**: 학습 초기에 loss.backward()가 성공해도 VCA 파라미터 gradient가 0으로 남아 학습이 전혀 일어나지 않음. sigma 값이 epoch 내내 고정.

**원인**: `vca_layer.last_sigma`는 detach된 값이어서 computation graph와 연결이 끊겨 있음.
```python
# WRONG: detach — backward 시 gradient 미전달
sigma_raw_for_loss = vca_layer.last_sigma
l_depth = l_depth_ranking(sigma_raw_for_loss, order)  # grad = 0
```

**해결(fix)**: `last_sigma_raw` 사용 (grad 있음).
```python
# CORRECT: grad 있는 raw — backward 정상 작동
sigma_raw_for_loss = vca_layer.last_sigma_raw
l_depth = l_depth_ranking(sigma_raw_for_loss, order)
```

---

### FM-I7: Phase 17 λ_depth=0.02 → depth 감독 신호 너무 약함

**현상**: Phase 17 full training (λ_depth=0.02, 60 epoch, 168 samples/epoch) 결과:
- sigma_separation=0.263 (Phase 16의 0.313보다 낮음)
- Phase 16 대비: chain(-0.065), dancers(-0.021), snakes(-0.010) — toy/zero-shot에서 열세
- 생성 품질은 보존 (DEGRADED 0회, ratio 항상 75x 이상)

**원인**: λ_depth를 1/50으로 줄였더니 depth ordering 학습 자체가 불충분해짐.
l_depth_weighted가 l_diff 대비 너무 작아(ratio 75~8095x) VCA가 depth를 사실상 무시.

**교훈**: 최적 λ_depth는 Phase 16(1.0, 파괴)과 Phase 17(0.02, 미약) 사이 어딘가.
→ Phase 18에서 λ_depth ∈ {0.1, 0.2, 0.3} 그리드 탐색으로 최적값 결정.
체크포인트: `checkpoints/phase17/best.pt`

---

### FM-I8: sigma_separation 지표가 random t 노이즈를 측정 (Phase 16~18 공통)

**현상**: 학습 곡선에서 sigma_separation이 매 epoch 크게 진동 (0.004~0.346).
실제로 모델이 학습/망각을 반복하는 것처럼 보이지만, 지표 자체가 불안정한 것임.

**원인**: 매 epoch 랜덤 timestep `t` 1회에서만 forward → hidden_states가 t에 따라 완전히 달라짐.
sigma는 hidden_states 기반이므로 weight 변화가 아니라 **t의 노이즈**를 반영.
실제 학습 진행도를 측정하지 못함.

**해결 (Phase 19 Fix 2)**: 고정 probe × 5개 t 값 평균으로 probe_sep 측정.
`PROBE_T_VALUES = [20, 60, 100, 140, 180]`

---

### FM-I9: Majority vote depth order → 50% 역방향 gradient (Phase 16~18 공통)

**현상**: depth_reversal_rate=0.744로 8프레임 중 평균 6프레임이 depth가 뒤집히는 시퀀스.
8프레임 전체에 단일 majority vote 순서를 적용하면 절반 프레임에 역방향 gradient가 들어감.

**원인**:
```python
# 기존 코드 (train_objaverse_vca.py)
front_votes = sum(1 for d in depth_orders if d[0] == 0)
order = [0, 1] if front_votes >= len(depth_orders) // 2 else [1, 0]
l_depth = l_depth_ranking(sigma_raw, order)  # 모든 프레임에 같은 순서
```
depth_reversal_rate=0.744인 시퀀스에서 majority vote로 결정해도,
나머지 26%+교차 구간 프레임들에 틀린 감독 신호가 들어감.

**해결 (Phase 19 Fix 1)**: 프레임별 독립 depth ranking loss.
```python
# Phase 19 코드
for fi, order in enumerate(depth_orders):
    frame_sigma = sigma_acc[layer_idx][fi:fi+1]
    loss += l_depth_ranking(frame_sigma, order)
```

---

### FM-I10: Phase 19 6-layer inference injection → zero-shot generalization 감소

**현상**: Phase 19는 학습 시 INJECT_KEYS_P19 (6개 레이어) 주입 + 4가지 수정 적용.
inference 비교(compare_phase19.py) 결과 Phase 19는 8개 프롬프트 중 1개(cat_dog, in-dist objaverse)만 이겼고,
zero-shot에서 Phase 16/18 대비 10~60% 수준으로 크게 감소 (p19 zero-shot 평균 0.006 vs p16 0.028).

| prompt_id | phase16 | phase18 | phase19 | winner |
|-----------|---------|---------|---------|--------|
| cat_dog (in-dist objaverse) | 0.006 | 0.007 | 0.033 | phase19 |
| dancers (zero-shot) | 0.036 | 0.034 | 0.002 | phase16 |
| snakes (zero-shot) | 0.051 | 0.084 | 0.015 | phase18 |
| fighters (zero-shot) | 0.008 | 0.011 | 0.001 | phase18 |

**가설 원인**:
1. **6-layer inference injection 과부하**: 학습 시 VCAProcessor는 TrainVCAProcessor로 sigma_acc 누적용으로 UNet을 단순 통과. inference 시에는 FixedContextVCAProcessor가 6개 레이어에서 entity context를 강제 주입 → UNet의 cross-attention 구조가 학습에서 의도한 것보다 더 강하게 변형됨.
2. **depth_pe_init_scale=0.3이 zero-shot에서 과도한 편향**: 학습 데이터(objaverse)의 entity embedding 분포에 depth_pe가 맞춰지므로, zero-shot 프롬프트에서는 잘못된 z-bin 편향이 오히려 방해됨.
3. **VCA layer 1개 공유 × 6 주입**: 단일 VCALayer를 6개 위치에 반복 주입하므로, 한 번의 forward에서 동일 entity context가 6번 누적되어 sigma가 포화(saturation)될 가능성.

**교훈**:
- Multi-layer injection에서 학습-inference 일관성이 critical. 학습 시 TrainVCAProcessor의 동작과 inference 시 FixedContextVCAProcessor의 동작이 다르게 동작할 수 있다.
- 6-layer injection은 in-dist objaverse에서는 효과적이지만 zero-shot generalization을 크게 해침.
- 다음 시도: mid_block 1개 + up_block 1개(2-layer) 또는 학습 시 FixedContextVCAProcessor와 동일한 동작으로 통일.

---

### FM-I11: Phase 20 Fix 1+2+3 (단일 주입) — in-dist 개선 없음, zero-shot 일부 회복

**현상**: Phase 20은 per-frame loss(Fix 1) + fixed probe(Fix 2) + depth_pe_init_scale=0.3(Fix 3)을 mid_block 단일 주입에 적용.
결과: 8개 프롬프트 중 2개 win (dancers, fighters — 둘 다 zero-shot), in-dist는 오히려 하락.

| 카테고리 | Phase16 avg | Phase18 avg | Phase19 avg | Phase20 avg |
|----------|-------------|-------------|-------------|-------------|
| in_dist_objaverse | 0.0121 | 0.0123 | 0.0231 | **0.0069** |
| in_dist_toy       | 0.0529 | 0.0363 | 0.0187 | **0.0112** |
| zero_shot         | 0.0279 | 0.0352 | 0.0057 | **0.0212** |

Phase 20 패턴:
- zero-shot 2개 (dancers: 0.046 최고, fighters: 0.013 최고) — per-frame loss가 zero-shot generalization에 기여
- in-dist는 Phase 16/18보다 낮음 — depth_pe_init_scale=0.3이 in-dist objaverse에서 오히려 방해?

**가설 원인**:
1. **probe entity가 학습 내내 고정** (a cat vs a dog): probe_sep 최적화 방향이 probe 시퀀스에 과적합될 수 있음. best 체크포인트가 probe 시퀀스에만 좋은 VCA 파라미터를 저장.
2. **depth_pe_init_scale=0.3 × single-layer**: Fix 3은 multi-layer 환경에서 설계됨. single-layer에서는 entity embedding(~1.0)과 depth_pe(~0.3)의 비율이 여전히 entity가 지배하여 z-bin 분리가 불충분할 수 있음.
3. **per-frame loss × 60 epoch adaptive lambda**: lambda가 빠르게 decay(epoch 0: 0.3→0.15)하여 depth supervision이 조기에 약화됨.

**교훈**: Fix 1+2+3이 zero-shot에는 도움되지만 in-dist 퇴보. 다음 시도 방향:
  a) probe 다양화: 매 epoch 랜덤 probe 3개 평균 (고정 probe 과적합 방지)
  b) adaptive lambda 완화: min_lambda 높이기 (0.075 이하로 내려가지 않도록)
  c) depth_pe_init_scale 재검토: single-layer에서 0.5~1.0이 더 적합할 수 있음

---

### FM-I12: Phase 1~20 공통 — VCA가 text cross-attention을 완전 대체 (구조적 버그)

**현상**: `debug/learning_quality`의 reconstruction.gif에서 Baseline(고양이 등 인식 가능한 이미지)에 비해
Phase 16/18/19/20 모두 격자 무늬, 추상 노이즈, 갈색 덩어리 등 품질이 크게 저하됨.
sigma_separation을 측정하고 있었지만 이미지 자체가 망가진 상태에서 측정.

**원인**: `FixedContextVCAProcessor.__call__()` 코드:
```python
encoder_hidden_states: Optional[torch.Tensor] = None,  # ignored ← 버그
vca_out = self.vca(x, ctx.float())   # entity context만 사용
attn_out = vca_out - x               # text attn 완전 대체
return attn_out
```
mid_block의 text cross-attention(77토큰 × 768dim)을 entity context(4벡터)로 대체.
Text guidance가 사라져 AnimateDiff가 프롬프트를 잃어버림 → 이미지 품질 파괴.

**Phase 21 수정**:
```python
class AdditiveVCAProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, ...):
        # 1. 원본 text cross-attention 유지
        text_out = self.orig(attn, hidden_states, encoder_hidden_states, ...)
        # 2. VCA depth delta (alpha=0.3 강도)
        vca_delta = (VCA(hidden_states) - hidden_states) * self.alpha
        # 3. additive blend
        return text_out + vca_delta
```

**검증**: 5epoch, 20 샘플 smoke test → `depth_rank_accuracy=0.906` (91%), `IDEA=WORKS`.
전체 데이터(168샘플) epoch 0: dra=0.588 (58.8% > 50% random baseline).

**교훈**: VCA는 text cross-attention을 대체하는 것이 아니라 depth bias로 추가해야 함.
additive 구조에서만 text quality + depth awareness가 동시에 달성 가능.
