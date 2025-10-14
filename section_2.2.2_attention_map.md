# 2.2.2 Attention Map 계산

## 개요

본 연구에서 사용하는 Attention Map은 특징맵의 공간적 활성화 강도를 정규화한 2차원 표현으로, 네트워크가 입력 이미지의 어느 공간적 위치에 주목하는지를 나타낸다. 이는 Zagoruyko와 Komodakis[1]가 제안한 Attention Transfer 방법론을 기반으로 하며, 채널 차원의 정보를 집계하여 공간적 중요도를 추출한다.

## 수학적 정의

중간 특징맵 \(F \in \mathbb{R}^{N \times C \times H \times W}\)가 주어졌을 때, 공간적 attention map \(A \in \mathbb{R}^{N \times H \times W}\)은 다음과 같이 계산된다:

\[
A(F) = \frac{\mathbf{a}}{\|\mathbf{a}\|_2 + \epsilon}
\]

여기서 \(\mathbf{a} \in \mathbb{R}^{N \times H \times W}\)는 채널 차원에 대한 활성화 강도의 합으로 정의된다:

\[
\mathbf{a} = \sum_{c=1}^{C} |F_{:,c,:,:}|^p
\]

각 기호의 의미는 다음과 같다:
- \(N\): 배치 크기 (batch size)
- \(C\): 채널 개수 (number of channels)
- \(H \times W\): 공간 해상도 (spatial resolution)
- \(p\): 거듭제곱 지수 (본 연구에서는 \(p=2\) 사용)
- \(\|\cdot\|_2\): L2 norm, 공간 차원 \((H, W)\)에 대해 계산
- \(\epsilon = 10^{-8}\): 수치 안정성을 위한 작은 상수

## 계산 과정

Attention map 계산은 다음 3단계로 이루어진다:

### Step 1: 활성화 강도 계산 (Activation Magnitude)

특징맵의 각 원소에 대해 절댓값의 \(p\)제곱을 계산한다:

\[
\tilde{F} = |F|^p
\]

여기서 \(p=2\)를 사용하여 높은 활성값을 더욱 강조한다. 이는 강하게 활성화된 위치가 더 중요하다는 가정에 기반한다.

**의미**:
- 양수와 음수 활성값을 동등하게 취급 (절댓값)
- 제곱 연산으로 높은 활성값의 영향력 증폭
- 예: 활성값 0.1 → 0.01, 활성값 1.0 → 1.0, 활성값 2.0 → 4.0

### Step 2: 채널 차원 합산 (Channel Aggregation)

모든 채널에 걸쳐 활성화 강도를 합산하여 공간적 맵을 생성한다:

\[
\mathbf{a}_{n,h,w} = \sum_{c=1}^{C} \tilde{F}_{n,c,h,w}
\]

이 연산을 통해 4차원 텐서 \((N, C, H, W)\)가 3차원 텐서 \((N, H, W)\)로 축소된다.

**의미**:
- 각 공간 위치 \((h, w)\)에서 **모든 채널의 활성값을 집계**
- 채널별 중요도가 아닌, **해당 위치의 전체 활성 강도** 측정
- 공간적 위치의 "중요도"를 단일 스칼라 값으로 표현

**예시**:
```
입력: F = (64, 512, 40, 40)  # Batch=64, Channels=512, H=40, W=40
Step 1: |F|^2 = (64, 512, 40, 40)
Step 2: sum(dim=1) → a = (64, 40, 40)  # 채널 차원 제거
```

### Step 3: L2 정규화 (L2 Normalization)

공간 차원에 대해 L2 norm을 계산하고 정규화한다:

\[
\|a\|_2 = \sqrt{\sum_{h=1}^{H} \sum_{w=1}^{W} \mathbf{a}_{n,h,w}^2}
\]

\[
A_{n,h,w} = \frac{\mathbf{a}_{n,h,w}}{\|\mathbf{a}_n\|_2 + \epsilon}
\]

**의미**:
- 각 샘플의 attention map을 단위 norm으로 정규화
- 스케일 불변성(scale invariance) 확보
- 서로 다른 레이어 간 비교 가능하도록 정규화
- 배치 내 서로 다른 이미지 간 비교 가능

**정규화의 효과**:
- 이미지 밝기, 대비에 무관하게 상대적 중요도 비교
- Teacher와 Student의 활성값 크기가 달라도 패턴 비교 가능
- Gradient 안정화 (역전파 시 수치적 안정성)

## 구현 코드

위의 수학적 정의는 다음과 같이 구현된다 (trainer.py:170-184):

```python
def attention_map(self, fm, p=2):
    """공간적 attention map 계산

    Args:
        fm (torch.Tensor): 특징맵, shape (N, C, H, W)
        p (int): 거듭제곱 지수, 기본값 2

    Returns:
        torch.Tensor: Attention map, shape (N, H, W)
    """
    # Step 1: 활성화 강도 계산 |F|^p
    am = torch.pow(torch.abs(fm), p)

    # Step 2: 채널 차원 합산 (C 차원 제거)
    am = torch.sum(am, dim=1, keepdim=False)  # (N, C, H, W) → (N, H, W)

    # Step 3: L2 정규화
    norm = torch.norm(am, p=2, dim=(1, 2), keepdim=True)  # (N, 1, 1)
    am = torch.div(am, norm + 1e-8)  # Broadcasting: (N, H, W) / (N, 1, 1)

    return am  # (N, H, W)
```

**코드 상세 설명**:

1. **`torch.pow(torch.abs(fm), p)`**
   - `torch.abs()`: 음수 활성값도 동등하게 취급
   - `torch.pow(·, 2)`: Element-wise 제곱 연산
   - 출력: (N, C, H, W), 입력과 동일한 크기

2. **`torch.sum(am, dim=1, keepdim=False)`**
   - `dim=1`: 채널 차원(C)에 대해 합산
   - `keepdim=False`: 합산된 차원 제거
   - 출력: (N, H, W), 공간적 맵

3. **`torch.norm(am, p=2, dim=(1, 2), keepdim=True)`**
   - `p=2`: L2 norm 계산
   - `dim=(1, 2)`: H와 W 차원에 대해 norm 계산
   - `keepdim=True`: Broadcasting을 위해 차원 유지
   - 출력: (N, 1, 1), 각 샘플의 norm 값

4. **`torch.div(am, norm + 1e-8)`**
   - Broadcasting: (N, H, W) / (N, 1, 1) → (N, H, W)
   - `1e-8`: Division by zero 방지
   - 출력: (N, H, W), 정규화된 attention map

## Attention Map의 특성

### 1. 공간적 표현 (Spatial Representation)

Attention map \(A \in \mathbb{R}^{N \times H \times W}\)는 **공간적 위치별 중요도**를 나타낸다:

- \(A_{n,h,w}\): 배치 내 \(n\)번째 이미지의 \((h, w)\) 위치의 중요도
- 높은 값: 해당 위치가 네트워크에 의해 강하게 활성화됨
- 낮은 값: 해당 위치가 네트워크에 의해 약하게 활성화됨

### 2. 스케일 불변성 (Scale Invariance)

L2 정규화로 인해 다음과 같은 불변성을 가진다:

\[
A(c \cdot F) = A(F), \quad \forall c > 0
\]

즉, 특징맵의 전체 크기가 상수배로 변해도 attention pattern은 동일하다.

**의미**:
- Teacher와 Student의 활성값 크기가 달라도 상대적 패턴 비교 가능
- 서로 다른 레이어 간 비교 시 스케일 차이 무시
- 학습 안정성 향상

### 3. 채널 독립성 제거 (Channel-Agnostic)

채널 차원 합산으로 인해 attention map은 **어떤 채널이 활성화되었는지와 무관**하게, **해당 위치가 얼마나 활성화되었는지**만 나타낸다.

**예시**:
- 특정 위치에서 채널 100번이 강하게 활성화: 높은 attention 값
- 특정 위치에서 여러 채널이 약하게 활성화: 낮은 attention 값
- 특정 위치에서 많은 채널이 중간 정도로 활성화: 중간 attention 값

### 4. 다중 스케일 적용 (Multi-Scale Application)

동일한 계산 방식을 서로 다른 해상도의 특징맵에 적용:

| Layer | Feature Map | Attention Map | Spatial Points |
|-------|-------------|---------------|----------------|
| 16 | (N, 256, 80, 80) | (N, 80, 80) | 6,400 |
| 6, 13, 19 | (N, 512, 40, 40) | (N, 40, 40) | 1,600 |
| 8, 22 | (N, 512, 20, 20) | (N, 20, 20) | 400 |

**의미**:
- 고해상도 레이어: 세밀한 공간 정보 (작은 객체)
- 저해상도 레이어: 전역적 공간 정보 (큰 객체)
- 모든 스케일에서 동일한 계산 방식 적용

## Teacher-Student Attention 비교

계산된 attention map을 사용하여 Teacher와 Student의 공간적 주의 패턴을 비교한다:

\[
\text{Distance} = \|A^T - A^S\|_2
\]

여기서:
- \(A^T \in \mathbb{R}^{N \times H \times W}\): Teacher의 attention map
- \(A^S \in \mathbb{R}^{N \times H \times W}\): Student의 attention map
- 두 맵의 L2 거리가 작을수록 유사한 공간적 패턴

**시각적 해석**:
- Teacher와 Student의 attention map을 heatmap으로 시각화
- 밝은 영역: 높은 attention (중요한 위치)
- 어두운 영역: 낮은 attention (덜 중요한 위치)
- 두 heatmap이 유사할수록 Student가 Teacher의 패턴을 잘 모방

**실험 결과** (attention_visualizations/):
- Teacher와 Student의 attention map이 시각적으로 유사
- 객체 위치(motorbike, sheep, person)에서 높은 activation
- Layer가 깊어질수록(6→8→13→16→19→22) 점점 더 집중된 패턴

## 수치적 예시

**예제 1: Layer 16 (80×80, 256 channels)**

```
Teacher Feature Map: F_T = (64, 256, 80, 80)

Step 1: |F_T|^2 = (64, 256, 80, 80)
Step 2: sum(dim=1) → a_T = (64, 80, 80)
        예: a_T[0, 40, 40] = 2500.0 (중심 위치에서 높은 값)
Step 3: norm(a_T[0]) = 8000.0 (전체 L2 norm)
        A_T[0, 40, 40] = 2500.0 / 8000.0 = 0.3125

Student Feature Map: F_S = (64, 128, 80, 80)

Step 1: |F_S|^2 = (64, 128, 80, 80)
Step 2: sum(dim=1) → a_S = (64, 80, 80)
        예: a_S[0, 40, 40] = 1800.0
Step 3: norm(a_S[0]) = 6500.0
        A_S[0, 40, 40] = 1800.0 / 6500.0 = 0.2769

Distance: |A_T[0, 40, 40] - A_S[0, 40, 40]| = |0.3125 - 0.2769| = 0.0356
```

**해석**:
- Teacher와 Student 모두 중심 위치(40, 40)를 중요하게 여김
- 정규화 후 상대적 중요도가 유사 (0.3125 vs 0.2769)
- 작은 차이(0.0356)는 학습을 통해 더 줄일 수 있음

**예제 2: 전체 맵 비교 (Layer 16)**

```
Teacher Attention Map (80×80):
  - Max value: 0.35 (객체 중심)
  - Min value: 0.001 (배경)
  - Mean value: 0.0156 (1/(80*80) ≈ 0.0156)

Student Attention Map (80×80):
  - Max value: 0.28 (객체 중심)
  - Min value: 0.002 (배경)
  - Mean value: 0.0156

L2 Distance: ||A_T - A_S||_2 = 0.42
```

**해석**:
- 두 맵 모두 객체 중심에서 peak
- Student의 peak가 약간 낮음 (0.28 vs 0.35)
- 학습 목표: L2 distance 0.42를 줄이기

## 계산 복잡도

각 레이어에서의 계산 복잡도:

**시간 복잡도**:
- Step 1 (|F|^p): \(O(N \cdot C \cdot H \cdot W)\)
- Step 2 (sum): \(O(N \cdot C \cdot H \cdot W)\)
- Step 3 (norm): \(O(N \cdot H \cdot W)\)
- **Total**: \(O(N \cdot C \cdot H \cdot W)\)

**공간 복잡도**:
- 중간 텐서: \(O(N \cdot C \cdot H \cdot W)\) (Step 1)
- Attention map: \(O(N \cdot H \cdot W)\) (Step 2 이후)

**전체 6개 레이어 계산량** (N=64):

| Layer | C | H×W | FLOPs (Step 1+2) | Memory |
|-------|---|-----|------------------|--------|
| 16 | 256 | 80×80 | 1.31M | 1.31 MB |
| 6, 13, 19 | 512 | 40×40 | 2.62M | 2.62 MB |
| 8, 22 | 512 | 20×20 | 0.66M | 0.66 MB |
| **Total** | - | - | **10.49M** | **10.49 MB** |

**참고**:
- 실제 학습 시 forward pass 계산량의 ~1% 미만
- 메모리 오버헤드 minimal
- GPU 병렬 처리로 빠른 계산

## 요약

Attention map 계산은 다음과 같이 요약된다:

1. **입력**: 4차원 특징맵 \((N, C, H, W)\)
2. **출력**: 3차원 attention map \((N, H, W)\)
3. **핵심 연산**:
   - 활성화 강도 계산 (\(|F|^p\))
   - 채널 차원 합산 (공간적 맵 생성)
   - L2 정규화 (스케일 불변성)
4. **의미**: 네트워크가 **어느 공간 위치**를 중요하게 여기는지
5. **용도**: Teacher-Student 간 **공간적 주의 패턴 전이**

---

## 참고문헌

[1] Zagoruyko, S., & Komodakis, N. (2016). Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. ICLR 2017. arXiv:1612.03928.
