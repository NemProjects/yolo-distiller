# 수정된 2.2 Attention-based Feature Distillation Framework

## 2.2 Attention Transfer 기반 특징 증류 프레임워크

### 2.2.1 개요

Attention Transfer 기반 증류는 Teacher와 Student의 중간 특징맵을 비교하여, Student가 Teacher의 **공간적 활성화 패턴(spatial activation pattern)**을 모방하도록 유도하는 방식이다. 이때 Attention Map은 특징맵의 **공간적 활성 강도를 정규화한 형태의 표현**으로, Teacher가 입력 이미지에서 **"어느 공간적 위치에 집중했는가"**를 명시적으로 반영한다.

본 연구에서는 YOLOv11의 다단계 특징 구조를 활용하여 Backbone 및 Neck의 주요 계층(Layers 6, 8, 13, 16, 19, 22)을 증류 대상으로 선정하였다. 이 6개 레이어는 **3가지 해상도 스케일**(80×80, 40×40, 20×20)에 걸쳐 분포하며, 각기 다른 receptive field를 통해 **작은 객체부터 큰 객체까지 다양한 크기의 객체 탐지에 특화된 특징**을 포착한다. 이러한 다중 스케일 구조는 Teacher의 공간적 표현력을 계층적으로 전이하는 데 적합하다.

---

## 2.2.2 Attention Map 계산

Teacher와 Student의 중간 특징맵 \(F \in \mathbb{R}^{N \times C \times H \times W}\)로부터 공간적 attention map \(A \in \mathbb{R}^{N \times H \times W}\)을 다음과 같이 계산한다:

\[
A(F) = \frac{\sum_{c=1}^{C} |F_c|^p}{\left\|\sum_{c=1}^{C} |F_c|^p\right\|_2 + \epsilon}
\]

여기서:
- \(p=2\): 활성화 강도를 제곱하여 높은 값을 더 강조
- \(\sum_{c=1}^{C}\): **채널 차원에 대한 합산(channel summation)**으로 공간적 맵 생성
- \(\|\cdot\|_2\): L2 정규화로 스케일 불변성 확보
- \(\epsilon = 10^{-8}\): 수치 안정성을 위한 작은 상수

이 과정을 통해 4차원 특징맵 \((N, C, H, W)\)이 **공간적 attention map** \((N, H, W)\)로 변환되며, 이는 각 공간 위치에서의 **전체 채널 활성화 강도**를 나타낸다. 즉, Teacher와 Student가 이미지의 **어느 위치(spatial location)**를 중요하게 여기는지를 비교 가능한 형태로 추출한다.

**구현 코드 (trainer.py:170-184):**
```python
def attention_map(self, fm, p=2):
    """공간적 attention map 계산

    Args:
        fm: Feature map (N, C, H, W)
        p: Power for attention calculation (default: 2)

    Returns:
        Attention map (N, H, W)
    """
    am = torch.pow(torch.abs(fm), p)           # |F|^p, element-wise
    am = torch.sum(am, dim=1, keepdim=False)   # 채널 합산 → (N, H, W)
    norm = torch.norm(am, p=2, dim=(1, 2), keepdim=True)  # Spatial L2 norm
    am = torch.div(am, norm + 1e-8)            # Normalize
    return am
```

---

## 2.2.3 다중 스케일 레이어 선정

본 연구에서 선정한 6개 증류 레이어는 YOLOv11의 Feature Pyramid 구조에서 **3가지 해상도 스케일**을 대표한다:

### 해상도별 레이어 분포

| Resolution | Layers | Feature Map Size | Receptive Field | 대상 객체 크기 | Feature Points |
|------------|--------|------------------|-----------------|---------------|----------------|
| **High (P3)** | 16 | 80×80 | 1/8 of input | **작은 객체** (병, 의자 등) | 6,400 |
| **Medium (P4)** | 6, 13, 19 | 40×40 | 1/16 of input | **중간 객체** (개, 고양이, 사람 등) | 1,600 |
| **Low (P5)** | 8, 22 | 20×20 | 1/32 of input | **큰 객체** (소파, 버스, 차량 등) | 400 |

### 레이어별 상세 특성

**1. Layer 16 (High Resolution - P3)**
- Feature Map: 80×80 (6,400 spatial points)
- Channels: Teacher 256, Student 128
- 역할: **세밀한 공간 해상도**로 작은 객체의 정확한 위치 포착
- Receptive Field: 작음 (local features)

**2. Layers 6, 13, 19 (Medium Resolution - P4)**
- Feature Map: 40×40 (1,600 spatial points)
- Channels: Teacher 512, Student 256
- 역할: **중간 수준 특징**으로 대부분의 일반적인 객체 탐지
- Receptive Field: 중간 (intermediate features)

**3. Layers 8, 22 (Low Resolution - P5)**
- Feature Map: 20×20 (400 spatial points)
- Channels: Teacher 512, Student 512
- 역할: **높은 의미론적 수준**으로 큰 객체 및 전역 context 파악
- Receptive Field: 큼 (high-level semantic features)

### 다중 스케일 선정의 장점

1. **다양한 객체 크기 대응**
   - VOC 데이터셋은 작은 객체(병)부터 큰 객체(버스)까지 다양
   - 단일 스케일로는 모든 크기의 객체를 효과적으로 탐지 불가
   - 3가지 스케일을 동시에 증류하여 **스케일 불변성(scale invariance)** 확보

2. **계층적 특징 학습**
   - Shallow layers (6, 8): 텍스처, 엣지 등 저수준 특징
   - Middle layers (13, 16): 형태, 부분 객체 등 중간 수준 특징
   - Deep layers (19, 22): 의미론적 패턴, 객체 전체 등 고수준 특징

3. **Feature Pyramid 구조 활용**
   - YOLOv11의 Neck 구조는 FPN(Feature Pyramid Network) 기반
   - 각 피라미드 레벨에서 attention을 전이하여 구조적 일관성 유지

4. **균형잡힌 Loss 기여도**
   - 각 스케일의 loss를 동등하게 합산 (equal weighting)
   - 특정 스케일에 편향되지 않고 전체적으로 균형잡힌 학습

---

## 2.2.4 Attention Transfer Loss

Teacher와 Student의 attention map 간 차이를 최소화하기 위해 L2 distance를 손실 함수로 사용한다. 각 레이어 \(l\)에서의 손실은 다음과 같이 계산된다:

\[
\mathcal{L}_{\text{att}}^{(l)} = \frac{1}{N} \left\|A^T_l - A^S_l\right\|_2
\]

여기서 \(A^T_l\)과 \(A^S_l\)은 각각 Teacher와 Student의 레이어 \(l\)에서의 attention map이며, \(N\)은 배치 크기이다.

최종 Attention Transfer Loss는 선정된 모든 레이어의 손실을 합산한다:

\[
\mathcal{L}_{\text{att}} = \sum_{l \in \{6, 8, 13, 16, 19, 22\}} \mathcal{L}_{\text{att}}^{(l)}
\]

**구현 코드 (trainer.py:186-208):**
```python
def forward(self, y_s, y_t):
    """Multi-scale attention transfer

    Args:
        y_s: Student feature maps (list of 6 tensors)
        y_t: Teacher feature maps (list of 6 tensors)

    Returns:
        Total attention transfer loss
    """
    assert len(y_s) == len(y_t), "Layer count mismatch"
    losses = []

    for s, t in zip(y_s, y_t):  # 6개 레이어 순회
        # 각 레이어에서 attention map 계산
        s_attention = self.attention_map(s, self.p)  # Student (N, H, W)
        t_attention = self.attention_map(t, self.p)  # Teacher (N, H, W)

        # L2 distance 계산 및 배치 정규화
        loss = torch.norm(s_attention - t_attention, p=2) / s_attention.shape[0]
        losses.append(loss)

    return sum(losses)  # 모든 스케일의 loss 합산
```

### Loss 계산 예시

입력 배치 크기 \(N=64\), 3가지 해상도에서의 loss 기여도:

| Layer | Resolution | Spatial Points | 상대적 기여도 |
|-------|------------|----------------|--------------|
| 16 | 80×80 | 6,400 | ~73% |
| 6, 13, 19 | 40×40 | 1,600 × 3 | ~22% |
| 8, 22 | 20×20 | 400 × 2 | ~5% |

**Note**: 실제로는 동등한 가중치(equal weighting)로 합산하지만, 고해상도 레이어(Layer 16)가 더 많은 spatial points를 가지므로 gradient 크기가 더 클 수 있다. 이는 L2 norm의 특성상 자연스러운 현상이며, 작은 객체 탐지에 더 집중하도록 유도하는 효과가 있다.

---

## 주요 수정 사항

### ❌ 원본 (부정확/모호)
1. "특징맵의 **채널별·공간적** 활성 강도" ← 틀림
2. "Teacher가 이미지에서 '어디에 집중했는가'" ← 모호
3. "저수준–고수준 특징 정보를 모두 포함" ← 너무 일반적
4. "멀티스케일 수준에서" ← 구체적 설명 없음

### ✅ 수정 (정확/구체적)
1. "특징맵의 **공간적 활성 강도**" + 수식으로 명확히 정의
2. "Teacher가 입력 이미지에서 **어느 공간적 위치에 집중했는가**"
3. **3가지 해상도 스케일 (80×80, 40×40, 20×20)** 명시
4. 각 스케일의 역할, receptive field, 대상 객체 크기 상세 설명
5. 다중 스케일 선정의 4가지 장점 추가
6. Loss 계산 수식 및 구현 코드 추가

---

## 영문 표현 제안

### 핵심 용어
- **Attention Transfer-based Feature Distillation**: 어텐션 전이 기반 특징 증류
- **Spatial Activation Pattern**: 공간적 활성화 패턴
- **Multi-scale Feature Pyramid**: 다중 스케일 특징 피라미드
- **Spatial Attention Map**: 공간적 어텐션 맵
- **Channel Summation**: 채널 합산
- **L2 Normalization**: L2 정규화
- **Scale Invariance**: 스케일 불변성

### 참고문헌
```
[1] Zagoruyko, S., & Komodakis, N. (2016).
    Paying more attention to attention: Improving the performance of
    convolutional neural networks via attention transfer.
    ICLR 2017. arXiv:1612.03928.

[2] Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017).
    Feature pyramid networks for object detection.
    CVPR 2017.
```
