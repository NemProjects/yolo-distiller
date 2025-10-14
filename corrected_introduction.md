# 수정된 서론 (Attention Transfer KD 기반)

## 1. 서론

임베디드 AI 시스템은 제한된 연산 자원과 전력 제약으로 인해, 고성능 딥러닝 모델을 실시간으로 구동하기 어렵다. 예를 들어, 자율주행차나 UAV, IoT 디바이스에서는 객체 탐지(Object Detection)와 같은 비전 작업을 온보드에서 수행해야 하지만, 대형 모델은 높은 지연(latency)과 전력 소모를 초래한다.

이러한 환경에서 널리 활용되는 YOLO(You Only Look Once) 계열 모델은 모델 크기에 따라 정확도–속도 간의 trade-off를 제공한다. 작은 모델(YOLOv11s)은 빠르지만 정확도가 낮고, 큰 모델(YOLOv11m)은 정확도가 높지만 연산량과 자원 소모가 크다. 즉, 임베디드 환경에서는 실시간성을 확보하기 위해 경량 모델을 선택해야 하지만, 이에 따른 성능 저하는 불가피하다.

이를 해결하기 위한 대표적 접근법이 지식 증류(Knowledge Distillation, KD)이다. KD는 대형 모델(Teacher)의 예측 결과나 중간 표현을 활용하여 경량 모델(Student)을 학습시킴으로써, 모델 경량화와 성능 향상을 동시에 달성한다. 분류(Classification) 분야에서는 이러한 기법이 활발히 연구되어 왔으며, Soft Target 혹은 Feature-level 지식을 모방하는 방식으로 Student의 일반화 성능을 높이는 데 성공하였다. 그러나 객체 탐지(Object Detection) 모델은 구조적 복잡성과 다중 출력(box, class, DFL 등)으로 인해, 분류 모델에서의 KD 방법을 그대로 적용하기 어렵다.

**본 연구에서는 이러한 한계를 극복하기 위해, YOLOv11 기반 객체 탐지 모델의 경량화와 성능 향상을 목표로 공간적 어텐션 전이 기반 지식 증류(Attention Transfer-based Knowledge Distillation) 방법을 적용한다. 본 방법은 Teacher(YOLOv11m)가 학습한 공간적 활성화 패턴(spatial activation pattern)을 Student(YOLOv11s)가 모방하도록 유도하며, 다중 스케일(multi-scale) 특징 계층에서 attention map을 정합함으로써 중요한 공간적 정보를 효율적으로 전이한다. 이를 통해 Student 모델은 Teacher의 공간적 주의 집중 메커니즘을 계승하면서도 경량 구조의 효율성을 유지할 수 있다.**

결과적으로, 본 연구는 지식 증류를 통한 경량 객체 탐지 모델의 성능 보완 전략을 제시하며, 이는 자원 제약이 큰 임베디드 환경에서 실시간 객체 탐지를 구현하는 데 실질적인 기여를 제공한다.

---

## 주요 수정 사항

### ❌ 원본 (틀린 표현)
- "채널 단위 어텐션 기반 지식 증류(Channel-wise Attention Distillation)"
- "채널별 표현 강도를 Student가 모방"
- "공간적 정합 과정 없이도"

### ✅ 수정 (정확한 표현)
- "공간적 어텐션 전이 기반 지식 증류(Attention Transfer-based Knowledge Distillation)"
- "공간적 활성화 패턴(spatial activation pattern)을 Student가 모방"
- "다중 스케일 특징 계층에서 attention map을 정합함으로써"

---

## 기술적 근거

### 1. Attention Transfer 방법론 (arxiv.org/abs/1612.03928)
```
AttentionLoss: 공간적 attention 전이
- Input: Feature map (N, C, H, W)
- Process: 채널 차원 합산 → (N, H, W) 공간적 맵
- Output: Spatial attention pattern
- Loss: L2 distance between teacher and student attention maps
```

### 2. 실제 구현 코드 (trainer.py:170-184)
```python
def attention_map(self, fm, p=2):
    """공간적 attention map 계산"""
    am = torch.pow(torch.abs(fm), p)
    am = torch.sum(am, dim=1, keepdim=False)  # 채널 합산 → 공간적 맵
    norm = torch.norm(am, p=2, dim=(1, 2), keepdim=True)
    am = torch.div(am, norm + 1e-8)
    return am  # (N, H, W) - 공간적 attention!
```

### 3. Multi-scale Feature Distillation
- Layer 16: 80×80 (High-res) - 작은 객체
- Layers 6, 13, 19: 40×40 (Mid-res) - 중간 객체
- Layers 8, 22: 20×20 (Low-res) - 큰 객체
- **총 6개 레이어에서 공간적 attention 전이**

### 4. CWD vs Attention Transfer 비교

| 특성 | CWD (Channel-Wise) | Attention Transfer (본 연구) |
|------|-------------------|------------------------------|
| **Target** | 채널별 표현 강도 | 공간적 활성화 패턴 |
| **Attention Map** | (N, C, 1, 1) | **(N, H, W)** ✅ |
| **계산** | Spatial pooling | **Channel summation** ✅ |
| **정합 차원** | Channel 차원 | **Spatial 차원** ✅ |
| **Loss** | Channel-wise L2 | **Spatial L2** ✅ |

---

## 영문 표현

### 제안하는 방법
- **Attention Transfer-based Knowledge Distillation**
- **Spatial Attention Transfer KD**
- **Multi-scale Attention Transfer for Object Detection**

### 핵심 개념
- **Spatial activation pattern**: 공간적 활성화 패턴
- **Attention map alignment**: Attention 맵 정합
- **Multi-scale feature distillation**: 다중 스케일 특징 증류
- **Spatial attention mechanism**: 공간적 주의 집중 메커니즘

---

## 참고문헌 추가 필요
```
[1] Zagoruyko, S., & Komodakis, N. (2016).
    Paying more attention to attention: Improving the performance of
    convolutional neural networks via attention transfer.
    arXiv preprint arXiv:1612.03928.
```
