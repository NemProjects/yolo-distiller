# 수정된 연구 방법 (Attention Transfer KD 기반)

## 2. 연구 방법

### 2.1 연구 개요
본 연구에서는 YOLOv11 계열 모델을 기반으로 한 Teacher–Student 지식 증류(knowledge distillation) 프레임워크를 제안한다. 대형 모델(YOLOv11m, 22M parameters)이 학습한 풍부한 표현 정보를 경량 모델(YOLOv11s, 9.4M parameters)로 효율적으로 전이하여, 경량 구조에서도 높은 탐지 성능을 달성하는 것을 목표로 한다.

전체 프레임워크는 **공간적 어텐션 전이 기반 특징 증류(Attention Transfer-based Feature Distillation)**를 중심으로 구성된다. Teacher는 학습 과정 동안 파라미터가 고정(frozen, `requires_grad=False`)되고 `torch.no_grad()` 컨텍스트 하에서 순전파만 수행하여 중간 특징맵(intermediate feature maps)을 제공한다. Student는 동일한 입력 이미지에 대해 순전파(forward) 및 역전파(backward)를 모두 수행하면서, Teacher의 중간 특징맵을 비교 지표로 삼아 학습한다.

이 과정에서 Student는 **Teacher가 입력 이미지 내에서 강조하는 공간적 주의 패턴(spatial attention patterns)**을 학습하도록 유도된다. 특히, **다중 스케일(multi-scale) 특징 계층**—고해상도(80×80), 중해상도(40×40), 저해상도(20×20)—에서 추출된 6개 레이어의 attention map을 동시에 정합함으로써, 다양한 크기의 객체에 대한 공간적 주의 집중 메커니즘을 효과적으로 전이한다. 이로써 Student 모델은 Teacher의 공간적 표현력을 계승하면서도 경량 구조의 효율성을 유지할 수 있다.

---

## 2.2 Teacher-Student 아키텍처

### 2.2.1 모델 구성

| 항목 | Teacher (YOLOv11m) | Student (YOLOv11s) |
|------|--------------------|--------------------|
| **Parameters** | 22.0M | 9.4M (42.7% of teacher) |
| **FLOPs** | ~150 GFLOPs | ~65 GFLOPs |
| **Layer 구조** | C2f + SPPF + Detect Head | C2f + SPPF + Detect Head |
| **역할** | Feature reference 제공 | Task + Distillation loss 학습 |
| **학습 상태** | Frozen (`requires_grad=False`) | Trainable |

### 2.2.2 Teacher 모델 상태
Teacher는 다음과 같이 설정되어 학습에 관여하지 않는다:

```python
# trainer.py:402-407
for param in self.teacher.parameters():
    param.requires_grad = False  # 파라미터 고정

# Forward pass 시 (trainer.py:622)
with torch.no_grad():
    teacher_outputs = self.teacher(batch["img"])  # Gradient 계산 안함
```

- **`requires_grad=False`**: Teacher의 모든 파라미터를 학습 불가능 상태로 설정
- **`torch.no_grad()`**: Forward pass 시 computational graph 생성 안함 → 메모리 효율
- **DDP 사용 시**: `self.teacher.eval()` 상태 유지 (BatchNorm, Dropout 등 inference mode)

---

## 2.3 Attention Transfer 기반 지식 증류

### 2.3.1 Attention Map 계산

Teacher와 Student의 중간 특징맵 \(F \in \mathbb{R}^{N \times C \times H \times W}\)으로부터 공간적 attention map \(A \in \mathbb{R}^{N \times H \times W}\)을 다음과 같이 계산한다:

\[
A(F) = \frac{\sum_{c=1}^{C} |F_c|^p}{\|\sum_{c=1}^{C} |F_c|^p\|_2 + \epsilon}
\]

여기서 \(p=2\)이며, 채널 차원에 대한 합산(channel summation)을 통해 공간적 활성화 강도를 집계한 뒤 L2 정규화를 수행한다.

**구현 코드 (trainer.py:170-184):**
```python
def attention_map(self, fm, p=2):
    """공간적 attention map 계산"""
    am = torch.pow(torch.abs(fm), p)           # |F|^p
    am = torch.sum(am, dim=1, keepdim=False)   # 채널 합산 → (N, H, W)
    norm = torch.norm(am, p=2, dim=(1, 2), keepdim=True)
    am = torch.div(am, norm + 1e-8)            # L2 정규화
    return am  # Shape: (N, H, W)
```

### 2.3.2 Multi-scale Feature Distillation

본 연구에서는 YOLOv11 백본의 **6개 중간 레이어**(Layers 6, 8, 13, 16, 19, 22)에서 특징맵을 추출하여 attention 증류를 수행한다. 이들은 다음과 같이 3가지 해상도 스케일로 분류된다:

| Layer | Resolution | Feature Points | Receptive Field | 대상 객체 크기 |
|-------|------------|----------------|-----------------|---------------|
| **16** | 80×80 (P3) | 6,400 | 1/8 | 작은 객체 |
| **6, 13, 19** | 40×40 (P4) | 1,600 | 1/16 | 중간 객체 |
| **8, 22** | 20×20 (P5) | 400 | 1/32 | 큰 객체 |

각 스케일에서 Teacher와 Student의 attention map 간 L2 거리를 계산하고, 모든 레이어의 손실을 합산하여 최종 증류 손실을 구한다:

\[
\mathcal{L}_{\text{att}} = \sum_{l \in \mathcal{L}} \frac{\|A^T_l - A^S_l\|_2}{N}
\]

여기서 \(\mathcal{L} = \{6, 8, 13, 16, 19, 22\}\)는 증류 대상 레이어 집합이며, \(A^T_l\)과 \(A^S_l\)은 각각 Teacher와 Student의 레이어 \(l\)에서의 attention map이다.

**구현 코드 (trainer.py:186-208):**
```python
def forward(self, y_s, y_t):
    """Multi-scale attention transfer"""
    losses = []

    for s, t in zip(y_s, y_t):  # 6개 레이어 순회
        s_attention = self.attention_map(s, self.p)  # Student attention
        t_attention = self.attention_map(t, self.p)  # Teacher attention

        # L2 loss between attention maps
        loss = torch.norm(s_attention - t_attention, p=2) / s_attention.shape[0]
        losses.append(loss)

    return sum(losses)  # 모든 스케일 loss 합산
```

### 2.3.3 Channel Alignment

Teacher(YOLOv11m)와 Student(YOLOv11s)는 동일한 아키텍처 구조를 공유하지만 채널 수가 다르다. 따라서 attention 계산 전에 Student의 특징맵을 Teacher의 채널 차원에 맞추는 정합(alignment) 과정이 필요하다.

**Channel Alignment 상세:**

| Layer | Teacher Channels | Student Channels | Alignment |
|-------|------------------|------------------|-----------|
| 6 | 512 | 256 | Conv1x1 + BN |
| 8 | 512 | 512 | 불필요 |
| 13 | 512 | 256 | Conv1x1 + BN |
| 16 | 256 | 128 | Conv1x1 + BN |
| 19 | 512 | 256 | Conv1x1 + BN |
| 22 | 512 | 512 | 불필요 |

**구현 코드 (trainer.py:290-308):**
```python
# Channel alignment modules
self.align_module = nn.ModuleList([
    nn.Sequential(
        nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(t_channel, affine=False)
    ).to(device)
    for s_channel, t_channel in zip(channels_s, channels_t)
])

# Forward 시 alignment 적용 (trainer.py:355-360)
if self.distiller in ["att", "spa", "hybrid"]:
    s = self.align_module[idx](s)  # Student 채널 확장
    stu_feats.append(s)
    tea_feats.append(t.detach())
```

---

## 2.4 최종 Loss 함수

Student 모델의 전체 학습 손실은 다음과 같이 구성된다:

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda(t) \cdot w_{\text{att}} \cdot \mathcal{L}_{\text{att}}
\]

- **\(\mathcal{L}_{\text{task}}\)**: YOLO detection loss (box, class, DFL)
- **\(\mathcal{L}_{\text{att}}\)**: Attention transfer loss
- **\(w_{\text{att}} = 0.75\)**: Attention loss 기본 가중치 (2.5 × 0.3)
- **\(\lambda(t)\)**: Epoch에 따른 적응형 가중치 스케줄링

### 2.4.1 Adaptive Lambda Scheduling (150 epochs)

증류 손실의 영향력을 학습 단계에 따라 동적으로 조절한다:

\[
\lambda(t) =
\begin{cases}
2.0 \cdot (1 - \frac{t}{40}) & \text{if } t < 40 \\
0.3 \cdot (1 - \frac{t-40}{60}) & \text{if } 40 \leq t < 100 \\
0.05 & \text{if } t \geq 100
\end{cases}
\]

**구현 코드 (trainer.py:763-768):**
```python
if epoch < 40:
    distill_weight = 2.0 * (1 - epoch / 40)      # 2.0 → 0
elif epoch < 100:
    distill_weight = 0.3 * (1 - (epoch - 40) / 60)  # 0.3 → 0
else:
    distill_weight = 0.05  # 고정
```

**스케줄링 전략:**
- **Phase 1 (0-40 epoch)**: 강한 증류 (λ = 2.0 → 0.0)
  - 초기 학습 가속화
  - Teacher의 공간적 패턴 빠르게 모방
- **Phase 2 (40-100 epoch)**: 중간 증류 (λ = 0.3 → 0.0)
  - Task loss와 균형 유지
  - Fine-tuning 단계
- **Phase 3 (100-150 epoch)**: 최소 증류 (λ = 0.05)
  - Student의 독립적 학습 강화
  - Overfitting 방지

**최종 유효 가중치 (Effective Weight = 0.75 × λ(t)):**

| Epoch | λ(t) | Effective Weight |
|-------|------|------------------|
| 0 | 2.0 | 1.50 |
| 20 | 1.0 | 0.75 |
| 40 | 0.0 | 0.00 |
| 70 | 0.15 | 0.1125 |
| 100 | 0.0 | 0.00 |
| **105 (best)** | 0.05 | **0.0375** |
| 150 | 0.05 | 0.0375 |

---

## 2.5 학습 설정

### 2.5.1 데이터셋
- **Dataset**: PASCAL VOC 2012
- **Training images**: 17,125
- **Validation images**: 4,952
- **Classes**: 20 object categories
- **Image size**: 640×640 (resized)

### 2.5.2 하이퍼파라미터
- **Epochs**: 150
- **Batch size**: 64
- **Workers**: 12
- **Optimizer**: SGD with momentum
- **Learning rate**: Cosine annealing
- **Weight decay**: 0.0005
- **Patience**: 100 (early stopping)

### 2.5.3 학습 환경
- **GPU**: NVIDIA A100 40GB
- **Framework**: PyTorch + Ultralytics YOLO
- **Training time**: ~5.8 hours

---

## 주요 수정 사항 요약

### ❌ 원본 (틀린 표현)
1. "Attention 기반 특징 증류" (모호)
2. "채널별 주의 특성(channel-wise attention characteristics)"
3. "고정(frozen)" (불완전한 설명)
4. Multi-scale 정보 누락
5. 문장 미완성 ("이로써" 로 끝남)

### ✅ 수정 (정확한 표현)
1. "**공간적 어텐션 전이 기반 특징 증류(Attention Transfer-based Feature Distillation)**"
2. "**공간적 주의 패턴(spatial attention patterns)**"
3. "**파라미터 고정(frozen, `requires_grad=False`) + `torch.no_grad()` 컨텍스트**"
4. **다중 스케일 특징 계층 (6 layers, 3 resolutions)** 명시
5. 완전한 문장 완성 + 기술적 세부사항 추가

---

## 참고문헌
```
[1] Zagoruyko, S., & Komodakis, N. (2016).
    Paying more attention to attention: Improving the performance of
    convolutional neural networks via attention transfer.
    arXiv:1612.03928.

[2] Wang, J., Chen, K., Xu, R., Liu, Z., Loy, C. C., & Lin, D. (2019).
    CARAFE: Content-Aware ReAssembly of FEatures.
    ICCV 2019.
```
