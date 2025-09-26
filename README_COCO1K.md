# COCO-1K Knowledge Distillation Validation

빠르고 효과적인 Knowledge Distillation 검증을 위한 COCO-1K 데이터셋과 실험 도구입니다.

## 🎯 목적

- **빠른 KD 검증**: 전체 COCO 대신 균형잡힌 1500개 이미지로 KD 효과 확인
- **시간 효율성**: 4-6시간 내 완전한 KD vs Baseline 비교
- **실제 성능**: 모든 COCO 클래스 포함으로 의미있는 결과

## 📦 구성 요소

### 1. `create_coco1k.py`
COCO 2017에서 균형잡힌 1500개 이미지를 샘플링하여 COCO-1K 데이터셋 생성

**특징:**
- 80개 클래스 각각에서 균등 샘플링
- 훈련용 ~1500개, 검증용 ~400개 이미지
- 원본 COCO 어노테이션 형식 유지
- Ultralytics 호환 YAML 설정 파일 생성

### 2. `test_coco1k_kd.py`
Knowledge Distillation vs Baseline 성능 비교 실험

**실험 내용:**
- Baseline: YOLOv11n 단독 훈련
- KD: YOLOv11n (Student) + YOLOv11l (Teacher)
- 성능 지표: mAP50, mAP50-95, 훈련 시간
- 결과 시각화 및 저장

### 3. `run_coco1k_kd.sh`
전체 파이프라인 자동 실행 스크립트

## 🚀 사용 방법

### 전체 파이프라인 실행 (추천)
```bash
./run_coco1k_kd.sh
```

### 단계별 실행

#### 1단계: COCO-1K 데이터셋 생성
```bash
python create_coco1k.py \
    --coco-root /path/to/coco2017 \
    --target-size 1500 \
    --output-dir coco1k
```

#### 2단계: 간단한 KD 테스트 (추천)
```bash
python simple_kd_example.py
```

#### 3단계: 완전한 KD 검증 실험
```bash
python test_coco1k_kd.py \
    --data coco1k/coco1k.yaml \
    --epochs 100 \
    --batch 16 \
    --teacher yolo11l.pt \
    --loss cwd
```

### 실제 예제 코드 형식
```python
from ultralytics import YOLO

teacher_model = YOLO("yolo11l.pt")
student_model = YOLO("yolo11n.pt")

student_model.train(
    data="coco1k/coco1k.yaml",
    teacher=teacher_model.model,  # None if you don't wanna use knowledge distillation
    distillation_loss="cwd",
    epochs=100,
    batch=16,
    workers=0,
    exist_ok=True,
)
```

## 📋 필요 조건

### 데이터셋
COCO 2017 데이터셋이 필요합니다:
```bash
# COCO 2017 다운로드
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# 압축 해제
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# 디렉토리 구조
/datasets/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### 하드웨어
- **GPU**: NVIDIA GPU 권장 (8GB+ VRAM)
- **메모리**: 16GB+ RAM
- **저장공간**: 10GB+ (데이터셋 + 결과)

## 📊 예상 결과

### 성능 개선
- **mAP50**: +2~5% 향상 기대
- **mAP50-95**: +1~3% 향상 기대
- **모델 크기**: 동일 (Student 모델 기준)

### 실험 시간
- **데이터셋 생성**: 10-20분
- **Baseline 훈련**: 2-3시간
- **KD 훈련**: 2.5-4시간
- **총 소요시간**: 4-6시간

## 📁 출력 구조

```
coco1k/                          # COCO-1K 데이터셋
├── train2017/                   # 훈련 이미지
├── val2017/                     # 검증 이미지
├── annotations/                 # 어노테이션
└── coco1k.yaml                  # YOLO 설정 파일

coco1k_kd_experiments/           # 실험 결과
├── baseline_YYYYMMDD_HHMMSS/    # Baseline 결과
├── kd_cwd_YYYYMMDD_HHMMSS/      # KD 실험 결과
├── comparison_YYYYMMDD_HHMMSS.json # 성능 비교 데이터
└── kd_comparison_YYYYMMDD_HHMMSS.png # 결과 시각화
```

## 🔧 커스터마이징

### 데이터셋 크기 조정
```bash
python create_coco1k.py --target-size 2000  # 2000개 이미지
```

### 다른 Teacher-Student 조합
```bash
python test_coco1k_kd.py --teacher yolo11x.pt  # 더 큰 Teacher
```

### 다른 Distillation Loss
```bash
python test_coco1k_kd.py --loss mgd  # MGD Loss 사용
```

## 🎓 Knowledge Distillation 이해

### Channel-Wise Distillation (CWD)
- 중간 레이어 feature map의 채널별 정보 전달
- Teacher의 채널별 활성화 패턴을 Student가 모방
- 논문: [Channel-wise Distillation for Semantic Segmentation](https://arxiv.org/abs/2011.13256)

### 검증 포인트
- **Loss 수렴**: Distillation loss가 안정적으로 감소하는지
- **Feature Alignment**: Student가 Teacher의 중간 표현을 학습하는지
- **성능 향상**: 동일 모델 크기에서 성능 개선되는지

## 🐛 문제 해결

### COCO 데이터셋을 찾을 수 없음
```bash
# COCO 경로 확인
python create_coco1k.py --coco-root /your/coco/path
```

### GPU 메모리 부족
```bash
# 배치 크기 줄이기
python test_coco1k_kd.py --batch 8
```

### 훈련 시간이 너무 김
```bash
# 에포크 줄이기 (빠른 검증용)
python test_coco1k_kd.py --epochs 50
```