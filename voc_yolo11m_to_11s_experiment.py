#!/usr/bin/env python3
"""
VOC Dataset YOLOv11m → YOLOv11s Knowledge Distillation Experiment
최적화된 설정으로 베이스라인과 KD 성능 비교
"""

import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from ultralytics import YOLO

def get_kst_timestamp():
    """한국 시간 타임스탬프 생성"""
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    return now.strftime("%Y%m%d_%H%M%S"), now.strftime("%Y-%m-%d %H:%M:%S KST")

def format_training_time(seconds):
    """학습 시간을 시:분:초 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"

def get_model_size(model_path):
    """모델 파일 크기를 MB 단위로 반환"""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        return size_bytes / (1024 * 1024)  # MB로 변환
    return 0

def run_baseline_yolo11s(epochs=100, batch=64, workers=12):
    """YOLOv11s 베이스라인 학습"""
    print("=" * 60)
    print("🚀 YOLOv11s 베이스라인 학습 시작")
    print(f"📊 설정: epochs={epochs}, batch={batch}, workers={workers}")
    print("=" * 60)

    start_time = time.time()

    # 학생 모델 초기화
    student_model = YOLO("yolo11s.pt")

    # 학습 실행
    results = student_model.train(
        data="VOC.yaml",
        epochs=epochs,
        batch=batch,
        workers=workers,
        exist_ok=True,
        name=f"voc_baseline_yolo11s_optimized_{get_kst_timestamp()[0]}"
    )

    end_time = time.time()
    training_time = end_time - start_time

    print(f"✅ 베이스라인 학습 완료! 소요시간: {format_training_time(training_time)}")

    return {
        "training_time": training_time,
        "training_time_str": format_training_time(training_time),
        "model_path": results.save_dir / "weights" / "best.pt",
        "results": results
    }

def run_kd_yolo11m_to_11s(epochs=100, batch=64, workers=12):
    """YOLOv11m → YOLOv11s Knowledge Distillation 학습"""
    print("=" * 60)
    print("🎓 YOLOv11m → YOLOv11s KD 학습 시작")
    print(f"📊 설정: epochs={epochs}, batch={batch}, workers={workers}")
    print("=" * 60)

    start_time = time.time()

    # 교사 모델과 학생 모델 초기화
    teacher_model = YOLO("yolo11m.pt")
    student_model = YOLO("yolo11s.pt")

    print(f"👨‍🏫 Teacher: YOLOv11m ({teacher_model.model.model[-1].nc} classes)")
    print(f"👨‍🎓 Student: YOLOv11s ({student_model.model.model[-1].nc} classes)")

    # KD 학습 실행
    results = student_model.train(
        data="VOC.yaml",
        teacher=teacher_model.model,
        distillation_loss="cwd",
        epochs=epochs,
        batch=batch,
        workers=workers,
        exist_ok=True,
        name=f"voc_kd_yolo11s_from_11m_optimized_{get_kst_timestamp()[0]}"
    )

    end_time = time.time()
    training_time = end_time - start_time

    print(f"✅ KD 학습 완료! 소요시간: {format_training_time(training_time)}")

    return {
        "training_time": training_time,
        "training_time_str": format_training_time(training_time),
        "model_path": results.save_dir / "weights" / "best.pt",
        "results": results
    }

def extract_metrics_from_results(results_dir):
    """results.csv에서 최종 메트릭 추출"""
    csv_path = os.path.join(results_dir, "results.csv")
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    last_row = df.iloc[-1]

    return {
        "metrics/precision(B)": last_row.get("metrics/precision(B)", 0),
        "metrics/recall(B)": last_row.get("metrics/recall(B)", 0),
        "metrics/mAP50(B)": last_row.get("metrics/mAP50(B)", 0),
        "metrics/mAP50-95(B)": last_row.get("metrics/mAP50-95(B)", 0),
        "fitness": last_row.get("fitness", 0) if "fitness" in last_row else 0
    }

def create_comparison_chart(baseline_metrics, kd_metrics, save_path):
    """비교 차트 생성"""
    metrics = ['precision', 'recall', 'mAP50', 'mAP50-95']
    baseline_values = [
        baseline_metrics.get("metrics/precision(B)", 0),
        baseline_metrics.get("metrics/recall(B)", 0),
        baseline_metrics.get("metrics/mAP50(B)", 0),
        baseline_metrics.get("metrics/mAP50-95(B)", 0)
    ]
    kd_values = [
        kd_metrics.get("metrics/precision(B)", 0),
        kd_metrics.get("metrics/recall(B)", 0),
        kd_metrics.get("metrics/mAP50(B)", 0),
        kd_metrics.get("metrics/mAP50-95(B)", 0)
    ]

    # 개선율 계산
    improvements = []
    for b, k in zip(baseline_values, kd_values):
        if b > 0:
            improvement = ((k - b) / b) * 100
            improvements.append(improvement)
        else:
            improvements.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 절대값 비교
    x = range(len(metrics))
    width = 0.35
    ax1.bar([i - width/2 for i in x], baseline_values, width, label='Baseline', alpha=0.8, color='skyblue')
    ax1.bar([i + width/2 for i in x], kd_values, width, label='Knowledge Distillation', alpha=0.8, color='lightgreen')

    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Value')
    ax1.set_title('Baseline vs Knowledge Distillation Performance\n(VOC Dataset - YOLOv11m→YOLOv11s)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 값 표시
    for i, (b, k) in enumerate(zip(baseline_values, kd_values)):
        ax1.text(i - width/2, b + 0.01, f'{b:.3f}', ha='center', va='bottom')
        ax1.text(i + width/2, k + 0.01, f'{k:.3f}', ha='center', va='bottom')

    # 개선율 비교
    colors = ['red' if imp < 0 else 'green' for imp in improvements]
    bars = ax2.bar(metrics, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Performance Change with Knowledge Distillation\n(VOC Dataset - YOLOv11m→YOLOv11s)')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # 개선율 값 표시
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -0.5),
                f'{imp:.2f}%', ha='center', va='bottom' if height >= 0 else 'top')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_full_experiment():
    """전체 실험 실행"""
    timestamp, kst_time = get_kst_timestamp()

    print("🔬 VOC YOLOv11m → YOLOv11s Knowledge Distillation 실험 시작")
    print(f"⏰ 시작 시간: {kst_time}")
    print("💪 최적화된 설정: batch=64, workers=12 (A100 40GB 최적화)")

    # 1. 베이스라인 학습
    baseline_result = run_baseline_yolo11s()
    baseline_metrics = extract_metrics_from_results(baseline_result["model_path"].parent.parent)

    # 2. KD 학습
    kd_result = run_kd_yolo11m_to_11s()
    kd_metrics = extract_metrics_from_results(kd_result["model_path"].parent.parent)

    # 3. 결과 정리
    experiment_results = {
        "experiment_info": {
            "timestamp": timestamp,
            "timestamp_kst": kst_time,
            "data": "VOC.yaml",
            "dataset_size": "VOC 2012 (17,125 train, 4,952 val)",
            "epochs": 100,
            "batch_size": 64,
            "workers": 12,
            "teacher_model": "yolo11m",
            "student_model": "yolo11s",
            "distillation_loss": "cwd",
            "device": "0"
        },
        "baseline": {
            "training_time": baseline_result["training_time"],
            "training_time_str": baseline_result["training_time_str"],
            "model_path": str(baseline_result["model_path"]),
            "metrics": baseline_metrics
        },
        "knowledge_distillation": {
            "training_time": kd_result["training_time"],
            "training_time_str": kd_result["training_time_str"],
            "model_path": str(kd_result["model_path"]),
            "metrics": kd_metrics
        }
    }

    # 비교 메트릭 계산
    comparison = {}
    for metric in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        baseline_val = baseline_metrics.get(metric, 0)
        kd_val = kd_metrics.get(metric, 0)
        improvement = ((kd_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0

        comparison[metric] = {
            "baseline": baseline_val,
            "kd": kd_val,
            "improvement_percent": improvement
        }

    # 학습 시간 비교
    comparison["training_time"] = {
        "baseline": baseline_result["training_time"],
        "kd": kd_result["training_time"],
        "baseline_str": baseline_result["training_time_str"],
        "kd_str": kd_result["training_time_str"]
    }

    # 모델 크기 비교
    comparison["model_size_mb"] = {
        "baseline": get_model_size(baseline_result["model_path"]),
        "kd": get_model_size(kd_result["model_path"])
    }

    experiment_results["comparison"] = comparison

    # 4. 결과 저장
    output_dir = f"kd_experiments_voc_11m_to_11s"
    os.makedirs(output_dir, exist_ok=True)

    # JSON 저장
    json_path = f"{output_dir}/kd_results_voc_11m_to_11s_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_results, f, indent=4, ensure_ascii=False)

    # CSV 요약 저장
    summary_data = {
        'Metric': ['precision', 'recall', 'mAP50', 'mAP50-95'],
        'Baseline': [
            baseline_metrics.get("metrics/precision(B)", 0),
            baseline_metrics.get("metrics/recall(B)", 0),
            baseline_metrics.get("metrics/mAP50(B)", 0),
            baseline_metrics.get("metrics/mAP50-95(B)", 0)
        ],
        'KD': [
            kd_metrics.get("metrics/precision(B)", 0),
            kd_metrics.get("metrics/recall(B)", 0),
            kd_metrics.get("metrics/mAP50(B)", 0),
            kd_metrics.get("metrics/mAP50-95(B)", 0)
        ]
    }

    for i, metric in enumerate(['precision', 'recall', 'mAP50', 'mAP50-95']):
        baseline_val = summary_data['Baseline'][i]
        kd_val = summary_data['KD'][i]
        improvement = ((kd_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
        if i == 0:
            summary_data['Improvement (%)'] = [improvement]
        else:
            summary_data['Improvement (%)'].append(improvement)

    df_summary = pd.DataFrame(summary_data)
    csv_path = f"{output_dir}/kd_summary_voc_11m_to_11s_{timestamp}.csv"
    df_summary.to_csv(csv_path, index=False, float_format='%.4f')

    # 차트 생성
    chart_path = f"{output_dir}/kd_comparison_voc_11m_to_11s_{timestamp}.png"
    create_comparison_chart(baseline_metrics, kd_metrics, chart_path)

    print("=" * 60)
    print("🎉 실험 완료!")
    print(f"📁 결과 저장 위치: {output_dir}/")
    print(f"📊 JSON: {json_path}")
    print(f"📈 CSV: {csv_path}")
    print(f"📊 차트: {chart_path}")
    print("=" * 60)

    return experiment_results

def test_one_epoch():
    """1 에포크 테스트"""
    print("🧪 1 에포크 테스트 시작")

    # 베이스라인 1 에포크
    print("1️⃣ YOLOv11s 베이스라인 1 에포크 테스트")
    baseline_result = run_baseline_yolo11s(epochs=1, batch=64, workers=12)

    # KD 1 에포크
    print("2️⃣ YOLOv11m → YOLOv11s KD 1 에포크 테스트")
    kd_result = run_kd_yolo11m_to_11s(epochs=1, batch=64, workers=12)

    print("✅ 1 에포크 테스트 완료!")
    print(f"📊 베이스라인 시간: {baseline_result['training_time_str']}")
    print(f"🎓 KD 시간: {kd_result['training_time_str']}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_one_epoch()
    else:
        run_full_experiment()