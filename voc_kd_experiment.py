#!/usr/bin/env python3
"""
VOC Dataset Knowledge Distillation Experiment
Following the exact example code structure
Teacher: YOLOv11l (87MB) -> Student: YOLOv11n (6.5MB)
"""

from ultralytics import YOLO
import time
from datetime import datetime
import pytz
from pathlib import Path
import json

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def get_seoul_time():
    """Get current Seoul time"""
    seoul_tz = pytz.timezone('Asia/Seoul')
    return datetime.now(seoul_tz).strftime("%Y-%m-%d %H:%M:%S KST")

def run_baseline_training():
    """Run baseline training without knowledge distillation"""
    print_section("BASELINE TRAINING (No Knowledge Distillation)")
    print(f"Start Time: {get_seoul_time()}")
    print(f"Model: YOLOv11n")
    print(f"Dataset: VOC")
    print(f"Epochs: 100")
    print(f"Batch Size: 16")
    print(f"Workers: 0")
    print("-"*80)

    start_time = time.time()

    # Following example code exactly - baseline without teacher
    student_model = YOLO("yolo11n.pt")

    results = student_model.train(
        data="VOC.yaml",
        teacher=None,  # No knowledge distillation
        epochs=100,
        batch=64,
        workers=0,
        exist_ok=True,
        name="voc_baseline_yolo11n"
    )

    training_time = time.time() - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)

    print(f"\n✅ Baseline training completed in {hours}h {minutes}m")
    print(f"End Time: {get_seoul_time()}")

    # Get the best model path for later comparison
    best_model_path = Path("runs/detect/voc_baseline_yolo11n/weights/best.pt")

    return {
        "training_time": training_time,
        "model_path": str(best_model_path),
        "results": results
    }

def run_kd_training():
    """Run knowledge distillation training"""
    print_section("KNOWLEDGE DISTILLATION TRAINING")
    print(f"Start Time: {get_seoul_time()}")
    print(f"Teacher: YOLOv11l (87MB)")
    print(f"Student: YOLOv11n (6.5MB)")
    print(f"Dataset: VOC")
    print(f"Distillation Loss: CWD")
    print(f"Epochs: 100")
    print(f"Batch Size: 16")
    print(f"Workers: 0")
    print("-"*80)

    start_time = time.time()

    # Following example code exactly
    teacher_model = YOLO("yolo11l.pt")
    student_model = YOLO("yolo11n.pt")

    results = student_model.train(
        data="VOC.yaml",
        teacher=teacher_model.model,  # Enable knowledge distillation
        distillation_loss="cwd",
        epochs=100,
        batch=64,
        workers=0,
        exist_ok=True,
        name="voc_kd_yolo11n_from_11l"
    )

    training_time = time.time() - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)

    print(f"\n✅ KD training completed in {hours}h {minutes}m")
    print(f"End Time: {get_seoul_time()}")

    # Get the best model path for later comparison
    best_model_path = Path("runs/detect/voc_kd_yolo11n_from_11l/weights/best.pt")

    return {
        "training_time": training_time,
        "model_path": str(best_model_path),
        "results": results
    }

def compare_results(baseline_results, kd_results):
    """Compare and display results"""
    print_section("RESULTS COMPARISON")

    print("\n📊 Training Time Comparison:")
    print("-"*50)
    baseline_time = baseline_results["training_time"]
    kd_time = kd_results["training_time"]

    print(f"Baseline: {baseline_time/3600:.2f} hours")
    print(f"KD:       {kd_time/3600:.2f} hours")
    print(f"Difference: {(kd_time - baseline_time)/3600:.2f} hours")

    # Validate both models for fair comparison
    print("\n📊 Validating Final Models on VOC...")
    print("-"*50)

    # Load and validate baseline model
    baseline_model = YOLO(baseline_results["model_path"])
    baseline_metrics = baseline_model.val(data="VOC.yaml")

    # Load and validate KD model
    kd_model = YOLO(kd_results["model_path"])
    kd_metrics = kd_model.val(data="VOC.yaml")

    # Display metrics comparison
    print("\n📈 Performance Metrics:")
    print("-"*50)
    print(f"{'Metric':<15} | {'Baseline':<12} | {'KD':<12} | {'Improvement':<12}")
    print("-"*50)

    metrics_to_compare = [
        ("mAP50-95", "metrics/mAP50-95(B)"),
        ("mAP50", "metrics/mAP50(B)"),
        ("Precision", "metrics/precision(B)"),
        ("Recall", "metrics/recall(B)")
    ]

    improvements = []
    for metric_name, metric_key in metrics_to_compare:
        baseline_val = baseline_metrics.results_dict.get(metric_key, 0)
        kd_val = kd_metrics.results_dict.get(metric_key, 0)

        if baseline_val > 0:
            improvement = ((kd_val - baseline_val) / baseline_val) * 100
        else:
            improvement = 0

        improvements.append(improvement)

        # Color code improvements
        if improvement > 0:
            imp_str = f"+{improvement:.2f}%"
            color = "\033[92m"  # Green
        elif improvement < 0:
            imp_str = f"{improvement:.2f}%"
            color = "\033[91m"  # Red
        else:
            imp_str = "0.00%"
            color = "\033[93m"  # Yellow

        print(f"{metric_name:<15} | {baseline_val:<12.4f} | {kd_val:<12.4f} | {color}{imp_str}\033[0m")

    # Average improvement
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0

    print("\n" + "="*80)
    print(f"📊 Average Performance Change: {avg_improvement:+.2f}%")

    if avg_improvement > 0:
        print("✅ Knowledge Distillation IMPROVED performance!")
    elif avg_improvement < 0:
        print("⚠️  Knowledge Distillation decreased performance.")
    else:
        print("➖ Knowledge Distillation showed no significant change.")

    # Save results to JSON
    results_data = {
        "experiment": "VOC Knowledge Distillation",
        "timestamp": get_seoul_time(),
        "teacher": "YOLOv11l",
        "student": "YOLOv11n",
        "dataset": "VOC",
        "epochs": 100,
        "batch_size": 16,
        "baseline": {
            "training_time": baseline_time,
            "mAP50-95": baseline_metrics.results_dict.get("metrics/mAP50-95(B)", 0),
            "mAP50": baseline_metrics.results_dict.get("metrics/mAP50(B)", 0)
        },
        "kd": {
            "training_time": kd_time,
            "mAP50-95": kd_metrics.results_dict.get("metrics/mAP50-95(B)", 0),
            "mAP50": kd_metrics.results_dict.get("metrics/mAP50(B)", 0)
        },
        "improvement": {
            "average": avg_improvement,
            "details": dict(zip([m[0] for m in metrics_to_compare], improvements))
        }
    }

    # Save to file
    results_file = Path("voc_kd_results.json")
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=4, default=str)

    print(f"\n💾 Results saved to: {results_file}")

    return results_data

def main():
    """Main experiment execution"""
    print_section("VOC KNOWLEDGE DISTILLATION EXPERIMENT")
    print(f"Experiment Start: {get_seoul_time()}")
    print(f"Configuration:")
    print(f"  - Dataset: Pascal VOC (~17K training images)")
    print(f"  - Teacher Model: YOLOv11l (87MB)")
    print(f"  - Student Model: YOLOv11n (6.5MB)")
    print(f"  - Teacher/Student Ratio: 13:1")
    print(f"  - Epochs: 100")
    print(f"  - Batch Size: 16")
    print(f"  - Workers: 0 (for stability)")
    print(f"  - Distillation Loss: CWD")

    try:
        # Check GPU availability
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("\n⚠️  No GPU detected - Training will be slow!")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Experiment cancelled.")
                return

        # Run experiments
        print("\n" + "="*80)
        print("  PHASE 1/3: Baseline Training")
        print("="*80)
        baseline_results = run_baseline_training()

        print("\n" + "="*80)
        print("  PHASE 2/3: Knowledge Distillation Training")
        print("="*80)
        kd_results = run_kd_training()

        print("\n" + "="*80)
        print("  PHASE 3/3: Results Comparison")
        print("="*80)
        comparison = compare_results(baseline_results, kd_results)

        print_section("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"Experiment End: {get_seoul_time()}")

        # Total time
        total_time = baseline_results["training_time"] + kd_results["training_time"]
        print(f"Total Experiment Time: {total_time/3600:.2f} hours")

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user.")
        print("Partial results may have been saved.")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        print("Please check the error and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()