#!/usr/bin/env python3
"""
COCO128 Baseline vs Knowledge Distillation Comparison - 100 Epochs
Complete performance comparison following exact user specification
"""

import time
import json
import gc
import torch
from datetime import datetime
import pytz
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params, get_flops

# Configuration
CONFIG = {
    'improvement_threshold': 0.01,  # 1% improvement threshold
    'expected_time_hours': 4.5,     # Expected total time
    'timezone': 'Asia/Seoul'
}

# Global timezone to avoid duplication
SEOUL_TZ = pytz.timezone(CONFIG['timezone'])

def run_baseline_experiment():
    """Run baseline training (no KD)"""
    print("\n" + "="*60)
    print("🔥 BASELINE EXPERIMENT (No Knowledge Distillation)")
    print("="*60)

    start_time_str = datetime.now(SEOUL_TZ).strftime("%Y-%m-%d %H:%M:%S KST")
    print(f"Start Time: {start_time_str}")
    print(f"Model: YOLOv11n (Student only)")
    print(f"Dataset: COCO128")
    print(f"Epochs: 100")
    print("-"*60)

    try:
        # Baseline training (exact same settings but no teacher)
        student_model = YOLO("yolo11n.pt")

        print("🚀 Starting Baseline training...")
        start_time = time.time()

        baseline_results = student_model.train(
            data="coco128.yaml",
            epochs=100,
            batch=16,
            workers=0,
            exist_ok=True,
            project="coco128_comparison",
            name="baseline_100ep"
        )

        baseline_time = time.time() - start_time
        end_time_str = datetime.now(SEOUL_TZ).strftime("%Y-%m-%d %H:%M:%S KST")

        # Extract metrics with validation
        baseline_map50 = float(baseline_results.results_dict.get('metrics/mAP50(B)', 0))
        baseline_map50_95 = float(baseline_results.results_dict.get('metrics/mAP50-95(B)', 0))

        # Validate metrics extraction
        if baseline_map50 == 0 and baseline_map50_95 == 0:
            print("⚠️ Warning: Extracted metrics are 0. Available keys:")
            print(list(baseline_results.results_dict.keys()))

        # Get model info with dependency check
        try:
            params = get_num_params(student_model.model)
            gflops = round(get_flops(student_model.model), 3) if get_flops(student_model.model) > 0 else 'N/A'
        except Exception as e:
            print(f"⚠️ Warning: Could not get model info: {e}")
            params = 'N/A'
            gflops = 'N/A'

        baseline_metrics = {
            'model': 'yolo11n_baseline',
            'epochs': 100,
            'training_time_hours': baseline_time / 3600,
            'training_time_minutes': baseline_time / 60,
            'map50': baseline_map50,
            'map50_95': baseline_map50_95,
            'params': params,
            'gflops': gflops,
            'start_time': start_time_str,
            'end_time': end_time_str
        }

        print(f"\n✅ Baseline Training Complete!")
        print(f"End Time: {end_time_str}")
        print(f"Training Time: {baseline_time/3600:.2f} hours")
        print(f"Final mAP50: {baseline_map50:.4f}")
        print(f"Final mAP50-95: {baseline_map50_95:.4f}")

        # Memory cleanup
        del student_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        return baseline_metrics

    except Exception as e:
        error_time = datetime.now(SEOUL_TZ).strftime("%Y-%m-%d %H:%M:%S KST")
        print(f"\n❌ Baseline Training Failed at {error_time}!")
        print(f"Error: {str(e)}")

        # Save error info
        error_metrics = {
            'model': 'yolo11n_baseline',
            'status': 'failed',
            'error': str(e),
            'failed_time': error_time,
            'start_time': start_time_str
        }

        # Memory cleanup on failure
        try:
            del student_model
        except:
            pass
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        return error_metrics

def run_kd_experiment():
    """Run Knowledge Distillation training"""
    print("\n" + "="*60)
    print("🎓 KNOWLEDGE DISTILLATION EXPERIMENT")
    print("="*60)

    start_time_str = datetime.now(SEOUL_TZ).strftime("%Y-%m-%d %H:%M:%S KST")
    print(f"Start Time: {start_time_str}")
    print(f"Teacher: YOLOv11l")
    print(f"Student: YOLOv11n")
    print(f"Dataset: COCO128")
    print(f"Epochs: 100")
    print(f"Distillation Loss: CWD")
    print("-"*60)

    try:
        # Following exact user specification
        print("📚 Loading Teacher model (YOLOv11l)...")
        teacher_model = YOLO("yolo11l.pt")

        print("🎓 Loading Student model (YOLOv11n)...")
        student_model = YOLO("yolo11n.pt")

        print("🚀 Starting Knowledge Distillation training...")
        start_time = time.time()

        kd_results = student_model.train(
            data="coco128.yaml",
            teacher=teacher_model.model,  # None if you don't wanna use knowledge distillation
            distillation_loss="cwd",
            epochs=100,
            batch=16,
            workers=0,
            exist_ok=True,
            project="coco128_comparison",
            name="kd_cwd_100ep"
        )

        kd_time = time.time() - start_time
        end_time_str = datetime.now(SEOUL_TZ).strftime("%Y-%m-%d %H:%M:%S KST")

        # Extract metrics with validation
        kd_map50 = float(kd_results.results_dict.get('metrics/mAP50(B)', 0))
        kd_map50_95 = float(kd_results.results_dict.get('metrics/mAP50-95(B)', 0))

        # Validate metrics extraction
        if kd_map50 == 0 and kd_map50_95 == 0:
            print("⚠️ Warning: Extracted metrics are 0. Available keys:")
            print(list(kd_results.results_dict.keys()))

        # Get model info with dependency check
        try:
            params = get_num_params(student_model.model)
            gflops = round(get_flops(student_model.model), 3) if get_flops(student_model.model) > 0 else 'N/A'
        except Exception as e:
            print(f"⚠️ Warning: Could not get model info: {e}")
            params = 'N/A'
            gflops = 'N/A'

        kd_metrics = {
            'model': 'yolo11n_kd_cwd',
            'teacher': 'yolo11l',
            'distillation_loss': 'cwd',
            'epochs': 100,
            'training_time_hours': kd_time / 3600,
            'training_time_minutes': kd_time / 60,
            'map50': kd_map50,
            'map50_95': kd_map50_95,
            'params': params,
            'gflops': gflops,
            'start_time': start_time_str,
            'end_time': end_time_str
        }

        print(f"\n✅ Knowledge Distillation Training Complete!")
        print(f"End Time: {end_time_str}")
        print(f"Training Time: {kd_time/3600:.2f} hours")
        print(f"Final mAP50: {kd_map50:.4f}")
        print(f"Final mAP50-95: {kd_map50_95:.4f}")

        # Memory cleanup
        del teacher_model, student_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        return kd_metrics

    except Exception as e:
        error_time = datetime.now(SEOUL_TZ).strftime("%Y-%m-%d %H:%M:%S KST")
        print(f"\n❌ Knowledge Distillation Training Failed at {error_time}!")
        print(f"Error: {str(e)}")

        # Save error info
        error_metrics = {
            'model': 'yolo11n_kd_cwd',
            'teacher': 'yolo11l',
            'status': 'failed',
            'error': str(e),
            'failed_time': error_time,
            'start_time': start_time_str
        }

        # Memory cleanup on failure
        try:
            del teacher_model, student_model
        except:
            pass
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        return error_metrics

def compare_results(baseline_metrics, kd_metrics):
    """Compare baseline vs KD results"""
    print("\n" + "="*80)
    print("🏆 FINAL COMPARISON: BASELINE vs KNOWLEDGE DISTILLATION")
    print("="*80)

    # Handle failed experiments
    if 'status' in baseline_metrics and baseline_metrics['status'] == 'failed':
        print("❌ BASELINE EXPERIMENT FAILED - Cannot compare results")
        print(f"Baseline Error: {baseline_metrics.get('error', 'Unknown error')}")
        return None

    if 'status' in kd_metrics and kd_metrics['status'] == 'failed':
        print("❌ KNOWLEDGE DISTILLATION EXPERIMENT FAILED - Cannot compare results")
        print(f"KD Error: {kd_metrics.get('error', 'Unknown error')}")
        return None

    # Calculate improvements
    map50_improvement = kd_metrics['map50'] - baseline_metrics['map50']
    map50_95_improvement = kd_metrics['map50_95'] - baseline_metrics['map50_95']
    map50_improvement_pct = (map50_improvement / baseline_metrics['map50']) * 100 if baseline_metrics['map50'] > 0 else 0
    map50_95_improvement_pct = (map50_95_improvement / baseline_metrics['map50_95']) * 100 if baseline_metrics['map50_95'] > 0 else 0

    print(f"📊 Dataset: COCO128 (128 images, 80 classes)")
    print(f"🔄 Training: 100 epochs")

    # Handle parameter display for both numeric and string values
    params_display = f"{kd_metrics['params']:,}" if isinstance(kd_metrics['params'], int) else kd_metrics['params']
    gflops_display = f"{kd_metrics['gflops']:.1f}" if isinstance(kd_metrics['gflops'], (int, float)) else kd_metrics['gflops']
    print(f"🏗️  Model Size: {params_display} parameters, {gflops_display} GFLOPs")

    print()
    print("📈 PERFORMANCE METRICS:")
    print("-" * 80)
    print(f"{'Metric':<15} {'Baseline':<12} {'KD (CWD)':<12} {'Improvement':<15} {'% Change':<10}")
    print("-" * 80)
    print(f"{'mAP50':<15} {baseline_metrics['map50']:<12.4f} {kd_metrics['map50']:<12.4f} {map50_improvement:<+15.4f} {map50_improvement_pct:<+10.2f}%")
    print(f"{'mAP50-95':<15} {baseline_metrics['map50_95']:<12.4f} {kd_metrics['map50_95']:<12.4f} {map50_95_improvement:<+15.4f} {map50_95_improvement_pct:<+10.2f}%")
    print("-" * 80)

    print("\n⏱️  TRAINING TIME:")
    print("-" * 50)
    print(f"Baseline:     {baseline_metrics['training_time_hours']:.2f} hours")
    print(f"KD:           {kd_metrics['training_time_hours']:.2f} hours")
    print(f"Overhead:     {kd_metrics['training_time_hours'] - baseline_metrics['training_time_hours']:+.2f} hours")

    # Evaluation with configurable threshold
    print("\n🎯 KNOWLEDGE DISTILLATION EVALUATION:")
    print("-" * 50)
    if map50_improvement > CONFIG['improvement_threshold']:  # Use configurable threshold
        print("✅ SUCCESS: Knowledge Distillation improved performance!")
        print(f"   → mAP50 improved by {map50_improvement_pct:.2f}%")
        print(f"   → mAP50-95 improved by {map50_95_improvement_pct:.2f}%")
    elif map50_improvement > 0:
        print("⚡ MARGINAL: Knowledge Distillation showed slight improvement")
        print(f"   → mAP50 improved by {map50_improvement_pct:.2f}%")
    else:
        print("❌ NO IMPROVEMENT: Knowledge Distillation did not help")
        print("   → Consider different teacher model, loss function, or hyperparameters")

    # Save comparison results
    comparison = {
        'experiment_type': 'coco128_baseline_vs_kd',
        'dataset': 'coco128.yaml',
        'epochs': 100,
        'baseline': baseline_metrics,
        'kd': kd_metrics,
        'improvements': {
            'map50_absolute': map50_improvement,
            'map50_relative_pct': map50_improvement_pct,
            'map50_95_absolute': map50_95_improvement,
            'map50_95_relative_pct': map50_95_improvement_pct,
        }
    }

    # Save to file
    results_dir = Path("coco128_comparison")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(results_dir / f"comparison_{timestamp}.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n💾 Results saved to: coco128_comparison/comparison_{timestamp}.json")
    print("="*80)

    return comparison

def main():
    """Run complete baseline vs KD comparison"""
    experiment_start = datetime.now(SEOUL_TZ).strftime("%Y-%m-%d %H:%M:%S KST")

    print("🔬 COCO128 BASELINE vs KNOWLEDGE DISTILLATION COMPARISON")
    print("="*80)
    print(f"Experiment Start: {experiment_start}")
    print(f"Total Expected Time: ~{CONFIG['expected_time_hours']:.1f} hours (2 experiments × 100 epochs)")
    print("="*80)

    # Initialize tracking variables
    baseline_metrics = None
    kd_metrics = None
    total_time = 0

    try:
        # Run both experiments
        print("\n🎬 Starting Experiment 1/2: Baseline Training")
        baseline_metrics = run_baseline_experiment()

        # Only proceed to KD if baseline succeeded
        if 'status' not in baseline_metrics or baseline_metrics['status'] != 'failed':
            print("\n🎬 Starting Experiment 2/2: Knowledge Distillation Training")
            kd_metrics = run_kd_experiment()
        else:
            print("\n⚠️ Skipping KD experiment due to baseline failure")
            kd_metrics = {'status': 'skipped', 'reason': 'baseline_failed'}

        # Compare results if both experiments completed
        if (baseline_metrics and kd_metrics and
            'status' not in baseline_metrics and 'status' not in kd_metrics):
            comparison_result = compare_results(baseline_metrics, kd_metrics)
            total_time = baseline_metrics.get('training_time_hours', 0) + kd_metrics.get('training_time_hours', 0)

            if comparison_result:
                print("✅ Comparison completed and saved successfully")
        else:
            print("\n⚠️ Cannot perform comparison due to experiment failures")

    except KeyboardInterrupt:
        print("\n\n⚠️ Experiment interrupted by user (Ctrl+C)")
        print("Partial results may be available in the output directories")
    except Exception as e:
        print(f"\n\n❌ Unexpected error in main(): {str(e)}")
    finally:
        # Final cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        experiment_end = datetime.now(SEOUL_TZ).strftime("%Y-%m-%d %H:%M:%S KST")
        print(f"\n🎉 Experiment Session Ended!")
        print(f"Experiment End: {experiment_end}")
        if total_time > 0:
            print(f"Total Training Time: {total_time:.2f} hours")
        print("Results saved in: coco128_comparison/ directory")

if __name__ == "__main__":
    main()