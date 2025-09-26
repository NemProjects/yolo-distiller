#!/usr/bin/env python3
"""
COCO-1K Knowledge Distillation Test Script
Test KD effectiveness using the created COCO-1K dataset
"""

import time
import json
from pathlib import Path
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

class COCO1KKDTester:
    def __init__(self, data_yaml="coco1k/coco1k.yaml", epochs=100, batch_size=16):
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch_size = batch_size
        self.results_dir = Path("coco1k_kd_experiments")
        self.results_dir.mkdir(exist_ok=True)

        # Use Seoul timezone (KST)
        seoul_tz = pytz.timezone('Asia/Seoul')
        self.timestamp = datetime.now(seoul_tz).strftime("%Y%m%d_%H%M%S")

        # GPU check
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  Using CPU - Training will be slower")

    def run_baseline_training(self):
        """Train YOLOv11n without knowledge distillation on COCO-1K"""
        print("\n" + "="*60)
        print("🚀 Starting BASELINE Training (YOLOv11n) on COCO-1K")
        print("="*60)

        seoul_tz = pytz.timezone('Asia/Seoul')
        current_time = datetime.now(seoul_tz).strftime("%Y-%m-%d %H:%M:%S KST")
        print(f"Start Time: {current_time}")
        print(f"Dataset: COCO-1K (~1500 training images)")
        print(f"Epochs: {self.epochs}")
        print(f"Batch Size: {self.batch_size}")
        print("-"*60)

        student_model = YOLO("yolo11n.pt")

        start_time = time.time()
        results = student_model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            workers=0,  # Same as example
            exist_ok=True,  # Same as example
            device=self.device,
            project=str(self.results_dir),
            name=f"baseline_{self.timestamp}",
            save=True,
            cache=True,
            patience=30
        )

        training_time = time.time() - start_time

        baseline_metrics = {
            'model': 'yolo11n_baseline',
            'training_time_hours': training_time / 3600,
            'final_map50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
            'final_map50_95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
            'params': student_model.model.get_n_params(),
            'gflops': student_model.model.get_gflops(),
        }

        print(f"\n📊 Baseline Results:")
        print(f"   Training Time: {training_time/3600:.2f} hours")
        print(f"   mAP50: {baseline_metrics['final_map50']:.4f}")
        print(f"   mAP50-95: {baseline_metrics['final_map50_95']:.4f}")
        print(f"   Parameters: {baseline_metrics['params']:,}")

        return baseline_metrics, results

    def run_kd_training(self, teacher_model="yolo11l.pt", distillation_loss="cwd"):
        """Train YOLOv11n with knowledge distillation - following exact example format"""
        print("\n" + "="*60)
        print(f"🎓 Starting KNOWLEDGE DISTILLATION Training")
        print(f"   Teacher: {teacher_model}")
        print(f"   Student: yolo11n.pt")
        print(f"   Distillation Loss: {distillation_loss}")
        print("="*60)

        seoul_tz = pytz.timezone('Asia/Seoul')
        current_time = datetime.now(seoul_tz).strftime("%Y-%m-%d %H:%M:%S KST")
        print(f"Start Time: {current_time}")
        print("-"*60)

        # Load models exactly as in the example
        teacher_model_obj = YOLO(teacher_model)
        student_model = YOLO("yolo11n.pt")

        start_time = time.time()

        # Train with KD exactly as in the example
        results = student_model.train(
            data=self.data_yaml,
            teacher=teacher_model_obj.model,  # Use .model as in example
            distillation_loss=distillation_loss,
            epochs=self.epochs,
            batch=self.batch_size,
            workers=0,  # Same as example
            exist_ok=True,  # Same as example
            device=self.device,
            project=str(self.results_dir),
            name=f"kd_{distillation_loss}_{self.timestamp}",
            save=True,
            cache=True,
            patience=30
        )

        training_time = time.time() - start_time

        kd_metrics = {
            'model': f'yolo11n_kd_{distillation_loss}',
            'teacher': teacher_model,
            'distillation_loss': distillation_loss,
            'training_time_hours': training_time / 3600,
            'final_map50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
            'final_map50_95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
            'params': student_model.model.get_n_params(),
            'gflops': student_model.model.get_gflops(),
        }

        print(f"\n📊 Knowledge Distillation Results:")
        print(f"   Training Time: {training_time/3600:.2f} hours")
        print(f"   mAP50: {kd_metrics['final_map50']:.4f}")
        print(f"   mAP50-95: {kd_metrics['final_map50_95']:.4f}")
        print(f"   Parameters: {kd_metrics['params']:,}")

        return kd_metrics, results

    def compare_results(self, baseline_metrics, kd_metrics):
        """Compare baseline vs KD results"""
        print("\n" + "="*60)
        print("📈 COMPARISON RESULTS")
        print("="*60)

        map50_improvement = kd_metrics['final_map50'] - baseline_metrics['final_map50']
        map50_95_improvement = kd_metrics['final_map50_95'] - baseline_metrics['final_map50_95']

        print(f"Model Size: {kd_metrics['params']:,} parameters")
        print(f"Computational Cost: {kd_metrics['gflops']:.2f} GFLOPs")
        print("\nPerformance Comparison:")
        print(f"  Baseline mAP50:     {baseline_metrics['final_map50']:.4f}")
        print(f"  KD mAP50:           {kd_metrics['final_map50']:.4f}")
        print(f"  Improvement:        {map50_improvement:+.4f} ({map50_improvement/baseline_metrics['final_map50']*100:+.2f}%)")
        print()
        print(f"  Baseline mAP50-95:  {baseline_metrics['final_map50_95']:.4f}")
        print(f"  KD mAP50-95:        {kd_metrics['final_map50_95']:.4f}")
        print(f"  Improvement:        {map50_95_improvement:+.4f} ({map50_95_improvement/baseline_metrics['final_map50_95']*100:+.2f}%)")

        print(f"\nTraining Time:")
        print(f"  Baseline:           {baseline_metrics['training_time_hours']:.2f} hours")
        print(f"  KD:                 {kd_metrics['training_time_hours']:.2f} hours")
        print(f"  Overhead:           {kd_metrics['training_time_hours'] - baseline_metrics['training_time_hours']:.2f} hours")

        # Save comparison results
        comparison = {
            'timestamp': self.timestamp,
            'baseline': baseline_metrics,
            'kd': kd_metrics,
            'improvements': {
                'map50_abs': map50_improvement,
                'map50_rel': map50_improvement/baseline_metrics['final_map50']*100,
                'map50_95_abs': map50_95_improvement,
                'map50_95_rel': map50_95_improvement/baseline_metrics['final_map50_95']*100,
            }
        }

        with open(self.results_dir / f"comparison_{self.timestamp}.json", 'w') as f:
            json.dump(comparison, f, indent=2)

        return comparison

    def create_plots(self, comparison):
        """Create visualization plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # mAP50 comparison
        models = ['Baseline', 'KD']
        map50_values = [comparison['baseline']['final_map50'], comparison['kd']['final_map50']]
        ax1.bar(models, map50_values, color=['lightblue', 'lightgreen'])
        ax1.set_ylabel('mAP50')
        ax1.set_title('mAP50 Comparison')
        ax1.set_ylim(0, max(map50_values) * 1.1)
        for i, v in enumerate(map50_values):
            ax1.text(i, v + max(map50_values)*0.01, f'{v:.4f}', ha='center')

        # mAP50-95 comparison
        map50_95_values = [comparison['baseline']['final_map50_95'], comparison['kd']['final_map50_95']]
        ax2.bar(models, map50_95_values, color=['lightblue', 'lightgreen'])
        ax2.set_ylabel('mAP50-95')
        ax2.set_title('mAP50-95 Comparison')
        ax2.set_ylim(0, max(map50_95_values) * 1.1)
        for i, v in enumerate(map50_95_values):
            ax2.text(i, v + max(map50_95_values)*0.01, f'{v:.4f}', ha='center')

        # Training time comparison
        time_values = [comparison['baseline']['training_time_hours'], comparison['kd']['training_time_hours']]
        ax3.bar(models, time_values, color=['lightcoral', 'lightsalmon'])
        ax3.set_ylabel('Training Time (hours)')
        ax3.set_title('Training Time Comparison')
        for i, v in enumerate(time_values):
            ax3.text(i, v + max(time_values)*0.01, f'{v:.2f}h', ha='center')

        # Improvement summary
        improvements = [
            comparison['improvements']['map50_rel'],
            comparison['improvements']['map50_95_rel']
        ]
        metrics = ['mAP50', 'mAP50-95']
        colors = ['green' if x > 0 else 'red' for x in improvements]
        ax4.bar(metrics, improvements, color=colors, alpha=0.7)
        ax4.set_ylabel('Improvement (%)')
        ax4.set_title('KD Performance Improvement')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        for i, v in enumerate(improvements):
            ax4.text(i, v + (max(improvements) if max(improvements) > 0 else min(improvements))*0.05,
                    f'{v:+.2f}%', ha='center')

        plt.tight_layout()
        plt.savefig(self.results_dir / f"kd_comparison_{self.timestamp}.png", dpi=300, bbox_inches='tight')
        print(f"\n📊 Plots saved: {self.results_dir}/kd_comparison_{self.timestamp}.png")

        return fig

    def run_full_experiment(self, teacher_model="yolo11l.pt", distillation_loss="cwd"):
        """Run complete KD validation experiment"""
        print("🔬 Starting COCO-1K Knowledge Distillation Validation")
        print(f"📁 Results will be saved to: {self.results_dir}")

        # Check if dataset exists
        if not Path(self.data_yaml).exists():
            print(f"❌ Dataset not found: {self.data_yaml}")
            print("Please run create_coco1k.py first to create the dataset")
            return

        # Run baseline
        baseline_metrics, baseline_results = self.run_baseline_training()

        # Run KD
        kd_metrics, kd_results = self.run_kd_training(teacher_model, distillation_loss)

        # Compare results
        comparison = self.compare_results(baseline_metrics, kd_metrics)

        # Create plots
        self.create_plots(comparison)

        print("\n" + "="*60)
        print("✅ COCO-1K KD Validation Complete!")
        print("="*60)
        print(f"📁 All results saved to: {self.results_dir}")

        return comparison

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Knowledge Distillation on COCO-1K')
    parser.add_argument('--data', default='coco1k/coco1k.yaml',
                       help='Path to COCO-1K YAML file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--teacher', default='yolo11l.pt',
                       help='Teacher model')
    parser.add_argument('--loss', default='cwd',
                       choices=['cwd', 'mgd'],
                       help='Distillation loss type')

    args = parser.parse_args()

    tester = COCO1KKDTester(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch
    )

    comparison = tester.run_full_experiment(
        teacher_model=args.teacher,
        distillation_loss=args.loss
    )

if __name__ == "__main__":
    main()