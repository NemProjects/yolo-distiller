#!/usr/bin/env python3
"""
Knowledge Distillation Test Script - Full COCO Dataset
Compare baseline YOLOv11n with knowledge distillation from YOLOv11m teacher
Using complete COCO 2017 dataset (118k training images)
"""

import json
import time
from pathlib import Path
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import torch


class COCOKnowledgeDistillationTester:
    def __init__(self, data_yaml="coco.yaml", epochs=25, batch_size=64):
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch_size = batch_size
        self.results_dir = Path("kd_experiments_coco")
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
        """Train YOLOv11n without knowledge distillation on full COCO"""
        print("\n" + "="*60)
        print("🚀 Starting BASELINE Training (YOLOv11n) on Full COCO")
        print("="*60)
        # Show current Seoul time
        seoul_tz = pytz.timezone('Asia/Seoul')
        current_time = datetime.now(seoul_tz).strftime("%Y-%m-%d %H:%M:%S KST")
        print(f"Start Time: {current_time}")
        print(f"Dataset: COCO 2017 (118,287 training images)")
        print(f"Epochs: {self.epochs}")
        print(f"Batch Size: {self.batch_size}")
        print("-"*60)

        model = YOLO("yolo11n.pt")

        start_time = time.time()
        results = model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            name=f"baseline_yolo11n_coco_{self.timestamp}",
            teacher=None,  # No teacher for baseline
            device=self.device,
            exist_ok=True,
            verbose=True,
            patience=10,  # Early stopping
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            plots=True,
            cache=False,  # Don't cache full COCO in memory
            workers=8,
            amp=True  # Automatic Mixed Precision for faster training
        )
        training_time = time.time() - start_time

        # Validate the model
        print("\n📊 Validating baseline model...")
        val_results = model.val(data=self.data_yaml)

        # Convert time to readable format
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        time_str = f"{hours}h {minutes}m {seconds}s"

        print(f"\n✅ Baseline training completed in {time_str}")

        return {
            "model": model,
            "training_results": results,
            "validation_results": val_results,
            "training_time": training_time,
            "training_time_str": time_str,
            "model_path": model.trainer.last,
            "best_model_path": model.trainer.best
        }

    def run_kd_training(self):
        """Train YOLOv11n with knowledge distillation from YOLOv11m on full COCO"""
        print("\n" + "="*60)
        print("🎓 Starting KNOWLEDGE DISTILLATION Training on Full COCO")
        print("Teacher: YOLOv11m → Student: YOLOv11n")
        print("="*60)
        # Show current Seoul time
        seoul_tz = pytz.timezone('Asia/Seoul')
        current_time = datetime.now(seoul_tz).strftime("%Y-%m-%d %H:%M:%S KST")
        print(f"Start Time: {current_time}")
        print(f"Dataset: COCO 2017 (118,287 training images)")
        print(f"Epochs: {self.epochs}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Distillation Loss: CWD (Channel-wise Distillation)")
        print("-"*60)

        # Load teacher and student models
        teacher = YOLO("yolo11m.pt")
        student = YOLO("yolo11n.pt")

        start_time = time.time()
        results = student.train(
            data=self.data_yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            name=f"kd_yolo11n_from_11m_coco_{self.timestamp}",
            teacher=teacher.model,  # Enable knowledge distillation
            distillation_loss="cwd",  # Channel-wise distillation
            device=self.device,
            exist_ok=True,
            verbose=True,
            patience=10,  # Early stopping
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            plots=True,
            cache=False,  # Don't cache full COCO in memory
            workers=8,
            amp=True  # Automatic Mixed Precision
        )
        training_time = time.time() - start_time

        # Validate the model
        print("\n📊 Validating KD model...")
        val_results = student.val(data=self.data_yaml)

        # Convert time to readable format
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        time_str = f"{hours}h {minutes}m {seconds}s"

        print(f"\n✅ KD training completed in {time_str}")

        return {
            "model": student,
            "training_results": results,
            "validation_results": val_results,
            "training_time": training_time,
            "training_time_str": time_str,
            "model_path": student.trainer.last,
            "best_model_path": student.trainer.best
        }

    def compare_results(self, baseline_results, kd_results):
        """Compare and analyze results between baseline and KD models"""
        print("\n" + "="*60)
        print("📈 RESULTS COMPARISON")
        print("="*60)

        comparison = {}

        # Extract key metrics
        baseline_metrics = baseline_results["validation_results"].results_dict
        kd_metrics = kd_results["validation_results"].results_dict

        # Calculate improvements
        metrics_to_compare = [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)"
        ]

        print("\n📊 Performance Metrics:")
        print("-" * 50)
        print(f"{'Metric':<15} | {'Baseline':<10} | {'KD':<10} | {'Change':<10}")
        print("-" * 50)

        for metric in metrics_to_compare:
            if metric in baseline_metrics and metric in kd_metrics:
                baseline_val = baseline_metrics[metric]
                kd_val = kd_metrics[metric]
                improvement = ((kd_val - baseline_val) / baseline_val) * 100 if baseline_val > 0 else 0

                comparison[metric] = {
                    "baseline": baseline_val,
                    "kd": kd_val,
                    "improvement_percent": improvement
                }

                metric_name = metric.split("/")[-1].replace("(B)", "")
                color = "\033[92m" if improvement > 0 else "\033[91m" if improvement < 0 else "\033[93m"
                reset = "\033[0m"
                print(f"{metric_name:<15} | {baseline_val:<10.4f} | {kd_val:<10.4f} | {color}{improvement:+.2f}%{reset}")

        # Training efficiency
        print("\n⏱️  Training Efficiency:")
        print("-" * 50)
        print(f"Baseline training time: {baseline_results['training_time_str']}")
        print(f"KD training time: {kd_results['training_time_str']}")
        time_diff = kd_results['training_time'] - baseline_results['training_time']
        time_diff_percent = (time_diff / baseline_results['training_time']) * 100
        print(f"Time difference: {time_diff:.0f}s ({time_diff_percent:+.1f}%)")

        comparison["training_time"] = {
            "baseline": baseline_results["training_time"],
            "kd": kd_results["training_time"],
            "baseline_str": baseline_results["training_time_str"],
            "kd_str": kd_results["training_time_str"]
        }

        # Model size comparison
        print("\n💾 Model Sizes:")
        print("-" * 50)
        baseline_size = Path(baseline_results["best_model_path"]).stat().st_size / (1024 * 1024)
        kd_size = Path(kd_results["best_model_path"]).stat().st_size / (1024 * 1024)
        print(f"Baseline model: {baseline_size:.2f} MB")
        print(f"KD model: {kd_size:.2f} MB")
        print(f"Size difference: {kd_size - baseline_size:.2f} MB")

        comparison["model_size_mb"] = {
            "baseline": baseline_size,
            "kd": kd_size
        }

        # Overall improvement
        avg_improvement = sum(v["improvement_percent"] for k, v in comparison.items()
                             if k.startswith("metrics/")) / len(metrics_to_compare)

        print("\n" + "="*60)
        print(f"📊 Average Performance Change: {avg_improvement:+.2f}%")

        if avg_improvement > 0:
            print("✅ Knowledge distillation IMPROVED performance!")
        elif avg_improvement < 0:
            print("⚠️  Knowledge distillation decreased performance.")
        else:
            print("➖ Knowledge distillation showed no change.")
        print("="*60)

        return comparison

    def visualize_results(self, comparison):
        """Create visualization of results comparison"""
        print("\n📈 Generating visualization...")

        # Prepare data for visualization
        metrics = ["precision", "recall", "mAP50", "mAP50-95"]
        baseline_values = []
        kd_values = []

        for metric in metrics:
            key = f"metrics/{metric}(B)"
            if key in comparison:
                baseline_values.append(comparison[key]["baseline"])
                kd_values.append(comparison[key]["kd"])

        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar plot comparison
        x = range(len(metrics))
        width = 0.35

        bars1 = ax1.bar([i - width/2 for i in x], baseline_values, width,
                        label='Baseline', color='#3498db', alpha=0.8)
        bars2 = ax1.bar([i + width/2 for i in x], kd_values, width,
                        label='Knowledge Distillation', color='#2ecc71', alpha=0.8)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        ax1.set_xlabel('Metrics', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title('Baseline vs Knowledge Distillation Performance\n(Full COCO Dataset)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, max(max(baseline_values), max(kd_values)) * 1.1])

        # Improvement percentage plot
        improvements = [comparison[f"metrics/{m}(B)"]["improvement_percent"]
                       for m in metrics if f"metrics/{m}(B)" in comparison]

        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        bars = ax2.bar(metrics[:len(improvements)], improvements, color=colors, alpha=0.8)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center',
                    va='bottom' if height > 0 else 'top', fontsize=10)

        ax2.set_xlabel('Metrics', fontsize=12)
        ax2.set_ylabel('Improvement (%)', fontsize=12)
        ax2.set_title('Performance Change with Knowledge Distillation\n(Full COCO Dataset)', fontsize=14, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)

        # Add dataset info
        fig.suptitle(f'COCO 2017 Full Dataset - 118,287 training images, 5,000 validation images',
                     fontsize=11, y=1.02)

        plt.tight_layout()

        # Save figure
        plot_path = self.results_dir / f"kd_comparison_coco_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {plot_path}")

        return plot_path

    def save_results(self, baseline_results, kd_results, comparison):
        """Save all results to JSON and CSV files"""
        results_data = {
            "experiment_info": {
                "timestamp": self.timestamp,
                "timestamp_kst": datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S KST"),
                "data": self.data_yaml,
                "dataset_size": "Full COCO 2017 (118,287 train, 5,000 val)",
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "teacher_model": "yolo11m",
                "student_model": "yolo11n",
                "distillation_loss": "cwd",
                "device": str(self.device)
            },
            "baseline": {
                "training_time": baseline_results["training_time"],
                "training_time_str": baseline_results["training_time_str"],
                "model_path": str(baseline_results["best_model_path"]),
                "metrics": baseline_results["validation_results"].results_dict
            },
            "knowledge_distillation": {
                "training_time": kd_results["training_time"],
                "training_time_str": kd_results["training_time_str"],
                "model_path": str(kd_results["best_model_path"]),
                "metrics": kd_results["validation_results"].results_dict
            },
            "comparison": comparison
        }

        # Save to JSON
        json_path = self.results_dir / f"kd_results_coco_{self.timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(results_data, f, indent=4, default=str)

        print(f"\n💾 Results saved to: {json_path}")

        # Create summary CSV
        summary_data = []
        for metric_key, values in comparison.items():
            if metric_key.startswith("metrics/"):
                metric_name = metric_key.split("/")[-1].replace("(B)", "")
                summary_data.append({
                    "Metric": metric_name,
                    "Baseline": f"{values['baseline']:.4f}",
                    "KD": f"{values['kd']:.4f}",
                    "Improvement (%)": f"{values['improvement_percent']:.2f}"
                })

        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_path = self.results_dir / f"kd_summary_coco_{self.timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"📊 Summary CSV saved to: {csv_path}")
            print("\n" + "="*60)
            print("SUMMARY TABLE:")
            print("="*60)
            print(df.to_string(index=False))
            print("="*60)

        return results_data

    def run_full_experiment(self):
        """Run the complete knowledge distillation experiment on full COCO"""
        print("\n" + "="*80)
        print("🚀 KNOWLEDGE DISTILLATION EXPERIMENT - FULL COCO DATASET 🚀")
        print("="*80)
        # Show current Seoul time
        seoul_tz = pytz.timezone('Asia/Seoul')
        current_time = datetime.now(seoul_tz).strftime("%Y-%m-%d %H:%M:%S KST")
        print(f"Experiment Start Time: {current_time}")
        print(f"Teacher Model: YOLOv11m")
        print(f"Student Model: YOLOv11n")
        print(f"Dataset: Full COCO 2017")
        print(f"  - Training: 118,287 images")
        print(f"  - Validation: 5,000 images")
        print(f"  - Total Size: ~20GB")
        print(f"Epochs: {self.epochs}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Device: {self.device}")
        print("="*80)

        # Note about dataset download
        print("\n⚠️  Note: COCO dataset will be automatically downloaded if not present.")
        print("   This may take some time (approximately 20GB).\n")

        # Run baseline training
        print("\n[1/5] Running baseline training...")
        baseline_results = self.run_baseline_training()

        # Run KD training
        print("\n[2/5] Running knowledge distillation training...")
        kd_results = self.run_kd_training()

        # Compare results
        print("\n[3/5] Comparing results...")
        comparison = self.compare_results(baseline_results, kd_results)

        # Visualize
        print("\n[4/5] Creating visualizations...")
        plot_path = self.visualize_results(comparison)

        # Save results
        print("\n[5/5] Saving results...")
        self.save_results(baseline_results, kd_results, comparison)

        print("\n" + "="*80)
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY! ✅")
        seoul_tz = pytz.timezone('Asia/Seoul')
        end_time = datetime.now(seoul_tz).strftime("%Y-%m-%d %H:%M:%S KST")
        print(f"Completion Time: {end_time}")
        print("="*80)

        return comparison


def main():
    """Main entry point for the script"""
    # Initialize tester with parameters
    tester = COCOKnowledgeDistillationTester(
        data_yaml="coco.yaml",  # Full COCO dataset
        epochs=25,
        batch_size=64
    )

    # Run the full experiment
    try:
        comparison = tester.run_full_experiment()

        # Print final message
        print("\n" + "📝 FINAL NOTES " + "📝")
        print("="*80)
        print("This experiment used the FULL COCO dataset with 118,287 training images.")
        print("The results should be more representative of real-world performance")
        print("compared to the COCO8 experiment (which used only 8 images).")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user.")
        print("Partial results may have been saved.")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        print("Please check the error and try again.")


if __name__ == "__main__":
    main()