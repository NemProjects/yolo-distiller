#!/usr/bin/env python3
"""
Training Analysis and Comparison Visualization
Baseline vs Attention KD performance comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_results(csv_path):
    """Load training results from CSV"""
    return pd.read_csv(csv_path)

def plot_metric_comparison(baseline_df, kd_df, save_dir):
    """Compare key metrics between baseline and KD"""
    metrics = [
        ('metrics/mAP50-95(B)', 'mAP50-95'),
        ('metrics/mAP50(B)', 'mAP50'),
        ('metrics/precision(B)', 'Precision'),
        ('metrics/recall(B)', 'Recall')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[idx]

        # Plot baseline (only up to epoch 70)
        baseline_data = baseline_df[baseline_df['epoch'] <= 70]
        ax.plot(baseline_data['epoch'], baseline_data[metric_key],
                label='Baseline (70 epochs)', linewidth=2, color='blue', alpha=0.7)

        # Plot KD (full 150 epochs)
        ax.plot(kd_df['epoch'], kd_df[metric_key],
                label='Attention KD (150 epochs)', linewidth=2, color='red', alpha=0.7)

        # Mark best epoch for KD
        best_idx = kd_df[metric_key].idxmax()
        best_epoch = kd_df.loc[best_idx, 'epoch']
        best_value = kd_df.loc[best_idx, metric_key]
        ax.scatter([best_epoch], [best_value], color='red', s=100, zorder=5,
                   label=f'Best: {best_value:.4f} (epoch {best_epoch})')

        # Mark baseline final value
        baseline_final = baseline_data[metric_key].iloc[-1]
        ax.axhline(y=baseline_final, color='blue', linestyle='--', alpha=0.5,
                   label=f'Baseline final: {baseline_final:.4f}')

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} Comparison', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "metrics_comparison_baseline_vs_kd.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved metrics comparison to {save_path}")
    plt.close()

def plot_loss_comparison(baseline_df, kd_df, save_dir):
    """Compare training losses"""
    losses = [
        ('train/box_loss', 'Box Loss'),
        ('train/cls_loss', 'Classification Loss'),
        ('train/dfl_loss', 'DFL Loss')
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (loss_key, loss_name) in enumerate(losses):
        ax = axes[idx]

        baseline_data = baseline_df[baseline_df['epoch'] <= 70]
        ax.plot(baseline_data['epoch'], baseline_data[loss_key],
                label='Baseline', linewidth=2, color='blue', alpha=0.7)

        ax.plot(kd_df['epoch'], kd_df[loss_key],
                label='Attention KD', linewidth=2, color='red', alpha=0.7)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(loss_name, fontsize=11)
        ax.set_title(f'{loss_name} Comparison', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "loss_comparison_baseline_vs_kd.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved loss comparison to {save_path}")
    plt.close()

def plot_improvement_over_baseline(baseline_df, kd_df, save_dir):
    """Plot improvement percentage over baseline"""
    metrics = {
        'metrics/mAP50-95(B)': 'mAP50-95',
        'metrics/mAP50(B)': 'mAP50',
        'metrics/precision(B)': 'Precision',
        'metrics/recall(B)': 'Recall'
    }

    # Get baseline final values (epoch 70)
    baseline_final = baseline_df[baseline_df['epoch'] == 70].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 6))

    for metric_key, metric_name in metrics.items():
        baseline_val = baseline_final[metric_key]
        improvement = ((kd_df[metric_key] - baseline_val) / baseline_val * 100).values
        ax.plot(kd_df['epoch'], improvement, label=metric_name, linewidth=2)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Improvement over Baseline (%)', fontsize=12)
    ax.set_title('Performance Improvement: Attention KD vs Baseline (70 epochs)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "improvement_over_baseline.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved improvement plot to {save_path}")
    plt.close()

def plot_learning_rate_schedule(kd_df, save_dir):
    """Visualize learning rate schedule"""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(kd_df['epoch'], kd_df['lr/pg0'], linewidth=2, color='navy')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule (Attention KD)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "learning_rate_schedule.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved LR schedule to {save_path}")
    plt.close()

def generate_summary_table(baseline_df, kd_df, save_dir):
    """Generate summary comparison table"""
    # Baseline final (epoch 70)
    baseline_final = baseline_df[baseline_df['epoch'] == 70].iloc[0]

    # KD best performance
    kd_best_idx = kd_df['metrics/mAP50-95(B)'].idxmax()
    kd_best = kd_df.loc[kd_best_idx]

    # Create summary
    summary = {
        'Metric': ['mAP50-95', 'mAP50', 'Precision', 'Recall', 'Best Epoch'],
        'Baseline (70ep)': [
            f"{baseline_final['metrics/mAP50-95(B)']*100:.2f}%",
            f"{baseline_final['metrics/mAP50(B)']*100:.2f}%",
            f"{baseline_final['metrics/precision(B)']*100:.2f}%",
            f"{baseline_final['metrics/recall(B)']*100:.2f}%",
            '70'
        ],
        'Attention KD (best)': [
            f"{kd_best['metrics/mAP50-95(B)']*100:.2f}%",
            f"{kd_best['metrics/mAP50(B)']*100:.2f}%",
            f"{kd_best['metrics/precision(B)']*100:.2f}%",
            f"{kd_best['metrics/recall(B)']*100:.2f}%",
            f"{int(kd_best['epoch'])}"
        ],
        'Improvement': [
            f"+{(kd_best['metrics/mAP50-95(B)'] - baseline_final['metrics/mAP50-95(B)'])*100:.2f}%p",
            f"+{(kd_best['metrics/mAP50(B)'] - baseline_final['metrics/mAP50(B)'])*100:.2f}%p",
            f"+{(kd_best['metrics/precision(B)'] - baseline_final['metrics/precision(B)'])*100:.2f}%p",
            f"+{(kd_best['metrics/recall(B)'] - baseline_final['metrics/recall(B)'])*100:.2f}%p",
            f"+{int(kd_best['epoch']) - 70}"
        ]
    }

    df_summary = pd.DataFrame(summary)

    # Save as CSV
    csv_path = save_dir / "performance_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"✅ Saved summary table to {csv_path}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df_summary.values,
                     colLabels=df_summary.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.25, 0.3, 0.25])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(df_summary.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style improvement column
    for i in range(1, len(df_summary) + 1):
        table[(i, 3)].set_facecolor('#90EE90')

    plt.title('Performance Summary: Baseline vs Attention KD', fontsize=14, fontweight='bold', pad=20)

    save_path = save_dir / "performance_summary_table.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved summary table visualization to {save_path}")
    plt.close()

    return df_summary

def main():
    """Generate all training analysis visualizations"""
    # Paths
    baseline_csv = "runs/detect/voc_baseline_yolo11s_optimized_20250928_095825/results.csv"
    kd_csv = "runs/detect/voc_kd_yolo11s_from_11m_optimized_20250930_102637_150ep_attention/results.csv"

    # Output directory
    output_dir = Path("training_analysis")
    output_dir.mkdir(exist_ok=True)

    print("📊 Loading training results...")
    baseline_df = load_results(baseline_csv)
    kd_df = load_results(kd_csv)

    print("📈 Generating visualizations...")

    # 1. Metrics comparison
    plot_metric_comparison(baseline_df, kd_df, output_dir)

    # 2. Loss comparison
    plot_loss_comparison(baseline_df, kd_df, output_dir)

    # 3. Improvement over baseline
    plot_improvement_over_baseline(baseline_df, kd_df, output_dir)

    # 4. Learning rate schedule
    plot_learning_rate_schedule(kd_df, output_dir)

    # 5. Summary table
    summary_df = generate_summary_table(baseline_df, kd_df, output_dir)

    print("\n" + "="*60)
    print("📊 PERFORMANCE SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)

    print(f"\n✅ All visualizations saved to {output_dir}/")
    print("\nGenerated files:")
    print("  - metrics_comparison_baseline_vs_kd.png")
    print("  - loss_comparison_baseline_vs_kd.png")
    print("  - improvement_over_baseline.png")
    print("  - learning_rate_schedule.png")
    print("  - performance_summary.csv")
    print("  - performance_summary_table.png")

if __name__ == "__main__":
    main()
