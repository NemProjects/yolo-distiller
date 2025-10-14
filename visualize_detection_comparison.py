#!/usr/bin/env python3
"""
Detection Result Comparison: Baseline vs Attention KD
Compare detection outputs side-by-side
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from ultralytics import YOLO

class DetectionComparator:
    """Compare detection results between Baseline and Attention KD models"""

    def __init__(self, baseline_path, kd_path, device='cuda'):
        self.device = device
        self.baseline_model = YOLO(baseline_path)
        self.kd_model = YOLO(kd_path)

        # VOC class names
        self.class_names = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    def visualize_comparison(self, image_path, save_path=None, conf=0.25):
        """
        Compare detection results side-by-side

        Args:
            image_path: Path to input image
            save_path: Path to save visualization
            conf: Confidence threshold
        """
        # Run predictions
        baseline_results = self.baseline_model.predict(
            image_path,
            conf=conf,
            verbose=False
        )[0]

        kd_results = self.kd_model.predict(
            image_path,
            conf=conf,
            verbose=False
        )[0]

        # Load original image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Baseline results
        baseline_img = baseline_results.plot()
        baseline_img_rgb = cv2.cvtColor(baseline_img, cv2.COLOR_BGR2RGB)
        axes[1].imshow(baseline_img_rgb)
        baseline_count = len(baseline_results.boxes)
        axes[1].set_title(f'Baseline (YOLOv11s)\n{baseline_count} detections',
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # Attention KD results
        kd_img = kd_results.plot()
        kd_img_rgb = cv2.cvtColor(kd_img, cv2.COLOR_BGR2RGB)
        axes[2].imshow(kd_img_rgb)
        kd_count = len(kd_results.boxes)
        axes[2].set_title(f'Attention KD (YOLOv11s)\n{kd_count} detections',
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved detection comparison to {save_path}")

        plt.close()
        return fig

    def visualize_confidence_comparison(self, image_path, save_path=None):
        """
        Compare confidence scores for same detections
        """
        # Run predictions
        baseline_results = self.baseline_model.predict(
            image_path,
            conf=0.1,  # Lower threshold to catch more detections
            verbose=False
        )[0]

        kd_results = self.kd_model.predict(
            image_path,
            conf=0.1,
            verbose=False
        )[0]

        # Load original image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract boxes and confidences
        baseline_boxes = baseline_results.boxes.xyxy.cpu().numpy() if len(baseline_results.boxes) > 0 else np.array([])
        baseline_confs = baseline_results.boxes.conf.cpu().numpy() if len(baseline_results.boxes) > 0 else np.array([])
        baseline_classes = baseline_results.boxes.cls.cpu().numpy().astype(int) if len(baseline_results.boxes) > 0 else np.array([])

        kd_boxes = kd_results.boxes.xyxy.cpu().numpy() if len(kd_results.boxes) > 0 else np.array([])
        kd_confs = kd_results.boxes.conf.cpu().numpy() if len(kd_results.boxes) > 0 else np.array([])
        kd_classes = kd_results.boxes.cls.cpu().numpy().astype(int) if len(kd_results.boxes) > 0 else np.array([])

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Baseline detections with confidence
        axes[0].imshow(img_rgb)
        for box, conf, cls in zip(baseline_boxes, baseline_confs, baseline_classes):
            x1, y1, x2, y2 = box
            axes[0].add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False, edgecolor='blue', linewidth=2
            ))
            axes[0].text(x1, y1-10, f'{self.class_names[cls]}: {conf:.2f}',
                        color='white', fontsize=10,
                        bbox=dict(facecolor='blue', alpha=0.7))
        axes[0].set_title(f'Baseline Confidence\n(Avg: {baseline_confs.mean():.3f})',
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Attention KD detections with confidence
        axes[1].imshow(img_rgb)
        for box, conf, cls in zip(kd_boxes, kd_confs, kd_classes):
            x1, y1, x2, y2 = box
            axes[1].add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False, edgecolor='red', linewidth=2
            ))
            axes[1].text(x1, y1-10, f'{self.class_names[cls]}: {conf:.2f}',
                        color='white', fontsize=10,
                        bbox=dict(facecolor='red', alpha=0.7))
        axes[1].set_title(f'Attention KD Confidence\n(Avg: {kd_confs.mean():.3f})',
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved confidence comparison to {save_path}")

        plt.close()
        return fig


def main():
    """Example usage"""
    # Model paths
    baseline_model = "voc_baseline_yolo11s_optimized_20250928_095825/weights/best.pt"
    kd_model = "runs/detect/voc_kd_yolo11s_from_11m_optimized_20250930_102637_150ep_attention/weights/best.pt"

    # Sample image
    image_path = "sample_voc_image.jpg"

    print(f"📸 Loading models...")
    print(f"   Baseline: {baseline_model}")
    print(f"   Attention KD: {kd_model}")
    print(f"   Sample image: {image_path}")

    # Create comparator
    comparator = DetectionComparator(baseline_model, kd_model)

    # Generate visualizations
    output_dir = Path("detection_comparison")
    output_dir.mkdir(exist_ok=True)

    print("\n🎨 Generating detection comparisons...")

    # 1. Side-by-side detection comparison
    print("  1. Detection result comparison...")
    comparator.visualize_comparison(
        image_path,
        save_path=output_dir / "detection_comparison.png",
        conf=0.25
    )

    # 2. Confidence score comparison
    print("  2. Confidence score comparison...")
    comparator.visualize_confidence_comparison(
        image_path,
        save_path=output_dir / "confidence_comparison.png"
    )

    print("\n✅ All detection comparisons completed!")
    print(f"📁 Saved to {output_dir}/")
    print("\nGenerated files:")
    print("  - detection_comparison.png")
    print("  - confidence_comparison.png")


if __name__ == "__main__":
    main()
