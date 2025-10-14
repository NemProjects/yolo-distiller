#!/usr/bin/env python3
"""
Attention Map Visualization for Knowledge Distillation
Teacher vs Student attention map comparison
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from ultralytics import YOLO
import yaml

class AttentionVisualizer:
    """Visualize attention maps from Teacher and Student models"""

    def __init__(self, teacher_path, student_path, device='cuda'):
        self.device = device
        self.teacher = YOLO(teacher_path).model.to(device).eval()
        self.student = YOLO(student_path).model.to(device).eval()

        # Layers to visualize (same as training)
        self.layers = ["6", "8", "13", "16", "19", "22"]
        self.teacher_features = []
        self.student_features = []

    def attention_map(self, fm, p=2):
        """Calculate attention map (same as AttentionLoss)"""
        am = torch.pow(torch.abs(fm), p)
        am = torch.sum(am, dim=1, keepdim=False)  # (N, H, W)
        norm = torch.norm(am, p=2, dim=(1, 2), keepdim=True)
        am = torch.div(am, norm + 1e-8)
        return am

    def register_hooks(self):
        """Register forward hooks to capture features"""
        self.teacher_features.clear()
        self.student_features.clear()
        self.hooks = []

        def make_hook(storage):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    storage.append(output.detach().clone())
            return hook

        # Find matching layers
        for name, module in self.teacher.named_modules():
            parts = name.split(".")
            if len(parts) >= 3 and parts[0] == "model" and parts[1] in self.layers:
                if "cv2" in parts[2] and hasattr(module, 'conv'):
                    self.hooks.append(module.register_forward_hook(make_hook(self.teacher_features)))

        for name, module in self.student.named_modules():
            parts = name.split(".")
            if len(parts) >= 3 and parts[0] == "model" and parts[1] in self.layers:
                if "cv2" in parts[2] and hasattr(module, 'conv'):
                    self.hooks.append(module.register_forward_hook(make_hook(self.student_features)))

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    @torch.no_grad()
    def extract_attention_maps(self, image_path):
        """Extract attention maps from both models"""
        # Load and preprocess image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 640x640 (YOLO input size)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Register hooks
        self.register_hooks()

        # Forward pass
        _ = self.teacher(img_tensor)
        _ = self.student(img_tensor)

        # Calculate attention maps
        teacher_attns = []
        student_attns = []

        n_layers = len(self.teacher_features)
        for t_feat, s_feat in zip(self.teacher_features[:n_layers], self.student_features[:n_layers]):
            teacher_attns.append(self.attention_map(t_feat))
            student_attns.append(self.attention_map(s_feat))

        # Remove hooks
        self.remove_hooks()

        return img_rgb, teacher_attns, student_attns

    def visualize_comparison(self, image_path, save_path=None, layer_idx=None):
        """Visualize Teacher vs Student attention comparison"""
        img_rgb, teacher_attns, student_attns = self.extract_attention_maps(image_path)

        if layer_idx is None:
            # Visualize all layers
            n_layers = len(teacher_attns)
            fig, axes = plt.subplots(3, n_layers, figsize=(4*n_layers, 10))

            for i in range(n_layers):
                # Original image
                axes[0, i].imshow(img_rgb)
                axes[0, i].set_title(f'Layer {self.layers[i]}' if i == 0 else f'Layer {self.layers[i]}')
                axes[0, i].axis('off')

                # Teacher attention
                t_attn = teacher_attns[i][0].cpu().numpy()
                t_attn_resized = cv2.resize(t_attn, (640, 640))
                im1 = axes[1, i].imshow(t_attn_resized, cmap='jet', alpha=0.6)
                axes[1, i].imshow(img_rgb, alpha=0.4)
                axes[1, i].set_title('Teacher' if i == 0 else '')
                axes[1, i].axis('off')

                # Student attention
                s_attn = student_attns[i][0].cpu().numpy()
                s_attn_resized = cv2.resize(s_attn, (640, 640))
                im2 = axes[2, i].imshow(s_attn_resized, cmap='jet', alpha=0.6)
                axes[2, i].imshow(img_rgb, alpha=0.4)
                axes[2, i].set_title('Student' if i == 0 else '')
                axes[2, i].axis('off')

            # Add colorbars
            fig.colorbar(im1, ax=axes[1, :].ravel().tolist(), shrink=0.6, label='Attention Weight')
            fig.colorbar(im2, ax=axes[2, :].ravel().tolist(), shrink=0.6, label='Attention Weight')

        else:
            # Visualize specific layer
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original
            axes[0].imshow(img_rgb)
            axes[0].set_title(f'Original Image (Layer {self.layers[layer_idx]})')
            axes[0].axis('off')

            # Teacher
            t_attn = teacher_attns[layer_idx][0].cpu().numpy()
            t_attn_resized = cv2.resize(t_attn, (640, 640))
            axes[1].imshow(img_rgb, alpha=0.4)
            im1 = axes[1].imshow(t_attn_resized, cmap='jet', alpha=0.6)
            axes[1].set_title('Teacher Attention')
            axes[1].axis('off')

            # Student
            s_attn = student_attns[layer_idx][0].cpu().numpy()
            s_attn_resized = cv2.resize(s_attn, (640, 640))
            axes[2].imshow(img_rgb, alpha=0.4)
            im2 = axes[2].imshow(s_attn_resized, cmap='jet', alpha=0.6)
            axes[2].set_title('Student Attention')
            axes[2].axis('off')

            # Colorbar
            fig.colorbar(im1, ax=axes[1], shrink=0.8, label='Attention Weight')
            fig.colorbar(im2, ax=axes[2], shrink=0.8, label='Attention Weight')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved visualization to {save_path}")

        plt.close()
        return fig

    def visualize_difference(self, image_path, save_path=None):
        """Visualize attention difference (Teacher - Student)"""
        img_rgb, teacher_attns, student_attns = self.extract_attention_maps(image_path)

        n_layers = len(teacher_attns)
        fig, axes = plt.subplots(2, n_layers, figsize=(4*n_layers, 7))

        for i in range(n_layers):
            # Attention difference
            t_attn = teacher_attns[i][0].cpu().numpy()
            s_attn = student_attns[i][0].cpu().numpy()

            # Resize to same size
            t_attn_resized = cv2.resize(t_attn, (640, 640))
            s_attn_resized = cv2.resize(s_attn, (640, 640))

            diff = t_attn_resized - s_attn_resized

            # Original + difference
            axes[0, i].imshow(img_rgb, alpha=0.4)
            im1 = axes[0, i].imshow(diff, cmap='RdBu_r', alpha=0.6, vmin=-0.02, vmax=0.02)
            axes[0, i].set_title(f'Layer {self.layers[i]} Diff')
            axes[0, i].axis('off')

            # L2 distance map
            l2_dist = np.abs(diff)
            axes[1, i].imshow(img_rgb, alpha=0.4)
            im2 = axes[1, i].imshow(l2_dist, cmap='hot', alpha=0.6)
            axes[1, i].set_title(f'Layer {self.layers[i]} |Diff|')
            axes[1, i].axis('off')

        # Colorbars
        fig.colorbar(im1, ax=axes[0, :].ravel().tolist(), shrink=0.6, label='T - S')
        fig.colorbar(im2, ax=axes[1, :].ravel().tolist(), shrink=0.6, label='|T - S|')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved difference map to {save_path}")

        plt.close()
        return fig


def main():
    """Example usage"""
    # Paths
    teacher_model = "yolo11m.pt"
    student_model = "runs/detect/voc_kd_yolo11s_from_11m_optimized_20250930_102637_150ep_attention/weights/best.pt"

    # Sample image from validation set
    image_path = "sample_voc_image.jpg"

    print(f"📸 Loading models...")
    print(f"   Teacher: {teacher_model}")
    print(f"   Student: {student_model}")
    print(f"   Sample image: {image_path}")

    # Create visualizer
    viz = AttentionVisualizer(teacher_model, student_model)

    # Generate visualizations
    output_dir = Path("attention_visualizations")
    output_dir.mkdir(exist_ok=True)

    print("\n🎨 Generating attention visualizations...")

    # 1. Full comparison (all layers)
    print("  1. Full layer comparison...")
    viz.visualize_comparison(
        image_path,
        save_path=output_dir / "attention_comparison_all_layers.png"
    )

    # 2. Single layer comparison (e.g., layer 13 - middle layer)
    print("  2. Single layer (layer 13) comparison...")
    viz.visualize_comparison(
        image_path,
        save_path=output_dir / "attention_comparison_layer13.png",
        layer_idx=2  # Layer 13 (index 2 in layers list)
    )

    # 3. Attention difference heatmap
    print("  3. Attention difference heatmap...")
    viz.visualize_difference(
        image_path,
        save_path=output_dir / "attention_difference.png"
    )

    print("\n✅ All attention visualizations completed!")
    print(f"📁 Saved to {output_dir}/")
    print("\nGenerated files:")
    print("  - attention_comparison_all_layers.png")
    print("  - attention_comparison_layer13.png")
    print("  - attention_difference.png")


if __name__ == "__main__":
    main()
