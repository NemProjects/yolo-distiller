#!/usr/bin/env python3
"""
Multi-scale Feature Analysis for Attention KD
Analyze feature map sizes across different layers
"""

import torch
from ultralytics import YOLO

def analyze_multiscale_features():
    """Analyze multi-scale features in YOLOv11 architecture"""

    # Load models
    teacher = YOLO('yolo11m.pt')
    student = YOLO('yolo11s.pt')

    # Move to GPU
    teacher.model.cuda()
    student.model.cuda()

    # Create dummy input (640x640)
    dummy_input = torch.randn(1, 3, 640, 640).cuda()

    # Define layers to analyze
    layers = ['6', '8', '13', '16', '19', '22']

    # Storage for features
    teacher_features = []
    student_features = []

    # Register hooks for teacher
    def make_teacher_hook(storage):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                storage.append(output.detach().clone())
        return hook

    # Register hooks for student
    def make_student_hook(storage):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                storage.append(output.detach().clone())
        return hook

    # Find and register hooks
    teacher_hooks = []
    student_hooks = []

    for name, module in teacher.model.named_modules():
        parts = name.split('.')
        if len(parts) >= 3 and parts[0] == 'model' and parts[1] in layers:
            if 'cv2' in parts[2] and hasattr(module, 'conv'):
                teacher_hooks.append(module.register_forward_hook(make_teacher_hook(teacher_features)))

    for name, module in student.model.named_modules():
        parts = name.split('.')
        if len(parts) >= 3 and parts[0] == 'model' and parts[1] in layers:
            if 'cv2' in parts[2] and hasattr(module, 'conv'):
                student_hooks.append(module.register_forward_hook(make_student_hook(student_features)))

    # Forward pass
    with torch.no_grad():
        _ = teacher.model(dummy_input)
        _ = student.model(dummy_input)

    # Remove hooks
    for hook in teacher_hooks:
        hook.remove()
    for hook in student_hooks:
        hook.remove()

    # Print analysis
    print("=" * 80)
    print("Multi-scale Feature Analysis for Attention Transfer KD")
    print("=" * 80)
    print(f"\nInput Image Size: 640 x 640\n")
    print("-" * 80)
    print(f"{'Layer':<10} {'Teacher Shape':<25} {'Student Shape':<25} {'Scale':<15}")
    print("-" * 80)

    for i, layer_id in enumerate(layers):
        if i < len(teacher_features) and i < len(student_features):
            t_shape = teacher_features[i].shape
            s_shape = student_features[i].shape

            # Calculate scale factor (relative to input 640x640)
            scale = t_shape[2]  # Height
            scale_factor = 640 / scale

            print(f"Layer {layer_id:<5} {str(tuple(t_shape)):<25} {str(tuple(s_shape)):<25} 1/{int(scale_factor):<15}")

    print("-" * 80)

    # Detailed analysis
    print("\n" + "=" * 80)
    print("Multi-scale Feature Distillation Strategy")
    print("=" * 80)

    print("\n1. Feature Pyramid Structure:")
    for i, layer_id in enumerate(layers):
        if i < len(teacher_features):
            t_shape = teacher_features[i].shape
            n, c, h, w = t_shape
            scale = 640 / h

            if h == 80:
                stage = "High Resolution (P3)"
            elif h == 40:
                stage = "Medium Resolution (P4)"
            elif h == 20:
                stage = "Low Resolution (P5)"
            else:
                stage = "Unknown"

            print(f"   Layer {layer_id}: {h}x{w} ({stage})")
            print(f"      - Channels: {c}")
            print(f"      - Receptive Field: 1/{int(scale)} of input")
            print(f"      - Feature Points: {h * w:,}")

    print("\n2. Attention Transfer Mechanism:")
    print("   - Each scale transfers spatial attention patterns")
    print("   - Attention Map: A(F) = ||∑|F|^2||_2 normalized")
    print("   - Loss: L2 distance between teacher and student attention maps")
    print("   - Aggregation: Sum of losses across all scales")

    print("\n3. Multi-scale Benefits:")
    print("   - High Resolution (80x80): Fine-grained details for small objects")
    print("   - Medium Resolution (40x40): Intermediate features for medium objects")
    print("   - Low Resolution (20x20): Semantic patterns for large objects")

    print("\n4. Channel Alignment:")
    for i, layer_id in enumerate(layers):
        if i < len(teacher_features) and i < len(student_features):
            t_c = teacher_features[i].shape[1]
            s_c = student_features[i].shape[1]

            if t_c != s_c:
                print(f"   Layer {layer_id}: Student {s_c} → Teacher {t_c} (Conv1x1 alignment)")
            else:
                print(f"   Layer {layer_id}: Student {s_c} = Teacher {t_c} (No alignment needed)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_multiscale_features()
