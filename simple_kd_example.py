#!/usr/bin/env python3
"""
Simple Knowledge Distillation Example - Exact format as provided
This script follows the exact example code format for KD training
"""

from ultralytics import YOLO

def main():
    print("🎓 Simple Knowledge Distillation Example")
    print("="*50)

    # Following the exact example format
    print("Loading models...")
    teacher_model = YOLO("yolo11l.pt")  # Large teacher
    student_model = YOLO("yolo11n.pt")  # Nano student

    print("Starting Knowledge Distillation training...")
    print("This follows the exact format from the example:")
    print("""
    from ultralytics import YOLO

    teacher_model = YOLO("<teacher-path>")
    student_model = YOLO("yolo11n.pt")

    student_model.train(
        data="<data-path>",
        teacher=teacher_model.model, # None if you don't wanna use knowledge distillation
        distillation_loss="cwd",
        epochs=100,
        batch=16,
        workers=0,
        exist_ok=True,
    )
    """)

    # Exact implementation
    student_model.train(
        data="coco1k/coco1k.yaml",  # Use COCO-1K dataset
        teacher=teacher_model.model,  # None if you don't wanna use knowledge distillation
        distillation_loss="cwd",
        epochs=50,  # Reduced for quick test
        batch=16,
        workers=0,
        exist_ok=True,
    )

    print("✅ Knowledge Distillation training completed!")

if __name__ == "__main__":
    main()