#!/usr/bin/env python3
"""
Evaluate Teacher Model Performance
"""

from ultralytics import YOLO
import json

# Load and evaluate teacher model
print("Evaluating Teacher Model (YOLOv11m)...")
teacher = YOLO("yolo11m.pt")
teacher_results = teacher.val(data="coco8.yaml")

# Print results
print("\n" + "="*50)
print("TEACHER MODEL (YOLOv11m) RESULTS")
print("="*50)
print(f"mAP50: {teacher_results.results_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP50-95: {teacher_results.results_dict['metrics/mAP50-95(B)']:.4f}")
print(f"Precision: {teacher_results.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall: {teacher_results.results_dict['metrics/recall(B)']:.4f}")

# Save results
results = {
    "model": "YOLOv11m (Teacher)",
    "metrics": teacher_results.results_dict
}

with open("teacher_results.json", "w") as f:
    json.dump(results, f, indent=4, default=str)

print("\nResults saved to teacher_results.json")