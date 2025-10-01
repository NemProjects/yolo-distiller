#!/usr/bin/env python3
"""
Hybrid Attention Knowledge Distillation Experiment
Combines Channel and Spatial Attention for improved knowledge transfer
"""

import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from ultralytics import YOLO


def run_hybrid_attention_kd_experiment():
    """Run Hybrid Attention KD training from YOLOv11m to YOLOv11s."""
    
    # Set timezone to KST
    kst = timezone(timedelta(hours=9))
    timestamp = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
    
    print("="*80)
    print(f"🚀 Starting Hybrid Attention KD Experiment - {timestamp}")
    print("="*80)
    print()
    print("Configuration:")
    print("  Teacher: YOLOv11m (pre-trained on VOC)")
    print("  Student: YOLOv11s")
    print("  Distillation: Hybrid Attention (Channel + Spatial)")
    print("  Dataset: VOC")
    print("  Epochs: 150")
    print("  Batch Size: 64")
    print()
    
    # Use pretrained YOLOv11m as teacher
    print("Loading teacher model: YOLOv11m (pretrained)")
    teacher_model = YOLO("yolo11m.pt")
    print("✓ Teacher model loaded successfully (using pretrained YOLOv11m)")
    
    # Initialize student model
    print("\nInitializing student model: YOLOv11s")
    student_model = YOLO("yolo11s.pt")
    
    # Training configuration
    training_args = {
        "data": "VOC.yaml",
        "epochs": 150,
        "batch": 64,
        "imgsz": 640,
        "device": 0,
        "workers": 12,
        "exist_ok": True,
        "patience": 100,
        "save": True,
        "pretrained": True,
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "name": f"voc_kd_yolo11s_from_11m_hybrid_{timestamp}",
        "seed": 42,
        "verbose": True,
    }
    
    # Add distillation parameters
    if teacher_model:
        training_args.update({
            "teacher": teacher_model.model,
            "distillation_loss": "hybrid",  # Use hybrid attention
        })
        print("\n✓ Knowledge Distillation enabled with Hybrid Attention")
        print("  - Combining Channel Attention (what) + Spatial Attention (where)")
        print("  - Using CBAM-style attention mechanism")
    else:
        print("\n⚠️ Training without Knowledge Distillation")
    
    # Start training
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Train the model
        results = student_model.train(**training_args)
        
        # Calculate training time
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        
        print("\n" + "="*80)
        print("✅ Training completed successfully!")
        print(f"Total training time: {hours}h {minutes}m")
        print("="*80)
        
        # Evaluate the model
        print("\nEvaluating the trained model...")
        metrics = student_model.val(data="VOC.yaml", batch=1, imgsz=640)
        
        print("\n📊 Final Performance Metrics:")
        print(f"  mAP50: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"  mAP50-95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")
        print(f"  Precision: {metrics.results_dict['metrics/precision(B)']:.4f}")
        print(f"  Recall: {metrics.results_dict['metrics/recall(B)']:.4f}")
        
        # Save results summary
        results_file = f"hybrid_kd_results_{timestamp}.txt"
        with open(results_file, 'w') as f:
            f.write(f"Hybrid Attention KD Experiment Results\n")
            f.write(f"="*50 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Teacher: YOLOv11m\n")
            f.write(f"Student: YOLOv11s\n")
            f.write(f"Distillation: Hybrid Attention (Channel + Spatial)\n")
            f.write(f"Training time: {hours}h {minutes}m\n")
            f.write(f"\nFinal Metrics:\n")
            f.write(f"  mAP50: {metrics.results_dict['metrics/mAP50(B)']:.4f}\n")
            f.write(f"  mAP50-95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}\n")
            f.write(f"  Precision: {metrics.results_dict['metrics/precision(B)']:.4f}\n")
            f.write(f"  Recall: {metrics.results_dict['metrics/recall(B)']:.4f}\n")
        
        print(f"\n📝 Results saved to: {results_file}")
        print(f"🎯 Model weights saved to: runs/detect/voc_kd_yolo11s_from_11m_hybrid_{timestamp}/weights/")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(run_hybrid_attention_kd_experiment())