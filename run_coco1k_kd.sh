#!/bin/bash
# COCO-1K Knowledge Distillation Complete Pipeline
# This script creates the dataset and runs KD validation

set -e  # Exit on any error

echo "🚀 COCO-1K Knowledge Distillation Pipeline"
echo "=========================================="

# Step 1: Create COCO-1K Dataset
echo "📦 Step 1: Creating COCO-1K dataset..."
if [ ! -d "coco1k" ]; then
    echo "Creating COCO-1K dataset from COCO 2017..."
    python create_coco1k.py \
        --coco-root /datasets/coco \
        --target-size 1500 \
        --output-dir coco1k

    if [ $? -eq 0 ]; then
        echo "✅ COCO-1K dataset created successfully!"
    else
        echo "❌ Failed to create COCO-1K dataset"
        echo "Please ensure COCO 2017 dataset is available at /datasets/coco"
        echo "Or modify the --coco-root parameter in this script"
        exit 1
    fi
else
    echo "✅ COCO-1K dataset already exists"
fi

# Step 2: Run KD Validation
echo ""
echo "🎓 Step 2: Running Knowledge Distillation validation..."
echo "This will:"
echo "  1. Train baseline YOLOv11n (no KD)"
echo "  2. Train YOLOv11n with KD from YOLOv11l teacher"
echo "  3. Compare results and generate plots"
echo ""

python test_coco1k_kd.py \
    --data coco1k/coco1k.yaml \
    --epochs 100 \
    --batch 16 \
    --teacher yolo11l.pt \
    --loss cwd

echo ""
echo "🎉 COCO-1K KD validation complete!"
echo "📊 Check the results in: coco1k_kd_experiments/"