# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of Ultralytics YOLO with additional knowledge distillation capabilities. The project enables training smaller student models using larger teacher models for improved performance-to-size ratios.

## Core Architecture

### Key Components
- **ultralytics/**: Main package containing YOLO models and utilities
- **ultralytics/engine/**: Core training, validation, prediction, and export engines
- **ultralytics/models/**: Model implementations (YOLO, RTDETR, SAM, FastSAM, etc.)
- **ultralytics/nn/**: Neural network modules, layers, and architectures
- **ultralytics/data/**: Dataset handling, data loaders, and augmentation
- **ultralytics/utils/**: Utility functions for losses, metrics, and operations

### Knowledge Distillation Architecture
The main enhancement is integrated directly in the trainer (ultralytics/engine/trainer.py):
- **CWDLoss** (lines 58-94): Channel-Wise Distillation implementing arxiv.org/abs/2011.13256
- **MGDLoss** (lines 96+): Mask Generation Distillation
- **DistillationLoss** (line 215+): Main orchestrator that manages feature alignment and loss computation
- Teacher model runs in parallel during training with feature hooks for intermediate layer matching
- Distillation weight scaling and loss integration happens in main training loop

## Development Commands

### Installation
```bash
pip install -r requirements.txt
# For development with all tools:
pip install -e ".[dev]"
```

### Training with Knowledge Distillation
```python
from ultralytics import YOLO

teacher_model = YOLO("<teacher-path>")
student_model = YOLO("yolo11n.pt")

student_model.train(
    data="<data-path>",
    teacher=teacher_model.model,  # None to disable distillation
    distillation_loss="cwd",      # Available: "cwd", "mgd"
    epochs=100,
    batch=16,
    workers=0,
    exist_ok=True,
)
```

### Standard YOLO Commands
```bash
# Training
yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01

# Prediction
yolo predict model=yolo11n-seg.pt source='path/to/image.jpg' imgsz=320

# Validation
yolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640

# Export
yolo export model=yolo11n-cls.pt format=onnx imgsz=224,128
```

### Testing and Quality Assurance
Based on pyproject.toml configuration:
```bash
# Run tests with pytest (includes doctests and slow tests)
pytest --doctest-modules --durations=30 --color=yes
pytest --slow  # Include slow tests
pytest -k "test_specific_function"  # Run specific test

# Code formatting with yapf
yapf --in-place --recursive ultralytics/

# Linting with ruff
ruff check ultralytics/
ruff format ultralytics/

# Run knowledge distillation experiment
python test_kd_coco_full.py
```

### Documentation
```bash
# Build documentation (requires dev dependencies)
mkdocs serve
```

## Important Notes

### Model Architecture Detection
The CLI automatically detects model architecture from filename:
- Files containing "rtdetr": Uses RTDETR class
- Files containing "fastsam": Uses FastSAM class
- Files containing "sam_", "sam2_", or "sam2.1_": Uses SAM class
- Default: Uses YOLO class

### Configuration System
- Default settings: ultralytics/cfg/default.yaml contains all training hyperparameters
- Argument parsing: ultralytics/cfg/__init__.py handles CLI and programmatic arguments
- Settings system: SETTINGS object manages global configurations
- Custom configs: Override with cfg=path/to/config.yaml

### Knowledge Distillation Implementation Details
- Teacher model integration happens in BaseTrainer.__init__() (ultralytics/engine/trainer.py:383+)
- Feature hooks are registered in DistillationLoss.register_hook() method
- Loss computation occurs during training loop with distillation weight scaling
- Both CWD and MGD losses operate on intermediate feature maps, not just final outputs
- Feature alignment modules handle channel dimension matching between student/teacher

### Key Directories and Output
- Training outputs: runs/detect/train/, runs/segment/train/, etc. (task-dependent)
- Weights: saved to runs/{task}/train/weights/ with best.pt and last.pt
- Results: CSV metrics, confusion matrix plots, training curves
- Docker support: 9 different Dockerfiles in docker/ for various platforms (ARM64, Jetson, CPU-only, etc.)

## Development Tips

### Adding New Distillation Methods
1. Implement loss class following CWDLoss/MGDLoss pattern in ultralytics/engine/trainer.py
2. Add condition in DistillationLoss.__init__() for your distiller name
3. Ensure forward() method accepts y_s (student) and y_t (teacher) feature lists
4. Handle feature alignment if student/teacher have different channel dimensions

### Testing Knowledge Distillation
- Use test_kd_coco_full.py as reference for full COCO experiments
- Supports Seoul timezone (KST) for logging
- Compare baseline vs distillation training with metrics tracking
- pytest markers: use `--slow` flag for time-intensive tests

### Model Extensions
New model types should follow the pattern in ultralytics/models/ with separate modules for train, val, predict functionality, and register in ultralytics/__init__.py imports.