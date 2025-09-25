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

### Knowledge Distillation Extension
The main enhancement is knowledge distillation support:
- Teacher model integration in training pipeline (ultralytics/engine/trainer.py)
- Distillation losses: Channel-Wise Distillation (CWD) and others
- Student-teacher architecture pairing for model compression

## Development Commands

### Installation
```bash
pip install -r requirements.txt
```

### Training with Knowledge Distillation
```python
from ultralytics import YOLO

teacher_model = YOLO("<teacher-path>")
student_model = YOLO("yolo11n.pt")

student_model.train(
    data="<data-path>",
    teacher=teacher_model.model,  # None to disable distillation
    distillation_loss="cwd",      # Available: "cwd", others
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
# Run tests with pytest
pytest --doctest-modules --durations=30 --color=yes

# Code formatting with yapf
yapf --in-place --recursive ultralytics/

# Linting with ruff
ruff check ultralytics/
ruff format ultralytics/

# Type checking (if mypy is available)
# No explicit mypy configuration found in pyproject.toml
```

### Documentation
```bash
# Build documentation (if mkdocs is available)
mkdocs serve
```

## Important Notes

### Model Architecture Detection
The CLI automatically detects model architecture from filename:
- Files containing "rtdetr": Uses RTDETR class
- Files containing "fastsam": Uses FastSAM class
- Files containing "sam_", "sam2_", or "sam2.1_": Uses SAM class
- Default: Uses YOLO class

### Configuration
- Main config: Uses ultralytics/cfg/__init__.py for argument parsing
- Default settings: Managed through SETTINGS system
- Custom configs: Can override with cfg=path/to/config.yaml

### Key Directories
- weights saved to: runs/detect/train/weights/ (or similar based on task)
- Results: CSV files and plots saved alongside weights
- Docker support: Multiple Dockerfiles available in docker/ directory

## Development Tips

### Adding New Distillation Methods
Look at ultralytics/models/utils/loss.py and the trainer implementation for existing distillation loss patterns.

### Model Extensions
New model types should follow the pattern in ultralytics/models/ with separate modules for train, val, predict functionality.

### Testing Specific Components
Use pytest markers for slow tests: `pytest --slow` to include slow tests, or omit flag to skip them.