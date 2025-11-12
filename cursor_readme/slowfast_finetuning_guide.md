# SlowFast Model Finetuning Guide for MMAction2

## Table of Contents
1. [Overview](#overview)
2. [Understanding MMAction2 Architecture](#understanding-mmaction2-architecture)
3. [SlowFast Model Architecture](#slowfast-model-architecture)
4. [Data Preparation](#data-preparation)
5. [Configuration Setup](#configuration-setup)
6. [Training Process](#training-process)
7. [Testing and Evaluation](#testing-and-evaluation)
8. [Advanced Tips](#advanced-tips)

## Overview

MMAction2 is OpenMMLab's comprehensive video understanding toolbox built on PyTorch. It provides state-of-the-art implementations of various video recognition, detection, and localization algorithms.

**SlowFast** is a dual-pathway 3D CNN architecture designed for video recognition:
- **Slow Pathway**: Operates at low frame rate to capture spatial semantics
- **Fast Pathway**: Operates at high frame rate to capture motion at fine temporal resolution
- The Fast pathway is lightweight (reduced channel capacity) but effective for temporal modeling

This guide will walk you through finetuning a pretrained SlowFast model on your custom dataset.

## Understanding MMAction2 Architecture

### Project Structure

```
mmaction2/
├── configs/              # Configuration files organized by task
│   ├── _base_/          # Base configs (models, schedules, runtime)
│   ├── recognition/     # Action recognition configs
│   │   ├── slowfast/    # SlowFast specific configs
│   │   └── ...
│   └── ...
├── mmaction/            # Core library code
│   ├── models/          # Model implementations
│   ├── datasets/        # Dataset implementations
│   ├── engine/          # Training/testing engines
│   └── ...
├── tools/               # Training and testing scripts
│   ├── train.py         # Main training script
│   ├── test.py          # Main testing script
│   └── data/            # Data preparation scripts
└── data/                # Dataset directory (you create this)
```

### Configuration System

MMAction2 uses a hierarchical configuration system with **inheritance**:
- Base configs define common settings
- Task-specific configs inherit and override base settings
- Your custom config can inherit from existing configs

## SlowFast Model Architecture

### Model Components

The SlowFast model consists of:

1. **Backbone** (`ResNet3dSlowFast`):
   - Slow pathway: ResNet3D with lower temporal resolution
   - Fast pathway: ResNet3D with higher temporal resolution and fewer channels
   - Lateral connections: Transfer information from Fast to Slow pathway

2. **Classification Head** (`SlowFastHead`):
   - Combines features from both pathways
   - Global average pooling
   - Dropout for regularization
   - Fully connected layer for classification

3. **Data Preprocessor**:
   - Normalization with ImageNet statistics
   - Format conversion (NCTHW)

### Key Configuration Parameters

From `configs/_base_/models/slowfast_r50.py`:

```python
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        resample_rate=8,      # tau: temporal downsampling
        speed_ratio=8,        # alpha: Fast/Slow frame rate ratio
        channel_ratio=8,      # beta_inv: Slow/Fast channel ratio
        slow_pathway=dict(
            type='resnet3d',
            depth=50,          # ResNet-50
            lateral=True,      # Enable lateral connections
            ...
        ),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            base_channels=8,   # Reduced channels for Fast pathway
            ...
        )
    ),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,      # 2048 (slow) + 256 (fast)
        num_classes=400,       # IMPORTANT: Change for your dataset
        dropout_ratio=0.5,
    ),
    ...
)
```

## Data Preparation

### Step 1: Organize Your Dataset

MMAction2 supports two data formats:

#### Option A: Video Format (Recommended for large datasets)

```
data/
└── your_dataset/
    ├── videos_train/
    │   ├── class1/
    │   │   ├── video1.mp4
    │   │   ├── video2.mp4
    │   │   └── ...
    │   ├── class2/
    │   │   ├── video1.mp4
    │   │   └── ...
    │   └── ...
    ├── videos_val/
    │   ├── class1/
    │   └── ...
    └── annotations/
        ├── train_list.txt
        └── val_list.txt
```

#### Option B: RawFrame Format (Faster but requires more storage)

```
data/
└── your_dataset/
    ├── rawframes_train/
    │   ├── class1_video1/
    │   │   ├── img_00001.jpg
    │   │   ├── img_00002.jpg
    │   │   └── ...
    │   └── ...
    └── annotations/
        ├── train_list.txt
        └── val_list.txt
```

### Step 2: Create Annotation Files

#### For Video Format (`VideoDataset`)

Annotation file format: `video_path label`

**Example: `train_list.txt`**
```
class1/video1.mp4 0
class1/video2.mp4 0
class2/video1.mp4 1
class2/video2.mp4 1
class3/video1.mp4 2
```

#### For RawFrame Format (`RawFrameDataset`)

Annotation file format: `frame_directory total_frames label`

**Example: `train_list.txt`**
```
class1_video1 150 0
class1_video2 200 0
class2_video1 180 1
class2_video2 220 1
class3_video1 175 2
```

### Step 3: Create Class Index Mapping (Optional but Recommended)

Create a `class_index.txt` file:
```
0 class1
1 class2
2 class3
...
```

### Data Preparation Script Example

If you have videos in a flat directory, here's a Python script to organize them:

```python
import os
import shutil
from pathlib import Path

def organize_dataset(
    video_dir: str,
    output_dir: str,
    label_mapping: dict,  # {video_name: label}
    split: str = 'train'
):
    """
    Organize videos into class folders and create annotation file.
    
    Args:
        video_dir: Directory containing all videos
        output_dir: Output directory for organized dataset
        label_mapping: Dictionary mapping video names to labels
        split: 'train' or 'val'
    """
    output_path = Path(output_dir) / f'videos_{split}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    ann_file = Path(output_dir) / 'annotations' / f'{split}_list.txt'
    ann_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(ann_file, 'w') as f:
        for video_name, label in label_mapping.items():
            # Create class directory
            class_dir = output_path / f'class{label}'
            class_dir.mkdir(exist_ok=True)
            
            # Copy video
            src = Path(video_dir) / video_name
            dst = class_dir / video_name
            if src.exists():
                shutil.copy(src, dst)
                
                # Write annotation
                relative_path = f'class{label}/{video_name}'
                f.write(f'{relative_path} {label}\n')
    
    print(f'Dataset organized at: {output_path}')
    print(f'Annotation file created at: {ann_file}')
```

## Configuration Setup

### Method 1: Inherit from Existing Config (Recommended)

Create a new config file: `configs/recognition/slowfast/slowfast_r50_custom.py`

```python
_base_ = [
    './slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py'
]

# =============================================================================
# 1. MODIFY MODEL: Change number of classes
# =============================================================================
model = dict(
    cls_head=dict(
        num_classes=10,  # Change to YOUR number of classes
    )
)

# =============================================================================
# 2. MODIFY DATASET: Point to your data
# =============================================================================
dataset_type = 'VideoDataset'
data_root = 'data/your_dataset/videos_train'
data_root_val = 'data/your_dataset/videos_val'
ann_file_train = 'data/your_dataset/annotations/train_list.txt'
ann_file_val = 'data/your_dataset/annotations/val_list.txt'

train_dataloader = dict(
    batch_size=8,  # Adjust based on GPU memory
    dataset=dict(
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root)
    )
)

val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val)
    )
)

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val)
    )
)

# =============================================================================
# 3. MODIFY TRAINING SCHEDULE: Finetune with lower LR and fewer epochs
# =============================================================================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,      # Reduce from 256 for finetuning
    val_begin=1,
    val_interval=2      # Validate every 2 epochs
)

# Adjust learning rate (typically 10x smaller for finetuning)
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.01,           # Reduced from 0.1
        momentum=0.9,
        weight_decay=1e-4
    ),
    clip_grad=dict(max_norm=40, norm_type=2)
)

# Update learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,              # Warmup for 5 epochs
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=50,           # Match max_epochs
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=50
    )
]

# =============================================================================
# 4. LOAD PRETRAINED WEIGHTS
# =============================================================================
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth'

# =============================================================================
# 5. OPTIONAL: Adjust logging and checkpointing
# =============================================================================
default_hooks = dict(
    checkpoint=dict(
        interval=5,          # Save every 5 epochs
        max_keep_ckpts=3     # Keep only 3 checkpoints
    ),
    logger=dict(interval=50)  # Log every 50 iterations
)

# Enable automatic learning rate scaling (optional)
auto_scale_lr = dict(
    enable=True,
    base_batch_size=64  # 8 GPUs x 8 batch_size
)
```

### Method 2: Full Custom Config (More Control)

If you need full control, create a complete config without inheritance:

```python
# configs/recognition/slowfast/slowfast_r50_custom_full.py

_base_ = [
    '../../_base_/models/slowfast_r50.py',
    '../../_base_/default_runtime.py'
]

# Model settings
model = dict(
    cls_head=dict(
        num_classes=10,  # YOUR number of classes
    )
)

# Dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/your_dataset/videos_train'
data_root_val = 'data/your_dataset/videos_val'
ann_file_train = 'data/your_dataset/annotations/train_list.txt'
ann_file_val = 'data/your_dataset/annotations/val_list.txt'

# Data pipeline
file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True
    ),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True
    ),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# Dataloader settings
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True
    )
)

# Evaluation settings
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

# Training settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_begin=1,
    val_interval=2
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer settings
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2)
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=50,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=50
    )
]

# Runtime settings
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3),
    logger=dict(interval=50)
)

# Load pretrained weights
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth'
```

## Training Process

### Step 1: Verify Installation

```bash
# Test MMAction2 installation
python -c "import mmaction; print(mmaction.__version__)"

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Verify Dataset and Config

Before full training, verify your setup:

```bash
# Test config file syntax
python tools/train.py configs/recognition/slowfast/slowfast_r50_custom.py --cfg-options dry_run=True

# Visualize data loading (optional)
# This helps debug data pipeline issues
```

### Step 3: Single GPU Training

For development and small datasets:

```bash
python tools/train.py \
    configs/recognition/slowfast/slowfast_r50_custom.py \
    --work-dir work_dirs/slowfast_custom \
    --seed 0 \
    --deterministic
```

**Arguments:**
- `--work-dir`: Directory to save logs and checkpoints
- `--seed`: Random seed for reproducibility
- `--deterministic`: Enable deterministic mode (slightly slower but reproducible)
- `--auto-scale-lr`: Automatically scale learning rate based on batch size

### Step 4: Multi-GPU Training (Recommended)

For larger datasets and faster training:

```bash
# Using 4 GPUs
bash tools/dist_train.sh \
    configs/recognition/slowfast/slowfast_r50_custom.py \
    4 \
    --work-dir work_dirs/slowfast_custom \
    --seed 0 \
    --auto-scale-lr
```

**With specific GPU devices:**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh \
    configs/recognition/slowfast/slowfast_r50_custom.py \
    4 \
    --work-dir work_dirs/slowfast_custom
```

### Step 5: Resume Training

If training is interrupted:

```bash
# Auto-resume from latest checkpoint
python tools/train.py \
    configs/recognition/slowfast/slowfast_r50_custom.py \
    --work-dir work_dirs/slowfast_custom \
    --resume

# Resume from specific checkpoint
python tools/train.py \
    configs/recognition/slowfast/slowfast_r50_custom.py \
    --work-dir work_dirs/slowfast_custom \
    --resume work_dirs/slowfast_custom/epoch_20.pth
```

### Step 6: Monitor Training

Training outputs are saved in `work_dirs/slowfast_custom/`:

```
work_dirs/slowfast_custom/
├── slowfast_r50_custom.py    # Copy of config
├── vis_data/                 # Visualization data
├── *.log                     # Training logs
├── epoch_*.pth               # Model checkpoints
└── last_checkpoint           # Path to latest checkpoint
```

**Monitor logs:**
```bash
# View training progress
tail -f work_dirs/slowfast_custom/*.log

# Or use tensorboard (if configured)
tensorboard --logdir work_dirs/slowfast_custom
```

## Testing and Evaluation

### Step 1: Test Single GPU

```bash
python tools/test.py \
    configs/recognition/slowfast/slowfast_r50_custom.py \
    work_dirs/slowfast_custom/best_acc_top1_epoch_*.pth \
    --dump results.pkl
```

### Step 2: Test Multi-GPU

```bash
bash tools/dist_test.sh \
    configs/recognition/slowfast/slowfast_r50_custom.py \
    work_dirs/slowfast_custom/best_acc_top1_epoch_*.pth \
    4 \
    --dump results.pkl
```

### Step 3: Inference on Videos

For inference on custom videos without annotation files:

```python
from mmaction.apis import inference_recognizer, init_recognizer

# Initialize model
config_file = 'configs/recognition/slowfast/slowfast_r50_custom.py'
checkpoint_file = 'work_dirs/slowfast_custom/best_acc_top1_epoch_40.pth'
model = init_recognizer(config_file, checkpoint_file, device='cuda:0')

# Run inference
video_path = 'path/to/your/video.mp4'
results = inference_recognizer(model, video_path)

# Get predictions
print(f"Top-5 predictions:")
for i, (score, label) in enumerate(zip(results.pred_score, results.pred_label)):
    print(f"{i+1}. Class {label}: {score:.4f}")
```

### Step 4: Batch Inference Script

Create `tools/inference_batch.py`:

```python
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

from mmaction.apis import inference_recognizer, init_recognizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def batch_inference(
    config_file: str,
    checkpoint_file: str,
    video_dir: str,
    output_csv: str,
    device: str = 'cuda:0'
) -> None:
    """
    Run batch inference on videos in a directory.
    
    Args:
        config_file: Path to config file
        checkpoint_file: Path to checkpoint file
        video_dir: Directory containing videos
        output_csv: Output CSV file path
        device: Device to use for inference
    """
    # Initialize model
    logger.info(f"Loading model from {checkpoint_file}")
    model = init_recognizer(config_file, checkpoint_file, device=device)
    
    # Get all videos
    video_dir = Path(video_dir)
    video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
    logger.info(f"Found {len(video_files)} videos")
    
    # Run inference
    results = []
    for video_file in tqdm(video_files, desc="Processing videos"):
        try:
            result = inference_recognizer(model, str(video_file))
            
            # Get top-5 predictions
            top5_scores = result.pred_score.topk(5)
            top5_labels = result.pred_label[:5]
            
            results.append({
                'video_name': video_file.name,
                'top1_label': int(top5_labels[0]),
                'top1_score': float(top5_scores.values[0]),
                'top5_labels': [int(l) for l in top5_labels],
                'top5_scores': [float(s) for s in top5_scores.values],
            })
        except Exception as e:
            logger.error(f"Error processing {video_file}: {e}")
            continue
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    logger.info(f"Results saved to {output_csv}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('video_dir', help='Directory containing videos')
    parser.add_argument('output_csv', help='Output CSV file')
    parser.add_argument('--device', default='cuda:0', help='Device')
    
    args = parser.parse_args()
    
    batch_inference(
        args.config,
        args.checkpoint,
        args.video_dir,
        args.output_csv,
        args.device
    )
```

Usage:
```bash
python tools/inference_batch.py \
    configs/recognition/slowfast/slowfast_r50_custom.py \
    work_dirs/slowfast_custom/best_acc_top1_epoch_40.pth \
    path/to/videos/ \
    predictions.csv
```

## Advanced Tips

### 1. Hyperparameter Tuning

**Learning Rate:**
```python
# Finding optimal learning rate
# Start with 10x smaller LR for finetuning
# Rule of thumb: base_lr * (your_batch_size / base_batch_size)

optim_wrapper = dict(
    optimizer=dict(
        lr=0.01,  # Try: 0.001, 0.005, 0.01, 0.05
    )
)
```

**Batch Size:**
```python
# Larger batch size = more stable gradients but requires more memory
# Adjust based on GPU memory:
# - 8-16 GB: batch_size=2-4
# - 16-24 GB: batch_size=4-8
# - 24+ GB: batch_size=8-16

train_dataloader = dict(
    batch_size=8,  # Adjust this
)
```

**Data Augmentation:**
```python
# Stronger augmentation can prevent overfitting
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter'),  # Add color jittering
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
```

### 2. Handling Imbalanced Datasets

If your dataset has class imbalance:

```python
from mmengine.dataset import DefaultSampler

# Option 1: Class-balanced sampling
train_dataloader = dict(
    sampler=dict(type='ClassBalancedSampler'),  # Instead of DefaultSampler
)

# Option 2: Use class weights in loss
model = dict(
    cls_head=dict(
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
            class_weight=[0.5, 1.0, 1.5, ...]  # Weights for each class
        )
    )
)
```

### 3. Reducing Memory Usage

If you encounter out-of-memory errors:

```python
# 1. Reduce batch size
train_dataloader = dict(batch_size=4)  # or even 2

# 2. Use gradient accumulation
optim_wrapper = dict(
    accumulative_counts=4,  # Accumulate gradients over 4 iterations
)

# 3. Enable automatic mixed precision (AMP)
# Add --amp flag during training
# bash tools/dist_train.sh config.py 4 --amp

# 4. Reduce input resolution
train_pipeline = [
    ...
    dict(type='Resize', scale=(192, 192), keep_ratio=False),  # 224 -> 192
    ...
]

# 5. Reduce clip length or frame interval
train_pipeline = [
    dict(type='SampleFrames', 
         clip_len=16,        # Reduced from 32
         frame_interval=4,   # Increased from 2
         num_clips=1),
    ...
]
```

### 4. Transfer Learning Strategies

**Strategy 1: Freeze backbone, train only head**
```python
# Freeze all parameters except classification head
model = dict(
    backbone=dict(
        frozen_stages=4,  # Freeze all backbone stages
    )
)
```

**Strategy 2: Gradual unfreezing**
```python
# First train only head for 10 epochs
# Then unfreeze and train all for remaining epochs
# This requires custom training logic or two-stage training
```

**Strategy 3: Different learning rates for different parts**
```python
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),    # 10x smaller LR for backbone
            'cls_head': dict(lr_mult=1.0),    # Normal LR for head
        }
    )
)
```

### 5. Using Different Pretrained Weights

MMAction2 provides various pretrained models:

```python
# Option 1: Kinetics-400 pretrained (default)
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth'

# Option 2: Local checkpoint
load_from = '/path/to/your/checkpoint.pth'

# Option 3: No pretraining (train from scratch)
load_from = None
```

### 6. Debugging Tips

**Test data pipeline:**
```python
from mmengine import Config
from mmaction.datasets import build_dataset

cfg = Config.fromfile('configs/recognition/slowfast/slowfast_r50_custom.py')
dataset = build_dataset(cfg.train_dataloader.dataset)

# Check first sample
sample = dataset[0]
print(f"Input shape: {sample['inputs'].shape}")
print(f"Label: {sample['data_samples'].gt_label}")
```

**Overfit on small subset:**
```bash
# Test if model can overfit on 10 samples (sanity check)
python tools/train.py config.py \
    --cfg-options train_dataloader.dataset.indices=[0,1,2,3,4,5,6,7,8,9] \
    train_cfg.max_epochs=100
```

### 7. Logging and Visualization

**Enable TensorBoard:**
```python
# Add to config
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend', save_dir='./work_dirs/tf_logs')
    ]
)

# Then view with:
# tensorboard --logdir ./work_dirs/tf_logs
```

**Log custom metrics:**
```python
# You can add custom hooks for logging additional metrics
custom_hooks = [
    dict(
        type='CustomMetricHook',
        # Your custom implementation
    )
]
```

### 8. Configuration Override at Runtime

Override config values without modifying files:

```bash
# Change learning rate
python tools/train.py config.py \
    --cfg-options optim_wrapper.optimizer.lr=0.005

# Change number of epochs
python tools/train.py config.py \
    --cfg-options train_cfg.max_epochs=100

# Change batch size and enable auto-scale-lr
python tools/train.py config.py \
    --cfg-options train_dataloader.batch_size=16 \
    --auto-scale-lr
```

### 9. Working with Limited Data

For small datasets (< 1000 videos):

1. **Use strong data augmentation**
2. **Reduce model capacity** (use smaller backbone)
3. **Use pretrained weights** (essential)
4. **Train for fewer epochs** with early stopping
5. **Use dropout** aggressively
6. **Consider few-shot learning** approaches

Example config for small dataset:
```python
model = dict(
    cls_head=dict(
        dropout_ratio=0.8,  # Increased from 0.5
    )
)

train_cfg = dict(
    max_epochs=30,  # Fewer epochs
)

# Add early stopping
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='acc/top1',
        patience=10,
        min_delta=0.001
    )
]
```

### 10. Multi-Dataset Training

Train on multiple datasets simultaneously:

```python
# Create combined dataset
train_dataloader = dict(
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='VideoDataset',
                ann_file='data/dataset1/train.txt',
                data_prefix=dict(video='data/dataset1/videos'),
                pipeline=train_pipeline
            ),
            dict(
                type='VideoDataset',
                ann_file='data/dataset2/train.txt',
                data_prefix=dict(video='data/dataset2/videos'),
                pipeline=train_pipeline
            ),
        ]
    )
)
```

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory

**Solutions:**
1. Reduce batch size
2. Enable gradient checkpointing
3. Use gradient accumulation
4. Reduce input resolution or clip length
5. Use mixed precision training (`--amp`)

### Issue 2: Model Not Learning (Loss Not Decreasing)

**Check:**
1. Is learning rate too high or too low?
2. Are labels correct? (verify annotation files)
3. Is data loading correctly? (check data pipeline)
4. Is gradient clipping too aggressive?
5. Try training from scratch on a tiny subset (should overfit)

### Issue 3: Low Validation Accuracy

**Possible causes:**
1. Overfitting: add more augmentation, reduce model capacity
2. Underfitting: train longer, increase model capacity, reduce regularization
3. Label noise in validation set
4. Domain shift between train and val

### Issue 4: Slow Training

**Optimizations:**
1. Increase `num_workers` in dataloader
2. Use `persistent_workers=True`
3. Use SSD instead of HDD for data storage
4. Enable `torch.backends.cudnn.benchmark=True`
5. Use video format instead of rawframes (less I/O)

### Issue 5: Config Inheritance Issues

**Debug:**
```python
from mmengine import Config

cfg = Config.fromfile('your_config.py')
print(cfg.pretty_text)  # Print full merged config
```

## Performance Benchmarks

Typical SlowFast R50 performance on different hardware:

| Hardware | Batch Size | Speed (videos/sec) | Memory Usage |
|----------|------------|-------------------|--------------|
| 1x RTX 3090 | 8 | ~40 | 16 GB |
| 1x V100 | 8 | ~35 | 14 GB |
| 1x RTX 2080 Ti | 4 | ~18 | 10 GB |
| 4x RTX 3090 | 32 (8x4) | ~160 | 64 GB total |

Training time estimate:
- Small dataset (1K videos): 2-4 hours
- Medium dataset (10K videos): 1-2 days
- Large dataset (100K videos): 1-2 weeks

## References

1. **SlowFast Paper**: Feichtenhofer et al., "SlowFast Networks for Video Recognition", ICCV 2019
2. **MMAction2 Docs**: https://mmaction2.readthedocs.io/
3. **Config System**: https://mmaction2.readthedocs.io/en/latest/user_guides/config.html
4. **Dataset Preparation**: https://mmaction2.readthedocs.io/en/latest/user_guides/prepare_dataset.html

## Next Steps

After successfully finetuning:

1. **Model Analysis**: Use analysis tools to understand what your model learned
2. **Ensemble Models**: Combine multiple models for better performance
3. **Model Compression**: Use knowledge distillation or pruning for deployment
4. **Deploy**: Export to ONNX or TorchScript for production

## Support and Community

- GitHub Issues: https://github.com/open-mmlab/mmaction2/issues
- Discord: OpenMMLab Discord Server
- Forum: OpenMMLab Community Forum

---

**Good luck with your SlowFast finetuning!** 🚀

