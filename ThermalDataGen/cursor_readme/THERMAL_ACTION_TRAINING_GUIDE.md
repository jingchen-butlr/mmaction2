# Thermal Action Detection Training Guide

**Version**: 1.0  
**Date**: November 12, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Format](#dataset-format)
3. [Dataset Generation](#dataset-generation)
4. [PyTorch DataLoader Usage](#pytorch-dataloader-usage)
5. [SlowFast Model Integration](#slowfast-model-integration)
6. [Training Configuration](#training-configuration)
7. [Example Training Script](#example-training-script)

---

## Overview

This guide explains how to use the thermal action detection dataset for training SlowFast models. The dataset is optimized for 64 consecutive frames (no temporal sampling gaps) and stores data in HDF5 format to minimize duplication.

### Key Features

- **64 consecutive frames** per sample (32 before + keyframe + 31 after)
- **HDF5 storage** for efficient sequential access (~10x storage savings)
- **COCO-style annotations** with AVA-compatible structure
- **14 action classes** from person subcategories
- **Thermal resolution**: 40 height × 60 width pixels (40 rows × 60 columns)
- **3-channel input**: Thermal data replicated to RGB-like format

---

## Dataset Format

### Directory Structure

```
thermal_action_dataset/
├── frames/
│   ├── SL14_R1.h5          # All frames for sensor SL14_R1
│   ├── SL14_R2.h5
│   ├── SL18_R1.h5
│   └── sensor_info.json    # Sensor metadata
├── annotations/
│   ├── train.json          # COCO-style training annotations
│   ├── val.json            # COCO-style validation annotations
│   └── class_mapping.json  # Action class ID to name mapping
├── statistics/
│   ├── dataset_stats.json
│   └── validation_report.json
└── dataset_info.json       # Overall dataset metadata
```

### HDF5 File Structure

Each `.h5` file contains:

```python
{
    'frames': [N, 40, 60] float32,      # Temperature in Celsius (N, height, width)
    'timestamps': [N] int64,             # Millisecond timestamps
    'frame_seqs': [N] int64,             # Frame sequence numbers
    'attrs': {
        'sensor_id': str,
        'mac_address': str,
        'total_frames': int,
        'corrupted_count': int,
        'min_timestamp': int,
        'max_timestamp': int
    }
}
```

### COCO Annotation Format

```json
{
  "images": [
    {
      "id": "SL18_R1_1760639220331",
      "sensor_id": "SL18_R1",
      "mac_address": "02:00:1a:62:51:67",
      "timestamp": 1760639220331,
      "width": 40,
      "height": 60,
      "frame_idx": 1523
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": "SL18_R1_1760639220331",
      "bbox": [0.32, 0.76, 0.185, 0.252],
      "category_id": 0,
      "category_name": "sitting",
      "object_id": 1
    }
  ],
  "categories": [
    {"id": 0, "name": "sitting"},
    {"id": 1, "name": "standing"},
    ...
  ]
}
```

**Note**: `bbox` is in YOLO format `[centerX, centerY, width, height]`, normalized to [0, 1].

### Action Classes (14 total)

| ID | Action Class |
|----|-------------|
| 0 | sitting |
| 1 | standing |
| 2 | walking |
| 3 | lying down-lying with risk |
| 4 | lying down-lying on the bed/couch |
| 5 | leaning |
| 6 | transition-normal transition |
| 7 | transition-lying with risk transition |
| 8 | transition-lying on the bed transition |
| 9 | lower position-other |
| 10 | lower position-kneeling |
| 11 | lower position-bending |
| 12 | lower position-crouching |
| 13 | other |

---

## Dataset Generation

### Quick Start

Generate the complete dataset from TDengine:

```bash
cd /Users/jma/Github/Butlr/YOLOv11

python scripts/thermal_action/generate_thermal_action_dataset.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --output-dir thermal_action_dataset \
  --val-split 0.2 \
  --buffer-frames 128
```

### Step-by-Step Generation

If you prefer manual control:

```bash
# Step 1: Create HDF5 frame storage
python scripts/thermal_action/create_hdf5_frames.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --output-dir thermal_action_dataset/frames \
  --buffer-frames 128

# Step 2: Convert annotations to COCO format
python scripts/thermal_action/convert_annotations_to_coco.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --hdf5-dir thermal_action_dataset/frames \
  --output-dir thermal_action_dataset/annotations \
  --val-split 0.2

# Step 3: Validate dataset
python scripts/thermal_action/validate_dataset.py \
  --hdf5-dir thermal_action_dataset/frames \
  --annotations-dir thermal_action_dataset/annotations \
  --num-samples 6
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--val-split` | 0.2 | Validation split ratio (0-1) |
| `--random-seed` | 42 | Random seed for reproducibility |
| `--buffer-frames` | 128 | Frames to fetch before/after annotations |
| `--compression` | gzip | HDF5 compression (gzip/lzf/none) |
| `--compression-level` | 4 | Compression level (0-9) |
| `--tdengine-host` | localhost | TDengine server address |
| `--tdengine-port` | 6041 | TDengine REST API port |

---

## PyTorch DataLoader Usage

### Basic Usage

```python
from scripts.thermal_action.thermal_action_dataset import (
    ThermalActionDataset,
    ThermalActionTransform,
    collate_fn
)
from torch.utils.data import DataLoader

# Create dataset
train_dataset = ThermalActionDataset(
    hdf5_root='thermal_action_dataset/frames',
    ann_file='thermal_action_dataset/annotations/train.json',
    transforms=ThermalActionTransform(is_train=True),
    frame_window=64
)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)

# Iterate over batches
for frames, boxes, labels, extras in train_loader:
    # frames: [B, 3, 64, 40, 60] - 3 channels, 64 frames, 40 height × 60 width
    # boxes: List of [N_i, 4] tensors (variable N per sample)
    # labels: List of [N_i] tensors (action class IDs)
    # extras: List of dicts with metadata
    
    # Forward pass
    output = model(frames, boxes, labels)
```

### Data Format

#### Input Frames

- **Shape**: `[B, 3, 64, 40, 60]`
- **Dtype**: float32
- **Range**: [0, 1] after normalization
- **Channels**: Thermal data replicated 3 times (R=G=B)
- **Temporal**: 64 consecutive frames (no gaps)
- **Spatial**: 40 height × 60 width (40 rows × 60 columns)

#### Bounding Boxes

- **Format**: List of tensors (not batched due to variable N)
- **Coordinates**: Normalized [0, 1] in `[centerX, centerY, width, height]` format
- **Per-sample**: Variable number of persons

#### Labels

- **Format**: List of tensors (not batched)
- **Values**: Action class IDs (0-13)
- **Multi-label**: Not currently supported (each person has one primary action)

### Custom Transforms

```python
class CustomThermalTransform:
    def __init__(self, is_train=True):
        self.is_train = is_train
    
    def __call__(self, frames, boxes):
        # frames: [64, 40, 60, 3] numpy array (64 frames, 40 height, 60 width, 3 channels)
        # boxes: [N, 4] numpy array
        
        # Temperature normalization (5-45°C to [0, 1])
        frames = np.clip(frames, 5.0, 45.0)
        frames = (frames - 5.0) / 40.0
        
        # Random horizontal flip (training only)
        if self.is_train and np.random.rand() < 0.5:
            frames = frames[:, :, ::-1, :]  # Flip W
            boxes = boxes.copy()
            boxes[:, 0] = 1.0 - boxes[:, 0]  # Flip centerX
        
        return frames, boxes
```

---

## SlowFast Model Integration

### Model Input Requirements

The SlowFast model expects dual-pathway inputs:

```python
# After loading from dataset:
frames = frames_batch  # [B, 3, 64, 40, 60]

# Split into slow and fast pathways (model-side)
# Slow pathway: Subsample temporally (e.g., every 8th frame)
# Fast pathway: Use all frames or subsample less (e.g., every 2nd frame)

# This splitting should be done INSIDE the model or as a preprocessing step
```

### Recommended Approach

**Option 1**: Let the model handle pathway splitting

```python
class SlowFastThermalModel(nn.Module):
    def __init__(self, slowfast_backbone):
        super().__init__()
        self.backbone = slowfast_backbone
    
    def forward(self, frames, boxes, labels):
        # frames: [B, 3, 64, 40, 60]
        
        # Split into pathways
        slow_frames = frames[:, :, ::8, :, :]  # Every 8th frame → [B, 3, 8, 40, 60]
        fast_frames = frames[:, :, ::2, :, :]  # Every 2nd frame → [B, 3, 32, 40, 60]
        
        # Forward through SlowFast backbone
        features = self.backbone([slow_frames, fast_frames])
        
        # ... rest of model (RoI pooling, action classification)
```

**Option 2**: Preprocess in dataset

```python
class SlowFastTransform:
    def __call__(self, frames, boxes):
        # frames: [64, 40, 60, 3]
        
        # Extract slow and fast pathways
        slow = frames[::8]   # [8, 40, 60, 3]
        fast = frames[::2]   # [32, 40, 60, 3]
        
        return (slow, fast), boxes
```

### Integration Example

```python
# Assuming AlphAction/SlowFast codebase structure
from slowfast.models import build_model
from alphaction.config import cfg

# Load config
cfg.merge_from_file('configs/thermal_action_slowfast.yaml')

# Modify config for thermal data
cfg.INPUT.VIDEO_SIZE = [40, 60]  # Height x Width (40 height, 60 width)
cfg.INPUT.CHANNELS = 3  # Replicated thermal
cfg.MODEL.NUM_CLASSES = 14  # Action classes
cfg.INPUT.FRAME_NUM = 64  # Total frames
cfg.INPUT.FRAME_SAMPLE_RATE = 1  # No sampling (consecutive)
cfg.INPUT.TAU = 8  # Slow pathway stride
cfg.INPUT.ALPHA = 4  # Fast/slow ratio

# Build model
model = build_model(cfg)

# Train
for frames, boxes, labels, extras in train_loader:
    # frames: [B, 3, 64, 40, 60]
    frames = frames.cuda()
    
    # Model handles pathway splitting internally
    loss_dict = model(frames, boxes, labels)
    
    loss = sum(loss_dict.values())
    loss.backward()
    optimizer.step()
```

---

## Training Configuration

### Recommended Hyperparameters

```yaml
# Config for thermal action detection

INPUT:
  VIDEO_SIZE: [40, 60]      # Height x Width (40 height, 60 width)
  CHANNELS: 3               # Replicated thermal to 3 channels
  FRAME_NUM: 64             # Total frames per sample
  FRAME_SAMPLE_RATE: 1      # No sampling (consecutive frames)
  TAU: 8                    # Slow pathway temporal stride
  ALPHA: 4                  # Fast pathway has 4x more frames
  
  # Temperature normalization
  PIXEL_MEAN: [20.0, 20.0, 20.0]  # Approximate room temperature
  PIXEL_STD: [10.0, 10.0, 10.0]   # Approximate thermal variance
  
  # Augmentation
  FLIP_PROB: 0.5
  MIN_SIZE_TRAIN: 60        # No resizing (keep native resolution)
  MAX_SIZE_TRAIN: 60
  COLOR_JITTER: False       # Skip for thermal data

MODEL:
  BACKBONE: 'SlowFast-8x8'
  NUM_CLASSES: 14           # Action classes
  ROI_HEAD:
    POOLER_RESOLUTION: 7
    POOLER_SAMPLING_RATIO: 0

SOLVER:
  BASE_LR: 0.01
  WEIGHT_DECAY: 0.0001
  MOMENTUM: 0.9
  WARMUP_EPOCHS: 5
  MAX_EPOCHS: 50
  LR_POLICY: 'cosine'

DATALOADER:
  VIDEOS_PER_BATCH: 8
  NUM_WORKERS: 4
  SIZE_DIVISIBILITY: 16
```

### Loss Functions

For action detection, use:

- **Focal Loss** for action classification (handles class imbalance)
- **Smooth L1 Loss** for bounding box regression (if needed)

```python
from torch.nn import functional as F

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss for action classification.
    
    Args:
        pred: [N, 14] logits
        target: [N] class labels (0-13)
    """
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    p_t = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
    return focal_loss.mean()
```

---

## Example Training Script

```python
#!/usr/bin/env python3
"""
Train SlowFast model for thermal action detection.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from scripts.thermal_action.thermal_action_dataset import (
    ThermalActionDataset,
    ThermalActionTransform,
    collate_fn
)

# Configuration
config = {
    'hdf5_root': 'thermal_action_dataset/frames',
    'train_ann': 'thermal_action_dataset/annotations/train.json',
    'val_ann': 'thermal_action_dataset/annotations/val.json',
    'batch_size': 8,
    'num_workers': 4,
    'max_epochs': 50,
    'base_lr': 0.01,
    'weight_decay': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Create datasets
train_dataset = ThermalActionDataset(
    hdf5_root=config['hdf5_root'],
    ann_file=config['train_ann'],
    transforms=ThermalActionTransform(is_train=True)
)

val_dataset = ThermalActionDataset(
    hdf5_root=config['hdf5_root'],
    ann_file=config['val_ann'],
    transforms=ThermalActionTransform(is_train=False)
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['num_workers'],
    collate_fn=collate_fn,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config['num_workers'],
    collate_fn=collate_fn
)

# Build model (placeholder - replace with actual SlowFast model)
class SimpleThermalActionModel(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        # Placeholder: Replace with actual SlowFast backbone
        self.conv3d = nn.Conv3d(3, 64, kernel_size=(3, 3, 3))
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, frames, boxes_list, labels_list):
        # frames: [B, 3, 64, 40, 60]
        x = self.conv3d(frames)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        
        # Compute loss (simplified)
        losses = {}
        if labels_list is not None:
            # Aggregate all labels
            all_labels = torch.cat(labels_list, dim=0).to(frames.device)
            # Repeat logits for each person (simplified)
            loss = nn.functional.cross_entropy(
                logits.repeat(len(all_labels), 1)[:len(all_labels)],
                all_labels
            )
            losses['action_loss'] = loss
        
        return losses if self.training else logits

model = SimpleThermalActionModel(num_classes=14)
model = model.to(config['device'])

# Optimizer and scheduler
optimizer = SGD(
    model.parameters(),
    lr=config['base_lr'],
    momentum=0.9,
    weight_decay=config['weight_decay']
)

scheduler = CosineAnnealingLR(optimizer, T_max=config['max_epochs'])

# Training loop
for epoch in range(config['max_epochs']):
    model.train()
    total_loss = 0
    
    for batch_idx, (frames, boxes, labels, extras) in enumerate(train_loader):
        frames = frames.to(config['device'])
        
        # Forward
        loss_dict = model(frames, boxes, labels)
        loss = sum(loss_dict.values())
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    scheduler.step()
    
    # Validation (simplified)
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for frames, boxes, labels, extras in val_loader:
            frames = frames.to(config['device'])
            loss_dict = model(frames, boxes, labels)
            val_loss += sum(loss_dict.values()).item()
    
    print(f"Epoch {epoch}: Train Loss={total_loss/len(train_loader):.4f}, Val Loss={val_loss/len(val_loader):.4f}")

# Save model
torch.save(model.state_dict(), 'thermal_action_model.pth')

# Cleanup
train_dataset.close()
val_dataset.close()
```

---

## Performance Tips

### 1. HDF5 File Caching

Keep HDF5 files open throughout training (already done in `ThermalActionDataset`):

```python
# Files remain open for fast slicing
self.hdf5_files[sensor_id] = h5py.File(h5_path, 'r')
```

### 2. Multi-Worker DataLoader

Use multiple workers for parallel data loading:

```python
train_loader = DataLoader(
    dataset,
    num_workers=4,  # Adjust based on CPU cores
    pin_memory=True  # Faster GPU transfer
)
```

### 3. Mixed Precision Training

Use automatic mixed precision for faster training:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for frames, boxes, labels, extras in train_loader:
    with autocast():
        loss = model(frames, boxes, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 4. Gradient Accumulation

For larger effective batch sizes:

```python
accumulation_steps = 4

for i, (frames, boxes, labels, extras) in enumerate(train_loader):
    loss = model(frames, boxes, labels)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or use gradient accumulation

```python
config['batch_size'] = 4  # Reduce from 8
```

### Issue: Slow Data Loading

**Solution**: Increase `num_workers` or check HDF5 compression level

```python
config['num_workers'] = 8  # Increase from 4
```

### Issue: NaN Loss

**Solution**: Check learning rate, add gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Issue: Class Imbalance

**Solution**: Use weighted loss or focal loss

```python
class_weights = torch.tensor([1.0, 1.5, 2.0, ...])  # Adjust per class
loss = F.cross_entropy(pred, target, weight=class_weights)
```

---

## Next Steps

1. **Generate Dataset**: Run `generate_thermal_action_dataset.py`
2. **Validate Dataset**: Run `validate_dataset.py`
3. **Integrate with SlowFast**: Adapt AlphAction codebase
4. **Experiment**: Try different temporal sampling strategies
5. **Evaluate**: Compute mAP on validation set

---

**For questions or issues**, refer to:
- Dataset generation logs: `thermal_action_dataset/dataset_info.json`
- Validation report: `thermal_action_dataset/statistics/validation_report.json`
- Visualizations: `thermal_action_dataset/statistics/visualizations/`

