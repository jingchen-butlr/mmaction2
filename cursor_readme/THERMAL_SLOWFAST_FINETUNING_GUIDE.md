## Thermal SlowFast Finetuning Guide for MMAction2

**Complete guide for training SlowFast on your thermal action dataset**

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Analysis](#dataset-analysis)
3. [Key Challenges & Solutions](#key-challenges--solutions)
4. [Setup Instructions](#setup-instructions)
5. [Training Workflow](#training-workflow)
6. [Monitoring & Debugging](#monitoring--debugging)
7. [Expected Results](#expected-results)
8. [Advanced Topics](#advanced-topics)

---

## Overview

### Your Thermal Dataset

**Location**: `mmaction2/ThermalDataGen/thermal_action_dataset/`

**Characteristics**:
- **Format**: HDF5 files with thermal frames
- **Frame size**: 40 height × 60 width (original)
- **Model input**: 256 height × 384 width (resized, maintains 2:3 aspect ratio)
- **Training samples**: 314
- **Validation samples**: 73
- **Action classes**: 14
- **Temporal window**: 64 consecutive frames
- **Data size**: 9 MB (compressed)

**Action Classes** (sorted by frequency):
1. `lying down-lying with risk` - 220 annotations (54%)
2. `standing` - 123 annotations (30%)
3. `lower position-kneeling` - 12 annotations (3%)
4. `transition-normal transition` - 11 annotations (3%)
5. `walking` - 10 annotations (2%)
6. Other classes - < 10 annotations each

---

## Dataset Analysis

### Strengths
✅ High-quality HDF5 storage (efficient)  
✅ Temporal continuity (64 consecutive frames)  
✅ Clean annotations in COCO format  
✅ Well-balanced train/val split (80/20)  

### Challenges
⚠️ **Very small dataset** (314 samples)  
⚠️ **Severe class imbalance** (54% in one class)  
⚠️ **Low resolution** (40×60 pixels)  
⚠️ **Domain shift** (thermal vs RGB pretraining)  

### Solutions Implemented

1. **Small Dataset → Heavy Data Augmentation**
   - ColorJitter, RandomErasing, RandomResizedCrop
   - Aggressive augmentation parameters
   - Longer training (100 epochs)

2. **Class Imbalance → Weighted Loss**
   - Automatic class weight computation
   - Focus on mean class accuracy metric
   - Consider oversampling minority classes

3. **Low Resolution → Smart Resizing**
   - 40×60 → 256×384 (maintains 2:3 aspect ratio)
   - Larger model input for better feature extraction

4. **Domain Shift → Careful Finetuning**
   - Lower learning rate (0.005 vs 0.05)
   - Different LR for backbone vs head
   - Long warmup period (10 epochs)

---

## Setup Instructions

### Step 1: Verify Environment

```bash
# Navigate to mmaction2 directory
cd /home/ec2-user/jingchen/mmaction2

# Verify MMAction2 installation
python -c "import mmaction; print(f'MMAction2: {mmaction.__version__}')"

# Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Check thermal dataset
ls -lh ThermalDataGen/thermal_action_dataset/
```

**Expected output**:
```
ThermalDataGen/thermal_action_dataset/
├── frames/          # 8 HDF5 files (~9 MB total)
├── annotations/     # train.json, val.json
└── statistics/      # Dataset stats
```

### Step 2: Download Pretrained Weights

```bash
# Download SlowFast R50 pretrained on Kinetics-400
python tools/download_pretrained_slowfast.py

# Verify download
ls -lh checkpoints/slowfast_*.pth
```

**Alternative** (manual download):
```bash
mkdir -p checkpoints
wget https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth \
    -O checkpoints/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth
```

### Step 3: Verify Dataset Loading

```bash
# Test custom dataset loader
python -c "
from mmaction.datasets import ThermalHDF5Dataset

dataset = ThermalHDF5Dataset(
    ann_file='ThermalDataGen/thermal_action_dataset/annotations/train.json',
    data_prefix={'hdf5': 'ThermalDataGen/thermal_action_dataset/frames'},
    pipeline=[],
    test_mode=False
)

print(f'✓ Dataset loaded: {len(dataset)} samples')
sample = dataset[0]
print(f'✓ Sample shape: {sample[\"imgs\"].shape}')
dataset.close()
"
```

**Expected output**:
```
✓ Dataset loaded: 314 samples
✓ Sample shape: (64, 40, 60, 3)
```

### Step 4: Verify Config

```bash
# Check config syntax
python tools/train.py \
    configs/recognition/slowfast/slowfast_thermal_finetuning.py \
    --cfg-options dry_run=True

# Should print config without errors
```

---

## Training Workflow

### Quick Start (Recommended)

```bash
# Single GPU training (recommended for small dataset)
python tools/train.py \
    configs/recognition/slowfast/slowfast_thermal_finetuning.py \
    --work-dir work_dirs/thermal_slowfast \
    --seed 42
```

### Multi-GPU Training

```bash
# 2 GPUs (if available)
bash tools/dist_train.sh \
    configs/recognition/slowfast/slowfast_thermal_finetuning.py \
    2 \
    --work-dir work_dirs/thermal_slowfast \
    --seed 42
```

### Training with Mixed Precision (Memory Efficient)

```bash
# Reduces memory usage by ~30%
python tools/train.py \
    configs/recognition/slowfast/slowfast_thermal_finetuning.py \
    --work-dir work_dirs/thermal_slowfast \
    --amp \
    --seed 42
```

### Resume Interrupted Training

```bash
python tools/train.py \
    configs/recognition/slowfast/slowfast_thermal_finetuning.py \
    --work-dir work_dirs/thermal_slowfast \
    --resume
```

### Override Config at Runtime

```bash
# Change learning rate
python tools/train.py \
    configs/recognition/slowfast/slowfast_thermal_finetuning.py \
    --cfg-options optim_wrapper.optimizer.lr=0.001

# Change batch size
python tools/train.py \
    configs/recognition/slowfast/slowfast_thermal_finetuning.py \
    --cfg-options train_dataloader.batch_size=2

# Change number of epochs
python tools/train.py \
    configs/recognition/slowfast/slowfast_thermal_finetuning.py \
    --cfg-options train_cfg.max_epochs=50
```

---

## Monitoring & Debugging

### Training Progress

```bash
# Watch training logs in real-time
tail -f work_dirs/thermal_slowfast/*.log

# Search for validation accuracy
grep "acc/top1" work_dirs/thermal_slowfast/*.log
```

### Output Directory Structure

```
work_dirs/thermal_slowfast/
├── slowfast_thermal_finetuning.py   # Config backup
├── *.log                            # Training logs
├── epoch_*.pth                      # Checkpoints (every 5 epochs)
├── best_acc_top1_epoch_*.pth        # Best model
├── last_checkpoint                  # Latest checkpoint reference
└── tf_logs/                         # TensorBoard logs
```

### TensorBoard Monitoring

```bash
# Start TensorBoard
tensorboard --logdir work_dirs/thermal_slowfast/tf_logs --port 6006

# Access in browser (forward port if remote)
# ssh -L 6006:localhost:6006 user@remote-server
```

### Common Training Issues

#### Issue 1: CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# 1. Reduce batch size
python tools/train.py config.py \
    --cfg-options train_dataloader.batch_size=2

# 2. Enable mixed precision
python tools/train.py config.py --amp

# 3. Reduce frame window (in config)
# Change: frame_window=64 → frame_window=32
```

#### Issue 2: Loss Not Decreasing

**Symptoms**: Loss stays constant or increases

**Checks**:
```python
# Verify data loading
python -c "
from mmaction.datasets import ThermalHDF5Dataset
ds = ThermalHDF5Dataset(...)
sample = ds[0]
print(f'Frame range: [{sample[\"imgs\"].min()}, {sample[\"imgs\"].max()}]')
print(f'Label: {sample[\"label\"]} (should be 0-13)')
"
```

**Solutions**:
- Check learning rate (try 10x smaller or larger)
- Verify data normalization
- Check class labels (must be 0-13, not 1-14)

#### Issue 3: Overfitting

**Symptoms**: Train accuracy >> Val accuracy

**Solutions**:
- Increase dropout: `cls_head.dropout_ratio=0.9`
- Add more augmentation
- Reduce model capacity
- Add early stopping

#### Issue 4: Slow Training

**Optimizations**:
```python
# In config:
train_dataloader = dict(
    num_workers=8,  # Increase workers
    persistent_workers=True,  # Keep workers alive
)

env_cfg = dict(
    cudnn_benchmark=True,  # Enable for fixed input size
)
```

---

## Expected Results

### Training Time

| Hardware | Batch Size | Speed | Time (100 epochs) |
|----------|------------|-------|-------------------|
| 1x RTX 3090 | 4 | ~15 samples/sec | 4-5 hours |
| 1x V100 | 4 | ~12 samples/sec | 5-6 hours |
| 2x RTX 3090 | 8 (4×2) | ~30 samples/sec | 2-3 hours |

### Accuracy Expectations

**Small dataset limits**:
- Top-1 accuracy: **50-70%** (realistic target)
- Mean class accuracy: **40-60%** (with imbalance)
- Top-3 accuracy: **70-85%**

**Class-specific performance**:
- High accuracy expected: `lying down-lying with risk` (220 samples)
- Medium accuracy: `standing`, `walking`
- Low accuracy: Rare classes with <10 samples

**Monitoring metrics**:
```bash
# Focus on these metrics in logs
- acc/top1: Overall accuracy
- acc/mean_class: Per-class average (better for imbalance)
- loss: Should steadily decrease
```

### Baseline Comparison

| Method | Top-1 Acc | Notes |
|--------|-----------|-------|
| Random guess | 7.1% | (1/14 classes) |
| Majority class | 54% | Always predict "lying with risk" |
| Finetuned SlowFast | **60-70%** | Target with augmentation |
| + More data (future) | **75-85%** | With 1000+ samples |

---

## Advanced Topics

### 1. Handling Class Imbalance

**Option A: Class Weights (Already Implemented)**

The config automatically computes class weights:
```python
from mmaction.datasets import get_class_weights

weights = get_class_weights('path/to/train.json')
# Weights: [high for rare classes, low for common classes]
```

**Option B: Oversampling Minority Classes**

Create a custom sampler:
```python
from torch.utils.data import WeightedRandomSampler

# In config:
train_dataloader = dict(
    sampler=dict(
        type='WeightedRandomSampler',
        weights=sample_weights,  # Based on class frequency
        num_samples=len(dataset) * 2,  # Oversample
        replacement=True
    )
)
```

**Option C: Focal Loss**

Replace CrossEntropyLoss with FocalLoss (focus on hard examples):
```python
model = dict(
    cls_head=dict(
        loss_cls=dict(
            type='FocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        )
    )
)
```

### 2. Data Augmentation Strategies

**Current augmentation** (already in config):
- RandomResizedCrop (0.7-1.0 scale)
- Horizontal flip (50%)
- ColorJitter
- RandomErasing (25%)

**Additional augmentation** (add to config):
```python
train_pipeline = [
    # ... existing transforms ...
    
    # Temporal augmentation
    dict(type='TemporalCrop', crop_ratio=0.9),
    
    # Mixup (mix two samples)
    dict(type='MixUp', alpha=0.2),
    
    # CutMix (replace patches between samples)
    dict(type='CutMix', alpha=1.0),
    
    # ... rest of pipeline ...
]
```

### 3. Transfer Learning Strategies

**Strategy A: Frozen Backbone (First 10 Epochs)**

```python
# Train only classification head first
model = dict(
    backbone=dict(
        frozen_stages=4,  # Freeze all 4 ResNet stages
    )
)

# After 10 epochs, unfreeze and continue training
# Requires manual config change or custom hook
```

**Strategy B: Differential Learning Rates (Already Implemented)**

```python
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # Backbone: 10x slower
            'cls_head': dict(lr_mult=1.0),  # Head: full speed
        }
    )
)
```

**Strategy C: Gradual Unfreezing**

Unfreeze layers progressively (requires custom hook).

### 4. Model Variants

**Try different SlowFast configurations**:

```bash
# Download SlowFast R101 (larger model, more capacity)
python tools/download_pretrained_slowfast.py --model slowfast_r101_8x8_kinetics400

# Update config:
_base_ = '../../_base_/models/slowfast_r101.py'
```

**Try SlowOnly** (simpler, faster):
```python
# Modify config to use SlowOnly instead of SlowFast
model = dict(
    type='Recognizer3D',
    backbone=dict(type='ResNet3dSlowOnly', ...)
)
```

### 5. Ensemble Methods

**Combine multiple models for better accuracy**:

```python
# Train multiple models with different seeds
python tools/train.py config.py --seed 42 --work-dir work_dirs/model_1
python tools/train.py config.py --seed 123 --work-dir work_dirs/model_2
python tools/train.py config.py --seed 456 --work-dirs/model_3

# Ensemble predictions (average probabilities)
# Create custom inference script
```

### 6. Model Deployment

**Export to ONNX** for production:

```python
from mmaction.apis import inference_recognizer, init_recognizer
import torch

# Load model
model = init_recognizer(
    'configs/recognition/slowfast/slowfast_thermal_finetuning.py',
    'work_dirs/thermal_slowfast/best_acc_top1_epoch_80.pth',
    device='cuda:0'
)

# Export to ONNX
dummy_input = torch.randn(1, 3, 64, 256, 384).cuda()
torch.onnx.export(
    model,
    dummy_input,
    'thermal_slowfast.onnx',
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

### 7. Dataset Expansion Plans

**When you collect more data**:

1. **Regenerate HDF5 files**:
   ```bash
   cd ThermalDataGen
   python scripts/thermal_action/generate_thermal_action_dataset.py \
       --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
       --output-dir thermal_action_dataset \
       --val-split 0.2
   ```

2. **Update config**:
   - Reduce dropout: `0.8 → 0.5`
   - Increase epochs if needed: `100 → 150`
   - Update class weights

3. **Consider progressive training**:
   - Finetune on current 314 samples
   - Continue training on expanded dataset
   - May achieve 75-85% accuracy with 1000+ samples

---

## Testing and Evaluation

### Test Best Model

```bash
python tools/test.py \
    configs/recognition/slowfast/slowfast_thermal_finetuning.py \
    work_dirs/thermal_slowfast/best_acc_top1_epoch_*.pth \
    --dump results.pkl
```

### Inference on Single Sample

```python
from mmaction.apis import inference_recognizer, init_recognizer

# Initialize model
model = init_recognizer(
    'configs/recognition/slowfast/slowfast_thermal_finetuning.py',
    'work_dirs/thermal_slowfast/best_acc_top1_epoch_80.pth',
    device='cuda:0'
)

# Load class names
import json
with open('ThermalDataGen/thermal_action_dataset/annotations/class_mapping.json') as f:
    class_mapping = json.load(f)
    classes = class_mapping['classes']

# Run inference on thermal HDF5 data
# (Requires custom inference function for HDF5)

print(f"Predicted action: {classes[pred_label]}")
print(f"Confidence: {pred_score:.2%}")
```

### Confusion Matrix Analysis

```python
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Collect predictions (from test.py results)
y_true = []  # True labels
y_pred = []  # Predicted labels

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualize
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=classes,
    yticklabels=classes,
    cmap='Blues'
)
plt.title('Confusion Matrix - Thermal Action Recognition')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)

# Print classification report
print(classification_report(y_true, y_pred, target_names=classes))
```

---

## Troubleshooting Checklist

Before asking for help, verify:

- [ ] Dataset loads correctly (test with script above)
- [ ] Pretrained weights downloaded
- [ ] Config file has no syntax errors
- [ ] GPU is available and has enough memory
- [ ] PYTHONPATH includes mmaction2 directory
- [ ] Data paths are correct (absolute or relative to mmaction2/)
- [ ] Checked training logs for specific error messages
- [ ] Tried reducing batch size if OOM errors

---

## Quick Reference

### File Locations

```
mmaction2/
├── configs/recognition/slowfast/
│   └── slowfast_thermal_finetuning.py      ← Main config
├── mmaction/datasets/
│   └── thermal_hdf5_dataset.py             ← Custom dataset
├── tools/
│   ├── train.py                            ← Training script
│   ├── test.py                             ← Testing script
│   └── download_pretrained_slowfast.py     ← Download weights
├── ThermalDataGen/
│   └── thermal_action_dataset/             ← Your data
│       ├── frames/                         ← HDF5 files
│       └── annotations/                    ← train.json, val.json
└── checkpoints/
    └── slowfast_r50_*.pth                  ← Pretrained weights
```

### Training Commands Summary

```bash
# 1. Download weights
python tools/download_pretrained_slowfast.py

# 2. Train
python tools/train.py configs/recognition/slowfast/slowfast_thermal_finetuning.py

# 3. Test
python tools/test.py configs/recognition/slowfast/slowfast_thermal_finetuning.py \
    work_dirs/thermal_slowfast/best_acc_top1_*.pth
```

### Key Parameters to Tune

| Parameter | Default | Try If... |
|-----------|---------|-----------|
| `lr` | 0.005 | Loss not decreasing: 0.01, 0.001 |
| `batch_size` | 4 | OOM error: 2, 1 |
| `dropout_ratio` | 0.8 | Overfitting: 0.9 |
| `max_epochs` | 100 | Poor convergence: 150, 200 |
| `warmup_epochs` | 10 | Unstable training: 20 |

---

## Support and Resources

- **MMAction2 Docs**: https://mmaction2.readthedocs.io/
- **SlowFast Paper**: https://arxiv.org/abs/1812.03982
- **Thermal Dataset Docs**: `ThermalDataGen/cursor_readme/`
- **Config Guide**: `cursor_readme/slowfast_finetuning_guide.md`

---

**Good luck with your thermal action recognition training!** 🚀

For questions or issues, check the troubleshooting section first, then review logs carefully.

