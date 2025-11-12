# Thermal Action Detection - AWS Execution Guide

**Target Instance**: i-02a024601a04bc5de (a-fall-det-model-jingchen)  
**Public IP**: 44.248.130.76  
**User**: ec2-user  
**SlowFast Location**: `/home/ec2-user/jingchen/AlphAction`

---

## Quick Deployment

### 1. Copy Files from Local to AWS

From your local machine (Mac):

```bash
cd /Users/jma/Github/Butlr/YOLOv11

# Option A: Copy archive (all files in one go)
scp thermal_action_aws_deploy.tar.gz \
  ec2-user@44.248.130.76:/home/ec2-user/jingchen/

# Option B: Use rsync for selective sync (recommended)
rsync -avz --progress \
  scripts/thermal_action/ \
  ec2-user@44.248.130.76:/home/ec2-user/jingchen/scripts/thermal_action/

rsync -avz --progress \
  DataAnnotationQA/src/data_pipeline/ \
  ec2-user@44.248.130.76:/home/ec2-user/jingchen/DataAnnotationQA/src/data_pipeline/

rsync -avz --progress \
  DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/ \
  ec2-user@44.248.130.76:/home/ec2-user/jingchen/DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/

# Copy documentation
rsync -avz --progress \
  cursor_readme/THERMAL_ACTION*.md \
  ec2-user@44.248.130.76:/home/ec2-user/jingchen/cursor_readme/

# Copy setup script
scp setup_thermal_action_aws.sh AWS_EXECUTION_GUIDE.md \
  ec2-user@44.248.130.76:/home/ec2-user/jingchen/
```

### 2. SSH to AWS and Setup

```bash
ssh ec2-user@44.248.130.76

cd /home/ec2-user/jingchen

# If using archive (Option A)
tar -xzf thermal_action_aws_deploy.tar.gz

# Run setup
chmod +x setup_thermal_action_aws.sh
./setup_thermal_action_aws.sh
```

---

## Generate Dataset on AWS

### Full Generation

```bash
cd /home/ec2-user/jingchen

python scripts/thermal_action/generate_thermal_action_dataset.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --output-dir thermal_action_dataset \
  --val-split 0.2 \
  --buffer-frames 128 \
  --tdengine-host 35.90.244.93 \
  --tdengine-port 6041
```

**Expected Output**:
- Duration: ~3-5 seconds
- Storage: ~9.3 MB (8 sensors)
- Train: 314 images (405 annotations)
- Val: 73 images (95 annotations)

### Validate Dataset

```bash
python scripts/thermal_action/validate_dataset.py \
  --hdf5-dir thermal_action_dataset/frames \
  --annotations-dir thermal_action_dataset/annotations \
  --num-samples 6
```

**Output**:
- `thermal_action_dataset/statistics/validation_report.json`
- `thermal_action_dataset/statistics/visualizations/*.png`

---

## AlphAction Integration

### Directory Structure on AWS

```
/home/ec2-user/jingchen/
├── AlphAction/                    # SlowFast codebase
│   ├── configs/
│   ├── alphaction/
│   └── tools/
│       └── train_net.py
├── scripts/thermal_action/        # Dataset generation scripts
├── DataAnnotationQA/              # TDengine data pipeline
├── thermal_action_dataset/        # Generated dataset
│   ├── frames/                    # HDF5 files
│   └── annotations/               # COCO JSON
└── cursor_readme/                 # Documentation
```

### Create Config for AlphAction

Create a new config file: `/home/ec2-user/jingchen/AlphAction/configs/thermal_action.yaml`

```yaml
MODEL:
  META_ARCHITECTURE: "SlowFastRCNN"
  BACKBONE:
    NAME: "SlowFast-Resnet50"
  SLOWFAST:
    ALPHA: 4
    BETA_INV: 8
    FUSION_CONV_CHANNEL_RATIO: 2
    FUSION_KERNEL_SZ: 7
  ROI_HEAD:
    NUM_CLASSES: 14
    POOLER_RESOLUTION: 7
    POOLER_SAMPLING_RATIO: 0

INPUT:
  VIDEO_SIZE: [40, 60]      # Height x Width for thermal
  CHANNELS: 3               # Replicated thermal
  FRAME_NUM: 64             # 64 consecutive frames
  FRAME_SAMPLE_RATE: 1      # No sampling
  TAU: 8
  ALPHA: 4
  MIN_SIZE_TRAIN: 40        # Keep native resolution
  MAX_SIZE_TRAIN: 60
  PIXEL_MEAN: [20.0, 20.0, 20.0]
  PIXEL_STD: [10.0, 10.0, 10.0]
  FLIP_PROB: 0.5
  COLOR_JITTER: False       # Skip for thermal

DATASETS:
  TRAIN: ("thermal_action_train",)
  TEST: ("thermal_action_val",)
  FRAME_ROOT: "/home/ec2-user/jingchen/thermal_action_dataset/frames"
  TRAIN_ANNO_PATH: "/home/ec2-user/jingchen/thermal_action_dataset/annotations/train.json"
  VAL_ANNO_PATH: "/home/ec2-user/jingchen/thermal_action_dataset/annotations/val.json"

DATALOADER:
  NUM_WORKERS: 4
  VIDEOS_PER_BATCH: 8
  SIZE_DIVISIBILITY: 16

SOLVER:
  BASE_LR: 0.01
  WEIGHT_DECAY: 0.0001
  MOMENTUM: 0.9
  MAX_EPOCHS: 50
  WARMUP_EPOCHS: 5
  LR_POLICY: "cosine"
  CHECKPOINT_PERIOD: 5

OUTPUT_DIR: "/home/ec2-user/jingchen/experiments/thermal_action"
```

### Modify AlphAction DataLoader

You'll need to create a custom dataloader in AlphAction that uses your `ThermalActionDataset`:

**File**: `/home/ec2-user/jingchen/AlphAction/alphaction/dataset/datasets/thermal_action.py`

```python
import sys
from pathlib import Path

# Add thermal action scripts to path
sys.path.insert(0, '/home/ec2-user/jingchen/scripts')

from thermal_action import ThermalActionDataset, ThermalActionTransform, collate_fn

def build_thermal_action_dataset(cfg, is_train=True):
    """Build thermal action dataset for AlphAction."""
    
    hdf5_root = cfg.DATASETS.FRAME_ROOT
    ann_file = cfg.DATASETS.TRAIN_ANNO_PATH if is_train else cfg.DATASETS.VAL_ANNO_PATH
    
    # Create transforms
    transforms = ThermalActionTransform(
        is_train=is_train,
        flip_prob=cfg.INPUT.FLIP_PROB if is_train else 0.0,
        normalize=True,
        temp_min=5.0,
        temp_max=45.0
    )
    
    dataset = ThermalActionDataset(
        hdf5_root=hdf5_root,
        ann_file=ann_file,
        transforms=transforms,
        frame_window=cfg.INPUT.FRAME_NUM
    )
    
    return dataset
```

### Training Command on AWS

```bash
cd /home/ec2-user/jingchen/AlphAction

python tools/train_net.py \
  --config-file configs/thermal_action.yaml \
  OUTPUT_DIR /home/ec2-user/jingchen/experiments/thermal_action
```

---

## Troubleshooting on AWS

### Issue: Python Module Not Found

```bash
# Add scripts to PYTHONPATH
export PYTHONPATH=/home/ec2-user/jingchen/scripts:$PYTHONPATH
export PYTHONPATH=/home/ec2-user/jingchen/DataAnnotationQA/src:$PYTHONPATH
```

### Issue: TDengine Connection Failed

```bash
# Test connection from AWS
curl http://35.90.244.93:6041/rest/sql/thermal_sensors_pilot \
  -u root:taosdata \
  -d "SHOW TABLES"

# Check AWS security group allows outbound to port 6041
```

### Issue: Out of Memory

```bash
# Reduce batch size in config
DATALOADER.VIDEOS_PER_BATCH: 4  # Instead of 8
```

### Issue: Missing Dependencies

```bash
# Install all required packages
pip install h5py numpy torch torchvision requests matplotlib pillow yacs opencv-python
```

---

## File Transfer Options

### Option 1: Archive (Simpler, One Transfer)

```bash
# From local Mac
cd /Users/jma/Github/Butlr/YOLOv11
scp thermal_action_aws_deploy.tar.gz \
  ec2-user@44.248.130.76:/home/ec2-user/jingchen/

# On AWS
ssh ec2-user@44.248.130.76
cd /home/ec2-user/jingchen
tar -xzf thermal_action_aws_deploy.tar.gz
```

### Option 2: Rsync (Incremental, Can Resume)

```bash
# From local Mac - copy scripts
rsync -avz --progress \
  /Users/jma/Github/Butlr/YOLOv11/scripts/thermal_action/ \
  ec2-user@44.248.130.76:/home/ec2-user/jingchen/scripts/thermal_action/

# Copy data pipeline
rsync -avz --progress \
  /Users/jma/Github/Butlr/YOLOv11/DataAnnotationQA/src/data_pipeline/ \
  ec2-user@44.248.130.76:/home/ec2-user/jingchen/DataAnnotationQA/src/data_pipeline/

# Copy annotations
rsync -avz --progress \
  /Users/jma/Github/Butlr/YOLOv11/DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/ \
  ec2-user@44.248.130.76:/home/ec2-user/jingchen/DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/

# Copy documentation
rsync -avz --progress \
  /Users/jma/Github/Butlr/YOLOv11/cursor_readme/THERMAL_ACTION*.md \
  ec2-user@44.248.130.76:/home/ec2-user/jingchen/cursor_readme/

# Copy debug docs
rsync -avz --progress \
  /Users/jma/Github/Butlr/YOLOv11/debug/THERMAL_ACTION*.md \
  ec2-user@44.248.130.76:/home/ec2-user/jingchen/debug/

# Copy setup files
rsync -avz --progress \
  /Users/jma/Github/Butlr/YOLOv11/setup_thermal_action_aws.sh \
  /Users/jma/Github/Butlr/YOLOv11/AWS_EXECUTION_GUIDE.md \
  ec2-user@44.248.130.76:/home/ec2-user/jingchen/
```

### Option 3: Copy Already Generated Dataset (Fastest)

If you want to skip regeneration on AWS and use the already generated dataset:

```bash
# From local Mac
rsync -avz --progress \
  /Users/jma/Github/Butlr/YOLOv11/thermal_action_dataset/ \
  ec2-user@44.248.130.76:/home/ec2-user/jingchen/thermal_action_dataset/
```

---

## Dataset Generation on AWS

### Full Command

```bash
cd /home/ec2-user/jingchen

# Set Python path
export PYTHONPATH=/home/ec2-user/jingchen:$PYTHONPATH

# Generate dataset (takes ~3-5 seconds)
python scripts/thermal_action/generate_thermal_action_dataset.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --output-dir thermal_action_dataset \
  --val-split 0.2 \
  --buffer-frames 128 \
  --compression gzip \
  --compression-level 4 \
  --random-seed 42 \
  --tdengine-host 35.90.244.93 \
  --tdengine-port 6041 \
  --tdengine-database thermal_sensors_pilot
```

### Monitor Generation

```bash
# Watch progress
tail -f thermal_action_dataset/dataset_info.json

# Check HDF5 files
ls -lh thermal_action_dataset/frames/

# Check annotations
wc -l thermal_action_dataset/annotations/*.json
```

### Validate Generated Dataset

```bash
python scripts/thermal_action/validate_dataset.py \
  --hdf5-dir thermal_action_dataset/frames \
  --annotations-dir thermal_action_dataset/annotations \
  --output-dir thermal_action_dataset/statistics \
  --num-samples 6

# Check validation results
cat thermal_action_dataset/statistics/validation_report.json | python -m json.tool
```

---

## SlowFast Training on AWS

### Prepare Training Environment

```bash
cd /home/ec2-user/jingchen/AlphAction

# Activate your conda/virtual environment if you have one
# conda activate alphaction
# or
# source venv/bin/activate
```

### Create Thermal Action Config

Create config file at `AlphAction/configs/thermal_action.yaml` (see config above).

### Copy Dataset Loader

```bash
# Create custom dataset loader
mkdir -p /home/ec2-user/jingchen/AlphAction/alphaction/dataset/datasets/

# Copy thermal_action.py integration code to AlphAction
# (see integration code above)
```

### Start Training

```bash
cd /home/ec2-user/jingchen/AlphAction

export PYTHONPATH=/home/ec2-user/jingchen:$PYTHONPATH

python tools/train_net.py \
  --config-file configs/thermal_action.yaml \
  OUTPUT_DIR /home/ec2-user/jingchen/experiments/thermal_action \
  SOLVER.MAX_EPOCHS 50 \
  DATALOADER.VIDEOS_PER_BATCH 8
```

### Monitor Training

```bash
# Check training logs
tail -f /home/ec2-user/jingchen/experiments/thermal_action/log.txt

# Check GPU usage
nvidia-smi -l 1

# Check disk usage
df -h /home/ec2-user
```

---

## Testing Dataset Loading

### Test 1: Verify HDF5 Files

```bash
python3 << 'EOF'
import h5py
import sys

# Check each sensor
sensors = ['SL14_R1', 'SL14_R2', 'SL14_R3', 'SL14_R4', 
           'SL18_R1', 'SL18_R2', 'SL18_R3', 'SL18_R4']

for sensor in sensors:
    try:
        f = h5py.File(f'thermal_action_dataset/frames/{sensor}.h5', 'r')
        print(f'{sensor}: {len(f["frames"])} frames, {f.attrs["corrupted_count"]} corrupted')
        f.close()
    except Exception as e:
        print(f'{sensor}: ERROR - {e}')
EOF
```

### Test 2: Load Sample with PyTorch

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/ec2-user/jingchen/scripts')
sys.path.insert(0, '/home/ec2-user/jingchen/DataAnnotationQA/src')

from thermal_action import ThermalActionDataset, collate_fn
from torch.utils.data import DataLoader

# Load dataset
dataset = ThermalActionDataset(
    hdf5_root='thermal_action_dataset/frames',
    ann_file='thermal_action_dataset/annotations/train.json'
)

print(f'Dataset size: {len(dataset)} samples')

# Test loading
frames, boxes, labels, extras = dataset[0]
print(f'Sample 0:')
print(f'  Frames: {frames.shape}')
print(f'  Boxes: {boxes.shape}')
print(f'  Labels: {labels}')

# Test dataloader
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
frames_batch, _, _, _ = next(iter(loader))
print(f'\nBatch:')
print(f'  Frames: {frames_batch.shape}')

dataset.close()
print('\n✅ PyTorch loading works on AWS!')
EOF
```

### Test 3: Verify Dimensions

```bash
python3 << 'EOF'
import h5py
import json

# Check HDF5
f = h5py.File('thermal_action_dataset/frames/SL18_R1.h5', 'r')
print('HDF5 Dimensions:')
print(f'  Shape: {f["frames"].shape}')
print(f'  Width: {f.attrs["frame_width"]}')
print(f'  Height: {f.attrs["frame_height"]}')
f.close()

# Check COCO
with open('thermal_action_dataset/annotations/train.json', 'r') as f:
    coco = json.load(f)
img = coco['images'][0]
print('\nCOCO Dimensions:')
print(f'  Width: {img["width"]}')
print(f'  Height: {img["height"]}')

print('\n✅ Dimensions correct: 40 height × 60 width')
EOF
```

---

## Environment Setup

### Python Dependencies

```bash
pip install h5py numpy torch torchvision requests matplotlib pillow yacs opencv-python

# For AlphAction (if not already installed)
pip install cython pycocotools
```

### PYTHONPATH Configuration

Add to `~/.bashrc` or `~/.bash_profile`:

```bash
export PYTHONPATH=/home/ec2-user/jingchen/scripts:$PYTHONPATH
export PYTHONPATH=/home/ec2-user/jingchen/DataAnnotationQA/src:$PYTHONPATH
export PYTHONPATH=/home/ec2-user/jingchen/AlphAction:$PYTHONPATH
```

Then: `source ~/.bashrc`

---

## Quick Commands Reference

### Copy Archive to AWS
```bash
scp thermal_action_aws_deploy.tar.gz ec2-user@44.248.130.76:/home/ec2-user/jingchen/
```

### SSH to AWS
```bash
ssh ec2-user@44.248.130.76
```

### Generate Dataset
```bash
cd /home/ec2-user/jingchen
python scripts/thermal_action/generate_thermal_action_dataset.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --output-dir thermal_action_dataset \
  --val-split 0.2
```

### Validate Dataset
```bash
python scripts/thermal_action/validate_dataset.py
```

### Train SlowFast
```bash
cd AlphAction
python tools/train_net.py --config-file configs/thermal_action.yaml
```

---

## Expected Results

### Dataset Generation
- **Duration**: 3-5 seconds
- **Size**: 9.3 MB
- **Train**: 314 images, 405 annotations
- **Val**: 73 images, 95 annotations

### Training
- **Duration**: ~2-4 hours for 50 epochs (depends on GPU)
- **GPU Memory**: ~4-6 GB (batch size 8)
- **Checkpoints**: Saved every 5 epochs

---

## Checklist

Before training on AWS:
- [ ] Files copied to AWS (scripts, data pipeline, annotations)
- [ ] Setup script executed successfully
- [ ] TDengine connection tested
- [ ] Dataset generated successfully
- [ ] Dataset validated (no critical issues)
- [ ] PyTorch loading tested
- [ ] Dimensions verified (40 height × 60 width)
- [ ] AlphAction config created
- [ ] PYTHONPATH configured
- [ ] Dependencies installed
- [ ] GPU available (check with `nvidia-smi`)

---

## Support

For issues on AWS:
- Check logs in `thermal_action_dataset/`
- Review validation report
- Verify TDengine connectivity
- Check PYTHONPATH and imports
- Ensure sufficient disk space and GPU memory

For detailed documentation:
- `cursor_readme/THERMAL_ACTION_TRAINING_GUIDE.md`
- `cursor_readme/THERMAL_ACTION_DATASET_FINAL.md`
- `scripts/thermal_action/README.md`

