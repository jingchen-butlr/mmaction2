# Thermal Action Detection - AWS ThermalDataGen

**Location**: `/home/ec2-user/jingchen/AlphAction/ThermalDataGen/`  
**Status**: ✅ Ready for SlowFast Training

---

## Directory Structure

```
ThermalDataGen/
├── scripts/thermal_action/          # Dataset generation scripts
│   ├── create_hdf5_frames.py
│   ├── convert_annotations_to_coco.py
│   ├── thermal_action_dataset.py
│   ├── validate_dataset.py
│   ├── generate_thermal_action_dataset.py
│   ├── __init__.py
│   ├── README.md
│   └── QUICK_COMMANDS.md
├── DataAnnotationQA/                # TDengine data pipeline + annotations
│   ├── src/data_pipeline/          # TDengine connector
│   └── Data/.../Annotations/       # 8 annotation JSON files (387 annotations)
├── thermal_action_dataset/          # Generated dataset (9 MB)
│   ├── frames/                     # 8 HDF5 files (3,976 frames)
│   ├── annotations/                # train.json, val.json
│   └── statistics/                 # Validation reports, visualizations
├── cursor_readme/                   # Complete documentation
│   ├── THERMAL_ACTION_TRAINING_GUIDE.md
│   ├── THERMAL_ACTION_IMPLEMENTATION_SUMMARY.md
│   └── THERMAL_ACTION_DATASET_FINAL.md
├── debug/                           # Debug documentation
├── setup_thermal_action_aws.sh     # Setup script
├── AWS_EXECUTION_GUIDE.md          # AWS execution guide
└── thermal_action_requirements.txt # Python dependencies
```

---

## Quick Commands

### Set Environment

```bash
cd /home/ec2-user/jingchen/AlphAction/ThermalDataGen
export PYTHONPATH=/home/ec2-user/jingchen/AlphAction/ThermalDataGen:$PYTHONPATH
```

### Regenerate Dataset

```bash
python scripts/thermal_action/generate_thermal_action_dataset.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --output-dir thermal_action_dataset \
  --val-split 0.2
```

### Validate Dataset

```bash
python scripts/thermal_action/validate_dataset.py \
  --hdf5-dir thermal_action_dataset/frames \
  --annotations-dir thermal_action_dataset/annotations
```

### Test PyTorch Loading

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/ec2-user/jingchen/AlphAction/ThermalDataGen/scripts')
sys.path.insert(0, '/home/ec2-user/jingchen/AlphAction/ThermalDataGen/DataAnnotationQA/src')

from thermal_action import ThermalActionDataset
ds = ThermalActionDataset(
    hdf5_root='thermal_action_dataset/frames',
    ann_file='thermal_action_dataset/annotations/train.json'
)
print(f'Dataset: {len(ds)} samples')
frames, boxes, labels, extras = ds[0]
print(f'Frames: {frames.shape}')
ds.close()
EOF
```

---

## Dataset Stats

- **Total sensors**: 8
- **Total frames**: 3,976
- **Storage**: 9.0 MB (HDF5 + gzip)
- **Train**: 314 images (405 annotations)
- **Val**: 73 images (95 annotations)
- **Action classes**: 14
- **Dimensions**: 40 height × 60 width (40 rows × 60 columns)
- **Model input**: [B, 3, 64, 40, 60]

---

## Integration with AlphAction

### Update Config Path

In your SlowFast training config, use these paths:

```yaml
DATASETS:
  FRAME_ROOT: "/home/ec2-user/jingchen/AlphAction/ThermalDataGen/thermal_action_dataset/frames"
  TRAIN_ANNO_PATH: "/home/ec2-user/jingchen/AlphAction/ThermalDataGen/thermal_action_dataset/annotations/train.json"
  VAL_ANNO_PATH: "/home/ec2-user/jingchen/AlphAction/ThermalDataGen/thermal_action_dataset/annotations/val.json"

INPUT:
  VIDEO_SIZE: [40, 60]  # Height x Width
  FRAME_NUM: 64
  FRAME_SAMPLE_RATE: 1

MODEL:
  NUM_CLASSES: 14
```

### Add to PYTHONPATH

Add to your `~/.bashrc`:

```bash
export PYTHONPATH=/home/ec2-user/jingchen/AlphAction/ThermalDataGen:$PYTHONPATH
```

---

## Documentation

- **AWS_EXECUTION_GUIDE.md** - Complete AWS execution guide
- **cursor_readme/THERMAL_ACTION_TRAINING_GUIDE.md** - Training guide
- **scripts/thermal_action/README.md** - Module documentation
- **scripts/thermal_action/QUICK_COMMANDS.md** - Command reference

---

## Troubleshooting

### Module Not Found

```bash
export PYTHONPATH=/home/ec2-user/jingchen/AlphAction/ThermalDataGen:$PYTHONPATH
export PYTHONPATH=/home/ec2-user/jingchen/AlphAction/ThermalDataGen/scripts:$PYTHONPATH
export PYTHONPATH=/home/ec2-user/jingchen/AlphAction/ThermalDataGen/DataAnnotationQA/src:$PYTHONPATH
```

### Regenerate Dataset

```bash
cd /home/ec2-user/jingchen/AlphAction/ThermalDataGen
rm -rf thermal_action_dataset
python scripts/thermal_action/generate_thermal_action_dataset.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --output-dir thermal_action_dataset \
  --val-split 0.2
```

---

**Status**: ✅ Ready for SlowFast Training

All files organized in AlphAction/ThermalDataGen/
