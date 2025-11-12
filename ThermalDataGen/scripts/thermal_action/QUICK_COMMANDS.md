# Thermal Action Detection - Quick Commands

## Complete Dataset Generation (One Command)

```bash
cd /Users/jma/Github/Butlr/YOLOv11

python scripts/thermal_action/generate_thermal_action_dataset.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --output-dir thermal_action_dataset \
  --val-split 0.2 \
  --buffer-frames 128 \
  --compression gzip \
  --compression-level 4 \
  --random-seed 42
```

**Time**: ~5-10 minutes depending on TDengine performance  
**Output**: `thermal_action_dataset/` with frames (HDF5) + annotations (COCO JSON)

---

## Step-by-Step Generation

### Step 1: Create HDF5 Frame Storage

```bash
python scripts/thermal_action/create_hdf5_frames.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --output-dir thermal_action_dataset/frames \
  --buffer-frames 128 \
  --tdengine-host localhost \
  --tdengine-port 6041 \
  --tdengine-database thermal_sensors_pilot
```

### Step 2: Convert Annotations to COCO

```bash
python scripts/thermal_action/convert_annotations_to_coco.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --hdf5-dir thermal_action_dataset/frames \
  --output-dir thermal_action_dataset/annotations \
  --val-split 0.2 \
  --random-seed 42
```

### Step 3: Validate Dataset

```bash
python scripts/thermal_action/validate_dataset.py \
  --hdf5-dir thermal_action_dataset/frames \
  --annotations-dir thermal_action_dataset/annotations \
  --output-dir thermal_action_dataset/statistics \
  --num-samples 6
```

---

## Custom Configurations

### Different Train/Val Split

```bash
# 70% train, 30% val
python scripts/thermal_action/generate_thermal_action_dataset.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --val-split 0.3
```

### Custom TDengine Server

```bash
# By default, uses remote server at 35.90.244.93
# To use a different server:
python scripts/thermal_action/generate_thermal_action_dataset.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --tdengine-host YOUR_HOST \
  --tdengine-port 6041
```

### Higher Compression (Slower, Smaller Files)

```bash
python scripts/thermal_action/generate_thermal_action_dataset.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --compression gzip \
  --compression-level 9
```

### More Buffer Frames (For Boundary Cases)

```bash
python scripts/thermal_action/generate_thermal_action_dataset.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --buffer-frames 256
```

---

## Testing & Validation

### Quick Test (Load First Sample)

```bash
python -c "
from scripts.thermal_action import ThermalActionDataset
dataset = ThermalActionDataset(
    hdf5_root='thermal_action_dataset/frames',
    ann_file='thermal_action_dataset/annotations/train.json'
)
print(f'Dataset size: {len(dataset)}')
if len(dataset) > 0:
    frames, boxes, labels, extras = dataset[0]
    print(f'Frames shape: {frames.shape}')
    print(f'Boxes shape: {boxes.shape}')
    print(f'Labels: {labels}')
dataset.close()
"
```

### Validate Specific Sensor

```bash
python -c "
import h5py
f = h5py.File('thermal_action_dataset/frames/SL18_R1.h5', 'r')
print('Sensor: SL18_R1')
print(f'Total frames: {len(f[\"frames\"])}')
print(f'Timestamps: {f[\"timestamps\"][0]} to {f[\"timestamps\"][-1]}')
print(f'Corrupted: {f.attrs[\"corrupted_count\"]}')
f.close()
"
```

### Check Dataset Statistics

```bash
cat thermal_action_dataset/dataset_info.json
cat thermal_action_dataset/statistics/dataset_stats.json
```

---

## PyTorch DataLoader Testing

```bash
python -c "
from scripts.thermal_action import ThermalActionDataset, ThermalActionTransform, collate_fn
from torch.utils.data import DataLoader

dataset = ThermalActionDataset(
    hdf5_root='thermal_action_dataset/frames',
    ann_file='thermal_action_dataset/annotations/train.json',
    transforms=ThermalActionTransform(is_train=True)
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)

print(f'Dataset size: {len(dataset)}')
print(f'Number of batches: {len(loader)}')

# Test first batch
frames, boxes, labels, extras = next(iter(loader))
print(f'\\nFirst batch:')
print(f'  Frames: {frames.shape}')
print(f'  Batch size: {len(boxes)}')
print(f'  Samples in batch: {[len(b) for b in boxes]}')

dataset.close()
"
```

---

## Regenerate Specific Components

### Only HDF5 (Skip if already exists)

```bash
python scripts/thermal_action/generate_thermal_action_dataset.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --skip-annotations
```

### Only Annotations (HDF5 must exist)

```bash
python scripts/thermal_action/generate_thermal_action_dataset.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --skip-hdf5
```

---

## Cleanup

```bash
# Remove entire dataset
rm -rf thermal_action_dataset/

# Remove only HDF5 files (keep annotations)
rm -rf thermal_action_dataset/frames/*.h5

# Remove only annotations (keep HDF5)
rm -rf thermal_action_dataset/annotations/*.json
```

---

## Troubleshooting Commands

### Check TDengine Connection

```bash
# Remote server (default)
curl http://35.90.244.93:6041/rest/sql/thermal_sensors_pilot \
  -u root:taosdata \
  -d "SHOW TABLES"

# Or if using localhost
curl http://localhost:6041/rest/sql/thermal_sensors_pilot \
  -u root:taosdata \
  -d "SHOW TABLES"
```

### Check Specific Sensor Table

```bash
curl http://35.90.244.93:6041/rest/sql/thermal_sensors_pilot \
  -u root:taosdata \
  -d "SELECT COUNT(*) FROM sensor_02_00_1a_62_51_67"
```

### List All Generated Files

```bash
tree thermal_action_dataset/
```

### Check File Sizes

```bash
du -sh thermal_action_dataset/
du -sh thermal_action_dataset/frames/
ls -lh thermal_action_dataset/frames/*.h5
```

---

## Expected Output Structure

```
thermal_action_dataset/
├── frames/
│   ├── SL14_R1.h5          (~50-100 MB)
│   ├── SL14_R2.h5          (~50-100 MB)
│   ├── SL14_R3.h5          (~50-100 MB)
│   ├── SL14_R4.h5          (~50-100 MB)
│   ├── SL18_R1.h5          (~50-100 MB)
│   ├── SL18_R2.h5          (~50-100 MB)
│   ├── SL18_R3.h5          (~50-100 MB)
│   ├── SL18_R4.h5          (~50-100 MB)
│   └── sensor_info.json
├── annotations/
│   ├── train.json          (~300 images)
│   ├── val.json            (~75 images)
│   └── class_mapping.json
├── statistics/
│   ├── dataset_stats.json
│   ├── validation_report.json
│   └── visualizations/
│       ├── sample_000_SL18_R1.png
│       ├── sample_001_SL18_R2.png
│       └── ...
└── dataset_info.json
```

---

## Next Steps After Generation

1. **Validate**: `python scripts/thermal_action/validate_dataset.py`
2. **Test Loading**: See "PyTorch DataLoader Testing" above
3. **Train Model**: Refer to `cursor_readme/THERMAL_ACTION_TRAINING_GUIDE.md`
4. **Monitor**: Check `thermal_action_dataset/dataset_info.json` for statistics

