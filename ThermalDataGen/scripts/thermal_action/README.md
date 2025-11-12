# Thermal Action Detection Dataset

Tools for creating and using thermal sensor datasets for human action detection with SlowFast models.

## Quick Start

### 1. Generate Dataset

Generate the complete dataset from TDengine:

```bash
python scripts/thermal_action/generate_thermal_action_dataset.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --output-dir thermal_action_dataset \
  --val-split 0.2
```

**Output**: HDF5 frame storage + COCO annotations ready for training

### 2. Validate Dataset

Verify dataset integrity and visualize samples:

```bash
python scripts/thermal_action/validate_dataset.py \
  --hdf5-dir thermal_action_dataset/frames \
  --annotations-dir thermal_action_dataset/annotations \
  --num-samples 6
```

**Output**: Validation report + sample visualizations

### 3. Use in Training

```python
from scripts.thermal_action import ThermalActionDataset, ThermalActionTransform, collate_fn
from torch.utils.data import DataLoader

# Create dataset
dataset = ThermalActionDataset(
    hdf5_root='thermal_action_dataset/frames',
    ann_file='thermal_action_dataset/annotations/train.json',
    transforms=ThermalActionTransform(is_train=True)
)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

# Iterate
for frames, boxes, labels, extras in loader:
    # frames: [B, 3, 64, 40, 60] - 64 frames, 40 height × 60 width
    # boxes: List of [N_i, 4] tensors
    # labels: List of [N_i] tensors (action class IDs 0-13)
    pass
```

## Key Features

- **HDF5 Storage**: Efficient sequential access, ~10x storage savings
- **64 Consecutive Frames**: No temporal gaps (FRAME_SAMPLE_RATE=1)
- **COCO Format**: AVA-compatible annotations
- **14 Action Classes**: Person subcategories from original annotations
- **Corruption Handling**: Replaces corrupted frames with zeros
- **128-Frame Buffer**: Ensures temporal context at boundaries

## Directory Structure

```
thermal_action_dataset/
├── frames/
│   ├── SL14_R1.h5          # Chronological frames per sensor
│   ├── SL14_R2.h5
│   └── sensor_info.json    # Sensor metadata
├── annotations/
│   ├── train.json          # Training annotations
│   ├── val.json            # Validation annotations
│   └── class_mapping.json  # Action class names
└── statistics/
    ├── dataset_stats.json
    └── validation_report.json
```

## Action Classes

| ID | Class |
|----|-------|
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

## Individual Modules

### create_hdf5_frames.py

Fetch frames from TDengine and store chronologically in HDF5.

```bash
python scripts/thermal_action/create_hdf5_frames.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --output-dir thermal_action_dataset/frames \
  --buffer-frames 128 \
  --tdengine-host localhost \
  --tdengine-port 6041
```

**Features**:
- Queries TDengine using `ts` TIMESTAMP and `frame_seq` BIGINT
- Fetches 128 frames buffer before/after annotations
- Replaces corrupted frames with all-zero arrays
- Stores with gzip compression (level 4)

### convert_annotations_to_coco.py

Convert YOLO-style annotations to COCO format.

```bash
python scripts/thermal_action/convert_annotations_to_coco.py \
  --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \
  --hdf5-dir thermal_action_dataset/frames \
  --output-dir thermal_action_dataset/annotations \
  --val-split 0.2 \
  --random-seed 42
```

**Features**:
- Filters to only "person" category with action subcategories
- Creates 80/20 train/val split (stratified by sensor)
- Validates temporal coverage (±32 frames)
- Generates class distribution statistics

### thermal_action_dataset.py

PyTorch Dataset for loading 64-frame sequences.

```python
from scripts.thermal_action import ThermalActionDataset

dataset = ThermalActionDataset(
    hdf5_root='thermal_action_dataset/frames',
    ann_file='thermal_action_dataset/annotations/train.json',
    transforms=None,  # Optional
    frame_window=64
)

frames, boxes, labels, extras = dataset[0]
# frames: [64, 40, 60, 3] - Thermal replicated to 3 channels (64 frames, 40 height, 60 width)
# boxes: [N, 4] - Normalized bboxes [centerX, centerY, width, height]
# labels: [N] - Action class IDs (0-13)
```

**Features**:
- Keeps HDF5 files open for fast slicing
- Loads 64 consecutive frames (32 before + keyframe + 31 after)
- Replicates thermal to 3 channels (R=G=B)
- Optional transforms (flip, normalize)

### validate_dataset.py

Validate dataset integrity and visualize samples.

```bash
python scripts/thermal_action/validate_dataset.py \
  --hdf5-dir thermal_action_dataset/frames \
  --annotations-dir thermal_action_dataset/annotations \
  --output-dir thermal_action_dataset/statistics \
  --num-samples 6
```

**Checks**:
- Temporal coverage (±32 frames exist)
- Annotation validity (bbox format, class IDs)
- Frame data integrity (shape, NaN/Inf, temperature range)
- Generates visualizations with overlaid bboxes

## Configuration

### TDengine Connection

```python
tdengine_config = {
    'host': '35.90.244.93',  # Remote server (default)
    'port': 6041,
    'database': 'thermal_sensors_pilot',
    'user': 'root',
    'password': 'taosdata'
}
```

### HDF5 Compression

```python
# Trade-off: compression level vs. speed
compression='gzip'
compression_level=4  # 0 (fast) to 9 (best compression)
```

### Dataset Split

```python
val_split=0.2  # 80% train, 20% val
random_seed=42  # For reproducibility
```

## Expected Performance

- **Storage**: ~10x reduction vs. storing duplicate 64-frame sequences
- **Loading Speed**: ~1ms per 64-frame read from HDF5
- **Memory**: Only load needed frames, not entire video
- **Scalability**: Easy to add new sensors (just add new .h5 file)

## Troubleshooting

### Issue: TDengine Connection Failed

**Solution**: Verify TDengine server is accessible

```bash
# Test REST API (remote server)
curl http://35.90.244.93:6041/rest/sql/thermal_sensors_pilot \
  -u root:taosdata \
  -d "SHOW TABLES"

# Check specific sensor
curl http://35.90.244.93:6041/rest/sql/thermal_sensors_pilot \
  -u root:taosdata \
  -d "SELECT COUNT(*) FROM sensor_02_00_1a_62_51_67"
```

### Issue: Corrupted Frames

**Solution**: Corrupted frames are automatically replaced with zeros. Check `corrupted_count` in `sensor_info.json`.

### Issue: Timestamp Not Found

**Solution**: Increase `--buffer-frames` or check annotation timestamps match TDengine data.

### Issue: Out of Memory

**Solution**: Reduce batch size in DataLoader or close HDF5 files when not in use.

## Documentation

For detailed training guide, see:
- [THERMAL_ACTION_TRAINING_GUIDE.md](../../cursor_readme/THERMAL_ACTION_TRAINING_GUIDE.md)

## Requirements

```
h5py>=3.0.0
numpy>=1.20.0
torch>=1.10.0
matplotlib>=3.3.0
Pillow>=8.0.0
requests>=2.25.0
```

## License

Same as parent project.

