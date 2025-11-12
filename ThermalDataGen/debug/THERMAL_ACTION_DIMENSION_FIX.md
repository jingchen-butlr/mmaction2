# Thermal Action Detection Dataset - Dimension Fix

**Date**: November 12, 2025  
**Status**: ✅ **FIXED AND VERIFIED**

---

## Issue

Dimension confusion between width/height specifications:
- User specified: **40 height × 60 width** (40 rows × 60 columns)
- Initial COCO annotations: width=40, height=60 (WRONG)
- HDF5 files: shape (N, 40, 60) (CORRECT)

---

## Root Cause

Confusion between numpy array indexing and image dimensions:
- **Numpy shape**: `(height, width)` = `(rows, columns)`
- **Image dimensions**: "width × height" often written as "columns × rows"
- **User specification**: "40 height × 60 width" = numpy shape (40, 60)

The HDF5 files were already storing data correctly as `(N, 40, 60)`, but the COCO annotations had the dimensions swapped.

---

## Fix Applied

### 1. COCO Annotation Converter

**File**: `scripts/thermal_action/convert_annotations_to_coco.py`

```python
# BEFORE (WRONG):
images.append({
    'width': 40,
    'height': 60,
    ...
})

# AFTER (CORRECT):
images.append({
    'width': 60,  # Columns
    'height': 40,  # Rows
    ...
})
```

### 2. HDF5 Metadata

**File**: `scripts/thermal_action/create_hdf5_frames.py`

```python
# BEFORE (WRONG):
f.attrs['frame_width'] = 40
f.attrs['frame_height'] = 60

# AFTER (CORRECT):
f.attrs['frame_width'] = 60  # Columns
f.attrs['frame_height'] = 40  # Rows
```

### 3. Documentation Comments

Updated all comments to clarify:
- "40 height × 60 width (40 rows × 60 columns)"
- Consistent numpy shape notation: `(40, 60)` = `(height, width)`

---

## Verification

### HDF5 Files

```python
import h5py
f = h5py.File('thermal_action_dataset/frames/SL14_R1.h5', 'r')

# Shape: (498, 40, 60) = (N, height, width) ✅
# Attributes: width=60, height=40 ✅
# Single frame: (40, 60) = (rows, columns) ✅
```

### COCO Annotations

```json
{
  "images": [{
    "width": 60,
    "height": 40
  }]
}
```
✅ Correct

### PyTorch Loading

```python
from scripts.thermal_action import ThermalActionDataset, collate_fn
from torch.utils.data import DataLoader

ds = ThermalActionDataset(...)
frames, boxes, labels, extras = ds[0]
# frames.shape: torch.Size([64, 40, 60, 3]) ✅

loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
frames_batch, _, _, _ = next(iter(loader))
# frames_batch.shape: torch.Size([4, 3, 64, 40, 60]) ✅
```

### Visualization

Bounding boxes correctly aligned with thermal signatures in `thermal_action_dataset/statistics/visualizations/` ✅

---

## Final Dataset Stats

### Generation
- Duration: 3.4 seconds
- Total sensors: 8
- Total frames: 3,976
- Storage: 9.3 MB (HDF5 + gzip compression)
- Corrupted frames: 0

### Annotations
- Train images: 314 (81.1%)
- Val images: 73 (18.9%)
- Train annotations: 405 person actions
- Val annotations: 95 person actions
- Action classes: 14

### Validation
- ✅ Temporal coverage: 0 issues
- ✅ Annotation format: 0 issues
- ⚠️ Frame data: 10 issues (corrupted pixels in ~2.6% of samples)

---

## Dimension Reference

### Thermal Sensor Specifications

- **Sensor resolution**: 40 rows × 60 columns
- **Numpy shape**: `(40, 60)` where first dimension is height, second is width
- **Image dimensions**: 60 width × 40 height
- **Total pixels**: 2,400

### Model Input Format

- **Single sample**: `[64, 40, 60, 3]`
  - 64 frames (temporal dimension)
  - 40 height (rows)
  - 60 width (columns)
  - 3 channels (R=G=B, replicated thermal)

- **Batch**: `[B, 3, 64, 40, 60]`
  - B batch size
  - 3 channels (RGB-like)
  - 64 frames
  - 40 height
  - 60 width

### Bounding Box Format

- **COCO format**: `[centerX, centerY, width, height]`
- **Normalized**: [0, 1] range
- **Canvas**: (60 width, 40 height)
- **Pixel coordinates**: Multiply by (60, 40) to get actual positions

---

## Status

✅ **ALL FIXED**

- Dimensions consistent across all components
- Dataset regenerated successfully
- PyTorch loading verified
- Visualizations confirmed correct bbox alignment
- Ready for SlowFast training

---

**Next Step**: Integrate with SlowFast model and start training

