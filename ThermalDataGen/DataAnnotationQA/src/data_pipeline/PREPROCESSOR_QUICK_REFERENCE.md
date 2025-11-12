# ThermalFramePreprocessor - Quick Reference

**One-stop preprocessing for thermal frames with automatic Kelvin/Celsius handling**

---

## 🚀 Quick Start

```python
from data_pipeline.thermal_preprocessor import ThermalFramePreprocessor

# 1. Create preprocessor
preprocessor = ThermalFramePreprocessor()

# 2. Process frame (auto-detects Kelvin/Celsius, converts, validates)
frame_celsius = preprocessor.process_frame(raw_frame)

# 3. Convert to image
img_8bit = preprocessor.thermal_to_image(frame_celsius)
```

---

## 📋 API Reference

### Constructor

```python
ThermalFramePreprocessor(
    window_min=5.0,           # Window minimum (°C)
    window_max=45.0,          # Window maximum (°C)
    expected_min=5.0,         # Expected min for indoor
    expected_max=45.0,        # Expected max for indoor
    absolute_min=-40.0,       # Absolute minimum valid
    absolute_max=80.0,        # Absolute maximum valid
    auto_convert=True,        # Auto-convert Kelvin to Celsius
    kelvin_threshold=200.0    # Mean temp threshold for Kelvin detection
)
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `process_frame(frame, validate=True)` | `np.ndarray` | **Main API**: Detect format, convert, validate |
| `thermal_to_image(frame, window_min=None, window_max=None)` | `np.ndarray` | Convert to 8-bit image with windowing |
| `check_temperature_format(frame)` | `str` | Detect 'kelvin' or 'celsius' |
| `kelvin_to_celsius(frame)` | `np.ndarray` | Convert Kelvin to Celsius |
| `validate_temperature_range(frame, raise_on_error=True)` | `bool` | Validate temperature range |
| `get_statistics(frame)` | `dict` | Get min/max/mean/std/format |

---

## 💡 Common Use Cases

### Use Case 1: Default Processing

```python
preprocessor = ThermalFramePreprocessor()
frame = preprocessor.process_frame(raw_frame)  # Auto-everything!
img = preprocessor.thermal_to_image(frame)
```

### Use Case 2: Custom Windowing

```python
# For warmer environments
preprocessor = ThermalFramePreprocessor(
    window_min=15.0,
    window_max=60.0
)
```

### Use Case 3: Manual Format Check

```python
format_type = preprocessor.check_temperature_format(frame)
print(f"Frame is in {format_type}")

if format_type == 'kelvin':
    frame = preprocessor.kelvin_to_celsius(frame)
```

### Use Case 4: Validation Only

```python
is_valid = preprocessor.validate_temperature_range(
    frame, 
    raise_on_error=False
)
if is_valid:
    # Process frame
    pass
```

### Use Case 5: Get Frame Info

```python
stats = preprocessor.get_statistics(frame)
print(f"Temperature: {stats['min']:.1f} to {stats['max']:.1f}°C")
print(f"Mean: {stats['mean']:.1f}°C")
print(f"Format: {stats['format']}")
```

---

## 🎯 Format Detection

The preprocessor automatically detects Kelvin vs Celsius:

| Indicator | Kelvin | Celsius |
|-----------|--------|---------|
| **Mean temp** | > 200 | < 200 |
| **Typical range** | 273-323 | 0-50 |
| **Example** | 293.15 K (20°C) | 20.0°C |

**Detection logic**:
- If `mean > 200`: Detected as **Kelvin**
- If `mean < 200`: Detected as **Celsius**
- If ambiguous: Assumes **Celsius** with warning

---

## ⚠️ Validation Rules

### Critical Errors (Raises exception)

1. **Below absolute zero**: `temp < -273°C`
2. **Still in Kelvin**: `mean > 200°C` after processing
3. **Corrupted pixels**: Any pixel `== -273.15°C`

### Warnings (Logged only)

1. **Below absolute min**: `temp < -40°C`
2. **Above absolute max**: `temp > 80°C`

### Info Messages (Logged only)

1. **Outside expected**: `temp < 5°C` or `temp > 45°C`

---

## 🎨 Windowing

**Purpose**: Map temperature range to 0-255 grayscale for better contrast

**Default**: 5°C to 45°C (indoor scenes)

**Mapping**:
```
5°C  → pixel value 0   (black)
25°C → pixel value 128 (mid-gray)
45°C → pixel value 255 (white)

< 5°C  → clipped to 0
> 45°C → clipped to 255
```

**Custom windowing**:
```python
# Per-preprocessor (affects all images)
preprocessor = ThermalFramePreprocessor(window_min=10.0, window_max=50.0)

# Per-image (one-time override)
img = preprocessor.thermal_to_image(frame, window_min=10.0, window_max=50.0)
```

---

## 🔗 Integration

### With TDengineConnector

```python
from data_pipeline.thermal_dataset import TDengineConnector
from data_pipeline.thermal_preprocessor import ThermalFramePreprocessor

# Create custom preprocessor
preprocessor = ThermalFramePreprocessor(window_min=10.0, window_max=50.0)

# Pass to connector
connector = TDengineConnector(preprocessor=preprocessor)

# Frames are now preprocessed automatically!
frame = connector.query_frame_by_timestamp(mac, timestamp)
```

### With Dataset Generator

```python
# The generator creates its own preprocessor automatically
generator = YOLODatasetGenerator(output_dir="datasets/my_dataset")

# Or provide custom settings via tdengine_config
# (preprocessor is created internally with default settings)
```

---

## 🐛 Troubleshooting

### Issue: "Data appears to be in Kelvin"

**Cause**: Frame has mean > 200°C  
**Solution**: Enable auto_convert:
```python
preprocessor = ThermalFramePreprocessor(auto_convert=True)
```

### Issue: "Invalid temperature: -273.20°C"

**Cause**: Corrupted frame with absolute zero pixels  
**Solution**: This is correct behavior - the frame is invalid and should be skipped

### Issue: "Ambiguous temperature format"

**Cause**: Mean temperature around threshold (~200°C) or mixed data  
**Solution**: Check your data source. Preprocessor will assume Celsius.

### Issue: Images too dark/bright

**Cause**: Wrong windowing range for your environment  
**Solution**: Adjust window parameters:
```python
# For warmer scenes
preprocessor = ThermalFramePreprocessor(window_min=15.0, window_max=60.0)

# For colder scenes
preprocessor = ThermalFramePreprocessor(window_min=0.0, window_max=35.0)
```

---

## 📊 Default Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `window_min` | 5.0°C | Window minimum |
| `window_max` | 45.0°C | Window maximum |
| `expected_min` | 5.0°C | Expected minimum (indoor) |
| `expected_max` | 45.0°C | Expected maximum (indoor) |
| `absolute_min` | -40.0°C | Absolute minimum valid |
| `absolute_max` | 80.0°C | Absolute maximum valid |
| `auto_convert` | True | Auto Kelvin→Celsius |
| `kelvin_threshold` | 200.0°C | Kelvin detection threshold |

---

## 🎓 Best Practices

### 1. Use Default Settings for Indoor

```python
# Default (5-45°C) optimized for indoor thermal imaging
preprocessor = ThermalFramePreprocessor()
```

### 2. Customize for Your Environment

```python
# Outdoor winter: -10 to 30°C
preprocessor = ThermalFramePreprocessor(window_min=-10.0, window_max=30.0)

# Industrial/hot: 20 to 100°C
preprocessor = ThermalFramePreprocessor(window_min=20.0, window_max=100.0)
```

### 3. Always Use process_frame() First

```python
# ✅ Good: Process first (handles format + validation)
frame = preprocessor.process_frame(raw_frame)
img = preprocessor.thermal_to_image(frame)

# ❌ Bad: Convert directly without processing
img = preprocessor.thermal_to_image(raw_frame)  # May fail if Kelvin!
```

### 4. Handle Validation Errors

```python
try:
    frame = preprocessor.process_frame(raw_frame)
except ValueError as e:
    print(f"Invalid frame: {e}")
    # Skip this frame or use default
```

### 5. Check Statistics for Debugging

```python
stats = preprocessor.get_statistics(raw_frame)
if stats['format'] == 'kelvin':
    print("Warning: Data is in Kelvin - will be converted")
```

---

## 📚 See Also

- **Full Documentation**: `REFACTORING_SUMMARY.md`
- **Source Code**: `thermal_preprocessor.py`
- **Usage Examples**: `REFACTORING_SUMMARY.md` (Section: Usage Examples)
- **Integration Guide**: `REFACTORING_SUMMARY.md` (Section: Files Modified)

---

**Quick Reference v1.0 - November 7, 2025**

