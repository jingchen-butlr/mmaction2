# MMAction2 SlowFast Finetuning Documentation

This folder contains comprehensive documentation and tools for finetuning SlowFast models on custom datasets using MMAction2.

## 🔥 **NEW: Thermal Action Recognition Integration**

Complete integration for finetuning SlowFast on your thermal sensor dataset! 

**Quick Start**: [`THERMAL_INTEGRATION_SUMMARY.md`](THERMAL_INTEGRATION_SUMMARY.md) (⭐ Start here!)

**Detailed Guide**: [`THERMAL_SLOWFAST_FINETUNING_GUIDE.md`](THERMAL_SLOWFAST_FINETUNING_GUIDE.md)

**Your Dataset**: 
- 314 train / 73 val samples
- 40×60 thermal frames (resized to 256×384)
- 14 action classes
- HDF5 format

**Status**: ✅ Ready to train!

```bash
# One command to start training:
bash tools/thermal_quickstart.sh
```

---

## 📚 General Documentation Files

### 1. [SlowFast Finetuning Guide](slowfast_finetuning_guide.md) ⭐
**Comprehensive guide covering everything you need to know**

Topics covered:
- MMAction2 architecture overview
- SlowFast model architecture details
- Complete data preparation workflow
- Configuration system explained
- Training and testing procedures
- Advanced hyperparameter tuning
- Common issues and solutions
- Performance benchmarks

**Who should read:** Everyone starting with SlowFast finetuning

---

### 2. [Quick Start Checklist](quickstart_checklist.md) 🚀
**TL;DR version for experienced users**

Quick reference for:
- Pre-training checklist
- Training commands
- Testing commands
- Common configuration adjustments
- Troubleshooting quick fixes
- Expected results timeline

**Who should read:** Those who want to get started quickly

---

### 3. [Example Config File](slowfast_custom_example.py) ⚙️
**Ready-to-use configuration template**

Features:
- Fully annotated config file
- All parameters explained with comments
- TODOs marked for customization
- Alternative configurations provided
- Memory optimization tips included
- Training commands in comments

**How to use:**
```bash
# Copy to configs directory
cp slowfast_custom_example.py ../configs/recognition/slowfast/my_dataset.py

# Edit the TODOs
vim ../configs/recognition/slowfast/my_dataset.py

# Start training
python tools/train.py configs/recognition/slowfast/my_dataset.py
```

---

### 4. [Dataset Preparation Script](prepare_custom_dataset.py) 🛠️
**Automated tool to organize your dataset**

Capabilities:
- Automatic video file discovery
- Label extraction from filenames or CSV
- Train/val splitting (stratified or random)
- Class folder organization
- Annotation file generation
- Class index mapping

**Usage examples:**

```bash
# Basic usage (labels from filename prefix)
python cursor_readme/prepare_custom_dataset.py \
    --video-dir /path/to/videos \
    --output-dir data/my_dataset \
    --val-ratio 0.2 \
    --stratify

# With CSV labels
python cursor_readme/prepare_custom_dataset.py \
    --video-dir /path/to/videos \
    --output-dir data/my_dataset \
    --label-csv labels.csv \
    --video-col "filename" \
    --label-col "class" \
    --organize-by-class \
    --copy-files

# Labels from parent folder
python cursor_readme/prepare_custom_dataset.py \
    --video-dir /path/to/videos \
    --output-dir data/my_dataset \
    --label-extractor parent \
    --organize-by-class
```

---

## 🎯 Quick Start Workflow

### For First-Time Users:

1. **Read the comprehensive guide** (30-60 minutes)
   ```bash
   cat cursor_readme/slowfast_finetuning_guide.md
   ```

2. **Prepare your dataset** using the script
   ```bash
   python cursor_readme/prepare_custom_dataset.py \
       --video-dir /path/to/your/videos \
       --output-dir data/my_dataset \
       --val-ratio 0.2 \
       --stratify
   ```

3. **Create your config** from the example
   ```bash
   cp cursor_readme/slowfast_custom_example.py \
      configs/recognition/slowfast/my_dataset.py
   # Edit the TODOs in the config file
   ```

4. **Start training**
   ```bash
   python tools/train.py \
       configs/recognition/slowfast/my_dataset.py \
       --work-dir work_dirs/my_dataset
   ```

### For Experienced Users:

1. **Check the quick start checklist**
   ```bash
   cat cursor_readme/quickstart_checklist.md
   ```

2. **Prepare data** and **start training** following the checklist

---

## 📁 Project Structure

After following this guide, your project should look like:

```
mmaction2/
├── configs/
│   └── recognition/
│       └── slowfast/
│           └── my_dataset.py              # Your custom config
├── data/
│   └── my_dataset/
│       ├── videos_train/                  # Training videos
│       │   ├── class1/
│       │   ├── class2/
│       │   └── ...
│       ├── videos_val/                    # Validation videos
│       │   ├── class1/
│       │   ├── class2/
│       │   └── ...
│       └── annotations/
│           ├── train_list.txt             # Train annotations
│           ├── val_list.txt               # Val annotations
│           └── class_index.txt            # Class mapping
├── work_dirs/
│   └── my_dataset/
│       ├── my_dataset.py                  # Config backup
│       ├── *.log                          # Training logs
│       ├── epoch_*.pth                    # Checkpoints
│       └── best_acc_top1_epoch_*.pth      # Best model
└── cursor_readme/                         # This folder
    ├── README.md
    ├── slowfast_finetuning_guide.md
    ├── quickstart_checklist.md
    ├── slowfast_custom_example.py
    └── prepare_custom_dataset.py
```

---

## 🔑 Key Concepts

### SlowFast Architecture
- **Dual Pathway**: Slow (spatial) + Fast (temporal)
- **Slow Pathway**: Low frame rate (4 FPS), full channels
- **Fast Pathway**: High frame rate (32 FPS), reduced channels (1/8)
- **Lateral Connections**: Transfer information from Fast to Slow

### Configuration Inheritance
MMAction2 uses hierarchical configs:
```python
_base_ = ['base_config.py']  # Inherit settings
model = dict(
    cls_head=dict(num_classes=10)  # Override specific values
)
```

### Data Pipeline
Video → Sample Frames → Decode → Augment → Format → Model

### Training Flow
1. Load pretrained weights (except head)
2. Finetune with smaller learning rate
3. Validate periodically
4. Save best model
5. Test on validation set

---

## 💡 Common Parameters

### Must Change for Your Dataset
```python
# Model
model = dict(
    cls_head=dict(num_classes=YOUR_NUM_CLASSES)
)

# Data paths
data_root = 'data/YOUR_DATASET/videos_train'
ann_file_train = 'data/YOUR_DATASET/annotations/train_list.txt'
```

### Usually Need Tuning
```python
# Learning rate (most important!)
optim_wrapper = dict(
    optimizer=dict(lr=0.01)  # Try: 0.001, 0.005, 0.01, 0.05
)

# Batch size (based on GPU memory)
train_dataloader = dict(
    batch_size=8  # Try: 2, 4, 8, 16
)

# Training duration
train_cfg = dict(
    max_epochs=50  # Try: 30, 50, 100
)
```

### Rarely Need Changing
```python
# Data sampling
dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1)

# Input resolution
dict(type='Resize', scale=(224, 224), keep_ratio=False)

# Gradient clipping
clip_grad=dict(max_norm=40, norm_type=2)
```

---

## 🚨 Common Issues

### 1. CUDA Out of Memory
**Solutions:**
- Reduce `batch_size`: 8 → 4 → 2
- Reduce `clip_len`: 32 → 16
- Increase `frame_interval`: 2 → 4
- Add `--amp` flag for mixed precision

### 2. Poor Accuracy
**Check:**
- Are annotations correct? (0-indexed labels)
- Is learning rate appropriate? (try 10x smaller/larger)
- Is model overfitting? (add more augmentation)
- Is training long enough? (check if loss still decreasing)

### 3. Slow Training
**Optimizations:**
- Increase `num_workers`: 2 → 8
- Enable `persistent_workers=True`
- Use SSD for data storage
- Check GPU utilization: `nvidia-smi`

### 4. Label Mismatch
**Verify:**
```bash
# Check annotation file format
head data/my_dataset/annotations/train_list.txt
# Should be: video_path label_index

# Check class index
cat data/my_dataset/annotations/class_index.txt
# Should be: label_index class_name
```

---

## 📊 Expected Performance

### Hardware Requirements
- **Minimum**: 1x GPU with 10GB+ VRAM
- **Recommended**: 4x GPU with 16GB+ VRAM each
- **Storage**: Fast SSD preferred (lots of I/O)

### Training Time (RTX 3090, batch_size=8)
| Dataset Size | Training Time | Epochs |
|--------------|---------------|--------|
| 500 videos   | 1-2 hours     | 50     |
| 5K videos    | 8-12 hours    | 50     |
| 50K videos   | 3-5 days      | 50     |

### Accuracy Expectations
Highly dependent on:
- Task complexity (fine-grained vs coarse)
- Dataset quality and size
- Domain similarity to Kinetics-400

**Typical ranges:**
- Small clean dataset (1K videos): 75-85%
- Medium dataset (10K videos): 80-90%
- Large dataset (100K+ videos): 85-95%

---

## 🔗 Useful Links

### Documentation
- [MMAction2 Official Docs](https://mmaction2.readthedocs.io/)
- [MMAction2 GitHub](https://github.com/open-mmlab/mmaction2)
- [Model Zoo](https://mmaction2.readthedocs.io/en/latest/modelzoo_statistics.html)

### Papers
- [SlowFast Networks](https://arxiv.org/abs/1812.03982) (ICCV 2019)
- [Kinetics Dataset](https://arxiv.org/abs/1705.06950)

### Pretrained Models
- [SlowFast R50 Kinetics-400](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth)
- [SlowFast R101 Kinetics-400](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb_20220818-9c0e09bd.pth)

---

## 🤝 Support

If you encounter issues:

1. **Check the troubleshooting section** in the comprehensive guide
2. **Search existing issues** on GitHub
3. **Ask on Discord/Forum** with:
   - Your config file
   - Error messages
   - System information (GPU, CUDA version, etc.)

---

## 📝 Notes

- All documentation follows user preferences (English, logging instead of print)
- Configuration files are designed to be modular and easy to customize
- Scripts include extensive error handling and logging
- Examples are tested with MMAction2 v1.x

---

## ✅ Checklist Before Training

- [ ] Environment setup (MMAction2, PyTorch, CUDA)
- [ ] Dataset prepared and verified
- [ ] Annotation files created and checked
- [ ] Config file created and customized
- [ ] Pretrained weights accessible
- [ ] GPU memory sufficient for batch size
- [ ] Training command tested (dry run)

**Ready to train?** 🎉

```bash
python tools/train.py configs/recognition/slowfast/my_dataset.py
```

---

**Good luck with your SlowFast finetuning!** 🚀

For detailed information, refer to the individual documentation files in this folder.

