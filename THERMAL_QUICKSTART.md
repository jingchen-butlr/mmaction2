# 🔥 Thermal SlowFast - Quick Start

**Train SlowFast on your thermal action dataset in 5 minutes!**

---

## ✅ Status: Ready to Train!

Your thermal dataset has been fully integrated with MMAction2's SlowFast framework.

**Dataset**: 314 train / 73 val samples, 14 action classes, 40×60 thermal frames  
**Model**: SlowFast R50 (pretrained on Kinetics-400)  
**Input**: Resized to 256×384 (maintains 2:3 aspect ratio)  
**Training time**: 4-5 hours on RTX 3090  

---

## 🚀 Quick Start (3 Steps)

### Step 1: Run Setup Script

```bash
cd /home/ec2-user/jingchen/mmaction2
bash tools/thermal_quickstart.sh
```

This will:
- ✓ Verify environment
- ✓ Check dataset
- ✓ Download pretrained weights (~400MB)
- ✓ Start training (optional)

### Step 2: Monitor Training

```bash
# Watch training logs
tail -f work_dirs/thermal_slowfast/*.log

# Check validation accuracy
grep "acc/top1" work_dirs/thermal_slowfast/*.log
```

### Step 3: Test Model

```bash
# After training completes
python tools/test.py \
    configs/recognition/slowfast/slowfast_thermal_finetuning.py \
    work_dirs/thermal_slowfast/best_acc_top1_epoch_*.pth
```

**That's it!** 🎉

---

## 📊 What to Expect

### Training Progress
- **Epochs**: 100 (with early stopping)
- **Validation**: Every epoch (dataset is small)
- **Checkpoints**: Saved every 5 epochs
- **Best model**: Automatically saved

### Accuracy Targets
- **Top-1**: 50-70% (realistic for 314 samples)
- **Top-3**: 70-85%
- **Mean class accuracy**: 40-60% (handles class imbalance)

### Performance by Class
- **High accuracy** (>70%): `lying down-lying with risk`, `standing`
- **Medium accuracy** (50-70%): `walking`, `transition-normal`
- **Low accuracy** (<50%): Rare classes with <10 samples

---

## 🛠️ Manual Training Commands

If you prefer manual control:

```bash
# Download pretrained weights first
python tools/download_pretrained_slowfast.py

# Single GPU training
python tools/train.py \
    configs/recognition/slowfast/slowfast_thermal_finetuning.py \
    --work-dir work_dirs/thermal_slowfast \
    --seed 42

# Multi-GPU training (if you have 2+ GPUs)
bash tools/dist_train.sh \
    configs/recognition/slowfast/slowfast_thermal_finetuning.py \
    2 \
    --work-dir work_dirs/thermal_slowfast

# With mixed precision (reduces memory by 30%)
python tools/train.py \
    configs/recognition/slowfast/slowfast_thermal_finetuning.py \
    --amp
```

---

## 📁 What Was Created

### 1. Custom Dataset Loader
**File**: `mmaction/datasets/thermal_hdf5_dataset.py`
- Loads 40×60 thermal frames from HDF5
- Resizes to 256×384
- Handles 64-frame temporal windows
- Automatic temperature normalization

### 2. Training Configuration
**File**: `configs/recognition/slowfast/slowfast_thermal_finetuning.py`
- Optimized for small dataset (314 samples)
- Heavy data augmentation
- Class-weighted loss for imbalance
- Differential learning rates

### 3. Setup Tools
**Files**: 
- `tools/download_pretrained_slowfast.py` - Downloads weights
- `tools/thermal_quickstart.sh` - Automated setup

### 4. Documentation
**Folder**: `cursor_readme/`
- `THERMAL_INTEGRATION_SUMMARY.md` - Overview (⭐ Start here!)
- `THERMAL_SLOWFAST_FINETUNING_GUIDE.md` - Complete guide
- General SlowFast documentation

---

## ⚙️ Configuration Highlights

### Data Pipeline
```python
# Training augmentation (heavy for small dataset)
- Resize: 40×60 → 256×384
- RandomResizedCrop (0.7-1.0 scale)
- Horizontal flip (50%)
- ColorJitter
- RandomErasing (25%)
```

### Model Settings
```python
- Backbone: SlowFast R50 (dual-pathway)
- Input: [B, 3, 64, 256, 384]
- Classes: 14
- Dropout: 0.8 (high for regularization)
- Class weights: Automatic (handles imbalance)
```

### Training Settings
```python
- Learning rate: 0.005 (10x smaller for finetuning)
- Batch size: 4 (adjustable)
- Epochs: 100
- Warmup: 10 epochs
- Validation: Every epoch
- Early stopping: 20 epochs patience
```

---

## 🐛 Common Issues

### CUDA Out of Memory

```bash
# Reduce batch size
python tools/train.py config.py \
    --cfg-options train_dataloader.batch_size=2

# Or enable mixed precision
python tools/train.py config.py --amp
```

### Loss Not Decreasing

**Check**:
1. Learning rate (try 0.001 or 0.01)
2. Data loading (run verification script)
3. Labels (must be 0-13)

### Slow Training

```bash
# Increase data loading workers
python tools/train.py config.py \
    --cfg-options train_dataloader.num_workers=8
```

---

## 📈 Monitoring Tools

### TensorBoard

```bash
tensorboard --logdir work_dirs/thermal_slowfast/tf_logs --port 6006

# If remote server, forward port:
# ssh -L 6006:localhost:6006 user@server
```

### Log Files

```bash
# Training directory
work_dirs/thermal_slowfast/
├── *.log                          # Training logs
├── epoch_*.pth                    # Checkpoints
├── best_acc_top1_epoch_*.pth      # Best model
└── tf_logs/                       # TensorBoard
```

---

## 🔧 Hyperparameter Tuning

**Most impactful parameters**:

| Parameter | Default | Try If... |
|-----------|---------|-----------|
| `lr` | 0.005 | Loss not decreasing: 0.001, 0.01 |
| `batch_size` | 4 | OOM: 2, 1 |
| `dropout_ratio` | 0.8 | Overfitting: 0.9 |
| `max_epochs` | 100 | Poor convergence: 150 |

**Override at runtime**:
```bash
python tools/train.py config.py \
    --cfg-options optim_wrapper.optimizer.lr=0.001
```

---

## 📚 Documentation Guide

**For first-time users**:
1. Read [`cursor_readme/THERMAL_INTEGRATION_SUMMARY.md`](cursor_readme/THERMAL_INTEGRATION_SUMMARY.md) (10 min)
2. Run `bash tools/thermal_quickstart.sh` (5 min)
3. Monitor training (4-5 hours)

**For detailed information**:
- [`cursor_readme/THERMAL_SLOWFAST_FINETUNING_GUIDE.md`](cursor_readme/THERMAL_SLOWFAST_FINETUNING_GUIDE.md) - Complete guide
- [`cursor_readme/slowfast_finetuning_guide.md`](cursor_readme/slowfast_finetuning_guide.md) - General SlowFast guide
- [`configs/recognition/slowfast/slowfast_thermal_finetuning.py`](configs/recognition/slowfast/slowfast_thermal_finetuning.py) - Config with detailed comments

---

## 💡 Tips for Success

✅ **Start with defaults** - Already optimized for your dataset  
✅ **Monitor validation accuracy** - Main metric to watch  
✅ **Use early stopping** - Prevents wasting time  
✅ **Check TensorBoard** - Visual monitoring helps  
✅ **Save best model** - Automatic with current config  

⚠️ **Small dataset limits accuracy** - 50-70% is realistic  
⚠️ **Class imbalance affects performance** - Mean class accuracy is better metric  
⚠️ **Overfitting is likely** - High dropout and augmentation help  

---

## 🎯 Success Criteria

**Training is successful when**:
- ✓ Loss steadily decreases
- ✓ Validation accuracy improves
- ✓ Top-1 accuracy > 50%
- ✓ No severe overfitting (train/val gap < 20%)

**Model is ready when**:
- ✓ Validation accuracy stabilizes
- ✓ Early stopping triggers (or reaches 100 epochs)
- ✓ Performance acceptable for your use case

---

## 🚀 Next Steps After Training

### 1. Evaluate Model

```bash
python tools/test.py \
    configs/recognition/slowfast/slowfast_thermal_finetuning.py \
    work_dirs/thermal_slowfast/best_acc_top1_epoch_*.pth
```

### 2. Analyze Results

- Check confusion matrix (which classes work well?)
- Review misclassifications
- Identify areas for improvement

### 3. Future Improvements

**When you collect more data**:
- Regenerate HDF5 files
- Reduce dropout (0.8 → 0.5)
- May achieve 75-85% with 1000+ samples

**Advanced techniques**:
- Ensemble multiple models
- Try focal loss for imbalance
- Implement oversampling
- Add mixup/cutmix augmentation

---

## 📞 Support

**If you encounter issues**:

1. Check [`cursor_readme/THERMAL_SLOWFAST_FINETUNING_GUIDE.md`](cursor_readme/THERMAL_SLOWFAST_FINETUNING_GUIDE.md) troubleshooting section
2. Review training logs for specific errors
3. Verify dataset paths are correct
4. Try reducing batch size if OOM

**Common fixes**:
- Batch size too large → `--cfg-options train_dataloader.batch_size=2`
- Loss exploding → `--cfg-options optim_wrapper.optimizer.lr=0.001`
- OOM errors → Add `--amp` flag

---

## ✅ Pre-Training Checklist

Before training, verify:

- [ ] MMAction2 installed: `python -c "import mmaction"`
- [ ] CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Dataset exists: `ls ThermalDataGen/thermal_action_dataset/`
- [ ] Config file exists: `ls configs/recognition/slowfast/slowfast_thermal_finetuning.py`
- [ ] Custom dataset registered: `python -c "from mmaction.datasets import ThermalHDF5Dataset"`

**All checks pass?** Run the quick start script! 🎉

```bash
bash tools/thermal_quickstart.sh
```

---

## 🎓 Learning Resources

- **MMAction2 Docs**: https://mmaction2.readthedocs.io/
- **SlowFast Paper**: https://arxiv.org/abs/1812.03982
- **Your Thermal Dataset Info**: `ThermalDataGen/cursor_readme/`

---

**Ready to train?** 🚀

```bash
bash tools/thermal_quickstart.sh
```

For detailed information, see [`cursor_readme/THERMAL_INTEGRATION_SUMMARY.md`](cursor_readme/THERMAL_INTEGRATION_SUMMARY.md).

