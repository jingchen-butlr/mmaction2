# SlowFast Finetuning Quick Start Checklist

This is a condensed checklist for quickly finetuning SlowFast on your custom dataset.

## ✅ Pre-Training Checklist

### 1. Data Preparation
- [ ] Videos are organized in class folders or flat structure
- [ ] Created `train_list.txt` annotation file (format: `video_path label`)
- [ ] Created `val_list.txt` annotation file
- [ ] Verified video files are readable (not corrupted)
- [ ] Checked data directory structure:
  ```
  data/your_dataset/
  ├── videos_train/
  │   └── [videos or class folders]
  ├── videos_val/
  │   └── [videos or class folders]
  └── annotations/
      ├── train_list.txt
      └── val_list.txt
  ```

### 2. Environment Setup
- [ ] MMAction2 installed: `python -c "import mmaction; print(mmaction.__version__)"`
- [ ] GPU available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Dependencies installed: `decord`, `mmcv`, `mmengine`

### 3. Configuration File
- [ ] Created custom config file (see example below)
- [ ] Changed `num_classes` to match your dataset
- [ ] Updated data paths (`data_root`, `ann_file_train`, `ann_file_val`)
- [ ] Set pretrained weights in `load_from`
- [ ] Adjusted batch size based on GPU memory
- [ ] Set appropriate learning rate (typically 10x smaller for finetuning)

### 4. Quick Validation
- [ ] Config syntax is valid: `python tools/train.py your_config.py --cfg-options dry_run=True`
- [ ] Can load first batch: Test with minimal training

## 🚀 Training Commands

### Single GPU
```bash
python tools/train.py \
    configs/recognition/slowfast/slowfast_custom.py \
    --work-dir work_dirs/slowfast_custom \
    --seed 0
```

### Multi-GPU (e.g., 4 GPUs)
```bash
bash tools/dist_train.sh \
    configs/recognition/slowfast/slowfast_custom.py \
    4 \
    --work-dir work_dirs/slowfast_custom \
    --auto-scale-lr
```

### Resume Training
```bash
python tools/train.py \
    configs/recognition/slowfast/slowfast_custom.py \
    --work-dir work_dirs/slowfast_custom \
    --resume
```

## 🧪 Testing Commands

### Test on Validation Set
```bash
python tools/test.py \
    configs/recognition/slowfast/slowfast_custom.py \
    work_dirs/slowfast_custom/best_acc_top1_epoch_*.pth
```

### Inference on Single Video
```python
from mmaction.apis import inference_recognizer, init_recognizer

model = init_recognizer(
    'configs/recognition/slowfast/slowfast_custom.py',
    'work_dirs/slowfast_custom/best_acc_top1_epoch_40.pth',
    device='cuda:0'
)

result = inference_recognizer(model, 'path/to/video.mp4')
print(f"Predicted class: {result.pred_label[0]}")
print(f"Confidence: {result.pred_score[0]:.4f}")
```

## 📊 Monitoring Training

### Check Training Logs
```bash
# Real-time log viewing
tail -f work_dirs/slowfast_custom/*.log

# Check training metrics
grep "acc/top1" work_dirs/slowfast_custom/*.log
```

### Output Directory Structure
```
work_dirs/slowfast_custom/
├── slowfast_custom.py      # Config backup
├── *.log                   # Training logs
├── epoch_*.pth             # Checkpoints
├── best_acc_top1_epoch_*.pth  # Best model
└── last_checkpoint         # Latest checkpoint path
```

## ⚙️ Common Configuration Adjustments

### Reduce Memory Usage
```python
# In your config:
train_dataloader = dict(batch_size=4)  # Reduce from 8
train_pipeline = [
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1),  # Reduce frames
    ...
]
```

### Adjust Learning Rate
```python
optim_wrapper = dict(
    optimizer=dict(lr=0.005)  # Try: 0.001, 0.005, 0.01, 0.05
)
```

### Early Stopping (if overfitting)
```python
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=30,  # Reduce epochs
    val_interval=1  # Validate every epoch
)
```

## 🐛 Troubleshooting

### CUDA Out of Memory
1. Reduce `batch_size` in config
2. Add `--amp` flag for mixed precision
3. Reduce `clip_len` or increase `frame_interval` in data pipeline

### Model Not Learning
1. Check data loading: verify annotation file format
2. Try smaller learning rate: `lr=0.001`
3. Verify labels are correct (0-indexed)
4. Increase batch size if possible

### Slow Training
1. Increase `num_workers` in dataloader
2. Enable `persistent_workers=True`
3. Use SSD for data storage
4. Verify GPU utilization: `nvidia-smi`

### Config Errors
```python
# Debug config
from mmengine import Config
cfg = Config.fromfile('your_config.py')
print(cfg.pretty_text)
```

## 📈 Expected Results Timeline

Small Dataset (< 1K videos):
- Setup: 30 min - 1 hour
- Training: 2-4 hours
- Expected accuracy: 70-85% (depends on task complexity)

Medium Dataset (1K-10K videos):
- Setup: 1-2 hours
- Training: 8-24 hours
- Expected accuracy: 75-90%

Large Dataset (> 10K videos):
- Setup: 2-4 hours
- Training: 1-7 days
- Expected accuracy: 80-95%

## 💡 Quick Tips

1. **Start Small**: Test with a subset first (100 videos) to verify pipeline
2. **Monitor Validation**: If val accuracy plateaus, stop training early
3. **Save Checkpoints**: Keep at least top-3 best models
4. **Pretrained Weights**: Always use pretrained weights for better results
5. **Learning Rate**: Most critical hyperparameter - tune this first
6. **Batch Size**: Larger is better (more stable), but limited by GPU memory
7. **Data Augmentation**: More augmentation = better generalization

## 🔗 Quick Links

- Full Guide: `cursor_readme/slowfast_finetuning_guide.md`
- Example Config: `cursor_readme/slowfast_custom_example.py`
- MMAction2 Docs: https://mmaction2.readthedocs.io/
- SlowFast Paper: https://arxiv.org/abs/1812.03982

## 📝 Notes

- Default input: 32 frames at 2-frame intervals (16 FPS video → 2 seconds)
- SlowFast has two pathways: Slow (4 frames) + Fast (32 frames)
- Requires ~10-16 GB GPU memory for batch_size=8
- Training speed: ~40 videos/sec on RTX 3090

---

**Need help?** Check the full guide or raise an issue on GitHub!

