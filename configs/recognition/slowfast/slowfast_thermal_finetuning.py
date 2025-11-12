"""
SlowFast Configuration for Thermal Action Recognition Finetuning

This configuration is specifically designed for training SlowFast on thermal sensor data
with the following characteristics:
- Input: 40×60 thermal frames (resized to 256×384)
- Small dataset: 314 train samples, 73 val samples
- 14 action classes (highly imbalanced)
- HDF5 data storage
- Heavy data augmentation to compensate for small dataset size

Key Modifications from Standard SlowFast:
1. Custom dataset loader (ThermalHDF5Dataset) for HDF5 files
2. Resizing pipeline: 40×60 → 256×384 (maintains 2:3 aspect ratio)
3. Aggressive data augmentation (ColorJitter, RandomErasing, etc.)
4. Class-weighted loss for imbalanced classes
5. Smaller learning rate and longer warmup for finetuning
6. Frequent validation (every epoch)

Author: Generated for thermal action recognition
Date: 2025-11-12
"""

# =============================================================================
# BASE CONFIGURATION
# =============================================================================
_base_ = ['../../_base_/default_runtime.py']

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,  # Will be loaded via load_from
        resample_rate=8,  # tau: temporal downsampling
        speed_ratio=8,    # alpha: Fast/Slow frame rate ratio  
        channel_ratio=8,  # beta_inv: Slow/Fast channel ratio
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False  # Enable BN training for finetuning
        ),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False  # Enable BN training for finetuning
        )
    ),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048 (slow) + 256 (fast)
        num_classes=14,    # THERMAL: 14 action classes
        spatial_type='avg',
        dropout_ratio=0.8,  # Increased dropout for small dataset (default 0.5)
        average_clips='prob',
        # Use weighted loss for class imbalance
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
            # Class weights will be set dynamically based on dataset
            class_weight=None  # Will be computed from training data
        )
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],  # ImageNet stats (thermal scaled to match)
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'
    )
)

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
dataset_type = 'ThermalHDF5Dataset'
data_root_hdf5 = 'ThermalDataGen/thermal_action_dataset/frames'
ann_file_train = 'ThermalDataGen/thermal_action_dataset/annotations/train.json'
ann_file_val = 'ThermalDataGen/thermal_action_dataset/annotations/val.json'

# =============================================================================
# DATA PIPELINE - TRAINING
# =============================================================================
# Training pipeline with HEAVY augmentation for small dataset
train_pipeline = [
    # Images are already loaded by ThermalHDF5Dataset
    # Input shape: [64, 40, 60, 3] uint8
    
    # Resize from 40×60 to 256×384 (maintains 2:3 aspect ratio)
    # 40×60 → 256×384 (both are 2:3 aspect ratio)
    dict(type='Resize', scale=(384, 256), keep_ratio=True),  # (W, H)
    
    # Random resized crop for augmentation
    dict(
        type='RandomResizedCrop',
        area_range=(0.7, 1.0),  # More aggressive cropping
        aspect_ratio_range=(0.85, 1.15)  # Allow some aspect ratio variation
    ),
    
    # Ensure final size
    dict(type='Resize', scale=(384, 256), keep_ratio=False),  # (W, H) -> 256×384 (H×W)
    
    # Horizontal flip
    dict(type='Flip', flip_ratio=0.5),
    
    # Color augmentation (thermal-appropriate, no hue adjustment)
    dict(
        type='ColorJitter',
        brightness=0.3,
        contrast=0.3,
        saturation=0,
        hue=0
    ),
    
    # Random erasing for robustness
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        min_area_ratio=0.02,
        max_area_ratio=0.2,
        fill_color=[128, 128, 128],
        fill_std=[64, 64, 64]
    ),
    
    # Format for model: [B, C, T, H, W]
    dict(type='FormatShape', input_format='NCTHW'),
    
    # Pack into standard format
    dict(type='PackActionInputs')
]

# =============================================================================
# DATA PIPELINE - VALIDATION
# =============================================================================
val_pipeline = [
    # Input shape: [64, 40, 60, 3] uint8
    
    # Resize to target size (no crop for validation)
    dict(type='Resize', scale=(384, 256), keep_ratio=True),
    
    # Center crop to ensure exact size
    dict(type='CenterCrop', crop_size=(256, 384)),  # (H, W)
    
    # Format for model
    dict(type='FormatShape', input_format='NCTHW'),
    
    # Pack
    dict(type='PackActionInputs')
]

# =============================================================================
# DATA PIPELINE - TESTING
# =============================================================================
test_pipeline = [
    # Similar to validation but can use multiple clips
    dict(type='Resize', scale=(384, 256), keep_ratio=True),
    dict(type='CenterCrop', crop_size=(256, 384)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# =============================================================================
# DATALOADER CONFIGURATION
# =============================================================================
train_dataloader = dict(
    batch_size=4,  # Small batch size for 314 samples (adjust based on GPU memory)
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(hdf5=data_root_hdf5),
        pipeline=train_pipeline,
        frame_window=64,
        test_mode=False
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(hdf5=data_root_hdf5),
        pipeline=val_pipeline,
        frame_window=64,
        test_mode=True
    )
)

test_dataloader = val_dataloader

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,  # More epochs for small dataset
    val_begin=1,     # Validate from first epoch
    val_interval=1   # Validate every epoch (dataset is small)
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================
# Use smaller learning rate for finetuning
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005,  # 10x smaller than from-scratch training (0.05 vs 0.5)
        momentum=0.9,
        weight_decay=1e-4
    ),
    clip_grad=dict(max_norm=40, norm_type=2),
    
    # Different learning rates for different parts (optional)
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # Backbone learns 10x slower
            'cls_head': dict(lr_mult=1.0),  # Head learns at full rate
        }
    )
)

# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================
param_scheduler = [
    # Long warmup for stability with small dataset
    dict(
        type='LinearLR',
        start_factor=0.01,  # Start very small
        by_epoch=True,
        begin=0,
        end=10,  # 10 epoch warmup
        convert_to_iter_based=True
    ),
    # Cosine annealing
    dict(
        type='CosineAnnealingLR',
        T_max=100,
        eta_min=0.0001,  # Don't go to zero
        by_epoch=True,
        begin=0,
        end=100
    )
]

# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=10,  # Log every 10 iterations (dataset is small)
        ignore_last=False
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,  # Save every 5 epochs
        max_keep_ckpts=5,
        save_best='auto',  # Save best model
        rule='greater'  # Higher accuracy is better
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook')
)

# =============================================================================
# LOGGING AND VISUALIZATION
# =============================================================================
log_processor = dict(
    type='LogProcessor',
    window_size=10,  # Small window for small dataset
    by_epoch=True
)

vis_backends = [
    dict(type='LocalVisBackend'),
    # Uncomment to enable TensorBoard
    dict(type='TensorboardVisBackend', save_dir='work_dirs/thermal_slowfast/tf_logs')
]

visualizer = dict(
    type='ActionVisualizer',
    vis_backends=vis_backends
)

log_level = 'INFO'

# =============================================================================
# PRETRAINED MODEL LOADING
# =============================================================================
# Download pretrained SlowFast weights from MMAction2 model zoo
# Use the download script: python tools/download_pretrained_slowfast.py
load_from = 'checkpoints/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth'

# Resume from checkpoint if training interrupted
resume = False

# =============================================================================
# ENVIRONMENT SETTINGS
# =============================================================================
env_cfg = dict(
    cudnn_benchmark=True,  # Enable for fixed input size (faster)
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# Random seed for reproducibility
randomness = dict(seed=42, deterministic=False, diff_rank_seed=False)

# =============================================================================
# MIXED PRECISION TRAINING (Optional)
# =============================================================================
# Enable with --amp flag during training
# Reduces memory usage by ~30% with minimal accuracy impact
# optim_wrapper.update(dict(
#     type='AmpOptimWrapper',
#     loss_scale='dynamic'
# ))

# =============================================================================
# AUTO LEARNING RATE SCALING
# =============================================================================
# Automatically scale LR based on batch size
# Base: 4 GPUs × 4 batch_size = 16
auto_scale_lr = dict(
    enable=True,
    base_batch_size=16
)

# =============================================================================
# CUSTOM HOOKS (Optional)
# =============================================================================
custom_hooks = [
    # Early stopping if validation accuracy doesn't improve
    dict(
        type='EarlyStoppingHook',
        monitor='acc/top1',
        patience=20,  # Stop if no improvement for 20 epochs
        min_delta=0.001,
        rule='greater'
    )
]

# =============================================================================
# MODEL EXPORT SETTINGS (For Deployment)
# =============================================================================
# Configuration for exporting to ONNX after training
onnx_config = dict(
    input_shape=(1, 3, 64, 256, 384),  # (B, C, T, H, W)
    normalize_cfg=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]
    ),
    opset_version=11
)

# =============================================================================
# NOTES AND RECOMMENDATIONS
# =============================================================================
"""
TRAINING RECOMMENDATIONS:

1. Data Augmentation:
   - Already aggressive to compensate for small dataset
   - Consider mixup/cutmix if accuracy plateaus
   
2. Class Imbalance:
   - Class weights are automatically computed (see get_class_weights)
   - "lying down-lying with risk" dominates (220/405 annotations)
   - Consider oversampling minority classes
   
3. Training Strategy:
   - Start with frozen backbone (first 10 epochs)
   - Then finetune entire network
   - Monitor for overfitting (train >> val accuracy)
   
4. Memory Optimization:
   - Reduce batch_size if OOM: 4 → 2 → 1
   - Enable gradient accumulation if needed
   - Use mixed precision (--amp flag)
   
5. Expected Results:
   - Small dataset limits performance
   - Target: 50-70% top-1 accuracy
   - Focus on mean class accuracy (handles imbalance)
   
6. Troubleshooting:
   - If loss explodes: reduce learning rate
   - If no learning: increase learning rate or disable warmup
   - If overfitting: increase dropout, add augmentation
   
TRAINING COMMANDS:

# Single GPU
python tools/train.py configs/recognition/slowfast/slowfast_thermal_finetuning.py

# Multi-GPU (2 GPUs)
bash tools/dist_train.sh configs/recognition/slowfast/slowfast_thermal_finetuning.py 2

# With mixed precision
python tools/train.py configs/recognition/slowfast/slowfast_thermal_finetuning.py --amp

# Resume training
python tools/train.py configs/recognition/slowfast/slowfast_thermal_finetuning.py --resume

TESTING:

# Test best model
python tools/test.py \\
    configs/recognition/slowfast/slowfast_thermal_finetuning.py \\
    work_dirs/slowfast_thermal_finetuning/best_acc_top1_*.pth

DATASET EXPANSION:

When you collect more data:
1. Regenerate HDF5 files with new annotations
2. Update class weights in config
3. Consider reducing dropout (0.8 → 0.5)
4. May need to increase epochs
"""

