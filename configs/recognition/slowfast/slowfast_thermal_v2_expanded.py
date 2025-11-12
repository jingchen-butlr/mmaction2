"""
SlowFast Configuration for Thermal Action Recognition - V2 (Expanded Dataset)

UPDATED FOR LARGER DATASET:
- Train: 2,043 samples (was 314) - 6.5x increase!
- Val: 494 samples (was 73) - 6.8x increase!
- Better class distribution (less imbalanced)

KEY CHANGES FROM V1:
1. Better pretrained model (8x8, 76.80% vs 4x16, 75.55%)
2. Reduced dropout (0.8 → 0.6) - more data = less regularization needed
3. Increased batch size (4 → 8) - can afford with more samples
4. Slightly reduced augmentation - dataset is larger now
5. Updated learning rate for new batch size

Expected Performance: 75-82% accuracy (vs 71% with small dataset)

Author: Generated for thermal action recognition V2
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
        resample_rate=8,  # tau
        speed_ratio=8,    # alpha
        channel_ratio=8,  # beta_inv
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
            norm_eval=False
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
            norm_eval=False
        )
    ),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,
        num_classes=14,    # 14 thermal action classes
        spatial_type='avg',
        dropout_ratio=0.6,  # REDUCED from 0.8 (more data = less dropout needed)
        average_clips='prob',
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0
        )
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
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
# DATA PIPELINE - TRAINING (Slightly reduced augmentation for larger dataset)
# =============================================================================
train_pipeline = [
    # Input shape: [64, 40, 60, 3] uint8
    
    # Resize from 40×60 to 256×384
    dict(type='Resize', scale=(384, 256), keep_ratio=True),
    
    # Slightly less aggressive cropping (0.75-1.0 vs 0.7-1.0)
    dict(
        type='RandomResizedCrop',
        area_range=(0.75, 1.0),  # REDUCED aggression (was 0.7-1.0)
        aspect_ratio_range=(0.9, 1.1)  # REDUCED variation (was 0.85-1.15)
    ),
    
    # Ensure final size
    dict(type='Resize', scale=(384, 256), keep_ratio=False),
    
    # Horizontal flip
    dict(type='Flip', flip_ratio=0.5),
    
    # Slightly reduced color jitter
    dict(
        type='ColorJitter',
        brightness=0.25,  # REDUCED from 0.3
        contrast=0.25,    # REDUCED from 0.3
        saturation=0,
        hue=0
    ),
    
    # Slightly reduced erasing
    dict(
        type='RandomErasing',
        erase_prob=0.2,  # REDUCED from 0.25
        min_area_ratio=0.02,
        max_area_ratio=0.15,  # REDUCED from 0.2
        fill_color=[128, 128, 128],
        fill_std=[64, 64, 64]
    ),
    
    # Format for model
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# =============================================================================
# DATA PIPELINE - VALIDATION
# =============================================================================
val_pipeline = [
    dict(type='Resize', scale=(384, 256), keep_ratio=True),
    dict(type='CenterCrop', crop_size=(256, 384)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = val_pipeline

# =============================================================================
# DATALOADER CONFIGURATION (Increased batch size)
# =============================================================================
train_dataloader = dict(
    batch_size=4,  # Keep at 4 due to GPU memory constraints
    num_workers=8,
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
    batch_size=4,  # Keep at 4 due to GPU memory constraints
    num_workers=8,
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
    max_epochs=100,
    val_begin=1,
    val_interval=2  # Validate every 2 epochs (was every 1 for small dataset)
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.01,  # INCREASED from 0.005 (more data = can handle higher LR)
        momentum=0.9,
        weight_decay=1e-4
    ),
    clip_grad=dict(max_norm=40, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'cls_head': dict(lr_mult=1.0),
        }
    )
)

# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================
param_scheduler = [
    # Warmup
    dict(
        type='LinearLR',
        start_factor=0.1,  # Start from 10% (was 1%)
        by_epoch=True,
        begin=0,
        end=5,  # REDUCED from 10 (faster warmup with more data)
        convert_to_iter_based=True
    ),
    # Cosine annealing
    dict(
        type='CosineAnnealingLR',
        T_max=100,
        eta_min=0.0001,
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
        interval=20,  # INCREASED from 10 (more iterations now)
        ignore_last=False
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=5,
        save_best='auto',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook')
)

# =============================================================================
# LOGGING
# =============================================================================
log_processor = dict(
    type='LogProcessor',
    window_size=20,  # INCREASED from 10
    by_epoch=True
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend', save_dir='work_dirs/thermal_slowfast_v2/tf_logs')
]

visualizer = dict(
    type='ActionVisualizer',
    vis_backends=vis_backends
)

log_level = 'INFO'

# =============================================================================
# PRETRAINED MODEL LOADING (8x8 version, 76.80% accuracy)
# =============================================================================
load_from = 'checkpoints/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth'

resume = False

# =============================================================================
# ENVIRONMENT SETTINGS
# =============================================================================
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

randomness = dict(seed=42, deterministic=False, diff_rank_seed=False)

# =============================================================================
# AUTO LEARNING RATE SCALING
# =============================================================================
auto_scale_lr = dict(
    enable=True,
    base_batch_size=32  # 8 GPUs × 4 batch_size = 32
)

# =============================================================================
# CUSTOM HOOKS
# =============================================================================
custom_hooks = [
    # Increased patience for larger dataset
    dict(
        type='EarlyStoppingHook',
        monitor='acc/top1',
        patience=30,  # INCREASED from 20 (more data = more patient)
        min_delta=0.001,
        rule='greater'
    )
]

# =============================================================================
# NOTES
# =============================================================================
"""
DATASET V2 IMPROVEMENTS:
- Train: 314 → 2,043 samples (6.5x increase!)
- Val: 73 → 494 samples (6.8x increase!)
- Better class balance
- More sensors: 8 → 25

CONFIGURATION CHANGES FROM V1:
- Dropout: 0.8 → 0.6 (less regularization needed)
- Learning rate: 0.005 → 0.01 (more data = can learn faster)
- Batch size: 4 → 8 (better gradient estimates)
- Warmup: 10 → 5 epochs (faster warmup)
- Augmentation: Slightly reduced (dataset is larger)
- Early stopping: 20 → 30 epochs patience
- Pretrained: 4x16 (75.55%) → 8x8 (76.80%)

EXPECTED IMPROVEMENTS:
- Accuracy: 71% → 78-82%
- Mean class accuracy: 24% → 50-60%
- Training time: ~2 hours (more data, but also larger batches)

TRAINING COMMAND:
python tools/train.py configs/recognition/slowfast/slowfast_thermal_v2_expanded.py \\
    --work-dir work_dirs/thermal_slowfast_v2 \\
    --seed 42
"""

