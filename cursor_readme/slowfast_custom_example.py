"""
SlowFast Custom Dataset Configuration Example

This is a ready-to-use configuration file for finetuning SlowFast on your custom dataset.

To use this config:
1. Copy it to configs/recognition/slowfast/ directory
2. Modify the parameters marked with # TODO
3. Run training: python tools/train.py configs/recognition/slowfast/this_file.py

Author: Generated for MMAction2 finetuning
Date: 2025
"""

# =============================================================================
# BASE CONFIGURATION: Inherit from existing config
# =============================================================================
_base_ = [
    '../../_base_/models/slowfast_r50.py',
    '../../_base_/default_runtime.py'
]

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
model = dict(
    # Model type and backbone are inherited from base config
    cls_head=dict(
        type='SlowFastHead',
        num_classes=10,  # TODO: Change to your number of classes
        spatial_type='avg',
        dropout_ratio=0.5,  # Increase to 0.6-0.8 for small datasets
        average_clips='prob'
    )
)

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
dataset_type = 'VideoDataset'  # Use 'RawFrameDataset' for extracted frames

# TODO: Update these paths to your dataset
data_root = 'data/your_dataset/videos_train'
data_root_val = 'data/your_dataset/videos_val'
ann_file_train = 'data/your_dataset/annotations/train_list.txt'
ann_file_val = 'data/your_dataset/annotations/val_list.txt'
ann_file_test = 'data/your_dataset/annotations/val_list.txt'  # Usually same as val

# =============================================================================
# DATA PIPELINE CONFIGURATION
# =============================================================================
file_client_args = dict(io_backend='disk')

# Training pipeline with data augmentation
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,           # Number of frames to sample
        frame_interval=2,      # Sample every 2 frames (temporal stride)
        num_clips=1            # Number of clips per video
    ),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),  # Resize shorter edge to 256
    dict(type='RandomResizedCrop'),        # Random crop for augmentation
    dict(type='Resize', scale=(224, 224), keep_ratio=False),  # Final resize
    dict(type='Flip', flip_ratio=0.5),     # Random horizontal flip
    dict(type='FormatShape', input_format='NCTHW'),  # Format: [N, C, T, H, W]
    dict(type='PackActionInputs')
]

# Validation pipeline (no augmentation)
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True
    ),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),  # Center crop instead of random
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# Test pipeline (multiple clips for better accuracy)
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,          # Sample 10 clips and average predictions
        test_mode=True
    ),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),  # 3-crop testing (left, center, right)
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# =============================================================================
# DATALOADER CONFIGURATION
# =============================================================================
train_dataloader = dict(
    batch_size=8,              # TODO: Adjust based on GPU memory (4-16)
    num_workers=8,             # Number of parallel data loading workers
    persistent_workers=True,   # Keep workers alive between epochs
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=8,              # Can be same or larger than training
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True
    )
)

test_dataloader = dict(
    batch_size=1,              # Usually 1 for multi-clip testing
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True
    )
)

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
val_evaluator = dict(
    type='AccMetric',
    metric_list=('top_k_accuracy', 'mean_class_accuracy')
)
test_evaluator = val_evaluator

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,         # TODO: Adjust (30-100 for finetuning)
    val_begin=1,           # Start validation from epoch 1
    val_interval=2         # Validate every 2 epochs
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.01,               # TODO: Tune this (try 0.001, 0.005, 0.01, 0.05)
        momentum=0.9,
        weight_decay=1e-4      # L2 regularization
    ),
    clip_grad=dict(max_norm=40, norm_type=2)  # Gradient clipping
)

# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================
param_scheduler = [
    # Warmup phase: gradually increase learning rate
    dict(
        type='LinearLR',
        start_factor=0.1,      # Start with 0.1 * lr
        by_epoch=True,
        begin=0,
        end=5,                 # Warmup for 5 epochs
        convert_to_iter_based=True
    ),
    # Main training: cosine annealing
    dict(
        type='CosineAnnealingLR',
        T_max=50,              # Should match max_epochs
        eta_min=0,             # Minimum learning rate
        by_epoch=True,
        begin=0,
        end=50
    )
]

# Alternative: Step decay (comment out above and uncomment below)
# param_scheduler = [
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=50,
#         by_epoch=True,
#         milestones=[20, 40],   # Decay at these epochs
#         gamma=0.1              # Multiply LR by 0.1
#     )
# ]

# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=50,           # Log every 50 iterations
        ignore_last=False
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,            # Save checkpoint every 5 epochs
        max_keep_ckpts=3,      # Keep only 3 latest checkpoints
        save_best='auto'       # Also save best model
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook')
)

# =============================================================================
# LOGGING AND VISUALIZATION
# =============================================================================
log_processor = dict(
    type='LogProcessor',
    window_size=20,
    by_epoch=True
)

vis_backends = [
    dict(type='LocalVisBackend'),
    # Uncomment to enable TensorBoard
    # dict(type='TensorboardVisBackend', save_dir='work_dirs/tf_logs')
]
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=vis_backends
)

log_level = 'INFO'

# =============================================================================
# PRETRAINED MODEL LOADING
# =============================================================================
# TODO: Choose pretrained weights:
# Option 1: Load from URL (recommended)
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth'

# Option 2: Load from local file
# load_from = 'checkpoints/slowfast_r50_kinetics400.pth'

# Option 3: Train from scratch (not recommended)
# load_from = None

# Resume from checkpoint if training interrupted
resume = False

# =============================================================================
# AUTOMATIC LEARNING RATE SCALING
# =============================================================================
# If you change batch size or number of GPUs, this will automatically
# scale the learning rate proportionally
auto_scale_lr = dict(
    enable=True,
    base_batch_size=64  # Base: 8 GPUs x 8 batch_size = 64
)

# =============================================================================
# CUSTOM SETTINGS
# =============================================================================

# Environment settings
env_cfg = dict(
    cudnn_benchmark=False,  # Set to True for fixed input size (faster)
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# Random seed for reproducibility
randomness = dict(seed=0, deterministic=False)

# =============================================================================
# NOTES AND TIPS
# =============================================================================
"""
MEMORY OPTIMIZATION TIPS:
- Reduce batch_size if OOM: 8 → 4 → 2
- Reduce clip_len: 32 → 16
- Increase frame_interval: 2 → 4
- Use mixed precision: add --amp flag

HYPERPARAMETER TUNING:
- Learning rate: Most important! Try 0.001, 0.005, 0.01, 0.05
- Batch size: Larger = more stable, but needs more memory
- Epochs: 30-50 for small datasets, 50-100 for large datasets
- Dropout: Increase (0.6-0.8) for small datasets to prevent overfitting

DATA AUGMENTATION (if needed):
Add to train_pipeline after 'Flip':
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='RandomErasing', probability=0.25),

TRAINING COMMANDS:
Single GPU:
    python tools/train.py configs/recognition/slowfast/this_file.py

Multi-GPU (4 GPUs):
    bash tools/dist_train.sh configs/recognition/slowfast/this_file.py 4

With mixed precision:
    python tools/train.py configs/recognition/slowfast/this_file.py --amp

Override config:
    python tools/train.py configs/recognition/slowfast/this_file.py \\
        --cfg-options optim_wrapper.optimizer.lr=0.005

TESTING:
    python tools/test.py \\
        configs/recognition/slowfast/this_file.py \\
        work_dirs/slowfast_custom/best_acc_top1_epoch_*.pth

EXPECTED TRAINING TIME (RTX 3090):
- 1K videos: 2-4 hours
- 10K videos: 1-2 days
- 100K videos: 1-2 weeks

EXPECTED ACCURACY:
Depends heavily on:
- Dataset quality and size
- Task difficulty (fine-grained vs coarse)
- Domain similarity to Kinetics-400
Typical range: 70-95% top-1 accuracy
"""

