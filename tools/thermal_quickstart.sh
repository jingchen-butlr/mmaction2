#!/bin/bash
# Thermal SlowFast Quick Start Script
#
# This script automates the setup and training of SlowFast on thermal dataset
#
# Usage: bash tools/thermal_quickstart.sh
#
# Author: Generated for thermal action recognition
# Date: 2025-11-12

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================="
echo "  Thermal SlowFast Finetuning - Quick Start"
echo "========================================================================="
echo ""

# Navigate to mmaction2 directory
cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"
echo ""

# Step 1: Verify environment
echo -e "${YELLOW}Step 1: Verifying environment...${NC}"

if ! python -c "import mmaction" 2>/dev/null; then
    echo -e "${RED}✗ MMAction2 not installed${NC}"
    echo "Please install MMAction2 first:"
    echo "  pip install -e ."
    exit 1
fi
echo -e "${GREEN}✓ MMAction2 installed${NC}"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${YELLOW}⚠ CUDA not available. Training will be slow.${NC}"
else
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    echo -e "${GREEN}✓ CUDA available (${GPU_COUNT} GPU(s))${NC}"
fi

# Step 2: Verify thermal dataset
echo ""
echo -e "${YELLOW}Step 2: Verifying thermal dataset...${NC}"

if [ ! -d "ThermalDataGen/thermal_action_dataset" ]; then
    echo -e "${RED}✗ Thermal dataset not found${NC}"
    echo "Expected location: ThermalDataGen/thermal_action_dataset/"
    exit 1
fi

TRAIN_JSON="ThermalDataGen/thermal_action_dataset/annotations/train.json"
VAL_JSON="ThermalDataGen/thermal_action_dataset/annotations/val.json"
FRAMES_DIR="ThermalDataGen/thermal_action_dataset/frames"

if [ ! -f "$TRAIN_JSON" ] || [ ! -f "$VAL_JSON" ] || [ ! -d "$FRAMES_DIR" ]; then
    echo -e "${RED}✗ Dataset files incomplete${NC}"
    exit 1
fi

HDF5_COUNT=$(ls -1 "$FRAMES_DIR"/*.h5 2>/dev/null | wc -l)
echo -e "${GREEN}✓ Thermal dataset found (${HDF5_COUNT} HDF5 files)${NC}"

# Step 3: Verify dataset loading
echo ""
echo -e "${YELLOW}Step 3: Testing dataset loading...${NC}"

python << EOF
import sys
try:
    from mmaction.datasets import ThermalHDF5Dataset
    
    dataset = ThermalHDF5Dataset(
        ann_file='$TRAIN_JSON',
        data_prefix={'hdf5': '$FRAMES_DIR'},
        pipeline=[],
        test_mode=False
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"✓ Sample shape: {sample['imgs'].shape}")
    
    dataset.close()
except Exception as e:
    print(f"✗ Error loading dataset: {e}", file=sys.stderr)
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Dataset loading failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Dataset loading successful${NC}"

# Step 4: Download pretrained weights
echo ""
echo -e "${YELLOW}Step 4: Downloading pretrained weights...${NC}"

CHECKPOINT_FILE="checkpoints/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo -e "${GREEN}✓ Pretrained weights already exist${NC}"
else
    echo "Downloading SlowFast R50 pretrained on Kinetics-400..."
    python tools/download_pretrained_slowfast.py
    
    if [ ! -f "$CHECKPOINT_FILE" ]; then
        echo -e "${RED}✗ Download failed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Pretrained weights downloaded${NC}"
fi

# Step 5: Verify config
echo ""
echo -e "${YELLOW}Step 5: Verifying configuration...${NC}"

CONFIG_FILE="configs/recognition/slowfast/slowfast_thermal_finetuning.py"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ Config file not found: ${CONFIG_FILE}${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Config file exists${NC}"

# Step 6: Training options
echo ""
echo "========================================================================="
echo "  Setup Complete!"
echo "========================================================================="
echo ""
echo "Ready to train SlowFast on thermal dataset."
echo ""
echo "Training options:"
echo ""
echo "1. Single GPU (Recommended):"
echo "   python tools/train.py $CONFIG_FILE"
echo ""
echo "2. Multi-GPU (if available):"
echo "   bash tools/dist_train.sh $CONFIG_FILE 2"
echo ""
echo "3. With mixed precision (memory efficient):"
echo "   python tools/train.py $CONFIG_FILE --amp"
echo ""
echo "4. With custom settings:"
echo "   python tools/train.py $CONFIG_FILE --cfg-options train_dataloader.batch_size=2"
echo ""
echo "========================================================================="
echo ""

# Ask if user wants to start training
read -p "Start training now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${GREEN}Starting training...${NC}"
    echo ""
    
    # Detect GPU count and use appropriate command
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo "Detected $GPU_COUNT GPUs, using distributed training..."
        bash tools/dist_train.sh "$CONFIG_FILE" "$GPU_COUNT" \
            --work-dir work_dirs/thermal_slowfast \
            --seed 42
    else
        echo "Using single GPU training..."
        python tools/train.py "$CONFIG_FILE" \
            --work-dir work_dirs/thermal_slowfast \
            --seed 42
    fi
else
    echo ""
    echo "Training not started. You can manually run:"
    echo "  python tools/train.py $CONFIG_FILE"
    echo ""
fi

echo ""
echo "For monitoring training:"
echo "  tail -f work_dirs/thermal_slowfast/*.log"
echo ""
echo "For TensorBoard:"
echo "  tensorboard --logdir work_dirs/thermal_slowfast/tf_logs"
echo ""

