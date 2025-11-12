"""
Verification Script for Thermal SlowFast Integration

This script verifies that all components are properly integrated and ready for training.

Usage:
    python tools/verify_thermal_integration.py

Author: Generated for thermal SlowFast integration
Date: 2025-11-12
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_imports():
    """Verify required imports."""
    logger.info("="*70)
    logger.info("Step 1: Checking imports...")
    logger.info("="*70)
    
    try:
        import mmaction
        logger.info(f"✓ MMAction2: {mmaction.__version__}")
    except ImportError as e:
        logger.error(f"✗ MMAction2 not installed: {e}")
        return False
    
    try:
        import torch
        logger.info(f"✓ PyTorch: {torch.__version__}")
        logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"✓ GPU count: {torch.cuda.device_count()}")
    except ImportError as e:
        logger.error(f"✗ PyTorch not installed: {e}")
        return False
    
    try:
        import h5py
        logger.info(f"✓ h5py: {h5py.__version__}")
    except ImportError as e:
        logger.error(f"✗ h5py not installed: {e}")
        logger.error("  Install: pip install h5py")
        return False
    
    try:
        import numpy
        logger.info(f"✓ NumPy: {numpy.__version__}")
    except ImportError as e:
        logger.error(f"✗ NumPy not installed: {e}")
        return False
    
    return True


def check_custom_dataset():
    """Verify custom dataset class."""
    logger.info("")
    logger.info("="*70)
    logger.info("Step 2: Checking custom dataset...")
    logger.info("="*70)
    
    try:
        from mmaction.datasets import ThermalHDF5Dataset
        logger.info("✓ ThermalHDF5Dataset imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import ThermalHDF5Dataset: {e}")
        return False
    
    try:
        from mmaction.datasets import get_class_weights
        logger.info("✓ get_class_weights imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import get_class_weights: {e}")
        return False
    
    return True


def check_dataset_files():
    """Verify dataset files exist."""
    logger.info("")
    logger.info("="*70)
    logger.info("Step 3: Checking dataset files...")
    logger.info("="*70)
    
    base_dir = Path('ThermalDataGen/thermal_action_dataset')
    
    if not base_dir.exists():
        logger.error(f"✗ Dataset directory not found: {base_dir}")
        return False
    
    logger.info(f"✓ Dataset directory exists: {base_dir}")
    
    # Check annotations
    train_json = base_dir / 'annotations' / 'train.json'
    val_json = base_dir / 'annotations' / 'val.json'
    
    if not train_json.exists():
        logger.error(f"✗ Training annotations not found: {train_json}")
        return False
    logger.info(f"✓ Training annotations found: {train_json}")
    
    if not val_json.exists():
        logger.error(f"✗ Validation annotations not found: {val_json}")
        return False
    logger.info(f"✓ Validation annotations found: {val_json}")
    
    # Check HDF5 files
    frames_dir = base_dir / 'frames'
    if not frames_dir.exists():
        logger.error(f"✗ Frames directory not found: {frames_dir}")
        return False
    
    hdf5_files = list(frames_dir.glob('*.h5'))
    if len(hdf5_files) == 0:
        logger.error(f"✗ No HDF5 files found in: {frames_dir}")
        return False
    
    logger.info(f"✓ Found {len(hdf5_files)} HDF5 files")
    for h5_file in sorted(hdf5_files):
        logger.info(f"  - {h5_file.name}")
    
    return True


def check_dataset_loading():
    """Test dataset loading."""
    logger.info("")
    logger.info("="*70)
    logger.info("Step 4: Testing dataset loading...")
    logger.info("="*70)
    
    try:
        from mmaction.datasets import ThermalHDF5Dataset
        
        dataset = ThermalHDF5Dataset(
            ann_file='ThermalDataGen/thermal_action_dataset/annotations/train.json',
            data_prefix={'hdf5': 'ThermalDataGen/thermal_action_dataset/frames'},
            pipeline=[],
            test_mode=False
        )
        
        logger.info(f"✓ Dataset initialized: {len(dataset)} samples")
        
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"✓ Sample loaded successfully")
            logger.info(f"  - Shape: {sample['imgs'].shape}")
            logger.info(f"  - Label: {sample['label']}")
            logger.info(f"  - Frame range: [{sample['imgs'].min()}, {sample['imgs'].max()}]")
        
        dataset.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_config_file():
    """Verify config file exists and is valid."""
    logger.info("")
    logger.info("="*70)
    logger.info("Step 5: Checking config file...")
    logger.info("="*70)
    
    config_file = Path('configs/recognition/slowfast/slowfast_thermal_finetuning.py')
    
    if not config_file.exists():
        logger.error(f"✗ Config file not found: {config_file}")
        return False
    
    logger.info(f"✓ Config file exists: {config_file}")
    
    try:
        from mmengine import Config
        cfg = Config.fromfile(str(config_file))
        logger.info("✓ Config file is valid")
        logger.info(f"  - Model type: {cfg.model.type}")
        logger.info(f"  - Num classes: {cfg.model.cls_head.num_classes}")
        logger.info(f"  - Batch size: {cfg.train_dataloader.batch_size}")
        logger.info(f"  - Learning rate: {cfg.optim_wrapper.optimizer.lr}")
        logger.info(f"  - Max epochs: {cfg.train_cfg.max_epochs}")
        return True
    except Exception as e:
        logger.error(f"✗ Config file is invalid: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_pretrained_weights():
    """Check if pretrained weights exist."""
    logger.info("")
    logger.info("="*70)
    logger.info("Step 6: Checking pretrained weights...")
    logger.info("="*70)
    
    checkpoint_file = Path('checkpoints/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth')
    
    if checkpoint_file.exists():
        size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Pretrained weights found: {checkpoint_file}")
        logger.info(f"  - Size: {size_mb:.1f} MB")
        return True
    else:
        logger.warning(f"⚠ Pretrained weights not found: {checkpoint_file}")
        logger.warning("  Run: python tools/download_pretrained_slowfast.py")
        return True  # Not critical, just a warning


def main():
    """Main verification function."""
    logger.info("="*70)
    logger.info("  Thermal SlowFast Integration Verification")
    logger.info("="*70)
    logger.info("")
    
    checks = [
        ("Imports", check_imports),
        ("Custom Dataset", check_custom_dataset),
        ("Dataset Files", check_dataset_files),
        ("Dataset Loading", check_dataset_loading),
        ("Config File", check_config_file),
        ("Pretrained Weights", check_pretrained_weights),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            logger.error(f"✗ Unexpected error in {name}: {e}")
            results[name] = False
    
    # Summary
    logger.info("")
    logger.info("="*70)
    logger.info("  Verification Summary")
    logger.info("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info("")
    logger.info(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("")
        logger.info("="*70)
        logger.info("  ✅ All checks passed! Ready to train.")
        logger.info("="*70)
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Download pretrained weights (if not already):")
        logger.info("   python tools/download_pretrained_slowfast.py")
        logger.info("")
        logger.info("2. Start training:")
        logger.info("   bash tools/thermal_quickstart.sh")
        logger.info("   OR")
        logger.info("   python tools/train.py configs/recognition/slowfast/slowfast_thermal_finetuning.py")
        logger.info("")
        return 0
    else:
        logger.error("")
        logger.error("="*70)
        logger.error(f"  ✗ {total - passed} check(s) failed. Please fix above errors.")
        logger.error("="*70)
        logger.error("")
        return 1


if __name__ == '__main__':
    sys.exit(main())

