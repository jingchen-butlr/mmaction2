"""
Download Pretrained SlowFast Weights from MMAction2 Model Zoo

This script downloads the SlowFast R50 model pretrained on Kinetics-400
for finetuning on thermal action recognition dataset.

Usage:
    python tools/download_pretrained_slowfast.py

Author: Generated for thermal SlowFast finetuning
Date: 2025-11-12
"""

import logging
import os
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model Zoo URLs
SLOWFAST_MODELS = {
    'slowfast_r50_4x16_kinetics400': {
        'url': 'https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth',
        'filename': 'slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth',
        'md5': None,  # Add if available
        'description': 'SlowFast R50, 4x16 frames, Kinetics-400 (75.55% top-1)'
    },
    'slowfast_r50_8x8_kinetics400': {
        'url': 'https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth',
        'filename': 'slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth',
        'md5': None,
        'description': 'SlowFast R50, 8x8 frames, Kinetics-400 (76.80% top-1)'
    },
    'slowfast_r101_8x8_kinetics400': {
        'url': 'https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb_20220818-9c0e09bd.pth',
        'filename': 'slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb_20220818-9c0e09bd.pth',
        'md5': None,
        'description': 'SlowFast R101, 8x8 frames, Kinetics-400 (78.65% top-1)'
    }
}

# Default model for thermal finetuning (best balance of accuracy and speed)
DEFAULT_MODEL = 'slowfast_r50_4x16_kinetics400'


def compute_md5(filepath: str, chunk_size: int = 8192) -> str:
    """
    Compute MD5 hash of a file.
    
    Args:
        filepath: Path to file
        chunk_size: Size of chunks to read
        
    Returns:
        MD5 hash string
    """
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def download_file(url: str, dest_path: str, desc: str = None) -> bool:
    """
    Download a file with progress reporting.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading: {desc or url}")
        logger.info(f"  URL: {url}")
        logger.info(f"  Destination: {dest_path}")
        
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                progress = count * block_size / total_size * 100
                if count % 50 == 0:  # Log every 50 blocks
                    logger.info(f"  Progress: {progress:.1f}%")
        
        urlretrieve(url, dest_path, reporthook=progress_hook)
        
        file_size = dest_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"  Downloaded: {file_size:.1f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def download_slowfast_pretrained(
    model_name: str = DEFAULT_MODEL,
    checkpoint_dir: str = 'checkpoints',
    force: bool = False
) -> str:
    """
    Download pretrained SlowFast model.
    
    Args:
        model_name: Name of model to download
        checkpoint_dir: Directory to save checkpoint
        force: Force re-download even if file exists
        
    Returns:
        Path to downloaded checkpoint
    """
    if model_name not in SLOWFAST_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(SLOWFAST_MODELS.keys())}"
        )
    
    model_info = SLOWFAST_MODELS[model_name]
    checkpoint_path = Path(checkpoint_dir) / model_info['filename']
    
    logger.info("="*70)
    logger.info(f"Downloading SlowFast Pretrained Weights")
    logger.info("="*70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Description: {model_info['description']}")
    
    # Check if already exists
    if checkpoint_path.exists() and not force:
        logger.info(f"Checkpoint already exists: {checkpoint_path}")
        logger.info("Use --force to re-download")
        
        # Verify integrity if MD5 available
        if model_info['md5']:
            logger.info("Verifying file integrity...")
            actual_md5 = compute_md5(str(checkpoint_path))
            if actual_md5 == model_info['md5']:
                logger.info("  Integrity check passed ✓")
                return str(checkpoint_path)
            else:
                logger.warning("  Integrity check failed! Re-downloading...")
                force = True
        else:
            return str(checkpoint_path)
    
    # Download
    if download_file(
        model_info['url'],
        str(checkpoint_path),
        model_info['description']
    ):
        logger.info("="*70)
        logger.info(f"Download completed successfully!")
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        logger.info("="*70)
        
        # Update config reference
        logger.info("\nUpdate your config file:")
        logger.info(f"  load_from = '{checkpoint_path}'")
        
        return str(checkpoint_path)
    else:
        raise RuntimeError("Download failed")


def list_available_models():
    """List all available pretrained models."""
    logger.info("Available SlowFast Pretrained Models:")
    logger.info("="*70)
    
    for name, info in SLOWFAST_MODELS.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Description: {info['description']}")
        logger.info(f"  Filename: {info['filename']}")
        logger.info(f"  URL: {info['url']}")
    
    logger.info("\n" + "="*70)
    logger.info(f"Default model for thermal finetuning: {DEFAULT_MODEL}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download pretrained SlowFast weights'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        choices=list(SLOWFAST_MODELS.keys()),
        help='Model name to download'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoint'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if file exists'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available models'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    try:
        checkpoint_path = download_slowfast_pretrained(
            model_name=args.model,
            checkpoint_dir=args.checkpoint_dir,
            force=args.force
        )
        
        logger.info("\nNext steps:")
        logger.info("1. Verify checkpoint exists:")
        logger.info(f"   ls -lh {checkpoint_path}")
        logger.info("\n2. Start training:")
        logger.info("   python tools/train.py configs/recognition/slowfast/slowfast_thermal_finetuning.py")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

