"""
Custom Dataset Preparation Script for MMAction2

This script helps organize your video dataset into the format required by MMAction2.
It will:
1. Split videos into train/val sets
2. Create annotation files (train_list.txt, val_list.txt)
3. Optionally organize videos into class folders
4. Generate class index mapping

Usage:
    python prepare_custom_dataset.py \\
        --video-dir /path/to/videos \\
        --output-dir data/my_dataset \\
        --val-ratio 0.2 \\
        --organize-by-class

Author: Generated for MMAction2 finetuning
"""

import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_video_files(video_dir: str, extensions: List[str] = None) -> List[Path]:
    """
    Get all video files from a directory.
    
    Args:
        video_dir: Directory containing videos
        extensions: List of video extensions to include
        
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    video_dir = Path(video_dir)
    video_files = []
    
    for ext in extensions:
        video_files.extend(video_dir.glob(f'**/*{ext}'))
        video_files.extend(video_dir.glob(f'**/*{ext.upper()}'))
    
    logger.info(f"Found {len(video_files)} video files")
    return video_files


def parse_labels_from_filename(
    video_files: List[Path],
    label_extractor: str = 'prefix'
) -> Dict[Path, str]:
    """
    Extract labels from video filenames.
    
    Args:
        video_files: List of video file paths
        label_extractor: How to extract labels
            - 'prefix': class_videoname.mp4
            - 'parent': class_folder/videoname.mp4
            - 'suffix': videoname_class.mp4
            
    Returns:
        Dictionary mapping video paths to class labels
    """
    label_mapping = {}
    
    for video_file in video_files:
        if label_extractor == 'prefix':
            # Extract label from filename prefix (e.g., class1_video001.mp4)
            label = video_file.stem.split('_')[0]
        elif label_extractor == 'parent':
            # Extract label from parent directory name
            label = video_file.parent.name
        elif label_extractor == 'suffix':
            # Extract label from filename suffix (e.g., video001_class1.mp4)
            label = video_file.stem.split('_')[-1]
        else:
            raise ValueError(f"Unknown label_extractor: {label_extractor}")
        
        label_mapping[video_file] = label
    
    return label_mapping


def load_labels_from_csv(
    csv_file: str,
    video_col: str = 'video',
    label_col: str = 'label'
) -> Dict[str, str]:
    """
    Load labels from a CSV file.
    
    Args:
        csv_file: Path to CSV file
        video_col: Column name for video filenames
        label_col: Column name for labels
        
    Returns:
        Dictionary mapping video names to labels
    """
    df = pd.read_csv(csv_file)
    label_mapping = dict(zip(df[video_col], df[label_col]))
    logger.info(f"Loaded labels for {len(label_mapping)} videos from {csv_file}")
    return label_mapping


def create_class_index(labels: List[str], output_file: str) -> Dict[str, int]:
    """
    Create class index mapping (class_name -> class_id).
    
    Args:
        labels: List of class labels
        output_file: Path to save class index file
        
    Returns:
        Dictionary mapping class names to indices
    """
    unique_labels = sorted(set(labels))
    class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Save to file
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for label, idx in class_to_idx.items():
            f.write(f"{idx} {label}\n")
    
    logger.info(f"Created class index with {len(class_to_idx)} classes")
    logger.info(f"Saved to {output_file}")
    
    return class_to_idx


def split_train_val(
    video_files: List[Path],
    label_mapping: Dict[Path, str],
    val_ratio: float = 0.2,
    stratify: bool = True,
    seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """
    Split videos into train and validation sets.
    
    Args:
        video_files: List of video file paths
        label_mapping: Dictionary mapping videos to labels
        val_ratio: Ratio of validation set (0.0-1.0)
        stratify: Whether to stratify split by class
        seed: Random seed
        
    Returns:
        Tuple of (train_files, val_files)
    """
    random.seed(seed)
    
    if stratify:
        # Group videos by class
        class_videos = defaultdict(list)
        for video_file in video_files:
            label = label_mapping[video_file]
            class_videos[label].append(video_file)
        
        # Split each class
        train_files = []
        val_files = []
        
        for label, videos in class_videos.items():
            random.shuffle(videos)
            n_val = max(1, int(len(videos) * val_ratio))
            val_files.extend(videos[:n_val])
            train_files.extend(videos[n_val:])
        
        logger.info(f"Stratified split: {len(train_files)} train, {len(val_files)} val")
    else:
        # Random split
        video_files_shuffled = video_files.copy()
        random.shuffle(video_files_shuffled)
        n_val = int(len(video_files) * val_ratio)
        val_files = video_files_shuffled[:n_val]
        train_files = video_files_shuffled[n_val:]
        
        logger.info(f"Random split: {len(train_files)} train, {len(val_files)} val")
    
    return train_files, val_files


def organize_videos(
    video_files: List[Path],
    label_mapping: Dict[Path, str],
    output_dir: Path,
    split: str = 'train',
    copy_files: bool = True
) -> None:
    """
    Organize videos into class folders.
    
    Args:
        video_files: List of video file paths
        label_mapping: Dictionary mapping videos to labels
        output_dir: Output directory
        split: 'train' or 'val'
        copy_files: If True, copy files; if False, create symlinks
    """
    video_output_dir = output_dir / f'videos_{split}'
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    for video_file in tqdm(video_files, desc=f"Organizing {split} videos"):
        label = label_mapping[video_file]
        class_dir = video_output_dir / label
        class_dir.mkdir(exist_ok=True)
        
        dst = class_dir / video_file.name
        
        if copy_files:
            if not dst.exists():
                shutil.copy2(video_file, dst)
        else:
            if not dst.exists():
                dst.symlink_to(video_file.absolute())
    
    logger.info(f"Organized videos in {video_output_dir}")


def create_annotation_file(
    video_files: List[Path],
    label_mapping: Dict[Path, str],
    class_to_idx: Dict[str, int],
    output_file: str,
    relative_to: Optional[Path] = None
) -> None:
    """
    Create annotation file for MMAction2.
    
    Args:
        video_files: List of video file paths
        label_mapping: Dictionary mapping videos to labels
        class_to_idx: Dictionary mapping class names to indices
        output_file: Path to save annotation file
        relative_to: Make paths relative to this directory
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for video_file in sorted(video_files):
            label_name = label_mapping[video_file]
            label_idx = class_to_idx[label_name]
            
            if relative_to:
                try:
                    video_path = video_file.relative_to(relative_to)
                except ValueError:
                    video_path = video_file
            else:
                video_path = video_file
            
            # Format: video_path label_index
            f.write(f"{video_path} {label_idx}\n")
    
    logger.info(f"Created annotation file: {output_file}")
    logger.info(f"  - {len(video_files)} videos")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare custom dataset for MMAction2'
    )
    
    # Input/Output
    parser.add_argument(
        '--video-dir',
        type=str,
        required=True,
        help='Directory containing video files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for organized dataset'
    )
    parser.add_argument(
        '--label-csv',
        type=str,
        default=None,
        help='CSV file containing labels (optional)'
    )
    
    # Label extraction
    parser.add_argument(
        '--label-extractor',
        type=str,
        default='prefix',
        choices=['prefix', 'parent', 'suffix'],
        help='How to extract labels from filenames'
    )
    parser.add_argument(
        '--video-col',
        type=str,
        default='video',
        help='Column name for video in CSV'
    )
    parser.add_argument(
        '--label-col',
        type=str,
        default='label',
        help='Column name for label in CSV'
    )
    
    # Split settings
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Ratio of validation set (0.0-1.0)'
    )
    parser.add_argument(
        '--stratify',
        action='store_true',
        help='Stratify split by class'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Organization
    parser.add_argument(
        '--organize-by-class',
        action='store_true',
        help='Organize videos into class folders'
    )
    parser.add_argument(
        '--copy-files',
        action='store_true',
        help='Copy files instead of creating symlinks'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video files
    logger.info("Step 1: Finding video files...")
    video_files = get_video_files(video_dir)
    
    if not video_files:
        logger.error(f"No video files found in {video_dir}")
        return
    
    # Get labels
    logger.info("Step 2: Extracting labels...")
    if args.label_csv:
        # Load from CSV
        label_mapping_dict = load_labels_from_csv(
            args.label_csv,
            args.video_col,
            args.label_col
        )
        # Map to file paths
        label_mapping = {}
        for video_file in video_files:
            video_name = video_file.name
            if video_name in label_mapping_dict:
                label_mapping[video_file] = label_mapping_dict[video_name]
            else:
                logger.warning(f"No label found for {video_name}, skipping")
    else:
        # Extract from filenames
        label_mapping = parse_labels_from_filename(
            video_files,
            args.label_extractor
        )
    
    # Create class index
    logger.info("Step 3: Creating class index...")
    all_labels = list(label_mapping.values())
    class_to_idx = create_class_index(
        all_labels,
        output_dir / 'annotations' / 'class_index.txt'
    )
    
    # Display class distribution
    label_counts = defaultdict(int)
    for label in all_labels:
        label_counts[label] += 1
    
    logger.info("Class distribution:")
    for label, count in sorted(label_counts.items()):
        logger.info(f"  {label}: {count} videos")
    
    # Split train/val
    logger.info("Step 4: Splitting train/val...")
    train_files, val_files = split_train_val(
        video_files,
        label_mapping,
        args.val_ratio,
        args.stratify,
        args.seed
    )
    
    # Organize videos (optional)
    if args.organize_by_class:
        logger.info("Step 5: Organizing videos into class folders...")
        organize_videos(
            train_files,
            label_mapping,
            output_dir,
            'train',
            args.copy_files
        )
        organize_videos(
            val_files,
            label_mapping,
            output_dir,
            'val',
            args.copy_files
        )
        relative_to = output_dir
    else:
        relative_to = video_dir
    
    # Create annotation files
    logger.info("Step 6: Creating annotation files...")
    create_annotation_file(
        train_files,
        label_mapping,
        class_to_idx,
        output_dir / 'annotations' / 'train_list.txt',
        relative_to
    )
    create_annotation_file(
        val_files,
        label_mapping,
        class_to_idx,
        output_dir / 'annotations' / 'val_list.txt',
        relative_to
    )
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("Dataset preparation completed!")
    logger.info("="*50)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total videos: {len(video_files)}")
    logger.info(f"  - Train: {len(train_files)}")
    logger.info(f"  - Val: {len(val_files)}")
    logger.info(f"Number of classes: {len(class_to_idx)}")
    logger.info("\nNext steps:")
    logger.info("1. Review the generated annotation files")
    logger.info("2. Update config file with your dataset paths")
    logger.info("3. Start training!")
    logger.info("\nExample config settings:")
    logger.info(f"  data_root = '{output_dir}/videos_train'")
    logger.info(f"  data_root_val = '{output_dir}/videos_val'")
    logger.info(f"  ann_file_train = '{output_dir}/annotations/train_list.txt'")
    logger.info(f"  ann_file_val = '{output_dir}/annotations/val_list.txt'")
    logger.info(f"  num_classes = {len(class_to_idx)}")


if __name__ == '__main__':
    main()

