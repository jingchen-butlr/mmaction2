"""
Thermal HDF5 Dataset for MMAction2

This dataset loader integrates thermal action detection data stored in HDF5 format
with MMAction2's SlowFast training pipeline.

Key Features:
- Loads 40×60 thermal frames from HDF5 files
- Resizes to 256×384 for model input (maintaining aspect ratio)
- 64-frame temporal window (32 before + keyframe + 31 after)
- Handles class imbalance with weighted sampling
- Strong data augmentation for small dataset (314 train samples)

Author: Generated for MMAction2 thermal finetuning
Date: 2025-11-12
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from mmaction.registry import DATASETS
from mmengine.fileio import load

logger = logging.getLogger(__name__)


@DATASETS.register_module()
class ThermalHDF5Dataset(Dataset):
    """
    Thermal action detection dataset loading from HDF5 files.
    
    This dataset is designed for action recognition from thermal sensor data.
    Each sample consists of 64 consecutive frames centered around an annotated keyframe.
    
    Dataset Structure:
        - Thermal frames: 40 height × 60 width (stored in HDF5)
        - Model input: 256 height × 384 width (resized, maintains aspect ratio)
        - Temporal window: 64 frames
        - Classes: 14 action categories
    
    Args:
        ann_file (str): Path to COCO-style annotation JSON file
        data_prefix (dict): Dict with 'hdf5' key pointing to HDF5 directory
        pipeline (list): List of data augmentation/processing operations
        frame_window (int): Number of frames per sample (default: 64)
        test_mode (bool): Whether in test mode
        **kwargs: Additional arguments passed to parent class
    """
    
    def __init__(
        self,
        ann_file: str,
        data_prefix: dict,
        pipeline: list,
        frame_window: int = 64,
        test_mode: bool = False,
        **kwargs
    ):
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.pipeline = pipeline
        self.frame_window = frame_window
        self.half_window = frame_window // 2
        self.test_mode = test_mode
        
        # Get HDF5 root directory
        self.hdf5_root = Path(data_prefix.get('hdf5', ''))
        if not self.hdf5_root.exists():
            raise ValueError(f"HDF5 directory not found: {self.hdf5_root}")
        
        # Load annotations
        logger.info(f"Loading thermal dataset from: {ann_file}")
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.categories = self.coco_data['categories']
        
        # Build annotation index: image_id -> list of annotations
        self.ann_index = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.ann_index:
                self.ann_index[image_id] = []
            self.ann_index[image_id].append(ann)
        
        # Open HDF5 files (keep handles open for performance)
        self.hdf5_files = {}
        self._open_hdf5_files()
        
        # Build data list for MMAction2 compatibility
        self.data_list = self._build_data_list()
        
        logger.info(f"Thermal dataset initialized:")
        logger.info(f"  Images: {len(self.images)}")
        logger.info(f"  Annotations: {len(self.coco_data['annotations'])}")
        logger.info(f"  Classes: {len(self.categories)}")
        logger.info(f"  Mode: {'Test' if test_mode else 'Train'}")
        logger.info(f"  Frame window: {self.frame_window}")
        logger.info(f"  HDF5 root: {self.hdf5_root}")
    
    def _open_hdf5_files(self):
        """Open all HDF5 files for sensors in dataset."""
        sensors = set(img['sensor_id'] for img in self.images)
        
        logger.info(f"Opening HDF5 files for {len(sensors)} sensors")
        for sensor_id in sensors:
            h5_path = self.hdf5_root / f"{sensor_id}.h5"
            if not h5_path.exists():
                logger.warning(f"  HDF5 file not found: {h5_path}")
                continue
            
            try:
                self.hdf5_files[sensor_id] = h5py.File(h5_path, 'r')
                total_frames = len(self.hdf5_files[sensor_id]['frames'])
                logger.info(f"  {sensor_id}: {total_frames} frames")
            except Exception as e:
                logger.error(f"  Failed to open {h5_path}: {e}")
    
    def _build_data_list(self) -> List[Dict]:
        """
        Build data list compatible with MMAction2 pipeline.
        
        Returns:
            List of data info dictionaries
        """
        data_list = []
        
        for img_info in self.images:
            image_id = img_info['id']
            annotations = self.ann_index.get(image_id, [])
            
            if len(annotations) == 0:
                continue
            
            # Get labels from annotations
            labels = [ann['category_id'] for ann in annotations]
            
            # For action recognition, use the most common label
            # (or first label if you want to keep all annotations)
            label = max(set(labels), key=labels.count) if labels else 0
            
            data_info = {
                'image_id': image_id,
                'sensor_id': img_info['sensor_id'],
                'frame_idx': img_info['frame_idx'],
                'timestamp': img_info['timestamp'],
                'label': label,
                'annotations': annotations,
                'frame_shape': (40, 60),  # Original thermal shape
                'total_frames': self.frame_window
            }
            
            data_list.append(data_info)
        
        return data_list
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def get_data_info(self, idx: int) -> Dict:
        """
        Get data info for a given index.
        
        This is called by MMAction2's pipeline system.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with data information
        """
        return self.data_list[idx].copy()
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a training sample.
        
        This method loads thermal frames from HDF5 and prepares them
        for the MMAction2 pipeline.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'imgs' and 'label' keys
        """
        data_info = self.get_data_info(idx)
        
        # Load frames from HDF5
        sensor_id = data_info['sensor_id']
        frame_idx = data_info['frame_idx']
        
        if sensor_id not in self.hdf5_files:
            raise ValueError(f"HDF5 file not found for sensor: {sensor_id}")
        
        h5_file = self.hdf5_files[sensor_id]
        
        # Load 64 consecutive frames
        start_idx = frame_idx - self.half_window
        end_idx = frame_idx + self.half_window
        
        # Handle boundary cases
        total_frames_in_file = len(h5_file['frames'])
        if start_idx < 0 or end_idx > total_frames_in_file:
            # Pad with edge frames if out of bounds
            start_idx = max(0, start_idx)
            end_idx = min(total_frames_in_file, end_idx)
        
        # Load frames: shape [T, 40, 60]
        frames = h5_file['frames'][start_idx:end_idx]
        frames = np.array(frames, dtype=np.float32)
        
        # Pad if we got fewer frames than expected
        if len(frames) < self.frame_window:
            pad_before = max(0, self.half_window - frame_idx)
            pad_after = self.frame_window - len(frames) - pad_before
            frames = np.pad(
                frames,
                ((pad_before, pad_after), (0, 0), (0, 0)),
                mode='edge'
            )
        
        # Replicate to 3 channels: [T, H, W] -> [T, H, W, 3]
        # This makes thermal data compatible with RGB-pretrained models
        frames = np.stack([frames, frames, frames], axis=-1)
        
        # Normalize temperature range (thermal data is in Celsius)
        # Typical range: 5°C to 45°C for indoor environments
        frames = np.clip(frames, 5.0, 45.0)
        frames = (frames - 5.0) / (45.0 - 5.0)  # Normalize to [0, 1]
        
        # Scale to uint8 range for compatibility with augmentation pipeline
        frames = (frames * 255).astype(np.uint8)
        
        # Prepare data dict for pipeline
        results = {
            'imgs': frames,  # [T, H, W, 3] in uint8
            'label': data_info['label'],
            'img_shape': frames.shape[1:3],  # (H, W)
            'original_shape': frames.shape[1:3],
            'modality': 'Thermal',
            'num_clips': 1,
            'clip_len': self.frame_window,
            **data_info
        }
        
        # Apply pipeline (augmentation + formatting)
        results = self._apply_pipeline(results)
        
        return results
    
    def _apply_pipeline(self, results: Dict) -> Dict:
        """
        Apply data pipeline transformations.
        
        Args:
            results: Data dictionary
            
        Returns:
            Transformed data dictionary
        """
        from mmengine.registry import TRANSFORMS
        
        # Build pipeline if not already built
        if not hasattr(self, '_built_pipeline'):
            self._built_pipeline = []
            for transform_cfg in self.pipeline:
                if isinstance(transform_cfg, dict):
                    transform = TRANSFORMS.build(transform_cfg)
                else:
                    transform = transform_cfg
                self._built_pipeline.append(transform)
        
        # Apply transforms
        for transform in self._built_pipeline:
            results = transform(results)
            if results is None:
                return None
        return results
    
    def prepare_train_frames(self, idx: int) -> Dict:
        """
        Prepare training frames (MMAction2 compatibility method).
        
        Args:
            idx: Sample index
            
        Returns:
            Prepared data dictionary
        """
        return self.__getitem__(idx)
    
    def prepare_test_frames(self, idx: int) -> Dict:
        """
        Prepare test frames (MMAction2 compatibility method).
        
        Args:
            idx: Sample index
            
        Returns:
            Prepared data dictionary
        """
        return self.__getitem__(idx)
    
    def evaluate(
        self,
        results: List,
        metrics: Union[str, List[str]] = 'top_k_accuracy',
        metric_options: Optional[Dict] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ) -> Dict:
        """
        Evaluate the dataset.
        
        Args:
            results: List of result dictionaries
            metrics: Metrics to compute
            metric_options: Options for metrics
            logger: Logger instance
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of evaluation results
        """
        # This will be handled by MMAction2's evaluation hooks
        # For now, return empty dict
        return {}
    
    def close(self):
        """Close all HDF5 file handles."""
        for f in self.hdf5_files.values():
            f.close()
        self.hdf5_files.clear()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass


def get_class_weights(ann_file: str) -> torch.Tensor:
    """
    Calculate class weights for handling imbalanced dataset.
    
    Args:
        ann_file: Path to annotation JSON file
        
    Returns:
        Tensor of class weights for loss weighting
    """
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Count annotations per class
    class_counts = {}
    for ann in data['annotations']:
        cat_id = ann['category_id']
        class_counts[cat_id] = class_counts.get(cat_id, 0) + 1
    
    num_classes = len(data['categories'])
    total_annotations = len(data['annotations'])
    
    # Calculate weights: inverse frequency
    weights = torch.zeros(num_classes)
    for cat_id in range(num_classes):
        count = class_counts.get(cat_id, 1)
        weights[cat_id] = total_annotations / (num_classes * count)
    
    logger.info(f"Class weights calculated:")
    for cat_id, cat in enumerate(data['categories']):
        logger.info(f"  {cat['name']}: {weights[cat_id]:.3f} (count: {class_counts.get(cat_id, 0)})")
    
    return weights


# Example usage
if __name__ == '__main__':
    # Test dataset
    dataset = ThermalHDF5Dataset(
        ann_file='ThermalDataGen/thermal_action_dataset/annotations/train.json',
        data_prefix={'hdf5': 'ThermalDataGen/thermal_action_dataset/frames'},
        pipeline=[],  # Empty pipeline for testing
        test_mode=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Images shape: {sample['imgs'].shape}")
        print(f"Label: {sample['label']}")
    
    dataset.close()

