"""
PyTorch Dataset for thermal action detection.

This module provides a PyTorch Dataset class for loading 64-frame sequences
from HDF5 files for action detection training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThermalActionDataset(Dataset):
    """
    PyTorch Dataset for thermal action detection.
    
    Loads 64 consecutive frames (32 before + keyframe + 31 after) from HDF5 files
    and returns them with corresponding bounding boxes and action labels.
    """
    
    def __init__(
        self,
        hdf5_root: str,
        ann_file: str,
        transforms=None,
        frame_window: int = 64
    ):
        """
        Initialize dataset.
        
        Args:
            hdf5_root: Path to frames/ directory with .h5 files
            ann_file: COCO JSON annotation file
            transforms: Optional video transforms
            frame_window: Number of frames per sample (default 64)
        """
        self.hdf5_root = Path(hdf5_root)
        self.transforms = transforms
        self.frame_window = frame_window
        self.half_window = frame_window // 2
        
        # Load annotations
        logger.info(f"Loading annotations from: {ann_file}")
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        self.images = coco_data['images']
        self.categories = coco_data['categories']
        
        # Build annotation index: image_id -> list of annotations
        self.ann_index = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.ann_index:
                self.ann_index[image_id] = []
            self.ann_index[image_id].append(ann)
        
        # Open HDF5 files (keep handles open for performance)
        self.hdf5_files = {}
        self._open_hdf5_files()
        
        logger.info(f"Loaded dataset:")
        logger.info(f"  Images: {len(self.images)}")
        logger.info(f"  Annotations: {sum(len(v) for v in self.ann_index.values())}")
        logger.info(f"  Categories: {len(self.categories)}")
        logger.info(f"  Frame window: {self.frame_window}")
    
    def _open_hdf5_files(self):
        """Open all HDF5 files referenced in annotations."""
        # Get unique sensors
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
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a training sample.
        
        Returns:
            frames: [64, 40, 60, 3] float32 array (thermal replicated to 3 channels)
                    64 frames, each 40 height × 60 width, 3 channels
            boxes: [N, 4] float32 tensor (normalized bounding boxes)
            labels: [N] int64 tensor (action class IDs)
            extras: Dictionary with metadata
        """
        # Get image info
        image_info = self.images[idx]
        sensor_id = image_info['sensor_id']
        frame_idx = image_info['frame_idx']
        image_id = image_info['id']
        
        # Load 64 frames from HDF5 (32 before + keyframe + 31 after)
        if sensor_id not in self.hdf5_files:
            raise ValueError(f"HDF5 file not found for sensor: {sensor_id}")
        
        h5_file = self.hdf5_files[sensor_id]
        
        # Load frames [frame_idx-32:frame_idx+32] = 64 frames
        # Thermal frame dimensions: 40 height × 60 width (40 rows × 60 columns)
        start_idx = frame_idx - self.half_window
        end_idx = frame_idx + self.half_window
        
        frames = h5_file['frames'][start_idx:end_idx]  # [64, 40, 60]
        
        # Replicate to 3 channels (model expects RGB-like input)
        frames = np.stack([frames, frames, frames], axis=-1)  # [64, 40, 60, 3]
        
        # Load bounding boxes and labels
        annotations = self.ann_index.get(image_id, [])
        
        if len(annotations) == 0:
            # No annotations for this image (shouldn't happen, but handle gracefully)
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            object_ids = np.zeros((0,), dtype=np.int64)
        else:
            boxes = []
            labels = []
            object_ids = []
            
            for ann in annotations:
                # bbox is [centerX, centerY, width, height] normalized [0-1]
                bbox = ann['bbox']
                boxes.append(bbox)
                labels.append(ann['category_id'])
                object_ids.append(ann['object_id'])
            
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            object_ids = np.array(object_ids, dtype=np.int64)
        
        # Apply transforms if provided
        if self.transforms is not None:
            frames, boxes = self.transforms(frames, boxes)
        
        # Convert to tensors
        frames = torch.from_numpy(frames).float()
        boxes = torch.from_numpy(boxes).float()
        labels = torch.from_numpy(labels).long()
        object_ids = torch.from_numpy(object_ids).long()
        
        # Prepare extras
        extras = {
            'image_id': image_id,
            'sensor_id': sensor_id,
            'timestamp': image_info['timestamp'],
            'frame_idx': frame_idx,
            'object_ids': object_ids
        }
        
        return frames, boxes, labels, extras
    
    def close(self):
        """Close all HDF5 files."""
        for f in self.hdf5_files.values():
            f.close()
        self.hdf5_files.clear()
    
    def __del__(self):
        """Cleanup HDF5 files on deletion."""
        self.close()


class ThermalActionTransform:
    """
    Video transforms for thermal action detection.
    
    Adapted from AVA pipeline but simplified for thermal data.
    """
    
    def __init__(
        self,
        is_train: bool = True,
        flip_prob: float = 0.5,
        normalize: bool = True,
        temp_min: float = 5.0,
        temp_max: float = 45.0
    ):
        """
        Initialize transforms.
        
        Args:
            is_train: Training mode (enables augmentation)
            flip_prob: Probability of horizontal flip
            normalize: Whether to normalize frames
            temp_min: Minimum temperature for windowing (Celsius)
            temp_max: Maximum temperature for windowing (Celsius)
        """
        self.is_train = is_train
        self.flip_prob = flip_prob
        self.normalize = normalize
        self.temp_min = temp_min
        self.temp_max = temp_max
    
    def __call__(
        self,
        frames: np.ndarray,
        boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply transforms to frames and boxes.
        
        Args:
            frames: [T, H, W, 3] float32 array (replicated thermal)
            boxes: [N, 4] float32 array (normalized bboxes in centerXYWH format)
        
        Returns:
            frames: Transformed frames
            boxes: Transformed boxes
        """
        # Normalize temperature to [0, 1] range
        if self.normalize:
            # Frames are in Celsius, window to [temp_min, temp_max]
            frames = np.clip(frames, self.temp_min, self.temp_max)
            frames = (frames - self.temp_min) / (self.temp_max - self.temp_min)
        
        # Random horizontal flip (training only)
        if self.is_train and len(boxes) > 0 and np.random.rand() < self.flip_prob:
            # Flip frames
            frames = frames[:, :, ::-1, :]  # Flip width dimension
            
            # Flip boxes (centerX, centerY, width, height)
            boxes = boxes.copy()
            boxes[:, 0] = 1.0 - boxes[:, 0]  # Flip centerX
        
        return frames, boxes


def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (frames, boxes, labels, extras) tuples
    
    Returns:
        frames_batch: [B, 3, 64, 40, 60] - Batched frames
        boxes_list: List of [N_i, 4] tensors
        labels_list: List of [N_i] tensors
        extras_list: List of metadata dicts
    """
    frames_list = []
    boxes_list = []
    labels_list = []
    extras_list = []
    
    for frames, boxes, labels, extras in batch:
        frames_list.append(frames)
        boxes_list.append(boxes)
        labels_list.append(labels)
        extras_list.append(extras)
    
    # Stack frames [B, T, H, W, 3]
    frames_batch = torch.stack(frames_list, dim=0)
    
    # Transpose to [B, 3, T, H, W] for model input
    frames_batch = frames_batch.permute(0, 4, 1, 2, 3)
    
    # Boxes and labels remain as lists (variable number per sample)
    return frames_batch, boxes_list, labels_list, extras_list


# Example usage
if __name__ == '__main__':
    # Test dataset loading
    dataset = ThermalActionDataset(
        hdf5_root='thermal_action_dataset/frames',
        ann_file='thermal_action_dataset/annotations/train.json'
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test loading first sample
        frames, boxes, labels, extras = dataset[0]
        logger.info(f"Sample 0:")
        logger.info(f"  Frames shape: {frames.shape}")
        logger.info(f"  Boxes shape: {boxes.shape}")
        logger.info(f"  Labels shape: {labels.shape}")
        logger.info(f"  Image ID: {extras['image_id']}")
    
    dataset.close()

