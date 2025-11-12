"""
Convert thermal sensor annotations to COCO format for action detection.

This module converts existing YOLO-style annotations to COCO format with
AVA-compatible structure for human action detection training.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import h5py

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Action class mapping (14 classes from person subcategories)
ACTION_CLASSES = [
    'sitting',                                    # 0
    'standing',                                   # 1
    'walking',                                    # 2
    'lying down-lying with risk',                 # 3
    'lying down-lying on the bed/couch',          # 4
    'leaning',                                    # 5
    'transition-normal transition',               # 6
    'transition-lying with risk transition',      # 7
    'transition-lying on the bed transition',     # 8
    'lower position-other',                       # 9
    'lower position-kneeling',                    # 10
    'lower position-bending',                     # 11
    'lower position-crouching',                   # 12
    'other'                                       # 13
]

# Create reverse mapping
ACTION_CLASS_TO_ID = {name: idx for idx, name in enumerate(ACTION_CLASSES)}


class ThermalAnnotationConverter:
    """Convert thermal annotations to COCO format."""
    
    def __init__(
        self,
        hdf5_dir: str = "thermal_action_dataset/frames",
        output_dir: str = "thermal_action_dataset/annotations",
        val_split: float = 0.2,
        random_seed: int = 42
    ):
        """
        Initialize annotation converter.
        
        Args:
            hdf5_dir: Directory containing HDF5 frame files
            output_dir: Output directory for COCO JSON files
            val_split: Validation split ratio
            random_seed: Random seed for reproducibility
        """
        self.hdf5_dir = Path(hdf5_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.val_split = val_split
        random.seed(random_seed)
        
        # Load HDF5 files and build timestamp index
        self.hdf5_files = {}
        self.timestamp_to_idx = {}  # {sensor_id: {timestamp: frame_idx}}
        
        logger.info(f"Loading HDF5 files from: {self.hdf5_dir}")
        self._load_hdf5_files()
    
    def _load_hdf5_files(self):
        """Load all HDF5 files and build timestamp index."""
        h5_files = list(self.hdf5_dir.glob("*.h5"))
        logger.info(f"Found {len(h5_files)} HDF5 files")
        
        for h5_path in h5_files:
            sensor_id = h5_path.stem  # e.g., "SL18_R1"
            
            try:
                f = h5py.File(h5_path, 'r')
                self.hdf5_files[sensor_id] = f
                
                # Build timestamp to index mapping
                timestamps = f['timestamps'][:]
                self.timestamp_to_idx[sensor_id] = {
                    int(ts): idx for idx, ts in enumerate(timestamps)
                }
                
                logger.info(f"  Loaded {sensor_id}: {len(timestamps)} frames")
                
            except Exception as e:
                logger.error(f"  Failed to load {h5_path}: {e}")
    
    def _find_frame_idx(self, sensor_id: str, timestamp_ms: int, tolerance_ms: int = 50) -> Optional[int]:
        """
        Find frame index for a given timestamp.
        
        Args:
            sensor_id: Sensor ID
            timestamp_ms: Target timestamp in milliseconds
            tolerance_ms: Tolerance for timestamp matching
        
        Returns:
            frame_idx: Frame index in HDF5 file, or None if not found
        """
        if sensor_id not in self.timestamp_to_idx:
            return None
        
        # Try exact match first
        if timestamp_ms in self.timestamp_to_idx[sensor_id]:
            return self.timestamp_to_idx[sensor_id][timestamp_ms]
        
        # Try nearest match within tolerance
        timestamps = np.array(list(self.timestamp_to_idx[sensor_id].keys()))
        diffs = np.abs(timestamps - timestamp_ms)
        nearest_idx = np.argmin(diffs)
        
        if diffs[nearest_idx] <= tolerance_ms:
            nearest_ts = timestamps[nearest_idx]
            return self.timestamp_to_idx[sensor_id][nearest_ts]
        
        return None
    
    def _validate_temporal_window(self, sensor_id: str, frame_idx: int, window_size: int = 32) -> bool:
        """
        Validate that ±window_size frames exist around the keyframe.
        
        Args:
            sensor_id: Sensor ID
            frame_idx: Frame index
            window_size: Number of frames before/after keyframe (default 32 for 64-frame window)
        
        Returns:
            valid: True if temporal window is complete
        """
        if sensor_id not in self.hdf5_files:
            return False
        
        total_frames = len(self.hdf5_files[sensor_id]['frames'])
        return frame_idx >= window_size and frame_idx < total_frames - window_size
    
    def convert_annotations(
        self,
        annotation_files: List[Path]
    ) -> Tuple[Dict, Dict]:
        """
        Convert annotations to COCO format.
        
        Args:
            annotation_files: List of annotation JSON files
        
        Returns:
            train_data: COCO-format training data
            val_data: COCO-format validation data
        """
        logger.info(f"Converting {len(annotation_files)} annotation files to COCO format")
        
        # Collect all annotations
        all_annotations = []
        
        for ann_file in annotation_files:
            logger.info(f"Reading: {ann_file.name}")
            
            with open(ann_file, 'r') as f:
                for line in f:
                    ann = json.loads(line.strip())
                    all_annotations.append(ann)
        
        logger.info(f"Total annotations: {len(all_annotations)}")
        
        # Filter and process annotations
        images = []
        annotations = []
        image_id_counter = 0
        annotation_id_counter = 0
        
        skipped_no_person = 0
        skipped_no_frame = 0
        skipped_boundary = 0
        
        for ann in all_annotations:
            sensor_id = ann['data_id']
            mac_address = ann['mac_address']
            timestamp_ms = ann['data_time']
            
            # Find frame index
            frame_idx = self._find_frame_idx(sensor_id, timestamp_ms)
            
            if frame_idx is None:
                skipped_no_frame += 1
                continue
            
            # Validate temporal window (need ±32 frames for 64-frame sequence)
            if not self._validate_temporal_window(sensor_id, frame_idx, window_size=32):
                skipped_boundary += 1
                continue
            
            # Filter to only person annotations with action subcategories
            person_annotations = [
                a for a in ann['annotations']
                if a['category'] == 'person' and a['subcategory'] in ACTION_CLASS_TO_ID
            ]
            
            if len(person_annotations) == 0:
                skipped_no_person += 1
                continue
            
            # Create image entry
            # Thermal frame dimensions: 40 height × 60 width (40 rows × 60 columns)
            image_id = f"{sensor_id}_{timestamp_ms}"
            images.append({
                'id': image_id,
                'sensor_id': sensor_id,
                'mac_address': mac_address,
                'timestamp': timestamp_ms,
                'width': 60,  # Columns
                'height': 40,  # Rows
                'frame_idx': frame_idx
            })
            
            # Create annotation entries
            for person_ann in person_annotations:
                bbox = person_ann['bbox']  # [centerX, centerY, width, height] normalized
                subcategory = person_ann['subcategory']
                object_id = person_ann['object_id']
                
                # Convert from YOLO format (center) to COCO format (still normalized)
                # Keep normalized for consistency with original annotations
                annotations.append({
                    'id': annotation_id_counter,
                    'image_id': image_id,
                    'bbox': bbox,  # [centerX, centerY, width, height] normalized [0-1]
                    'category_id': ACTION_CLASS_TO_ID[subcategory],
                    'category_name': subcategory,
                    'object_id': object_id,
                    'attribute_obscured': person_ann.get('attribute_obscured', None)
                })
                annotation_id_counter += 1
        
        logger.info(f"\nProcessing summary:")
        logger.info(f"  Valid samples: {len(images)}")
        logger.info(f"  Total person annotations: {len(annotations)}")
        logger.info(f"  Skipped (no person): {skipped_no_person}")
        logger.info(f"  Skipped (frame not found): {skipped_no_frame}")
        logger.info(f"  Skipped (boundary): {skipped_boundary}")
        
        # Create categories
        categories = [
            {'id': idx, 'name': name}
            for idx, name in enumerate(ACTION_CLASSES)
        ]
        
        # Split into train/val (stratified by sensor)
        # Group images by sensor
        sensor_to_images = {}
        for img in images:
            sensor_id = img['sensor_id']
            if sensor_id not in sensor_to_images:
                sensor_to_images[sensor_id] = []
            sensor_to_images[sensor_id].append(img)
        
        logger.info(f"\nSplitting into train/val (ratio: {1-self.val_split:.1f}/{self.val_split:.1f})")
        
        train_images = []
        val_images = []
        
        for sensor_id, sensor_images in sensor_to_images.items():
            # Shuffle images for this sensor
            random.shuffle(sensor_images)
            
            # Split
            n_val = int(len(sensor_images) * self.val_split)
            val_images.extend(sensor_images[:n_val])
            train_images.extend(sensor_images[n_val:])
            
            logger.info(f"  {sensor_id}: {len(sensor_images)} total, {len(sensor_images)-n_val} train, {n_val} val")
        
        logger.info(f"\nFinal split:")
        logger.info(f"  Train: {len(train_images)} images")
        logger.info(f"  Val: {len(val_images)} images")
        
        # Create train/val image ID sets
        train_image_ids = {img['id'] for img in train_images}
        val_image_ids = {img['id'] for img in val_images}
        
        # Split annotations
        train_annotations = [a for a in annotations if a['image_id'] in train_image_ids]
        val_annotations = [a for a in annotations if a['image_id'] in val_image_ids]
        
        logger.info(f"  Train: {len(train_annotations)} annotations")
        logger.info(f"  Val: {len(val_annotations)} annotations")
        
        # Create COCO data structures
        train_data = {
            'images': train_images,
            'annotations': train_annotations,
            'categories': categories,
            'info': {
                'description': 'Thermal Sensor Human Action Detection Dataset (Training)',
                'version': '1.0',
                'year': 2025
            }
        }
        
        val_data = {
            'images': val_images,
            'annotations': val_annotations,
            'categories': categories,
            'info': {
                'description': 'Thermal Sensor Human Action Detection Dataset (Validation)',
                'version': '1.0',
                'year': 2025
            }
        }
        
        return train_data, val_data
    
    def save_coco_files(
        self,
        train_data: Dict,
        val_data: Dict
    ):
        """Save COCO-format JSON files."""
        train_path = self.output_dir / 'train.json'
        val_path = self.output_dir / 'val.json'
        
        logger.info(f"\nSaving COCO annotations:")
        logger.info(f"  Train: {train_path}")
        logger.info(f"  Val: {val_path}")
        
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(val_path, 'w') as f:
            json.dump(val_data, f, indent=2)
        
        # Save class mapping
        class_mapping_path = self.output_dir / 'class_mapping.json'
        class_mapping = {
            'classes': ACTION_CLASSES,
            'class_to_id': ACTION_CLASS_TO_ID
        }
        with open(class_mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        logger.info(f"  Class mapping: {class_mapping_path}")
        
        # Generate statistics
        self._generate_statistics(train_data, val_data)
    
    def _generate_statistics(self, train_data: Dict, val_data: Dict):
        """Generate dataset statistics."""
        stats_path = self.output_dir.parent / 'statistics' / 'dataset_stats.json'
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Count annotations per class
        train_class_counts = {}
        val_class_counts = {}
        
        for ann in train_data['annotations']:
            cat_id = ann['category_id']
            cat_name = ann['category_name']
            train_class_counts[cat_name] = train_class_counts.get(cat_name, 0) + 1
        
        for ann in val_data['annotations']:
            cat_id = ann['category_id']
            cat_name = ann['category_name']
            val_class_counts[cat_name] = val_class_counts.get(cat_name, 0) + 1
        
        # Compile statistics
        stats = {
            'total_classes': len(ACTION_CLASSES),
            'train': {
                'images': len(train_data['images']),
                'annotations': len(train_data['annotations']),
                'class_distribution': train_class_counts
            },
            'val': {
                'images': len(val_data['images']),
                'annotations': len(val_data['annotations']),
                'class_distribution': val_class_counts
            }
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"\nDataset statistics saved to: {stats_path}")
        logger.info("\nClass distribution (train):")
        for class_name in ACTION_CLASSES:
            count = train_class_counts.get(class_name, 0)
            logger.info(f"  {class_name}: {count}")
    
    def close(self):
        """Close all HDF5 files."""
        for f in self.hdf5_files.values():
            f.close()


def main():
    parser = argparse.ArgumentParser(description='Convert annotations to COCO format')
    parser.add_argument(
        '--annotation-files',
        nargs='+',
        required=True,
        help='Annotation JSON files'
    )
    parser.add_argument(
        '--hdf5-dir',
        default='thermal_action_dataset/frames',
        help='Directory containing HDF5 frame files'
    )
    parser.add_argument(
        '--output-dir',
        default='thermal_action_dataset/annotations',
        help='Output directory for COCO JSON files'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Create converter
    converter = ThermalAnnotationConverter(
        hdf5_dir=args.hdf5_dir,
        output_dir=args.output_dir,
        val_split=args.val_split,
        random_seed=args.random_seed
    )
    
    # Convert annotations
    annotation_files = [Path(f) for f in args.annotation_files]
    train_data, val_data = converter.convert_annotations(annotation_files)
    
    # Save COCO files
    converter.save_coco_files(train_data, val_data)
    
    # Close HDF5 files
    converter.close()
    
    logger.info("\n✅ Annotation conversion complete!")


if __name__ == '__main__':
    main()

