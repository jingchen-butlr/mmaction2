"""
Validate and visualize thermal action detection dataset.

This script validates dataset integrity and creates visualizations to verify
data quality before training.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import h5py
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Action class names
ACTION_CLASSES = [
    'sitting', 'standing', 'walking',
    'lying down-lying with risk', 'lying down-lying on the bed/couch',
    'leaning',
    'transition-normal transition', 'transition-lying with risk transition',
    'transition-lying on the bed transition',
    'lower position-other', 'lower position-kneeling',
    'lower position-bending', 'lower position-crouching',
    'other'
]


class DatasetValidator:
    """Validate thermal action detection dataset."""
    
    def __init__(
        self,
        hdf5_dir: str,
        annotations_dir: str,
        output_dir: str = "thermal_action_dataset/statistics"
    ):
        """
        Initialize validator.
        
        Args:
            hdf5_dir: Directory containing HDF5 frame files
            annotations_dir: Directory containing COCO annotation files
            output_dir: Output directory for validation reports
        """
        self.hdf5_dir = Path(hdf5_dir)
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load annotations
        self.train_data = self._load_json(self.annotations_dir / 'train.json')
        self.val_data = self._load_json(self.annotations_dir / 'val.json')
        
        # Open HDF5 files
        self.hdf5_files = {}
        self._open_hdf5_files()
    
    def _load_json(self, path: Path) -> Dict:
        """Load JSON file."""
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return {}
        with open(path, 'r') as f:
            return json.load(f)
    
    def _open_hdf5_files(self):
        """Open all HDF5 files."""
        h5_files = list(self.hdf5_dir.glob("*.h5"))
        logger.info(f"Opening {len(h5_files)} HDF5 files")
        
        for h5_path in h5_files:
            sensor_id = h5_path.stem
            try:
                self.hdf5_files[sensor_id] = h5py.File(h5_path, 'r')
            except Exception as e:
                logger.error(f"Failed to open {h5_path}: {e}")
    
    def validate_temporal_coverage(self) -> Dict:
        """Validate that all keyframes have required temporal context."""
        logger.info("\n" + "="*80)
        logger.info("VALIDATING TEMPORAL COVERAGE")
        logger.info("="*80)
        
        issues = []
        
        for split_name, data in [('train', self.train_data), ('val', self.val_data)]:
            if not data:
                continue
            
            logger.info(f"\nChecking {split_name} split ({len(data['images'])} images)")
            
            for img in data['images']:
                sensor_id = img['sensor_id']
                frame_idx = img['frame_idx']
                
                if sensor_id not in self.hdf5_files:
                    issues.append({
                        'split': split_name,
                        'image_id': img['id'],
                        'issue': 'HDF5 file not found',
                        'sensor_id': sensor_id
                    })
                    continue
                
                total_frames = len(self.hdf5_files[sensor_id]['frames'])
                
                # Check if ±32 frames exist
                if frame_idx < 32 or frame_idx >= total_frames - 32:
                    issues.append({
                        'split': split_name,
                        'image_id': img['id'],
                        'issue': 'Insufficient temporal context',
                        'frame_idx': frame_idx,
                        'total_frames': total_frames
                    })
        
        logger.info(f"\n✅ Temporal coverage check complete")
        logger.info(f"   Issues found: {len(issues)}")
        
        return {'issues': issues, 'total_checked': len(self.train_data.get('images', [])) + len(self.val_data.get('images', []))}
    
    def validate_annotations(self) -> Dict:
        """Validate annotation format and values."""
        logger.info("\n" + "="*80)
        logger.info("VALIDATING ANNOTATIONS")
        logger.info("="*80)
        
        issues = []
        
        for split_name, data in [('train', self.train_data), ('val', self.val_data)]:
            if not data:
                continue
            
            logger.info(f"\nChecking {split_name} annotations ({len(data.get('annotations', []))} total)")
            
            for ann in data.get('annotations', []):
                # Check bounding box
                bbox = ann['bbox']
                if len(bbox) != 4:
                    issues.append({
                        'split': split_name,
                        'annotation_id': ann['id'],
                        'issue': 'Invalid bbox format',
                        'bbox': bbox
                    })
                    continue
                
                # Check bbox values (should be normalized [0, 1])
                cx, cy, w, h = bbox
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    issues.append({
                        'split': split_name,
                        'annotation_id': ann['id'],
                        'issue': 'Bbox out of range',
                        'bbox': bbox
                    })
                
                # Check category ID
                cat_id = ann['category_id']
                if not (0 <= cat_id < 14):
                    issues.append({
                        'split': split_name,
                        'annotation_id': ann['id'],
                        'issue': 'Invalid category ID',
                        'category_id': cat_id
                    })
        
        logger.info(f"\n✅ Annotation validation complete")
        logger.info(f"   Issues found: {len(issues)}")
        
        return {'issues': issues}
    
    def validate_frame_data(self, num_samples: int = 10) -> Dict:
        """Validate frame data integrity."""
        logger.info("\n" + "="*80)
        logger.info(f"VALIDATING FRAME DATA ({num_samples} samples)")
        logger.info("="*80)
        
        issues = []
        
        # Sample random images from train split
        if self.train_data and 'images' in self.train_data:
            images = self.train_data['images'][:num_samples]
            
            for img in images:
                sensor_id = img['sensor_id']
                frame_idx = img['frame_idx']
                
                if sensor_id not in self.hdf5_files:
                    continue
                
                try:
                    # Load 64 frames
                    start_idx = frame_idx - 32
                    end_idx = frame_idx + 32
                    frames = self.hdf5_files[sensor_id]['frames'][start_idx:end_idx]
                    
                    # Check shape (should be 64 frames x height x width)
                    # Get expected dimensions from image metadata
                    expected_height = img['height']
                    expected_width = img['width']
                    if frames.shape != (64, expected_height, expected_width):
                        issues.append({
                            'image_id': img['id'],
                            'issue': 'Invalid frame shape',
                            'expected': (64, expected_height, expected_width),
                            'actual': frames.shape
                        })
                    
                    # Check for NaN or inf
                    if np.any(np.isnan(frames)) or np.any(np.isinf(frames)):
                        issues.append({
                            'image_id': img['id'],
                            'issue': 'NaN or Inf values in frames'
                        })
                    
                    # Check temperature range (should be reasonable Celsius values)
                    temp_min = np.min(frames)
                    temp_max = np.max(frames)
                    
                    if temp_min < -50 or temp_max > 100:
                        issues.append({
                            'image_id': img['id'],
                            'issue': 'Unreasonable temperature range',
                            'min': float(temp_min),
                            'max': float(temp_max)
                        })
                    
                except Exception as e:
                    issues.append({
                        'image_id': img['id'],
                        'issue': f'Frame loading error: {str(e)}'
                    })
        
        logger.info(f"\n✅ Frame data validation complete")
        logger.info(f"   Issues found: {len(issues)}")
        
        return {'issues': issues, 'samples_checked': num_samples}
    
    def generate_statistics(self) -> Dict:
        """Generate dataset statistics."""
        logger.info("\n" + "="*80)
        logger.info("GENERATING STATISTICS")
        logger.info("="*80)
        
        stats = {
            'train': self._compute_split_stats(self.train_data),
            'val': self._compute_split_stats(self.val_data),
            'hdf5': self._compute_hdf5_stats()
        }
        
        # Log statistics
        for split in ['train', 'val']:
            logger.info(f"\n{split.upper()} split:")
            logger.info(f"  Images: {stats[split]['total_images']}")
            logger.info(f"  Annotations: {stats[split]['total_annotations']}")
            logger.info(f"  Avg annotations per image: {stats[split]['avg_annotations_per_image']:.2f}")
            logger.info(f"  Class distribution:")
            for class_name, count in sorted(stats[split]['class_distribution'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    {class_name}: {count}")
        
        logger.info(f"\nHDF5 files:")
        logger.info(f"  Total sensors: {stats['hdf5']['total_sensors']}")
        logger.info(f"  Total frames: {stats['hdf5']['total_frames']}")
        logger.info(f"  Corrupted frames: {stats['hdf5']['corrupted_frames']}")
        logger.info(f"  Total size (MB): {stats['hdf5']['total_size_mb']:.2f}")
        
        return stats
    
    def _compute_split_stats(self, data: Dict) -> Dict:
        """Compute statistics for a data split."""
        if not data:
            return {}
        
        images = data.get('images', [])
        annotations = data.get('annotations', [])
        
        # Class distribution
        class_dist = {}
        for ann in annotations:
            class_name = ann.get('category_name', ACTION_CLASSES[ann['category_id']])
            class_dist[class_name] = class_dist.get(class_name, 0) + 1
        
        return {
            'total_images': len(images),
            'total_annotations': len(annotations),
            'avg_annotations_per_image': len(annotations) / len(images) if images else 0,
            'class_distribution': class_dist
        }
    
    def _compute_hdf5_stats(self) -> Dict:
        """Compute HDF5 file statistics."""
        total_frames = 0
        corrupted_frames = 0
        total_size = 0
        
        for sensor_id, h5_file in self.hdf5_files.items():
            total_frames += len(h5_file['frames'])
            corrupted_frames += h5_file.attrs.get('corrupted_count', 0)
            
            # Get file size
            h5_path = self.hdf5_dir / f"{sensor_id}.h5"
            if h5_path.exists():
                total_size += h5_path.stat().st_size
        
        return {
            'total_sensors': len(self.hdf5_files),
            'total_frames': total_frames,
            'corrupted_frames': corrupted_frames,
            'total_size_mb': total_size / (1024 * 1024)
        }
    
    def visualize_samples(self, num_samples: int = 6):
        """Visualize sample frames with annotations."""
        logger.info("\n" + "="*80)
        logger.info(f"VISUALIZING {num_samples} SAMPLES")
        logger.info("="*80)
        
        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        if not self.train_data or 'images' not in self.train_data:
            logger.warning("No training data to visualize")
            return
        
        # Build annotation index
        ann_index = {}
        for ann in self.train_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in ann_index:
                ann_index[image_id] = []
            ann_index[image_id].append(ann)
        
        # Sample random images
        images = self.train_data['images'][:num_samples]
        
        for i, img in enumerate(images):
            sensor_id = img['sensor_id']
            frame_idx = img['frame_idx']
            image_id = img['id']
            
            if sensor_id not in self.hdf5_files:
                continue
            
            try:
                # Load keyframe (middle of 64-frame sequence)
                frames = self.hdf5_files[sensor_id]['frames'][frame_idx-32:frame_idx+32]
                keyframe = frames[32]  # The annotated frame
                
                # Normalize for visualization
                frame_norm = (keyframe - 5.0) / (45.0 - 5.0)
                frame_norm = np.clip(frame_norm * 255, 0, 255).astype(np.uint8)
                
                # Create figure
                fig, ax = plt.subplots(1, 1, figsize=(8, 12))
                ax.imshow(frame_norm, cmap='hot', interpolation='nearest')
                
                # Draw bounding boxes and labels
                annotations = ann_index.get(image_id, [])
                for ann in annotations:
                    bbox = ann['bbox']  # [centerX, centerY, width, height] normalized
                    cx, cy, w, h = bbox
                    
                    # Convert to pixel coordinates
                    # Frame is 40 height × 60 width (40 rows × 60 columns)
                    # X is along width (60), Y is along height (40)
                    img_width = img['width']   # 60
                    img_height = img['height'] # 40
                    
                    x1 = (cx - w/2) * img_width   # Multiply by width (60)
                    y1 = (cy - h/2) * img_height  # Multiply by height (40)
                    width = w * img_width
                    height = h * img_height
                    
                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2, edgecolor='cyan', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    class_name = ann['category_name']
                    ax.text(
                        x1, y1 - 2,
                        class_name,
                        color='cyan',
                        fontsize=6,
                        bbox=dict(facecolor='black', alpha=0.7, pad=1)
                    )
                
                ax.set_title(f"{image_id}\n{len(annotations)} annotations", fontsize=10)
                ax.axis('off')
                
                # Save
                output_path = vis_dir / f"sample_{i:03d}_{sensor_id}.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"  Saved: {output_path.name}")
                
            except Exception as e:
                logger.error(f"  Failed to visualize {image_id}: {e}")
        
        logger.info(f"\n✅ Visualizations saved to: {vis_dir}")
    
    def generate_report(self, num_samples: int = 6):
        """Generate complete validation report."""
        logger.info("\n" + "="*80)
        logger.info("GENERATING VALIDATION REPORT")
        logger.info("="*80)
        
        # Run all validations, pass num_samples to validate_frame_data
        temporal_result = self.validate_temporal_coverage()
        annotation_result = self.validate_annotations()
        frame_result = self.validate_frame_data(num_samples=num_samples)
        stats = self.generate_statistics()
        
        # Compile report
        report = {
            'temporal_coverage': temporal_result,
            'annotation_validation': annotation_result,
            'frame_data_validation': frame_result,
            'statistics': stats
        }
        
        # Save report (convert numpy types to Python types for JSON serialization)
        report_path = self.output_dir / 'validation_report.json'
        
        # Convert numpy types to Python types recursively
        def convert_to_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        report = convert_to_json_serializable(report)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✅ Validation report saved to: {report_path}")
        
        # Generate visualizations, pass num_samples to visualize_samples
        self.visualize_samples(num_samples=num_samples)
        
        # Print summary
        total_issues = (
            len(temporal_result.get('issues', [])) +
            len(annotation_result.get('issues', [])) +
            len(frame_result.get('issues', []))
        )
        
        logger.info("\n" + "="*80)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total issues found: {total_issues}")
        if total_issues == 0:
            logger.info("✅ Dataset is ready for training!")
        else:
            logger.warning("⚠️  Please review issues in validation_report.json")
        logger.info("="*80)
    
    def close(self):
        """Close all HDF5 files."""
        for f in self.hdf5_files.values():
            f.close()


def main():
    parser = argparse.ArgumentParser(description='Validate thermal action detection dataset')
    parser.add_argument(
        '--hdf5-dir',
        default='thermal_action_dataset/frames',
        help='Directory containing HDF5 frame files'
    )
    parser.add_argument(
        '--annotations-dir',
        default='thermal_action_dataset/annotations',
        help='Directory containing COCO annotation files'
    )
    parser.add_argument(
        '--output-dir',
        default='thermal_action_dataset/statistics',
        help='Output directory for validation reports'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=6,
        help='Number of samples to visualize'
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = DatasetValidator(
        hdf5_dir=args.hdf5_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir
    )
    
    # Generate report, pass num_samples to generate_report
    validator.generate_report(num_samples=args.num_samples)
    
    # Close HDF5 files
    validator.close()


if __name__ == '__main__':
    main()

