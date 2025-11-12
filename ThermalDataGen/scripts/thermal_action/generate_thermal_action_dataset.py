"""
Main script to generate thermal action detection dataset.

This script orchestrates the complete dataset generation process:
1. Create HDF5 frame storage from TDengine
2. Convert annotations to COCO format
3. Validate dataset integrity
4. Generate statistics report
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from create_hdf5_frames import ThermalFrameHDF5Creator
from convert_annotations_to_coco import ThermalAnnotationConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Generate thermal action detection dataset from TDengine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Generate full dataset from all annotation files
  python generate_thermal_action_dataset.py \\
    --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \\
    --output-dir thermal_action_dataset \\
    --val-split 0.2
  
  # Generate with custom TDengine connection
  python generate_thermal_action_dataset.py \\
    --annotation-files DataAnnotationQA/Data/Gen3_Annotated_Data_MVP/Annotations/*.json \\
    --tdengine-host 192.168.1.100 \\
    --tdengine-port 6041
        """
    )
    
    # Input/output arguments
    parser.add_argument(
        '--annotation-files',
        nargs='+',
        required=True,
        help='Annotation JSON files (supports wildcards)'
    )
    parser.add_argument(
        '--output-dir',
        default='thermal_action_dataset',
        help='Output directory for dataset (default: thermal_action_dataset)'
    )
    
    # Dataset split arguments
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Validation split ratio (default: 0.2)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # HDF5 arguments
    parser.add_argument(
        '--buffer-frames',
        type=int,
        default=128,
        help='Number of frames to fetch before/after annotations (default: 128)'
    )
    parser.add_argument(
        '--compression',
        default='gzip',
        choices=['gzip', 'lzf', 'none'],
        help='HDF5 compression method (default: gzip)'
    )
    parser.add_argument(
        '--compression-level',
        type=int,
        default=4,
        help='HDF5 compression level 0-9 (default: 4)'
    )
    
    # TDengine arguments
    parser.add_argument(
        '--tdengine-host',
        default='35.90.244.93',
        help='TDengine host (default: 35.90.244.93 - remote server)'
    )
    parser.add_argument(
        '--tdengine-port',
        type=int,
        default=6041,
        help='TDengine REST API port (default: 6041)'
    )
    parser.add_argument(
        '--tdengine-database',
        default='thermal_sensors_pilot',
        help='TDengine database name (default: thermal_sensors_pilot)'
    )
    
    # Control arguments
    parser.add_argument(
        '--skip-hdf5',
        action='store_true',
        help='Skip HDF5 creation (assumes HDF5 files already exist)'
    )
    parser.add_argument(
        '--skip-annotations',
        action='store_true',
        help='Skip annotation conversion (assumes COCO files already exist)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / 'frames'
    annotations_dir = output_dir / 'annotations'
    stats_dir = output_dir / 'statistics'
    
    # Create directories
    for d in [frames_dir, annotations_dir, stats_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Start time
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info("THERMAL ACTION DETECTION DATASET GENERATION")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Annotation files: {len(args.annotation_files)}")
    logger.info(f"Val split: {args.val_split}")
    logger.info(f"Random seed: {args.random_seed}")
    logger.info("="*80)
    
    annotation_files = [Path(f) for f in args.annotation_files]
    
    # Step 1: Create HDF5 frame storage
    if not args.skip_hdf5:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: CREATE HDF5 FRAME STORAGE")
        logger.info("="*80)
        
        tdengine_config = {
            'host': args.tdengine_host,
            'port': args.tdengine_port,
            'database': args.tdengine_database,
            'user': 'root',
            'password': 'taosdata'
        }
        
        hdf5_creator = ThermalFrameHDF5Creator(
            tdengine_config=tdengine_config,
            output_dir=str(frames_dir),
            compression=args.compression if args.compression != 'none' else None,
            compression_level=args.compression_level
        )
        
        sensor_info = hdf5_creator.process_annotation_files(
            annotation_files=annotation_files,
            buffer_frames=args.buffer_frames
        )
        
        # Save sensor info
        sensor_info_path = frames_dir / 'sensor_info.json'
        with open(sensor_info_path, 'w') as f:
            json.dump(sensor_info, f, indent=2)
        
        logger.info(f"\n✅ HDF5 creation complete!")
        logger.info(f"   Sensors: {len(sensor_info)}")
        logger.info(f"   Total frames: {sum(s['total_frames'] for s in sensor_info.values())}")
        logger.info(f"   Corrupted frames: {sum(s['corrupted_frames'] for s in sensor_info.values())}")
    else:
        logger.info("\n⏭️  Skipping HDF5 creation (--skip-hdf5)")
    
    # Step 2: Convert annotations to COCO format
    if not args.skip_annotations:
        logger.info("\n" + "="*80)
        logger.info("STEP 2: CONVERT ANNOTATIONS TO COCO FORMAT")
        logger.info("="*80)
        
        converter = ThermalAnnotationConverter(
            hdf5_dir=str(frames_dir),
            output_dir=str(annotations_dir),
            val_split=args.val_split,
            random_seed=args.random_seed
        )
        
        train_data, val_data = converter.convert_annotations(annotation_files)
        converter.save_coco_files(train_data, val_data)
        converter.close()
        
        logger.info(f"\n✅ Annotation conversion complete!")
        logger.info(f"   Train images: {len(train_data['images'])}")
        logger.info(f"   Val images: {len(val_data['images'])}")
        logger.info(f"   Train annotations: {len(train_data['annotations'])}")
        logger.info(f"   Val annotations: {len(val_data['annotations'])}")
    else:
        logger.info("\n⏭️  Skipping annotation conversion (--skip-annotations)")
    
    # Step 3: Generate summary report
    logger.info("\n" + "="*80)
    logger.info("STEP 3: GENERATE SUMMARY REPORT")
    logger.info("="*80)
    
    # Load statistics
    if (annotations_dir / 'train.json').exists():
        with open(annotations_dir / 'train.json', 'r') as f:
            train_data = json.load(f)
        with open(annotations_dir / 'val.json', 'r') as f:
            val_data = json.load(f)
    
    if (frames_dir / 'sensor_info.json').exists():
        with open(frames_dir / 'sensor_info.json', 'r') as f:
            sensor_info = json.load(f)
    else:
        sensor_info = {}
    
    # Create summary report
    summary = {
        'generation_time': datetime.now().isoformat(),
        'duration_seconds': (datetime.now() - start_time).total_seconds(),
        'configuration': {
            'val_split': args.val_split,
            'random_seed': args.random_seed,
            'buffer_frames': args.buffer_frames,
            'compression': args.compression,
            'compression_level': args.compression_level
        },
        'frames': {
            'total_sensors': len(sensor_info),
            'total_frames': sum(s['total_frames'] for s in sensor_info.values()),
            'corrupted_frames': sum(s['corrupted_frames'] for s in sensor_info.values()),
            'storage_directory': str(frames_dir)
        },
        'annotations': {
            'train_images': len(train_data['images']) if 'train_data' in locals() else 0,
            'val_images': len(val_data['images']) if 'val_data' in locals() else 0,
            'train_annotations': len(train_data['annotations']) if 'train_data' in locals() else 0,
            'val_annotations': len(val_data['annotations']) if 'val_data' in locals() else 0,
            'total_classes': 14,
            'storage_directory': str(annotations_dir)
        }
    }
    
    summary_path = output_dir / 'dataset_info.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✅ Summary report saved to: {summary_path}")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("DATASET GENERATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Duration: {summary['duration_seconds']:.1f} seconds")
    logger.info(f"\nDataset statistics:")
    logger.info(f"  Sensors: {summary['frames']['total_sensors']}")
    logger.info(f"  Total frames: {summary['frames']['total_frames']}")
    logger.info(f"  Train images: {summary['annotations']['train_images']}")
    logger.info(f"  Val images: {summary['annotations']['val_images']}")
    logger.info(f"  Train annotations: {summary['annotations']['train_annotations']}")
    logger.info(f"  Val annotations: {summary['annotations']['val_annotations']}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Validate dataset: python scripts/thermal_action/validate_dataset.py")
    logger.info(f"  2. Train SlowFast model (see cursor_readme/THERMAL_ACTION_TRAINING_GUIDE.md)")
    logger.info("="*80)


if __name__ == '__main__':
    main()

