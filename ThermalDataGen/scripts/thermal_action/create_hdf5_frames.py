"""
Create HDF5 frame storage from TDengine thermal sensor data.

This module fetches all frames for each sensor from TDengine and stores them
chronologically in HDF5 files for efficient sequential access during training.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
import calendar

import numpy as np
import h5py
import requests
import zlib
import base64

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "DataAnnotationQA" / "src"))
from data_pipeline.thermal_preprocessor import ThermalFramePreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThermalFrameHDF5Creator:
    """Create HDF5 files from TDengine thermal sensor data."""
    
    def __init__(
        self,
        tdengine_config: Optional[Dict] = None,
        output_dir: str = "thermal_action_dataset/frames",
        compression: str = "gzip",
        compression_level: int = 4
    ):
        """
        Initialize HDF5 creator.
        
        Args:
            tdengine_config: TDengine connection config (host, port, database)
            output_dir: Output directory for HDF5 files
            compression: HDF5 compression method
            compression_level: Compression level (0-9)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.compression = compression
        self.compression_level = compression_level
        
        # TDengine configuration (use same defaults as existing code)
        if tdengine_config is None:
            tdengine_config = {
                'host': '35.90.244.93',  # Same as DataAnnotationQA/src/data_pipeline/thermal_dataset.py
                'port': 6041,
                'database': 'thermal_sensors_pilot',
                'user': 'root',
                'password': 'taosdata'
            }
        self.tdengine_config = tdengine_config
        self.base_url = f"http://{tdengine_config['host']}:{tdengine_config['port']}/rest/sql"
        
        # Thermal preprocessor for validation and conversion
        self.preprocessor = ThermalFramePreprocessor(
            window_min=5.0,
            window_max=45.0,
            auto_convert=True
        )
        
        logger.info(f"Initialized HDF5 creator with output dir: {self.output_dir}")
    
    def _mac_to_table_name(self, mac_address: str) -> str:
        """Convert MAC address to TDengine table name."""
        return f"sensor_{mac_address.replace(':', '_')}"
    
    def _query_tdengine(self, sql: str) -> Optional[List]:
        """Execute TDengine SQL query."""
        try:
            url = f"{self.base_url}/{self.tdengine_config['database']}"
            response = requests.post(
                url,
                auth=(self.tdengine_config['user'], self.tdengine_config['password']),
                data=sql,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"TDengine query failed: {response.status_code}")
                return None
            
            result = response.json()
            
            if result.get('code') != 0:
                logger.error(f"TDengine error: {result.get('desc')}")
                return None
            
            return result.get('data', [])
            
        except Exception as e:
            logger.error(f"TDengine query exception: {e}")
            return None
    
    def _decompress_frame(
        self,
        frame_data: bytes,
        width: int,
        height: int
    ) -> Optional[np.ndarray]:
        """
        Decompress and process thermal frame.
        
        Returns:
            frame: [height, width] float32 array in Celsius, or None if corrupted
        """
        try:
            # Decompress
            if isinstance(frame_data, str):
                compressed_data = base64.b64decode(frame_data)
            else:
                compressed_data = frame_data
            
            decompressed_data = zlib.decompress(compressed_data)
            
            # Parse format
            if len(decompressed_data) == width * height * 4:
                # float32 (Celsius)
                frame = np.frombuffer(decompressed_data, dtype=np.float32).reshape(height, width)
            elif len(decompressed_data) == width * height * 2:
                # int16 (deciKelvin)
                frame = np.frombuffer(decompressed_data, dtype=np.int16).reshape(height, width)
                frame = frame.astype(np.float32) / 10.0 - 273.15
            else:
                logger.warning(f"Unknown frame format: size={len(decompressed_data)}")
                return None
            
            # Make a copy (don't flip - annotations are in original orientation)
            frame = frame.copy()
            
            # Process frame: auto-detect format, convert if needed, validate
            frame = self.preprocessor.process_frame(frame, validate=False)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Frame decompression failed: {str(e)[:100]}")
            return None
    
    def fetch_frames_for_sensor(
        self,
        sensor_id: str,
        mac_address: str,
        min_timestamp_ms: int,
        max_timestamp_ms: int,
        buffer_frames: int = 128
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Fetch all frames for a sensor in the given time range.
        
        Args:
            sensor_id: Sensor ID (e.g., "SL18_R1")
            mac_address: Sensor MAC address
            min_timestamp_ms: Minimum timestamp in milliseconds
            max_timestamp_ms: Maximum timestamp in milliseconds
            buffer_frames: Number of frames to fetch before/after range
        
        Returns:
            frames: [N, 40, 60] float32 array (N frames, 40 height, 60 width)
            timestamps: [N] int64 array (milliseconds)
            frame_seqs: [N] int64 array
            corrupted_count: Number of corrupted frames replaced with zeros
        """
        logger.info(f"Fetching frames for sensor {sensor_id} ({mac_address})")
        logger.info(f"  Time range: {min_timestamp_ms} to {max_timestamp_ms}")
        
        # Convert timestamps to TDengine format (milliseconds)
        # Add buffer: Assume ~10 FPS, so 128 frames ≈ 12.8 seconds
        buffer_ms = buffer_frames * 100  # 100ms per frame at 10 FPS
        query_min = min_timestamp_ms - buffer_ms
        query_max = max_timestamp_ms + buffer_ms
        
        # Convert to TDengine timestamp format
        min_dt = datetime.fromtimestamp(query_min / 1000.0, tz=timezone.utc)
        max_dt = datetime.fromtimestamp(query_max / 1000.0, tz=timezone.utc)
        
        table_name = self._mac_to_table_name(mac_address)
        
        # Query frames ordered by frame_seq for chronological order
        sql = f"""
        SELECT ts, frame_data, width, height, frame_seq
        FROM {table_name}
        WHERE ts >= '{min_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}'
          AND ts <= '{max_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}'
        ORDER BY frame_seq ASC
        """
        
        logger.info(f"  Querying TDengine table: {table_name}")
        data = self._query_tdengine(sql)
        
        if not data:
            logger.error(f"  No frames found for sensor {sensor_id}")
            return np.array([]), np.array([]), np.array([]), 0
        
        logger.info(f"  Retrieved {len(data)} frames from TDengine")
        
        # Process frames
        frames_list = []
        timestamps_list = []
        frame_seqs_list = []
        corrupted_count = 0
        
        for row in data:
            ts_str = row[0]  # Timestamp string
            frame_data = row[1]
            width = row[2] if len(row) > 2 else 60
            height = row[3] if len(row) > 3 else 40
            frame_seq = row[4] if len(row) > 4 else 0
            
            # Parse timestamp to milliseconds (handle ISO 8601 format from TDengine)
            try:
                # TDengine returns timestamp in ISO 8601 format: '2025-10-14T22:50:47.375Z'
                # Clean it up to match the format we can parse
                ts_clean = ts_str[:23].replace('T', ' ').replace('Z', '')
                dt = datetime.strptime(ts_clean, '%Y-%m-%d %H:%M:%S.%f')
                # Convert to milliseconds (UTC)
                timestamp_ms = int(calendar.timegm(dt.timetuple()) * 1000 + dt.microsecond // 1000)
            except Exception as e:
                logger.warning(f"  Failed to parse timestamp '{ts_str}': {e}")
                continue
            
            # Decompress frame
            frame = self._decompress_frame(frame_data, width, height)
            
            if frame is None:
                # Create all-zero frame for corrupted data
                frame = np.zeros((height, width), dtype=np.float32)
                corrupted_count += 1
            
            frames_list.append(frame)
            timestamps_list.append(timestamp_ms)
            frame_seqs_list.append(frame_seq)
        
        if not frames_list:
            logger.error(f"  No valid frames for sensor {sensor_id}")
            return np.array([]), np.array([]), np.array([]), 0
        
        frames = np.stack(frames_list, axis=0)  # [N, 40, 60] (N, height, width)
        timestamps = np.array(timestamps_list, dtype=np.int64)
        frame_seqs = np.array(frame_seqs_list, dtype=np.int64)
        
        logger.info(f"  Processed {len(frames)} frames ({corrupted_count} corrupted, replaced with zeros)")
        
        return frames, timestamps, frame_seqs, corrupted_count
    
    def create_hdf5(
        self,
        sensor_id: str,
        mac_address: str,
        frames: np.ndarray,
        timestamps: np.ndarray,
        frame_seqs: np.ndarray,
        corrupted_count: int
    ) -> Path:
        """
        Create HDF5 file for sensor.
        
        Args:
            sensor_id: Sensor ID
            mac_address: Sensor MAC address
            frames: [N, 40, 60] float32 array (N frames, 40 height, 60 width)
            timestamps: [N] int64 array (milliseconds)
            frame_seqs: [N] int64 array
            corrupted_count: Number of corrupted frames
        
        Returns:
            output_path: Path to created HDF5 file
        """
        output_path = self.output_dir / f"{sensor_id}.h5"
        
        logger.info(f"Creating HDF5 file: {output_path}")
        logger.info(f"  Shape: {frames.shape}")
        logger.info(f"  Dtype: {frames.dtype}")
        logger.info(f"  Compression: {self.compression} level {self.compression_level}")
        
        with h5py.File(output_path, 'w') as f:
            # Store frames with compression
            # Chunk shape must be (T, H, W) = (min(64, N), 40, 60)
            chunk_size = (min(64, len(frames)), frames.shape[1], frames.shape[2])
            f.create_dataset(
                'frames',
                data=frames,
                dtype=np.float32,
                compression=self.compression,
                compression_opts=self.compression_level,
                chunks=chunk_size  # Optimize for 64-frame reads
            )
            
            # Store timestamps
            f.create_dataset(
                'timestamps',
                data=timestamps,
                dtype=np.int64
            )
            
            # Store frame sequences
            f.create_dataset(
                'frame_seqs',
                data=frame_seqs,
                dtype=np.int64
            )
            
            # Store metadata
            # Thermal frame dimensions: 40 height × 60 width (40 rows × 60 columns)
            f.attrs['sensor_id'] = sensor_id
            f.attrs['mac_address'] = mac_address
            f.attrs['total_frames'] = len(frames)
            f.attrs['corrupted_count'] = corrupted_count
            f.attrs['frame_width'] = 60  # Columns
            f.attrs['frame_height'] = 40  # Rows
            f.attrs['min_timestamp'] = int(timestamps[0])
            f.attrs['max_timestamp'] = int(timestamps[-1])
            f.attrs['compression'] = self.compression
            f.attrs['compression_level'] = self.compression_level
            f.attrs['created_at'] = datetime.now(timezone.utc).isoformat()
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Created HDF5 file: {file_size_mb:.2f} MB")
        
        return output_path
    
    def process_annotation_files(
        self,
        annotation_files: List[Path],
        buffer_frames: int = 128
    ) -> Dict[str, Dict]:
        """
        Process all annotation files and create HDF5 files for each sensor.
        
        Args:
            annotation_files: List of annotation JSON files
            buffer_frames: Number of frames to fetch before/after annotations
        
        Returns:
            sensor_info: Dictionary with sensor metadata
        """
        logger.info(f"Processing {len(annotation_files)} annotation files")
        
        # Group annotations by sensor and find time ranges
        sensor_data = {}  # {sensor_id: {mac, min_ts, max_ts}}
        
        for ann_file in annotation_files:
            logger.info(f"Reading annotations from: {ann_file.name}")
            
            with open(ann_file, 'r') as f:
                for line in f:
                    ann = json.loads(line.strip())
                    sensor_id = ann['data_id']
                    mac_address = ann['mac_address']
                    timestamp = ann['data_time']
                    
                    if sensor_id not in sensor_data:
                        sensor_data[sensor_id] = {
                            'mac_address': mac_address,
                            'min_timestamp': timestamp,
                            'max_timestamp': timestamp,
                            'annotation_count': 0
                        }
                    else:
                        sensor_data[sensor_id]['min_timestamp'] = min(
                            sensor_data[sensor_id]['min_timestamp'], timestamp
                        )
                        sensor_data[sensor_id]['max_timestamp'] = max(
                            sensor_data[sensor_id]['max_timestamp'], timestamp
                        )
                    
                    sensor_data[sensor_id]['annotation_count'] += 1
        
        logger.info(f"Found {len(sensor_data)} unique sensors")
        
        # Create HDF5 files for each sensor
        sensor_info = {}
        
        for sensor_id, data in sensor_data.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing sensor: {sensor_id}")
            logger.info(f"  MAC: {data['mac_address']}")
            logger.info(f"  Annotations: {data['annotation_count']}")
            logger.info(f"  Time range: {data['min_timestamp']} to {data['max_timestamp']}")
            
            # Fetch frames from TDengine
            frames, timestamps, frame_seqs, corrupted_count = self.fetch_frames_for_sensor(
                sensor_id=sensor_id,
                mac_address=data['mac_address'],
                min_timestamp_ms=data['min_timestamp'],
                max_timestamp_ms=data['max_timestamp'],
                buffer_frames=buffer_frames
            )
            
            if len(frames) == 0:
                logger.warning(f"  Skipping sensor {sensor_id}: no frames retrieved")
                continue
            
            # Create HDF5 file
            output_path = self.create_hdf5(
                sensor_id=sensor_id,
                mac_address=data['mac_address'],
                frames=frames,
                timestamps=timestamps,
                frame_seqs=frame_seqs,
                corrupted_count=corrupted_count
            )
            
            sensor_info[sensor_id] = {
                'hdf5_path': str(output_path),
                'mac_address': data['mac_address'],
                'total_frames': len(frames),
                'corrupted_frames': corrupted_count,
                'annotation_count': data['annotation_count'],
                'min_timestamp': int(timestamps[0]),
                'max_timestamp': int(timestamps[-1])
            }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"HDF5 creation complete!")
        logger.info(f"  Total sensors: {len(sensor_info)}")
        logger.info(f"  Output directory: {self.output_dir}")
        
        return sensor_info


def main():
    parser = argparse.ArgumentParser(description='Create HDF5 frame storage from TDengine')
    parser.add_argument(
        '--annotation-files',
        nargs='+',
        required=True,
        help='Annotation JSON files'
    )
    parser.add_argument(
        '--output-dir',
        default='thermal_action_dataset/frames',
        help='Output directory for HDF5 files'
    )
    parser.add_argument(
        '--buffer-frames',
        type=int,
        default=128,
        help='Number of frames to fetch before/after annotations'
    )
    parser.add_argument(
        '--tdengine-host',
        default='35.90.244.93',
        help='TDengine host (default: 35.90.244.93 - remote server)'
    )
    parser.add_argument(
        '--tdengine-port',
        type=int,
        default=6041,
        help='TDengine REST API port'
    )
    parser.add_argument(
        '--tdengine-database',
        default='thermal_sensors_pilot',
        help='TDengine database name'
    )
    
    args = parser.parse_args()
    
    # TDengine configuration
    tdengine_config = {
        'host': args.tdengine_host,
        'port': args.tdengine_port,
        'database': args.tdengine_database,
        'user': 'root',
        'password': 'taosdata'
    }
    
    # Create HDF5 creator
    creator = ThermalFrameHDF5Creator(
        tdengine_config=tdengine_config,
        output_dir=args.output_dir
    )
    
    # Process annotation files
    annotation_files = [Path(f) for f in args.annotation_files]
    sensor_info = creator.process_annotation_files(
        annotation_files=annotation_files,
        buffer_frames=args.buffer_frames
    )
    
    # Save sensor info
    info_path = Path(args.output_dir) / 'sensor_info.json'
    with open(info_path, 'w') as f:
        json.dump(sensor_info, f, indent=2)
    logger.info(f"\nSaved sensor info to: {info_path}")


if __name__ == '__main__':
    main()

