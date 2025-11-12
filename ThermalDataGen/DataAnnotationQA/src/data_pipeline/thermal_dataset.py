"""
PyTorch Custom Dataset for Thermal Sensor Data with Annotations

This module implements a custom PyTorch Dataset that:
1. Reads annotation JSON files
2. Fetches raw thermal data directly from TDengine (in-memory, no disk files)
3. Returns tensors ready for deep learning training
"""

import json
import logging
import requests
import zlib
import struct
import base64
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
from pathlib import Path

from .thermal_preprocessor import ThermalFramePreprocessor

logger = logging.getLogger(__name__)


class TDengineConnector:
    """
    Direct TDengine connection for fetching thermal data into memory.
    No disk I/O - data decompressed directly into numpy arrays.
    """
    
    def __init__(self, host: str = "35.90.244.93", port: int = 6041,
                 user: str = "root", password: str = "taosdata",
                 database: str = "thermal_sensors_pilot",
                 preprocessor: Optional[ThermalFramePreprocessor] = None):
        """
        Initialize TDengine connector.
        
        Args:
            host: TDengine server host
            port: TDengine REST API port
            user: Database user
            password: Database password
            database: Database name
            preprocessor: Optional ThermalFramePreprocessor instance
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.base_url = f"http://{host}:{port}/rest/sql"
        
        # Initialize preprocessor with default settings if not provided
        self.preprocessor = preprocessor or ThermalFramePreprocessor(
            window_min=5.0,
            window_max=45.0,
            auto_convert=True
        )
        
        logger.info(f"Initialized TDengine connector: {host}:{port}/{database}")
        logger.info(f"Using {self.preprocessor}")
    
    def query_frame_by_timestamp(self, mac_address: str, timestamp_ms: int,
                                 tolerance_ms: int = 100) -> Optional[np.ndarray]:
        """
        Query a single frame by timestamp, fetch directly into memory.
        
        Args:
            mac_address: Sensor MAC address (e.g., "02:00:1a:62:51:67")
            timestamp_ms: Timestamp in milliseconds (UTC)
            tolerance_ms: Matching tolerance in milliseconds
            
        Returns:
            Thermal frame as numpy array (40, 60) in Celsius, or None if not found
        """
        # Convert timestamp to TDengine format (UTC)
        from datetime import timezone
        timestamp_sec = timestamp_ms / 1000.0
        dt = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
        ts_str = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Millisecond precision
        
        # Calculate time window
        start_sec = (timestamp_ms - tolerance_ms) / 1000.0
        end_sec = (timestamp_ms + tolerance_ms) / 1000.0
        start_dt = datetime.fromtimestamp(start_sec, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_sec, tz=timezone.utc)
        start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        end_str = end_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Build table name from MAC
        table_name = f"sensor_{mac_address.replace(':', '_')}"
        
        # Query frame data within tolerance window
        sql = f"""
        SELECT ts, frame_data, width, height
        FROM {table_name}
        WHERE ts >= '{start_str}' AND ts <= '{end_str}'
        ORDER BY ts ASC
        LIMIT 1
        """
        
        try:
            url = f"{self.base_url}/{self.database}"
            response = requests.post(
                url,
                auth=(self.user, self.password),
                data=sql,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"HTTP error {response.status_code}")
                return None
            
            result = response.json()
            
            if result.get('code') != 0:
                error_msg = result.get('desc', result.get('msg', 'Unknown error'))
                logger.error(f"Query failed: {error_msg}")
                return None
            
            data = result.get('data', [])
            
            if not data:
                logger.warning(f"No frame found for timestamp {timestamp_ms}")
                return None
            
            # Extract frame data
            row = data[0]
            encoded_frame_data = row[1]  # Compressed frame data
            width = row[2] if len(row) > 2 else 60
            height = row[3] if len(row) > 3 else 40
            
            # Decompress frame data directly into memory
            frame = self._decompress_frame_data(encoded_frame_data, width, height)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error fetching frame: {e}")
            return None
    
    def query_temporal_frames(
        self,
        mac_address: str,
        timestamp_ms: int,
        window_ms: int = 2000,
        max_gap_ms: int = 1000
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Query 3 consecutive frames using Strategy 2 (wide query with smart selection).
        
        Downloads frames in [timestamp - 2s, timestamp + 2s] window, then finds the best
        3 consecutive frames centered around the target timestamp.
        
        Args:
            mac_address: Sensor MAC address (e.g., "02:00:1a:62:51:67")
            timestamp_ms: Target timestamp in milliseconds
            window_ms: Time window to query (default 2000ms = ±2 seconds)
            max_gap_ms: Maximum acceptable gap between consecutive frames (default 1000ms)
            
        Returns:
            Tuple of (frame_before, frame_target, frame_after) as numpy arrays in Celsius
            Any of these can be None if not found or gaps too large
        """
        table_name = f"sensor_{mac_address.replace(':', '_')}"
        
        # Query wide window of frames around target (±2 seconds)
        start_ms = timestamp_ms - window_ms
        end_ms = timestamp_ms + window_ms
        
        start_dt = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc)
        
        start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        end_str = end_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Query all frames in window (no limit, get everything)
        sql = f"""
        SELECT ts, frame_data, width, height
        FROM {table_name}
        WHERE ts >= '{start_str}' AND ts <= '{end_str}'
        ORDER BY ts ASC
        """
        
        try:
            url = f"{self.base_url}/{self.database}"
            response = requests.post(
                url,
                auth=(self.user, self.password),
                data=sql,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"HTTP error {response.status_code}")
                return None, None, None
            
            result = response.json()
            
            if result.get('code') != 0:
                error_msg = result.get('desc', result.get('msg', 'Unknown error'))
                logger.error(f"Query failed: {error_msg}")
                return None, None, None
            
            data = result.get('data', [])
            
            if not data or len(data) < 3:
                logger.warning(f"Insufficient frames in window: {len(data) if data else 0} frames")
                return None, None, None
            
            # Parse all timestamps and decompress frames
            frames_info = []
            for row in data:
                frame_ts_str = row[0]
                frame_data_encoded = row[1]
                width = row[2] if len(row) > 2 else 60
                height = row[3] if len(row) > 3 else 40
                
                # Parse timestamp
                ts_clean = frame_ts_str[:23].replace('T', ' ').replace('Z', '')
                dt = datetime.strptime(ts_clean, '%Y-%m-%d %H:%M:%S.%f')
                import calendar
                ts_ms = int(calendar.timegm(dt.timetuple()) * 1000 + dt.microsecond // 1000)
                
                frames_info.append({
                    'timestamp_ms': ts_ms,
                    'distance_to_target': abs(ts_ms - timestamp_ms),
                    'frame_data': frame_data_encoded,
                    'width': width,
                    'height': height
                })
            
            # Find all valid 3-frame sequences
            valid_sequences = []
            
            for i in range(len(frames_info) - 2):
                frame_before = frames_info[i]
                frame_target = frames_info[i + 1]
                frame_after = frames_info[i + 2]
                
                gap_before_target = frame_target['timestamp_ms'] - frame_before['timestamp_ms']
                gap_target_after = frame_after['timestamp_ms'] - frame_target['timestamp_ms']
                
                # Check if gaps are acceptable
                if gap_before_target <= max_gap_ms and gap_target_after <= max_gap_ms:
                    # Score this sequence based on:
                    # 1. How close the middle frame is to target (primary)
                    # 2. How small the gaps are (secondary)
                    # 3. How symmetric the gaps are (tertiary)
                    
                    target_closeness_score = frame_target['distance_to_target']
                    gap_size_score = (gap_before_target + gap_target_after) / 2
                    gap_symmetry_score = abs(gap_before_target - gap_target_after)
                    
                    # Weighted combination (prioritize target closeness)
                    total_score = (
                        target_closeness_score * 10.0 +  # Most important
                        gap_size_score * 0.5 +            # Prefer smaller gaps
                        gap_symmetry_score * 0.1          # Slight preference for symmetry
                    )
                    
                    valid_sequences.append({
                        'indices': (i, i+1, i+2),
                        'score': total_score,
                        'gap_before': gap_before_target,
                        'gap_after': gap_target_after,
                    })
            
            if not valid_sequences:
                logger.warning(f"No valid 3-frame sequences found (gaps > {max_gap_ms}ms)")
                return None, None, None
            
            # Select the best sequence (lowest score = best)
            best_sequence = min(valid_sequences, key=lambda s: s['score'])
            before_idx, target_idx, after_idx = best_sequence['indices']
            
            # Decompress frames with corruption handling
            # Try to decompress all 3 frames, skip corrupted ones
            decompressed_frames = []
            for idx, frame_type in [(before_idx, 'before'), (target_idx, 'target'), (after_idx, 'after')]:
                try:
                    frame = self._decompress_frame_data(
                        frames_info[idx]['frame_data'],
                        frames_info[idx]['width'],
                        frames_info[idx]['height']
                    )
                    decompressed_frames.append({'idx': idx, 'frame': frame, 'type': frame_type})
                except Exception as e:
                    # Frame is corrupted, log and skip
                    logger.debug(f"Corrupted {frame_type} frame at index {idx}: {str(e)[:50]}")
                    decompressed_frames.append({'idx': idx, 'frame': None, 'type': frame_type})
            
            # Check what we got
            frame_before = decompressed_frames[0]['frame'] if decompressed_frames[0]['frame'] is not None else None
            frame_target = decompressed_frames[1]['frame'] if decompressed_frames[1]['frame'] is not None else None
            frame_after = decompressed_frames[2]['frame'] if decompressed_frames[2]['frame'] is not None else None
            
            # If target is corrupted, we must fail (target is essential)
            if frame_target is None:
                logger.warning(f"Target frame corrupted, cannot recover")
                return None, None, None
            
            # Fallback strategy for corrupted before/after frames
            # Look for alternative clean frames in the window
            
            if frame_before is None:
                # Before frame corrupted, try to find earlier clean frame
                logger.debug(f"Before frame corrupted, searching for alternative...")
                for alt_idx in range(before_idx - 1, -1, -1):
                    if alt_idx >= len(frames_info):
                        continue
                    gap = frames_info[target_idx]['timestamp_ms'] - frames_info[alt_idx]['timestamp_ms']
                    if gap > max_gap_ms:
                        break  # Too far, stop searching
                    try:
                        frame_before = self._decompress_frame_data(
                            frames_info[alt_idx]['frame_data'],
                            frames_info[alt_idx]['width'],
                            frames_info[alt_idx]['height']
                        )
                        logger.info(f"Recovered: Using frame at -{gap}ms as 'before' (original corrupted)")
                        break
                    except Exception:
                        continue  # This frame also corrupted, try next
            
            if frame_after is None:
                # After frame corrupted, try to find later clean frame
                logger.debug(f"After frame corrupted, searching for alternative...")
                for alt_idx in range(after_idx + 1, len(frames_info)):
                    gap = frames_info[alt_idx]['timestamp_ms'] - frames_info[target_idx]['timestamp_ms']
                    if gap > max_gap_ms:
                        break  # Too far, stop searching
                    try:
                        frame_after = self._decompress_frame_data(
                            frames_info[alt_idx]['frame_data'],
                            frames_info[alt_idx]['width'],
                            frames_info[alt_idx]['height']
                        )
                        logger.info(f"Recovered: Using frame at +{gap}ms as 'after' (original corrupted)")
                        break
                    except Exception:
                        continue  # This frame also corrupted, try next
            
            # Final check
            if frame_before is None or frame_target is None or frame_after is None:
                logger.warning(f"Could not find 3 clean frames within {max_gap_ms}ms window")
                return None, None, None
            
            logger.debug(f"Temporal frames retrieved successfully")
            
            return frame_before, frame_target, frame_after
            
        except Exception as e:
            logger.error(f"Error fetching temporal frames: {e}")
            return None, None, None
    
    def _decompress_frame_data(self, encoded_data: str, width: int = 60, 
                               height: int = 40) -> np.ndarray:
        """
        Decompress frame data from TDengine format into numpy array in memory.
        
        Args:
            encoded_data: Hex or base64 encoded compressed data
            width: Frame width
            height: Frame height
            
        Returns:
            Numpy array (height, width) in Celsius
        """
        try:
            # Decode from hex or base64
            try:
                if all(c in '0123456789abcdefABCDEF' for c in encoded_data):
                    compressed_bytes = bytes.fromhex(encoded_data)
                else:
                    compressed_bytes = base64.b64decode(encoded_data)
            except:
                compressed_bytes = base64.b64decode(encoded_data)
            
            # Decompress with zlib
            decompressed = zlib.decompress(compressed_bytes)
            
            num_pixels = width * height
            
            # Detect format by size
            if len(decompressed) == num_pixels * 2:
                # int16 format (deciKelvin)
                frame_data = struct.unpack(f'{num_pixels}h', decompressed)
                # Convert deciKelvin to Celsius
                frame_celsius = np.array([(val / 10.0) - 273.15 for val in frame_data], dtype=np.float32)
            elif len(decompressed) == num_pixels * 4:
                # float32 format (Celsius)
                frame_data = struct.unpack(f'{num_pixels}f', decompressed)
                frame_celsius = np.array(frame_data, dtype=np.float32)
            else:
                raise ValueError(f"Unexpected data size: {len(decompressed)}")
            
            # Reshape to (height, width)
            frame = frame_celsius.reshape(height, width)
            
            # DON'T flip - annotations were created on original sensor orientation
            # Applying fliplr would break bounding box alignment with annotations
            # Make a copy to avoid negative stride issues with PyTorch
            frame = frame.copy()  # NO FLIP!
            
            # Process frame: auto-detect format, convert if needed, and validate
            frame = self.preprocessor.process_frame(frame, validate=True)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error decompressing frame: {e}")
            raise
    
    def batch_query_frames(self, mac_address: str, timestamps_ms: List[int],
                          tolerance_ms: int = 100) -> Dict[int, np.ndarray]:
        """
        Query multiple frames by timestamps (batch query for efficiency).
        
        Args:
            mac_address: Sensor MAC address
            timestamps_ms: List of timestamps in milliseconds
            tolerance_ms: Matching tolerance
            
        Returns:
            Dictionary mapping timestamp_ms to frame arrays
        """
        frames = {}
        
        # Calculate overall time range (UTC)
        from datetime import timezone
        min_ts = min(timestamps_ms) - tolerance_ms
        max_ts = max(timestamps_ms) + tolerance_ms
        
        min_dt = datetime.fromtimestamp(min_ts / 1000.0, tz=timezone.utc)
        max_dt = datetime.fromtimestamp(max_ts / 1000.0, tz=timezone.utc)
        
        start_str = min_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        end_str = max_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Build table name
        table_name = f"sensor_{mac_address.replace(':', '_')}"
        
        # Query all frames in range
        sql = f"""
        SELECT ts, frame_data, width, height
        FROM {table_name}
        WHERE ts >= '{start_str}' AND ts <= '{end_str}'
        ORDER BY ts ASC
        """
        
        try:
            url = f"{self.base_url}/{self.database}"
            response = requests.post(
                url,
                auth=(self.user, self.password),
                data=sql,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"HTTP error {response.status_code}")
                return frames
            
            result = response.json()
            
            if result.get('code') != 0:
                error_msg = result.get('desc', result.get('msg', 'Unknown error'))
                logger.error(f"Query failed: {error_msg}")
                return frames
            
            data = result.get('data', [])
            logger.info(f"Batch query found {len(data)} frames")
            
            # Log first few timestamps for debugging
            if data and len(data) > 0:
                logger.info(f"First TDengine timestamp: {data[0][0]}")
                logger.info(f"Last TDengine timestamp: {data[-1][0]}")
                logger.info(f"Target timestamps (first 3): {timestamps_ms[:3]}")
            
            # Process each frame
            for row in data:
                frame_ts_str = row[0]
                frame_data = row[1]
                width = row[2] if len(row) > 2 else 60
                height = row[3] if len(row) > 3 else 40
                
                # Parse timestamp (handle both formats: space and T separator)
                # TDengine returns UTC timestamps, parse as UTC
                frame_ts_clean = frame_ts_str[:23].replace('T', ' ')
                frame_dt = datetime.strptime(frame_ts_clean, '%Y-%m-%d %H:%M:%S.%f')
                # Use utcfromtimestamp instead of fromtimestamp to handle UTC correctly
                import calendar
                frame_ts_ms = int(calendar.timegm(frame_dt.timetuple()) * 1000 + frame_dt.microsecond // 1000)
                
                # Debug: log first few timestamps
                if len(frames) < 3:
                    logger.info(f"Frame timestamp: {frame_ts_str} -> {frame_ts_ms}")
                
                # Match to requested timestamps
                for target_ts in timestamps_ms:
                    if abs(frame_ts_ms - target_ts) <= tolerance_ms:
                        # Decompress frame
                        frame = self._decompress_frame_data(frame_data, width, height)
                        frames[target_ts] = frame
                        break
            
            logger.info(f"Matched {len(frames)}/{len(timestamps_ms)} frames")
            return frames
            
        except Exception as e:
            logger.error(f"Batch query error: {e}")
            return frames


class ThermalAnnotationDataset(Dataset):
    """
    PyTorch custom Dataset for thermal sensor data with annotations.
    
    Follows PyTorch Dataset pattern with __init__, __len__, __getitem__.
    Fetches thermal data directly from TDengine into memory (no disk files).
    
    Reference: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """
    
    def __init__(self, annotation_file: str, mac_address: str,
                 tdengine_config: Optional[Dict] = None,
                 transform=None, target_transform=None,
                 cache_frames: bool = True):
        """
        Initialize the dataset.
        
        Args:
            annotation_file: Path to annotation JSON file (one JSON per line)
            mac_address: Sensor MAC address for TDengine queries
            tdengine_config: Optional TDengine connection config
            transform: Optional transform to apply to frames
            target_transform: Optional transform to apply to annotations
            cache_frames: Whether to cache fetched frames in memory
        """
        self.annotation_file = annotation_file
        self.mac_address = mac_address
        self.transform = transform
        self.target_transform = target_transform
        self.cache_frames = cache_frames
        
        # Load annotations from JSON file
        self.annotations = self._load_annotations()
        
        # Initialize TDengine connector
        tdengine_config = tdengine_config or {}
        self.tdengine = TDengineConnector(**tdengine_config)
        
        # Frame cache (in-memory)
        self.frame_cache = {} if cache_frames else None
        
        # Category mapping for labels
        self.category_to_id = {}
        self.id_to_category = {}
        self._build_category_mapping()
        
        logger.info(f"Initialized ThermalAnnotationDataset:")
        logger.info(f"  Annotation file: {annotation_file}")
        logger.info(f"  MAC address: {mac_address}")
        logger.info(f"  Total samples: {len(self.annotations)}")
        logger.info(f"  Categories: {len(self.category_to_id)}")
        logger.info(f"  Cache enabled: {cache_frames}")
    
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from JSON file (one JSON object per line)."""
        annotations = []
        
        with open(self.annotation_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        ann = json.loads(line.strip())
                        annotations.append(ann)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
        
        logger.info(f"Loaded {len(annotations)} annotations from {self.annotation_file}")
        return annotations
    
    def _build_category_mapping(self):
        """Build category to ID mapping for classification tasks."""
        next_id = 0
        
        for ann in self.annotations:
            for obj in ann.get('annotations', []):
                category = obj.get('category', '')
                subcategory = obj.get('subcategory', '')
                full_category = f"{category}/{subcategory}"
                
                if full_category not in self.category_to_id:
                    self.category_to_id[full_category] = next_id
                    self.id_to_category[next_id] = full_category
                    next_id += 1
        
        logger.info(f"Built category mapping with {len(self.category_to_id)} categories")
    
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        Required by PyTorch Dataset.
        """
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a sample from the dataset at the given index.
        Required by PyTorch Dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (thermal_frame_tensor, annotation_dict)
            - thermal_frame_tensor: torch.Tensor of shape (1, H, W) in Celsius
            - annotation_dict: Dict with bboxes, labels, and metadata
        """
        if idx >= len(self.annotations):
            raise IndexError(f"Index {idx} out of range (0-{len(self.annotations)-1})")
        
        # Get annotation for this index
        annotation = self.annotations[idx]
        data_time_ms = annotation['data_time']  # Milliseconds
        
        # Check cache first
        if self.cache_frames and data_time_ms in self.frame_cache:
            frame = self.frame_cache[data_time_ms]
        else:
            # Fetch frame from TDengine directly into memory
            frame = self.tdengine.query_frame_by_timestamp(
                self.mac_address,
                data_time_ms,
                tolerance_ms=100
            )
            
            if frame is None:
                logger.warning(f"Frame not found for timestamp {data_time_ms}, using zeros")
                frame = np.zeros((40, 60), dtype=np.float32)
            
            # Cache if enabled
            if self.cache_frames:
                self.frame_cache[data_time_ms] = frame
        
        # Convert numpy to torch tensor
        # Add channel dimension: (H, W) → (1, H, W)
        frame_tensor = torch.from_numpy(frame).unsqueeze(0)
        
        # Apply transform if provided
        if self.transform:
            frame_tensor = self.transform(frame_tensor)
        
        # Process annotations into training format
        target = self._process_annotation(annotation)
        
        # Apply target transform if provided
        if self.target_transform:
            target = self.target_transform(target)
        
        return frame_tensor, target
    
    def _process_annotation(self, annotation: Dict) -> Dict:
        """
        Process annotation into training-ready format.
        
        Args:
            annotation: Raw annotation dictionary
            
        Returns:
            Processed annotation with tensors for bboxes and labels
        """
        objects = annotation.get('annotations', [])
        
        if not objects:
            # No objects in frame
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'image_id': annotation.get('data_id', ''),
                'timestamp': annotation.get('data_time', 0),
            }
        
        # Extract bboxes and labels
        boxes = []
        labels = []
        
        for obj in objects:
            bbox = obj.get('bbox', [])
            category = obj.get('category', '')
            subcategory = obj.get('subcategory', '')
            
            if len(bbox) == 4:
                boxes.append(bbox)  # Already in YOLO format [cx, cy, w, h]
                
                # Get category ID
                full_category = f"{category}/{subcategory}"
                label_id = self.category_to_id.get(full_category, 0)
                labels.append(label_id)
        
        # Convert to tensors
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        
        return {
            'boxes': boxes_tensor,  # Shape: (N, 4) - YOLO format
            'labels': labels_tensor,  # Shape: (N,)
            'image_id': annotation.get('data_id', ''),
            'timestamp': annotation.get('data_time', 0),
            'num_objects': len(objects),
        }
    
    def get_category_name(self, label_id: int) -> str:
        """Get category name from label ID."""
        return self.id_to_category.get(label_id, 'unknown')
    
    def prefetch_all_frames(self):
        """
        Prefetch all frames from TDengine into memory cache.
        Useful for training to avoid repeated network queries.
        """
        if not self.cache_frames:
            logger.warning("Cache is disabled, prefetch has no effect")
            return
        
        logger.info(f"Prefetching {len(self.annotations)} frames...")
        
        # Collect all timestamps
        timestamps = [ann['data_time'] for ann in self.annotations]
        
        # Batch query for efficiency
        frames = self.tdengine.batch_query_frames(
            self.mac_address,
            timestamps,
            tolerance_ms=100
        )
        
        # Update cache
        self.frame_cache.update(frames)
        
        logger.info(f"Prefetched {len(frames)} frames into cache")
        logger.info(f"Cache hit rate will be: {len(frames)}/{len(timestamps)} ({len(frames)/len(timestamps)*100:.1f}%)")
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            'total_samples': len(self.annotations),
            'num_categories': len(self.category_to_id),
            'categories': self.category_to_id,
            'mac_address': self.mac_address,
            'cache_size': len(self.frame_cache) if self.cache_frames else 0,
            'cached_frames': list(self.frame_cache.keys()) if self.cache_frames else [],
        }


def create_dataloader(annotation_file: str, mac_address: str,
                     batch_size: int = 8, shuffle: bool = True,
                     num_workers: int = 0, prefetch: bool = True,
                     tdengine_config: Optional[Dict] = None,
                     **dataloader_kwargs) -> DataLoader:
    """
    Create a PyTorch DataLoader for thermal annotation data.
    
    Args:
        annotation_file: Path to annotation JSON file
        mac_address: Sensor MAC address
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        prefetch: Whether to prefetch all frames before training
        tdengine_config: Optional TDengine connection config
        **dataloader_kwargs: Additional arguments for DataLoader
        
    Returns:
        PyTorch DataLoader ready for training
        
    Example:
        >>> dataloader = create_dataloader(
        ...     'annotations.json',
        ...     '02:00:1a:62:51:67',
        ...     batch_size=16,
        ...     shuffle=True,
        ...     prefetch=True
        ... )
        >>> for frames, targets in dataloader:
        ...     # frames: (batch_size, 1, H, W)
        ...     # targets: list of dicts with boxes and labels
        ...     train_step(frames, targets)
    """
    # Create dataset
    dataset = ThermalAnnotationDataset(
        annotation_file=annotation_file,
        mac_address=mac_address,
        tdengine_config=tdengine_config,
        cache_frames=True  # Always enable cache for training
    )
    
    # Prefetch all frames if requested
    if prefetch:
        logger.info("Prefetching all frames into memory...")
        dataset.prefetch_all_frames()
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **dataloader_kwargs
    )
    
    logger.info(f"Created DataLoader:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Shuffle: {shuffle}")
    logger.info(f"  Num workers: {num_workers}")
    logger.info(f"  Total batches: {len(dataloader)}")
    
    return dataloader


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Custom collate function for batching samples.
    
    Args:
        batch: List of (frame_tensor, target_dict) tuples
        
    Returns:
        Tuple of (batched_frames, list_of_targets)
        - batched_frames: (batch_size, 1, H, W)
        - list_of_targets: List of target dicts (varies by sample)
    """
    frames = []
    targets = []
    
    for frame, target in batch:
        frames.append(frame)
        targets.append(target)
    
    # Stack frames into batch
    frames_batch = torch.stack(frames, dim=0)
    
    return frames_batch, targets

