"""
Thermal Action Detection Dataset Package

This package provides tools for creating and using thermal sensor datasets
for human action detection with SlowFast models.
"""

from .thermal_action_dataset import (
    ThermalActionDataset,
    ThermalActionTransform,
    collate_fn
)

from .create_hdf5_frames import ThermalFrameHDF5Creator
from .convert_annotations_to_coco import ThermalAnnotationConverter

__all__ = [
    'ThermalActionDataset',
    'ThermalActionTransform',
    'collate_fn',
    'ThermalFrameHDF5Creator',
    'ThermalAnnotationConverter'
]

__version__ = '1.0.0'

