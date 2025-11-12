"""
Thermal Data Training Pipeline

PyTorch custom Dataset and DataLoader for thermal sensor data with annotations.
Fetches data directly from TDengine into memory for training.
"""

from .thermal_dataset import ThermalAnnotationDataset, create_dataloader, TDengineConnector
from .thermal_preprocessor import ThermalFramePreprocessor

__version__ = "1.0.0"
__all__ = [
    "ThermalAnnotationDataset",
    "TDengineConnector",
    "ThermalFramePreprocessor",
    "create_dataloader",
]

