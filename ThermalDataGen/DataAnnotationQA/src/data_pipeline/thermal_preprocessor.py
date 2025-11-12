"""
Thermal Frame Preprocessor

Unified preprocessing class for thermal frames that handles:
- Kelvin vs Celsius format detection and conversion
- Temperature validation
- Windowing and normalization for image conversion
"""

import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ThermalFramePreprocessor:
    """
    Preprocessor for thermal frames with automatic format detection and conversion.
    
    This class provides a unified interface for:
    1. Detecting whether data is in Kelvin or Celsius
    2. Converting Kelvin to Celsius if needed
    3. Validating temperature ranges
    4. Converting thermal frames to 8-bit images with windowing
    
    Usage:
        preprocessor = ThermalFramePreprocessor(
            window_min=5.0,
            window_max=45.0,
            auto_convert=True
        )
        
        # Process frame (auto-detects format and converts if needed)
        frame_celsius = preprocessor.process_frame(frame)
        
        # Convert to image
        img_8bit = preprocessor.thermal_to_image(frame_celsius)
    """
    
    def __init__(
        self,
        window_min: float = 5.0,
        window_max: float = 45.0,
        expected_min: float = 5.0,
        expected_max: float = 45.0,
        absolute_min: float = -40.0,
        absolute_max: float = 80.0,
        auto_convert: bool = True,
        kelvin_threshold: float = 200.0
    ):
        """
        Initialize thermal frame preprocessor.
        
        Args:
            window_min: Minimum temperature for windowing (°C)
            window_max: Maximum temperature for windowing (°C)
            expected_min: Expected minimum temperature for indoor scenes (°C)
            expected_max: Expected maximum temperature for indoor scenes (°C)
            absolute_min: Absolute minimum valid temperature (°C)
            absolute_max: Absolute maximum valid temperature (°C)
            auto_convert: Automatically convert Kelvin to Celsius
            kelvin_threshold: Mean temperature above which data is considered Kelvin (°C)
        """
        self.window_min = window_min
        self.window_max = window_max
        self.expected_min = expected_min
        self.expected_max = expected_max
        self.absolute_min = absolute_min
        self.absolute_max = absolute_max
        self.auto_convert = auto_convert
        self.kelvin_threshold = kelvin_threshold
        
        logger.debug(
            f"ThermalFramePreprocessor initialized: "
            f"window=[{window_min},{window_max}]°C, "
            f"expected=[{expected_min},{expected_max}]°C, "
            f"auto_convert={auto_convert}"
        )
    
    def check_temperature_format(self, frame: np.ndarray) -> str:
        """
        Detect whether frame is in Kelvin or Celsius.
        
        Args:
            frame: Thermal frame array
            
        Returns:
            'kelvin' or 'celsius'
        """
        frame_mean = frame.mean()
        frame_min = frame.min()
        frame_max = frame.max()
        
        # Check if mean is in typical Kelvin range (> 200)
        if frame_mean > self.kelvin_threshold:
            logger.info(
                f"Temperature format detected: KELVIN "
                f"(mean={frame_mean:.1f}, range=[{frame_min:.1f}, {frame_max:.1f}])"
            )
            return 'kelvin'
        
        # Check if values are in typical Celsius range
        if self.absolute_min < frame_mean < self.absolute_max:
            logger.debug(
                f"Temperature format detected: CELSIUS "
                f"(mean={frame_mean:.1f}°C, range=[{frame_min:.1f}, {frame_max:.1f}]°C)"
            )
            return 'celsius'
        
        # Ambiguous - default to celsius with warning
        logger.warning(
            f"Ambiguous temperature format (mean={frame_mean:.1f}). "
            f"Assuming CELSIUS. Range: [{frame_min:.1f}, {frame_max:.1f}]"
        )
        return 'celsius'
    
    def kelvin_to_celsius(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert Kelvin to Celsius.
        
        Args:
            frame: Frame in Kelvin
            
        Returns:
            Frame in Celsius
        """
        frame_celsius = frame - 273.15
        logger.info(
            f"Converted Kelvin to Celsius: "
            f"{frame.mean():.1f}K -> {frame_celsius.mean():.1f}°C"
        )
        return frame_celsius
    
    def validate_temperature_range(
        self, 
        frame: np.ndarray,
        raise_on_error: bool = True
    ) -> bool:
        """
        Validate that frame temperatures are in reasonable Celsius range.
        
        Args:
            frame: Thermal frame in Celsius
            raise_on_error: Whether to raise exception on validation failure
            
        Returns:
            True if validation passed, False otherwise
            
        Raises:
            ValueError: If validation fails and raise_on_error=True
        """
        frame_min = frame.min()
        frame_max = frame.max()
        frame_mean = frame.mean()
        
        # Check for absolute zero or extreme values (strong indicators of issues)
        if frame_min < -273.0:
            msg = (
                f"Invalid temperature: {frame_min:.2f}°C. "
                f"Values below absolute zero detected!"
            )
            if raise_on_error:
                raise ValueError(msg)
            logger.error(msg)
            return False
        
        # Check if data might still be in Kelvin
        if frame_mean > self.kelvin_threshold:
            msg = (
                f"Invalid temperature: mean={frame_mean:.2f}°C. "
                f"Data appears to be in Kelvin! Expected Celsius range: "
                f"{self.expected_min}-{self.expected_max}°C"
            )
            if raise_on_error:
                raise ValueError(msg)
            logger.error(msg)
            return False
        
        # Check absolute limits (soft warnings)
        if frame_min < self.absolute_min:
            logger.warning(
                f"Temperature below absolute minimum: {frame_min:.2f}°C "
                f"(limit: {self.absolute_min}°C)"
            )
        
        if frame_max > self.absolute_max:
            logger.warning(
                f"Temperature above absolute maximum: {frame_max:.2f}°C "
                f"(limit: {self.absolute_max}°C)"
            )
        
        # Log if outside expected range (info only, not error)
        if frame_min < self.expected_min or frame_max > self.expected_max:
            logger.info(
                f"Temperature outside expected range: "
                f"{frame_min:.2f}°C to {frame_max:.2f}°C "
                f"(expected: {self.expected_min}-{self.expected_max}°C for indoor scenes)"
            )
        
        # Check for absolute zero pixels (corruption indicator)
        if np.any(frame <= -273.0):
            num_corrupted = np.sum(frame <= -273.0)
            msg = f"Frame contains {num_corrupted} absolute zero pixels (corrupted data!)"
            if raise_on_error:
                raise ValueError(msg)
            logger.error(msg)
            return False
        
        logger.debug(
            f"Temperature validation passed: {frame_min:.2f}°C to {frame_max:.2f}°C "
            f"(mean: {frame_mean:.2f}°C)"
        )
        return True
    
    def process_frame(
        self, 
        frame: np.ndarray,
        validate: bool = True
    ) -> np.ndarray:
        """
        Process thermal frame: detect format, convert if needed, and validate.
        
        This is the main entry point for frame preprocessing.
        
        Args:
            frame: Input thermal frame (Kelvin or Celsius)
            validate: Whether to validate temperature range
            
        Returns:
            Frame in Celsius, validated and ready for use
            
        Raises:
            ValueError: If validation fails
        """
        # Detect format
        format_type = self.check_temperature_format(frame)
        
        # Convert if needed
        if format_type == 'kelvin' and self.auto_convert:
            frame = self.kelvin_to_celsius(frame)
        elif format_type == 'kelvin' and not self.auto_convert:
            raise ValueError(
                f"Frame is in Kelvin but auto_convert=False. "
                f"Mean temperature: {frame.mean():.1f}K"
            )
        
        # Validate
        if validate:
            self.validate_temperature_range(frame, raise_on_error=True)
        
        return frame
    
    def thermal_to_image(
        self, 
        frame: np.ndarray,
        window_min: Optional[float] = None,
        window_max: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert thermal frame (Celsius) to 8-bit grayscale image using windowing.
        
        Windowing improves contrast by mapping a specific temperature range
        to the full 0-255 grayscale range. Default is 5-45°C for indoor scenes.
        
        Args:
            frame: Thermal frame in Celsius (H, W)
            window_min: Override minimum temperature for window (°C)
            window_max: Override maximum temperature for window (°C)
            
        Returns:
            8-bit grayscale image (H, W)
        """
        # Use provided window or default
        win_min = window_min if window_min is not None else self.window_min
        win_max = window_max if window_max is not None else self.window_max
        
        # Apply windowing: clip to range and normalize
        # Temperatures below win_min -> 0 (black)
        # Temperatures above win_max -> 255 (white)
        # Temperatures in between -> linear mapping
        
        # Clip to window range
        windowed = np.clip(frame, win_min, win_max)
        
        # Normalize to 0-1 range
        normalized = (windowed - win_min) / (win_max - win_min)
        
        # Convert to 8-bit (0-255)
        img_8bit = (normalized * 255).astype(np.uint8)
        
        logger.debug(
            f"Windowing: {win_min}°C to {win_max}°C | "
            f"Frame range: {frame.min():.1f}°C to {frame.max():.1f}°C"
        )
        
        return img_8bit
    
    def get_statistics(self, frame: np.ndarray) -> dict:
        """
        Get temperature statistics for a frame.
        
        Args:
            frame: Thermal frame
            
        Returns:
            Dictionary with statistics
        """
        return {
            'min': float(frame.min()),
            'max': float(frame.max()),
            'mean': float(frame.mean()),
            'std': float(frame.std()),
            'median': float(np.median(frame)),
            'format': self.check_temperature_format(frame)
        }
    
    def __repr__(self) -> str:
        return (
            f"ThermalFramePreprocessor("
            f"window=[{self.window_min},{self.window_max}]°C, "
            f"expected=[{self.expected_min},{self.expected_max}]°C, "
            f"auto_convert={self.auto_convert})"
        )

