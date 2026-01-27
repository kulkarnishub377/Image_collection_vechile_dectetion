"""
Utility Functions Module
Advanced image processing, quality checks, and helper functions
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging
from pathlib import Path
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class ImageQualityChecker:
    """Advanced image quality assessment"""
    
    @staticmethod
    def calculate_blur(image: np.ndarray) -> float:
        """Calculate image blur using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        """Calculate average brightness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return np.mean(gray)
    
    @staticmethod
    def is_quality_acceptable(image: np.ndarray, 
                            min_blur: float = 100.0,
                            min_brightness: int = 30,
                            max_brightness: int = 220) -> Tuple[bool, dict]:
        """
        Check if image meets quality standards
        
        Returns:
            (is_acceptable, metrics_dict)
        """
        blur_score = ImageQualityChecker.calculate_blur(image)
        brightness = ImageQualityChecker.calculate_brightness(image)
        
        metrics = {
            'blur_score': blur_score,
            'brightness': brightness,
            'is_sharp': blur_score >= min_blur,
            'is_well_lit': min_brightness <= brightness <= max_brightness
        }
        
        is_acceptable = metrics['is_sharp'] and metrics['is_well_lit']
        
        return is_acceptable, metrics
    
    @staticmethod
    def enhance_image(image: np.ndarray, 
                     auto_contrast: bool = True,
                     denoise: bool = False) -> np.ndarray:
        """Apply image enhancement"""
        enhanced = image.copy()
        
        if auto_contrast:
            # CLAHE for adaptive histogram equalization
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        if denoise:
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return enhanced


class GeometryUtils:
    """Geometric calculations for tracking"""
    
    @staticmethod
    def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection area
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Union area
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def calculate_angle(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate angle in degrees"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return np.degrees(np.arctan2(dy, dx))
    
    @staticmethod
    def get_box_center(box: np.ndarray) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    @staticmethod
    def expand_box(box: np.ndarray, 
                   padding: int, 
                   img_shape: Tuple[int, int]) -> np.ndarray:
        """Expand bounding box with padding"""
        h, w = img_shape[:2]
        x1, y1, x2, y2 = box
        
        x1 = max(0, int(x1 - padding))
        y1 = max(0, int(y1 - padding))
        x2 = min(w, int(x2 + padding))
        y2 = min(h, int(y2 + padding))
        
        return np.array([x1, y1, x2, y2])


class DirectionDetector:
    """Detect vehicle movement direction"""
    
    DIRECTIONS = {
        0: "East →",
        45: "Northeast ↗",
        90: "North ↑",
        135: "Northwest ↖",
        180: "West ←",
        225: "Southwest ↙",
        270: "South ↓",
        315: "Southeast ↘"
    }
    
    @staticmethod
    def get_direction(trajectory: List[Tuple[int, int]], 
                     min_points: int = 5) -> Optional[str]:
        """
        Determine direction from trajectory
        
        Args:
            trajectory: List of (x, y) center points
            min_points: Minimum points needed
            
        Returns:
            Direction string or None
        """
        if len(trajectory) < min_points:
            return None
        
        # Use first and last points
        start = trajectory[0]
        end = trajectory[-1]
        
        # Calculate angle
        angle = GeometryUtils.calculate_angle(start, end)
        
        # Normalize to 0-360
        angle = (angle + 360) % 360
        
        # Find closest direction
        closest_dir = min(DirectionDetector.DIRECTIONS.keys(), 
                         key=lambda x: abs(x - angle))
        
        return DirectionDetector.DIRECTIONS[closest_dir]


class SpeedEstimator:
    """Estimate vehicle speed from trajectory"""
    
    def __init__(self, pixels_per_meter: float = 10.0, fps: float = 25.0):
        """
        Args:
            pixels_per_meter: Calibration factor (needs manual calibration)
            fps: Frames per second
        """
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
    
    def estimate_speed(self, 
                      trajectory: List[Tuple[int, int]], 
                      time_window: int = 10) -> Optional[float]:
        """
        Estimate speed in km/h
        
        Args:
            trajectory: List of (x, y) center points
            time_window: Number of frames to consider
            
        Returns:
            Speed in km/h or None
        """
        if len(trajectory) < 2:
            return None
        
        # Use last N points
        recent_trajectory = trajectory[-time_window:]
        
        if len(recent_trajectory) < 2:
            return None
        
        # Calculate total distance
        total_distance = 0
        for i in range(1, len(recent_trajectory)):
            distance = GeometryUtils.calculate_distance(
                recent_trajectory[i-1], 
                recent_trajectory[i]
            )
            total_distance += distance
        
        # Convert to meters
        distance_meters = total_distance / self.pixels_per_meter
        
        # Calculate time
        time_seconds = (len(recent_trajectory) - 1) / self.fps
        
        if time_seconds == 0:
            return None
        
        # Calculate speed (m/s to km/h)
        speed_mps = distance_meters / time_seconds
        speed_kmh = speed_mps * 3.6
        
        return round(speed_kmh, 2)


class FileManager:
    """Advanced file management utilities"""
    
    @staticmethod
    def save_image_high_quality(image: np.ndarray, 
                               filepath: Path, 
                               format: str = "jpg",
                               quality: int = 100) -> bool:
        """
        Save image with maximum quality
        
        Args:
            image: Image array
            filepath: Output path
            format: jpg or png
            quality: Quality setting
            
        Returns:
            Success status
        """
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "jpg":
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            else:
                params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
            
            success = cv2.imwrite(str(filepath), image, params)
            
            if success:
                logger.debug(f"Saved image: {filepath}")
            else:
                logger.error(f"Failed to save image: {filepath}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False
    
    @staticmethod
    def generate_unique_filename(base_name: str, 
                                extension: str, 
                                counter: int) -> str:
        """Generate unique filename"""
        return f"{base_name}_{counter}.{extension}"
    
    @staticmethod
    def calculate_image_hash(image: np.ndarray) -> str:
        """Calculate perceptual hash for image deduplication"""
        # Resize to standard size
        resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Calculate hash
        avg = gray.mean()
        hash_array = (gray > avg).astype(int)
        hash_str = ''.join(hash_array.flatten().astype(str))
        return hash_str


class VisualizationUtils:
    """Advanced visualization utilities"""
    
    @staticmethod
    def draw_text_with_background(image: np.ndarray,
                                  text: str,
                                  position: Tuple[int, int],
                                  font_scale: float = 0.6,
                                  thickness: int = 2,
                                  text_color: Tuple[int, int, int] = (255, 255, 255),
                                  bg_color: Tuple[int, int, int] = (0, 0, 0),
                                  padding: int = 5):
        """Draw text with background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        x, y = position
        
        # Draw background
        cv2.rectangle(image, 
                     (x - padding, y - h - padding), 
                     (x + w + padding, y + padding), 
                     bg_color, -1)
        
        # Draw text
        cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)
    
    @staticmethod
    def draw_trajectory(image: np.ndarray,
                       trajectory: List[Tuple[int, int]],
                       color: Tuple[int, int, int] = (0, 255, 255),
                       thickness: int = 2):
        """Draw vehicle trajectory"""
        if len(trajectory) < 2:
            return
        
        for i in range(1, len(trajectory)):
            cv2.line(image, trajectory[i-1], trajectory[i], color, thickness)
    
    @staticmethod
    def create_info_panel(width: int = 400,
                         height: int = 200,
                         bg_color: Tuple[int, int, int] = (0, 0, 0),
                         alpha: float = 0.7) -> np.ndarray:
        """Create semi-transparent info panel"""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = bg_color
        return panel


class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.metrics = {
            'fps': [],
            'processing_time': [],
            'detection_count': [],
            'tracking_count': []
        }
        self.start_time = datetime.now()
    
    def update(self, metric_name: str, value: float):
        """Update metric"""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
    
    def get_average(self, metric_name: str, window: int = 100) -> float:
        """Get average of recent values"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return 0.0
        
        recent = self.metrics[metric_name][-window:]
        return np.mean(recent)
    
    def get_summary(self) -> dict:
        """Get performance summary"""
        summary = {}
        for metric_name in self.metrics:
            if self.metrics[metric_name]:
                summary[f"{metric_name}_avg"] = np.mean(self.metrics[metric_name])
                summary[f"{metric_name}_max"] = np.max(self.metrics[metric_name])
                summary[f"{metric_name}_min"] = np.min(self.metrics[metric_name])
        
        summary['uptime'] = (datetime.now() - self.start_time).total_seconds()
        return summary
