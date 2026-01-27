"""
Configuration Management Module
Professional-grade configuration with validation and environment support
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import json


@dataclass
class CameraConfig:
    """Individual camera configuration"""
    name: str
    rtsp_url: str
    enabled: bool = True
    resolution: Tuple[int, int] = (1920, 1080)
    fps_target: int = 25
    buffer_size: int = 2


@dataclass
class ModelConfig:
    """YOLO model configuration"""
    model_path: str = "bestv4.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    vehicle_classes: List[str] = field(default_factory=lambda: [
        'auto_rickshaw', 'bike', 'bus', 'car', 'mini_bus', 'tractor', 'truck'
    ])
    use_half_precision: bool = False  # FP16 for faster inference
    device: str = "0"  # GPU device or "cpu"


@dataclass
class TrackingConfig:
    """Advanced tracking configuration"""
    tracker_type: str = "bytetrack.yaml"
    max_age: int = 30  # Max frames to keep lost tracks
    min_hits: int = 3  # Min detections before confirming track
    iou_threshold: float = 0.3
    use_kalman_filter: bool = True
    enable_speed_estimation: bool = True
    enable_direction_detection: bool = True
    enable_trajectory_smoothing: bool = True


@dataclass
class StorageConfig:
    """Storage and dataset configuration"""
    base_dir: Path = Path("images")
    frame_dir: str = "frame"
    crop_dir: str = "croped"
    save_format: str = "jpg"  # jpg or png
    jpeg_quality: int = 100
    png_compression: int = 0
    save_full_frame: bool = True
    save_cropped: bool = True
    crop_padding: int = 10
    min_crop_size: Tuple[int, int] = (50, 50)  # Min width, height


@dataclass
class DisplayConfig:
    """Display and visualization configuration"""
    window_name: str = "Vehicle Tracking System - Multi-Camera View"
    display_width: int = 1920
    display_height: int = 1080
    show_roi: bool = True
    show_tracking: bool = True
    show_fps: bool = True
    show_trajectories: bool = True
    roi_color: Tuple[int, int, int] = (0, 255, 0)
    roi_thickness: int = 2
    bbox_thickness: int = 2
    font_scale: float = 0.6
    info_panel_alpha: float = 0.7


@dataclass
class ROIConfig:
    """ROI configuration"""
    config_file: str = "roi_config.json"
    point_radius: int = 5
    allow_multiple_rois: bool = False  # Future: multiple ROIs per camera
    min_points: int = 3


@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    enable_multithreading: bool = True
    enable_async_save: bool = True
    max_queue_size: int = 100
    enable_gpu: bool = True
    tensorrt_optimization: bool = False
    batch_processing: bool = False
    frame_skip: int = 0  # Process every Nth frame (0 = all frames)


@dataclass
class AdvancedConfig:
    """Advanced features configuration"""
    enable_quality_check: bool = True
    min_blur_threshold: float = 100.0  # Laplacian variance
    min_brightness: int = 30
    max_brightness: int = 220
    enable_deduplication: bool = True
    spatial_dedup_threshold: float = 0.8  # IoU threshold
    temporal_dedup_frames: int = 10
    enable_vehicle_classification: bool = True
    enable_metrics_logging: bool = True
    metrics_interval: int = 60  # seconds


class SystemConfig:
    """Main system configuration manager"""
    
    def __init__(self, config_file: str = None):
        # Initialize all sub-configs
        self.cameras: Dict[str, CameraConfig] = self._init_cameras()
        self.model = ModelConfig()
        self.tracking = TrackingConfig()
        self.storage = StorageConfig()
        self.display = DisplayConfig()
        self.roi = ROIConfig()
        self.performance = PerformanceConfig()
        self.advanced = AdvancedConfig()
        
        # Load from file if provided
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
    
    def _init_cameras(self) -> Dict[str, CameraConfig]:
        """Initialize camera configurations"""
        return {
            "overview": CameraConfig(
                name="overview",
                rtsp_url="rtsp://admin:Arya@123@125.18.39.10:5554/cam/realmonitor?channel=1&subtype=0"
            ),
            "anpr": CameraConfig(
                name="anpr",
                rtsp_url="rtsp://admin:BE04_ViDeS@125.18.39.10:5555/Streaming/Channels/101"
            ),
            "ptz": CameraConfig(
                name="ptz",
                rtsp_url="rtsp://admin:Arya_123@125.18.39.10:554/cam/realmonitor?channel=1&subtype=0"
            )
        }
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            # TODO: Parse and update configs
            print(f"Loaded configuration from {config_file}")
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def save_to_file(self, config_file: str):
        """Save configuration to JSON file"""
        try:
            # TODO: Serialize all configs
            with open(config_file, 'w') as f:
                json.dump({}, f, indent=2)
            print(f"Saved configuration to {config_file}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def validate(self) -> bool:
        """Validate configuration"""
        # Check model exists
        if not Path(self.model.model_path).exists():
            print(f"Error: Model file not found: {self.model.model_path}")
            return False
        
        # Check camera URLs
        for cam_name, cam_config in self.cameras.items():
            if not cam_config.rtsp_url:
                print(f"Error: No RTSP URL for camera: {cam_name}")
                return False
        
        # Validate thresholds
        if not (0 <= self.model.confidence_threshold <= 1):
            print(f"Error: Invalid confidence threshold: {self.model.confidence_threshold}")
            return False
        
        return True
    
    def get_enabled_cameras(self) -> Dict[str, CameraConfig]:
        """Get only enabled cameras"""
        return {name: cfg for name, cfg in self.cameras.items() if cfg.enabled}


# Global configuration instance
config = SystemConfig()
