"""
Advanced Vehicle Tracker Module
High-performance tracking with Kalman filtering, trajectory analysis, and deduplication
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from utils import (
    ImageQualityChecker, GeometryUtils, DirectionDetector, 
    SpeedEstimator, FileManager, VisualizationUtils
)
from roi_manager import ROIManager

logger = logging.getLogger(__name__)


@dataclass
class VehicleTrack:
    """Individual vehicle track information"""
    track_id: int
    class_id: int
    class_name: str
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    trajectory: deque = field(default_factory=lambda: deque(maxlen=50))
    boxes: deque = field(default_factory=lambda: deque(maxlen=10))
    confidences: List[float] = field(default_factory=list)
    saved: bool = False
    frame_count: int = 0
    direction: Optional[str] = None
    speed: Optional[float] = None
    
    def update(self, center: Tuple[int, int], box: np.ndarray, conf: float):
        """Update track with new detection"""
        self.last_seen = datetime.now()
        self.trajectory.append(center)
        self.boxes.append(box)
        self.confidences.append(conf)
        self.frame_count += 1
    
    def get_avg_confidence(self) -> float:
        """Get average confidence"""
        return np.mean(self.confidences) if self.confidences else 0.0
    
    def get_lifetime(self) -> float:
        """Get track lifetime in seconds"""
        return (self.last_seen - self.first_seen).total_seconds()
    
    def is_stationary(self, threshold: float = 10.0) -> bool:
        """Check if vehicle is stationary"""
        if len(self.trajectory) < 5:
            return False
        
        # Calculate total movement
        total_dist = 0
        for i in range(1, len(self.trajectory)):
            dist = GeometryUtils.calculate_distance(
                self.trajectory[i-1], 
                self.trajectory[i]
            )
            total_dist += dist
        
        return total_dist < threshold


class AdvancedVehicleTracker:
    """
    Advanced vehicle tracking system
    
    Features:
    - ByteTrack integration
    - Kalman filtering for smooth tracking
    - Trajectory analysis
    - Speed estimation
    - Direction detection
    - Quality-based saving
    - Duplicate prevention
    - Stationary vehicle filtering
    """
    
    def __init__(self,
                 camera_name: str,
                 roi_manager: ROIManager,
                 save_dir_frame: Path,
                 save_dir_crop: Path,
                 vehicle_classes: List[str],
                 config: dict = None):
        """
        Initialize advanced tracker
        
        Args:
            camera_name: Camera identifier
            roi_manager: ROI manager instance
            save_dir_frame: Directory for full frames
            save_dir_crop: Directory for cropped images
            vehicle_classes: List of vehicle class names
            config: Configuration dict
        """
        self.camera_name = camera_name
        self.roi_manager = roi_manager
        self.save_dir_frame = save_dir_frame
        self.save_dir_crop = save_dir_crop
        self.vehicle_classes = vehicle_classes
        
        # Create directories
        self.save_dir_frame.mkdir(parents=True, exist_ok=True)
        self.save_dir_crop.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.config = config or {}
        self.min_blur_threshold = self.config.get('min_blur_threshold', 100.0)
        self.crop_padding = self.config.get('crop_padding', 10)
        self.save_format = self.config.get('save_format', 'jpg')
        self.save_quality = self.config.get('save_quality', 100)
        self.enable_quality_check = self.config.get('enable_quality_check', True)
        self.enable_speed_estimation = self.config.get('enable_speed_estimation', True)
        self.enable_direction_detection = self.config.get('enable_direction_detection', True)
        
        # Tracking state
        self.active_tracks: Dict[int, VehicleTrack] = {}
        self.saved_track_ids: set = set()
        
        # Counters
        self.total_vehicles = 0
        self.saved_frames = 0
        self.rejected_quality = 0
        self.rejected_stationary = 0
        
        # Speed estimator
        self.speed_estimator = SpeedEstimator(
            pixels_per_meter=self.config.get('pixels_per_meter', 10.0),
            fps=self.config.get('fps', 25.0)
        )
        
        # Direction detector
        self.direction_detector = DirectionDetector()
        
        # Quality checker
        self.quality_checker = ImageQualityChecker()
        
        # File manager
        self.file_manager = FileManager()
        
        logger.info(f"Initialized advanced tracker for {camera_name}")
        logger.info(f"Quality check: {self.enable_quality_check}")
        logger.info(f"Speed estimation: {self.enable_speed_estimation}")
        logger.info(f"Direction detection: {self.enable_direction_detection}")
    
    def process_detections(self, frame: np.ndarray, results):
        """
        Process YOLO tracking results
        
        Args:
            frame: Current frame
            results: YOLO tracking results
        """
        current_frame_tracks = set()
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Extract detection info
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                
                # Get tracking ID
                if not hasattr(boxes, 'id') or boxes.id is None:
                    continue
                
                track_id = int(boxes.id[i])
                
                # Calculate center
                center = GeometryUtils.get_box_center(box)
                
                # Check ROI
                if not self.roi_manager.is_point_in_roi(center):
                    continue
                
                # Track is valid and in ROI
                current_frame_tracks.add(track_id)
                
                # Update or create track
                if track_id not in self.active_tracks:
                    # New track
                    self._create_new_track(track_id, cls, center, box, conf)
                else:
                    # Update existing track
                    self._update_track(track_id, center, box, conf)
                
                # Check if we should save this track
                if track_id not in self.saved_track_ids:
                    self._try_save_vehicle(frame, track_id)
        
        # Clean up lost tracks
        self._cleanup_lost_tracks(current_frame_tracks)
    
    def _create_new_track(self, 
                         track_id: int, 
                         cls: int, 
                         center: Tuple[int, int], 
                         box: np.ndarray, 
                         conf: float):
        """Create new vehicle track"""
        track = VehicleTrack(
            track_id=track_id,
            class_id=cls,
            class_name=self.vehicle_classes[cls]
        )
        track.update(center, box, conf)
        
        self.active_tracks[track_id] = track
        self.total_vehicles += 1
        
        logger.debug(f"New track: ID={track_id}, Class={track.class_name}")
    
    def _update_track(self, 
                     track_id: int, 
                     center: Tuple[int, int], 
                     box: np.ndarray, 
                     conf: float):
        """Update existing track"""
        track = self.active_tracks[track_id]
        track.update(center, box, conf)
        
        # Update direction
        if self.enable_direction_detection and len(track.trajectory) >= 5:
            track.direction = self.direction_detector.get_direction(
                list(track.trajectory)
            )
        
        # Update speed
        if self.enable_speed_estimation and len(track.trajectory) >= 3:
            track.speed = self.speed_estimator.estimate_speed(
                list(track.trajectory)
            )
    
    def _try_save_vehicle(self, frame: np.ndarray, track_id: int):
        """
        Try to save vehicle if it meets criteria
        
        Args:
            frame: Current frame
            track_id: Track ID to save
        """
        track = self.active_tracks[track_id]
        
        # Check if track is mature enough
        if track.frame_count < 3:
            return
        
        # Check if vehicle is stationary (skip if yes)
        if track.is_stationary():
            logger.debug(f"Skipping stationary vehicle: ID={track_id}")
            self.rejected_stationary += 1
            self.saved_track_ids.add(track_id)  # Mark as processed
            return
        
        # Get latest box
        box = track.boxes[-1]
        
        # Extract crop
        expanded_box = GeometryUtils.expand_box(box, self.crop_padding, frame.shape)
        x1, y1, x2, y2 = map(int, expanded_box)
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            logger.warning(f"Empty crop for track {track_id}")
            return
        
        # Quality check
        if self.enable_quality_check:
            is_good, metrics = self.quality_checker.is_quality_acceptable(
                crop, 
                min_blur=self.min_blur_threshold
            )
            
            if not is_good:
                logger.debug(f"Quality check failed: ID={track_id}, metrics={metrics}")
                self.rejected_quality += 1
                return
        
        # Save images
        success = self._save_images(frame, crop, track)
        
        if success:
            self.saved_track_ids.add(track_id)
            self.saved_frames += 1
            logger.info(f"[{self.camera_name}] Saved vehicle #{self.saved_frames} "
                       f"(ID={track_id}, Class={track.class_name}, "
                       f"Conf={track.get_avg_confidence():.2f}, "
                       f"Speed={track.speed:.1f if track.speed else 0:.1f} km/h, "
                       f"Dir={track.direction or 'N/A'})")
    
    def _save_images(self, 
                    full_frame: np.ndarray, 
                    crop: np.ndarray, 
                    track: VehicleTrack) -> bool:
        """
        Save full frame and cropped image
        
        Args:
            full_frame: Full frame image
            crop: Cropped vehicle image
            track: Vehicle track info
            
        Returns:
            True if saved successfully
        """
        try:
            # Generate filenames with timestamp and camera for uniqueness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            frame_filename = f"{self.camera_name}_vehicle_{self.saved_frames}_{timestamp}.{self.save_format}"
            crop_filename = f"{self.camera_name}_vehicle_{self.saved_frames}_{timestamp}.{self.save_format}"
            
            frame_path = self.save_dir_frame / frame_filename
            crop_path = self.save_dir_crop / crop_filename
            
            # Save full frame
            frame_success = self.file_manager.save_image_high_quality(
                full_frame, frame_path, self.save_format, self.save_quality
            )
            
            # Save crop
            crop_success = self.file_manager.save_image_high_quality(
                crop, crop_path, self.save_format, self.save_quality
            )
            
            return frame_success and crop_success
            
        except Exception as e:
            logger.error(f"Error saving images: {e}")
            return False
    
    def _cleanup_lost_tracks(self, current_tracks: set):
        """Remove tracks that haven't been seen recently"""
        max_age = 30  # frames
        
        to_remove = []
        for track_id, track in self.active_tracks.items():
            if track_id not in current_tracks:
                # Track not seen in current frame
                age = track.frame_count
                if age > max_age:
                    to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.active_tracks[track_id]
            logger.debug(f"Removed lost track: ID={track_id}")
    
    def draw_tracking_overlay(self, frame: np.ndarray):
        """
        Draw tracking visualization on frame
        
        Args:
            frame: Frame to draw on
        """
        for track_id, track in self.active_tracks.items():
            if not track.boxes:
                continue
            
            # Get latest box and info
            box = track.boxes[-1]
            center = track.trajectory[-1]
            
            x1, y1, x2, y2 = map(int, box)
            
            # Choose color
            if track_id in self.saved_track_ids:
                color = (0, 255, 0)  # Green = saved
            else:
                color = (0, 255, 255)  # Yellow = tracking
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(frame, center, 5, color, -1)
            
            # Draw trajectory
            if len(track.trajectory) > 1:
                VisualizationUtils.draw_trajectory(
                    frame, list(track.trajectory), color, 2
                )
            
            # Build label
            label_parts = [
                f"ID:{track_id}",
                track.class_name,
                f"{track.get_avg_confidence():.2f}"
            ]
            
            if track.speed is not None:
                label_parts.append(f"{track.speed:.0f}km/h")
            
            if track.direction:
                label_parts.append(track.direction)
            
            label = " ".join(label_parts)
            
            # Draw label with background
            VisualizationUtils.draw_text_with_background(
                frame, label, (x1, y1 - 10), 
                font_scale=0.5, thickness=1,
                text_color=(255, 255, 255), bg_color=color
            )
    
    def get_statistics(self) -> dict:
        """Get tracker statistics"""
        return {
            'camera': self.camera_name,
            'total_vehicles': self.total_vehicles,
            'saved_frames': self.saved_frames,
            'active_tracks': len(self.active_tracks),
            'rejected_quality': self.rejected_quality,
            'rejected_stationary': self.rejected_stationary,
            'save_rate': (self.saved_frames / self.total_vehicles * 100) 
                        if self.total_vehicles > 0 else 0
        }
