"""
ROI Manager Module
Advanced ROI handling with polygon support, persistence, and validation
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ROIManager:
    """
    Advanced ROI management with polygon support
    Features:
    - Interactive polygon drawing
    - ROI persistence (save/load)
    - Point-in-polygon testing
    - Multiple ROI support (future)
    - ROI validation
    """
    
    def __init__(self, camera_name: str, roi_config_path: str):
        self.camera_name = camera_name
        self.roi_config_path = roi_config_path
        self.roi_points: List[Tuple[int, int]] = []
        self.is_drawing = False
        self.temp_point: Optional[Tuple[int, int]] = None
        
        # Visual properties
        self.roi_color = (0, 255, 0)
        self.point_radius = 5
        self.line_thickness = 2
        
        logger.info(f"Initialized ROI manager for {camera_name}")
    
    def load_roi(self) -> bool:
        """
        Load ROI from configuration file
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            roi_path = Path(self.roi_config_path)
            if not roi_path.exists():
                logger.warning(f"ROI config file not found: {self.roi_config_path}")
                return False
            
            with open(roi_path, 'r') as f:
                data = json.load(f)
            
            if self.camera_name not in data:
                logger.warning(f"No ROI found for camera: {self.camera_name}")
                return False
            
            # Load points
            points = data[self.camera_name]
            self.roi_points = [tuple(p) for p in points]
            
            # Validate
            if len(self.roi_points) < 3:
                logger.warning(f"Invalid ROI (need at least 3 points): {self.camera_name}")
                self.roi_points = []
                return False
            
            logger.info(f"Loaded ROI for {self.camera_name}: {len(self.roi_points)} points")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ROI: {e}")
            return False
    
    def save_roi(self) -> bool:
        """
        Save ROI to configuration file
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Load existing data
            data = {}
            roi_path = Path(self.roi_config_path)
            
            if roi_path.exists():
                with open(roi_path, 'r') as f:
                    data = json.load(f)
            
            # Update with current ROI
            data[self.camera_name] = self.roi_points
            
            # Save
            with open(roi_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved ROI for {self.camera_name}: {len(self.roi_points)} points")
            return True
            
        except Exception as e:
            logger.error(f"Error saving ROI: {e}")
            return False
    
    def draw_roi_interactive(self, frame: np.ndarray) -> bool:
        """
        Interactive ROI drawing interface
        
        Args:
            frame: Frame to draw on
            
        Returns:
            True if ROI was successfully drawn
        """
        clone = frame.copy()
        window_name = f"Draw ROI: {self.camera_name} | Click: add point | 'c': complete | 'r': reset | 'q': quit"
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal clone
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # Add point
                self.roi_points.append((x, y))
                logger.debug(f"Added ROI point: ({x}, {y})")
                
                # Redraw
                clone = frame.copy()
                self._draw_roi_preview(clone)
                cv2.imshow(window_name, clone)
            
            elif event == cv2.EVENT_MOUSEMOVE:
                # Update temporary point
                self.temp_point = (x, y)
                
                # Redraw
                clone = frame.copy()
                self._draw_roi_preview(clone)
                cv2.imshow(window_name, clone)
        
        # Setup window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        cv2.imshow(window_name, clone)
        
        logger.info(f"Draw ROI for {self.camera_name}")
        logger.info("Controls: Click to add points | 'c' to complete | 'r' to reset | 'q' to quit")
        
        # Drawing loop
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # Complete ROI
                if len(self.roi_points) >= 3:
                    cv2.destroyWindow(window_name)
                    self.save_roi()
                    logger.info(f"ROI complete: {len(self.roi_points)} points")
                    return True
                else:
                    logger.warning("Need at least 3 points to complete ROI")
            
            elif key == ord('r'):
                # Reset ROI
                self.roi_points = []
                self.temp_point = None
                clone = frame.copy()
                cv2.imshow(window_name, clone)
                logger.info("ROI reset")
            
            elif key == ord('q'):
                # Quit without saving
                cv2.destroyWindow(window_name)
                logger.info("ROI drawing cancelled")
                return False
        
        return False
    
    def _draw_roi_preview(self, frame: np.ndarray):
        """Draw ROI preview during interactive drawing"""
        # Draw existing points
        for i, point in enumerate(self.roi_points):
            cv2.circle(frame, point, self.point_radius, self.roi_color, -1)
            cv2.putText(frame, str(i+1), 
                       (point[0] + 10, point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.roi_color, 2)
        
        # Draw lines between points
        if len(self.roi_points) > 1:
            for i in range(len(self.roi_points)):
                pt1 = self.roi_points[i]
                pt2 = self.roi_points[(i + 1) % len(self.roi_points)]
                cv2.line(frame, pt1, pt2, self.roi_color, self.line_thickness)
        
        # Draw temporary line from last point to cursor
        if self.temp_point and len(self.roi_points) > 0:
            cv2.line(frame, self.roi_points[-1], self.temp_point, 
                    (255, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, self.temp_point, self.point_radius, 
                      (255, 255, 0), 1)
        
        # Draw instruction text
        cv2.putText(frame, f"Points: {len(self.roi_points)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def is_point_in_roi(self, point: Tuple[int, int]) -> bool:
        """
        Check if point is inside ROI polygon
        
        Args:
            point: (x, y) coordinates
            
        Returns:
            True if point is inside ROI
        """
        # No ROI means all points are valid
        if len(self.roi_points) < 3:
            return True
        
        # Convert to numpy array
        roi_array = np.array(self.roi_points, dtype=np.int32)
        
        # Use OpenCV's point polygon test
        result = cv2.pointPolygonTest(roi_array, point, False)
        
        return result >= 0
    
    def draw_roi_overlay(self, frame: np.ndarray, alpha: float = 0.2):
        """
        Draw ROI overlay on frame with semi-transparent fill
        
        Args:
            frame: Frame to draw on
            alpha: Transparency (0.0 to 1.0)
        """
        if len(self.roi_points) < 3:
            return
        
        # Create overlay
        overlay = frame.copy()
        roi_array = np.array(self.roi_points, dtype=np.int32)
        
        # Fill polygon
        cv2.fillPoly(overlay, [roi_array], self.roi_color)
        
        # Blend with original
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw outline
        cv2.polylines(frame, [roi_array], True, self.roi_color, 
                     self.line_thickness, cv2.LINE_AA)
    
    def get_roi_area(self) -> float:
        """
        Calculate ROI area in pixels
        
        Returns:
            Area in square pixels
        """
        if len(self.roi_points) < 3:
            return 0.0
        
        roi_array = np.array(self.roi_points, dtype=np.int32)
        return cv2.contourArea(roi_array)
    
    def get_roi_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box of ROI
        
        Returns:
            (x_min, y_min, x_max, y_max) or None
        """
        if len(self.roi_points) < 3:
            return None
        
        roi_array = np.array(self.roi_points)
        x_min = roi_array[:, 0].min()
        y_min = roi_array[:, 1].min()
        x_max = roi_array[:, 0].max()
        y_max = roi_array[:, 1].max()
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def validate_roi(self, frame_shape: Tuple[int, int]) -> bool:
        """
        Validate ROI against frame dimensions
        
        Args:
            frame_shape: (height, width) of frame
            
        Returns:
            True if ROI is valid
        """
        if len(self.roi_points) < 3:
            logger.warning("ROI has less than 3 points")
            return False
        
        h, w = frame_shape[:2]
        
        # Check all points are within frame
        for point in self.roi_points:
            if not (0 <= point[0] < w and 0 <= point[1] < h):
                logger.warning(f"ROI point outside frame: {point}")
                return False
        
        # Check area is reasonable
        area = self.get_roi_area()
        frame_area = h * w
        
        if area < 100:  # Too small
            logger.warning(f"ROI area too small: {area}")
            return False
        
        if area > frame_area * 0.95:  # Too large
            logger.warning(f"ROI area covers almost entire frame: {area}")
        
        return True
    
    def clear_roi(self):
        """Clear current ROI"""
        self.roi_points = []
        self.temp_point = None
        logger.info(f"Cleared ROI for {self.camera_name}")
    
    def get_roi_mask(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create binary mask for ROI
        
        Args:
            frame_shape: (height, width) of frame
            
        Returns:
            Binary mask (255 inside ROI, 0 outside)
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if len(self.roi_points) >= 3:
            roi_array = np.array(self.roi_points, dtype=np.int32)
            cv2.fillPoly(mask, [roi_array], 255)
        
        return mask
