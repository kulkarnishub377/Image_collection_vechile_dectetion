"""
Main System Orchestrator
Professional vehicle capture system with modular architecture
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Optional
import signal
import sys

from config import SystemConfig, config
from roi_manager import ROIManager
from stream_reader import RTSPStreamReader, MultiStreamManager
from tracker import AdvancedVehicleTracker
from utils import VisualizationUtils, PerformanceMonitor
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vehicle_capture.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VehicleCaptureSystem:
    """
    Professional Vehicle Capture System
    
    Architecture:
    - Modular design with separation of concerns
    - Multi-threaded RTSP capture
    - Advanced tracking with ByteTrack + Kalman filtering
    - ROI-based filtering
    - Quality-controlled image saving
    - Real-time visualization and monitoring
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize system
        
        Args:
            config: System configuration
        """
        self.config = config
        self.running = False
        
        logger.info("=" * 80)
        logger.info("PROFESSIONAL VEHICLE CAPTURE SYSTEM")
        logger.info("=" * 80)
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid configuration")
        
        # Initialize YOLO model
        logger.info(f"Loading YOLO model: {self.config.model.model_path}")
        self.model = YOLO(self.config.model.model_path)
        
        # Verify model classes
        model_classes = self.model.names
        logger.info(f"Model classes: {model_classes}")
        
        # Set device
        if self.config.performance.enable_gpu:
            logger.info(f"Using device: {self.config.model.device}")
        
        # Initialize components
        self.stream_manager = MultiStreamManager()
        self.roi_managers: Dict[str, ROIManager] = {}
        self.trackers: Dict[str, AdvancedVehicleTracker] = {}
        self.performance_monitor = PerformanceMonitor()
        
        # Setup cameras
        self._setup_cameras()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("System initialized successfully")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _setup_cameras(self):
        """Setup all enabled cameras"""
        enabled_cameras = self.config.get_enabled_cameras()
        
        logger.info(f"Setting up {len(enabled_cameras)} cameras...")
        
        for cam_name, cam_config in enabled_cameras.items():
            logger.info(f"Setting up camera: {cam_name}")
            
            # Create camera-specific directories
            frame_dir = self.config.storage.base_dir / self.config.storage.frame_dir / cam_name
            crop_dir = self.config.storage.base_dir / self.config.storage.crop_dir / cam_name
            
            # Initialize ROI manager
            self.roi_managers[cam_name] = ROIManager(
                cam_name, 
                self.config.roi.config_file
            )
            
            # Initialize tracker
            tracker_config = {
                'min_blur_threshold': self.config.advanced.min_blur_threshold,
                'crop_padding': self.config.storage.crop_padding,
                'save_format': self.config.storage.save_format,
                'save_quality': self.config.storage.jpeg_quality,
                'enable_quality_check': self.config.advanced.enable_quality_check,
                'enable_speed_estimation': self.config.tracking.enable_speed_estimation,
                'enable_direction_detection': self.config.tracking.enable_direction_detection,
            }
            
            self.trackers[cam_name] = AdvancedVehicleTracker(
                cam_name,
                self.roi_managers[cam_name],
                frame_dir,
                crop_dir,
                self.config.model.vehicle_classes,
                tracker_config
            )
            
            logger.info(f"Camera setup complete: {cam_name}")
    
    def setup_rois(self):
        """Setup ROI for all cameras"""
        logger.info("Starting ROI setup...")
        
        # Start streams temporarily
        for cam_name, cam_config in self.config.get_enabled_cameras().items():
            self.stream_manager.add_stream(
                cam_name,
                cam_config.rtsp_url,
                target_resolution=cam_config.resolution,
                buffer_size=cam_config.buffer_size
            )
        
        # Wait for streams to connect
        logger.info("Waiting for streams to connect...")
        time.sleep(3)
        
        # Setup ROI for each camera
        for cam_name, roi_manager in self.roi_managers.items():
            # Try to load existing ROI
            if roi_manager.load_roi():
                logger.info(f"Using existing ROI for {cam_name}")
                
                # Validate ROI
                stream = self.stream_manager.get_stream(cam_name)
                frame = stream.wait_for_frame(timeout=10)
                
                if frame is not None and roi_manager.validate_roi(frame.shape):
                    logger.info(f"ROI validated for {cam_name}")
                    continue
                else:
                    logger.warning(f"ROI validation failed for {cam_name}, redrawing...")
            
            # Draw new ROI
            stream = self.stream_manager.get_stream(cam_name)
            frame = stream.wait_for_frame(timeout=10)
            
            if frame is not None:
                logger.info(f"Draw ROI for camera: {cam_name}")
                roi_manager.draw_roi_interactive(frame)
                
                # Validate
                if roi_manager.validate_roi(frame.shape):
                    logger.info(f"ROI setup complete for {cam_name}")
                else:
                    logger.error(f"Invalid ROI for {cam_name}")
            else:
                logger.error(f"Could not get frame from {cam_name}")
        
        # Stop temporary streams
        self.stream_manager.stop_all()
        
        logger.info("ROI setup complete")
    
    def run(self):
        """Main processing loop"""
        logger.info("Starting main processing loop...")
        
        # Start all streams
        for cam_name, cam_config in self.config.get_enabled_cameras().items():
            self.stream_manager.add_stream(
                cam_name,
                cam_config.rtsp_url,
                target_resolution=cam_config.resolution,
                buffer_size=cam_config.buffer_size
            )
        
        # Wait for streams
        time.sleep(2)
        
        self.running = True
        
        # Create display window
        cv2.namedWindow(self.config.display.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.config.display.window_name,
            self.config.display.display_width,
            self.config.display.display_height
        )
        
        logger.info("System running. Press 'q' to quit, 's' for statistics")
        
        frame_count = 0
        last_stats_time = time.time()
        
        try:
            while self.running:
                loop_start = time.time()
                
                frames_to_display = []
                
                # Process each camera
                for cam_name in self.config.get_enabled_cameras().keys():
                    frame = self._process_camera(cam_name)
                    if frame is not None:
                        frames_to_display.append(frame)
                
                # Combine and display frames
                if frames_to_display:
                    display_frame = self._combine_frames(frames_to_display)
                    cv2.imshow(self.config.display.window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit signal received")
                    break
                elif key == ord('s'):
                    self._print_statistics()
                
                # Update performance metrics
                loop_time = time.time() - loop_start
                self.performance_monitor.update('processing_time', loop_time)
                
                frame_count += 1
                
                # Periodic statistics logging
                if time.time() - last_stats_time >= 60:  # Every minute
                    self._log_periodic_stats()
                    last_stats_time = time.time()
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        
        finally:
            self.stop()
    
    def _process_camera(self, cam_name: str) -> Optional[np.ndarray]:
        """
        Process single camera frame
        
        Args:
            cam_name: Camera name
            
        Returns:
            Processed frame or None
        """
        # Get stream
        stream = self.stream_manager.get_stream(cam_name)
        if not stream:
            return None
        
        # Get latest frame
        frame = stream.get_frame()
        
        if frame is None:
            # Create placeholder
            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, f"Waiting for {cam_name}...", 
                (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3
            )
            return placeholder
        
        # Run detection + tracking
        results = self.model.track(
            frame,
            conf=self.config.model.confidence_threshold,
            iou=self.config.model.iou_threshold,
            persist=True,
            tracker=self.config.tracking.tracker_type,
            verbose=False,
            device=self.config.model.device
        )
        
        # Process detections
        tracker = self.trackers[cam_name]
        tracker.process_detections(frame, results)
        
        # Draw overlays
        if self.config.display.show_roi:
            self.roi_managers[cam_name].draw_roi_overlay(frame)
        
        if self.config.display.show_tracking:
            tracker.draw_tracking_overlay(frame)
        
        # Draw info panel
        self._draw_info_panel(frame, cam_name)
        
        return frame
    
    def _draw_info_panel(self, frame: np.ndarray, cam_name: str):
        """Draw information panel on frame"""
        h, w = frame.shape[:2]
        
        # Get stats
        stream = self.stream_manager.get_stream(cam_name)
        stream_stats = stream.get_stats() if stream else None
        tracker_stats = self.trackers[cam_name].get_statistics()
        
        # Create semi-transparent panel
        overlay = frame.copy()
        panel_h = 180
        cv2.rectangle(overlay, (10, 10), (450, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(
            overlay, 
            self.config.display.info_panel_alpha, 
            frame, 
            1 - self.config.display.info_panel_alpha, 
            0, 
            frame
        )
        
        # Draw text info
        y = 35
        line_height = 28
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.display.font_scale
        thickness = 2
        
        # Camera name
        cv2.putText(frame, f"Camera: {cam_name}", (20, y), 
                   font, font_scale, (0, 255, 0), thickness)
        y += line_height
        
        # FPS
        if stream_stats:
            fps_color = (0, 255, 0) if stream_stats.is_connected else (0, 0, 255)
            cv2.putText(frame, f"FPS: {stream_stats.fps:.1f}", (20, y), 
                       font, font_scale, fps_color, thickness)
        y += line_height
        
        # Vehicle counts
        cv2.putText(frame, f"Total Vehicles: {tracker_stats['total_vehicles']}", 
                   (20, y), font, font_scale, (255, 255, 255), thickness)
        y += line_height
        
        cv2.putText(frame, f"Frames Saved: {tracker_stats['saved_frames']}", 
                   (20, y), font, font_scale, (0, 255, 255), thickness)
        y += line_height
        
        cv2.putText(frame, f"Active Tracks: {tracker_stats['active_tracks']}", 
                   (20, y), font, font_scale, (255, 255, 255), thickness)
        y += line_height
        
        # Save rate
        save_rate = tracker_stats.get('save_rate', 0)
        cv2.putText(frame, f"Save Rate: {save_rate:.1f}%", 
                   (20, y), font, font_scale, (255, 255, 255), thickness)
    
    def _combine_frames(self, frames: list) -> np.ndarray:
        """Combine multiple camera frames into one display"""
        if not frames:
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Target size for each frame
        target_h = 540
        target_w = 960
        
        # Resize all frames
        resized = [cv2.resize(f, (target_w, target_h)) for f in frames]
        
        # Arrange frames
        if len(resized) == 1:
            return resized[0]
        elif len(resized) == 2:
            return np.hstack(resized)
        elif len(resized) == 3:
            # 2 on top, 1 on bottom (centered)
            top_row = np.hstack(resized[:2])
            bottom_frame = resized[2]
            
            # Center bottom frame
            pad_width = (top_row.shape[1] - bottom_frame.shape[1]) // 2
            if pad_width > 0:
                left_pad = np.zeros((bottom_frame.shape[0], pad_width, 3), dtype=np.uint8)
                right_pad = np.zeros((bottom_frame.shape[0], pad_width, 3), dtype=np.uint8)
                bottom_frame = np.hstack([left_pad, bottom_frame, right_pad])
            
            return np.vstack([top_row, bottom_frame])
        else:
            # 2x2 grid
            top_row = np.hstack(resized[:2])
            bottom_row = np.hstack(resized[2:4])
            return np.vstack([top_row, bottom_row])
    
    def _print_statistics(self):
        """Print detailed statistics"""
        logger.info("=" * 80)
        logger.info("SYSTEM STATISTICS")
        logger.info("=" * 80)
        
        # Stream statistics
        logger.info("\n--- Stream Health ---")
        health = self.stream_manager.get_health_status()
        for cam_name, is_connected in health.items():
            status = "Connected" if is_connected else "Disconnected"
            logger.info(f"{cam_name}: {status}")
        
        # Tracker statistics
        logger.info("\n--- Tracking Statistics ---")
        for cam_name, tracker in self.trackers.items():
            stats = tracker.get_statistics()
            logger.info(f"\n{cam_name}:")
            for key, value in stats.items():
                if key != 'camera':
                    logger.info(f"  {key}: {value}")
        
        # Performance statistics
        logger.info("\n--- Performance ---")
        perf_summary = self.performance_monitor.get_summary()
        for key, value in perf_summary.items():
            logger.info(f"{key}: {value:.2f}")
        
        logger.info("=" * 80)
    
    def _log_periodic_stats(self):
        """Log periodic statistics"""
        total_saved = sum(t.saved_frames for t in self.trackers.values())
        total_detected = sum(t.total_vehicles for t in self.trackers.values())
        
        logger.info(f"Periodic stats: Detected={total_detected}, Saved={total_saved}")
    
    def stop(self):
        """Stop system gracefully"""
        logger.info("Stopping system...")
        
        self.running = False
        
        # Stop all streams
        self.stream_manager.stop_all()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Final statistics
        self._print_statistics()
        
        logger.info("System stopped successfully")


def main():
    """Main entry point"""
    try:
        # Load configuration
        logger.info("Loading configuration...")
        sys_config = config
        
        # Create system
        system = VehicleCaptureSystem(sys_config)
        
        # Setup ROIs
        system.setup_rois()
        
        # Run main loop
        system.run()
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
