"""
PROFESSIONAL VEHICLE DETECTION & TRACKING SYSTEM
================================================
Purpose: High-performance data collection for vehicle AI training
Features: Multi-camera RTSP, ROI-based tracking, duplicate prevention, dataset generation
"""

import cv2
import numpy as np
from ultralytics import YOLO
import threading
from queue import Queue
import time
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """System configuration"""
    
    # RTSP Streams
    CAMERAS = {
       ggdg#put you g
    }
    
    # Model
    MODEL_PATH = "yolov8n.pt"
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    
    # Vehicle classes
    VEHICLE_CLASSES = #put the vechile class of your models
    
    # Paths
    BASE_DIR = Path("images")
    FRAME_DIR = BASE_DIR / "frame"
    CROP_DIR = BASE_DIR / "croped"
    ROI_CONFIG = "roi_config.json"
    
    # Display
    WINDOW_NAME = "Vehicle Tracking System - Multi-Camera View"
    DISPLAY_WIDTH = 1920
    DISPLAY_HEIGHT = 1080
    
    # Performance
    FRAME_BUFFER_SIZE = 2
    SAVE_FORMAT = "jpg"
    SAVE_QUALITY = 100  # Maximum quality
    
    # Tracking
    TRACKER_TYPE = "bytetrack.yaml"  # Built-in ByteTrack
    MAX_TRACK_AGE = 30  # Frames to keep lost tracks
    
    # ROI
    ROI_COLOR = (0, 255, 0)  # Green
    ROI_THICKNESS = 2
    POINT_RADIUS = 5


# ============================================================================
# LOGGER SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ROI MANAGER
# ============================================================================

class ROIManager:
    """Handles ROI drawing and validation"""
    
    def __init__(self, camera_name, roi_config_path):
        self.camera_name = camera_name
        self.roi_config_path = roi_config_path
        self.roi_points = []
        self.is_drawing = False
        self.temp_point = None
        
    def load_roi(self):
        """Load ROI from config file"""
        try:
            if Path(self.roi_config_path).exists():
                with open(self.roi_config_path, 'r') as f:
                    data = json.load(f)
                    if self.camera_name in data:
                        self.roi_points = [tuple(p) for p in data[self.camera_name]]
                        logger.info(f"Loaded ROI for {self.camera_name}: {len(self.roi_points)} points")
                        return True
        except Exception as e:
            logger.error(f"Error loading ROI: {e}")
        return False
    
    def save_roi(self):
        """Save ROI to config file"""
        try:
            data = {}
            if Path(self.roi_config_path).exists():
                with open(self.roi_config_path, 'r') as f:
                    data = json.load(f)
            
            data[self.camera_name] = self.roi_points
            
            with open(self.roi_config_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved ROI for {self.camera_name}")
        except Exception as e:
            logger.error(f"Error saving ROI: {e}")
    
    def draw_roi_interactive(self, frame):
        """Interactive ROI drawing"""
        clone = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal clone
            
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi_points.append((x, y))
                clone = frame.copy()
                self._draw_roi(clone, show_temp=True)
                cv2.imshow("Draw ROI - Click to add points, Press 'c' to complete, 'r' to reset", clone)
            
            elif event == cv2.EVENT_MOUSEMOVE:
                self.temp_point = (x, y)
                clone = frame.copy()
                self._draw_roi(clone, show_temp=True)
                cv2.imshow("Draw ROI - Click to add points, Press 'c' to complete, 'r' to reset", clone)
        
        cv2.namedWindow("Draw ROI - Click to add points, Press 'c' to complete, 'r' to reset")
        cv2.setMouseCallback("Draw ROI - Click to add points, Press 'c' to complete, 'r' to reset", mouse_callback)
        cv2.imshow("Draw ROI - Click to add points, Press 'c' to complete, 'r' to reset", clone)
        
        logger.info(f"Draw ROI for {self.camera_name}. Press 'c' to complete, 'r' to reset")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and len(self.roi_points) >= 3:
                cv2.destroyWindow("Draw ROI - Click to add points, Press 'c' to complete, 'r' to reset")
                self.save_roi()
                break
            elif key == ord('r'):
                self.roi_points = []
                clone = frame.copy()
                cv2.imshow("Draw ROI - Click to add points, Press 'c' to complete, 'r' to reset", clone)
    
    def _draw_roi(self, frame, show_temp=False):
        """Draw ROI on frame"""
        if len(self.roi_points) > 0:
            # Draw points
            for point in self.roi_points:
                cv2.circle(frame, point, Config.POINT_RADIUS, Config.ROI_COLOR, -1)
            
            # Draw lines
            for i in range(len(self.roi_points)):
                pt1 = self.roi_points[i]
                pt2 = self.roi_points[(i + 1) % len(self.roi_points)]
                cv2.line(frame, pt1, pt2, Config.ROI_COLOR, Config.ROI_THICKNESS)
            
            # Draw temporary line
            if show_temp and self.temp_point and len(self.roi_points) > 0:
                cv2.line(frame, self.roi_points[-1], self.temp_point, (255, 255, 0), 1)
    
    def is_point_in_roi(self, point):
        """Check if point is inside ROI polygon"""
        if len(self.roi_points) < 3:
            return True  # No ROI means all points valid
        
        roi_array = np.array(self.roi_points, dtype=np.int32)
        result = cv2.pointPolygonTest(roi_array, point, False)
        return result >= 0
    
    def draw_roi_overlay(self, frame):
        """Draw ROI overlay on frame"""
        if len(self.roi_points) >= 3:
            overlay = frame.copy()
            roi_array = np.array(self.roi_points, dtype=np.int32)
            cv2.fillPoly(overlay, [roi_array], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [roi_array], True, Config.ROI_COLOR, Config.ROI_THICKNESS)


# ============================================================================
# RTSP STREAM READER (THREADED)
# ============================================================================

class RTSPReader:
    """Thread-safe RTSP stream reader with no frame loss"""
    
    def __init__(self, camera_name, rtsp_url):
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread = None
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
    def start(self):
        """Start reading stream"""
        self.running = True
        self.thread = threading.Thread(target=self._read_stream, daemon=True)
        self.thread.start()
        logger.info(f"Started RTSP reader for {self.camera_name}")
    
    def _read_stream(self):
        """Continuous frame reading (runs in separate thread)"""
        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.FRAME_BUFFER_SIZE)
        
        # Try to set high resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        logger.info(f"Connected to {self.camera_name}")
        
        while self.running:
            ret, frame = cap.read()
            
            if ret:
                with self.frame_lock:
                    self.frame = frame
                    self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_time = current_time
            else:
                logger.warning(f"Failed to read frame from {self.camera_name}, reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.FRAME_BUFFER_SIZE)
        
        cap.release()
        logger.info(f"Stopped RTSP reader for {self.camera_name}")
    
    def get_frame(self):
        """Get latest frame (thread-safe)"""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        """Stop reading stream"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)


# ============================================================================
# VEHICLE TRACKER
# ============================================================================

class VehicleTracker:
    """Tracks vehicles and prevents duplicates"""
    
    def __init__(self, camera_name, roi_manager, save_dir_frame, save_dir_crop):
        self.camera_name = camera_name
        self.roi_manager = roi_manager
        self.save_dir_frame = save_dir_frame
        self.save_dir_crop = save_dir_crop
        
        # Create save directories
        self.save_dir_frame.mkdir(parents=True, exist_ok=True)
        self.save_dir_crop.mkdir(parents=True, exist_ok=True)
        
        # Tracking state
        self.tracked_ids = set()  # IDs we've already saved
        self.active_tracks = {}  # Currently active tracks
        
        # Counters
        self.total_vehicles = 0
        self.saved_frames = 0
        
        logger.info(f"Initialized tracker for {self.camera_name}")
    
    def process_detections(self, frame, results):
        """Process YOLO tracking results"""
        current_tracks = {}
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get detection info
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                
                # Get tracking ID (if available)
                track_id = None
                if hasattr(boxes, 'id') and boxes.id is not None:
                    track_id = int(boxes.id[i])
                
                # Calculate center point
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Check if center is in ROI
                if not self.roi_manager.is_point_in_roi((center_x, center_y)):
                    continue
                
                # If we have tracking ID
                if track_id is not None:
                    current_tracks[track_id] = {
                        'box': box,
                        'center': (center_x, center_y),
                        'conf': conf,
                        'cls': cls
                    }
                    
                    # New vehicle detected
                    if track_id not in self.tracked_ids:
                        self.tracked_ids.add(track_id)
                        self.total_vehicles += 1
                        
                        # Save images
                        self._save_vehicle_images(frame, box, track_id, cls)
        
        self.active_tracks = current_tracks
    
    def _save_vehicle_images(self, frame, box, track_id, cls):
        """Save full frame and cropped vehicle image"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Save full frame
            frame_filename = f"frame_{self.saved_frames}.{Config.SAVE_FORMAT}"
            frame_path = self.save_dir_frame / frame_filename
            
            if Config.SAVE_FORMAT == "jpg":
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, Config.SAVE_QUALITY])
            else:
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            # Save cropped vehicle
            x1, y1, x2, y2 = map(int, box)
            # Add padding
            h, w = frame.shape[:2]
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0:
                crop_filename = f"frame_{self.saved_frames}.{Config.SAVE_FORMAT}"
                crop_path = self.save_dir_crop / crop_filename
                
                if Config.SAVE_FORMAT == "jpg":
                    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, Config.SAVE_QUALITY])
                else:
                    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            self.saved_frames += 1
            logger.info(f"[{self.camera_name}] Saved vehicle #{self.saved_frames} (ID: {track_id}, Class: {Config.VEHICLE_CLASSES[cls]})")
            
        except Exception as e:
            logger.error(f"Error saving images: {e}")
    
    def draw_tracking_overlay(self, frame):
        """Draw tracking visualization"""
        for track_id, track_info in self.active_tracks.items():
            box = track_info['box']
            center = track_info['center']
            conf = track_info['conf']
            cls = track_info['cls']
            
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            color = (0, 255, 0) if track_id in self.tracked_ids else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(frame, center, 5, color, -1)
            
            # Draw label
            label = f"ID:{track_id} {Config.VEHICLE_CLASSES[cls]} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


# ============================================================================
# MAIN SYSTEM
# ============================================================================

class VehicleCapturSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        logger.info("Initializing Vehicle Capture System...")
        
        # Load YOLO model
        logger.info(f"Loading YOLO model: {Config.MODEL_PATH}")
        self.model = YOLO(Config.MODEL_PATH)
        
        # Verify model classes
        model_names = self.model.names
        logger.info(f"Model classes: {model_names}")
        
        # Initialize components
        self.rtsp_readers = {}
        self.roi_managers = {}
        self.trackers = {}
        
        # Setup cameras
        self._setup_cameras()
        
        # System state
        self.running = False
        
    def _setup_cameras(self):
        """Setup all camera components"""
        for cam_name, rtsp_url in Config.CAMERAS.items():
            # Create camera-specific directories
            frame_dir = Config.FRAME_DIR / cam_name
            crop_dir = Config.CROP_DIR / cam_name
            
            # RTSP Reader
            self.rtsp_readers[cam_name] = RTSPReader(cam_name, rtsp_url)
            
            # ROI Manager
            self.roi_managers[cam_name] = ROIManager(cam_name, Config.ROI_CONFIG)
            
            # Vehicle Tracker
            self.trackers[cam_name] = VehicleTracker(cam_name, self.roi_managers[cam_name], frame_dir, crop_dir)
            
            logger.info(f"Setup complete for camera: {cam_name}")
    
    def setup_rois(self):
        """Setup ROI for all cameras"""
        logger.info("Starting ROI setup...")
        
        # Start RTSP readers temporarily
        for reader in self.rtsp_readers.values():
            reader.start()
        
        # Wait for first frames
        time.sleep(3)
        
        for cam_name, roi_manager in self.roi_managers.items():
            # Try to load existing ROI
            if roi_manager.load_roi():
                logger.info(f"Using existing ROI for {cam_name}")
                continue
            
            # Get frame for ROI drawing
            frame = self.rtsp_readers[cam_name].get_frame()
            
            if frame is not None:
                logger.info(f"Draw ROI for camera: {cam_name}")
                roi_manager.draw_roi_interactive(frame)
            else:
                logger.error(f"Could not get frame from {cam_name}")
        
        # Stop readers (will restart in main loop)
        for reader in self.rtsp_readers.values():
            reader.stop()
        
        logger.info("ROI setup complete")
    
    def run(self):
        """Main processing loop"""
        logger.info("Starting main processing loop...")
        
        # Start RTSP readers
        for reader in self.rtsp_readers.values():
            reader.start()
        
        # Wait for streams to stabilize
        time.sleep(2)
        
        self.running = True
        
        # Create display window
        cv2.namedWindow(Config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(Config.WINDOW_NAME, Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT)
        
        logger.info("System running. Press 'q' to quit")
        
        while self.running:
            frames_to_display = []
            
            for cam_name in Config.CAMERAS.keys():
                # Get latest frame
                frame = self.rtsp_readers[cam_name].get_frame()
                
                if frame is None:
                    # Create placeholder
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(frame, f"Waiting for {cam_name}...", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    # Run detection + tracking
                    results = self.model.track(
                        frame,
                        conf=Config.CONFIDENCE_THRESHOLD,
                        iou=Config.IOU_THRESHOLD,
                        persist=True,
                        tracker=Config.TRACKER_TYPE,
                        verbose=False
                    )
                    
                    # Process detections
                    self.trackers[cam_name].process_detections(frame, results)
                    
                    # Draw overlays
                    self.roi_managers[cam_name].draw_roi_overlay(frame)
                    self.trackers[cam_name].draw_tracking_overlay(frame)
                    
                    # Draw info panel
                    self._draw_info_panel(frame, cam_name)
                
                frames_to_display.append(frame)
            
            # Combine frames
            display_frame = self._combine_frames(frames_to_display)
            
            # Show
            cv2.imshow(Config.WINDOW_NAME, display_frame)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit signal received")
                self.running = False
        
        # Cleanup
        self.stop()
    
    def _draw_info_panel(self, frame, cam_name):
        """Draw information panel on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text info
        fps = self.rtsp_readers[cam_name].fps
        total = self.trackers[cam_name].total_vehicles
        saved = self.trackers[cam_name].saved_frames
        active = len(self.trackers[cam_name].active_tracks)
        
        y_offset = 35
        cv2.putText(frame, f"Camera: {cam_name}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 30
        cv2.putText(frame, f"FPS: {fps}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Vehicles Detected: {total}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Frames Saved: {saved}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Active Tracks: {active}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _combine_frames(self, frames):
        """Combine multiple camera frames into one display"""
        if not frames:
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Resize frames to same size
        target_h = 540
        target_w = 960
        
        resized = []
        for frame in frames:
            resized.append(cv2.resize(frame, (target_w, target_h)))
        
        # Stack frames
        if len(resized) == 1:
            return resized[0]
        elif len(resized) == 2:
            return np.hstack(resized)
        else:
            # 3 cameras: 2 on top, 1 on bottom
            top_row = np.hstack(resized[:2])
            bottom_frame = resized[2]
            # Pad bottom frame to match width
            pad_width = top_row.shape[1] - bottom_frame.shape[1]
            if pad_width > 0:
                padding = np.zeros((bottom_frame.shape[0], pad_width, 3), dtype=np.uint8)
                bottom_frame = np.hstack([bottom_frame, padding])
            return np.vstack([top_row, bottom_frame])
    
    def stop(self):
        """Stop all components"""
        logger.info("Stopping system...")
        
        # Stop RTSP readers
        for reader in self.rtsp_readers.values():
            reader.stop()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Print final statistics
        logger.info("=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 60)
        for cam_name, tracker in self.trackers.items():
            logger.info(f"{cam_name}:")
            logger.info(f"  Total Vehicles Detected: {tracker.total_vehicles}")
            logger.info(f"  Frames Saved: {tracker.saved_frames}")
        logger.info("=" * 60)
        logger.info("System stopped successfully")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    try:
        # Create system
        system = VehicleCapturSystem()
        
        # Setup ROIs
        system.setup_rois()
        
        # Run main loop
        system.run()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
    finally:
        logger.info("Exiting...")


if __name__ == "__main__":
    main()
