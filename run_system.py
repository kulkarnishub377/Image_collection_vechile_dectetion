"""
SIMPLIFIED WORKING VERSION - Vehicle Capture System
Run this file directly for immediate functionality
"""

import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

# RTSP Streams
CAMERAS = {
    "overview": "rtsp://admin:Arya@123@125.18.39.10:5554/cam/realmonitor?channel=1&subtype=0",
    "anpr": "rtsp://admin:BE04_ViDeS@125.18.39.10:5555/Streaming/Channels/101",
    "ptz": "rtsp://admin:Arya_123@125.18.39.10:554/cam/realmonitor?channel=1&subtype=0"
}

# Model settings
MODEL_PATH = "bestv4.pt"
CONFIDENCE = 0.5
IOU_THRESHOLD = 0.45

# Paths
BASE_DIR = Path("images")
FRAME_DIR = BASE_DIR / "frame"
CROP_DIR = BASE_DIR / "croped"
ROI_CONFIG = "roi_config.json"

# Quality settings
SAVE_FORMAT = "jpg"
JPEG_QUALITY = 100  # Maximum quality
RESOLUTION = (1920, 1080)
BUFFER_SIZE = 3

# Vehicle classes
VEHICLE_CLASSES = ['auto_rickshaw', 'bike', 'bus', 'car', 'mini_bus', 'tractor', 'truck']

# Global counters
GLOBAL_SAVED_COUNT = 0
GLOBAL_VEHICLE_COUNT = 0
counter_lock = threading.Lock()

print("=" * 80)
print("PROFESSIONAL VEHICLE CAPTURE SYSTEM")
print("=" * 80)


# ============================================================================
# ROI MANAGER
# ============================================================================

class SimpleROI:
    def __init__(self, camera_name):
        self.camera_name = camera_name
        self.points = []
        self.temp_point = None
        
    def load(self):
        try:
            if Path(ROI_CONFIG).exists():
                with open(ROI_CONFIG, 'r') as f:
                    data = json.load(f)
                    if self.camera_name in data:
                        self.points = [tuple(p) for p in data[self.camera_name]]
                        print(f"✓ Loaded ROI for {self.camera_name}")
                        return True
        except:
            pass
        return False
    
    def save(self):
        try:
            data = {}
            if Path(ROI_CONFIG).exists():
                with open(ROI_CONFIG, 'r') as f:
                    data = json.load(f)
            data[self.camera_name] = self.points
            with open(ROI_CONFIG, 'w') as f:
                json.dump(data, f)
            print(f"✓ Saved ROI for {self.camera_name}")
        except Exception as e:
            print(f"Error saving ROI: {e}")
    
    def draw_interactive(self, frame):
        clone = frame.copy()
        window = f"ROI: {self.camera_name} - Click points, 'c'=done, 'r'=reset"
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal clone
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
            elif event == cv2.EVENT_MOUSEMOVE:
                self.temp_point = (x, y)
            
            clone = frame.copy()
            self._draw_preview(clone)
            cv2.imshow(window, clone)
        
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, mouse_callback)
        cv2.imshow(window, clone)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(self.points) >= 3:
                cv2.destroyWindow(window)
                self.save()
                return True
            elif key == ord('r'):
                self.points = []
                clone = frame.copy()
                cv2.imshow(window, clone)
    
    def _draw_preview(self, frame):
        for i, pt in enumerate(self.points):
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (pt[0]+10, pt[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if len(self.points) > 1:
            for i in range(len(self.points)):
                cv2.line(frame, self.points[i], 
                        self.points[(i+1) % len(self.points)], 
                        (0, 255, 0), 2)
        
        if self.temp_point and self.points:
            cv2.line(frame, self.points[-1], self.temp_point, (255, 255, 0), 1)
    
    def is_inside(self, point):
        if len(self.points) < 3:
            return True
        roi_array = np.array(self.points, dtype=np.int32)
        return cv2.pointPolygonTest(roi_array, point, False) >= 0
    
    def draw_overlay(self, frame):
        if len(self.points) >= 3:
            overlay = frame.copy()
            roi_array = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(overlay, [roi_array], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [roi_array], True, (0, 255, 0), 2)


# ============================================================================
# STREAM READER
# ============================================================================

class StreamReader:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.fps = 0
        
    def start(self):
        self.running = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        print(f"✓ Started stream: {self.name}")
    
    def _read_loop(self):
        cap = None
        fps_count = 0
        fps_time = time.time()
        
        while self.running:
            try:
                if cap is None or not cap.isOpened():
                    print(f"Connecting to {self.name}...")
                    cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    
                    if cap.isOpened():
                        # Set highest quality
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"✓ Connected {self.name}: {actual_w}x{actual_h}")
                    else:
                        time.sleep(2)
                        continue
                
                ret, frame = cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                    
                    # Calculate FPS
                    fps_count += 1
                    if time.time() - fps_time >= 1.0:
                        self.fps = fps_count
                        fps_count = 0
                        fps_time = time.time()
                else:
                    print(f"⚠ Frame read failed: {self.name}")
                    if cap:
                        cap.release()
                    cap = None
                    time.sleep(2)
            
            except Exception as e:
                print(f"Error in {self.name}: {e}")
                if cap:
                    cap.release()
                cap = None
                time.sleep(2)
        
        if cap:
            cap.release()
    
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.running = False


# ============================================================================
# TRACKER
# ============================================================================

class VehicleTracker:
    def __init__(self, camera_name, roi, save_frame_dir, save_crop_dir):
        self.camera_name = camera_name
        self.roi = roi
        self.save_frame_dir = save_frame_dir
        self.save_crop_dir = save_crop_dir
        
        self.save_frame_dir.mkdir(parents=True, exist_ok=True)
        self.save_crop_dir.mkdir(parents=True, exist_ok=True)
        
        self.saved_ids = set()
        self.active_tracks = {}
        
        self.camera_vehicle_count = 0
        self.camera_saved_count = 0
        
        print(f"✓ Initialized tracker: {camera_name}")
    
    def process(self, frame, results):
        global GLOBAL_SAVED_COUNT, GLOBAL_VEHICLE_COUNT
        
        current_tracks = {}
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                
                # Get tracking ID
                if not hasattr(boxes, 'id') or boxes.id is None:
                    continue
                
                track_id = int(boxes.id[i])
                
                # Calculate center
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Check ROI
                if not self.roi.is_inside((center_x, center_y)):
                    continue
                
                # Track info
                current_tracks[track_id] = {
                    'box': box,
                    'center': (center_x, center_y),
                    'conf': conf,
                    'cls': cls
                }
                
                # New vehicle
                if track_id not in self.saved_ids:
                    self.saved_ids.add(track_id)
                    
                    with counter_lock:
                        self.camera_vehicle_count += 1
                        GLOBAL_VEHICLE_COUNT += 1
                    
                    # Save images
                    self._save_vehicle(frame, box, track_id, cls)
        
        self.active_tracks = current_tracks
    
    def _save_vehicle(self, frame, box, track_id, cls):
        global GLOBAL_SAVED_COUNT
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Filenames with global counter
            with counter_lock:
                count = GLOBAL_SAVED_COUNT
                GLOBAL_SAVED_COUNT += 1
                self.camera_saved_count += 1
            
            frame_name = f"vehicle_{count:06d}_{self.camera_name}_{timestamp}.{SAVE_FORMAT}"
            crop_name = f"vehicle_{count:06d}_{self.camera_name}_{timestamp}.{SAVE_FORMAT}"
            
            # Save full frame (highest quality)
            frame_path = self.save_frame_dir / frame_name
            params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY, 
                     cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                     cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
            cv2.imwrite(str(frame_path), frame, params)
            
            # Save crop
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crop_path = self.save_crop_dir / crop_name
                cv2.imwrite(str(crop_path), crop, params)
            
            class_name = VEHICLE_CLASSES[cls] if cls < len(VEHICLE_CLASSES) else f"class_{cls}"
            print(f"✓ [{self.camera_name}] Saved #{count:06d} - {class_name} (ID:{track_id})")
        
        except Exception as e:
            print(f"Error saving: {e}")
    
    def draw_overlay(self, frame):
        for track_id, info in self.active_tracks.items():
            box = info['box']
            center = info['center']
            conf = info['conf']
            cls = info['cls']
            
            x1, y1, x2, y2 = map(int, box)
            
            # Color: green if saved, yellow if tracking
            color = (0, 255, 0) if track_id in self.saved_ids else (0, 255, 255)
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, center, 5, color, -1)
            
            # Label
            class_name = VEHICLE_CLASSES[cls] if cls < len(VEHICLE_CLASSES) else f"cls{cls}"
            label = f"ID:{track_id} {class_name} {conf:.2f}"
            
            # Background for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1-th-5), (x1+tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


# ============================================================================
# MAIN SYSTEM
# ============================================================================

def main():
    global GLOBAL_SAVED_COUNT, GLOBAL_VEHICLE_COUNT
    
    print("\n🚀 Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print(f"✓ Model loaded: {model.names}")
    
    # Setup cameras
    print("\n📹 Setting up cameras...")
    streams = {}
    rois = {}
    trackers = {}
    
    for cam_name, url in CAMERAS.items():
        # Stream reader
        streams[cam_name] = StreamReader(cam_name, url)
        
        # ROI
        rois[cam_name] = SimpleROI(cam_name)
        
        # Tracker
        frame_dir = FRAME_DIR / cam_name
        crop_dir = CROP_DIR / cam_name
        trackers[cam_name] = VehicleTracker(cam_name, rois[cam_name], frame_dir, crop_dir)
    
    # Start streams
    print("\n📡 Starting streams...")
    for stream in streams.values():
        stream.start()
    
    time.sleep(3)
    
    # Setup ROIs
    print("\n🎯 Setting up ROIs...")
    for cam_name, roi in rois.items():
        if roi.load():
            continue
        
        # Wait for frame
        for _ in range(50):
            frame = streams[cam_name].get_frame()
            if frame is not None:
                break
            time.sleep(0.1)
        
        if frame is not None:
            print(f"\n>>> Draw ROI for {cam_name}")
            roi.draw_interactive(frame)
    
    # Main loop
    print("\n✅ System running! Press 'q' to quit\n")
    
    cv2.namedWindow("Vehicle Tracking System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vehicle Tracking System", 1920, 1080)
    
    try:
        while True:
            frames_to_show = []
            
            for cam_name in CAMERAS.keys():
                frame = streams[cam_name].get_frame()
                
                if frame is None:
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(frame, f"Waiting for {cam_name}...", (400, 360),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    # Detect and track
                    results = model.track(
                        frame,
                        conf=CONFIDENCE,
                        iou=IOU_THRESHOLD,
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False
                    )
                    
                    # Process
                    trackers[cam_name].process(frame, results)
                    
                    # Draw overlays
                    rois[cam_name].draw_overlay(frame)
                    trackers[cam_name].draw_overlay(frame)
                    
                    # Info panel
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    y = 35
                    cv2.putText(frame, f"Camera: {cam_name}", (20, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y += 30
                    cv2.putText(frame, f"FPS: {streams[cam_name].fps}", (20, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y += 30
                    cv2.putText(frame, f"Camera Vehicles: {trackers[cam_name].camera_vehicle_count}", 
                               (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y += 30
                    cv2.putText(frame, f"Camera Saved: {trackers[cam_name].camera_saved_count}", 
                               (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y += 30
                    cv2.putText(frame, f"Active: {len(trackers[cam_name].active_tracks)}", 
                               (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y += 35
                    
                    # Global count (bigger, highlighted)
                    cv2.putText(frame, f"TOTAL SAVED: {GLOBAL_SAVED_COUNT} / {GLOBAL_VEHICLE_COUNT}", 
                               (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)
                
                frames_to_show.append(frame)
            
            # Combine frames
            if len(frames_to_show) == 1:
                display = frames_to_show[0]
            elif len(frames_to_show) == 2:
                display = np.hstack([cv2.resize(f, (960, 540)) for f in frames_to_show])
            else:
                top = np.hstack([cv2.resize(frames_to_show[0], (960, 540)),
                                cv2.resize(frames_to_show[1], (960, 540))])
                bottom = cv2.resize(frames_to_show[2], (960, 540))
                padding = np.zeros((540, 960, 3), dtype=np.uint8)
                bottom = np.hstack([bottom, padding])
                display = np.vstack([top, bottom])
            
            cv2.imshow("Vehicle Tracking System", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n⏹ Stopped by user")
    
    finally:
        print("\n📊 Final Statistics:")
        print("=" * 60)
        print(f"TOTAL VEHICLES DETECTED: {GLOBAL_VEHICLE_COUNT}")
        print(f"TOTAL FRAMES SAVED: {GLOBAL_SAVED_COUNT}")
        print("=" * 60)
        
        for cam_name in CAMERAS.keys():
            print(f"\n{cam_name}:")
            print(f"  Vehicles: {trackers[cam_name].camera_vehicle_count}")
            print(f"  Saved: {trackers[cam_name].camera_saved_count}")
        
        print("\n🛑 Shutting down...")
        for stream in streams.values():
            stream.stop()
        
        cv2.destroyAllWindows()
        print("✅ Done!")


if __name__ == "__main__":
    main()
