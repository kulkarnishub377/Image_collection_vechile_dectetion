"""
RTSP Stream Reader Module
High-performance threaded video capture with frame buffering and reconnection
"""

import cv2
import numpy as np
import threading
import time
from queue import Queue, Full
from typing import Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class StreamStats:
    """Stream statistics"""
    fps: float = 0.0
    frame_count: int = 0
    dropped_frames: int = 0
    reconnect_count: int = 0
    total_bytes: int = 0
    avg_latency: float = 0.0
    is_connected: bool = False


class RTSPStreamReader:
    """
    High-performance RTSP stream reader
    
    Features:
    - Threaded capture (no frame loss)
    - Automatic reconnection
    - Frame buffering
    - FPS monitoring
    - Latency tracking
    - Resolution adaptation
    """
    
    def __init__(self, 
                 camera_name: str, 
                 rtsp_url: str,
                 target_resolution: Tuple[int, int] = (1920, 1080),
                 buffer_size: int = 2,
                 reconnect_delay: float = 2.0):
        """
        Initialize RTSP reader
        
        Args:
            camera_name: Unique camera identifier
            rtsp_url: RTSP stream URL
            target_resolution: Desired (width, height)
            buffer_size: Frame buffer size (lower = less latency)
            reconnect_delay: Seconds to wait before reconnection
        """
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url
        self.target_resolution = target_resolution
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        
        # Thread management
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = StreamStats()
        self.frame_timestamps = []
        self.last_fps_calc = time.time()
        
        # Performance tracking
        self.capture_times = []
        
        logger.info(f"Initialized RTSP reader for {camera_name}")
        logger.debug(f"Stream URL: {rtsp_url}")
        logger.debug(f"Target resolution: {target_resolution}")
    
    def start(self):
        """Start reading stream in background thread"""
        if self.running:
            logger.warning(f"{self.camera_name} already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"Started RTSP reader for {self.camera_name}")
    
    def stop(self):
        """Stop reading stream"""
        logger.info(f"Stopping RTSP reader for {self.camera_name}")
        
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                logger.warning(f"Thread did not stop gracefully: {self.camera_name}")
        
        logger.info(f"Stopped RTSP reader for {self.camera_name}")
    
    def _capture_loop(self):
        """Main capture loop (runs in separate thread)"""
        cap = None
        
        while self.running:
            try:
                # Create/recreate capture
                if cap is None or not cap.isOpened():
                    cap = self._create_capture()
                    if cap is None:
                        time.sleep(self.reconnect_delay)
                        continue
                
                # Read frame
                start_time = time.time()
                ret, frame = cap.read()
                capture_time = time.time() - start_time
                
                if ret and frame is not None:
                    # Update frame (thread-safe)
                    with self.frame_lock:
                        self.frame = frame
                    
                    # Update statistics
                    self._update_stats(capture_time)
                    self.stats.is_connected = True
                
                else:
                    # Read failed
                    logger.warning(f"Failed to read frame from {self.camera_name}")
                    self.stats.is_connected = False
                    self.stats.dropped_frames += 1
                    
                    # Reconnect
                    if cap:
                        cap.release()
                        cap = None
                    
                    time.sleep(self.reconnect_delay)
            
            except Exception as e:
                logger.error(f"Error in capture loop for {self.camera_name}: {e}")
                self.stats.is_connected = False
                
                if cap:
                    cap.release()
                    cap = None
                
                time.sleep(self.reconnect_delay)
        
        # Cleanup
        if cap:
            cap.release()
        
        logger.info(f"Capture loop ended for {self.camera_name}")
    
    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        """
        Create video capture with optimal settings
        
        Returns:
            VideoCapture object or None
        """
        try:
            logger.info(f"Connecting to {self.camera_name}...")
            
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                logger.error(f"Failed to open stream: {self.camera_name}")
                return None
            
            # Set buffer size (lower = less latency)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # Try to set target resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_resolution[1])
            
            # Get actual resolution
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Connected to {self.camera_name}")
            logger.info(f"Resolution: {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")
            
            # Update reconnect count
            self.stats.reconnect_count += 1
            
            return cap
        
        except Exception as e:
            logger.error(f"Error creating capture for {self.camera_name}: {e}")
            return None
    
    def _update_stats(self, capture_time: float):
        """Update stream statistics"""
        current_time = time.time()
        
        # Update frame count
        self.stats.frame_count += 1
        
        # Track capture times for latency calculation
        self.capture_times.append(capture_time)
        if len(self.capture_times) > 100:
            self.capture_times.pop(0)
        
        self.stats.avg_latency = np.mean(self.capture_times) if self.capture_times else 0.0
        
        # Calculate FPS (every second)
        if current_time - self.last_fps_calc >= 1.0:
            # Count frames in last second
            self.frame_timestamps = [t for t in self.frame_timestamps 
                                    if current_time - t < 1.0]
            self.stats.fps = len(self.frame_timestamps)
            self.last_fps_calc = current_time
        
        # Add timestamp
        self.frame_timestamps.append(current_time)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get latest frame (thread-safe)
        
        Returns:
            Frame copy or None if no frame available
        """
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def get_stats(self) -> StreamStats:
        """
        Get current stream statistics
        
        Returns:
            StreamStats object
        """
        return self.stats
    
    def is_connected(self) -> bool:
        """Check if stream is connected"""
        return self.stats.is_connected
    
    def wait_for_frame(self, timeout: float = 10.0) -> Optional[np.ndarray]:
        """
        Wait for first frame with timeout
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            First frame or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            frame = self.get_frame()
            if frame is not None:
                return frame
            time.sleep(0.1)
        
        logger.warning(f"Timeout waiting for frame from {self.camera_name}")
        return None
    
    def get_resolution(self) -> Optional[Tuple[int, int]]:
        """
        Get current frame resolution
        
        Returns:
            (width, height) or None
        """
        frame = self.get_frame()
        if frame is not None:
            h, w = frame.shape[:2]
            return (w, h)
        return None


class MultiStreamManager:
    """
    Manage multiple RTSP streams
    
    Features:
    - Centralized stream management
    - Health monitoring
    - Statistics aggregation
    """
    
    def __init__(self):
        self.readers: dict[str, RTSPStreamReader] = {}
        logger.info("Initialized multi-stream manager")
    
    def add_stream(self, 
                   camera_name: str, 
                   rtsp_url: str,
                   **kwargs) -> RTSPStreamReader:
        """
        Add and start a new stream
        
        Args:
            camera_name: Unique camera identifier
            rtsp_url: RTSP stream URL
            **kwargs: Additional arguments for RTSPStreamReader
            
        Returns:
            RTSPStreamReader instance
        """
        if camera_name in self.readers:
            logger.warning(f"Stream already exists: {camera_name}")
            return self.readers[camera_name]
        
        reader = RTSPStreamReader(camera_name, rtsp_url, **kwargs)
        reader.start()
        self.readers[camera_name] = reader
        
        logger.info(f"Added stream: {camera_name}")
        return reader
    
    def remove_stream(self, camera_name: str):
        """Remove and stop a stream"""
        if camera_name in self.readers:
            self.readers[camera_name].stop()
            del self.readers[camera_name]
            logger.info(f"Removed stream: {camera_name}")
    
    def get_stream(self, camera_name: str) -> Optional[RTSPStreamReader]:
        """Get stream reader by name"""
        return self.readers.get(camera_name)
    
    def get_all_frames(self) -> dict[str, Optional[np.ndarray]]:
        """Get latest frames from all streams"""
        return {name: reader.get_frame() 
                for name, reader in self.readers.items()}
    
    def get_all_stats(self) -> dict[str, StreamStats]:
        """Get statistics from all streams"""
        return {name: reader.get_stats() 
                for name, reader in self.readers.items()}
    
    def stop_all(self):
        """Stop all streams"""
        logger.info("Stopping all streams...")
        for reader in self.readers.values():
            reader.stop()
        self.readers.clear()
        logger.info("All streams stopped")
    
    def get_health_status(self) -> dict[str, bool]:
        """Get connection status of all streams"""
        return {name: reader.is_connected() 
                for name, reader in self.readers.items()}
