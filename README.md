# Professional Vehicle Capture System

A production-grade vehicle detection and tracking system for building perfect AI training datasets.

## 🏗️ Architecture

**Modular Design** with separation of concerns:

```
├── main.py              # System orchestrator
├── config.py            # Configuration management
├── stream_reader.py     # RTSP stream handling
├── roi_manager.py       # ROI management
├── tracker.py           # Advanced vehicle tracking
├── utils.py             # Utility functions
└── vehicle_capture_system.py  # Legacy monolithic version (backup)
```

## ✨ Features

### 🎯 Core Capabilities
- **Multi-Camera Support**: Handle multiple RTSP streams simultaneously
- **Advanced Tracking**: ByteTrack + Kalman filtering for zero duplicates
- **ROI-Based Filtering**: User-drawn polygon ROIs per camera
- **Quality Control**: Blur detection, brightness check, stationary vehicle filtering
- **High Performance**: Multi-threaded capture, maximum FPS
- **Production-Ready**: Logging, error handling, graceful shutdown

### 📊 Advanced Features
- **Speed Estimation**: Real-time vehicle speed calculation
- **Direction Detection**: 8-way direction classification
- **Trajectory Analysis**: Path smoothing and analysis
- **Image Quality Check**: Blur and brightness validation
- **Performance Monitoring**: FPS, latency, and system metrics
- **Automatic Reconnection**: Stream failure recovery

### 💾 Dataset Generation
- **100% Quality Images**: Lossless PNG or maximum JPEG quality
- **Organized Storage**: Camera-wise folder structure
- **Sequential Naming**: `frame_0.jpg`, `frame_1.jpg`, etc.
- **Dual Save**: Full frames + cropped vehicles
- **No Duplicates**: Each vehicle saved exactly once

## 🚀 Quick Start

### Installation

```bash
pip install ultralytics opencv-python numpy
```

### Run the System

```bash
python main.py
```

### First-Time Setup

1. **ROI Drawing**: For each camera, draw polygon ROI:
   - Click to add points
   - Press `c` to complete
   - Press `r` to reset
   - ROI saved to `roi_config.json`

2. **Monitoring**: Watch live view with:
   - FPS counter
   - Vehicle count
   - Saved frames count
   - Active tracks
   - Speed and direction

3. **Controls**:
   - Press `q` to quit
   - Press `s` for statistics

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Camera settings
CAMERAS = {
    "overview": "rtsp://...",
    "anpr": "rtsp://...",
    "ptz": "rtsp://..."
}

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Quality settings
JPEG_QUALITY = 100
MIN_BLUR_THRESHOLD = 100.0

# Tracking settings
TRACKER_TYPE = "bytetrack.yaml"
ENABLE_SPEED_ESTIMATION = True
ENABLE_DIRECTION_DETECTION = True
```

## 📁 Output Structure

```
images/
├── frame/
│   ├── overview/
│   │   ├── frame_0.jpg
│   │   ├── frame_1.jpg
│   │   └── ...
│   ├── anpr/
│   └── ptz/
└── croped/
    ├── overview/
    │   ├── frame_0.jpg
    │   ├── frame_1.jpg
    │   └── ...
    ├── anpr/
    └── ptz/
```

## 🔧 Advanced Usage

### Custom Vehicle Classes

The system uses your custom-trained YOLO model (`bestv4.pt`) with classes:
- auto_rickshaw
- bike
- bus
- car
- mini_bus
- tractor
- truck

### Speed Calibration

For accurate speed estimation, calibrate `pixels_per_meter` in `config.py`:

1. Measure a known distance in the frame (e.g., 5 meters)
2. Count pixels for that distance
3. Set `pixels_per_meter = pixels / meters`

### Performance Tuning

**For Maximum FPS**:
- Set `buffer_size = 1` (lower latency)
- Enable GPU: `device = "0"`
- Disable quality checks: `enable_quality_check = False`

**For Best Quality**:
- Set `min_blur_threshold = 150`
- Enable quality checks
- Use PNG format for lossless compression

## 📊 Logging

System logs to:
- **Console**: Real-time status
- **File**: `vehicle_capture.log` (detailed logs)

Log levels:
- INFO: Normal operation
- WARNING: Non-critical issues
- ERROR: Critical errors

## 🎯 How It Works

1. **RTSP Capture**: Threaded readers continuously pull frames
2. **Detection**: YOLO detects vehicles in each frame
3. **Tracking**: ByteTrack assigns unique IDs and follows vehicles
4. **ROI Filtering**: Only vehicles with center in ROI are tracked
5. **Quality Check**: Blur and brightness validation
6. **Duplicate Prevention**: Each track ID saved only once
7. **Image Saving**: Full frame + crop saved with sequential naming

## 🔬 Technical Details

### Tracking Algorithm

- **ByteTrack**: Multi-object tracking with ID persistence
- **Kalman Filtering**: Smooth trajectory prediction
- **IoU Association**: Match detections to tracks
- **Lost Track Management**: Remove stale tracks after 30 frames

### Quality Metrics

- **Blur Score**: Laplacian variance (threshold: 100)
- **Brightness**: Mean pixel value (30-220 range)
- **Stationary Detection**: Total movement < 10 pixels

### Performance

- **Multi-threading**: Each camera has dedicated capture thread
- **Frame Buffering**: Minimal buffer size for low latency
- **Lock-free Reads**: Latest frame always available
- **Async Saves**: Image saving doesn't block processing

## 📈 Statistics

Access detailed statistics:
- Press `s` during runtime
- Check `vehicle_capture.log`
- View final summary on shutdown

Metrics include:
- Total vehicles detected
- Frames saved
- Rejection reasons (quality, stationary)
- Save rate percentage
- FPS per camera
- Processing latency

## 🛠️ Troubleshooting

**No frames appearing**:
- Check RTSP URLs
- Verify network connectivity
- Check camera credentials

**Low FPS**:
- Reduce resolution
- Enable GPU
- Increase `buffer_size`

**No vehicles detected**:
- Check confidence threshold
- Verify model path
- Test model separately

**ROI not working**:
- Ensure at least 3 points
- Check points are within frame
- Re-draw ROI if needed

## 🎓 Best Practices

1. **ROI Placement**: Draw tight ROIs around lanes
2. **Camera Angles**: Prefer frontal or rear views
3. **Lighting**: Ensure adequate lighting for quality
4. **Storage**: Use SSD for fast image writes
5. **Monitoring**: Check statistics regularly

## 📝 License

Professional-grade code for production use.

## 🤝 Support

For issues or questions, check logs first. Most problems are configuration-related.

---

**Built with:**
- Ultralytics YOLO
- OpenCV
- NumPy
- Python 3.8+
