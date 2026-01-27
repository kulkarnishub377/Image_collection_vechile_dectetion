"""
Stream Quality Test - Check your RTSP streams before running main system
"""

import cv2
import numpy as np
import time

# Your streams
STREAMS = {
    "overview": "rtsp://admin:Arya@123@125.18.39.10:5554/cam/realmonitor?channel=1&subtype=0",
    "anpr": "rtsp://admin:BE04_ViDeS@125.18.39.10:5555/Streaming/Channels/101",
    "ptz": "rtsp://admin:Arya_123@125.18.39.10:554/cam/realmonitor?channel=1&subtype=0"
}

def test_stream(name, url):
    """Test single stream quality"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    try:
        # Try TCP transport first (more reliable)
        rtsp_url = url + ("?tcp" if "?" not in url else "&tcp")
        print(f"Connecting with TCP transport...")
        
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            print(f"❌ Failed to connect!")
            return False
        
        # Set quality parameters
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Get actual settings
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        print(f"✓ Connected!")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Codec: {codec}")
        
        # Read test frames
        print(f"\nReading test frames...")
        frame_count = 0
        start_time = time.time()
        
        for i in range(30):
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                if i == 0:
                    # Save first frame
                    cv2.imwrite(f"test_{name}.jpg", frame, 
                               [cv2.IMWRITE_JPEG_QUALITY, 100])
                    print(f"  Saved test frame: test_{name}.jpg")
            else:
                print(f"  ⚠ Frame {i+1} failed")
        
        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n✓ Test complete!")
        print(f"  Frames read: {frame_count}/30")
        print(f"  Actual FPS: {actual_fps:.1f}")
        print(f"  Frame size: {frame.shape if frame_count > 0 else 'N/A'}")
        
        cap.release()
        
        if frame_count < 20:
            print(f"⚠ WARNING: Low frame success rate!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("="*60)
    print("RTSP STREAM QUALITY TEST")
    print("="*60)
    print("\nThis will test each camera stream for:")
    print("  - Connection stability")
    print("  - Resolution quality")
    print("  - Frame rate")
    print("  - Codec support")
    
    results = {}
    
    for name, url in STREAMS.items():
        results[name] = test_stream(name, url)
        time.sleep(1)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{name:15} {status}")
    
    if all(results.values()):
        print("\n✅ All streams OK! Ready to run main system.")
    else:
        print("\n⚠ Some streams failed. Check network/credentials.")
    
    print("\nTest images saved as: test_<camera>.jpg")
    print("Check these images for quality verification.")

if __name__ == "__main__":
    main()
