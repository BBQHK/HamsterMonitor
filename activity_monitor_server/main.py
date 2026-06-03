from flask import Flask, jsonify
import cv2
import numpy as np
from datetime import datetime
import os
from ai_activity_detector import HamsterActivityDetector
import threading
import time

# Initialize Flask app
app = Flask(__name__)

# Initialize AI activity detector
activity_detector = HamsterActivityDetector("best.pt")

# Configuration
SERVER_URL = os.getenv("HARDWARE_SERVER_URL", "http://192.168.50.167:8081")  # URL of hamster monitoring hardware server
PROCESSING_INTERVAL = float(os.getenv("PROCESSING_INTERVAL", "0.2"))  # Process frames every 200ms (5 FPS)

# Store latest detection result
latest_detection_result = {
    'activity': 'Unknown',
    'activity_probability': 0.0,
    'all_probabilities': {},
    'timestamp': None
}

def process_frame(frame):
    """Process a BGR frame and return activity results."""
    try:
        if frame is None:
            return None

        activity, activity_probs = activity_detector.detect_activity(frame)
        
        # Check if all probabilities are 0.0
        if all(prob == 0.0 for prob in activity_probs.values()):
            activity = "Unknown"
            activity_probability = 0.0
        else:
            activity_probability = float(activity_probs[activity])
        
        # Prepare result
        result = {
            "activity": activity,
            "activity_probability": activity_probability,
            "all_probabilities": {k: float(v) for k, v in activity_probs.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def monitor_camera_stream():
    """Monitor the H.264 MPEG-TS feed from the hardware server."""
    global latest_detection_result

    stream_url = f"{SERVER_URL}/camera4"
    last_process_time = 0.0

    while True:
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"Failed to open H.264 stream: {stream_url}")
            time.sleep(1)
            continue

        print("Connected to H.264 camera stream")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Lost camera stream, reconnecting...")
                    break

                now = time.time()
                if now - last_process_time < PROCESSING_INTERVAL:
                    continue
                last_process_time = now

                result = process_frame(frame)
                if result:
                    latest_detection_result.update(result)
        except Exception as e:
            print(f"Error in stream monitoring: {e}")
        finally:
            cap.release()
            time.sleep(1)

@app.route('/detection_result')
def get_detection_result():
    """API endpoint to get the latest detection result."""
    return jsonify(latest_detection_result)


if __name__ == '__main__':
    # Start the camera stream monitoring thread
    stream_monitoring_thread = threading.Thread(target=monitor_camera_stream, daemon=True)
    stream_monitoring_thread.start()
    
    app.run(host='0.0.0.0', port=8082, threaded=True)
