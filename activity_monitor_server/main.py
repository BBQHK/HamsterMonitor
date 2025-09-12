from flask import Flask, Response, request, jsonify
import cv2
import numpy as np
from datetime import datetime
import json
import os
from ai_activity_detector import HamsterActivityDetector
from database import HamsterDatabase
import requests
from io import BytesIO
from PIL import Image
import threading
import time

# Initialize Flask app
app = Flask(__name__)

# Initialize AI activity detector
activity_detector = HamsterActivityDetector("best.pt")

# Initialize database for logging
db = HamsterDatabase("hamster_activity.db")

# Configuration
SERVER_URL = "http://192.168.50.167:8081"  # URL of hamster monitoring hardware server
PROCESSING_INTERVAL = 0.2  # Process frames every 200ms (5 FPS)

# Store latest detection result
latest_detection_result = {
    'activity': 'Unknown',
    'activity_probability': 0.0,
    'all_probabilities': {},
    'timestamp': None
}

def process_frame_from_bytes(frame_bytes):
    """Process a frame from bytes and return activity results."""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None

        # Use AI to detect activity
        activity, activity_probs = activity_detector.detect_activity(frame)
        
        # Check if all probabilities are 0.0
        if all(prob == 0.0 for prob in activity_probs.values()):
            activity = "Unknown"
            activity_probability = 0.0
        else:
            activity_probability = float(activity_probs[activity])
        
        # Log detection data to database
        try:
            # Get motion intensity from the detector
            motion_intensity = activity_detector.detect_motion(frame)
            
            # Log the activity detection
            db.log_activity(
                activity=activity,
                confidence=activity_probability,
                all_probabilities={k: float(v) for k, v in activity_probs.items()},
                motion_intensity=motion_intensity,
                frame_shape=frame.shape
            )
        except Exception as db_error:
            print(f"Database logging error: {db_error}")
            
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
    """Monitor the streaming feed from camera server and process frames."""
    global latest_detection_result
    
    while True:
        try:
            # Connect to the streaming feed
            response = requests.get(f"{SERVER_URL}/camera0", stream=True, timeout=10)
            if response.status_code == 200:
                print("Connected to camera stream")
                
                # Parse MJPEG stream
                buffer = b''
                frame_count = 0
                
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        buffer += chunk
                        
                        # Look for frame boundaries
                        while b'--frame\r\n' in buffer:
                            # Find the start of a frame
                            frame_start = buffer.find(b'--frame\r\n')
                            if frame_start == -1:
                                break
                                
                            # Find the end of this frame
                            frame_end = buffer.find(b'--frame\r\n', frame_start + 10)
                            if frame_end == -1:
                                # Need more data
                                break
                                
                            # Extract the frame data
                            frame_data = buffer[frame_start:frame_end]
                            
                            # Find the JPEG data (after headers)
                            jpeg_start = frame_data.find(b'\r\n\r\n')
                            if jpeg_start != -1:
                                jpeg_data = frame_data[jpeg_start + 4:]
                                
                                # Process every few frames to avoid overwhelming the system
                                if frame_count % 3 == 0:  # Process every 3rd frame
                                    result = process_frame_from_bytes(jpeg_data)
                                    if result:
                                        latest_detection_result.update(result)
                                
                                frame_count += 1
                            
                            # Remove processed frame from buffer
                            buffer = buffer[frame_end:]
                            
        except requests.RequestException as e:
            print(f"Error monitoring camera stream: {e}")
            time.sleep(1)  # Wait before retrying
        except Exception as e:
            print(f"Error in stream monitoring: {e}")
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
