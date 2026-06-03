from flask import Flask, jsonify
import cv2
import numpy as np
from datetime import datetime
import os
from ai_activity_detector import HamsterActivityDetector
import threading
import time
import subprocess
import shutil

# Initialize Flask app
app = Flask(__name__)

# Initialize AI activity detector
activity_detector = HamsterActivityDetector("best.pt")

# Configuration
SERVER_URL = os.getenv("HARDWARE_SERVER_URL", "http://192.168.50.167:8081")  # URL of hamster monitoring hardware server
PROCESSING_INTERVAL = float(os.getenv("PROCESSING_INTERVAL", "0.2"))  # Process frames every 200ms (5 FPS)
STREAM_FRAME_WIDTH = int(os.getenv("STREAM_FRAME_WIDTH", "640"))
STREAM_FRAME_HEIGHT = int(os.getenv("STREAM_FRAME_HEIGHT", "480"))
FRAME_BYTES = STREAM_FRAME_WIDTH * STREAM_FRAME_HEIGHT * 3

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

def _drain_stderr(proc, stop_event):
    try:
        for line in iter(proc.stderr.readline, b''):
            if stop_event.is_set():
                break
            text = line.decode(errors='replace').strip()
            if text:
                print(f"ffmpeg decode: {text}")
    except Exception as e:
        print(f"ffmpeg stderr reader error: {e}")

def monitor_camera_stream():
    """Decode the hardware server's H.264 MP4 stream with ffmpeg."""
    global latest_detection_result

    stream_url = f"{SERVER_URL}/camera4"
    last_process_time = 0.0
    decode_cmd = [
        'ffmpeg',
        '-nostdin',
        '-loglevel', 'warning',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-i', stream_url,
        '-an',
        '-sn',
        '-dn',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        'pipe:1',
    ]

    while True:
        if shutil.which('ffmpeg') is None:
            print("ffmpeg is required to decode the H.264 camera stream")
            time.sleep(5)
            continue

        stop_event = threading.Event()
        proc = subprocess.Popen(
            decode_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        threading.Thread(
            target=_drain_stderr, args=(proc, stop_event), daemon=True
        ).start()

        print(f"Connected to H.264 camera stream: {stream_url}")
        try:
            while True:
                raw = proc.stdout.read(FRAME_BYTES)
                if len(raw) != FRAME_BYTES:
                    if proc.poll() is not None:
                        print(f"ffmpeg decode exited (code {proc.returncode})")
                    else:
                        print("Incomplete frame from H.264 stream")
                    break

                now = time.time()
                if now - last_process_time < PROCESSING_INTERVAL:
                    continue
                last_process_time = now

                frame = np.frombuffer(raw, np.uint8).reshape(
                    (STREAM_FRAME_HEIGHT, STREAM_FRAME_WIDTH, 3)
                )
                result = process_frame(frame)
                if result:
                    latest_detection_result.update(result)
        except Exception as e:
            print(f"Error in stream monitoring: {e}")
        finally:
            stop_event.set()
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
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
