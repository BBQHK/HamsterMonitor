from flask import Flask, Response
import cv2
import random
import time
import numpy as np
from datetime import datetime

# Constants
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15
FONT_SCALE = 0.5
FONT_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
BACKGROUND_ALPHA = 0.5
TEXT_COLOR = (255, 255, 255)  # White
BACKGROUND_COLOR = (0, 0, 0)  # Black
TEXT_PADDING = 10
TEXT_START_X = 10
TEXT_START_Y = 20
LINE_SPACING = 20

# Initialize Flask app
app = Flask(__name__)

# Camera setup
def setup_camera():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)
    return camera

camera = setup_camera()

def get_simulated_readings():
    """Generate simulated temperature and humidity readings."""
    temperature = random.uniform(20.0, 30.0)
    humidity = random.uniform(40.0, 60.0)
    return temperature, humidity

def get_current_timestamp():
    """Get current timestamp in formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_text_overlay(frame, texts):
    """Add text overlay with semi-transparent background to the frame."""
    # Calculate maximum text width and total height
    max_width = 0
    total_height = 0
    text_sizes = []
    
    for text in texts:
        (width, height), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        text_sizes.append((width, height))
        max_width = max(max_width, width)
        total_height += height + LINE_SPACING
    
    # Add semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (TEXT_START_X - TEXT_PADDING, TEXT_START_Y - TEXT_PADDING),
        (TEXT_START_X + max_width + TEXT_PADDING, TEXT_START_Y + total_height),
        BACKGROUND_COLOR,
        -1
    )
    cv2.addWeighted(overlay, BACKGROUND_ALPHA, frame, 1 - BACKGROUND_ALPHA, 0, frame)
    
    # Add text
    y = TEXT_START_Y
    for i, text in enumerate(texts):
        cv2.putText(frame, text, (TEXT_START_X, y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        y += text_sizes[i][1] + LINE_SPACING

def generate_frames():
    """Generate video frames with sensor data overlay."""
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Get sensor readings
        temperature, humidity = get_simulated_readings()
        current_time = get_current_timestamp()
        
        # Prepare text for overlay
        texts = [
            f"Time: {current_time}",
            f"Temp: {temperature:.1f}C  Hum: {humidity:.1f}%"
        ]
        
        # Add text overlay to frame
        add_text_overlay(frame, texts)

        # Encode frame as JPEG for MJPEG streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Stream video feed with sensor data overlay."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8081, threaded=True)
    finally:
        # Release camera resources when the application stops
        camera.release()
