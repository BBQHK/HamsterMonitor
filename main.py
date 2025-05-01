from flask import Flask, Response
import cv2
import random
import time
import numpy as np
from datetime import datetime

# Constants
CAMERA_INDEX_1 = 0  # First camera
CAMERA_INDEX_2 = 1  # Second camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15
FONT_SCALE = 0.5
FONT_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
BACKGROUND_ALPHA = 0.5
TEXT_COLOR = (255, 255, 255)  # White
BACKGROUND_COLOR = (0, 0, 0)  # Black
TEXT_PADDING = 5

# Activity detection constants
MOVEMENT_THRESHOLD = 1000  # Minimum pixels that need to change to consider movement
WHEEL_DETECTION_AREA = (200, 200, 440, 280)  # (x1, y1, x2, y2) - area where the wheel is
FOOD_AREA = (50, 300, 150, 400)  # Area where food is placed
WATER_AREA = (450, 300, 550, 400)  # Area where water is placed
SLEEPING_THRESHOLD = 5  # Frames with minimal movement to consider sleeping

# Initialize Flask app
app = Flask(__name__)

# Camera setup
def setup_camera(camera_index):
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)
    return camera

# Initialize both cameras
camera1 = setup_camera(CAMERA_INDEX_1)
camera2 = setup_camera(CAMERA_INDEX_2)

# Initialize background subtractors
bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)
bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)

def get_simulated_readings():
    """Generate simulated temperature and humidity readings."""
    temperature = random.uniform(20.0, 30.0)
    humidity = random.uniform(40.0, 60.0)
    return temperature, humidity

def detect_hamster_activity(frame, bg_subtractor, prev_activity, no_movement_frames):
    """Detect hamster activity based on movement patterns."""
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate total movement
    movement = cv2.countNonZero(thresh)
    
    # Check if hamster is in wheel area
    wheel_roi = thresh[WHEEL_DETECTION_AREA[1]:WHEEL_DETECTION_AREA[3], WHEEL_DETECTION_AREA[0]:WHEEL_DETECTION_AREA[2]]
    wheel_movement = cv2.countNonZero(wheel_roi)
    
    # Check if hamster is in food area
    food_roi = thresh[FOOD_AREA[1]:FOOD_AREA[3], FOOD_AREA[0]:FOOD_AREA[2]]
    food_movement = cv2.countNonZero(food_roi)
    
    # Check if hamster is in water area
    water_roi = thresh[WATER_AREA[1]:WATER_AREA[3], WATER_AREA[0]:WATER_AREA[2]]
    water_movement = cv2.countNonZero(water_roi)
    
    # Update no movement frames counter
    if movement < MOVEMENT_THRESHOLD:
        no_movement_frames += 1
    else:
        no_movement_frames = 0
    
    # Determine activity based on movement patterns
    if no_movement_frames >= SLEEPING_THRESHOLD:
        return "Sleeping", no_movement_frames
    elif wheel_movement > MOVEMENT_THRESHOLD * 0.5:
        return "Running on wheel", no_movement_frames
    elif food_movement > MOVEMENT_THRESHOLD * 0.3:
        return "Eating", no_movement_frames
    elif water_movement > MOVEMENT_THRESHOLD * 0.3:
        return "Drinking water", no_movement_frames
    elif movement > MOVEMENT_THRESHOLD:
        return "Exploring", no_movement_frames
    else:
        return prev_activity, no_movement_frames

def get_current_timestamp():
    """Get current timestamp in formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_text_overlay(frame, text1, text2, text3):
    """Add text overlay to a frame."""
    # Get text size
    (text1_width, text1_height), _ = cv2.getTextSize(text1, FONT, FONT_SCALE, FONT_THICKNESS)
    (text2_width, text2_height), _ = cv2.getTextSize(text2, FONT, FONT_SCALE, FONT_THICKNESS)
    (text3_width, text3_height), _ = cv2.getTextSize(text3, FONT, FONT_SCALE, FONT_THICKNESS)
    
    # Add semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (TEXT_PADDING, TEXT_PADDING),
        (TEXT_PADDING + max(text1_width, text2_width, text3_width) + TEXT_PADDING * 2, TEXT_PADDING + text1_height + text2_height + text3_height + TEXT_PADDING * 4),
        BACKGROUND_COLOR,
        -1
    )
    cv2.addWeighted(overlay, BACKGROUND_ALPHA, frame, 1 - BACKGROUND_ALPHA, 0, frame)
    
    # Add text
    cv2.putText(frame, text1, (TEXT_PADDING + 5, TEXT_PADDING + text1_height + 5), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
    cv2.putText(frame, text2, (TEXT_PADDING + 5, TEXT_PADDING + text1_height + text2_height + TEXT_PADDING * 2), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
    cv2.putText(frame, text3, (TEXT_PADDING + 5, TEXT_PADDING + text1_height + text2_height + text3_height + TEXT_PADDING * 3), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

def generate_camera_frames(camera, bg_subtractor):
    """Generate video frames from a camera with sensor data overlay."""
    prev_activity = "Exploring"
    no_movement_frames = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Get sensor readings and detect activity
        temperature, humidity = get_simulated_readings()
        current_time = get_current_timestamp()
        activity, no_movement_frames = detect_hamster_activity(frame, bg_subtractor, prev_activity, no_movement_frames)
        prev_activity = activity
        
        # Prepare text for overlay
        text1 = f"Time: {current_time}"
        text2 = f"Temp: {temperature:.1f}C  Hum: {humidity:.1f}%"
        text3 = f"Activity: {activity}"
        
        # Add text overlay
        add_text_overlay(frame, text1, text2, text3)

        # Encode frame as JPEG for MJPEG streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/camera1')
def camera1_feed():
    """Stream video feed from camera 1 with sensor data overlay."""
    return Response(generate_camera_frames(camera1, bg_subtractor1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera2')
def camera2_feed():
    """Stream video feed from camera 2 with sensor data overlay."""
    return Response(generate_camera_frames(camera2, bg_subtractor2), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Serve a simple HTML page with both camera feeds."""
    return """
    <html>
        <head>
            <title>Dual Camera Feed</title>
            <style>
                body { margin: 0; padding: 20px; background: #333; }
                .container { display: flex; gap: 20px; }
                .camera-feed { flex: 1; }
                img { width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="camera-feed">
                    <h2 style="color: white;">Camera 1</h2>
                    <img src="/camera1" />
                </div>
                <div class="camera-feed">
                    <h2 style="color: white;">Camera 2</h2>
                    <img src="/camera2" />
                </div>
            </div>
        </body>
    </html>
    """

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8081, threaded=True)
    finally:
        # Release camera resources when the application stops
        camera1.release()
        camera2.release()
