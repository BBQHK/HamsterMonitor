from flask import Flask, Response
import cv2
import requests
import json
import numpy as np
from datetime import datetime

# Constants
CAMERA_INDICES = [0, 2]  # List of camera indices to use
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
MAIN_API_URL = "http://192.168.50.168:8081/process_frame"  # URL of main.py API
FRAME_SKIP = 3  # Process every 3rd frame

# Text overlay constants
FONT_SCALE = 0.5
FONT_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
BACKGROUND_ALPHA = 0.5
TEXT_COLOR = (255, 255, 255)  # White
BACKGROUND_COLOR = (0, 0, 0)  # Black
TEXT_PADDING = 5

# Initialize Flask app
app = Flask(__name__)

# Dictionary to store camera objects
cameras = {}

# Store last activity result (shared across all cameras)
last_activity_result = {
    'activity': 'Unknown',
    'activity_probability': 0.0
}

# Global server status
server_down = False

def get_current_timestamp():
    """Get current timestamp in formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_simulated_readings():
    """Generate simulated temperature and humidity readings."""
    temperature = 22.5
    humidity = 40.2
    return temperature, humidity

def add_text_overlay(frame, texts):
    """Add text overlay to a frame.
    
    Args:
        frame: The frame to add text overlay to
        texts: Array of text strings to display
    """
    if not texts:
        return
        
    # Calculate total height needed and max width
    total_height = 0
    max_width = 0
    text_sizes = []
    
    for text in texts:
        (width, height), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        text_sizes.append((width, height))
        max_width = max(max_width, width)
        total_height += height
    
    # Add padding between texts and around the box
    padding = TEXT_PADDING
    total_height += padding * (len(texts) + 1)  # Padding between texts and around the box
    max_width += padding * 2  # Padding on both sides
    
    # Add semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (padding, padding),
        (padding + max_width, padding + total_height),
        BACKGROUND_COLOR,
        -1
    )
    cv2.addWeighted(overlay, BACKGROUND_ALPHA, frame, 1 - BACKGROUND_ALPHA, 0, frame)
    
    # Add text
    y = padding + text_sizes[0][1] + padding
    for i, text in enumerate(texts):
        cv2.putText(
            frame, 
            text, 
            (padding + 5, y), 
            FONT, 
            FONT_SCALE, 
            TEXT_COLOR, 
            FONT_THICKNESS
        )
        if i < len(texts) - 1:
            y += text_sizes[i + 1][1] + padding

def setup_camera(camera_index):
    """Setup a camera with specified index."""
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)
    return camera

def initialize_cameras():
    """Initialize all cameras at startup."""
    for camera_index in CAMERA_INDICES:
        camera = setup_camera(camera_index)
        if camera.isOpened():
            cameras[camera_index] = camera
            print(f"Successfully initialized camera {camera_index}")
        else:
            print(f"Failed to initialize camera {camera_index}")

def get_camera(camera_index):
    """Get a camera object for the given index."""
    return cameras.get(camera_index)

def generate_frames(camera_index):
    """Generate video frames from specified camera."""
    global server_down
    camera = get_camera(camera_index)
    if camera is None:
        return
    
    frame_count = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            break

        try:
            # Get local readings
            current_time = get_current_timestamp()
            temperature, humidity = get_simulated_readings()
            
            # Only process frames from camera0
            if camera_index == 0:
                # Process frame only every FRAME_SKIP frames
                if frame_count % FRAME_SKIP == 0:
                    try:
                        # Encode frame as JPEG
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_bytes = buffer.tobytes()
                        
                        # Send frame to main.py for processing
                        response = requests.post(MAIN_API_URL, data=frame_bytes)
                        if response.status_code == 200:
                            # Update the shared activity result
                            last_activity_result.update(response.json())
                            server_down = False
                    except requests.exceptions.RequestException:
                        server_down = True
            
            # Use the shared activity result for overlay
            texts = [
                f"Time: {current_time}",
                f"Temp: {temperature:.1f}C  Hum: {humidity:.1f}%"
            ]
            
            if server_down:
                texts.append("Activity: Server Down")
            else:
                texts.append(f"Activity: {last_activity_result['activity']} ({last_activity_result['activity_probability']*100:.1f}%)")
            
            # Add text overlay to frame
            add_text_overlay(frame, texts)
            
            frame_count += 1
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Add error message to frame
            cv2.putText(frame, f"Error: {str(e)}", (50, FRAME_HEIGHT//2), 
                       FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

        # Encode processed frame as JPEG for MJPEG streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/camera<int:camera_index>')
def camera_feed(camera_index):
    """Stream video feed from specified camera index."""
    if camera_index not in CAMERA_INDICES:
        return "Invalid camera index", 400
    return Response(generate_frames(camera_index), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Serve a simple HTML page with camera feeds."""
    camera_feed_html = ""
    for camera_index in CAMERA_INDICES:
        camera_feed_html += f"""
                <div class="camera-feed">
                    <h3>/camera{camera_index}</h3>
                    <img src="/camera{camera_index}" />
                </div>
        """

    return f"""
    <html>
        <head>
            <title>Multi-Camera Feed</title>
            <style>
                body {{ margin: 0; padding: 20px; background: #333; color: white; }}
                .camera-grid {{ 
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                    max-width: 1600px;
                    margin: 0 auto;
                }}
                .camera-feed {{
                    background: #444;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .camera-feed h3 {{
                    margin: 0 0 10px 0;
                    color: #4CAF50;
                }}
                img {{ width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Multi-Camera Feed</h1>
            <div class="camera-grid">
                {camera_feed_html}
            </div>
        </body>
    </html>
    """

if __name__ == '__main__':
    try:
        # Initialize all cameras before starting the server
        initialize_cameras()
        app.run(host='0.0.0.0', port=8081, threaded=True)
    finally:
        # Release all camera resources when the application stops
        for camera in cameras.values():
            camera.release()
