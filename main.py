from flask import Flask, Response, request, jsonify
import cv2
import random
import time
import numpy as np
from datetime import datetime
import json
import os
from ai_activity_detector import HamsterActivityDetector

# Constants
CAMERA_INDEX_1 = 0  # First camera
# CAMERA_INDEX_2 = 1  # Second camera
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

# Initialize Flask app
app = Flask(__name__)

# Initialize AI activity detector
activity_detector = HamsterActivityDetector()

# Global settings
show_temp_hum = True

def list_available_cameras():
    """List all available cameras on the system."""
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def setup_camera(camera_index):
    """Setup camera with proper configuration for Windows."""
    # Try to open the camera
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use DirectShow for Windows
    
    if not camera.isOpened():
        # Try without DirectShow if it fails
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            raise RuntimeError(f"Failed to open camera with index {camera_index}")
    
    # Set camera properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)
    
    # Try to set auto exposure and focus
    try:
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode
        camera.set(cv2.CAP_PROP_EXPOSURE, -4)  # Adjust exposure (value depends on camera)
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 = manual focus
    except:
        print("Warning: Could not set camera properties. Some features may not work.")
    
    return camera

# Initialize cameras
try:
    available_cameras = list_available_cameras()
    print(f"Available cameras: {available_cameras}")
    
    if not available_cameras:
        raise RuntimeError("No cameras found!")
    
    camera1 = setup_camera(available_cameras[0])
    print(f"Successfully initialized camera {available_cameras[0]}")
    
    # Uncomment if you want to use a second camera
    # if len(available_cameras) > 1:
    #     camera2 = setup_camera(available_cameras[1])
    #     print(f"Successfully initialized camera {available_cameras[1]}")
    # else:
    #     camera2 = None
    #     print("No second camera found")
    
except Exception as e:
    print(f"Error initializing cameras: {e}")
    camera1 = None
    # camera2 = None

def get_simulated_readings():
    """Generate simulated temperature and humidity readings."""
    temperature = 22.5
    humidity = 40.2
    return temperature, humidity

def get_current_timestamp():
    """Get current timestamp in formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

def generate_camera_frames(camera, show_config=False):
    """Generate video frames from a camera with sensor data overlay."""
    if camera is None:
        # Generate a black frame with error message
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera not available", (50, FRAME_HEIGHT//2), 
                   FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return

    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera")
            break

        # Get sensor readings and detect activity
        temperature, humidity = get_simulated_readings()
        current_time = get_current_timestamp()
        
        # Use AI to detect activity
        activity, activity_probs = activity_detector.detect_activity(frame)
        
        # Add AI activity overlay
        frame = activity_detector.draw_activity_overlay(frame, activity, activity_probs)
        
        # Prepare text for overlay
        texts = [f"Time: {current_time}"]
        
        # Add temperature and humidity if enabled
        if show_temp_hum:
            texts.append(f"Temp: {temperature:.1f}C  Hum: {humidity:.1f}%")
        
        # Add text overlay
        add_text_overlay(frame, texts)

        # Encode frame as JPEG for MJPEG streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/camera1')
def camera1_feed():
    """Stream video feed from camera 1 with sensor data overlay."""
    return Response(generate_camera_frames(camera1), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/camera2')
# def camera2_feed():
#     """Stream video feed from camera 2 with sensor data overlay."""
#     return Response(generate_camera_frames(camera2), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    """Handle configuration updates."""
    global show_temp_hum
    if request.method == 'POST':
        new_config = request.json
        show_temp_hum = new_config.get('SHOW_TEMP_HUM', show_temp_hum)
        return jsonify({"status": "success"})
    return jsonify({"SHOW_TEMP_HUM": show_temp_hum})

@app.route('/activity_pattern')
def get_activity_pattern():
    """Get the current activity pattern analysis."""
    pattern = activity_detector.get_activity_pattern()
    return jsonify(pattern)

@app.route('/')
def index():
    """Serve a simple HTML page with camera feed and configuration interface."""
    return """
    <html>
        <head>
            <title>Hamster Monitor</title>
            <style>
                body { margin: 0; padding: 20px; background: #333; color: white; }
                .container { max-width: 1200px; margin: 0 auto; }
                .camera-feed { margin-bottom: 20px; }
                img { width: 100%; height: auto; }
                .controls { margin: 20px 0; }
                button { padding: 10px 20px; margin: 0 10px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
                button:hover { background: #45a049; }
                .status-message { margin: 10px 0; padding: 10px; border-radius: 4px; }
                .success { background: #4CAF50; }
                .error { background: #f44336; }
                .activity-chart { margin-top: 20px; padding: 10px; background: #444; border-radius: 5px; }
                .activity-bar { height: 20px; margin: 5px 0; background: #666; border-radius: 3px; }
                .activity-bar-fill { height: 100%; background: #4CAF50; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="controls">
                    <button onclick="toggleTempHum()" id="temp-hum-btn">Hide Temperature/Humidity</button>
                </div>
                <div class="camera-feed">
                    <h2>Camera Feed</h2>
                    <img src="/camera1" id="camera1" />
                    <div class="activity-chart">
                        <h3>Activity Analysis</h3>
                        <div id="activity-bars"></div>
                    </div>
                </div>
            </div>
            <script>
                let showTempHum = true;
                
                function showStatus(message, isError = false) {
                    const statusDiv = document.createElement('div');
                    statusDiv.className = `status-message ${isError ? 'error' : 'success'}`;
                    statusDiv.textContent = message;
                    document.body.insertBefore(statusDiv, document.querySelector('.container'));
                    setTimeout(() => {
                        statusDiv.remove();
                    }, 3000);
                }
                
                function toggleTempHum() {
                    showTempHum = !showTempHum;
                    const button = document.getElementById('temp-hum-btn');
                    button.textContent = showTempHum ? 'Hide Temperature/Humidity' : 'Show Temperature/Humidity';
                    
                    // Update the configuration
                    fetch('/config', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ SHOW_TEMP_HUM: showTempHum })
                    })
                    .then(response => response.json())
                    .then(data => {
                        showStatus('Configuration saved successfully!');
                    })
                    .catch(error => {
                        showStatus('Error saving configuration: ' + error, true);
                    });
                }
                
                // Load initial configuration
                fetch('/config')
                    .then(response => response.json())
                    .then(config => {
                        showTempHum = config.SHOW_TEMP_HUM;
                        document.getElementById('temp-hum-btn').textContent = 
                            showTempHum ? 'Hide Temperature/Humidity' : 'Show Temperature/Humidity';
                    });
                
                // Update activity chart
                function updateActivityChart() {
                    fetch('/activity_pattern')
                        .then(response => response.json())
                        .then(data => {
                            const container = document.getElementById('activity-bars');
                            container.innerHTML = '';
                            
                            for (const [activity, probability] of Object.entries(data)) {
                                const bar = document.createElement('div');
                                bar.className = 'activity-bar';
                                
                                const fill = document.createElement('div');
                                fill.className = 'activity-bar-fill';
                                fill.style.width = `${probability * 100}%`;
                                
                                const label = document.createElement('span');
                                label.textContent = `${activity}: ${(probability * 100).toFixed(1)}%`;
                                label.style.marginLeft = '10px';
                                
                                bar.appendChild(fill);
                                bar.appendChild(label);
                                container.appendChild(bar);
                            }
                        });
                }
                
                // Update activity chart every 2 seconds
                setInterval(updateActivityChart, 2000);
                
                // Initial update
                updateActivityChart();
            </script>
        </body>
    </html>
    """

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8081, threaded=True)
    finally:
        # Release camera resources when the application stops
        if camera1:
            camera1.release()
        # if camera2:
        #     camera2.release()
