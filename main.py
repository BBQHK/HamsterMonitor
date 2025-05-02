from flask import Flask, Response, request, jsonify
import cv2
import random
import time
import numpy as np
from datetime import datetime
import json
import os

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

# Configuration file path
CONFIG_FILE = 'activity_areas.json'

# Default activity detection constants
DEFAULT_CONFIG = {
    'MOVEMENT_THRESHOLD': 1000,
    'SLEEPING_THRESHOLD': 5,
    'ACTIVITY_DETECTION_ENABLED': True,
    'WHEEL_AREA': {'x1': 200, 'y1': 200, 'x2': 440, 'y2': 280},
    'FOOD_AREA': {'x1': 50, 'y1': 300, 'x2': 150, 'y2': 400},
    'WATER_AREA': {'x1': 450, 'y1': 300, 'x2': 550, 'y2': 400}
}

# Load or create configuration
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_CONFIG

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# Initialize configuration
config = load_config()

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
# camera2 = setup_camera(CAMERA_INDEX_2)

# Initialize background subtractors
bg_subtractor1 = cv2.createBackgroundSubtractorKNN(history=500, detectShadows=False, dist2Threshold=400.0)
# bg_subtractor2 = cv2.createBackgroundSubtractorKNN(history=500, detectShadows=False, dist2Threshold=400.0)

# Store previous frames for frame differencing
prev_frame1 = None
# prev_frame2 = None

def get_simulated_readings():
    """Generate simulated temperature and humidity readings."""
    temperature = 22.5
    humidity = 40.2
    return temperature, humidity

def detect_lighting_condition(frame):
    """Detect if the frame is in day or night vision mode."""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Calculate average brightness
    avg_brightness = np.mean(gray)
    
    # If average brightness is below 50, consider it night vision
    return avg_brightness < 50

def detect_hamster_activity(frame, bg_subtractor, prev_activity, no_movement_frames, prev_frame=None):
    """Detect hamster activity based on movement patterns."""
    if not config['ACTIVITY_DETECTION_ENABLED']:
        return "Activity detection disabled", no_movement_frames, frame
        
    # Convert to grayscale if not already
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Detect lighting condition
    is_night_vision = detect_lighting_condition(frame)
    print(is_night_vision)
    
    # Adjust parameters based on lighting condition
    if is_night_vision:
        # More sensitive settings for night vision
        bg_subtractor.setDist2Threshold(400.0)  # Higher threshold for night vision
        frame_diff_threshold = 25
        morph_kernel_size = 3
    else:
        # Less sensitive settings for daylight
        bg_subtractor.setDist2Threshold(600.0)  # Lower threshold for daylight
        frame_diff_threshold = 30
        morph_kernel_size = 5
    
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(gray)
    
    # If the mask is mostly empty, try frame differencing as fallback
    if cv2.countNonZero(fg_mask) < 100 and prev_frame is not None:
        # Calculate absolute difference between frames
        frame_diff = cv2.absdiff(gray, prev_frame)
        # Apply threshold to get binary image
        _, fg_mask = cv2.threshold(frame_diff, frame_diff_threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate total movement
    movement = cv2.countNonZero(thresh)
    
    # Check if hamster is in wheel area
    wheel_area = config['WHEEL_AREA']
    wheel_roi = thresh[wheel_area['y1']:wheel_area['y2'], wheel_area['x1']:wheel_area['x2']]
    wheel_movement = cv2.countNonZero(wheel_roi)
    
    # Check if hamster is in food area
    food_area = config['FOOD_AREA']
    food_roi = thresh[food_area['y1']:food_area['y2'], food_area['x1']:food_area['x2']]
    food_movement = cv2.countNonZero(food_roi)
    
    # Check if hamster is in water area
    water_area = config['WATER_AREA']
    water_roi = thresh[water_area['y1']:water_area['y2'], water_area['x1']:water_area['x2']]
    water_movement = cv2.countNonZero(water_roi)
    
    # Update no movement frames counter
    if movement < config['MOVEMENT_THRESHOLD']:
        no_movement_frames += 1
    else:
        no_movement_frames = 0
    
    # Determine activity based on movement patterns
    if no_movement_frames >= config['SLEEPING_THRESHOLD']:
        return "Sleeping", no_movement_frames, gray
    elif wheel_movement > config['MOVEMENT_THRESHOLD'] * 0.5:
        return "Running on wheel", no_movement_frames, gray
    elif food_movement > config['MOVEMENT_THRESHOLD'] * 0.3:
        return "Eating", no_movement_frames, gray
    elif water_movement > config['MOVEMENT_THRESHOLD'] * 0.3:
        return "Drinking water", no_movement_frames, gray
    elif movement > config['MOVEMENT_THRESHOLD']:
        return "Exploring", no_movement_frames, gray
    else:
        return prev_activity, no_movement_frames, gray

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

def draw_config_areas(frame):
    """Draw the configured areas on the frame."""
    # Draw wheel area
    wheel = config['WHEEL_AREA']
    cv2.rectangle(frame, (wheel['x1'], wheel['y1']), (wheel['x2'], wheel['y2']), (0, 255, 0), 2)
    cv2.putText(frame, "Wheel", (wheel['x1'], wheel['y1'] - 10), FONT, FONT_SCALE, (0, 255, 0), FONT_THICKNESS)
    
    # Draw food area
    food = config['FOOD_AREA']
    cv2.rectangle(frame, (food['x1'], food['y1']), (food['x2'], food['y2']), (255, 0, 0), 2)
    cv2.putText(frame, "Food", (food['x1'], food['y1'] - 10), FONT, FONT_SCALE, (255, 0, 0), FONT_THICKNESS)
    
    # Draw water area
    water = config['WATER_AREA']
    cv2.rectangle(frame, (water['x1'], water['y1']), (water['x2'], water['y2']), (0, 0, 255), 2)
    cv2.putText(frame, "Water", (water['x1'], water['y1'] - 10), FONT, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)

def generate_camera_frames(camera, bg_subtractor, show_config=False):
    """Generate video frames from a camera with sensor data overlay."""
    prev_activity = "Exploring"
    no_movement_frames = 0
    prev_frame = None
    
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Get sensor readings and detect activity
        temperature, humidity = get_simulated_readings()
        current_time = get_current_timestamp()
        activity, no_movement_frames, prev_frame = detect_hamster_activity(frame, bg_subtractor, prev_activity, no_movement_frames, prev_frame)
        prev_activity = activity
        
        # Draw configuration areas if in config mode
        if show_config:
            draw_config_areas(frame)
        
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
    show_config = request.args.get('config', 'false').lower() == 'true'
    return Response(generate_camera_frames(camera1, bg_subtractor1, show_config), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/camera2')
# def camera2_feed():
#     """Stream video feed from camera 2 with sensor data overlay."""
#     show_config = request.args.get('config', 'false').lower() == 'true'
#     return Response(generate_camera_frames(camera2, bg_subtractor2, show_config), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    """Handle configuration updates."""
    if request.method == 'POST':
        new_config = request.json
        global config
        config = new_config
        save_config(config)
        return jsonify({"status": "success"})
    return jsonify(config)

@app.route('/')
def index():
    """Serve a simple HTML page with camera feed and configuration interface."""
    return """
    <html>
        <head>
            <title>Hamster Monitor</title>
            <style>
                body { margin: 0; padding: 20px; background: #333; color: white; }
                .container { display: flex; gap: 20px; }
                .camera-feed { flex: 2; }
                .config-panel { flex: 1; background: #444; padding: 20px; border-radius: 5px; }
                img { width: 100%; height: auto; }
                .controls { margin: 20px 0; }
                button { padding: 10px 20px; margin: 0 10px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
                button:hover { background: #45a049; }
                .config-section { margin-bottom: 20px; }
                .config-section h4 { margin: 10px 0; color: #4CAF50; }
                .config-input { margin: 5px 0; }
                .config-input label { display: inline-block; width: 150px; }
                .config-input input { width: 80px; padding: 5px; margin-right: 10px; }
                .area-box { border: 1px solid #666; padding: 10px; margin: 10px 0; background: #555; }
                .status-message { margin: 10px 0; padding: 10px; border-radius: 4px; }
                .success { background: #4CAF50; }
                .error { background: #f44336; }
            </style>
        </head>
        <body>
            <div class="controls">
                <button onclick="toggleConfig()">Toggle Configuration Mode</button>
                <button onclick="saveConfig()">Save Configuration</button>
                <button onclick="toggleActivityDetection()" id="activity-detection-btn">Disable Activity Detection</button>
            </div>
            <div class="container">
                <div class="camera-feed">
                    <h2>Camera Feed</h2>
                    <img src="/camera1" id="camera1" />
                </div>
                <div class="config-panel">
                    <h3>Configuration Settings</h3>
                    <div id="status-message" class="status-message" style="display: none;"></div>
                    
                    <div class="config-section">
                        <h4>General Settings</h4>
                        <div class="config-input">
                            <label>Movement Threshold:</label>
                            <input type="number" id="movement-threshold" value="1000">
                        </div>
                        <div class="config-input">
                            <label>Sleeping Threshold:</label>
                            <input type="number" id="sleeping-threshold" value="5">
                        </div>
                    </div>

                    <div class="config-section">
                        <h4>Wheel Area</h4>
                        <div class="area-box">
                            <div class="config-input">
                                <label>X1:</label>
                                <input type="number" id="wheel-x1" value="200">
                                <label>Y1:</label>
                                <input type="number" id="wheel-y1" value="200">
                            </div>
                            <div class="config-input">
                                <label>X2:</label>
                                <input type="number" id="wheel-x2" value="440">
                                <label>Y2:</label>
                                <input type="number" id="wheel-y2" value="280">
                            </div>
                        </div>
                    </div>

                    <div class="config-section">
                        <h4>Food Area</h4>
                        <div class="area-box">
                            <div class="config-input">
                                <label>X1:</label>
                                <input type="number" id="food-x1" value="50">
                                <label>Y1:</label>
                                <input type="number" id="food-y1" value="300">
                            </div>
                            <div class="config-input">
                                <label>X2:</label>
                                <input type="number" id="food-x2" value="150">
                                <label>Y2:</label>
                                <input type="number" id="food-y2" value="400">
                            </div>
                        </div>
                    </div>

                    <div class="config-section">
                        <h4>Water Area</h4>
                        <div class="area-box">
                            <div class="config-input">
                                <label>X1:</label>
                                <input type="number" id="water-x1" value="450">
                                <label>Y1:</label>
                                <input type="number" id="water-y1" value="300">
                            </div>
                            <div class="config-input">
                                <label>X2:</label>
                                <input type="number" id="water-x2" value="550">
                                <label>Y2:</label>
                                <input type="number" id="water-y2" value="400">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <script>
                let configMode = false;
                let activityDetectionEnabled = true;
                
                function showStatus(message, isError = false) {
                    const statusDiv = document.getElementById('status-message');
                    statusDiv.textContent = message;
                    statusDiv.className = `status-message ${isError ? 'error' : 'success'}`;
                    statusDiv.style.display = 'block';
                    setTimeout(() => {
                        statusDiv.style.display = 'none';
                    }, 3000);
                }
                
                function toggleConfig() {
                    configMode = !configMode;
                    document.getElementById('camera1').src = '/camera1?config=' + configMode;
                }
                
                function toggleActivityDetection() {
                    activityDetectionEnabled = !activityDetectionEnabled;
                    const button = document.getElementById('activity-detection-btn');
                    button.textContent = activityDetectionEnabled ? 'Disable Activity Detection' : 'Enable Activity Detection';
                    
                    // Update the configuration
                    const config = {
                        ...getCurrentConfig(),
                        ACTIVITY_DETECTION_ENABLED: activityDetectionEnabled
                    };
                    
                    saveConfigToServer(config);
                }
                
                function getCurrentConfig() {
                    return {
                        MOVEMENT_THRESHOLD: parseInt(document.getElementById('movement-threshold').value),
                        SLEEPING_THRESHOLD: parseInt(document.getElementById('sleeping-threshold').value),
                        ACTIVITY_DETECTION_ENABLED: activityDetectionEnabled,
                        WHEEL_AREA: {
                            x1: parseInt(document.getElementById('wheel-x1').value),
                            y1: parseInt(document.getElementById('wheel-y1').value),
                            x2: parseInt(document.getElementById('wheel-x2').value),
                            y2: parseInt(document.getElementById('wheel-y2').value)
                        },
                        FOOD_AREA: {
                            x1: parseInt(document.getElementById('food-x1').value),
                            y1: parseInt(document.getElementById('food-y1').value),
                            x2: parseInt(document.getElementById('food-x2').value),
                            y2: parseInt(document.getElementById('food-y2').value)
                        },
                        WATER_AREA: {
                            x1: parseInt(document.getElementById('water-x1').value),
                            y1: parseInt(document.getElementById('water-y1').value),
                            x2: parseInt(document.getElementById('water-x2').value),
                            y2: parseInt(document.getElementById('water-y2').value)
                        }
                    };
                }
                
                function saveConfigToServer(config) {
                    fetch('/config', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(config)
                    })
                    .then(response => response.json())
                    .then(data => {
                        showStatus('Configuration saved successfully!');
                    })
                    .catch(error => {
                        showStatus('Error saving configuration: ' + error, true);
                    });
                }
                
                function saveConfig() {
                    saveConfigToServer(getCurrentConfig());
                }
                
                // Load initial configuration
                fetch('/config')
                    .then(response => response.json())
                    .then(config => {
                        // Set general settings
                        document.getElementById('movement-threshold').value = config.MOVEMENT_THRESHOLD;
                        document.getElementById('sleeping-threshold').value = config.SLEEPING_THRESHOLD;
                        activityDetectionEnabled = config.ACTIVITY_DETECTION_ENABLED;
                        document.getElementById('activity-detection-btn').textContent = 
                            activityDetectionEnabled ? 'Disable Activity Detection' : 'Enable Activity Detection';
                        
                        // Set wheel area
                        document.getElementById('wheel-x1').value = config.WHEEL_AREA.x1;
                        document.getElementById('wheel-y1').value = config.WHEEL_AREA.y1;
                        document.getElementById('wheel-x2').value = config.WHEEL_AREA.x2;
                        document.getElementById('wheel-y2').value = config.WHEEL_AREA.y2;
                        
                        // Set food area
                        document.getElementById('food-x1').value = config.FOOD_AREA.x1;
                        document.getElementById('food-y1').value = config.FOOD_AREA.y1;
                        document.getElementById('food-x2').value = config.FOOD_AREA.x2;
                        document.getElementById('food-y2').value = config.FOOD_AREA.y2;
                        
                        // Set water area
                        document.getElementById('water-x1').value = config.WATER_AREA.x1;
                        document.getElementById('water-y1').value = config.WATER_AREA.y1;
                        document.getElementById('water-x2').value = config.WATER_AREA.x2;
                        document.getElementById('water-y2').value = config.WATER_AREA.y2;
                    });
            </script>
        </body>
    </html>
    """

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8081, threaded=True)
    finally:
        # Release camera resources when the application stops
        camera1.release()
        # camera2.release()
