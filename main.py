from flask import Flask, Response, request, jsonify
import cv2
import numpy as np
from datetime import datetime
import json
import os
from ai_activity_detector import HamsterActivityDetector
import requests
from io import BytesIO
from PIL import Image

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_INDICES = [0, 2]  # List of camera indices to use
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
activity_detector = HamsterActivityDetector("best.pt")

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

def process_frame(frame_bytes):
    """Process a frame from the camera feed with activity detection."""
    # Convert bytes to numpy array
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        # Generate a black frame with error message
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(frame, "Failed to process frame", (50, FRAME_HEIGHT//2), 
                   FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        return frame

    # Get sensor readings and detect activity
    temperature, humidity = get_simulated_readings()
    current_time = get_current_timestamp()
    
    # Use AI to detect activity
    activity, activity_probs = activity_detector.detect_activity(frame)
    
    # Prepare text for overlay
    texts = [f"Time: {current_time}"]
    texts.append(f"Temp: {temperature:.1f}C  Hum: {humidity:.1f}%")
    texts.append(f"Activity: {activity} ({activity_probs[activity]*100:.1f}%)")
    
    # Add text overlay
    add_text_overlay(frame, texts)
    
    return frame

@app.route('/camera<int:camera_index>')
def camera_feed(camera_index):
    """Stream video feed from specified camera with activity detection overlay."""
    if camera_index not in CAMERA_INDICES:
        return "Invalid camera index", 400
        
    def generate():
        while True:
            try:
                # Get frame from start_cameras.py
                response = requests.get(f'http://192.168.50.167:8081/camera{camera_index}')
                if response.status_code == 200:
                    # Process the frame
                    frame = process_frame(response.content)
                    
                    # Encode frame as JPEG for MJPEG streaming
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error processing frame: {e}")
                # Generate error frame
                frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, f"Error: {str(e)}", (50, FRAME_HEIGHT//2), 
                           FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/activity_pattern')
def get_activity_pattern():
    """Get the current activity pattern analysis."""
    pattern = activity_detector.get_activity_pattern()
    # Convert numpy float32 values to regular Python floats
    pattern = {k: float(v) for k, v in pattern.items()}
    return jsonify(pattern)

@app.route('/')
def index():
    """Serve a simple HTML page with camera feeds and configuration interface."""
    camera_feed_html = ""
    for camera_index in CAMERA_INDICES:
        camera_feed_html += f"""
                    <div class="camera-feed">
                        <h2>/camera{camera_index}</h2>
                        <img src="/camera{camera_index}" id="camera{camera_index}" />
                    </div>
        """

    return f"""
    <html>
        <head>
            <title>Hamster Monitor</title>
            <style>
                :root {{
                    --primary-color: #4CAF50;
                    --background-dark: #333;
                    --background-light: #444;
                    --text-color: white;
                    --text-secondary: #888;
                    --border-radius: 8px;
                    --box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    --spacing: 20px;
                }}

                body {{ 
                    margin: 0; 
                    padding: var(--spacing); 
                    background: var(--background-dark); 
                    color: var(--text-color);
                    font-family: Arial, sans-serif;
                }}

                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                }}

                .main-content {{
                    display: flex;
                    gap: var(--spacing);
                    align-items: flex-start;
                }}

                .camera-grid {{
                    flex: 2;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: var(--spacing);
                }}

                .camera-feed {{ 
                    background: var(--background-light);
                    padding: var(--spacing);
                    border-radius: var(--border-radius);
                    box-shadow: var(--box-shadow);
                }}

                .camera-feed h2 {{
                    margin: 0 0 var(--spacing) 0;
                    color: var(--primary-color);
                    font-size: 1.5em;
                }}

                .activity-chart {{ 
                    flex: 1;
                    padding: var(--spacing); 
                    background: var(--background-light); 
                    border-radius: var(--border-radius);
                    box-shadow: var(--box-shadow);
                    position: sticky;
                    top: var(--spacing);
                }}

                .activity-chart h3 {{ 
                    margin: 0 0 var(--spacing) 0;
                    color: var(--primary-color);
                    font-size: 1.2em;
                }}

                img {{ 
                    width: 100%; 
                    height: auto;
                    border-radius: var(--border-radius);
                }}

                .activity-bar {{ 
                    height: 30px; 
                    margin: 8px 0; 
                    background: var(--background-dark); 
                    border-radius: var(--border-radius);
                    display: flex;
                    align-items: center;
                    padding: 0 10px;
                }}

                .activity-bar-fill {{ 
                    height: 20px; 
                    background: var(--primary-color); 
                    border-radius: var(--border-radius);
                    transition: width 0.3s ease;
                }}

                .activity-label {{
                    margin-left: 10px;
                    font-size: 0.9em;
                    min-width: 150px;
                }}

                .activity-probability {{
                    margin-left: auto;
                    font-weight: bold;
                    color: var(--primary-color);
                }}

                .activity-timestamp {{
                    font-size: 0.8em;
                    color: var(--text-secondary);
                    margin-top: var(--spacing);
                    text-align: right;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="main-content">
                    <div class="camera-grid">
                        {camera_feed_html}
                    </div>
                    <div class="activity-chart">
                        <h3>Activity Analysis</h3>
                        <div id="activity-bars"></div>
                    </div>
                </div>
            </div>
            <script>
                // Update activity chart
                function updateActivityChart() {{
                    fetch('/activity_pattern')
                        .then(response => response.json())
                        .then(data => {{
                            const container = document.getElementById('activity-bars');
                            container.innerHTML = '';
                            
                            // Sort activities by probability
                            const sortedActivities = Object.entries(data)
                                .sort((a, b) => b[1] - a[1]);
                            
                            for (const [activity, probability] of sortedActivities) {{
                                const bar = document.createElement('div');
                                bar.className = 'activity-bar';
                                
                                const fill = document.createElement('div');
                                fill.className = 'activity-bar-fill';
                                fill.style.width = `${{probability * 100}}%`;
                                
                                const label = document.createElement('span');
                                label.className = 'activity-label';
                                label.textContent = activity;
                                
                                const prob = document.createElement('span');
                                prob.className = 'activity-probability';
                                prob.textContent = `${{(probability * 100).toFixed(1)}}%`;
                                
                                bar.appendChild(fill);
                                bar.appendChild(label);
                                bar.appendChild(prob);
                                container.appendChild(bar);
                            }}
                            
                            // Add timestamp
                            const timestamp = document.createElement('div');
                            timestamp.className = 'activity-timestamp';
                            timestamp.textContent = `Last updated: ${{new Date().toLocaleTimeString()}}`;
                            container.appendChild(timestamp);
                        }});
                }}
                
                // Update activity chart every 2 seconds
                setInterval(updateActivityChart, 2000);
                
                // Initial update
                updateActivityChart();
            </script>
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, threaded=True)
