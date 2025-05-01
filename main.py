from flask import Flask, Response
import cv2
import random
import time
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Camera setup (UVC camera, usually index 0; adjust if needed)
camera = cv2.VideoCapture(0)  # CAP_DSHOW for Windows
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Resolution: 640x480
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 15)           # Framerate: 15 fps

def get_simulated_readings():
    # Simulate temperature between 20-30°C and humidity between 40-60%
    temperature = random.uniform(20.0, 30.0)
    humidity = random.uniform(40.0, 60.0)
    return temperature, humidity

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Get simulated temperature and humidity
        temperature, humidity = get_simulated_readings()
        
        # Get current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add text overlay - using 'C' instead of degree symbol
        text1 = f"Temp: {temperature:.1f}C  Hum: {humidity:.1f}%"
        text2 = f"Time: {current_time}"
        
        # Get text size
        font_scale = 0.5
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text1_width, text1_height), _ = cv2.getTextSize(text1, font, font_scale, thickness)
        (text2_width, text2_height), _ = cv2.getTextSize(text2, font, font_scale, thickness)
        
        # Add semi-transparent background for both lines
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (5 + max(text1_width, text2_width) + 10, 5 + text1_height + text2_height + 20), (0, 0, 0), -1)
        alpha = 0.5  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add text
        cv2.putText(frame, text1, (10, 20), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, text2, (10, 40), font, font_scale, (255, 255, 255), thickness)

        # Encode frame as JPEG for MJPEG streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, threaded=True)
