from flask import Flask, Response
import cv2
import random
import time

app = Flask(__name__)

# Camera setup (UVC camera, usually index 0; adjust if needed)
camera = cv2.VideoCapture(0)  # CAP_DSHOW for Windows
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Resolution: 640x480
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)           # Framerate: 30 fps

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
        
        # Add text overlay
        text = f"Temp: {temperature:.1f}°C  Hum: {humidity:.1f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
