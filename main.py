from flask import Flask, Response
import cv2

# Constants
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Initialize Flask app
app = Flask(__name__)

# Camera setup
def setup_camera(camera_index):
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)
    return camera

# Initialize camera
camera = setup_camera(CAMERA_INDEX)

def generate_frames():
    """Generate video frames from camera."""
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Encode frame as JPEG for MJPEG streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/camera')
def camera_feed():
    """Stream video feed from camera."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Serve a simple HTML page with camera feed."""
    return """
    <html>
        <head>
            <title>Camera Feed</title>
            <style>
                body { margin: 0; padding: 20px; background: #333; color: white; }
                .camera-feed { max-width: 800px; margin: 0 auto; }
                img { width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <div class="camera-feed">
                <h2>Camera Feed</h2>
                <img src="/camera" />
            </div>
        </body>
    </html>
    """

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8081, threaded=True)
    finally:
        # Release camera resources when the application stops
        camera.release()
