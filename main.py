from flask import Flask, Response
import cv2

# Constants
CAMERA_INDICES = [0, 2]  # List of camera indices to use
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Initialize Flask app
app = Flask(__name__)

# Dictionary to store camera objects
cameras = {}

def setup_camera(camera_index):
    """Setup a camera with specified index."""
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)
    return camera

def get_camera(camera_index):
    """Get or create a camera object for the given index."""
    if camera_index not in cameras:
        camera = setup_camera(camera_index)
        if camera.isOpened():
            cameras[camera_index] = camera
        else:
            return None
    return cameras[camera_index]

def generate_frames(camera_index):
    """Generate video frames from specified camera."""
    camera = get_camera(camera_index)
    if camera is None:
        return
    
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Encode frame as JPEG for MJPEG streaming
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
        app.run(host='0.0.0.0', port=8081, threaded=True)
    finally:
        # Release all camera resources when the application stops
        for camera in cameras.values():
            camera.release()
