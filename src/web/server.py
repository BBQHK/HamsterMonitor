"""Web server module for handling camera feeds and status endpoints."""

from flask import Flask, Response
import cv2
import json
from datetime import datetime
from src.config import settings

class WebServer:
    def __init__(self, camera_manager, sensor_manager):
        self.app = Flask(__name__)
        self.camera_manager = camera_manager
        self.sensor_manager = sensor_manager
        self._setup_routes()

    def _setup_routes(self):
        """Setup all Flask routes."""
        self.app.route('/camera<int:camera_index>')(self.camera_feed)
        self.app.route('/')(self.index)
        self.app.route('/status')(self.get_status)

    def _add_text_overlay(self, frame, texts):
        """Add text overlay to a frame."""
        if not texts:
            return
            
        # Calculate total height needed and max width
        total_height = 0
        max_width = 0
        text_sizes = []
        
        for text in texts:
            (width, height), _ = cv2.getTextSize(text, getattr(cv2, settings.FONT), 
                                               settings.FONT_SCALE, settings.FONT_THICKNESS)
            text_sizes.append((width, height))
            max_width = max(max_width, width)
            total_height += height
        
        # Add padding between texts and around the box
        padding = settings.TEXT_PADDING
        total_height += padding * (len(texts) + 1)
        max_width += padding * 2
        
        # Add semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (padding, padding),
            (padding + max_width, padding + total_height),
            settings.BACKGROUND_COLOR,
            -1
        )
        cv2.addWeighted(overlay, settings.BACKGROUND_ALPHA, frame, 1 - settings.BACKGROUND_ALPHA, 0, frame)
        
        # Add text
        y = padding + text_sizes[0][1] + padding
        for i, text in enumerate(texts):
            cv2.putText(
                frame, 
                text, 
                (padding + 5, y), 
                getattr(cv2, settings.FONT), 
                settings.FONT_SCALE, 
                settings.TEXT_COLOR, 
                settings.FONT_THICKNESS
            )
            if i < len(texts) - 1:
                y += text_sizes[i + 1][1] + padding

    def _generate_frames(self, camera_index):
        """Generate video frames from specified camera."""
        camera = self.camera_manager.get_camera(camera_index)
        if camera is None:
            return
        
        while True:
            success, frame = camera.read()
            if not success:
                break

            try:
                # Get sensor readings
                temperature, humidity, air_quality, air_quality_ppm = self.sensor_manager.get_readings()
                
                # Get activity status
                activity, activity_prob = self.camera_manager.get_activity_status()
                
                # Prepare text overlay
                texts = [
                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Temp: {temperature:.1f}C  Hum: {humidity:.1f}%",
                    f"Air Quality: {air_quality} ({air_quality_ppm:.1f} PPM)"
                ]
                
                if activity == "API Unavailable":
                    texts.append("Activity: API Unavailable")
                elif activity == "Unknown":
                    texts.append("Activity: Unknown")
                else:
                    texts.append(f"Activity: {activity} ({activity_prob*100:.1f}%)")
                
                # Add text overlay to frame
                self._add_text_overlay(frame, texts)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                cv2.putText(frame, f"Error: {str(e)}", (50, settings.FRAME_HEIGHT//2), 
                           getattr(cv2, settings.FONT), settings.FONT_SCALE, 
                           settings.TEXT_COLOR, settings.FONT_THICKNESS)

            # Encode processed frame as JPEG for MJPEG streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def camera_feed(self, camera_index):
        """Stream video feed from specified camera index."""
        if camera_index not in settings.CAMERA_INDICES:
            return "Invalid camera index", 400
        return Response(self._generate_frames(camera_index), 
                       mimetype='multipart/x-mixed-replace; boundary=frame')

    def index(self):
        """Serve a simple HTML page with camera feeds."""
        camera_feed_html = ""
        for camera_index in settings.CAMERA_INDICES:
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

    def get_status(self):
        """Return current cage status including timestamp, temperature, humidity, and activity."""
        # Get sensor status
        status = self.sensor_manager.get_status()
        
        # Add activity status
        activity, activity_prob = self.camera_manager.get_activity_status()
        status.update({
            'hamster_activity': activity,
            'hamster_activity_probability': activity_prob
        })
        
        return json.dumps(status, indent=2)

    def run(self):
        """Run the Flask web server."""
        self.app.run(host=settings.HOST, port=settings.PORT, threaded=True) 