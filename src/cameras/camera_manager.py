"""Camera management module for handling multiple camera feeds."""

import cv2
import threading
from queue import Queue
import requests
import json
from src.config import settings

class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.frame_queue = Queue(maxsize=10)
        self.api_error_count = 0
        self.last_activity_result = {
            'activity': 'Unknown',
            'activity_probability': 0.0
        }
        self._initialize_cameras()
        self._start_processing_thread()

    def _initialize_cameras(self):
        """Initialize all cameras at startup."""
        for camera_index in settings.CAMERA_INDICES:
            camera = self._setup_camera(camera_index)
            if camera.isOpened():
                self.cameras[camera_index] = camera
                print(f"Successfully initialized camera {camera_index}")
            else:
                print(f"Failed to initialize camera {camera_index}")

    def _setup_camera(self, camera_index):
        """Setup a camera with specified index."""
        camera = cv2.VideoCapture(camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, settings.FPS)
        return camera

    def _process_frame_async(self):
        """Process frames from the queue asynchronously."""
        while True:
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    continue
                    
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                try:
                    # Send frame to main.py for processing with timeout
                    response = requests.post(settings.MAIN_API_URL, data=frame_bytes, timeout=2)
                    if response.status_code == 200:
                        self.last_activity_result.update(response.json())
                        self.api_error_count = 0
                except (requests.RequestException, json.JSONDecodeError) as e:
                    self.api_error_count += 1
                    print(f"API Error: {e}")
                    
            except Exception as e:
                print(f"Error in async processing: {e}")

    def _start_processing_thread(self):
        """Start the async processing thread."""
        self.processing_thread = threading.Thread(target=self._process_frame_async, daemon=True)
        self.processing_thread.start()

    def _feed_camera0_frames(self):
        """Continuously feed frames from camera 0 into the queue."""
        camera = self.cameras.get(0)
        if camera is None:
            print("Failed to start camera 0 frame feeding - camera not initialized")
            return
            
        frame_count = 0
        while True:
            try:
                success, frame = camera.read()
                if not success:
                    print("Failed to read frame from camera 0")
                    continue
                    
                # Only process every FRAME_SKIP frames
                if frame_count % settings.FRAME_SKIP == 0:
                    try:
                        self.frame_queue.put(frame, block=False)
                    except:
                        pass  # Skip this frame if queue is full
                
                frame_count += 1
                    
            except Exception as e:
                print(f"Error in camera 0 frame feeding: {e}")

    def start_camera0_feeding(self):
        """Start the camera 0 frame feeding thread."""
        self.camera0_thread = threading.Thread(target=self._feed_camera0_frames, daemon=True)
        self.camera0_thread.start()

    def get_camera(self, camera_index):
        """Get a camera object for the given index."""
        return self.cameras.get(camera_index)

    def get_activity_status(self):
        """Get the current activity status."""
        if self.api_error_count >= settings.API_ERROR_THRESHOLD:
            return "API Unavailable", 0.0
        return self.last_activity_result['activity'], self.last_activity_result['activity_probability']

    def cleanup(self):
        """Release all camera resources."""
        for camera in self.cameras.values():
            camera.release() 