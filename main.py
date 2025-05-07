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

@app.route('/process_frame', methods=['POST'])
def process_frame_api():
    """API endpoint to process a frame and return activity results."""
    try:
        # Get frame data from request
        frame_bytes = request.get_data()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Failed to process frame"}), 400

        # Use AI to detect activity
        activity, activity_probs = activity_detector.detect_activity(frame)
        
        # Prepare response with only activity information
        response = {
            "activity": activity,
            "activity_probability": float(activity_probs[activity]),
            "all_probabilities": {k: float(v) for k, v in activity_probs.items()}
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, threaded=True)
