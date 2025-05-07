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
        
        # Check if all probabilities are 0.0
        if all(prob == 0.0 for prob in activity_probs.values()):
            activity = "Unknown"
            activity_probability = 0.0
        else:
            activity_probability = float(activity_probs[activity])
            
        # Prepare response with only activity information
        response = {
            "activity": activity,
            "activity_probability": activity_probability,
            "all_probabilities": {k: float(v) for k, v in activity_probs.items()}
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, threaded=True)
