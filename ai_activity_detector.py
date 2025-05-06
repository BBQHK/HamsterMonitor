import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import Tuple, List, Dict
import time
import os
from pathlib import Path

class HamsterActivityDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize the Hamster Activity Detector.
        
        Args:
            model_path: Path to the custom trained YOLOv8 model weights
                       If None, will use the default YOLOv8n model (not recommended for production)
        """
        if model_path is None:
            print("WARNING: Using default YOLOv8n model. This model is not trained for hamster activity detection.")
            print("Please train a custom model using the train() method before using in production.")
            self.model = YOLO('yolov8n.pt')
        else:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = YOLO(model_path)
            
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Activity states
        self.activity_states = {
            'running': 0,
            'eating': 0,
            'drinking': 0
        }
        
        # Activity thresholds
        self.activity_thresholds = {
            'running': 0.5,
            'eating': 0.4,
            'drinking': 0.4
        }
        
        # Activity history for temporal analysis
        self.activity_history = []
        self.max_history = 30  # Keep last 30 frames of activity
        
    def detect_activity(self, frame: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Detect hamster activity in the given frame.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Tuple containing:
            - Most likely activity
            - Dictionary of activity probabilities
        """
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        # Get detection results
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        
        # Initialize activity probabilities
        activity_probs = {
            'running': 0.0,
            'eating': 0.0,
            'drinking': 0.0
        }
        
        # Process detections
        for conf, cls in zip(confidences, classes):
            # Map YOLO classes to activities
            if cls == 0:  # wheel
                activity_probs['eating'] = conf
            elif cls == 1:  # food
                activity_probs['running'] = conf
            elif cls == 2:  # water
                activity_probs['drinking'] = conf
        
        # Update activity history
        self.activity_history.append(activity_probs)
        if len(self.activity_history) > self.max_history:
            self.activity_history.pop(0)
        
        # Determine most likely activity
        most_likely_activity = max(activity_probs.items(), key=lambda x: x[1])[0]
        
        return most_likely_activity, activity_probs
    
    def get_activity_pattern(self) -> Dict[str, float]:
        """
        Analyze activity patterns over time.
        
        Returns:
            Dictionary of average activity probabilities
        """
        if not self.activity_history:
            return {k: 0.0 for k in self.activity_states.keys()}
        
        # Calculate average probabilities
        avg_probs = {k: 0.0 for k in self.activity_states.keys()}
        for probs in self.activity_history:
            for act, prob in probs.items():
                avg_probs[act] += prob
        
        for act in avg_probs:
            avg_probs[act] /= len(self.activity_history)
        
        return avg_probs 