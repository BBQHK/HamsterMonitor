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
            'drinking': 0,
            'sleeping': 0,
            'exploring': 0
        }
        
        # Activity thresholds
        self.activity_thresholds = {
            'running': 0.5,
            'eating': 0.4,
            'drinking': 0.4,
            'sleeping': 0.3,  # Threshold for motion detection
            'exploring': 0.4  # Threshold for exploring detection
        }
        
        # Activity history for temporal analysis
        self.activity_history = []
        self.max_history = 30  # Keep last 30 frames of activity
        
        # Background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        
        # Motion detection parameters
        self.motion_threshold = 500  # Minimum area of motion to consider
        self.motion_history = []
        self.motion_history_size = 10  # Keep last 10 frames of motion data
        
    def detect_motion(self, frame: np.ndarray) -> float:
        """
        Detect motion in the frame using background subtraction.
        
        Args:
            frame: Input frame
            
        Returns:
            float: Motion intensity (0.0 to 1.0)
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5,5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate motion intensity
        motion_area = np.sum(fg_mask > 0)
        motion_intensity = min(1.0, motion_area / self.motion_threshold)
        
        # Update motion history
        self.motion_history.append(motion_intensity)
        if len(self.motion_history) > self.motion_history_size:
            self.motion_history.pop(0)
            
        # Calculate average motion over history
        avg_motion = np.mean(self.motion_history)
        
        return avg_motion
        
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
            'drinking': 0.0,
            'sleeping': 0.0,
            'exploring': 0.0
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
        
        # Detect motion
        motion_intensity = self.detect_motion(frame)
        
        # Check if there's significant motion but no specific activities detected
        has_specific_activity = any(prob > 0.3 for act, prob in activity_probs.items() 
                                  if act in ['running', 'eating', 'drinking'])
        
        if motion_intensity > self.activity_thresholds['exploring'] and not has_specific_activity:
            # High motion without specific activities = exploring
            activity_probs['exploring'] = motion_intensity
            activity_probs['sleeping'] = 0.0  # Can't be sleeping if exploring
        else:
            # Update sleeping probability based on motion
            if motion_intensity < self.activity_thresholds['sleeping']:
                activity_probs['sleeping'] = 1.0 - motion_intensity
            else:
                # If there's significant motion, reduce sleeping probability
                activity_probs['sleeping'] = max(0.0, 1.0 - motion_intensity)
        
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