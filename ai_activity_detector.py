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
            'sleeping': 0.6,
            'exploring': 0.3
        }
        
        # Activity history for temporal analysis
        self.activity_history = []
        self.max_history = 30  # Keep last 30 frames of activity
        
    @staticmethod
    def train(data_yaml: str, epochs: int = 100, imgsz: int = 640, batch: int = 16):
        """
        Train a custom YOLOv8 model for hamster activity detection.
        
        Args:
            data_yaml: Path to the YAML file containing dataset configuration
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size for training
            
        Returns:
            Path to the trained model
        """
        # Create a new YOLOv8 model
        model = YOLO('yolov8n.pt')
        
        # Train the model
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='hamster_activity_detector'
        )
        
        # Return path to the best model
        return Path('runs/detect/hamster_activity_detector/weights/best.pt')
        
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
            if cls == 0:  # hamster
                # Analyze position and movement
                activity_probs['exploring'] = conf
            elif cls == 1:  # wheel
                activity_probs['running'] = conf
            elif cls == 2:  # food
                activity_probs['eating'] = conf
            elif cls == 3:  # water
                activity_probs['drinking'] = conf
            elif cls == 4:  # sleeping area
                activity_probs['sleeping'] = conf
        
        # Update activity history
        self.activity_history.append(activity_probs)
        if len(self.activity_history) > self.max_history:
            self.activity_history.pop(0)
        
        # Determine most likely activity
        most_likely_activity = max(activity_probs.items(), key=lambda x: x[1])[0]
        
        return most_likely_activity, activity_probs
    
    def draw_activity_overlay(self, frame: np.ndarray, activity: str, probabilities: Dict[str, float]) -> np.ndarray:
        """
        Draw activity information overlay on the frame.
        
        Args:
            frame: Input frame
            activity: Current activity
            probabilities: Activity probabilities
            
        Returns:
            Frame with activity overlay
        """
        # Create overlay
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw semi-transparent background
        cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw activity text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Activity: {activity}", (10, 30), font, 0.7, (255, 255, 255), 2)
        
        # Draw probabilities
        y = 60
        for act, prob in probabilities.items():
            cv2.putText(frame, f"{act}: {prob:.2f}", (10, y), font, 0.5, (255, 255, 255), 1)
            y += 20
        
        return frame
    
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