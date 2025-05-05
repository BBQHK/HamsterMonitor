from ai_activity_detector import HamsterActivityDetector
import os
from pathlib import Path

def train_model():
    # Initialize the detector
    detector = HamsterActivityDetector()
    
    # Training parameters
    epochs = 100  # Number of training epochs
    imgsz = 640   # Image size
    batch = 16    # Batch size
    
    # Path to the YAML configuration file
    data_yaml = "hamster_data.yaml"
    
    # Train the model
    print("Starting training...")
    model_path = detector.train(
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch
    )
    
    print(f"Training completed! Model saved to: {model_path}")
    return model_path

if __name__ == "__main__":
    train_model() 