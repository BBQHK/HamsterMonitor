import os
import cv2
import json
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime

def create_dataset_structure():
    """Create the necessary directory structure for the dataset."""
    # Create main dataset directory
    dataset_dir = Path('hamster_dataset')
    dataset_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (dataset_dir / 'images').mkdir(exist_ok=True)
    (dataset_dir / 'labels').mkdir(exist_ok=True)
    
    # Create train/val/test splits
    for split in ['train', 'val', 'test']:
        (dataset_dir / 'images' / split).mkdir(exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(exist_ok=True)
    
    return dataset_dir

def create_dataset_yaml(dataset_dir: Path):
    """Create the dataset YAML file required for YOLOv8 training."""
    yaml_content = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'hamster',
            1: 'wheel',
            2: 'food',
            3: 'water',
            4: 'sleeping_area'
        }
    }
    
    with open(dataset_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f)
    
    return dataset_dir / 'dataset.yaml'

def capture_training_samples(camera_index: int = 0, num_samples: int = 100):
    """
    Capture training samples from the camera.
    
    Args:
        camera_index: Camera device index
        num_samples: Number of samples to capture
    """
    # Create dataset structure
    dataset_dir = create_dataset_structure()
    yaml_path = create_dataset_yaml(dataset_dir)
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    
    print("Starting sample capture...")
    print("Press 's' to save a sample, 'q' to quit")
    print("After saving, enter the activity label (0-4):")
    print("0: hamster, 1: wheel, 2: food, 3: water, 4: sleeping_area")
    
    sample_count = 0
    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue
            
        # Display frame
        cv2.imshow('Capture Samples', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            img_path = dataset_dir / 'images' / 'train' / f'sample_{timestamp}.jpg'
            cv2.imwrite(str(img_path), frame)
            
            # Get label from user
            print("\nEnter label (0-4):")
            label = input()
            try:
                label = int(label)
                if 0 <= label <= 4:
                    # Create YOLO format label file
                    label_path = dataset_dir / 'labels' / 'train' / f'sample_{timestamp}.txt'
                    with open(label_path, 'w') as f:
                        # YOLO format: class_id x_center y_center width height
                        # For now, we'll use dummy coordinates - you'll need to adjust these
                        f.write(f"{label} 0.5 0.5 0.2 0.2\n")
                    sample_count += 1
                    print(f"Saved sample {sample_count}/{num_samples}")
                else:
                    print("Invalid label. Please enter a number between 0 and 4.")
            except ValueError:
                print("Invalid input. Please enter a number between 0 and 4.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nDataset preparation complete!")
    print(f"Dataset YAML file: {yaml_path}")
    print("\nNext steps:")
    print("1. Review and adjust the labels in the dataset")
    print("2. Run the training script:")
    print("   from ai_activity_detector import HamsterActivityDetector")
    print("   detector = HamsterActivityDetector()")
    print("   model_path = detector.train('hamster_dataset/dataset.yaml')")

if __name__ == '__main__':
    capture_training_samples() 