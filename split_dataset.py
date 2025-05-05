import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir: str, train_ratio: float = 0.8):
    """
    Split the dataset into training and validation sets.
    
    Args:
        source_dir: Path to the source directory containing images
        train_ratio: Ratio of training data (default: 0.8)
    """
    # Create paths
    source_path = Path(source_dir)
    train_path = source_path / 'train'
    val_path = source_path / 'val'
    
    # Create validation directory if it doesn't exist
    val_path.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = list(train_path.glob('*.jpg'))
    random.shuffle(image_files)
    
    # Calculate split point
    split_idx = int(len(image_files) * train_ratio)
    
    # Move files to validation set
    for img_file in image_files[split_idx:]:
        # Move image
        shutil.move(str(img_file), str(val_path / img_file.name))
        
        # Move corresponding label file if it exists
        label_file = Path('dataset/labels/train') / f"{img_file.stem}.txt"
        if label_file.exists():
            val_label_path = Path('dataset/labels/val')
            val_label_path.mkdir(exist_ok=True)
            shutil.move(str(label_file), str(val_label_path / label_file.name))

if __name__ == "__main__":
    # Split the dataset
    split_dataset('dataset/images')
    print("Dataset split completed successfully!") 