import cv2
import os
from pathlib import Path
import argparse
from tqdm import tqdm

def extract_frames(video_path: str, output_dir: str, frame_interval: int = 1, prefix: str = "frame"):
    """
    Extract frames from a video file and save them as images.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the extracted frames
        frame_interval: Extract every Nth frame (default: 1, meaning every frame)
        prefix: Prefix for the output image filenames
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties:")
    print(f"- Total frames: {total_frames}")
    print(f"- FPS: {fps}")
    print(f"- Duration: {total_frames/fps:.2f} seconds")
    print(f"- Extracting every {frame_interval} frame(s)")
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save frame if it's the right interval
            if frame_count % frame_interval == 0:
                frame_path = output_path / f"{prefix}_{saved_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_count += 1
                
            frame_count += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    
    print(f"\nExtraction complete!")
    print(f"- Total frames processed: {frame_count}")
    print(f"- Frames saved: {saved_count}")
    print(f"- Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Extract frames from a video file')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('output_dir', help='Directory to save the extracted frames')
    parser.add_argument('--interval', type=int, default=1, 
                        help='Extract every Nth frame (default: 1)')
    parser.add_argument('--prefix', default='frame',
                        help='Prefix for output filenames (default: frame)')
    
    args = parser.parse_args()
    
    extract_frames(args.video_path, args.output_dir, args.interval, args.prefix)

if __name__ == "__main__":
    main() 