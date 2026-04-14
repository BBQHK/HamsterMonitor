import cv2
from ai_activity_detector import HamsterActivityDetector
import time
import numpy as np
import os
from datetime import datetime

def format_time(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def render_video(video_path, model_path="best.pt"):
    """
    Render a labeled video with activity detection.
    
    Args:
        video_path: Path to the input video file
        model_path: Path to the trained model weights
    """
    # Initialize the detector
    detector = HamsterActivityDetector(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory if it doesn't exist
    output_dir = "labeled_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_labeled_{timestamp}.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print(f"Rendering labeled video to: {output_path}")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get current time
        current_time = frame_count / fps
        
        # Detect activity
        activity, probs = detector.detect_activity(frame)
        
        # Add activity information
        text = f"Activity: {activity}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add probabilities
        y = 70
        for act, prob in probs.items():
            text = f"{act}: {prob:.2f}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 30
        
        # Write frame to output video
        out.write(frame)
        
        # Update progress
        frame_count += 1
        if frame_count % 30 == 0:  # Update progress every second
            elapsed = time.time() - start_time
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) - Elapsed: {elapsed:.1f}s")
    
    # Release resources
    cap.release()
    out.release()
    
    # Print final statistics
    total_time = time.time() - start_time
    print(f"\nFinished rendering video to: {output_path}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average processing speed: {frame_count/total_time:.1f} FPS")

if __name__ == "__main__":
    # Replace with your video file path
    video_path = "Hamster monitor-20250425-232521.mp4"
    render_video(video_path) 