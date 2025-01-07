import os
import cv2
import logging

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'frame_extraction.log'), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_video_frames(video_path, save_dir):
    """Extract frames from the video and save them as images."""
    video_name = os.path.basename(video_path).split('.')[0]
    save_path = os.path.join(save_dir, video_name)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Unable to open video file: {video_path}")
        return None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Extracting {frame_count} frames from {video_path}")
    
    frame_idx = 0
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save each frame as an image
        frame_file = os.path.join(save_path, f'frame_{frame_idx:05d}.png')
        cv2.imwrite(frame_file, frame)
        frames.append(frame_file)
        frame_idx += 1
    
    cap.release()
    logger.info(f"Extracted and saved {len(frames)} frames from {video_path}")
    return frames

def main():
    video_dir = 'data/videos'  # Path to the video directory
    save_dir = 'data/frames'   # Directory to save the extracted frames
    os.path.join('..', 'data', 'Train.xlsx')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)
            extract_video_frames(video_path, save_dir)

if __name__ == "__main__":
    main()
