import os
import logging
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import EfficientNetB0
from utils import (
    extract_visual_token,
    extract_emotion_token,
    extract_gesture_token
)
import numpy as np
import cv2  # Ensure OpenCV is imported for frame processing

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'feature_extraction_debug.log'), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_features(frames_dir, save_dir_visual, save_dir_emotion, save_dir_gesture):
    """Extract visual, emotion, and gesture features from frames and save them."""
    
    # Initialize models and layers
    logger.info("Initializing models and layers...")
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    visual_projection_layer = Dense(512, activation=None)
    emotion_dense_layer = Dense(512, activation=None)
    gesture_dense_layer = Dense(512, activation=None)
    logger.info("Models and layers initialized.")

    # Iterate over the datasets: train, test, and dev
    for subset in ['train', 'test', 'dev']:
        logger.info(f"Processing subset: {subset}")
        subset_frames_dir = os.path.join(frames_dir, subset)
        
        if not os.path.exists(subset_frames_dir):
            logger.warning(f"Frames directory for subset {subset} does not exist: {subset_frames_dir}")
            continue
        
        # Ensure feature directories exist
        subset_visual_dir = os.path.join(save_dir_visual, subset)
        subset_emotion_dir = os.path.join(save_dir_emotion, subset)
        subset_gesture_dir = os.path.join(save_dir_gesture, subset)
        for path in [subset_visual_dir, subset_emotion_dir, subset_gesture_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

        # Process each video (i.e., each folder of frames)
        for video_name in os.listdir(subset_frames_dir):
            logger.info(f"Processing video: {video_name}")
            video_frame_dir = os.path.join(subset_frames_dir, video_name)

            if not os.path.isdir(video_frame_dir):
                logger.warning(f"Skipping non-directory file: {video_frame_dir}")
                continue

            # Load frames for the current video
            frames = []
            for frame_file in sorted(os.listdir(video_frame_dir)):
                frame_path = os.path.join(video_frame_dir, frame_file)
                logger.info(f"Loading frame: {frame_path}")
                try:
                    frame = tf.io.read_file(frame_path)
                    frame = tf.image.decode_image(frame, channels=3)
                    frames.append(frame)
                except Exception as e:
                    logger.warning(f"Error processing frame {frame_file}: {e}. Skipping.")

            if not frames:
                logger.warning(f"No valid frames found for video: {video_name}. Skipping.")
                continue

            # Stack frames into tensor
            try:
                frames_tensor = tf.stack(frames)
                logger.info(f"Frames stacked into tensor for video: {video_name}")
            except Exception as e:
                logger.error(f"Error stacking frames for video {video_name}: {e}. Skipping.")
                continue

            # Extract visual tokens
            visual_tokens = extract_visual_token(
                frames_tensor, efficientnet_model, visual_projection_layer,
                video_name, save_dir=subset_visual_dir  # Pass video_name and save_dir
            )

            if visual_tokens is None:
                logger.warning(f"No valid visual tokens for video {video_name}. Skipping.")
                continue

            visual_token_file = os.path.join(subset_visual_dir, f'{video_name}.npy')
            logger.info(f"Saving visual tokens for video {video_name} to {visual_token_file}")
            np.save(visual_token_file, visual_tokens.numpy())

            # Extract emotion tokens
            emotion_tokens = extract_emotion_token(
                frames_tensor, emotion_dense_layer,
                video_name, save_dir=subset_emotion_dir  # Pass video_name and save_dir
            )

            if emotion_tokens is None:
                logger.warning(f"No valid emotion tokens for video {video_name}. Skipping.")
                continue

            emotion_token_file = os.path.join(subset_emotion_dir, f'{video_name}.npy')
            logger.info(f"Saving emotion tokens for video {video_name} to {emotion_token_file}")
            np.save(emotion_token_file, emotion_tokens.numpy())

            # Extract gesture tokens
            gesture_tokens = extract_gesture_token(
                frames_tensor, gesture_dense_layer,
                video_name, save_dir=subset_gesture_dir  # Pass video_name and save_dir
            )

            if gesture_tokens is None:
                logger.warning(f"No valid gesture tokens for video {video_name}. Skipping.")
                continue

            gesture_token_file = os.path.join(subset_gesture_dir, f'{video_name}.npy')
            logger.info(f"Saving gesture tokens for video {video_name} to {gesture_token_file}")
            np.save(gesture_token_file, gesture_tokens.numpy())

def main():
    # Directory containing extracted frames
    frames_dir = 'data/output_frames'
    
    # Directories to save extracted features
    visual_feature_dir = 'data/features/visual'
    emotion_feature_dir = 'data/features/emotion'
    gesture_feature_dir = 'data/features/gesture'

    # Check if frames directory exists
    if not os.path.exists(frames_dir):
        logger.error(f"Frames directory does not exist: {frames_dir}")
        return

    logger.info("Starting feature extraction...")
    # Extract features
    extract_features(frames_dir, visual_feature_dir, emotion_feature_dir, gesture_feature_dir)
    logger.info("Feature extraction completed.")

if __name__ == "__main__":
    main()
