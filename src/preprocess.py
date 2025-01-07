import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import EfficientNetB0
from utils_train import (
    load_annotations_from_excel,
    extract_video_frames,
    extract_visual_token,
    extract_emotion_token,
    extract_gesture_token,
)
from models.custom_layers_train import create_padding_mask, create_look_ahead_mask, create_attention_masks

# Initialize logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'preprocess.log'), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocess_features(train_excel, video_dir):
    annotations = load_annotations_from_excel(train_excel)

    # Initialize feature extraction models
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    visual_projection_layer = Dense(512, activation=None, dtype='float32')   
    emotion_dense_layer = Dense(512, activation=None, dtype='float32')      
    gesture_dense_layer = Dense(512, activation=None, dtype='float32')        

    # Define directories
    visual_feature_dir = 'data/features/visual'
    emotion_feature_dir = 'data/features/emotion'
    gesture_feature_dir = 'data/features/gesture'

    # Ensure directories exist
    for directory in [visual_feature_dir, emotion_feature_dir, gesture_feature_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

    # Preprocess and save features
    for video_name in annotations.keys():
        try:
            logger.debug(f"Preprocessing video: {video_name}")
            video_path = os.path.join(video_dir, f"{video_name}.mp4")

            # Check if features already exist
            visual_path = os.path.join(visual_feature_dir, f"{video_name}.npy")
            emotion_path = os.path.join(emotion_feature_dir, f"{video_name}.npy")
            gesture_path = os.path.join(gesture_feature_dir, f"{video_name}.npy")

            if os.path.exists(visual_path) and os.path.exists(emotion_path) and os.path.exists(gesture_path):
                logger.info(f"Features already extracted for {video_name}. Skipping extraction.")
                continue

            # Extract video frames
            frames = extract_video_frames(video_path)
            if not frames:
                logger.warning(f"No frames extracted from {video_path}. Skipping.")
                continue

            # Extract tokens for each modality
            visual_tokens = extract_visual_token(
                frames, efficientnet_model, visual_projection_layer=None,
                video_name=video_name, save_dir=None
            )
            if visual_tokens is None:
                logger.warning(f"Skipping video {video_path} due to no valid visual tokens.")
                continue
            logger.info(f"Visual Tokens Shape: {visual_tokens.shape}")
            np.save(visual_path, visual_tokens.numpy())

            emotion_tokens = extract_emotion_token(
                frames, emotion_dense_layer,
                video_name=video_name, save_dir=None
            )
            if emotion_tokens is None:
                logger.warning(f"Skipping video {video_path} due to no valid emotion tokens.")
                continue
            logger.info(f"Emotion Tokens Shape: {emotion_tokens.shape}")
            np.save(emotion_path, emotion_tokens.numpy())

            gesture_tokens = extract_gesture_token(
                frames, gesture_dense_layer,
                video_name=video_name, save_dir=None
            )
            if gesture_tokens is None:
                logger.warning(f"Skipping video {video_path} due to no valid gesture tokens.")
                continue
            logger.info(f"Gesture Tokens Shape: {gesture_tokens.shape}")
            np.save(gesture_path, gesture_tokens.numpy())

            logger.info(f"Features saved for video: {video_name}")

        except Exception as e:
            logger.error(f"Error preprocessing video {video_name}: {e}")
            continue
    logger.info("Feature extraction and saving completed.")

def main():
    train_excel = os.path.join('data', 'Train1.xlsx')
    video_dir = os.path.join('data', '')
    preprocess_features(train_excel, video_dir)

if __name__ == "__main__":
    main()