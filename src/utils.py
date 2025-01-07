# utils.py

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import mediapipe as mp
import json
import unicodedata
import logging
import pandas as pd
import string

logger = logging.getLogger(__name__)

# Load Mediapipe models once
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

def sanitize_filename(filename):
    """
    Replaces path separators and invalid characters in the filename.
    """
    # Replace path separators and other invalid characters with underscores
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    # Remove non-ASCII characters
    filename = ''.join(c for c in filename if ord(c) < 128)
    # Remove leading and trailing whitespaces
    filename = filename.strip()
    return filename

# Function to load annotations from an Excel file
def load_annotations_from_excel(excel_file):
    logger.info(f"Loading annotations from: {excel_file}")
    data = pd.read_excel(excel_file)
    annotations = {}
    for idx, row in data.iterrows():
        video_name = row['name']
        gloss = row['gloss']
        translation = row['text']
        annotations[video_name] = {"gloss": gloss, "text": translation}
    logger.info(f"Loaded {len(annotations)} annotations from {excel_file}.")

    # Log first training example
    if 'Train' in excel_file:
        first_key = next(iter(annotations))
        first_annotation = annotations[first_key]
        logger.info("First training example:")
        logger.info(f"Video: {first_key}")
        logger.info(f"Gloss: {first_annotation['gloss']}")
        logger.info(f"Translation: {first_annotation['text']}")

    return annotations

# Function to load video files and extract frames
def extract_video_frames(video_path):
    logger.info(f"Extracting frames from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        logger.warning(f"Failed to open video file: {video_path}")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    return frames

# Function to extract visual tokens from frames using EfficientNetB0
def extract_visual_token(frames, model, projection_layer, video_name, save_dir='data/features/visual'):
    logger.info(f"Extracting visual tokens from frames for video: {video_name}")

    # Sanitize video_name to prevent directory traversal
    safe_video_name = sanitize_filename(video_name)

    feature_path = os.path.join(save_dir, f"{safe_video_name}.npy")

    if os.path.exists(feature_path):
        logger.info(f"Loading visual features from {feature_path}")
        visual_tokens_projected = np.load(feature_path)
        visual_tokens_projected = tf.convert_to_tensor(visual_tokens_projected)
        return visual_tokens_projected

    processed_frames = []

    for idx, frame in enumerate(frames):
        if frame is None:
            logger.warning(f"Warning: Frame {idx} is None. Skipping.")
            continue
        try:
            img = cv2.resize(frame, (224, 224))  # Resize to EfficientNet input size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            processed_frames.append(img)
        except Exception as e:
            logger.warning(f"Error processing frame {idx}: {e}. Skipping.")
            continue

    if len(processed_frames) == 0:
        logger.warning("No valid frames to process for visual tokens.")
        return None  # Or handle appropriately

    processed_frames = np.array(processed_frames)
    processed_frames = preprocess_input(processed_frames)  # Preprocess input

    # Predict features for all frames at once
    visual_features = model.predict(processed_frames, verbose=0)  # Shape: (num_frames, 1280)

    # Project features to 512 dimensions
    visual_tokens_projected = projection_layer(visual_features)  # Shape: (num_frames, 512)

    # Ensure the directory exists before saving
    os.makedirs(save_dir, exist_ok=True)

    # Save the features to disk
    np.save(feature_path, visual_tokens_projected.numpy())
    logger.info(f"Saved visual features to {feature_path}")

    return visual_tokens_projected  # Shape: (num_frames, 512)

# Function to extract emotion tokens using Mediapipe Face Mesh and project to 512 dimensions
def extract_emotion_token(frames, dense_layer, video_name, save_dir='data/features/emotion'):
    logger.info(f"Extracting emotion tokens from frames for video: {video_name}")

    # Sanitize video_name
    safe_video_name = sanitize_filename(video_name)

    feature_path = os.path.join(save_dir, f"{safe_video_name}.npy")

    if os.path.exists(feature_path):
        logger.info(f"Loading emotion features from {feature_path}")
        emotion_tokens_projected = np.load(feature_path)
        emotion_tokens_projected = tf.convert_to_tensor(emotion_tokens_projected)
        return emotion_tokens_projected

    emotion_tokens = []

    for idx, frame in enumerate(frames):
        if frame is None:
            logger.warning(f"Warning: Frame {idx} is None. Skipping.")
            continue
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            # Default to zeros if no face is detected
            landmarks = np.zeros((468, 3), dtype=np.float16)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark], dtype=np.float16)

            emotion_tokens.append(landmarks)
        except Exception as e:
            logger.warning(f"Error processing frame {idx} for emotion tokens: {e}. Skipping.")
            continue

    if len(emotion_tokens) == 0:
        logger.warning("No valid frames to process for emotion tokens.")
        return None

    emotion_tokens = np.array(emotion_tokens)  # Shape: (num_frames, 468, 3)

    # Flatten the landmarks to feed into the Dense layer
    num_frames = emotion_tokens.shape[0]
    flattened_tokens = emotion_tokens.reshape(num_frames, -1)  # Shape: (num_frames, 468*3)

    # Project to 512 dimensions
    emotion_tokens_projected = dense_layer(flattened_tokens)  # Shape: (num_frames, 512)

    # Ensure the directory exists before saving
    os.makedirs(save_dir, exist_ok=True)

    # Save the features to disk
    np.save(feature_path, emotion_tokens_projected.numpy())
    logger.info(f"Saved emotion features to {feature_path}")

    return emotion_tokens_projected  # Shape: (num_frames, 512)

# Function to extract gesture tokens using Mediapipe Hands and project to 512 dimensions
def extract_gesture_token(frames, dense_layer, video_name, save_dir='data/features/gesture'):
    logger.info(f"Extracting gesture tokens from frames for video: {video_name}")

    # Sanitize video_name
    safe_video_name = sanitize_filename(video_name)

    feature_path = os.path.join(save_dir, f"{safe_video_name}.npy")

    if os.path.exists(feature_path):
        logger.info(f"Loading gesture features from {feature_path}")
        gesture_tokens_projected = np.load(feature_path)
        gesture_tokens_projected = tf.convert_to_tensor(gesture_tokens_projected)
        return gesture_tokens_projected

    gesture_tokens = []

    for idx, frame in enumerate(frames):
        if frame is None:
            logger.warning(f"Warning: Frame {idx} is None. Skipping.")
            continue
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Initialize array for a fixed number of landmarks
            landmarks = np.zeros((42, 3), dtype=np.float16)  # 21 landmarks per hand, 2 hands

            if results.multi_hand_landmarks:
                landmark_index = 0
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        if landmark_index < 42:
                            landmarks[landmark_index] = [lm.x, lm.y, lm.z]
                            landmark_index += 1

            gesture_tokens.append(landmarks)
        except Exception as e:
            logger.warning(f"Error processing frame {idx} for gesture tokens: {e}. Skipping.")
            continue

    if len(gesture_tokens) == 0:
        logger.warning("No valid frames to process for gesture tokens.")
        return None

    gesture_tokens = np.array(gesture_tokens)  # Shape: (num_frames, 42, 3)

    # Flatten the landmarks to feed into the Dense layer
    num_frames = gesture_tokens.shape[0]
    flattened_tokens = gesture_tokens.reshape(num_frames, -1)  # Shape: (num_frames, 42*3)

    # Project to 512 dimensions
    gesture_tokens_projected = dense_layer(flattened_tokens)  # Shape: (num_frames, 512)

    # Ensure the directory exists before saving
    os.makedirs(save_dir, exist_ok=True)

    # Save the features to disk
    np.save(feature_path, gesture_tokens_projected.numpy())
    logger.info(f"Saved gesture features to {feature_path}")

    return gesture_tokens_projected  # Shape: (num_frames, 512)

# Function to load vocabulary from a JSON file
def load_vocab(vocab_file):
    """
    Loads the vocabulary and creates an inverse vocabulary.

    Args:
        vocab_file (str): Path to the vocabulary JSON file.

    Returns:
        tuple: A tuple containing the vocabulary dict and inverse vocabulary dict.
    """
    with open(vocab_file, 'r', encoding='utf-8') as file:
        vocab = json.load(file)
    
    # Ensure that the keys in vocab are strings that can be converted to integers
    inverse_vocab = {int(index): word for word, index in vocab.items()}
    
    return vocab, inverse_vocab


# Updated normalize_text function
def normalize_text(text):
    # Remove any non-printable characters, but keep diacritics
    text = ''.join(c for c in text if c.isprintable())
    return text

# Function to tokenize text (convert text into token indices)
def tokenize_sequence(text, vocab, max_seq_len=400):
    tokens = ['<START>'] + text.split() + ['<END>']
    token_ids = []
    for token in tokens:
        token_id = vocab.get(token, vocab.get('<OOV>', 0))
        if isinstance(token_id, int):
            token_ids.append(token_id)
        else:
            logger.warning(f"Token ID for '{token}' is not an integer. Using <OOV> token ID.")
            token_ids.append(vocab.get('<OOV>', 0))

    if len(token_ids) == 0:
        logger.error("Token IDs list is empty after tokenization.")
        token_ids = [vocab.get('<OOV>', 0)]

    token_ids = token_ids[:max_seq_len] + [vocab.get('<PAD>', 0)] * max(0, max_seq_len - len(token_ids))

    return tf.constant(token_ids, dtype=tf.int32)  # Ensure int32

def replace_umlauts(text):
    replacements = {
        'ä': 'ae',
        'ö': 'oe',
        'ü': 'ue',
        'ß': 'ss'
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text

# Function to save model checkpoints
def save_model_checkpoint(model, checkpoint_dir, epoch):
    """
    Saves the model at the given epoch as a checkpoint.
    """
    try:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            logger.info(f"Created checkpoint directory at {checkpoint_dir}")
        # Use .weights.h5 extension
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.weights.h5")
        model.save_weights(checkpoint_path)
        logger.info(f"Model checkpoint saved at {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error saving model checkpoint: {e}")

# Function to load a specific model checkpoint
def load_model_checkpoint(model, latest_checkpoint):
    """
    Loads the model checkpoint from the specified epoch.
    """
    checkpoint_path = os.path.join('checkpoints', latest_checkpoint)
    model.load_weights(checkpoint_path)
    logger.info(f"Loaded model from {checkpoint_path}")

