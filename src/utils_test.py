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

# Load Mediapipe models once
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

def sanitize_filename(filename):
    """
    Replaces path separators and invalid characters in the filename.
    """
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    filename = ''.join(c for c in filename if ord(c) < 128)
    filename = filename.strip()
    return filename

# Function to load annotations from an Excel file
def load_annotations_from_excel(excel_file):
    logger.info(f"Loading annotations from: {excel_file}")
    try:
        data = pd.read_excel(excel_file)
    except Exception as e:
        logger.error(f"Failed to read Excel file {excel_file}: {e}")
        raise

    required_columns = {'name', 'gloss', 'text'}
    if not required_columns.issubset(data.columns):
        logger.error(f"Excel file must contain columns: {required_columns}")
        raise ValueError(f"Missing columns in Excel file. Required columns: {required_columns}")

    annotations = {}
    for idx, row in data.iterrows():
        video_name = row.get('name')
        gloss = row.get('gloss')
        text = row.get('text')

        if pd.isna(video_name) or not isinstance(video_name, str):
            logger.warning(f"Row {idx} has invalid or missing 'video_name'. Skipping.")
            continue

        video_name = video_name.strip()
        if not video_name:
            logger.warning(f"Row {idx} has empty 'video_name'. Skipping.")
            continue

        annotations[video_name] = {
            'gloss': gloss if not pd.isna(gloss) else '',
            'text': text if not pd.isna(text) else ''
        }

    logger.info(f"Loaded {len(annotations)} annotations from {excel_file}.")
    return annotations

# Function to load video files and extract frames
def extract_video_frames(video_path):
    logger.info(f"Extracting frames from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
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
    # Ensure at least one valid frame is processed
    if not frames:
        logger.warning(f"No frames to process for video: {video_name}")
        return None

    logger.info(f"Extracting visual tokens from frames for video: {video_name}")

    safe_video_name = sanitize_filename(video_name)
    feature_path = os.path.join(save_dir, f"{safe_video_name}.npy")

    if os.path.exists(feature_path):
        logger.info(f"Loading existing visual features from: {feature_path}")
        visual_tokens_projected = np.load(feature_path)
        return tf.convert_to_tensor(visual_tokens_projected, dtype=tf.float32)

    processed_frames = []

    for idx, frame in enumerate(frames):
        if frame is None:
            logger.warning(f"Frame {idx} is None. Skipping.")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(frame)

    if len(processed_frames) == 0:
        logger.warning(f"No valid frames after processing for video: {video_name}")
        return None

    processed_frames = np.array(processed_frames)
    processed_frames = preprocess_input(processed_frames)
    visual_features = model.predict(processed_frames, verbose=0)
    visual_tokens_projected = projection_layer(visual_features)

    os.makedirs(save_dir, exist_ok=True)
    np.save(feature_path, visual_tokens_projected.numpy())
    logger.info(f"Saved visual features to {feature_path}")

    return tf.convert_to_tensor(visual_tokens_projected, dtype=tf.float32)

# Function to extract emotion tokens using Mediapipe Face Mesh and project to 512 dimensions
def extract_emotion_token(frames, dense_layer, video_name, save_dir='data/features/emotion'):
    # Ensure at least one valid frame is processed
    if not frames:
        logger.warning(f"No frames to process for video: {video_name}")
        return None

    logger.info(f"Extracting emotion tokens from frames for video: {video_name}")

    safe_video_name = sanitize_filename(video_name)
    feature_path = os.path.join(save_dir, f"{safe_video_name}.npy")

    if os.path.exists(feature_path):
        logger.info(f"Loading existing emotion features from: {feature_path}")
        emotion_tokens_projected = np.load(feature_path)
        return tf.convert_to_tensor(emotion_tokens_projected, dtype=tf.float32)

    emotion_tokens = []

    for idx, frame in enumerate(frames):
        if frame is None:
            logger.warning(f"Frame {idx} is None. Skipping.")
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            # Example: Use average of landmarks as emotion feature
            landmarks = results.multi_face_landmarks[0].landmark
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            emotion_tokens.append(coords)
        else:
            # If no face detected, append zeros
            emotion_tokens.append(np.zeros(468 * 3, dtype=np.float32))

    if len(emotion_tokens) == 0:
        logger.warning(f"No valid emotion tokens after processing for video: {video_name}")
        return None

    emotion_tokens = np.array(emotion_tokens)
    flattened_tokens = emotion_tokens.reshape(emotion_tokens.shape[0], -1)
    emotion_tokens_projected = dense_layer(flattened_tokens)

    os.makedirs(save_dir, exist_ok=True)
    np.save(feature_path, emotion_tokens_projected.numpy())
    logger.info(f"Saved emotion features to {feature_path}")

    return tf.convert_to_tensor(emotion_tokens_projected, dtype=tf.float32)

# Function to extract gesture tokens using Mediapipe Hands and project to 512 dimensions
def extract_gesture_token(frames, dense_layer, video_name, save_dir='data/features/gesture'):
    # Ensure at least one valid frame is processed
    if not frames:
        logger.warning(f"No frames to process for video: {video_name}")
        return None

    logger.info(f"Extracting gesture tokens from frames for video: {video_name}")

    safe_video_name = sanitize_filename(video_name)
    feature_path = os.path.join(save_dir, f"{safe_video_name}.npy")

    if os.path.exists(feature_path):
        logger.info(f"Loading existing gesture features from: {feature_path}")
        gesture_tokens_projected = np.load(feature_path)
        return tf.convert_to_tensor(gesture_tokens_projected, dtype=tf.float32)

    gesture_tokens = []

    for idx, frame in enumerate(frames):
        if frame is None:
            logger.warning(f"Frame {idx} is None. Skipping.")
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            # Example: Use average of landmarks from all hands as gesture feature
            landmarks = [landmark for hand in results.multi_hand_landmarks for landmark in hand.landmark]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            gesture_tokens.append(coords)
        else:
            # If no hands detected, append zeros
            gesture_tokens.append(np.zeros(21 * 3 * 2, dtype=np.float32))  # 21 landmarks for 2 hands

    if len(gesture_tokens) == 0:
        logger.warning(f"No valid gesture tokens after processing for video: {video_name}")
        return None

    gesture_tokens = np.array(gesture_tokens)
    flattened_tokens = gesture_tokens.reshape(gesture_tokens.shape[0], -1)
    gesture_tokens_projected = dense_layer(flattened_tokens)

    os.makedirs(save_dir, exist_ok=True)
    np.save(feature_path, gesture_tokens_projected.numpy())
    logger.info(f"Saved gesture features to {feature_path}")

    return tf.convert_to_tensor(gesture_tokens_projected, dtype=tf.float32)

# Function to tokenize text
def tokenize_sequence(text, vocab, max_seq_len=400):
    tokens = ['<START>'] + text.split() + ['<END>']
    token_ids = []
    for token in tokens:
        token_id = vocab.get(token, vocab.get('<OOV>', 0))
        token_ids.append(token_id)

    token_ids = token_ids[:max_seq_len] + [vocab.get('<PAD>', 0)] * max(0, max_seq_len - len(token_ids))

    return tf.constant(token_ids, dtype=tf.int32)

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

def replace_umlauts(text):
    import unicodedata
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if unicodedata.category(c) != 'Mn'
    )

# Function to save model checkpoints
def save_model_checkpoint(model, checkpoint_dir, epoch):
    """
    Saves the model checkpoint.

    Args:
        model (tf.keras.Model): The model to save.
        checkpoint_dir (str): Directory to save the checkpoint.
        epoch (int): Current epoch number.
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.ckpt")
    model.save_weights(checkpoint_path)
    logger.info(f"Saved model checkpoint at epoch {epoch} to {checkpoint_path}")

# Function to load a specific model checkpoint
def load_model_checkpoint(model, latest_checkpoint):
    """
    Loads the model checkpoint.

    Args:
        model (tf.keras.Model): The model to load weights into.
        latest_checkpoint (str): Path to the checkpoint.

    Returns:
        tf.keras.Model: The model with loaded weights.
    """
    if latest_checkpoint:
        model.load_weights(latest_checkpoint)
        logger.info(f"Loaded model weights from {latest_checkpoint}")
    else:
        logger.warning("No checkpoint found. Using randomly initialized weights.")
    return model