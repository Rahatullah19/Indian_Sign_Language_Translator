import tensorflow as tf
from models.transformer_test import SignLanguageTransformer
from utils_test import (
    extract_video_frames,
    extract_visual_token,
    extract_emotion_token,
    extract_gesture_token,
    load_annotations_from_excel
)
from models.custom_layers_test import create_attention_masks_encoder_only, create_attention_masks_decoder
import json
import os
import logging

logger = logging.getLogger(__name__)

def load_vocab(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as file:
        vocab = json.load(file)
    inverse_vocab = {int(index): str(word) for word, index in vocab.items()}  
    return vocab, inverse_vocab

# Function to initialize the model and load weights
def initialize_model():
    vocab_file = os.path.join('data', 'vocab.json')
    vocab, inverse_vocab = load_vocab(vocab_file)
    model = SignLanguageTransformer(
        visual_dim=512,
        emotion_dim=512,
        gesture_dim=512,
        gloss_vocab_size=len(vocab),
        text_vocab_size=len(vocab),
        inverse_vocab=inverse_vocab,
        num_layers=2,
        num_heads=8,
        ff_dim=512,
        dropout_rate=0.1,
        start_token_id=1,
        end_token_id=2,
        pad_token_id=0,
        max_positional_encoding=5000
    )

    # Dummy inputs to build the model
    visual_tokens = tf.random.uniform((1, 10, 512), dtype=tf.float32)
    emotion_tokens = tf.random.uniform((1, 10, 512), dtype=tf.float32)
    gesture_tokens = tf.random.uniform((1, 10, 512), dtype=tf.float32)
    gloss_target = tf.random.uniform((1, 10), maxval=4000, dtype=tf.int32)
    text_target = tf.random.uniform((1, 10), maxval=4000, dtype=tf.int32)

    batch_size = tf.shape(visual_tokens)[0]
    encoder_seq_len = tf.shape(visual_tokens)[1]  # 10
    decoder_seq_len = tf.shape(gloss_target)[1]   # 10

    # Create masks
    masks = {
        'encoder_padding_mask': create_attention_masks_encoder_only(seq_len=encoder_seq_len, batch_size=batch_size),
        'combined_mask': create_attention_masks_decoder(seq_len_decoder=decoder_seq_len, pad_token_id=0, batch_size=batch_size),
        'cross_attention_mask': tf.ones((batch_size, 1, decoder_seq_len, encoder_seq_len), dtype=tf.float32)
    }

    # Call the model to build it
    gloss_logits, text_logits = model(
        visual_tokens=visual_tokens,
        emotion_tokens=emotion_tokens,
        gesture_tokens=gesture_tokens,
        gloss_target=gloss_target,
        text_target=text_target,
        training=False,
        masks=masks
    )

    # Print model summary
    model.summary()

    # Load pre-trained weights by layer name
    try:
        model.load_weights('checkpoints/model_epoch_best.weights.h5', by_name=True)
        logger.info("Model weights loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        raise

    return model

# Function to process a single video and generate predictions
def predict_single_video(model, video_path, video_name, inverse_vocab, max_length=50, pad_token_id=0):
    """
    Predicts gloss and text translations for a single video.

    Args:
        model: The trained model to use for prediction.
        video_path (str): Path to the video file.
        video_name (str): Name of the video file.
        inverse_vocab (dict): Inverse vocabulary mapping token IDs to words.
        max_length (int): Maximum length for the prediction sequence.
        pad_token_id (int): Padding token ID for attention masks.

    Returns:
        tuple: gloss_translation (list), text_translation (list)
    """
    try:
        # Initialize modality-specific layers
        efficientnet_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
        visual_projection_layer = tf.keras.layers.Dense(512, activation=None)
        emotion_dense_layer = tf.keras.layers.Dense(512, activation=None)
        gesture_dense_layer = tf.keras.layers.Dense(512, activation=None)

        # Extract frames from the video
        frames = extract_video_frames(video_path)
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")

        # Extract tokens for each modality
        visual_tokens = extract_visual_token(
            frames, efficientnet_model, visual_projection_layer, video_name,
            save_dir='data/features/visual'
        )
        emotion_tokens = extract_emotion_token(
            frames, emotion_dense_layer, video_name,
            save_dir='data/features/emotion'
        )
        gesture_tokens = extract_gesture_token(
            frames, gesture_dense_layer, video_name,
            save_dir='data/features/gesture'
        )

        # Ensure tokens are valid
        if visual_tokens is None or emotion_tokens is None or gesture_tokens is None:
            raise ValueError("Failed to extract tokens from the video.")

        # Debug: Print token shapes
        logger.debug(f"Visual tokens shape: {visual_tokens.shape if visual_tokens is not None else 'None'}")
        logger.debug(f"Emotion tokens shape: {emotion_tokens.shape if emotion_tokens is not None else 'None'}")
        logger.debug(f"Gesture tokens shape: {gesture_tokens.shape if gesture_tokens is not None else 'None'}")

        # Add batch dimension
        visual_tokens = tf.expand_dims(visual_tokens, axis=0)  # Shape: (1, seq_len, 512)
        emotion_tokens = tf.expand_dims(emotion_tokens, axis=0)
        gesture_tokens = tf.expand_dims(gesture_tokens, axis=0)

        # Prepare masks with correct sequence lengths
        encoder_seq_len = tf.shape(visual_tokens)[1]  # e.g., 181
        decoder_seq_len = max_length  # e.g., 50

        encoder_padding_mask = create_attention_masks_encoder_only(
            seq_len=encoder_seq_len, batch_size=1
        )

        combined_mask = create_attention_masks_decoder(
            seq_len_decoder=decoder_seq_len,
            pad_token_id=pad_token_id,
            batch_size=1
        )

        cross_attention_mask = tf.ones(
            (1, 1, decoder_seq_len, encoder_seq_len), dtype=tf.float32
        )

        masks = {
            'encoder_padding_mask': encoder_padding_mask,  # (1, 1, 1, seq_len)
            'combined_mask': combined_mask,                # (1, 1, max_length, max_length)
            'cross_attention_mask': cross_attention_mask   # (1, 1, max_length, seq_len)
        }

        logger.info(f"Masks: {masks}")

        # Predict translations using the model
        gloss_translation, text_translation = model.predict_translation(
            visual_tokens=visual_tokens,
            emotion_tokens=emotion_tokens,
            gesture_tokens=gesture_tokens,
            masks=masks,
            inverse_vocab=inverse_vocab,
            max_length=max_length
        )

        return gloss_translation, text_translation

    except Exception as e:
        logger.error(f"Error predicting for video {video_name}: {e}")
        raise

# Function to iterate over videos in the annotation file
def process_videos_from_annotations(model, annotation_file, video_dir, vocab, inverse_vocab):
    annotations = load_annotations_from_excel(annotation_file)

    for video_name, annotation in annotations.items():
        video_path = os.path.join(video_dir, f"{video_name}.mp4")

        try:
            logger.info(f"Processing video: {video_name}")

            # Generate predictions
            gloss_translation, text_translation = predict_single_video(
                model, video_path, video_name, inverse_vocab
            )

            # Print predictions
            logger.info(f"Gloss Prediction: {gloss_translation}")
            logger.info(f"Text Prediction: {text_translation}")

            # Compare predictions with ground truth
            logger.info(f"Ground Truth Gloss: {annotation['gloss']}")
            logger.info(f"Ground Truth Text: {annotation['text']}")
            logger.info("-" * 50)

        except Exception as e:
            logger.error(f"Error processing video {video_name}: {e}")

# Main function to run the prediction process
def main():
    # Initialize the model
    model = initialize_model()

    # Paths and configuration
    annotation_file = "data/test.xlsx"  # Path to your annotation file
    video_dir = "data"                  # Directory containing video files
    vocab_file = "data/vocab.json"      # Path to vocabulary file

    # Load vocabulary and inverse vocabulary
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    inverse_vocab = {int(v): k for k, v in vocab.items()}

    # Update the model's inverse_vocab
    model.inverse_vocab = inverse_vocab

    # Process videos from the annotation file
    process_videos_from_annotations(model, annotation_file, video_dir, vocab, inverse_vocab)

if __name__ == "__main__":
    main()