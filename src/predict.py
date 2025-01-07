# predict.py

import os
import logging
import numpy as np
import tensorflow as tf
import json
from models.transformer_test_video import SignLanguageTransformer
from utils import (
    extract_video_frames,
    extract_visual_token,
    extract_emotion_token,
    extract_gesture_token,
    load_annotations_from_excel, 
)
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import EfficientNetB0

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'test.log'), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Disable mixed precision for debugging
tf.keras.mixed_precision.set_global_policy("float32")

def load_video_names(test_excel):
    annotations = load_annotations_from_excel(test_excel)
    video_names = list(annotations.keys())
    return video_names

def load_vocab(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as file:
        vocab = json.load(file)
    inverse_vocab = {int(index): word for word, index in vocab.items()}
    
    # Ensure special tokens are present
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    for token in special_tokens:
        assert token in vocab, f"{token} not found in vocab."
    return vocab, inverse_vocab

def handle_oov_tokens(predictions, inverse_vocab):
    predicted_words = []
    logger.debug(f"Predicted Token IDs: {predictions}")
    for word_idx in predictions:
        word_idx_int = int(word_idx)
        word = inverse_vocab.get(word_idx_int, '<UNK>')
        logger.debug(f"Token ID: {word_idx_int}, Word: {word}")
        if word in ('<PAD>', '<EOS>', '<SOS>'):
            continue  # Skip <PAD>, <EOS>, and <SOS> tokens
        predicted_words.append(word)
    logger.debug(f"Predicted Words: {predicted_words}")
    return predicted_words

def validate_vocab(vocab, inverse_vocab):
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    for token in special_tokens:
        assert token in vocab, f"{token} not found in vocab."
        assert vocab[token] in inverse_vocab, f"Inverse mapping for {token} is missing."
    logger.info("Vocabulary validation passed.")

def log_model_summary(model, logger):
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_str = stream.getvalue()
    logger.info(f"Model Summary:\n{summary_str}")

def test(model, video_names, video_dir, inverse_vocab):
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    visual_projection_layer = Dense(512, activation=None, name='visual_projection')
    emotion_dense_layer = Dense(512, activation=None, name='emotion_dense')
    gesture_dense_layer = Dense(512, activation=None, name='gesture_dense')

    for video_name in video_names:
        video_path = os.path.join(video_dir, video_name + '.mp4')
        logger.info(f"Processing video: {video_path}")

        frames = extract_video_frames(video_path)
        if not frames:
            logger.warning(f"No frames extracted from {video_path}. Skipping.")
            continue

        visual_tokens = extract_visual_token(frames, efficientnet_model, visual_projection_layer, video_name)
        emotion_tokens = extract_emotion_token(frames, emotion_dense_layer, video_name)
        gesture_tokens = extract_gesture_token(frames, gesture_dense_layer, video_name)

        tokens_list = [visual_tokens, emotion_tokens, gesture_tokens]
        for i, token in enumerate(tokens_list):
            if token is None:
                token_name = ["visual", "emotion", "gesture"][i]
                logger.error(f"{token_name} tokens could not be generated for {video_name}.")
                continue
            if len(token.shape) == 2:
                token = tf.expand_dims(token, axis=0)
            tokens_list[i] = token

        visual_tokens, emotion_tokens, gesture_tokens = tokens_list

        logger.debug(f"Visual Tokens Shape: {visual_tokens.shape}")
        logger.debug(f"Emotion Tokens Shape: {emotion_tokens.shape}")
        logger.debug(f"Gesture Tokens Shape: {gesture_tokens.shape}")

        # Model inference: obtain predicted token sequences
        gloss_pred_ids, text_pred_ids = model(
            visual_tokens=visual_tokens,
            emotion_tokens=emotion_tokens,
            gesture_tokens=gesture_tokens,
            training=False
        )

        # Convert token IDs to words
        gloss_pred_ids = gloss_pred_ids.numpy()
        text_pred_ids = text_pred_ids.numpy()

        logger.debug(f"Raw Gloss Prediction IDs: {gloss_pred_ids}")
        logger.debug(f"Raw Text Prediction IDs: {text_pred_ids}")

        gloss_words = handle_oov_tokens(gloss_pred_ids, inverse_vocab)
        text_words = handle_oov_tokens(text_pred_ids, inverse_vocab)

        gloss_prediction = ' '.join(gloss_words)
        text_prediction = ' '.join(text_words)

        # Logging predictions
        logger.info(f"Video: {video_name}")
        logger.info(f"Gloss Prediction: {gloss_prediction}")
        logger.info(f"Text Prediction: {text_prediction}")
        logger.info("-" * 50)

def main():
    vocab_file = os.path.join('data', 'vocab.json') 
    vocab, inverse_vocab = load_vocab(vocab_file)  
    validate_vocab(vocab, inverse_vocab)

    # Initialize the model before logging the summary
    if 0 not in inverse_vocab:
        inverse_vocab[0] = '<PAD>'

    start_token_id = vocab.get('<SOS>', 1)  # Adjust as per your vocab
    end_token_id = vocab.get('<EOS>', 2)    # Adjust as per your vocab

    test_excel = os.path.join('data', 'test.xlsx')  
    video_dir = os.path.join('data', '')  

    video_names = load_video_names(test_excel)

    # Ensure these parameters match those used during training
    num_layers = 4     # Set to the value used during training
    num_heads = 8      # Set to the value used during training
    ff_dim = 512       # Set to the value used during training
    max_length = 50    # Set to the value used during training

    model = SignLanguageTransformer(
        visual_dim=512,
        emotion_dim=512,
        gesture_dim=512,
        gloss_vocab_size=len(vocab),
        text_vocab_size=len(vocab),
        start_token_id=start_token_id,
        end_token_id=end_token_id,
        max_length=max_length,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim
    )

    # Create dummy inputs with correct shapes
    seq_len = max_length  # Adjust as per your sequence length
    sample_visual_tokens = tf.zeros((1, seq_len, 512), dtype='float32')
    sample_emotion_tokens = tf.zeros((1, seq_len, 512), dtype='float32')
    sample_gesture_tokens = tf.zeros((1, seq_len, 512), dtype='float32')

    # Call the model to initialize variables and build the model
    _ = model(
        visual_tokens=sample_visual_tokens,
        emotion_tokens=sample_emotion_tokens,
        gesture_tokens=sample_gesture_tokens,
        training=False
    )

    # Log the model summary after the model has been built
    log_model_summary(model, logger)

    weights_path = 'checkpoints10/model_epoch_10.weights.h5'  
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path, by_name=True)
            logger.info("All weights successfully loaded.")
            logger.info(f"Loaded model weights from {weights_path}")
            
            # Verify weight loading for each layer
            for layer in model.layers:
                if layer.weights:
                    logger.debug(f"Layer '{layer.name}' loaded with weights.")
                else:
                    logger.warning(f"Layer '{layer.name}' has no weights loaded.")
            
        except tf.errors.ResourceExhaustedError as e:
            logger.error("Error loading weights. Details:")
            logger.error(e)
            return
        except AssertionError as e:
            logger.error("Some weights were not consumed during loading.")
            logger.error(e)
            return
    else:
        logger.error(f"No weights file found at {weights_path}")
        return

    logger.info("Starting inference on test data.")
    test(model, video_names, video_dir, inverse_vocab)

    # Optional: Test with dummy input (if necessary)
    # test_dummy_input(model, inverse_vocab)

if __name__ == "__main__":
    main()