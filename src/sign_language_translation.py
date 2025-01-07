
import os
import logging
import numpy as np
import tensorflow as tf
import json
from models.transformer import SignLanguageTransformer
from utils import (
    extract_video_frames,
    extract_visual_token,
    extract_emotion_token,
    extract_gesture_token,
    tokenize_sequence,
    load_annotations_from_excel,
)
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import EfficientNetB0

if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'test_video.log'), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

tf.keras.mixed_precision.set_global_policy("mixed_float16")

def load_test_data(test_excel):
    annotations = load_annotations_from_excel(test_excel)
    return annotations

def load_vocab(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as file:
        vocab = json.load(file)
    inverse_vocab = {int(index): word for word, index in vocab.items()}
    return vocab, inverse_vocab

def handle_oov_tokens(predictions, inverse_vocab):
    predicted_words = []
    for word_idx in predictions:
        word_idx_int = int(np.array(word_idx).item())
        word = inverse_vocab.get(word_idx_int, '<UNK>')
        if word == '<PAD>':
            break  
        predicted_words.append(word)
    return predicted_words

def test(model, test_excel, video_dir, vocab, inverse_vocab):
    annotations = load_test_data(test_excel)
    
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    visual_projection_layer = Dense(512, activation=None)
    emotion_dense_layer = Dense(512, activation=None)
    gesture_dense_layer = Dense(512, activation=None)

    for video_name, annotation in annotations.items():
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

        gloss_target = tokenize_sequence(annotation['gloss'], vocab, max_seq_len=50)
        text_target = tokenize_sequence(annotation['text'], vocab, max_seq_len=50)

        gloss_target = tf.expand_dims(gloss_target, axis=0)
        text_target = tf.expand_dims(text_target, axis=0)

        gloss_logits, text_logits = model(
            visual_tokens=visual_tokens,
            emotion_tokens=emotion_tokens,
            gesture_tokens=gesture_tokens,
            gloss_target=gloss_target,
            text_target=text_target,
            training=False
        )

        gloss_predictions = tf.argmax(gloss_logits, axis=-1).numpy().flatten()
        gloss_predicted_words = handle_oov_tokens(gloss_predictions, inverse_vocab)
        gloss_prediction = ' '.join(gloss_predicted_words)

        text_predictions = tf.argmax(text_logits, axis=-1).numpy().flatten()
        text_predicted_words = handle_oov_tokens(text_predictions, inverse_vocab)
        text_prediction = ' '.join(text_predicted_words)

        logger.info(f"Video: {video_name}")
        logger.info("=" * 80)
        logger.info(f"Gloss Reference: {annotation['gloss']}")
        logger.info(f"Gloss Prediction: {gloss_prediction}")
        logger.info("-" * 80)
        logger.info(f"Text Reference: {annotation['text']}")
        logger.info(f"Text Prediction: {text_prediction}")
        logger.info("=" * 80)

def main():
    vocab_file = os.path.join('data', 'vocab.json') 
    vocab, inverse_vocab = load_vocab(vocab_file)  

    if 0 not in inverse_vocab:
        inverse_vocab[0] = '<PAD>'

    test_excel = os.path.join('data', 'test_video.xlsx')  
    video_dir = os.path.join('data', '')  

    model = SignLanguageTransformer(
        visual_dim=512,
        emotion_dim=512,
        gesture_dim=512,
        gloss_vocab_size=len(vocab),
        text_vocab_size=len(vocab)
    )

    seq_len = 50  # Adjust as per your sequence length
    sample_visual_tokens = tf.zeros((1, seq_len, 512))
    sample_emotion_tokens = tf.zeros((1, seq_len, 512))
    sample_gesture_tokens = tf.zeros((1, seq_len, 512))
    sample_gloss_target = tf.zeros((1, seq_len), dtype=tf.int32)
    sample_text_target = tf.zeros((1, seq_len), dtype=tf.int32)

    _ = model(
        visual_tokens=sample_visual_tokens,
        emotion_tokens=sample_emotion_tokens,
        gesture_tokens=sample_gesture_tokens,
        gloss_target=sample_gloss_target,
        text_target=sample_text_target,
        training=False
    )

    weights_path = 'checkpoints/model_epoch_best.weights.h5'  
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        logger.info(f"Loaded model weights from {weights_path}")
    else:
        logger.error(f"No weights file found at {weights_path}")
        return

    logger.info("Starting inference on test data.")
    test(model, test_excel, video_dir, vocab, inverse_vocab)

if __name__ == "__main__":
    main()