# FILE:TEST.PY
# filepath: /C:/Users/Admin/Rahul/islt_multi_modality_phoenix_dataset_scratch/src/test.py
from calendar import c
import os
import logging
import numpy as np
import json
import tensorflow as tf
from models.transformer import SignLanguageTransformer
from utils import (
    extract_video_frames,
    extract_visual_token,
    extract_emotion_token,
    extract_gesture_token,
    tokenize_sequence,
    load_annotations_from_excel,
    load_model_checkpoint,
    normalize_text
)
from evaluation import handle_oov_tokens
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import EfficientNetB0
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sacrebleu import CHRF
from jiwer import wer, compute_measures

# Initialize logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'test.log'), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_vocab(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as file:
        vocab = json.load(file)
    inverse_vocab = {int(index): str(word) for word, index in vocab.items()}
    return vocab, inverse_vocab

def evaluate_on_test(model, test_excel, video_dir, vocab, inverse_vocab, checkpoint_dir):
    annotations = load_annotations_from_excel(test_excel)
    logger.info(f"Loaded {len(annotations)} test annotations from {test_excel}.")

    # Initialize feature extraction models
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    visual_projection_layer = Dense(512, activation=None, dtype='float32')
    emotion_dense_layer = Dense(512, activation=None, dtype='float32')
    gesture_dense_layer = Dense(512, activation=None, dtype='float32')

    # Metrics accumulators
    total_wer = 0.0
    total_chrf = 0.0
    total_rouge = 0.0
    total_bleu = 0.0
    total_samples = 0

    for video_name, annotation in annotations.items():
        try:
            video_path = os.path.join(video_dir, video_name + '.mp4')
            logger.info(f"Processing video: {video_path}")

            # Extract video frames
            frames = extract_video_frames(video_path)
            if not frames:
                logger.warning(f"No frames extracted from {video_path}. Skipping.")
                continue

            # Extract tokens for each modality
            visual_tokens = extract_visual_token(
                frames, efficientnet_model, visual_projection_layer,
                video_name, save_dir='data/features/visual'
            )
            if visual_tokens is None:
                logger.warning(f"No visual tokens for {video_name}. Skipping.")
                continue
            visual_tokens = tf.cast(visual_tokens, dtype=tf.float32)
            visual_tokens = tf.expand_dims(visual_tokens, axis=0)

            emotion_tokens = extract_emotion_token(
                frames, emotion_dense_layer,
                video_name, save_dir='data/features/emotion'
            )
            if emotion_tokens is None:
                logger.warning(f"No emotion tokens for {video_name}. Skipping.")
                continue
            emotion_tokens = tf.cast(emotion_tokens, dtype=tf.float32)
            emotion_tokens = tf.expand_dims(emotion_tokens, axis=0)

            gesture_tokens = extract_gesture_token(
                frames, gesture_dense_layer,
                video_name, save_dir='data/features/gesture'
            )
            if gesture_tokens is None:
                logger.warning(f"No gesture tokens for {video_name}. Skipping.")
                continue
            gesture_tokens = tf.cast(gesture_tokens, dtype=tf.float32)
            gesture_tokens = tf.expand_dims(gesture_tokens, axis=0)

            # Generate encoder output
            enc_output = model.encode_inputs(
                visual_tokens=visual_tokens,
                emotion_tokens=emotion_tokens,
                gesture_tokens=gesture_tokens,
                training=False
            )

            # Perform beam search decoding
            predictions = model.beam_search_decode(
                enc_output=enc_output,
                embedding_layer=model.gloss_embedding,
                decoder=model.gloss_decoder,
                output_layer=model.gloss_output_layer,
                max_length=50,
                start_token_id=model.start_token_id,
                end_token_id=model.end_token_id,
                beam_size=1,
                alpha=0.6,
                training=False
            )

            # Handle OOV tokens and decode
            decoded_sentences = model.handle_oov_tokens(predictions, max_length=50)
            if not decoded_sentences:
                logger.warning(f"No decoded sentences for {video_name}. Skipping.")
                continue
            hypothesis = decoded_sentences[0]
            reference = annotation['gloss']

            # Compute WER
            wer_score = wer(reference, hypothesis)

            # Compute CHRF
            chrf = CHRF().sentence_score(hypothesis, reference).score

            # Compute ROUGE-L
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_score = scorer.score(reference, hypothesis)['rougeL'].fmeasure * 100

            # Compute BLEU
            smoothie = SmoothingFunction().method4
            bleu = sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie) * 100

            logger.info(f"Video: {video_name} | WER: {wer_score:.2f}% | CHRF: {chrf:.2f} | ROUGE-L: {rouge_score:.2f}% | BLEU: {bleu:.2f}")

            # Accumulate metrics
            total_wer += wer_score
            total_chrf += chrf
            total_rouge += rouge_score
            total_bleu += bleu
            total_samples += 1

        except Exception as e:
            logger.error(f"Error processing video {video_name}: {e}")
            continue

    # Compute average metrics
    if total_samples > 0:
        avg_wer = total_wer / total_samples
        avg_chrf = total_chrf / total_samples
        avg_rouge = total_rouge / total_samples
        avg_bleu = total_bleu / total_samples
        logger.info(f"Test Evaluation Completed: {total_samples} samples")
        logger.info(f"Average WER: {avg_wer:.2f}%")
        logger.info(f"Average CHRF: {avg_chrf:.2f}")
        logger.info(f"Average ROUGE-L: {avg_rouge:.2f}%")
        logger.info(f"Average BLEU: {avg_bleu:.2f}%")
    else:
        logger.warning("No samples were evaluated.")

def main():
    vocab_file = os.path.join('data', 'vocab.json')
    test_excel = os.path.join('data', 'test.xlsx')
    video_dir = os.path.join('data', '')

    # Load vocabulary
    vocab, inverse_vocab = load_vocab(vocab_file)
    logger.info(f"Loaded vocab with {len(vocab)} entries.")

    # Initialize the Transformer model
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
        pad_token_id=vocab.get('<PAD>', 0),
        max_positional_encoding=1000
    )

    # ========== Add Model Building Here ==========
    # Define dummy input shapes based on your model's expected input
    dummy_visual_tokens = tf.zeros((1, 100, 512), dtype=tf.float32)
    dummy_emotion_tokens = tf.zeros((1, 100, 512), dtype=tf.float32)
    dummy_gesture_tokens = tf.zeros((1, 100, 512), dtype=tf.float32)

    # Perform a dummy forward pass to build the model
    try:
        _ = model.encode_inputs(
            visual_tokens=dummy_visual_tokens,
            emotion_tokens=dummy_emotion_tokens,
            gesture_tokens=dummy_gesture_tokens,
            training=False
        )
        logger.info("Model built successfully with dummy inputs.")
    except Exception as e:
        logger.error(f"Error during model building with dummy inputs: {e}")
        raise e
    # =============================================

    checkpoint_dir = 'checkpointsfull'
    weight_file = os.path.join(checkpoint_dir, 'model_epoch_best.weights.h5')
    if os.path.exists(weight_file):
        try:
            model.load_weights(weight_file, by_name=True, skip_mismatch=True)
            logger.info(f"Loaded model weights from: {weight_file}")
        except Exception as e:
            logger.error(f"Error loading weights from {weight_file}: {e}")
            raise e
    else:
        logger.error(f"No checkpoint found at {weight_file}. Please train the model before testing.")
        exit(1)

    # Perform evaluation on the test set
    evaluate_on_test(model, test_excel, video_dir, vocab, inverse_vocab, checkpoint_dir)

if __name__ == "__main__":
    main()