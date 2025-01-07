# evaluation.py

import os
import logging
import time
import tensorflow as tf
from utils import (
    extract_video_frames,
    extract_visual_token,
    extract_emotion_token,
    extract_gesture_token,
    tokenize_sequence,
    load_annotations_from_excel,
    normalize_text
)
from models.transformer import SignLanguageTransformer
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.chrf_score import sentence_chrf  # Import the function
import jiwer

logger = logging.getLogger(__name__)
def load_dev_data(dev_excel, video_dir, vocab):
    """
    Loads and preprocesses the development data.

    Args:
        dev_excel (str): Path to the development Excel file containing annotations.
        video_dir (str): Directory containing video files.
        vocab (dict): Vocabulary mapping for tokenization.

    Returns:
        list: A list of tuples containing processed inputs and targets.
    """
    annotations = load_annotations_from_excel(dev_excel)
    dev_data = []

    # Initialize the layers similar to training
    efficientnet_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet', include_top=False, pooling='avg'
    )
    visual_projection_layer = tf.keras.layers.Dense(512, activation=None)
    emotion_dense_layer = tf.keras.layers.Dense(512, activation=None)
    gesture_dense_layer = tf.keras.layers.Dense(512, activation=None)

    for video_name, annotation in annotations.items():
        video_path = os.path.join(video_dir, video_name + '.mp4')
        frames = extract_video_frames(video_path)
        if not frames:
            continue  # Skip if no frames are extracted

        visual_tokens = extract_visual_token(
            frames, efficientnet_model, visual_projection_layer,
            video_name, save_dir='data/features/visual'
        )
        emotion_tokens = extract_emotion_token(
            frames, emotion_dense_layer,
            video_name, save_dir='data/features/emotion'
        )
        gesture_tokens = extract_gesture_token(
            frames, gesture_dense_layer,
            video_name, save_dir='data/features/gesture'
        )

        # Ensure the tokens have the correct dimensions
        visual_tokens = tf.expand_dims(visual_tokens, axis=0) if len(visual_tokens.shape) == 2 else visual_tokens
        emotion_tokens = tf.expand_dims(emotion_tokens, axis=0) if len(emotion_tokens.shape) == 2 else emotion_tokens
        gesture_tokens = tf.expand_dims(gesture_tokens, axis=0) if len(gesture_tokens.shape) == 2 else gesture_tokens

        gloss_target = tokenize_sequence(annotation['gloss'], vocab, max_seq_len=50)
        text_target = tokenize_sequence(annotation['text'], vocab, max_seq_len=50)
        gloss_target = tf.expand_dims(gloss_target, axis=0) if len(gloss_target.shape) == 1 else gloss_target
        text_target = tf.expand_dims(text_target, axis=0) if len(text_target.shape) == 1 else text_target
        gloss_target = tf.cast(gloss_target, dtype=tf.int32)
        text_target = tf.cast(text_target, dtype=tf.int32)

        dev_data.append((
            {
                'visual_tokens': visual_tokens,
                'emotion_tokens': emotion_tokens,
                'gesture_tokens': gesture_tokens,
                'gloss_target': gloss_target,
                'text_target': text_target
            },
            {
                'gloss_output': gloss_target,
                'text_output': text_target
            }
        ))

    return dev_data

def evaluate(model, dev_excel, video_dir, vocab, epoch, step):
    # Load and preprocess your development data
    dev_data = load_dev_data(dev_excel, video_dir, vocab)
    
    # Generate predictions
    gloss_predictions = model.predict(dev_data)
    
    # Convert to tensor if it's a list
    if isinstance(gloss_predictions, list):
        gloss_predictions = tf.convert_to_tensor(gloss_predictions)
    
    # Ensure the tensor has the expected shape
    # Example: (batch_size, sequence_length, vocab_size)
    if len(gloss_predictions.shape) < 3:
        logger.error("gloss_predictions does not have the expected number of dimensions.")
        return
    
    # Reshape predictions for loss computation or other operations
    gloss_predictions_reshaped = tf.reshape(gloss_predictions[:, :-1], [-1, gloss_predictions.shape[-1]])
    
    # Similarly, reshape targets if necessary
    gloss_targets = tf.convert_to_tensor(load_gloss_targets(dev_excel, vocab))
    gloss_targets_reshaped = tf.reshape(gloss_targets[:, :-1], [-1])
    
    # Compute loss (example using Sparse Categorical Crossentropy)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = loss_fn(gloss_targets_reshaped, gloss_predictions_reshaped)
    
    logger.info(f"Validation result at epoch {epoch}, step {step} - Loss: {loss.numpy()}")
    
    return loss.numpy()

def compute_wer(references, hypotheses):
    """
    Computes the average Word Error Rate (WER) between references and hypotheses.
    """
    wers = []
    for ref, hyp in zip(references, hypotheses):
        ref_str = ' '.join(ref)
        hyp_str = ' '.join(hyp)
        error = jiwer.wer(ref_str, hyp_str)
        wers.append(error)
    avg_wer = np.mean(wers) * 100  # Convert to percentage
    return avg_wer

def get_alignment(reference, hypothesis):
    """
    Generates an alignment string showing matches (spaces) and mismatches (S for substitution, I for insertion, D for deletion).
    """
    # Using a simple alignment algorithm (edit distance)
    import difflib
    matcher = difflib.SequenceMatcher(None, reference, hypothesis)
    alignment = []
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            alignment.extend([' ' for _ in range(a1 - a0)])
        elif opcode == 'replace':
            alignment.extend(['S' for _ in range(a1 - a0)])
        elif opcode == 'insert':
            alignment.extend(['I' for _ in range(b1 - b0)])
        elif opcode == 'delete':
            alignment.extend(['D' for _ in range(a1 - a0)])
    return ' '.join(alignment)

def evaluate_beam_search(model, dev_excel, video_dir, vocab, beam_sizes=[1,2,3,4,5], alpha_values=[0.6], max_length=50):
    """
    Evaluates the model using beam search decoding with various beam sizes and length penalty values.
    """
    logger.info("Starting beam search evaluation")
    annotations = load_annotations_from_excel(dev_excel)
    id_to_word = {v: k for k, v in vocab.items()}
    start_token_id = vocab['<START>']
    end_token_id = vocab['<END>']

    # Initialize models and layers once
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    visual_projection_layer = Dense(512, activation=None)

    # Initialize Dense layers for emotion and gesture tokens
    emotion_dense_layer = Dense(512, activation=None)
    gesture_dense_layer = Dense(512, activation=None)

    # Define feature directories
    visual_feature_dir = 'data/features/visual'
    emotion_feature_dir = 'data/features/emotion'
    gesture_feature_dir = 'data/features/gesture'

    for beam_size in beam_sizes:
        for alpha in alpha_values:
            logger.info(f"Evaluating with beam size {beam_size} and alpha {alpha}")
            text_references = []
            text_hypotheses = []

            for video_name, annotation in annotations.items():
                video_path = os.path.join(video_dir, video_name + '.mp4')

                # Extract frames from video
                frames = extract_video_frames(video_path)
                if not frames:
                    continue

                # Extract or load visual tokens
                visual_tokens = extract_visual_token(
                    frames, efficientnet_model, visual_projection_layer,
                    video_name, save_dir=visual_feature_dir
                )
                if visual_tokens is None:
                    continue

                # Extract or load emotion tokens
                emotion_tokens = extract_emotion_token(
                    frames, emotion_dense_layer,
                    video_name, save_dir=emotion_feature_dir
                )
                if emotion_tokens is None:
                    continue

                # Extract or load gesture tokens
                gesture_tokens = extract_gesture_token(
                    frames, gesture_dense_layer,
                    video_name, save_dir=gesture_feature_dir
                )
                if gesture_tokens is None:
                    continue

                # Ensure batch dimension for tokens
                if len(visual_tokens.shape) == 2:
                    visual_tokens = tf.expand_dims(visual_tokens, axis=0)
                if len(emotion_tokens.shape) == 2:
                    emotion_tokens = tf.expand_dims(emotion_tokens, axis=0)
                if len(gesture_tokens.shape) == 2:
                    gesture_tokens = tf.expand_dims(gesture_tokens, axis=0)

                # Beam search decoding
                predicted_ids = model.predict_translation(
                    visual_tokens=visual_tokens,
                    emotion_tokens=emotion_tokens,
                    gesture_tokens=gesture_tokens,
                    max_length=max_length,
                    start_token_id=start_token_id,
                    end_token_id=end_token_id,
                    beam_size=beam_size,
                    alpha=alpha
                )

                # Convert predicted IDs to words
                predicted_tokens = [id_to_word.get(token_id, '<UNK>') for token_id in predicted_ids]

                # Reference translation
                translation = annotation['text']
                translation = normalize_text(translation.lower())
                text_ref_words = translation.split()

                text_references.append([text_ref_words])  # Note the extra list for BLEU
                text_hypotheses.append(predicted_tokens)

            # Compute evaluation metrics
            # Text WER
            text_wer = compute_wer([ref[0] for ref in text_references], text_hypotheses)

            # BLEU score
            bleu_scores = []
            smoothing_fn = SmoothingFunction().method1
            for ref, hyp in zip(text_references, text_hypotheses):
                score = sentence_bleu(ref, hyp, smoothing_function=smoothing_fn)
                bleu_scores.append(score)
            avg_bleu = np.mean(bleu_scores) * 100  # Convert to percentage

            logger.info(f"Beam size {beam_size}, alpha {alpha}: Text WER: {text_wer:.2f}, BLEU-4: {avg_bleu:.2f}")

def final_evaluation(model, dev_excel, video_dir, vocab, checkpoint_dir):
    """
    Performs final evaluation on the development set using the best model checkpoint.
    """
    # Load the best model checkpoint
    best_checkpoint = os.path.join(checkpoint_dir, 'model_epoch_best.weights.h5')
    model.load_weights(best_checkpoint)
    logger.info(f"Loaded best model checkpoint from {best_checkpoint}")

    # Evaluate on the development set
    logger.info("Final Evaluation on Development Set:")
    evaluate(model, dev_excel, video_dir, vocab, epoch=0, step=0)
