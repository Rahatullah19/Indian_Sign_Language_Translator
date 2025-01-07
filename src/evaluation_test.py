
from turtle import st
from models.transformer import SignLanguageTransformer
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense
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
from models.custom_layers import create_padding_mask, create_look_ahead_mask, create_attention_masks
from jiwer import wer, compute_measures
from sacrebleu.metrics import CHRF
from sacrebleu import sentence_chrf
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import difflib
import numpy as np
import tensorflow as tf
import logging
import os
import time


logger = logging.getLogger(__name__)

def loss_function(real, pred):
    """
    Computes the masked Sparse Categorical Crossentropy loss with higher weighting for the <END> token.
    """
    tf.debugging.assert_rank(real, 2, message="real labels should be 2-dimensional")
    tf.debugging.assert_rank(pred, 3, message="predictions should be 3-dimensional")
    tf.debugging.assert_equal(
        tf.shape(real), tf.shape(pred)[:2], 
        message="labels and logits must have the same batch and sequence dimensions"
    )
    pred = tf.cast(pred, dtype=tf.float32)  
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_fn(real, pred)  

    pad_token_id = 0
    end_token_id = 2
    end_weight = 2.0

    mask = tf.cast(tf.not_equal(real, pad_token_id), dtype=loss_.dtype)
    end_weight_tensor = tf.where(
        tf.equal(real, end_token_id),
        tf.cast(end_weight, loss_.dtype),
        tf.cast(1.0, loss_.dtype)
    )
    loss_ *= mask * end_weight_tensor

    mask_sum = tf.reduce_sum(mask)
    loss = tf.math.divide_no_nan(tf.reduce_sum(loss_), mask_sum)

    tf.debugging.check_numerics(loss, 'Loss contains NaN')
    tf.debugging.assert_non_negative(mask_sum, message="mask_sum should be non-negative")
    loss = tf.cond(
        tf.equal(mask_sum, 0),
        lambda: tf.print("Mask sum is zero. No valid tokens to compute loss.") or loss,
        lambda: loss
    )
    tf.debugging.assert_all_finite(loss, 'Loss contains non-finite values')
    return loss

def compute_wer_detailed(reference, hypothesis):
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    logger.debug(f"Reference text: '{reference}', Hypothesis text: '{hypothesis}'")

    if not reference.strip() or not hypothesis.strip():
        logger.warning("Empty reference or hypothesis detected. This causes 100% WER.")

    measures = compute_measures(reference, hypothesis)
    wer_score = measures['wer'] * 100  
    substitutions = measures['substitutions']
    deletions = measures['deletions']
    insertions = measures['insertions']
    # Compute error rates
    total_words = measures['hits'] + substitutions + deletions
    sub_rate = (substitutions / total_words) * 100 if total_words > 0 else 0
    del_rate = (deletions / total_words) * 100 if total_words > 0 else 0
    ins_rate = (insertions / total_words) * 100 if total_words > 0 else 0
    return {
        'wer': wer_score,
        'sub_rate': sub_rate,
        'del_rate': del_rate,
        'ins_rate': ins_rate
    }


def handle_oov_tokens(predictions, inverse_vocab, max_length=50):
    if isinstance(predictions, (int, np.integer)):
        predictions = [[predictions]]

    if isinstance(predictions, tf.Tensor):
        predictions = tf.argmax(predictions, axis=-1).numpy()

    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()

    if isinstance(predictions, list) and all(isinstance(x, (int, np.integer)) for x in predictions):
        predictions = [predictions]

    predicted_sentences = []
    for seq in predictions:  
        tokens = []
        for token_id in seq:
            word = inverse_vocab.get(int(token_id), '<OOV>')
            if word not in ['<START>', '<END>', '<PAD>']:
                tokens.append(word)
        predicted_sentences.append(" ".join(tokens[:max_length]))

    return predicted_sentences

def generate_alignment(reference_tokens, hypothesis_tokens):
    """
    Generates alignment symbols between reference and hypothesis tokens.

    Args:
        reference_tokens (list): Ground truth tokens.
        hypothesis_tokens (list): Predicted tokens.

    Returns:
        str: Alignment symbols as a space-separated string.
    """
    align_symbols = []
    matcher = difflib.SequenceMatcher(None, reference_tokens, hypothesis_tokens)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            align_symbols.extend(['='] * (i2 - i1))
        elif tag == 'replace':
            align_symbols.extend(['R'] * max(i2 - i1, j2 - j1))
        elif tag == 'delete':
            align_symbols.extend(['D'] * (i2 - i1))
        elif tag == 'insert':
            align_symbols.extend(['I'] * (j2 - j1))

    # Ensure all alignment symbols are strings
    align_symbols = [str(symbol) for symbol in align_symbols]
    return ' '.join(align_symbols)


def compute_chrf_and_rouge(text_references, text_hypotheses):
    chrf_scores = []
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for ref, hyp in zip(text_references, text_hypotheses):
        if isinstance(ref, list):
            ref = ' '.join(ref)
        if isinstance(hyp, list):
            hyp = ' '.join(hyp)
        if not ref.strip() or not hyp.strip():
            continue
        chrf_score1 = sentence_chrf(hyp, [ref]).score
        chrf_scores.append(chrf_score1)
        rouge_score = scorer.score(ref, hyp)
        rouge_scores.append(rouge_score['rougeL'].fmeasure)

    avg_chrf = np.mean(chrf_scores) if chrf_scores else 0.0
    avg_rouge = np.mean(rouge_scores) if rouge_scores else 0.0
    return {
        'CHRF': avg_chrf,
        'ROUGE-L': avg_rouge
    }

def compute_bleu_scores_batch(references, hypotheses):
    smoothie = SmoothingFunction().method4
    bleu_scores = [0.0, 0.0, 0.0, 0.0]
    num_samples = len(hypotheses)

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        for i in range(1, 5):
            bleu = sentence_bleu([ref_tokens], hyp_tokens, weights=(1.0/i,)*i + (0.0,)*(4-i), smoothing_function=smoothie)
            bleu_scores[i-1] += bleu

    if num_samples > 0:
        bleu_scores = [(score / num_samples) * 100 for score in bleu_scores]
    else:
        bleu_scores = [0.0, 0.0, 0.0, 0.0]

    return bleu_scores

def evaluate(model, dev_excel, video_dir, vocab, inverse_vocab, epoch, step, max_length=50):
    
    # Type-safe logging for epoch and step
    if isinstance(epoch, int) and isinstance(step, int):
        epoch_display = epoch
        step_display = step
    else:
        epoch_display = 'N/A'
        step_display = 'N/A'

    logger.info(f"Validation result at epoch {epoch_display}, step {step_display}")
    start_time = time.time()

    annotations = load_annotations_from_excel(dev_excel)
    total_wer = 0.0
    total_loss = 0.0
    total_samples = 0

    gloss_references = []
    gloss_hypotheses = []
    text_references = []
    text_hypotheses = []
    sample_outputs = []

    logger.info("Starting evaluation...")
    # Initialize feature extraction models
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    visual_projection_layer = Dense(512, activation=None, dtype='float32')  # Ensure float32
    emotion_dense_layer = Dense(512, activation=None, dtype='float32')     # Ensure float32
    gesture_dense_layer = Dense(512, activation=None, dtype='float32')     # Ensure float32

    # Prepare directories for feature storage
    visual_feature_dir = 'data/features/visual'
    emotion_feature_dir = 'data/features/emotion'
    gesture_feature_dir = 'data/features/gesture'

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
                video_name, save_dir=visual_feature_dir
            )
            if visual_tokens is None:
                logger.warning(f"Skipping video {video_path} due to no valid visual tokens.")
                continue
            logger.info(f"Visual Tokens Shape: {visual_tokens.shape}")
            visual_tokens = tf.cast(visual_tokens, dtype=tf.float32)
            visual_tokens = tf.expand_dims(visual_tokens, axis=0)  # Add batch dimension

            emotion_tokens = extract_emotion_token(
                frames, emotion_dense_layer,
                video_name, save_dir=emotion_feature_dir
            )
            if emotion_tokens is None:
                logger.warning(f"Skipping video {video_path} due to no valid emotion tokens.")
                continue
            logger.info(f"Emotion Tokens Shape: {emotion_tokens.shape}")
            emotion_tokens = tf.cast(emotion_tokens, dtype=tf.float32)
            emotion_tokens = tf.expand_dims(emotion_tokens, axis=0)  # Add batch dimension

            gesture_tokens = extract_gesture_token(
                frames, gesture_dense_layer,
                video_name, save_dir=gesture_feature_dir
            )
            if gesture_tokens is None:
                logger.warning(f"Skipping video {video_path} due to no valid gesture tokens.")
                continue
            logger.info(f"Gesture Tokens Shape: {gesture_tokens.shape}")
            gesture_tokens = tf.cast(gesture_tokens, dtype=tf.float32)
            gesture_tokens = tf.expand_dims(gesture_tokens, axis=0)  # Add batch dimension

            if tf.reduce_any(tf.math.is_nan(visual_tokens)) or tf.reduce_any(tf.math.is_inf(visual_tokens)):
                logger.error(f"Invalid visual tokens for video {video_name}")
                continue

            if tf.reduce_any(tf.math.is_nan(emotion_tokens)) or tf.reduce_any(tf.math.is_inf(emotion_tokens)):
                logger.error(f"Invalid emotion tokens for video {video_name}")
                continue

            if tf.reduce_any(tf.math.is_nan(gesture_tokens)) or tf.reduce_any(tf.math.is_inf(gesture_tokens)):
                logger.error(f"Invalid gesture tokens for video {video_name}")
                continue

            # Convert annotation gloss/text to string if needed
            gloss_text = annotation.get('gloss', '')
            if isinstance(gloss_text, list):
                gloss_text = ' '.join(gloss_text)

            text_text = annotation.get('text', '')
            if isinstance(text_text, list):
                text_text = ' '.join(text_text)

            # Tokenize and pad targets
            gloss_target = tokenize_sequence(gloss_text, vocab, max_seq_len=400)
            text_target = tokenize_sequence(text_text, vocab, max_seq_len=400)

            # Ensure batch dimension
            gloss_target = tf.expand_dims(gloss_target, axis=0)  
            text_target = tf.expand_dims(text_target, axis=0)    

            batch_size = tf.shape(visual_tokens)[0]

            # Shift gloss_target for decoder input and target
            gloss_decoder_input = gloss_target[:, :-1] 
            gloss_decoder_target = gloss_target[:, 1:]  

            # Shift text_target for decoder input and target
            text_decoder_input = text_target[:, :-1]    
            text_decoder_target = text_target[:, 1:]    

            # **Add the following checks to ensure decoder inputs are not empty**
            decoder_input_shape = tf.shape(gloss_decoder_input)[1]
            if decoder_input_shape <= 0:
                logger.error(f"Decoder input sequence length is zero for video {video_name}. Skipping this sample.")
                continue

            # Use gloss_decoder_input for creating masks
            masks = create_attention_masks(
                encoder_token_ids=tf.ones([batch_size, tf.shape(visual_tokens)[1] * 3], dtype=tf.int32),
                decoder_token_ids=gloss_decoder_input,
                pad_token_id=tf.constant(vocab.get('<PAD>', 0), dtype=tf.int32)
            )

            # Forward pass
            gloss_predictions, text_predictions = model(
                visual_tokens=visual_tokens,
                emotion_tokens=emotion_tokens,
                gesture_tokens=gesture_tokens,
                gloss_target=gloss_decoder_input,  
                text_target=text_decoder_input,    
                training=False,
                masks=masks
            )
            
            gloss_predictions = tf.cast(gloss_predictions, tf.float32)
            text_predictions = tf.cast(text_predictions, tf.float32)

            gloss_loss = loss_function(gloss_decoder_target, gloss_predictions)
            text_loss = loss_function(text_decoder_target, text_predictions)
            total_loss += (gloss_loss + text_loss).numpy()

            # Compute predictions
            gloss_predicted_ids = tf.argmax(gloss_predictions, axis=-1).numpy()[0]
            text_predicted_ids = tf.argmax(text_predictions, axis=-1).numpy()[0]

            # Convert token IDs to words
            gloss_predicted = handle_oov_tokens(gloss_predicted_ids, inverse_vocab, max_length=max_length)[0]
            text_predicted = handle_oov_tokens(text_predicted_ids, inverse_vocab, max_length=max_length)[0]

            gloss_reference = gloss_text
            text_reference = text_text

            gloss_references.append(gloss_reference)
            text_references.append(text_reference)
            gloss_hypotheses.append(gloss_predicted)
            text_hypotheses.append(text_predicted)

            wer_scores = compute_wer_detailed(text_reference, text_predicted)
            total_wer += wer_scores['wer']
            total_samples += 1

            logger.debug(f"Processed video: {video_name}, WER: {wer_scores}")

        except Exception as e:
            logger.error(f"Error evaluating video {video_name}: {e}")
            continue

    if total_samples == 0:
        logger.error("No valid samples for evaluation.")
        return None, None

    average_loss = total_loss / total_samples
    average_wer = total_wer / total_samples

    chrf_and_rouge = compute_chrf_and_rouge(text_references, text_hypotheses)
    bleu_scores = compute_bleu_scores_batch(text_references, text_hypotheses)

    logger.info(f"Validation Loss: {average_loss:.4f}, WER: {average_wer:.2f}%")
    logger.info(f"CHRF Score: {chrf_and_rouge['CHRF']:.2f}, ROUGE-L Score: {chrf_and_rouge['ROUGE-L']:.2f}")
    logger.info(f"BLEU Scores (1-4 grams): {bleu_scores}")

    end_time = time.time()
    logger.info(f"Evaluation Time Taken: {end_time - start_time:.2f} seconds")

    return average_loss, average_wer

def evaluate_beam_search(
    model, dev_excel, video_dir, vocab, inverse_vocab,
    beam_sizes=[1, 2, 3, 4, 5],
    alpha_values=[0.6], max_length=50
):
    
    logger.info("Starting beam search evaluation.")
    annotations = load_annotations_from_excel(dev_excel)
    logger.info(f"Loaded {len(annotations)} annotations from {dev_excel}.")

    # Initialize feature extraction models
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    visual_projection_layer = Dense(512, activation=None, dtype='float32')  
    emotion_dense_layer = Dense(512, activation=None, dtype='float32')     
    gesture_dense_layer = Dense(512, activation=None, dtype='float32')    

    # Prepare directories for feature storage
    visual_feature_dir = 'data/features/visual'
    emotion_feature_dir = 'data/features/emotion'
    gesture_feature_dir = 'data/features/gesture'

    for beam_size in beam_sizes:
        for alpha in alpha_values:
            logger.info(f"Evaluating with beam size: {beam_size}, alpha: {alpha}")
            gloss_references = []
            gloss_hypotheses = []
            text_references = []
            text_hypotheses = []

            for video_name, annotation in annotations.items():
                try:
                    logger.debug(f"Processing video: {video_name}")
                    video_path = os.path.join(video_dir, video_name + '.mp4')

                    # Extract video frames
                    frames = extract_video_frames(video_path)
                    if not frames:
                        logger.warning(f"No frames extracted from {video_path}. Skipping.")
                        continue

                    # Extract tokens for each modality
                    visual_tokens = extract_visual_token(
                        frames, efficientnet_model, visual_projection_layer,
                        video_name, save_dir=visual_feature_dir
                    )
                    if visual_tokens is None:
                        logger.warning(f"Skipping video {video_path} due to no valid visual tokens.")
                        continue
                    logger.info(f"Visual Tokens Shape: {visual_tokens.shape}")
                    visual_tokens = tf.cast(visual_tokens, dtype=tf.float32)
                    visual_tokens = tf.expand_dims(visual_tokens, axis=0)  

                    emotion_tokens = extract_emotion_token(
                        frames, emotion_dense_layer,
                        video_name, save_dir=emotion_feature_dir
                    )
                    if emotion_tokens is None:
                        logger.warning(f"Skipping video {video_path} due to no valid emotion tokens.")
                        continue
                    logger.info(f"Emotion Tokens Shape: {emotion_tokens.shape}")
                    emotion_tokens = tf.cast(emotion_tokens, dtype=tf.float32)
                    emotion_tokens = tf.expand_dims(emotion_tokens, axis=0)  

                    gesture_tokens = extract_gesture_token(
                        frames, gesture_dense_layer,
                        video_name, save_dir=gesture_feature_dir
                    )
                    if gesture_tokens is None:
                        logger.warning(f"Skipping video {video_path} due to no valid gesture tokens.")
                        continue
                    logger.info(f"Gesture Tokens Shape: {gesture_tokens.shape}")
                    gesture_tokens = tf.cast(gesture_tokens, dtype=tf.float32)
                    gesture_tokens = tf.expand_dims(gesture_tokens, axis=0) 

                    # Tokenize targets
                    gloss_text = annotation['gloss']
                    text_text = annotation['text']

                    gloss_target = tokenize_sequence(gloss_text, vocab, max_seq_len=400)
                    text_target = tokenize_sequence(text_text, vocab, max_seq_len=400)

                    # Perform encoding
                    enc_output = model.encode_inputs(visual_tokens, emotion_tokens, gesture_tokens, training=False)

                    # Beam search decoding for gloss
                    gloss_translation = model.beam_search_decode(
                        enc_output=enc_output,
                        embedding_layer=model.gloss_embedding,
                        decoder=model.gloss_decoder,
                        output_layer=model.gloss_output_layer,
                        max_length=max_length,
                        start_token_id=model.start_token_id,
                        end_token_id=model.end_token_id,
                        beam_size=beam_size,
                        alpha=alpha,
                        training=False
                    )
                    gloss_translation = model.handle_oov_tokens(gloss_translation, max_length=max_length)

                    # Beam search decoding for text
                    text_translation = model.beam_search_decode(
                        enc_output=enc_output,
                        embedding_layer=model.text_embedding,
                        decoder=model.text_decoder,
                        output_layer=model.text_output_layer,
                        max_length=max_length,
                        start_token_id=model.start_token_id,
                        end_token_id=model.end_token_id,
                        beam_size=beam_size,
                        alpha=alpha,
                        training=False
                    )
                    text_translation = model.handle_oov_tokens(text_translation, max_length=max_length)

                    # Ensure references are strings
                    gloss_reference = str(gloss_text) if not isinstance(gloss_text, str) else gloss_text
                    text_reference = str(text_text) if not isinstance(text_text, str) else text_text

                    gloss_references.append(gloss_reference)
                    gloss_hypotheses.append(gloss_translation[0] if gloss_translation else "")
                    text_references.append(text_reference)
                    text_hypotheses.append(text_translation[0] if text_translation else "")

                except Exception as e:
                    logger.error(f"Error during beam search evaluation for video {video_name}: {e}")
                    continue

            try:
                chrf_and_rouge = compute_chrf_and_rouge(text_references, text_hypotheses)
                logger.info(f"CHRF Score: {chrf_and_rouge['CHRF']:.2f}, ROUGE-L Score: {chrf_and_rouge['ROUGE-L']:.2f}")
            except Exception as e:
                chrf_score = 0.0
                logger.error(f"CHRF & ROUGE-L computation failed: {e}")

            # Compute ROUGE
            try:
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                rouge_scores = [scorer.score(ref, hyp)['rougeL'].fmeasure for ref, hyp in zip(gloss_references, gloss_hypotheses)]
                avg_rouge = np.mean(rouge_scores) if rouge_scores else 0.0
                logger.info(f"Average ROUGE-L Score: {avg_rouge}")
            except Exception as e:
                logger.error(f"ROUGE computation failed: {e}")

            # Compute BLEU
            try:
                smoothie = SmoothingFunction().method4
                bleu_scores = [sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie) for ref, hyp in zip(gloss_references, gloss_hypotheses)]
                avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
                logger.info(f"Average BLEU Score: {avg_bleu}")
            except Exception as e:
                logger.error(f"BLEU computation failed: {e}")

            # Log the results
            logger.info(f"Beam Size: {beam_size}, Alpha: {alpha} | CHRF Score: {chrf_and_rouge['CHRF']:.2f}, ROUGE-L Score: {chrf_and_rouge['ROUGE-L']:.2f}, BLEU: {bleu_scores}")


def final_evaluation(model, dev_excel, video_dir, vocab, inverse_vocab, checkpoint_dir):
    logger.info("Loading the best model checkpoint for final evaluation.")

    load_model_checkpoint(model, 'model_epoch_best.weights.h5')
    
    logger.info("Performing final evaluation on the development set.")
    avg_loss, avg_wer = evaluate(model, dev_excel, video_dir, vocab, inverse_vocab, epoch='final', step='final')
    logger.info(f"Final Evaluation Average Loss: {avg_loss:.2f}%")
    logger.info(f"Final Evaluation WER: {avg_wer:.2f}%")
