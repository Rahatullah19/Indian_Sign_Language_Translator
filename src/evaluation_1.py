import os
import logging
import time
from turtle import st
import numpy as np
import tensorflow as tf
from models.transformer_1 import SignLanguageTransformer
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from utils_1 import (
    load_annotations_from_excel,
    load_model_checkpoint,
    normalize_text,
    tokenize_sequence,
    sanitize_filename
)
from models.custom_layers_1 import create_padding_mask, create_look_ahead_mask, create_attention_masks, handle_oov_tokens, generate_alignment
from jiwer import wer, compute_measures
from sacrebleu.metrics import CHRF
from sacrebleu import sentence_chrf
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import difflib

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
    pred = tf.cast(pred, dtype=tf.float32)  # Ensure predictions are float32 for loss computation
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_fn(real, pred)  

    pad_token_id = 0
    end_token_id = 2
    end_weight = 1.0  # Reduced from 2.0 to 1.0

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
    # logger.debug(f"Reference text: '{reference}', Hypothesis text: '{hypothesis}'")

    if not reference.strip() or not hypothesis.strip():
        return {
            'wer': 100.0 if not reference.strip() else 0.0,
            'sub_rate': 100.0 if not reference.strip() else 0.0,
            'del_rate': 100.0 if not reference.strip() else 0.0,
            'ins_rate': 100.0 if not reference.strip() else 0.0
        }

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

def compute_chrf_and_rouge(text_references, text_hypotheses):
    chrf_scores = []
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for ref, hyp in zip(text_references, text_hypotheses):
    
        # Ensure ref and hyp are strings
        if isinstance(ref, list):
            ref = ' '.join([str(r) for r in ref])
            logger.debug(f"Converted reference list to string: {ref}")
        if isinstance(hyp, list):
            hyp = ' '.join([str(h) for h in hyp])
            logger.debug(f"Converted hypothesis list to string: {hyp}")

        if not ref.strip() or not hyp.strip():
            #logger.warning("Empty reference or hypothesis detected. Skipping CHRF and ROUGE calculations for this pair.")
            continue

        chrf = sentence_chrf(hyp, [ref]).score
        chrf_scores.append(chrf)

        rouge = scorer.score(ref, hyp)
        rouge_scores.append(rouge['rougeL'].fmeasure * 100)

    avg_chrf = np.mean(chrf_scores) if chrf_scores else 0.0
    avg_rouge = np.mean(rouge_scores) if rouge_scores else 0.0
    return {
        'CHRF': avg_chrf,
        'ROUGE-L': avg_rouge
    }


def compute_bleu_scores_batch(references, hypotheses):

    smoothie = SmoothingFunction().method4
    bleu_scores = [0.0, 0.0, 0.0, 0.0]  # BLEU-1 to BLEU-4
    num_samples = len(hypotheses)

    for ref, hyp in zip(references, hypotheses):
        # Ensure ref and hyp are strings
        if isinstance(ref, list):
            ref = ' '.join([str(r) for r in ref])
            logger.debug(f"Converted reference list to string: {ref}")
        if isinstance(hyp, list):
            hyp = ' '.join([str(h) for h in hyp])
            logger.debug(f"Converted hypothesis list to string: {hyp}")

        if not ref.strip() or not hyp.strip():
            #logger.warning("Empty reference or hypothesis detected. Skipping BLEU score for this pair.")
            continue

        reference_tokens = ref.split()
        hypothesis_tokens = hyp.split()

        try:
            # Calculate BLEU scores
            bleu1 = sentence_bleu([reference_tokens], hypothesis_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
            bleu2 = sentence_bleu([reference_tokens], hypothesis_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
            bleu3 = sentence_bleu([reference_tokens], hypothesis_tokens, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smoothie)
            bleu4 = sentence_bleu([reference_tokens], hypothesis_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

            bleu_scores[0] += bleu1
            bleu_scores[1] += bleu2
            bleu_scores[2] += bleu3
            bleu_scores[3] += bleu4
        except Exception as e:
            logger.error(f"Error computing BLEU scores for reference: '{ref}' and hypothesis: '{hyp}': {e}")
            continue

    if num_samples > 0:
        avg_bleu_scores = [score / num_samples * 100 for score in bleu_scores]
    else:
        avg_bleu_scores = [0.0, 0.0, 0.0, 0.0]
        logger.warning("No samples available to compute BLEU scores.")

    return avg_bleu_scores

def evaluate(model, dev_excel, video_dir, vocab, inverse_vocab, epoch, step, max_length=400):
    """
    Evaluates the model on the development set.

    Args:
        model (SignLanguageTransformer): The trained model.
        dev_excel (str): Path to the development Excel file.
        video_dir (str): Directory containing videos.
        vocab (dict): Vocabulary mapping.
        inverse_vocab (dict): Inverse vocabulary mapping.
        epoch (int): Current epoch number.
        step (int): Current training step.
        max_length (int, optional): Maximum length for generation. Defaults to 50.

    Returns:
        tuple: Average loss and WER.
    """
    # Type-safe logging for epoch and step
    if isinstance(epoch, int) and isinstance(step, int):
        epoch_display = epoch
        step_display = step
    else:
        epoch_display = 'Unknown'
        step_display = 'Unknown'

    logger.info(f"Validation result at epoch {epoch_display}, step {step_display}")
    start_time = time.time()

    annotations = load_annotations_from_excel(dev_excel)
    total_wer = 0.0
    total_wer_g = 0.0
    total_loss = 0.0
    total_samples = 0

    gloss_references = []
    gloss_hypotheses = []
    text_references = []
    text_hypotheses = []
    gloss_alignments = []
    text_alignments = []

    logger.info("Starting evaluation...")

    # Prepare directories for feature storage
    visual_feature_dir = 'data/features/visual'
    emotion_feature_dir = 'data/features/emotion'
    gesture_feature_dir = 'data/features/gesture'

    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    for video_name, annotation in annotations.items():
        try:
            video_path = os.path.join(video_dir, f"{video_name}.mp4")
            video_name_sanitized = sanitize_filename(video_name)
            logger.info(f"Processing video: {video_path}")

            # Load pre-extracted features
            visual_path = os.path.join(visual_feature_dir, f"{video_name_sanitized}.npy")
            emotion_path = os.path.join(emotion_feature_dir, f"{video_name_sanitized}.npy")
            gesture_path = os.path.join(gesture_feature_dir, f"{video_name_sanitized}.npy")

            if not (os.path.exists(visual_path) and os.path.exists(emotion_path) and os.path.exists(gesture_path)):
                logger.warning(f"Feature files missing for {video_name_sanitized}. Skipping.")
                continue

            visual_tokens = np.load(visual_path)
            emotion_tokens = np.load(emotion_path)
            gesture_tokens = np.load(gesture_path)

            visual_tokens = tf.convert_to_tensor(visual_tokens, dtype=tf.float16)
            emotion_tokens = tf.convert_to_tensor(emotion_tokens, dtype=tf.float16)
            gesture_tokens = tf.convert_to_tensor(gesture_tokens, dtype=tf.float16)

            visual_tokens = tf.expand_dims(visual_tokens, axis=0)  
            emotion_tokens = tf.expand_dims(emotion_tokens, axis=0)  
            gesture_tokens = tf.expand_dims(gesture_tokens, axis=0)  

            # Check for NaN or Inf in tokens
            if tf.reduce_any(tf.math.is_nan(visual_tokens)) or tf.reduce_any(tf.math.is_inf(visual_tokens)):
                logger.error(f"Invalid visual tokens for video {video_name_sanitized}")
                continue

            if tf.reduce_any(tf.math.is_nan(emotion_tokens)) or tf.reduce_any(tf.math.is_inf(emotion_tokens)):
                logger.error(f"Invalid emotion tokens for video {video_name_sanitized}")
                continue

            if tf.reduce_any(tf.math.is_nan(gesture_tokens)) or tf.reduce_any(tf.math.is_inf(gesture_tokens)):
                logger.error(f"Invalid gesture tokens for video {video_name_sanitized}")
                continue

            # Convert annotation gloss/text to string if needed
            gloss_text = annotation['gloss']
            if isinstance(gloss_text, list):
                gloss_text = ' '.join(gloss_text)

            text_text = annotation['text']
            if isinstance(text_text, list):
                text_text = ' '.join(text_text)

            # Add normalization
            gloss_text = normalize_text(gloss_text)
            text_text = normalize_text(text_text)

            # Tokenize and pad targets
            gloss_target = tokenize_sequence(gloss_text, vocab, max_seq_len=400)
            text_target = tokenize_sequence(text_text, vocab, max_seq_len=400)

            # Ensure batch dimension
            gloss_target = tf.expand_dims(gloss_target, axis=0)  # Shape: (1, 400)
            text_target = tf.expand_dims(text_target, axis=0)    # Shape: (1, 400)

            # Get batch size
            batch_size = tf.shape(visual_tokens)[0]

            # Shift gloss_target for decoder input and target
            gloss_decoder_input = gloss_target[:, :-1]  
            gloss_decoder_target = gloss_target[:, 1:] 

            # Shift text_target for decoder input and target
            text_decoder_input = text_target[:, :-1]   
            text_decoder_target = text_target[:, 1:]    

            # Add checks to ensure decoder inputs are not empty
            decoder_input_shape = tf.shape(gloss_decoder_input)[1]
            if decoder_input_shape <= 0:
                logger.error(f"Decoder input sequence length is zero for video {video_name_sanitized}. Skipping this sample.")
                continue

            masks = create_attention_masks(
                encoder_token_ids=tf.ones([batch_size, tf.shape(visual_tokens)[1]], dtype=tf.int32),
                decoder_token_ids=gloss_decoder_input,
                pad_token_id=tf.constant(vocab.get('<PAD>', 0), dtype=tf.int32),
                num_heads=model.num_heads
            )

            # Cast masks to float16
            masks = {k: tf.cast(v, tf.float16) for k, v in masks.items()}

            # **Add Logging and Assertions**
            logger.debug(f"Encoder Padding Mask Shape: {masks['encoder_padding_mask'].shape}")
            logger.debug(f"Combined Mask Shape: {masks['combined_mask'].shape}")
            logger.debug(f"Cross Attention Mask Shape: {masks['cross_attention_mask'].shape}")

            # Assert that encoder_seq_len matches enc_output_seq_len later in the code
            encoder_seq_len = tf.shape(visual_tokens)[1]
            logger.debug(f"Encoder Sequence Length: {encoder_seq_len}")

            # Example Assertion before passing masks to the model
            tf.debugging.assert_equal(
                tf.shape(masks['encoder_padding_mask'])[3],
                encoder_seq_len,
                message="Encoder padding mask's sequence length does not match encoder's output sequence length."
            )

            # Forward pass
            with tf.GradientTape() as tape:
                outputs = model(
                    visual_tokens=visual_tokens,
                    emotion_tokens=emotion_tokens,
                    gesture_tokens=gesture_tokens,
                    gloss_target=gloss_decoder_input,  
                    text_target=text_decoder_input,    
                    training=False,
                    masks=masks
                )
                
                # Ensure outputs are correctly returned
                if outputs is None or len(outputs) != 2:
                    logger.error(f"Model returned invalid outputs for video {video_name_sanitized}. Skipping.")
                    continue
                
                gloss_predictions, text_predictions = outputs

                # Cast predictions to float32 if they are not already
                gloss_predictions = tf.cast(gloss_predictions, tf.float32)
                text_predictions = tf.cast(text_predictions, tf.float32)

                # Compute loss
                gloss_loss = loss_function(gloss_decoder_target, gloss_predictions)
                text_loss = loss_function(text_decoder_target, text_predictions)
                total_loss_sample = gloss_loss + text_loss

            total_loss += total_loss_sample.numpy()
            total_samples += 1

            # Generate translations
            gloss_pred = model.predict_translation(
                visual_tokens,
                emotion_tokens=emotion_tokens,
                gesture_tokens=gesture_tokens,
                masks=masks,
                max_length=max_length,
                mode='gloss',
                return_token_ids=True
            )
            text_pred = model.predict_translation(
                visual_tokens,
                emotion_tokens=emotion_tokens,
                gesture_tokens=gesture_tokens,
                masks=masks,
                max_length=max_length,
                mode='text',
                return_token_ids=True
            )

            gloss_pred_text = handle_oov_tokens(gloss_pred, inverse_vocab)
            text_pred_text = handle_oov_tokens(text_pred, inverse_vocab)

            gloss_hypotheses.append(gloss_pred_text[0])
            gloss_references.append(gloss_text)
            text_hypotheses.append(text_pred_text[0])
            text_references.append(text_text)

            # Compute WER
            wer_gloss = compute_wer_detailed(gloss_text, gloss_pred_text[0])
            wer_text = compute_wer_detailed(text_text, text_pred_text[0])
            total_wer += wer_text['wer']
            total_wer_g += wer_gloss['wer']

            # Compute Alignments
            gloss_alignment = generate_alignment(gloss_text.split(), gloss_pred_text[0].split())
            text_alignment = generate_alignment(text_text.split(), text_pred_text[0].split())
            gloss_alignments.append(gloss_alignment)
            text_alignments.append(text_alignment)

            # Log references, hypotheses, and alignments
            logger.info(" ")
            logger.info("=" * 119)
            logger.info(f"Gloss Reference: {gloss_text}")
            logger.info(f"Gloss Hypothesis: {gloss_pred_text[0]}")
            logger.info(f"Gloss Alignment: {gloss_alignment}")
            logger.info("-" * 119)
            logger.info(f"Text Reference: {text_text}")
            logger.info(f"Text Hypothesis: {text_pred_text[0]}")
            logger.info(f"Text Alignment: {text_alignment}")
            logger.info("=" * 119)
            logger.info(" ")

            total_samples += 1

            logger.debug(f"Processed video: {video_name_sanitized}, Gloss WER: {wer_gloss}, Text WER: {wer_text}")

        except Exception as e:
            logger.error(f"Error evaluating video {video_name}: {e}")
            continue

    if total_samples == 0:
        logger.warning("No samples were evaluated. Unable to compute average metrics.")
        return None, None

    average_loss = total_loss / total_samples
    average_wer = total_wer / total_samples
    average_wer_g = total_wer_g / total_samples

    # Compute CHRF and ROUGE scores for Text
    chrf_and_rouge_text = compute_chrf_and_rouge(text_references, text_hypotheses)

    # Compute CHRF and ROUGE scores for Gloss
    chrf_and_rouge_gloss = compute_chrf_and_rouge(gloss_references, gloss_hypotheses)

    # Compute BLEU scores for Text
    bleu_scores_text = compute_bleu_scores_batch(text_references, text_hypotheses)

    # Compute BLEU scores for Gloss
    bleu_scores_gloss = compute_bleu_scores_batch(gloss_references, gloss_hypotheses)

    # Log aggregate metrics
    logger.info(" ")
    logger.info("*" * 119)
    logger.info(f"Validation Loss: {average_loss:.4f}")
    logger.info(f"Gloss WER: {average_wer_g:.2f}%")
    logger.info(f"Gloss CHRF Score: {chrf_and_rouge_gloss['CHRF']:.2f}, ROUGE-L Score: {chrf_and_rouge_gloss['ROUGE-L']:.2f}")
    logger.info(f"Gloss BLEU Scores (1-4 grams): BLEU-1: {bleu_scores_gloss[0]:.2f}, BLEU-2: {bleu_scores_gloss[1]:.2f}, BLEU-3: {bleu_scores_gloss[2]:.2f}, BLEU-4: {bleu_scores_gloss[3]:.2f}")
    logger.info(f"Text WER: {average_wer:.2f}%")
    logger.info(f"Text CHRF Score: {chrf_and_rouge_text['CHRF']:.2f}, ROUGE-L Score: {chrf_and_rouge_text['ROUGE-L']:.2f}")
    logger.info(f"Text BLEU Scores (1-4 grams): BLEU-1: {bleu_scores_text[0]:.2f}, BLEU-2: {bleu_scores_text[1]:.2f}, BLEU-3: {bleu_scores_text[2]:.2f}, BLEU-4: {bleu_scores_text[3]:.2f}")
    logger.info("*" * 119)
    logger.info(" ")

    end_time = time.time()
    logger.info(f"Evaluation Time Taken: {end_time - start_time:.2f} seconds")

    return average_loss, average_wer + average_wer_g

def evaluate_beam_search(
    model, dev_excel, video_dir, vocab, inverse_vocab,
    beam_sizes=[1, 2, 3, 4, 5],
    alpha_values=[0.6], max_length=50
):
    logger.info("Starting beam search evaluation.")
    annotations = load_annotations_from_excel(dev_excel)
    total_samples = len(annotations)
    logger.info(f"Total samples for beam search evaluation: {total_samples} from {dev_excel}.")

    visual_feature_dir = 'data/features/visual'
    emotion_feature_dir = 'data/features/emotion'
    gesture_feature_dir = 'data/features/gesture'

    for beam_size in beam_sizes:
        for alpha in alpha_values:
            logger.info(f"Evaluating with Beam Size: {beam_size}, Alpha: {alpha}")
            gloss_references = []
            gloss_hypotheses = []
            text_references = []
            text_hypotheses = []

            for video_name, annotation in annotations.items():
                try:
                    video_name_clean = sanitize_filename(video_name)
                    visual_path = os.path.join(visual_feature_dir, f"{video_name_clean}.npy")
                    emotion_path = os.path.join(emotion_feature_dir, f"{video_name_clean}.npy")
                    gesture_path = os.path.join(gesture_feature_dir, f"{video_name_clean}.npy")

                    if not (os.path.exists(visual_path) and os.path.exists(emotion_path) and os.path.exists(gesture_path)):
                        logger.warning(f"Feature files missing for {video_name}. Skipping.")
                        continue

                    # Load and convert features
                    visual_tokens = tf.convert_to_tensor(np.load(visual_path), dtype=tf.float16)
                    emotion_tokens = tf.convert_to_tensor(np.load(emotion_path), dtype=tf.float16)
                    gesture_tokens = tf.convert_to_tensor(np.load(gesture_path), dtype=tf.float16)
                    visual_tokens = tf.expand_dims(visual_tokens, axis=0)
                    emotion_tokens = tf.expand_dims(emotion_tokens, axis=0)
                    gesture_tokens = tf.expand_dims(gesture_tokens, axis=0)

                    # Ensure all tokens are float16
                    visual_tokens = tf.cast(visual_tokens, tf.float16)
                    emotion_tokens = tf.cast(emotion_tokens, tf.float16)
                    gesture_tokens = tf.cast(gesture_tokens, tf.float16)

                    # Prepare reference texts
                    gloss_ref = annotation['gloss']
                    if isinstance(gloss_ref, list):
                        gloss_ref = ' '.join(gloss_ref)
                    gloss_ref = normalize_text(gloss_ref)

                    text_ref = annotation['text']
                    if isinstance(text_ref, list):
                        text_ref = ' '.join(text_ref)
                    text_ref = normalize_text(text_ref)

                    # Encode inputs
                    combined_tokens = model.combine_inputs(visual_tokens, emotion_tokens, gesture_tokens)
                    enc_input = model.encoder_positional_encoding(combined_tokens)
                    enc_output, _ = model.encoder(enc_input, training=False, attention_mask=None)

                    # Beam search for gloss
                    gloss_pred_ids = model.beam_search_decode(
                        enc_output=enc_output,
                        embedding_layer=model.gloss_embedding,
                        decoder=model.gloss_decoder,
                        output_layer=model.gloss_output_layer,
                        max_length=max_length,
                        num_heads=model.num_heads,
                        start_token_id=vocab.get('<START>', 1),
                        end_token_id=vocab.get('<END>', 2),
                        beam_size=beam_size,
                        alpha=alpha,
                        training=False
                    )
                    gloss_pred = handle_oov_tokens([gloss_pred_ids], inverse_vocab, max_length=max_length)[0]

                    # Beam search for text
                    text_pred_ids = model.beam_search_decode(
                        enc_output=enc_output,
                        embedding_layer=model.text_embedding,
                        decoder=model.text_decoder,
                        output_layer=model.text_output_layer,
                        max_length=max_length,
                        num_heads=model.num_heads,
                        start_token_id=vocab.get('<START>', 1),
                        end_token_id=vocab.get('<END>', 2),
                        beam_size=beam_size,
                        alpha=alpha,
                        training=False
                    )
                    text_pred = handle_oov_tokens([text_pred_ids], inverse_vocab, max_length=max_length)[0]

                    # Collect references and hypotheses
                    gloss_references.append(gloss_ref)
                    gloss_hypotheses.append(gloss_pred)
                    text_references.append(text_ref)
                    text_hypotheses.append(text_pred)

                except Exception as e:
                    logger.error(f"Error during beam search evaluation for video {video_name}: {e}")
                    continue

            chrf_and_rouge_gloss = compute_chrf_and_rouge(gloss_references, gloss_hypotheses)
            chrf_and_rouge_text = compute_chrf_and_rouge(text_references, text_hypotheses)
            bleu_scores_gloss = compute_bleu_scores_batch(gloss_references, gloss_hypotheses)
            bleu_scores_text = compute_bleu_scores_batch(text_references, text_hypotheses)

            logger.info(f"Beam Size: {beam_size}, Alpha: {alpha}")
            logger.info(f"Gloss CHRF: {chrf_and_rouge_gloss['CHRF']:.2f}, ROUGE-L: {chrf_and_rouge_gloss['ROUGE-L']:.2f}")
            logger.info(
                f"Gloss BLEU (1-4): {bleu_scores_gloss[0]:.2f}, {bleu_scores_gloss[1]:.2f}, "
                f"{bleu_scores_gloss[2]:.2f}, {bleu_scores_gloss[3]:.2f}"
            )
            logger.info(f"Text CHRF: {chrf_and_rouge_text['CHRF']:.2f}, ROUGE-L: {chrf_and_rouge_text['ROUGE-L']:.2f}")
            logger.info(
                f"Text BLEU (1-4): {bleu_scores_text[0]:.2f}, {bleu_scores_text[1]:.2f}, "
                f"{bleu_scores_text[2]:.2f}, {bleu_scores_text[3]:.2f}"
            )
            logger.info("*" * 50)

def final_evaluation(model, dev_excel, video_dir, vocab, inverse_vocab, checkpoint_dir):
    logger.info("Loading the best model checkpoint for final evaluation.")

    load_model_checkpoint(model, 'model_epoch_best.weights.h5')
    
    logger.info("Performing final evaluation on the development set.")
    avg_loss, avg_wer = evaluate(model, dev_excel, video_dir, vocab, inverse_vocab, epoch='final', step='final')
    if avg_loss is not None and avg_wer is not None:
        logger.info(f"Final Evaluation Average Loss: {avg_loss:.2f}%")
        logger.info(f"Final Evaluation WER: {avg_wer:.2f}%")
    else:
        logger.warning("Final evaluation metrics are unavailable.")