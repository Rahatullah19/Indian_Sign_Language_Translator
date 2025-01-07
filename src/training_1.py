
import os
import logging
import time
import numpy as np
import tensorflow as tf

# Set GPU configurations before importing TensorFlow
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_GPU_OPERATION_LOGGING'] = '1'

import json
from models.transformer_1 import SignLanguageTransformer
from utils_1 import (
    load_model_checkpoint,
    save_model_checkpoint,
    load_annotations_from_excel,
    tokenize_sequence,
    normalize_text,
    extract_video_frames,
    extract_visual_token,
    extract_emotion_token,
    extract_gesture_token,
    sanitize_filename
)
from evaluation_1 import evaluate, evaluate_beam_search, final_evaluation
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import EfficientNetB0
from io import StringIO
from tensorflow.keras import mixed_precision


from models.custom_layers_1 import create_padding_mask, create_look_ahead_mask, create_attention_masks

# Enable device placement logging
tf.debugging.set_log_device_placement(True)

# Initialize logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'train_train.log'), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Num GPUs Available: %d", len(tf.config.list_physical_devices('GPU')))

# Enable mixed precision
mixed_precision.set_global_policy("mixed_float16")
logger.info(f"Mixed precision policy set to: {mixed_precision.global_policy()}")

# Verify that TensorFlow sees the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(gpus)}")
    except RuntimeError as e:
        logger.error(e)
else:
    logger.error("No GPUs found. Please ensure that TensorFlow is configured to use the GPU.")

def load_vocab(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as file:
        vocab = json.load(file)
    
    inverse_vocab = {int(index): str(word) for word, index in vocab.items()}  
    
    return vocab, inverse_vocab

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

def preprocess_and_save_features(annotations, video_dir, visual_model, visual_layer, emotion_layer, gesture_layer,
                                visual_dir, emotion_dir, gesture_dir):
    logger.info("Starting feature extraction and saving.")
    for video_name in annotations.keys():
        try:
            logger.debug(f"Preprocessing video: {video_name}")
            video_path = os.path.join(video_dir, f"{video_name}.mp4")

            video_name = sanitize_filename(video_name)

            # Check if features already exist
            visual_path = os.path.join(visual_dir, f"{video_name}.npy")
            emotion_path = os.path.join(emotion_dir, f"{video_name}.npy")
            gesture_path = os.path.join(gesture_dir, f"{video_name}.npy")

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
                frames, visual_model, visual_layer,
                video_name=video_name, save_dir=visual_dir
            )
            if visual_tokens is None:
                logger.warning(f"Skipping video {video_path} due to no valid visual tokens.")
                continue
            logger.info(f"Visual Tokens Shape: {visual_tokens.shape}")
            np.save(visual_path, visual_tokens.numpy())

            emotion_tokens = extract_emotion_token(
                frames, emotion_layer,
                video_name=video_name, save_dir=emotion_dir
            )
            if emotion_tokens is None:
                logger.warning(f"Skipping video {video_path} due to no valid emotion tokens.")
                continue
            logger.info(f"Emotion Tokens Shape: {emotion_tokens.shape}")
            np.save(emotion_path, emotion_tokens.numpy())

            gesture_tokens = extract_gesture_token(
                frames, gesture_layer,
                video_name=video_name, save_dir=gesture_dir
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

def train(model, train_excel, dev_excel, video_dir, vocab, inverse_vocab, epochs, learning_rate, patience=3):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    annotations = load_annotations_from_excel(train_excel)

    # Initialize feature extraction models
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    visual_projection_layer = Dense(512, activation=None, dtype='float32')  # Changed to float32
    emotion_dense_layer = Dense(512, activation=None, dtype='float32')    # Changed to float32
    gesture_dense_layer = Dense(512, activation=None, dtype='float32')    # Changed to float32

    logger.info("Building the model with specified input shapes.")

    # Preprocess and save all features before training
    visual_feature_dir = 'data/features/visual'
    emotion_feature_dir = 'data/features/emotion'
    gesture_feature_dir = 'data/features/gesture'

    for directory in [visual_feature_dir, emotion_feature_dir, gesture_feature_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

    preprocess_and_save_features(
        annotations, video_dir, efficientnet_model, visual_projection_layer, emotion_dense_layer, gesture_dense_layer,
        visual_feature_dir, emotion_feature_dir, gesture_feature_dir
    )

    # Create dummy input tensors for initial model building
    dummy_batch_size = 1
    dummy_seq_len = 252

    dummy_visual_tokens = tf.random.uniform((dummy_batch_size, dummy_seq_len, 512), dtype=tf.float32)  # float32
    dummy_emotion_tokens = tf.random.uniform((dummy_batch_size, dummy_seq_len, 512), dtype=tf.float32)  # float32
    dummy_gesture_tokens = tf.random.uniform((dummy_batch_size, dummy_seq_len, 512), dtype=tf.float32)  # float32
    dummy_gloss_target = tf.random.uniform((dummy_batch_size, 400), maxval=3521, dtype=tf.int32)
    dummy_text_target = tf.random.uniform((dummy_batch_size, 400), maxval=3521, dtype=tf.int32)

    encoder_seq_len = dummy_seq_len

    dummy_masks = create_attention_masks(
        encoder_token_ids=tf.ones([dummy_batch_size, encoder_seq_len], dtype=tf.int32),
        decoder_token_ids=dummy_gloss_target,
        pad_token_id=tf.constant(vocab.get('<PAD>', 0), dtype=tf.int32),
        num_heads=model.num_heads
    )

    # Ensure masks are float32 (no casting needed)
    dummy_masks = {k: v for k, v in dummy_masks.items()}

    assert dummy_masks['encoder_padding_mask'].shape[-1] == encoder_seq_len, \
        f"Encoder Mask Seq_len {dummy_masks['encoder_padding_mask'].shape[-1]} does not match encoder_seq_len {encoder_seq_len}"

    # Perform a dummy forward pass to build the model
    try:
        outputs = model(
            visual_tokens=dummy_visual_tokens,
            emotion_tokens=dummy_emotion_tokens,
            gesture_tokens=dummy_gesture_tokens,
            gloss_target=dummy_gloss_target,
            text_target=dummy_text_target,
            training=True,  # Set to True for dummy forward pass
            masks=dummy_masks
        )
        gloss_predictions, text_predictions = outputs
        logger.info("Dummy forward pass successful.")
    except Exception as e:
        logger.error(f"Dummy forward pass failed: {e}")
        return

    total_params = np.sum([np.prod(v.shape) for v in model.variables])
    if total_params == 0:
        logger.error("Model has no variables after building. Cannot proceed.")
        return
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")

    # Log model summary
    logger.info("Model Summary:")
    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
    logger.info(stream.getvalue())

    # Prepare checkpointing
    checkpoint_dir = os.path.join('checkpoints_train', '')
    logger.info(f"Checkpoint directory is {checkpoint_dir}")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        logger.info(f"Created checkpoint directory: {checkpoint_dir}")
    initial_epoch = 0

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        load_model_checkpoint(model, latest_checkpoint)
        initial_epoch = int(latest_checkpoint.split('-')[-1])
        logger.info(f"Loaded checkpoint from epoch {initial_epoch}")
    else:
        logger.info("No checkpoint found. Starting from scratch.")

    best_val_loss = float('inf') 
    epochs_without_improvement = 0

    for epoch in range(initial_epoch, epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        epoch_start_time = time.time()
        epoch_loss = 0.0
        step = 0

        for video_name, annotation in annotations.items():
            try:
                video_path = os.path.join(video_dir, f"{video_name}.mp4")
                video_name = sanitize_filename(video_name)
                logger.info(f"Processing video: {video_path}")

                # Load pre-extracted features
                visual_path = os.path.join(visual_feature_dir, f"{video_name}.npy")
                emotion_path = os.path.join(emotion_feature_dir, f"{video_name}.npy")
                gesture_path = os.path.join(gesture_feature_dir, f"{video_name}.npy")

                if not (os.path.exists(visual_path) and os.path.exists(emotion_path) and os.path.exists(gesture_path)):
                    logger.warning(f"Feature files missing for {video_name}. Skipping.")
                    continue

                visual_tokens = np.load(visual_path)
                emotion_tokens = np.load(emotion_path)
                gesture_tokens = np.load(gesture_path)

                visual_tokens = tf.convert_to_tensor(visual_tokens, dtype=tf.float32)  # Changed to float32
                emotion_tokens = tf.convert_to_tensor(emotion_tokens, dtype=tf.float32)  # Changed to float32
                gesture_tokens = tf.convert_to_tensor(gesture_tokens, dtype=tf.float32)  # Changed to float32

                visual_tokens = tf.expand_dims(visual_tokens, axis=0)  
                emotion_tokens = tf.expand_dims(emotion_tokens, axis=0)  
                gesture_tokens = tf.expand_dims(gesture_tokens, axis=0)  

                # Check for NaN or Inf in tokens
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
                    logger.error(f"Decoder input sequence length is zero for video {video_name}. Skipping this sample.")
                    continue

                masks = create_attention_masks(
                    encoder_token_ids=tf.ones([batch_size, tf.shape(visual_tokens)[1]], dtype=tf.int32),
                    decoder_token_ids=gloss_decoder_input,
                    pad_token_id=tf.constant(vocab.get('<PAD>', 0), dtype=tf.int32),
                    num_heads=model.num_heads
                )

                # Ensure masks are float32 (no casting)
                masks = {k: v for k, v in masks.items()}  # No casting

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
                        training=True,
                        masks=masks
                    )
                    
                    # Ensure outputs are correctly returned
                    if outputs is None or len(outputs) != 2:
                        logger.error(f"Model returned invalid outputs for video {video_name}. Skipping.")
                        continue
                    
                    gloss_predictions, text_predictions = outputs

                    # Compute loss
                    gloss_loss = loss_function(gloss_decoder_target, gloss_predictions)
                    text_loss = loss_function(text_decoder_target, text_predictions)
                    total_loss_sample = gloss_loss + text_loss

                # Compute gradients
                gradients = tape.gradient(total_loss_sample, model.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                logger.debug(f"Total Loss: {total_loss_sample.numpy()}")
                epoch_loss += total_loss_sample.numpy()
                step += 1

                # Additional logging for debugging
                if tf.math.is_nan(total_loss_sample):
                    logger.error("Encountered NaN loss during training.")
                    break

                # Logging every 100 steps
                if step % 100 == 0:
                    avg_loss = epoch_loss / step
                    logger.info(f"[Epoch: {epoch + 1:03d} Step: {step:06d}] "
                                f"Average Training Loss: {avg_loss:.6f}")
            except Exception as e:
                logger.error(f"Error evaluating video {video_name}: {e}")
                continue
    
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / max(step, 1)
        logger.info(f"Epoch {epoch + 1}: Average Training Loss {avg_epoch_loss:.4f} "
                    f"Time Taken: {epoch_duration:.2f} seconds")

        # Evaluate on validation set
        logger.info(f"Evaluating model after Epoch {epoch + 1}")
        val_loss, val_wer = evaluate(model, dev_excel, video_dir, vocab, inverse_vocab, epoch, step, max_length=400)
        if val_loss is None or val_wer is None:
            logger.warning(f"Evaluation failed at epoch {epoch + 1}. Skipping validation metrics.")
            continue

        # Save checkpoint
        logger.info(f"Attempting to save checkpoint at epoch {epoch + 1}")
        save_model_checkpoint(model, checkpoint_dir=checkpoint_dir, epoch=epoch + 1)
        logger.info(f"Checkpoint saved at the end of epoch {epoch + 1}")

        # Log evaluation metrics
        logger.info(f"Validation Loss: {val_loss:.4f}, WER: {val_wer:.2f}%")
    
        # Early Stopping Logic based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            logger.info(f"Validation loss improved to {val_loss:.4f} at epoch {epoch + 1}")
            save_model_checkpoint(model, checkpoint_dir=checkpoint_dir, epoch='best')
            logger.info(f"Best checkpoint updated at epoch {epoch + 1}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement in validation loss for {epochs_without_improvement} epochs.")
            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered.")
                break

    # Final evaluations after training
    logger.info("Starting final evaluations.")
    final_evaluation(model, dev_excel, video_dir, vocab, inverse_vocab, checkpoint_dir)
    '''
    # Perform beam search evaluation
    logger.info("Starting beam search evaluation...")
    
    evaluate_beam_search(
        model, dev_excel, video_dir, vocab, inverse_vocab,
        beam_sizes=[1],
        alpha_values=[0.6],
        max_length=50
    )
    '''

def find_max_combined_sequence_length(annotations, video_dir, logger):
    max_visual = max_emotion = max_gesture = 0
    visual_feature_dir = 'data/features/visual'
    emotion_feature_dir = 'data/features/emotion'
    gesture_feature_dir = 'data/features/gesture'

    for video_name, annotation in annotations.items():
        try:
            video_path = os.path.join(video_dir, f"{video_name}.mp4")
            video_name = sanitize_filename(video_name)
            logger.info(f"Processing video: {video_path}")

            # Load pre-extracted features
            visual_path = os.path.join(visual_feature_dir, f"{video_name}.npy")
            emotion_path = os.path.join(emotion_feature_dir, f"{video_name}.npy")
            gesture_path = os.path.join(gesture_feature_dir, f"{video_name}.npy")

            if not (os.path.exists(visual_path) and os.path.exists(emotion_path) and os.path.exists(gesture_path)):
                logger.warning(f"Feature files missing for {video_name}. Skipping.")
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

            max_visual = max(max_visual, visual_tokens.shape[1])
            max_emotion = max(max_emotion, emotion_tokens.shape[1])
            max_gesture = max(max_gesture, gesture_tokens.shape[1])

        except Exception as e:
            logger.error(f"Error processing video {video_name}: {e}")
            continue

    max_combined = max_visual + max_emotion + max_gesture
    logger.info(f"Maximum combined sequence length: {max_combined}")
    return max_combined

def main():
    vocab_file = os.path.join('data', 'vocab.json')
    logger.info("Vocabulary file path: %s", vocab_file)
    logger.info("Does the vocabulary file exist? %s", os.path.isfile(vocab_file))
    vocab, inverse_vocab = load_vocab(vocab_file)
    logger.info(f"Loaded vocab with {len(vocab)} entries and inverse_vocab with {len(inverse_vocab)} entries.")

    train_excel = os.path.join('data', 'Train1.xlsx')
    dev_excel = os.path.join('data', 'dev1.xlsx')
    video_dir = os.path.join('data', '')
    epochs = 1
    learning_rate = 0.001

    # Load annotations to find maximum sequence length
    annotations = load_annotations_from_excel(train_excel)
    #max_combined = find_max_combined_sequence_length(annotations, video_dir, logger)
    #logger.info(f"Max combined sequence length is : {max_combined}")

    # Initialize the Transformer model with pad_token_id and appropriate max_positional_encoding
    model = SignLanguageTransformer(
        visual_dim=512,
        emotion_dim=512,
        gesture_dim=512,
        gloss_vocab_size=len(vocab),
        text_vocab_size=len(vocab),
        inverse_vocab=inverse_vocab,
        num_layers=4,
        num_heads=16,
        ff_dim=512,
        dropout_rate=0.1,
        start_token_id=1,
        end_token_id=2,
        pad_token_id=vocab.get('<PAD>', 0),  
        max_positional_encoding=1500
    )

    logger.info("Hello! This is your Sign Language Transformer training script.")

    logger.info("Training configuration:")
    # Log configuration settings
    logger.info("Configuration Settings:")
    logger.info(f"Vocabulary Size: {len(vocab)}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Learning Rate: {learning_rate}")

    # Define the patience parameter
    patience = 20 

    # **Begin: Build the model by performing a dummy forward pass**
    # Build the model by performing a dummy forward pass.
    logger.info("Building the model by performing a dummy forward pass.")
    dummy_batch_size = 1
    dummy_seq_len = 252

    dummy_visual_tokens = tf.random.uniform((dummy_batch_size, dummy_seq_len, 512), dtype=tf.float32)
    dummy_emotion_tokens = tf.random.uniform((dummy_batch_size, dummy_seq_len, 512), dtype=tf.float32)
    dummy_gesture_tokens = tf.random.uniform((dummy_batch_size, dummy_seq_len, 512), dtype=tf.float32)
    dummy_gloss_target = tf.random.uniform((dummy_batch_size, 400), maxval=3521, dtype=tf.int32)
    dummy_text_target = tf.random.uniform((dummy_batch_size, 400), maxval=3521, dtype=tf.int32)

    dummy_masks = create_attention_masks(
        encoder_token_ids=tf.ones([dummy_batch_size, dummy_seq_len], dtype=tf.int32),
        decoder_token_ids=dummy_gloss_target,
        pad_token_id=tf.constant(vocab.get('<PAD>', 0), dtype=tf.int32),
        num_heads=model.num_heads
    )

    dummy_masks = {k: v for k, v in dummy_masks.items()}

    try:
        gloss_predictions, text_predictions = model(
            visual_tokens=dummy_visual_tokens,
            emotion_tokens=dummy_emotion_tokens,
            gesture_tokens=dummy_gesture_tokens,
            gloss_target=dummy_gloss_target,
            text_target=dummy_text_target,
            training=False,
            masks=dummy_masks
        )
        logger.info("Dummy forward pass successful. Model is built.")
    except Exception as e:
        logger.error(f"Dummy forward pass failed: {e}")
        return
    # **End: Build the model**

    # **Begin: Log device placement for variables**
    logger.info("Logging device placement for trainable variables:")
    for layer in model.layers:
        for var in layer.trainable_variables:
            logger.info(f"{var.name} is on device: {var.device}")
    # **End: Log device placement**

    # Start training
    train(model, train_excel, dev_excel, video_dir, vocab, inverse_vocab, epochs, learning_rate, patience=patience)

if __name__ == "__main__":
    main()
