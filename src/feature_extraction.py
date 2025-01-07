# training.py

import os
import logging
import time
import numpy as np
import tensorflow as tf
from models.transformer import SignLanguageTransformer
from utils import (
    extract_video_frames,
    extract_visual_token,
    load_vocab,
    extract_emotion_token,
    extract_gesture_token,
    tokenize_sequence,
    save_model_checkpoint,
    load_model_checkpoint,
    load_annotations_from_excel,
)
from evaluation import evaluate, evaluate_beam_search, final_evaluation
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import EfficientNetB0
from io import StringIO

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'train.log'), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train(model, train_excel, dev_excel, video_dir, vocab, epochs, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    annotations = load_annotations_from_excel(train_excel)

    # Initialize models and layers once
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    visual_projection_layer = Dense(512, activation=None)

    # Initialize Dense layers for emotion and gesture tokens
    emotion_dense_layer = Dense(512, activation=None)
    gesture_dense_layer = Dense(512, activation=None)

    # Build the model by passing dummy inputs
    logger.info("Building the model with dummy inputs to initialize weights.")

    # Create dummy inputs matching the expected input shapes
    dummy_visual_input = tf.zeros([1, 50, 512])  # Adjust sequence length as needed
    dummy_emotion_input = tf.zeros([1, 50, 512])
    dummy_gesture_input = tf.zeros([1, 50, 512])
    dummy_gloss_target = tf.zeros([1, 50], dtype=tf.int32)
    dummy_text_target = tf.zeros([1, 50], dtype=tf.int32)

    # Build the model by passing dummy inputs
    _ = model(
        visual_tokens=dummy_visual_input,
        emotion_tokens=dummy_emotion_input,
        gesture_tokens=dummy_gesture_input,
        gloss_target=dummy_gloss_target,   # Changed to 'gloss_target'
        text_target=dummy_text_target,     # Changed to 'text_target'
        training=False
    )

    # Now the model is built, and variables are initialized
    # Calculate total and trainable parameters
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
    
    # Re-added missing lines
    total_steps = epochs * len(annotations)
    step = 0

    # Define feature directories
    visual_feature_dir = 'data/features/visual'
    emotion_feature_dir = 'data/features/emotion'
    gesture_feature_dir = 'data/features/gesture'

    ## Check for existing checkpoints to resume training
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    initial_epoch = 0

    # Load the latest checkpoint if available
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        logger.info(f"Found checkpoint at {latest_checkpoint}, attempting to load.")
        # Use load_model_checkpoint to load the model weights
        load_model_checkpoint(model, latest_checkpoint)
        # Extract the initial epoch number from the checkpoint filename
        # Assuming the filename is in the format 'model_epoch_{epoch}.weights.h5'
        epoch_str = os.path.basename(latest_checkpoint).split('_')[-1].split('.')[0]
        if epoch_str.isdigit():
            initial_epoch = int(epoch_str)
            logger.info(f"Resuming training from epoch {initial_epoch}")
        elif 'best' in latest_checkpoint:
            initial_epoch = 0  # Or set to the appropriate epoch
            logger.info(f"Resuming training from the best model checkpoint")
        else:
            initial_epoch = 0
    else:
        logger.info("No checkpoint found, starting training from scratch.")

    best_val_loss = float('inf')  # Initialize best validation loss

    for epoch in range(initial_epoch, epochs):
        logger.info(f"EPOCH {epoch + 1}")
        epoch_start_time = time.time()
        epoch_loss = 0.0
        step = 0  # Reset step counter for each epoch

        for video_name, annotation in annotations.items():
            video_path = os.path.join(video_dir, video_name + '.mp4')
            logger.info(f"Processing video: {video_path}")

            # Extract frames from video
            frames = extract_video_frames(video_path)
            if not frames:
                logger.warning(f"No frames extracted from {video_path}. Skipping.")
                continue

            # Extract or load visual tokens
            visual_tokens = extract_visual_token(
                frames, efficientnet_model, visual_projection_layer,
                video_name, save_dir=visual_feature_dir
            )
            if visual_tokens is None:
                logger.warning(f"Skipping video {video_path} due to no valid visual tokens.")
                continue

            # Extract or load emotion tokens
            emotion_tokens = extract_emotion_token(
                frames, emotion_dense_layer,
                video_name, save_dir=emotion_feature_dir
            )
            if emotion_tokens is None:
                logger.warning(f"Skipping video {video_path} due to no valid emotion tokens.")
                continue

            # Extract or load gesture tokens
            gesture_tokens = extract_gesture_token(
                frames, gesture_dense_layer,
                video_name, save_dir=gesture_feature_dir
            )
            if gesture_tokens is None:
                logger.warning(f"Skipping video {video_path} due to no valid gesture tokens.")
                continue

            # Ensure batch dimension for tokens
            if len(visual_tokens.shape) == 2:
                visual_tokens = tf.expand_dims(visual_tokens, axis=0)
            if len(emotion_tokens.shape) == 2:
                emotion_tokens = tf.expand_dims(emotion_tokens, axis=0)
            if len(gesture_tokens.shape) == 2:
                gesture_tokens = tf.expand_dims(gesture_tokens, axis=0)

            # Tokenize gloss and translation
            gloss_target = tokenize_sequence(annotation['gloss'], vocab, max_seq_len=50)  # Renamed variable
            text_target = tokenize_sequence(annotation['text'], vocab, max_seq_len=50)    # Renamed variable

            # Ensure batch dimension and correct data type for targets
            if len(gloss_target.shape) == 1:
                gloss_target = tf.expand_dims(gloss_target, axis=0)
            gloss_target = tf.cast(gloss_target, dtype=tf.int32)

            if len(text_target.shape) == 1:
                text_target = tf.expand_dims(text_target, axis=0)
            text_target = tf.cast(text_target, dtype=tf.int32)

            # Print shapes for debugging
            print("visual_tokens shape:", visual_tokens.shape)
            print("emotion_tokens shape:", emotion_tokens.shape)
            print("gesture_tokens shape:", gesture_tokens.shape)
            print("gloss_target shape:", gloss_target.shape)
            print("text_target shape:", text_target.shape)

            with tf.GradientTape() as tape:
                # Forward pass
                gloss_logits, text_logits = model(
                    visual_tokens=visual_tokens,
                    emotion_tokens=emotion_tokens,
                    gesture_tokens=gesture_tokens,
                    gloss_target=gloss_target,
                    text_target=text_target,
                    training=True
                )

                # Compute losses
                gloss_loss = loss_fn(gloss_target, gloss_logits)
                text_loss = loss_fn(text_target, text_logits)
                total_loss = gloss_loss + text_loss

            # Backward pass and optimization
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_loss += total_loss.numpy()
            step += 1

            # Periodically log training progress
            if step % 100 == 0:
                avg_loss = epoch_loss / step
                logger.info(f"[Epoch: {epoch + 1:03d} Step: {step:06d}] "
                            f"Average Loss: {avg_loss:.6f} || "
                            f"Learning Rate: {optimizer.learning_rate.numpy():.6f}")

        # At the end of each epoch
        epoch_duration = time.time() - epoch_start_time
        epoch_loss = epoch_loss / len(annotations)  # Average loss per sample
        logger.info(f"Epoch {epoch + 1}: Total Training Loss {epoch_loss:.2f}")

        # Evaluate on the validation set
        logger.info(f"Evaluating model after Epoch {epoch + 1}")
        val_loss = evaluate(model, dev_excel, video_dir, vocab, epoch, step)

        # Save model checkpoint at the end of each epoch
        logger.info(f"Attempting to save checkpoint at epoch {epoch + 1}")
        save_model_checkpoint(model, checkpoint_dir=checkpoint_dir, epoch=epoch + 1)
        logger.info(f"Checkpoint saved at the end of epoch {epoch + 1}")

        # Save model checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"Validation loss improved to {val_loss:.4f} at epoch {epoch + 1}")
            save_model_checkpoint(model, checkpoint_dir=checkpoint_dir, epoch='best')
            logger.info(f"Best model updated at epoch {epoch + 1} with validation loss {val_loss:.4f}")

    # Final evaluation
    final_evaluation(model, dev_excel, video_dir, vocab, checkpoint_dir)
    # Beam search evaluation
    evaluate_beam_search(model, dev_excel, video_dir, vocab, beam_sizes=[1,2,3,4,5], alpha_values=[0.6], max_length=50)

def main():
    vocab_file = os.path.join('data', 'vocab.json')  # Path to vocab file
    print("Vocabulary file path:", vocab_file)
    print("Does the vocabulary file exist?", os.path.isfile(vocab_file))
    vocab = load_vocab(vocab_file)  # Load the vocabulary

    train_excel = os.path.join('data', 'Train1.xlsx')  # Path to Train.xlsx
    dev_excel = os.path.join('data', 'dev1.xlsx')      # Path to dev.xlsx
    video_dir = os.path.join('data', '')         # Video directory path
    epochs = 5
    learning_rate = 0.001

    model = SignLanguageTransformer(
        visual_dim=512,
        emotion_dim=512,
        gesture_dim=512,
        gloss_vocab_size=len(vocab),
        text_vocab_size=len(vocab)
    )

    logger.info("Hello! This is your Sign Language Transformer training script.")

    logger.info("Training configuration:")
    # Log configuration settings
    logger.info("Configuration Settings:")
    logger.info(f"Vocabulary Size: {len(vocab)}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Learning Rate: {learning_rate}")

    train(model, train_excel, dev_excel, video_dir, vocab, epochs, learning_rate)

if __name__ == "__main__":
    main()
