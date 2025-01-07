import os
import tensorflow as tf
import logging
from models.transformer import SignLanguageTransformer
from utils import (
    load_vocab,
    load_model_checkpoint,
    load_annotations_from_excel,
    extract_video_frames,
    extract_visual_token,
    extract_emotion_token,
    extract_gesture_token,
    tokenize_sequence
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
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
            logger.warning(f"No frames extracted from {video_path}. Skipping.")
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

def evaluate(model, dev_data, epoch, step):
    """
    Evaluates the model on the development data.

    Args:
        model (tf.keras.Model): The trained model.
        dev_data (list): A list of tuples containing inputs and targets.
        epoch (int): Current epoch number.
        step (int): Current step number.

    Returns:
        float: The average validation loss.
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    total_loss = 0.0
    total_steps = 0

    for batch in dev_data:
        inputs, targets = batch
        # Forward pass
        gloss_logits, text_logits = model(
            visual_tokens=inputs['visual_tokens'],
            emotion_tokens=inputs['emotion_tokens'],
            gesture_tokens=inputs['gesture_tokens'],
            gloss_target=inputs['gloss_target'],
            text_target=inputs['text_target'],
            training=False
        )

        # Compute losses
        gloss_loss = loss_fn(targets['gloss_output'], gloss_logits)
        text_loss = loss_fn(targets['text_output'], text_logits)
        total_loss += gloss_loss + text_loss
        total_steps += 1

    average_loss = total_loss / total_steps if total_steps > 0 else 0.0
    logger.info(f"Validation result at epoch {epoch}, step {step} - Loss: {average_loss.numpy():.4f}")

    return average_loss.numpy()

def load_and_evaluate(checkpoint_path, dev_excel, video_dir, vocab):
    # Initialize the model
    model = SignLanguageTransformer(
        visual_dim=512,
        emotion_dim=512,
        gesture_dim=512,
        gloss_vocab_size=len(vocab),
        text_vocab_size=len(vocab)
    )

    # Build the model by calling it with dummy inputs to initialize weights
    dummy_visual_input = tf.zeros([1, 50, 512])
    dummy_emotion_input = tf.zeros([1, 50, 512])
    dummy_gesture_input = tf.zeros([1, 50, 512])
    dummy_gloss_target = tf.zeros([1, 50], dtype=tf.int32)
    dummy_text_target = tf.zeros([1, 50], dtype=tf.int32)

    model(
        visual_tokens=dummy_visual_input,
        emotion_tokens=dummy_emotion_input,
        gesture_tokens=dummy_gesture_input,
        gloss_target=dummy_gloss_target,
        text_target=dummy_text_target,
        training=False
    )

    # Load weights
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        logger.info(f"Loaded model weights from {checkpoint_path}")
    else:
        logger.error(f"No weights file found at {checkpoint_path}")
        return

    # Load and preprocess development data
    dev_data = load_dev_data(dev_excel, video_dir, vocab)
    logger.info(f"Loaded {len(dev_data)} samples from development data.")

    # Evaluate the model
    val_loss = evaluate(model, dev_data, epoch=0, step=0)
    logger.info(f"Validation Loss: {val_loss}")

def main():
    vocab_file = os.path.join('data', 'vocab.json')
    vocab = load_vocab(vocab_file)

    dev_excel = os.path.join('data', 'dev1.xlsx')
    video_dir = os.path.join('data', '')
    checkpoint_path = os.path.join('checkpoints', 'model_epoch_best.weights.h5')

    load_and_evaluate(checkpoint_path, dev_excel, video_dir, vocab)

if __name__ == "__main__":
    main()