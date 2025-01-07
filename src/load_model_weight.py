import tensorflow as tf
from models.transformer import SignLanguageTransformer

# Initialize the model
visual_dim = 512
emotion_dim = 512
gesture_dim = 512
gloss_vocab_size = 3521  # Adjust based on your vocab size
text_vocab_size = 3521  # Adjust based on your vocab size
inverse_vocab = {}  # Load or define your inverse vocab here

model = SignLanguageTransformer(
    visual_dim=visual_dim,
    emotion_dim=emotion_dim,
    gesture_dim=gesture_dim,
    gloss_vocab_size=gloss_vocab_size,
    text_vocab_size=text_vocab_size,
    inverse_vocab=inverse_vocab,
    num_layers=2,
    num_heads=8,
    ff_dim=512,
    dropout_rate=0.1,
    start_token_id=1,
    end_token_id=2,
    pad_token_id=0,
    max_positional_encoding=1000
)

# Create dummy input tensors to initialize the model's variables
dummy_visual_tokens = tf.random.uniform((1, 10, visual_dim))  # (batch_size, seq_len, feature_dim)
dummy_emotion_tokens = tf.random.uniform((1, 10, emotion_dim))
dummy_gesture_tokens = tf.random.uniform((1, 10, gesture_dim))
dummy_gloss_target = tf.random.uniform((1, 10), maxval=gloss_vocab_size, dtype=tf.int32)
dummy_text_target = tf.random.uniform((1, 10), maxval=text_vocab_size, dtype=tf.int32)

# Perform a dummy forward pass
model(
    visual_tokens=dummy_visual_tokens,
    emotion_tokens=dummy_emotion_tokens,
    gesture_tokens=dummy_gesture_tokens,
    gloss_target=dummy_gloss_target,
    text_target=dummy_text_target,
    training=False
)

# Load weights
model.load_weights('checkpoints/model_epoch_best.weights.h5')
print("Weights loaded successfully!")


# Display model summary
model.summary()