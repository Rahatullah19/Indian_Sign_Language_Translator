
import difflib
import tensorflow as tf
import numpy as np
import math
import json
import logging
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense  

logger = logging.getLogger(__name__)

def create_padding_mask(token_ids, pad_token_id):
    token_ids = tf.cast(token_ids, tf.int32)
    pad_token_id = tf.cast(pad_token_id, tf.int32)
    # Cast mask to float32 for consistency
    mask = tf.cast(tf.math.equal(token_ids, pad_token_id), tf.float32)  # [batch_size, seq_len]
    mask = tf.expand_dims(mask, axis=1)  # [batch_size, 1, seq_len]
    mask = tf.expand_dims(mask, axis=1)  # [batch_size, 1, 1, seq_len]
    return mask

def create_look_ahead_mask(size, dtype=tf.float32):
    # Use float32 for the look-ahead mask
    mask = 1 - tf.linalg.band_part(tf.ones((size, size), dtype=dtype), -1, 0)
    mask = tf.reshape(mask, [1, 1, size, size])  
    return mask

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
            align_symbols.extend(['X'] * max(i2 - i1, j2 - j1))
        elif tag == 'delete':
            align_symbols.extend(['-'] * (i2 - i1))
        elif tag == 'insert':
            align_symbols.extend(['+'] * (j2 - j1))
        else:
            align_symbols.extend(['?'] * (max(i2 - i1, j2 - j1)))

    # Ensure all alignment symbols are strings
    align_symbols = [str(symbol) for symbol in align_symbols]
    return ' '.join(align_symbols)

def handle_oov_tokens(predictions, inverse_vocab, max_length=50, end_token_id=2):
    if isinstance(predictions, tf.Tensor):
        predictions = predictions.numpy().tolist()
    elif isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
    elif isinstance(predictions, list):
        pass
    else:
        raise ValueError("Unsupported type for predictions.")

    if not all(isinstance(seq, list) for seq in predictions):
        raise ValueError("Each prediction should be a list of token IDs.")

    predicted_sentences = []
    for seq in predictions:
        sentence = []
        for token_id in seq:
            if token_id == end_token_id:
                break
            if token_id in inverse_vocab:
                sentence.append(inverse_vocab[token_id])
            else:
                sentence.append('<OOV>')
        predicted_sentences.append(' '.join(sentence))

    logger.debug(f"Predicted Sentences (without <END> and '.'): {predicted_sentences}")
    return predicted_sentences

def create_attention_masks(encoder_token_ids, decoder_token_ids, pad_token_id, num_heads):
    encoder_padding_mask = create_padding_mask(encoder_token_ids, pad_token_id)
    encoder_padding_mask = tf.tile(encoder_padding_mask, [1, num_heads, 1, 1])
    encoder_padding_mask = tf.cast(encoder_padding_mask, tf.float16)  # Cast to float16

    decoder_padding_mask = create_padding_mask(decoder_token_ids, pad_token_id)
    decoder_seq_len = tf.shape(decoder_token_ids)[1]
    look_ahead_mask = create_look_ahead_mask(decoder_seq_len, dtype=tf.float32)
    combined_mask = tf.maximum(decoder_padding_mask, look_ahead_mask)
    combined_mask = tf.squeeze(combined_mask, axis=1)
    combined_mask = tf.expand_dims(combined_mask, 1)
    combined_mask = tf.tile(combined_mask, [1, num_heads, 1, 1])
    combined_mask = tf.cast(combined_mask, tf.float16)  # Cast to float16

    cross_attention_mask = tf.squeeze(encoder_padding_mask, axis=2)
    cross_attention_mask = tf.expand_dims(cross_attention_mask, axis=2)
    # cross_attention_mask is already float16 after tile+cast above
    return {
        'encoder_padding_mask': encoder_padding_mask,
        'combined_mask': combined_mask,
        'cross_attention_mask': cross_attention_mask
    }

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)  # Removed dtype
    
    def call(self, query, key, value, training, attention_mask=None):
        attn_output = self.mha(query=query, key=key, value=value, attention_mask=attention_mask, training=training)
        return attn_output

class MultiHeadCrossAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dtype='float16')
    
    def call(self, query, key, value, training, attention_mask=None):
        attn_output = self.mha(query=query, key=key, value=value, attention_mask=attention_mask, training=training)
        return attn_output

class FeedForwardNetwork(layers.Layer):
    def __init__(self, embed_dim, ff_dim, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),  # Removed dtype
            layers.Dense(embed_dim)  # Removed dtype
        ])
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        x = self.ffn(x)
        x = self.dropout(x, training=training)
        return x

class TransformerEncoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        # Ensure all sublayers use float16
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dtype='float16')
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout_rate)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, dtype='float16')
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, dtype='float16')
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training, attention_mask=None):
        # Self-attention
        attn_output = self.mha(
            query=x, 
            value=x, 
            key=x, 
            attention_mask=attention_mask, 
            training=training
        )  # Output dtype: float16
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # dtype: float16

        # Feed-forward network
        ffn_output = self.ffn(out1, training=training)  # dtype: float16
        out2 = self.layernorm2(out1 + ffn_output)      # dtype: float16

        return out2

class TransformerEncoder(layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.enc_layers = [
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate) 
            for _ in range(num_layers)
        ]
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training, attention_mask=None):
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, attention_mask)
        return x

class TransformerDecoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        # Ensure all sublayers use float16
        self.mha1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dtype='float16')
        self.mha2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dtype='float16')
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout_rate)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, dtype='float16')
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, dtype='float16')
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6, dtype='float16')
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training, attention_mask=None, cross_attention_mask=None):
        # Self-attention
        attn1 = self.mha1(
            query=x, 
            value=x, 
            key=x, 
            attention_mask=attention_mask, 
            training=training
        )  # Output dtype: float16
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)  # dtype: float16

        # Cross-attention
        attn2 = self.mha2(
            query=out1,
            key=enc_output,
            value=enc_output,
            attention_mask=cross_attention_mask,
            training=training
        )  # Output dtype: float16
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)  # dtype: float16

        # Feed-forward network
        ffn_output = self.ffn(out2, training=training)  # dtype: float16
        out3 = self.layernorm3(out2 + ffn_output)      # dtype: float16

        return out3

class TransformerDecoder(layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.dec_layers = [
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout_rate) 
            for _ in range(num_layers)
        ]
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training, attention_mask=None, cross_attention_mask=None):
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training, attention_mask, cross_attention_mask)
        return x

class PositionalEncoding(layers.Layer):
    def __init__(self, embed_dim, max_len=1500):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)
    
    def get_angles(self, pos, i, embed_dim):
        angle_rates = 1 / tf.pow(10000.0, (2 * (i//2)) / tf.cast(embed_dim, tf.float32))
        return tf.cast(pos, tf.float32) * angle_rates
    
    def positional_encoding(self, max_len, embed_dim):
        angle_rads = self.get_angles(
            pos=tf.range(start=0, limit=max_len, delta=1, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(start=0, limit=embed_dim, delta=1, dtype=tf.float32)[tf.newaxis, :],
            embed_dim=embed_dim
        )
        angle_rads = tf.where(tf.math.equal(tf.cast(tf.range(embed_dim) % 2, tf.bool), True),
                              tf.math.sin(angle_rads),
                              tf.math.cos(angle_rads))
        pos_encoding = tf.expand_dims(angle_rads, 0)  # [1, max_len, embed_dim]
        pos_encoding = tf.cast(pos_encoding, tf.float16)  # Convert to float16
        return pos_encoding
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]