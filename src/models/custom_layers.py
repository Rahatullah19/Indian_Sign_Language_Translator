# custom_layers.py
import difflib
import tensorflow as tf
import numpy as np
import json
import logging
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense  

logger = logging.getLogger(__name__)

def create_padding_mask(token_ids, pad_token_id):
    
    token_ids = tf.cast(token_ids, tf.int32)
    pad_token_id = tf.cast(pad_token_id, tf.int32)
    mask = tf.cast(tf.math.equal(token_ids, pad_token_id), tf.float32) 
    return mask

def create_look_ahead_mask(size, dtype=tf.float32): 
    mask = 1 - tf.linalg.band_part(tf.ones((size, size), dtype=dtype), -1, 0)
    mask = tf.reshape(mask, [1, 1, size, size])  
    return mask

def generate_alignment(reference_tokens, hypothesis_tokens):
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

    align_symbols = [str(symbol) for symbol in align_symbols]
    return ' '.join(align_symbols)

def handle_oov_tokens(predictions, inverse_vocab, max_length=50):
    
    if isinstance(predictions, (int, np.integer)):
        predictions = [[predictions]]
        logger.debug("Wrapped single integer prediction into a list.")

    elif isinstance(predictions, tf.Tensor):
        predictions = tf.argmax(predictions, axis=-1).numpy().tolist()
        logger.debug("Converted Tensor predictions to list using argmax.")

    elif isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
        logger.debug("Converted numpy.ndarray predictions to list.")

    elif isinstance(predictions, list):
        if all(isinstance(x, (int, np.integer)) for x in predictions):
            predictions = [predictions]
            logger.debug("Wrapped flat list of integers into a list of lists.")

    else:
        logger.error(f"Unsupported prediction type: {type(predictions)}. Returning empty list.")
        return ['<OOV>']

    if not all(isinstance(seq, list) for seq in predictions):
        logger.error("Predictions are not properly formatted as a list of lists.")
        return ['<OOV>']

    predicted_sentences = []
    for seq in predictions:
        if not isinstance(seq, list):
            logger.error(f"Expected a list of token IDs, but got: {type(seq)}. Skipping.")
            continue

        tokens = []
        for token_id in seq:
            if not isinstance(token_id, (int, np.integer)):
                logger.warning(f"Token ID '{token_id}' is not an integer. Using '<OOV>'.")
                tokens.append('<OOV>')
                continue

            word = inverse_vocab.get(int(token_id), '<OOV>')
            if word not in ['<START>', '<END>', '<PAD>']:
                tokens.append(word)

        sentence = " ".join(tokens[:max_length]) if tokens else '<EMPTY>'
        predicted_sentences.append(sentence)

    return predicted_sentences

def create_attention_masks(encoder_token_ids, decoder_token_ids, pad_token_id):
    pad_token_id = tf.cast(pad_token_id, tf.int32)
    encoder_token_ids = tf.cast(encoder_token_ids, tf.int32)
    decoder_token_ids = tf.cast(decoder_token_ids, tf.int32)

    encoder_padding_mask = create_padding_mask(encoder_token_ids, pad_token_id)
    decoder_padding_mask = create_padding_mask(decoder_token_ids, pad_token_id)

    decoder_seq_len = tf.cast(tf.shape(decoder_token_ids)[1], tf.int32)  # Cast to int32
    look_ahead_mask = create_look_ahead_mask(decoder_seq_len, dtype=tf.float32)

    combined_mask = tf.maximum(
        tf.expand_dims(decoder_padding_mask, 1),
        look_ahead_mask
    )

    cross_attention_mask = tf.tile(
        tf.expand_dims(encoder_padding_mask, 1),
        tf.cast([1, decoder_seq_len, 1], tf.int32)
    )

    return {
        'encoder_padding_mask': tf.cast(encoder_padding_mask, tf.float32),
        'combined_mask': tf.cast(combined_mask, tf.float32),
        'cross_attention_mask': tf.cast(cross_attention_mask, tf.float32)
    }


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=0.1,
        )

    def call(self, query, value, key, training, attention_mask=None):
        return self.mha(
            query=query,
            value=value,
            key=key,
            attention_mask=attention_mask,
            training=training
        )

class MultiHeadCrossAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=0.1,
        )

    def call(self, query, value, key, training, attention_mask=None):
        return self.mha(
            query=query,
            value=value,
            key=key,
            attention_mask=attention_mask,
            training=training
        )

class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, embed_dim, ff_dim, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu', dtype='float32'),
            tf.keras.layers.Dropout(dropout_rate),
            Dense(embed_dim, dtype='float32'),
            tf.keras.layers.Dropout(dropout_rate)
        ])

    def call(self, x, training=False):
        return self.ffn(x, training=training)

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.msa = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, attention_mask=None):
        attn_output = self.msa(query=x, value=x, key=x, training=training, attention_mask=attention_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output) 

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output) 

        return out2

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.enc_layers = [
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, attention_mask=None):
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, attention_mask)
        return x

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.msa = MultiHeadSelfAttention(embed_dim, num_heads)
        self.cross_mha = MultiHeadCrossAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, attention_mask=None, cross_attention_mask=None):
        
        attn1 = self.msa(query=x, value=x, key=x, training=training, attention_mask=attention_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)  

        # Cross-Attention
        attn2 = self.cross_mha(query=out1, value=enc_output, key=enc_output, training=training, attention_mask=cross_attention_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2) 

        # Feed-Forward Network
        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)  

        return out3

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.dec_layers = [
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, attention_mask=None, cross_attention_mask=None):
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training, attention_mask, cross_attention_mask)
        return x

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len  

        position = np.arange(max_len)[:, np.newaxis]
        i = np.arange(embed_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(embed_dim))
        angle_rads = position * angle_rates

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return self.pos_encoding[:, :tf.shape(inputs)[1], :]