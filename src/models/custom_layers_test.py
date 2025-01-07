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
            logger.debug("Wrapped single list of token IDs into a list of lists.")
        else:
            logger.error("Inconsistent token ID formatting in predictions.")
            return ['<OOV>']

    else:
        logger.error(f"Unsupported prediction type: {type(predictions)}. Returning empty list.")
        return ['<OOV>']

    if not all(isinstance(seq, list) for seq in predictions):
        logger.error("Predictions are not properly formatted as a list of lists.")
        return ['<OOV>']

    predicted_sentences = []
    for seq in predictions:
        if not isinstance(seq, list):
            logger.error("Each prediction sequence must be a list of token IDs.")
            predicted_sentences.append('<OOV>')
            continue

        tokens = []
        for token_id in seq:
            word = inverse_vocab.get(token_id, '<OOV>')
            tokens.append(word)

        sentence = " ".join(tokens[:max_length]) if tokens else '<EMPTY>'
        predicted_sentences.append(sentence)

    return predicted_sentences

def create_attention_masks(encoder_token_ids, decoder_token_ids, pad_token_id):
    print(f"Encoder token IDs shape: {encoder_token_ids.shape}")
    print(f"Decoder token IDs shape: {decoder_token_ids.shape}")
    pad_token_id = tf.cast(pad_token_id, tf.int32)
    encoder_token_ids = tf.cast(encoder_token_ids, tf.int32)
    decoder_token_ids = tf.cast(decoder_token_ids, tf.int32)

    encoder_padding_mask = create_padding_mask(encoder_token_ids, pad_token_id)
    decoder_padding_mask = create_padding_mask(decoder_token_ids, pad_token_id)

    decoder_seq_len = tf.cast(tf.shape(decoder_token_ids)[1], tf.int32)
    look_ahead_mask = create_look_ahead_mask(decoder_seq_len, dtype=tf.float32)

    combined_mask = tf.maximum(
        tf.expand_dims(decoder_padding_mask, 1),
        look_ahead_mask
    )

    cross_attention_mask = tf.tile(
        tf.expand_dims(encoder_padding_mask, 1),
        [1, decoder_seq_len, 1]
    )

    return {
        'encoder_padding_mask': tf.cast(encoder_padding_mask, tf.float32),
        'combined_mask': tf.cast(combined_mask, tf.float32),
        'cross_attention_mask': tf.cast(cross_attention_mask, tf.float32)
    }

def create_attention_masks_encoder_only(seq_len, batch_size=1):
    """
    Creates an encoder padding mask with all ones (no padding).
    Shape: (batch_size, 1, 1, seq_len)
    """
    return tf.ones((batch_size, 1, 1, seq_len), dtype=tf.float32)

def create_attention_masks_decoder(seq_len_decoder, pad_token_id=0, batch_size=1):
    """
    Creates look-ahead and combined masks for the decoder.
    Shape: (batch_size, 1, seq_len_decoder, seq_len_decoder)
    """
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len_decoder, seq_len_decoder), dtype=tf.float32), -1, 0)
    pad_mask = tf.expand_dims(create_padding_mask(tf.ones((batch_size, seq_len_decoder), dtype=tf.int32) * pad_token_id, pad_token_id), axis=1)
    combined_mask = tf.maximum(look_ahead_mask, pad_mask)
    combined_mask = tf.expand_dims(combined_mask, axis=1)  # Shape: (batch_size, 1, seq_len_decoder, seq_len_decoder)
    return combined_mask

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, name=None):
        super(MultiHeadSelfAttention, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = embed_dim // num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.key_dim)
        self.dropout = layers.Dropout(0.1)
        # Set axis to [1, 2] to match saved weights
        self.layernorm = layers.LayerNormalization(epsilon=1e-6, axis=[1, 2])

    def call(self, query, value, key, training, masks=None):
        attn_output = self.attention(query, value, key, attention_mask=masks.get('combined_mask'))
        attn_output = self.dropout(attn_output, training=training)
        out = self.layernorm(query + attn_output)
        return out

class MultiHeadCrossAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, name=None):
        super(MultiHeadCrossAttention, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = embed_dim // num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.key_dim)
        self.dropout = layers.Dropout(0.1)
        # Set axis to [1, 2] to match saved weights
        self.layernorm = layers.LayerNormalization(epsilon=1e-6, axis=[1, 2])

    def call(self, query, value, key, training, masks=None):
        attn_output = self.attention(query, value, key, attention_mask=masks.get('cross_attention_mask'))
        attn_output = self.dropout(attn_output, training=training)
        out = self.layernorm(query + attn_output)
        return out

class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, embed_dim, ff_dim, dropout_rate=0.1, name=None):
        super(FeedForwardNetwork, self).__init__(name=name)
        self.dense1 = Dense(ff_dim, activation='relu')
        self.dense2 = Dense(embed_dim)
        self.dropout = layers.Dropout(dropout_rate)
        # Set axis to [1, 2] to match saved weights
        self.layernorm = layers.LayerNormalization(epsilon=1e-6, axis=[1, 2])

    def call(self, x, training=False):
        ffn_output = self.dense1(x)
        ffn_output = self.dropout(ffn_output, training=training)
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout(ffn_output, training=training)
        out = self.layernorm(x + ffn_output)
        return out

# Ensure all Transformer layers use axis=[1, 2] in LayerNormalization
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, name=None):
        super(TransformerEncoderLayer, self).__init__(name=name)
        self.mha = MultiHeadSelfAttention(embed_dim, num_heads, name=f"{name}/multi_head_self_attention")
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout_rate, name=f"{name}/feed_forward_network")
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training, attention_mask=None):
        attn_output = self.mha(x, x, x, training, masks={'combined_mask': attention_mask})
        attn_output = self.dropout1(attn_output, training=training)
        out1 = layers.add([x, attn_output])
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = layers.add([out1, ffn_output])
        return out2

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout_rate=0.1, name=None):
        super(TransformerEncoder, self).__init__(name=name)
        self.num_layers = num_layers
        self.layers_list = [
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate, name=f"encoder_layer_{i}")
            for i in range(num_layers)
        ]

    def call(self, x, training, attention_mask=None):
        for i in range(self.num_layers):
            x = self.layers_list[i](x, training, attention_mask)
        return x

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, name=None):
        super(TransformerDecoderLayer, self).__init__(name=name)
        self.mha1 = MultiHeadSelfAttention(embed_dim, num_heads, name=f"{name}/multi_head_self_attention")
        self.mha2 = MultiHeadCrossAttention(embed_dim, num_heads, name=f"{name}/multi_head_cross_attention")
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout_rate, name=f"{name}/feed_forward_network")
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, attention_mask=None, cross_attention_mask=None):
        attn1 = self.mha1(x, x, x, training, masks={'combined_mask': attention_mask})
        attn1 = self.dropout1(attn1, training=training)
        out1 = layers.add([x, attn1])

        attn2 = self.mha2(out1, enc_output, enc_output, training, masks={'cross_attention_mask': cross_attention_mask})
        attn2 = self.dropout2(attn2, training=training)
        out2 = layers.add([out1, attn2])

        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = layers.add([out2, ffn_output])

        return out3

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout_rate=0.1, name=None):
        super(TransformerDecoder, self).__init__(name=name)
        self.num_layers = num_layers
        self.layers_list = [
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout_rate, name=f"decoder_layer_{i}")
            for i in range(num_layers)
        ]

    def call(self, x, enc_output, training, attention_mask=None, cross_attention_mask=None):
        for i in range(self.num_layers):
            x = self.layers_list[i](x, enc_output, training, attention_mask, cross_attention_mask)
        return x

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len=5000, embed_dim=512, name=None):
        super(PositionalEncoding, self).__init__(name=name)
        self.pos_encoding = self.generate_positional_encoding(max_len, embed_dim)

    def generate_positional_encoding(self, max_len, embed_dim):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(embed_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(embed_dim))
        angle_rads = pos * angle_rates

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]