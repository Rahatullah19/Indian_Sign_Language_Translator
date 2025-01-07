
from pydoc import text
import tensorflow as tf
import logging
import difflib
import numpy as np
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.models import Model
from models.custom_layers_1 import (
    TransformerEncoder,
    TransformerDecoder,
    PositionalEncoding,
    create_padding_mask,
    create_look_ahead_mask,
    handle_oov_tokens,
    generate_alignment,
    create_attention_masks
)

logger = logging.getLogger(__name__)


def custom_handle_oov_tokens(predictions, inverse_vocab, max_length=50):
    return handle_oov_tokens(predictions, inverse_vocab, max_length)

def custom_generate_alignment(reference_tokens, hypothesis_tokens):
    return generate_alignment(reference_tokens, hypothesis_tokens)

class SignLanguageTransformer(tf.keras.Model):
    def __init__(
        self,
        visual_dim,
        emotion_dim,
        gesture_dim,
        gloss_vocab_size,
        text_vocab_size,
        inverse_vocab,
        num_layers=4,
        num_heads=16,
        ff_dim=512,
        dropout_rate=0.1,
        start_token_id=1,
        end_token_id=2,
        pad_token_id=0,
        max_positional_encoding=1500
    ):
        super(SignLanguageTransformer, self).__init__()

        self.inverse_vocab = inverse_vocab
        self.visual_dim = visual_dim
        self.emotion_dim = emotion_dim
        self.gesture_dim = gesture_dim
        self.gloss_vocab_size = gloss_vocab_size
        self.text_vocab_size = text_vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.pad_token_id = pad_token_id

        # Keep PositionalEncoding as float16
        self.encoder_positional_encoding = PositionalEncoding(
            embed_dim=visual_dim,
            max_len=max_positional_encoding
        )
        self.decoder_positional_encoding = PositionalEncoding(
            embed_dim=visual_dim,
            max_len=max_positional_encoding
        )

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=visual_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate
        )

        self.gloss_decoder = TransformerDecoder(
            num_layers=num_layers,
            embed_dim=visual_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate
        )
        self.text_decoder = TransformerDecoder(
            num_layers=num_layers,
            embed_dim=visual_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate
        )

        self.gloss_output_layer = Dense(gloss_vocab_size, dtype='float32')
        self.text_output_layer = Dense(text_vocab_size, dtype='float32')

        # Use float16 for embeddings
        self.gloss_embedding = Embedding(
            input_dim=gloss_vocab_size,
            output_dim=visual_dim,
            mask_zero=True,
            dtype='float16'
        )
        self.text_embedding = Embedding(
            input_dim=text_vocab_size,
            output_dim=visual_dim,
            mask_zero=True,
            dtype='float16'
        )

        # Use float32 for this layer but cast the result to float16 later
        self.input_projection = Dense(
            units=visual_dim,
            activation=None,
            dtype='float32',
            name='input_projection'
        )

    def pad_to_max_length(self, tensor, max_seq_len):
        return tf.pad(tensor, [[0, 0], [0, max_seq_len - tf.shape(tensor)[1]], [0, 0]], constant_values=0.0)

    def combine_inputs(self, visual_tokens, emotion_tokens, gesture_tokens):
        # Cast to float32 for the Dense layer, then convert to float16
        visual_tokens = tf.cast(visual_tokens, tf.float32)
        emotion_tokens = tf.cast(emotion_tokens, tf.float32)
        gesture_tokens = tf.cast(gesture_tokens, tf.float32)
        combined = tf.concat([visual_tokens, emotion_tokens, gesture_tokens], axis=-1)
        combined = self.input_projection(combined)           # float32
        combined = tf.cast(combined, tf.float16)             # Convert to float16
        return combined

    def call(
        self,
        visual_tokens,
        emotion_tokens=None,
        gesture_tokens=None,
        gloss_target=None,
        text_target=None,
        training=False,
        masks=None
    ):
        enc_input = self.combine_inputs(visual_tokens, emotion_tokens, gesture_tokens)  # float16
        enc_input = self.encoder_positional_encoding(enc_input)                        # float16
        enc_output = self.encoder(enc_input, training, masks['encoder_padding_mask'])  # float16

        # Decoder for Gloss
        gloss_emb = self.gloss_embedding(gloss_target)           # float16
        gloss_emb = self.decoder_positional_encoding(gloss_emb)  # float16
        gloss_dec_output = self.gloss_decoder(
            gloss_emb, enc_output, training, masks['combined_mask'], masks['cross_attention_mask']
        )
        gloss_predictions = self.gloss_output_layer(gloss_dec_output)  # float32

        # Decoder for Text
        text_emb = self.text_embedding(text_target)               # float16
        text_emb = self.decoder_positional_encoding(text_emb)      # float16
        text_dec_output = self.text_decoder(
            text_emb, enc_output, training, masks['combined_mask'], masks['cross_attention_mask']
        )
        text_predictions = self.text_output_layer(text_dec_output)  # float32

        # Return both predictions
        return gloss_predictions, text_predictions

    def handle_oov_tokens(self, predictions, inverse_vocab, max_length=50):
        return handle_oov_tokens(predictions, inverse_vocab, max_length)

    def generate_alignment(self, reference_tokens, hypothesis_tokens):
        return generate_alignment(reference_tokens, hypothesis_tokens)

    def encode_inputs(self, visual_tokens, emotion_tokens, gesture_tokens, training, attention_mask=None):
        enc_input = self.combine_inputs(visual_tokens, emotion_tokens, gesture_tokens)
        enc_input = self.encoder_positional_encoding(enc_input)
        enc_output = self.encoder(enc_input, training, attention_mask)
        return enc_output

    def generate_translation(
        self,
        enc_output,
        embedding_layer,
        decoder,
        output_layer,
        max_length,
        start_token_id,
        end_token_id,
        training=False,
        attention_mask=None,
        cross_attention_mask=None
    ):
        # Initialize the decoder input with the start token
        decoder_input = tf.constant([[start_token_id]], dtype=tf.int32)
        for _ in range(max_length):
            emb = embedding_layer(decoder_input)
            emb = self.decoder_positional_encoding(emb)
            dec_output = decoder(emb, enc_output, training, attention_mask, cross_attention_mask)
            logits = output_layer(dec_output)
            predicted_id = tf.argmax(logits, axis=-1)
            decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)
            if tf.reduce_any(tf.equal(predicted_id, end_token_id)):
                break
        return decoder_input

    def beam_search_decode(
        self,
        enc_output,
        embedding_layer,
        decoder,
        output_layer,
        max_length,
        start_token_id,
        end_token_id,
        beam_size=3,
        alpha=0.6,
        training=False,
        attention_mask=None
    ):
        """
        Generates translation using beam search decoding.

        Args:
            enc_output (tf.Tensor): Encoder output.
            embedding_layer (tf.keras.layers.Layer): Embedding layer.
            decoder (tf.keras.layers.Layer): Decoder layer.
            output_layer (tf.keras.layers.Layer): Output dense layer.
            max_length (int): Maximum length of the generated sequence.
            start_token_id (int): Start token ID.
            end_token_id (int): End token ID.
            beam_size (int, optional): Number of beams. Defaults to 3.
            alpha (float, optional): Length penalty factor. Defaults to 0.6.
            training (bool, optional): Training flag. Defaults to False.
            attention_mask (dict, optional): Attention masks. Defaults to None.

        Returns:
            list: List of generated token ID sequences.
        """
        # Initialize beams with the start token
        beams = [([start_token_id], 0.0)]
        completed_beams = []

        for _ in range(max_length):
            all_candidates = []
            for seq, score in beams:
                if seq[-1] == end_token_id:
                    completed_beams.append((seq, score))
                    continue
                decoder_input = tf.expand_dims(seq, 0)  # Batch size 1
                emb = embedding_layer(decoder_input)
                emb = self.decoder_positional_encoding(emb)
                dec_output = decoder(emb, enc_output, training, attention_mask['combined_mask'], attention_mask['cross_attention_mask'])
                logits = output_layer(dec_output)
                log_probs = tf.nn.log_softmax(logits[:, -1, :], axis=-1)
                top_k = tf.math.top_k(log_probs, k=beam_size)
                for i in range(beam_size):
                    token_id = top_k.indices[0][i].numpy()
                    token_log_prob = top_k.values[0][i].numpy()
                    new_seq = seq + [token_id]
                    new_score = score + token_log_prob
                    all_candidates.append((new_seq, new_score))
            # Select top beam_size beams
            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            beams = ordered[:beam_size]
            # If all beams are completed
            if len(completed_beams) >= beam_size:
                break
        # Select the best completed beams
        if not completed_beams:
            completed_beams = beams
        completed_beams = sorted(completed_beams, key=lambda tup: tup[1], reverse=True)
        return [seq for seq, score in completed_beams[:beam_size]]

    def trainable(self, value=True):
        self.trainable = value

    def predict_translation(
        self,
        visual_tokens,
        emotion_tokens=None,
        gesture_tokens=None,
        masks=None,
        max_length=400,
        mode='text',
        return_token_ids=False
    ):
        enc_output = self.encode_inputs(
            visual_tokens, emotion_tokens, gesture_tokens, training=False, attention_mask=masks['encoder_padding_mask']
        )

        if mode == 'gloss':
            predictions = self.generate_translation(
                enc_output,
                self.gloss_embedding,
                self.gloss_decoder,
                self.gloss_output_layer,
                max_length,
                self.start_token_id,
                self.end_token_id,
                training=False,
                attention_mask={
                    'combined_mask': masks['combined_mask'],
                    'cross_attention_mask': masks['cross_attention_mask']
                }
            )
            predictions = predictions[:, 1:]
        elif mode == 'text':
            predictions = self.generate_translation(
                enc_output,
                self.text_embedding,
                self.text_decoder,
                self.text_output_layer,
                max_length,
                self.start_token_id,
                self.end_token_id,
                training=False,
                attention_mask={
                    'combined_mask': masks['combined_mask'],
                    'cross_attention_mask': masks['cross_attention_mask']
                }
            )
            predictions = predictions[:, 1:]
        else:
            logger.error(f"Invalid mode '{mode}' specified for prediction.")
            return []

        if return_token_ids:
            return predictions.numpy().tolist()
        else:
            return self.handle_oov_tokens(predictions, self.inverse_vocab, max_length)