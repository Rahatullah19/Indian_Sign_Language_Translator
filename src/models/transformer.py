# transformer.py
from pydoc import text
import tensorflow as tf
import logging
import difflib
import numpy as np
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.models import Model
from models.custom_layers import (
    TransformerEncoder,
    TransformerDecoder,
    PositionalEncoding,
    create_padding_mask,
    create_look_ahead_mask,
    handle_oov_tokens,
    generate_alignment
)
from models.custom_layers import create_padding_mask, create_look_ahead_mask, create_attention_masks

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
        num_layers=2,
        num_heads=8,
        ff_dim=512,
        dropout_rate=0.1,
        start_token_id=1,
        end_token_id=2,
        pad_token_id=0,
        max_positional_encoding=1000
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

        self.gloss_embedding = Embedding(
            input_dim=gloss_vocab_size,
            output_dim=visual_dim,
            mask_zero=True
        )
        self.text_embedding = Embedding(
            input_dim=text_vocab_size,
            output_dim=visual_dim,
            mask_zero=True
        )

    def pad_to_max_length(self, tensor, max_seq_len):
        current_length = tf.shape(tensor)[1]
        padding_length = max_seq_len - current_length
        padding = tf.zeros([tf.shape(tensor)[0], padding_length, tf.shape(tensor)[2]], dtype=tf.float32)
        return tf.concat([tensor, padding], axis=1)

    def combine_inputs(self, visual_tokens, emotion_tokens, gesture_tokens):
        max_seq_len = tf.reduce_max([
            tf.shape(visual_tokens)[1],
            tf.shape(emotion_tokens)[1],
            tf.shape(gesture_tokens)[1]
        ])

        combined_seq_len = max_seq_len * 3
        tf.debugging.assert_less_equal(
            combined_seq_len,
            self.encoder_positional_encoding.max_len,
            message=f"Combined sequence length {combined_seq_len} exceeds max_positional_encoding {self.encoder_positional_encoding.max_len}"
        )

        visual_tokens = self.pad_to_max_length(visual_tokens, max_seq_len)
        emotion_tokens = self.pad_to_max_length(emotion_tokens, max_seq_len)
        gesture_tokens = self.pad_to_max_length(gesture_tokens, max_seq_len)
        logger.info(f"Shapes after padding: Visual: {visual_tokens.shape}, Emotion: {emotion_tokens.shape}, Gesture: {gesture_tokens.shape}")
        combined_tokens = tf.concat([visual_tokens, emotion_tokens, gesture_tokens], axis=1)
        logger.info(f"Combined Tokens Shape: {combined_tokens.shape}")

        return combined_tokens

    def call(
        self,
        visual_tokens,
        emotion_tokens,
        gesture_tokens,
        gloss_target=None,
        text_target=None,
        training=False,
        masks=None
    ):
        combined_tokens = self.combine_inputs(visual_tokens, emotion_tokens, gesture_tokens)
        enc_pos_encoding = self.encoder_positional_encoding(combined_tokens)
        combined_tokens = combined_tokens + enc_pos_encoding

        if masks is not None:
            encoder_padding_mask = masks.get('encoder_padding_mask', None)    
            combined_mask = masks.get('combined_mask', None)              
            cross_attention_mask = masks.get('cross_attention_mask', None)   
        else:
            encoder_padding_mask = None
            combined_mask = None
            cross_attention_mask = None

        enc_output = self.encoder(combined_tokens, training, encoder_padding_mask)

        gloss_predictions = None
        text_predictions = None

        # Decode Gloss
        if gloss_target is not None:
            gloss_embeddings = self.gloss_embedding(gloss_target)
            gloss_embeddings = gloss_embeddings + self.decoder_positional_encoding(gloss_embeddings)
            gloss_decoder_output = self.gloss_decoder(
                gloss_embeddings,
                enc_output,
                training,
                attention_mask=combined_mask,            # Self-Attention Mask
                cross_attention_mask=cross_attention_mask # Cross-Attention Mask
            )
            gloss_predictions = self.gloss_output_layer(gloss_decoder_output)

        # Decode Text
        if text_target is not None:
            text_embeddings = self.text_embedding(text_target)
            text_embeddings = text_embeddings + self.decoder_positional_encoding(text_embeddings)
            text_decoder_output = self.text_decoder(
                text_embeddings,
                enc_output,
                training,
                attention_mask=combined_mask,            # Self-Attention Mask
                cross_attention_mask=cross_attention_mask # Cross-Attention Mask
            )
            text_predictions = self.text_output_layer(text_decoder_output)

        # Inference Mode (if targets are None)
        if gloss_target is None and text_target is None:
            gloss_predictions = self.generate_translation(
                enc_output,
                self.gloss_embedding,
                self.gloss_decoder,
                self.gloss_output_layer,
                max_length=50,
                start_token_id=self.start_token_id,
                end_token_id=self.end_token_id,
                training=training,
                attention_mask=None  
            )
            text_predictions = self.generate_translation(
                enc_output,
                self.text_embedding,
                self.text_decoder,
                self.text_output_layer,
                max_length=50,
                start_token_id=self.start_token_id,
                end_token_id=self.end_token_id,
                training=training,
                attention_mask=None  
            )

        return gloss_predictions, text_predictions

    def handle_oov_tokens(self, predictions, max_length=50):
        return custom_handle_oov_tokens(predictions, self.inverse_vocab, max_length)

    def generate_alignment(self, reference_tokens, hypothesis_tokens):
        return custom_generate_alignment(reference_tokens, hypothesis_tokens)

    def encode_inputs(self, visual_tokens, emotion_tokens, gesture_tokens, training, attention_mask=None):
        combined_tokens = self.combine_inputs(visual_tokens, emotion_tokens, gesture_tokens)
        enc_pos_encoding = self.encoder_positional_encoding(combined_tokens)
        combined_tokens += enc_pos_encoding
        enc_output = self.encoder(combined_tokens, training, attention_mask)
        return enc_output

    def handle_oov_tokens(self, predictions, max_length=50):
        return custom_handle_oov_tokens(predictions, self.inverse_vocab, max_length)

    def generate_alignment(self, reference_tokens, hypothesis_tokens):
        return custom_generate_alignment(reference_tokens, hypothesis_tokens)

    def encode_inputs(self, visual_tokens, emotion_tokens, gesture_tokens, training, attention_mask=None):
        combined_tokens = self.combine_inputs(visual_tokens, emotion_tokens, gesture_tokens)
        enc_pos_encoding = self.encoder_positional_encoding(combined_tokens)
        combined_tokens += enc_pos_encoding
        enc_output = self.encoder(combined_tokens, training, attention_mask)
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
        attention_mask=None
    ):
        # Initialize the translation with the start token
        decoder_input = tf.expand_dims([start_token_id], 0) 

        translations = []

        for _ in range(max_length):
            embedded = embedding_layer(decoder_input)  
            pos_encoding = self.decoder_positional_encoding(embedded)
            embedded += pos_encoding

            # Create masks for the decoder
            look_ahead_mask = create_look_ahead_mask(tf.shape(decoder_input)[1])
            decoder_padding_mask = create_padding_mask(decoder_input, self.pad_token_id)
            combined_mask = tf.maximum(decoder_padding_mask, look_ahead_mask)

            decoded = decoder(
                embedded,
                enc_output,
                training,
                attention_mask=combined_mask,
                cross_attention_mask=attention_mask
            )

            predictions = output_layer(decoded)  
            predictions = predictions[:, -1:, :]  
            predicted_id = tf.argmax(predictions, axis=-1)  

            decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

            if predicted_id.numpy()[0][0] == end_token_id:
                break

            translations.append(predicted_id.numpy()[0][0])

        return translations

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
        beam = [([start_token_id], 0.0)]
        for _ in range(max_length):
            all_candidates = []
            for seq, score in beam:
                if seq[-1] == end_token_id:
                    all_candidates.append((seq, score))
                    continue
                decoder_input = tf.convert_to_tensor(seq, dtype=tf.int32)[tf.newaxis, :] 
                embedded_input = tf.cast(embedding_layer(decoder_input), tf.float32)

                dec_output = decoder(
                    embedded_input,
                    enc_output,
                    training=training,
                    attention_mask=None,   
                    cross_attention_mask=None
                )

                logits = output_layer(dec_output)[:, -1, :]
                log_probs = tf.nn.log_softmax(logits, axis=-1).numpy()[0]
                top_indices = log_probs.argsort()[-beam_size:][::-1]
                for idx in top_indices:
                    all_candidates.append((seq + [idx], score + log_probs[idx]))

            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            beam = ordered[:beam_size]
            if all(seq[-1] == end_token_id for seq, _ in beam):
                break

        return [seq for seq, _ in beam]

    def trainable(self, value=True):
        self.trainable = value

    def predict_translation(
        self,
        visual_tokens,
        emotion_tokens,
        gesture_tokens,
        inverse_vocab,
        max_length=100,
        start_token_id=1,
        end_token_id=2,
        beam_size=3,
        alpha=0.6
    ):
        enc_output = self.encode_inputs(visual_tokens, emotion_tokens, gesture_tokens, training=False)

        gloss_translation_ids = self.generate_translation(
            enc_output,
            self.gloss_embedding,
            self.gloss_decoder,
            self.gloss_output_layer,
            max_length,
            start_token_id,
            end_token_id,
            training=False
        )
        gloss_translation = custom_handle_oov_tokens(gloss_translation_ids, inverse_vocab, max_length)

        text_translation_ids = self.generate_translation(
            enc_output,
            self.text_embedding,
            self.text_decoder,
            self.text_output_layer,
            max_length,
            start_token_id,
            end_token_id,
            training=False
        )
        text_translation = custom_handle_oov_tokens(text_translation_ids, inverse_vocab, max_length)

        return gloss_translation, text_translation