# transformer.py
from pydoc import text
import tensorflow as tf
import logging
import difflib
import numpy as np
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.models import Model
from models.custom_layers_test import (
    TransformerEncoder,
    TransformerDecoder,
    PositionalEncoding,
    create_padding_mask,
    create_look_ahead_mask,
    create_attention_masks,
    handle_oov_tokens,
    generate_alignment
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
        num_layers=2,
        num_heads=8,
        ff_dim=512,
        dropout_rate=0.1,
        start_token_id=1,
        end_token_id=2,
        pad_token_id=0,
        max_positional_encoding=5000
    ):
        super(SignLanguageTransformer, self).__init__(name="sign_language_transformer")

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

        # Initialize positional encodings with embed_dim=512
        self.encoder_positional_encoding = PositionalEncoding(
            max_len=max_positional_encoding,
            embed_dim=512,
            name="encoder_positional_encoding"
        )

        self.decoder_positional_encoding = PositionalEncoding(
            max_len=max_positional_encoding,
            embed_dim=512,
            name="decoder_positional_encoding"
        )

        self.embedding_projection = Dense(512, activation=None, name="embedding_projection")

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=512,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name="transformer_encoder"
        )

        self.gloss_decoder = TransformerDecoder(
            num_layers=num_layers,
            embed_dim=512,            
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name="gloss_transformer_decoder"
        )
        self.text_decoder = TransformerDecoder(
            num_layers=num_layers,
            embed_dim=512,            
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name="text_transformer_decoder"
        )

        self.gloss_output_layer = Dense(gloss_vocab_size, dtype='float32', name="gloss_output_layer")
        self.text_output_layer = Dense(text_vocab_size, dtype='float32', name="text_output_layer")

        self.gloss_embedding = Embedding(
            input_dim=gloss_vocab_size,
            output_dim=512,            
            mask_zero=True,
            name="gloss_embedding"
        )
        self.text_embedding = Embedding(
            input_dim=text_vocab_size,
            output_dim=512,            
            mask_zero=True,
            name="text_embedding"
        )

    def pad_to_max_length(self, tensor, max_seq_len):
        current_length = tf.shape(tensor)[1]
        padding_length = tf.maximum(0, max_seq_len - current_length)
        padding = tf.zeros([tf.shape(tensor)[0], padding_length, tf.shape(tensor)[2]], dtype=tensor.dtype)
        padded_tensor = tf.concat([tensor, padding], axis=1)
        return padded_tensor

    def combine_inputs(self, visual_tokens, emotion_tokens, gesture_tokens):
        # Determine the maximum sequence length among the modalities
        max_seq_len = tf.reduce_max([
            tf.shape(visual_tokens)[1],
            tf.shape(emotion_tokens)[1],
            tf.shape(gesture_tokens)[1]
        ])

        # Pad each modality to the maximum sequence length
        visual_tokens = self.pad_to_max_length(visual_tokens, max_seq_len)
        emotion_tokens = self.pad_to_max_length(emotion_tokens, max_seq_len)
        gesture_tokens = self.pad_to_max_length(gesture_tokens, max_seq_len)

        # Concatenate along the embedding dimension (last axis)
        combined_tokens = tf.concat([visual_tokens, emotion_tokens, gesture_tokens], axis=-1)  # (batch, max_seq_len, 1536)

        # Project combined embeddings to 512 dimensions
        projected_tokens = self.embedding_projection(combined_tokens)
        return projected_tokens

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
        combined = self.combine_inputs(visual_tokens, emotion_tokens, gesture_tokens)
        combined = self.encoder_positional_encoding(combined)
        enc_output = self.encoder(combined, training=training, attention_mask=masks['encoder_padding_mask'])

        # Process Gloss Decoder
        if gloss_target is not None:
            gloss_embedded = self.gloss_embedding(gloss_target)
            gloss_embedded = self.decoder_positional_encoding(gloss_embedded)
            gloss_dec_output = self.gloss_decoder(
                gloss_embedded,
                enc_output,
                training=training,
                attention_mask=masks['combined_mask'],
                cross_attention_mask=masks['cross_attention_mask']
            )
            gloss_predictions = self.gloss_output_layer(gloss_dec_output)
        else:
            gloss_predictions = None

        # Process Text Decoder
        if text_target is not None:
            text_embedded = self.text_embedding(text_target)
            text_embedded = self.decoder_positional_encoding(text_embedded)
            text_dec_output = self.text_decoder(
                text_embedded,
                enc_output,
                training=training,
                attention_mask=masks['combined_mask'],
                cross_attention_mask=masks['cross_attention_mask']
            )
            text_predictions = self.text_output_layer(text_dec_output)
        else:
            text_predictions = None

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
        return handle_oov_tokens(predictions, self.inverse_vocab, max_length)

    def generate_alignment(self, reference_tokens, hypothesis_tokens):
        return generate_alignment(reference_tokens, hypothesis_tokens)

    def encode_inputs(self, visual_tokens, emotion_tokens, gesture_tokens, training, attention_mask=None):
        combined = self.combine_inputs(visual_tokens, emotion_tokens, gesture_tokens)
        combined = self.encoder_positional_encoding(combined)
        enc_output = self.encoder(combined, training=training, attention_mask=attention_mask)
        return enc_output

    def generate_translation(
        self,
        visual_tokens,
        emotion_tokens,
        gesture_tokens,
        masks,
        inverse_vocab,
        max_length=50,
        attention_mask=None
    ):
        enc_output = self.encode_inputs(
            visual_tokens, emotion_tokens, gesture_tokens, training=False, attention_mask=masks['encoder_padding_mask']
        )

        # Generate gloss translation
        gloss_translation = self._generate_single_translation(
            decoder=self.gloss_decoder,
            output_layer=self.gloss_output_layer,
            embeddings=self.gloss_embedding,
            positional_encoding=self.encoder_positional_encoding,
            enc_output=enc_output,
            masks=masks,
            inverse_vocab=inverse_vocab,
            max_length=max_length
        )

        # Generate text translation
        text_translation = self._generate_single_translation(
            decoder=self.text_decoder,
            output_layer=self.text_output_layer,
            embeddings=self.text_embedding,
            positional_encoding=self.encoder_positional_encoding,
            enc_output=enc_output,
            masks=masks,
            inverse_vocab=inverse_vocab,
            max_length=max_length
        )

        return gloss_translation, text_translation

    def _generate_single_translation(
        self,
        decoder,
        output_layer,
        embeddings,
        positional_encoding,
        enc_output,
        masks,
        inverse_vocab,
        max_length
    ):
        # Implement beam search or greedy decoding here
        # For simplicity, using greedy decoding
        tokens = tf.constant([[self.start_token_id]], dtype=tf.int32)
        for _ in range(max_length):
            embedded = embeddings(tokens)
            embedded = positional_encoding(embedded)
            dec_output = decoder(
                embedded,
                enc_output,
                training=False,
                attention_mask=masks['combined_mask'],
                cross_attention_mask=masks['cross_attention_mask']
            )
            logits = output_layer(dec_output)
            next_token = tf.argmax(logits[:, -1, :], axis=-1)
            tokens = tf.concat([tokens, tf.expand_dims(next_token, axis=1)], axis=1)
            if tf.reduce_any(tf.equal(next_token, self.end_token_id)):
                break
        return self.handle_oov_tokens(tokens.numpy()[0], max_length)

    def trainable(self, value=True):
        super(SignLanguageTransformer, self).trainable = value

    def predict_translation(
        self,
        visual_tokens,
        emotion_tokens,
        gesture_tokens,
        masks,
        inverse_vocab,
        max_length=50
    ):
        return self.generate_translation(
            visual_tokens=visual_tokens,
            emotion_tokens=emotion_tokens,
            gesture_tokens=gesture_tokens,
            masks=masks,
            inverse_vocab=inverse_vocab,
            max_length=max_length
        )