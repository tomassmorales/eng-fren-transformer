import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
import numpy as np
from multi_head_attention import MultiHeadAttention

# Decoder Layer
class DecoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    @tf.function
    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(enc_output, enc_output, out1, padding_mask) # En el segundo MHA vamos a pasar por Q y K la salida de los encoders.
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3

# Decoder
class Decoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 target_vocab_size, max_seq_length, rate=0.1):
        super(Decoder, self).__init__()

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
    @tf.function
    def call(self, inputs, enc_output, training,
             look_ahead_mask, padding_mask):
      x = inputs

      for i in range(len(self.dec_layers)):
          x = self.dec_layers[i](
              x, enc_output,
              training=training,
              look_ahead_mask=look_ahead_mask,
              padding_mask=padding_mask)
      return x  # (batch, max_seq_length, d_model)