import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
import numpy as np
from multi_head_attention import MultiHeadAttention

# Encoder Layer
class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    @tf.function
    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask) # Computamos el MHA con la salida del PreLayer.
        attn_output = self.dropout1(attn_output, training=training) # Aplicamos Dropout para evitar overfitting.
        out1 = self.layernorm1(x + attn_output) # Hacemos el skip connection o conexion residual para no perder el contexto.

        ffn_output = self.ffn(out1) # Pasamos la salida del skip connection por el FFN. (max_seq_length, d_model) x (d_model, dff) => (max_seq_length, dff) x (dff, d_model) => (max_seq_length, d_model)
        ffn_output = self.dropout2(ffn_output, training=training) # Dropout de vuelta para reducir overfitting.
        out2 = self.layernorm2(out1 + ffn_output) # Hacemos nuevamente el skip connection para no perder este contexto.
        return out2

# Encoder
class Encoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, max_seq_length, rate=0.1):
        super(Encoder, self).__init__()
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)] # Creamos tantos encoders como le digamos.
        self.num_layers = num_layers

    @tf.function
    def call(self, inputs, training, mask):
      x = inputs
      for i in range(self.num_layers):
          x = self.enc_layers[i](x, training = training, mask = mask) # Pasamos la mascara de padding y el
      return x, mask  # (batch, max_seq_length, d_model)