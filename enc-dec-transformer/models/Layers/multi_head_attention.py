import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
import numpy as np
import os

# Masking
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (max_seq_length, max_seq_length)

# Multi-head Attention
class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0,2,1,3])  # (batch, h, max_seq_length, depth)

    @tf.function
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q) # Pasamos x por la red densa para crear Q. (max_seq_length, d_model) x (d_model, d_model) = (max_seq_length, d_model)
        k = self.wk(k) # Pasamos x por la red densa para crear K. (max_seq_length, d_model) x (d_model, d_model) = (max_seq_length, d_model)
        v = self.wv(v) # Pasamos x por la red densa para crear V. (max_seq_length, d_model) x (d_model, d_model) = (max_seq_length, d_model)

        #Dividimos las dimensiones para pasarlas por las cabezas de atencion correspondientes.
        q = self.split_heads(q, batch_size) # (h, max_seq_length, depth)
        k = self.split_heads(k, batch_size) # (h, max_seq_length, depth)
        v = self.split_heads(v, batch_size) # (h, max_seq_length, depth)

        matmul_qk = tf.matmul(q, k, transpose_b=True) #Multiplicamos las matrices QxK con K traspuesta para matchear las shapes.
        dk = tf.cast(tf.shape(k)[-1], tf.float32) # Escalamos el resultado.
        pre_score = matmul_qk / tf.math.sqrt(dk) # Computamos el score.

        if mask is not None:
            pre_score += (mask * -1e9)

        attention_scores = tf.nn.softmax(pre_score, axis=-1) # Aplicamos softmax al score
        output = tf.matmul(attention_scores, v) # Multiplicamos los scores con la matriz V.

        output = tf.transpose(output, perm=[0,2,1,3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model)) # Concatenamos los resultados de cada head.
        out = self.dense(concat_attention) # Pasamos los resultados por una red densa de dimension d_model.
        return out # Salida de tamaño (max_seq_length, d_model)