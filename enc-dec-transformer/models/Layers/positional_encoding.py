import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dropout, TextVectorization
import os

# Definimos la matriz inicial de Positional Encoding
def pos_ratio(pos, i, d_model):
    pos_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    #print('Pos', pos)
    return pos * pos_rates

@tf.function
def positional_encoding(max_seq_length, d_model):

    #print('Position', max_seq_length)
    pos_ratios = pos_ratio(
        np.arange(max_seq_length)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model)

    pos_ratios[:, 0::2] = np.sin(pos_ratios[:, 0::2])
    pos_ratios[:, 1::2] = np.cos(pos_ratios[:, 1::2])
    pos_encoding = pos_ratios[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

#Definimos el procesamiento previo a 
class PreLayer(Layer):
  def __init__(self, d_model, vocab_size, max_seq_length, enc_dec = 0):
      super(PreLayer, self).__init__()
      self.d_model = d_model
      self.max_seq_length = max_seq_length
      self.enc_dec = enc_dec

      #Vectorizers
      #Para adapt y obtener vocabulario.
      if enc_dec == 0:
        self.vectorizer = TextVectorization(max_tokens= vocab_size, output_mode='int', output_sequence_length = max_seq_length, standardize = None)
      elif enc_dec == 1:
        self.vectorizer = TextVectorization(max_tokens= vocab_size, output_mode='int', output_sequence_length = max_seq_length + 1, standardize = None)

      #Embedding
      self.embedding = Embedding(input_dim= vocab_size, output_dim= d_model, mask_zero=True) #Mask Zero sirve para manejar los tokens de padding sin que se les asigne un peso significativo.

      #Positional Encoding
      self.pos_encoding = positional_encoding(max_seq_length, d_model)

      # Dropout
      self.dropout = Dropout(0.1)

  @tf.function
  def adapt(self, dataset):
    self.vectorizer.adapt(dataset)
    return self.vectorizer

  @tf.function
  def get_vocabulary(self):
    return self.vectorizer.get_vocabulary()

  @tf.function
  def call(self, inputs, training):

        x = inputs

        #print('Inputs',inputs)

        # Mascara de padding extra para la logica de atencion.
        mask = tf.cast(tf.not_equal(x, 0), tf.float32)

        #print('Padding Mask',mask)

        # Embedding y positional encoding
        x = self.embedding(x)
        #print('Embedded',x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) #Equilibra la magnitud del embedding y del positional encoding para que el modelo aprenda más rápido.
        seq_len = tf.shape(x)[1]
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        # Mascara para atenttion.
        padding_mask = mask[:, tf.newaxis, tf.newaxis, :]
        #print('Masked',padding_mask)
        return x, padding_mask