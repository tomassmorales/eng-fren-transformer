import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
from Layers.positional_encoding import PreLayer
from Layers.multi_head_attention import create_look_ahead_mask
from Layers.encoder import Encoder
from Layers.decoder import Decoder

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, max_seq_length, rate=0.1):
        super(Transformer, self).__init__()

        #PreLayers
        self.pre_layer_encoder = PreLayer(d_model= d_model, vocab_size= input_vocab_size, max_seq_length=max_seq_length, enc_dec = 0)
        self.pre_layer_decoder = PreLayer(d_model= d_model, vocab_size= target_vocab_size, max_seq_length=max_seq_length, enc_dec= 1)

        #Encoder
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)

        #Decoder
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        #Final Layer
        self.final_layer = Dense(target_vocab_size, activation='softmax') # Usamos el target vocab size para tener la cantidad de tokens como clases.

    @tf.function
    def call(self, inputs, training):
        encoder_inputs, decoder_inputs = inputs

        x_encoder, encoder_padding_mask = self.pre_layer_encoder(encoder_inputs, training = training) # Computamos Embedding y Positional Enconding en el PreLayer del Encoder

        enc_output, enc_mask = self.encoder(x_encoder, training=training, mask=encoder_padding_mask) # Pasamos la salida de la PreLayer del Encoder al Encoder.

        x_decoder,_ = self.pre_layer_decoder(decoder_inputs, training = training)

        seq_len = tf.shape(decoder_inputs)[1] # Obtenemos la longitud de las secuencias de los inputs del decoder.
        look_ahead_mask = create_look_ahead_mask(seq_len) # Creamos la mascara de look ahead.

        # Combinamos la look ahead mask con la de decoder
        # Máscara de padding para la secuencia de entrada del decoder (target)
        # Esta es la que se usará en combinación con la look_ahead_mask

        dec_target_padding_mask = tf.cast(tf.not_equal(decoder_inputs, 0), tf.float32)
        dec_target_padding_mask = dec_target_padding_mask[:, tf.newaxis, tf.newaxis, :] # Ajustamos las dimensiones

        # La look_ahead_mask y la dec_target_padding_mask se combinan para
        # el primer MHA (self-attention) en cada DecoderLayer.
        combined_mask = tf.maximum(look_ahead_mask, dec_target_padding_mask)

        dec_output = self.decoder(x_decoder, enc_output, training=training, look_ahead_mask=combined_mask, padding_mask=enc_mask)# encoder_padding_mask

        final_output = self.final_layer(dec_output) # Capa de prediccion de siguiente Token
        return final_output