import tensorflow as tf
import pandas as pd
import numpy as np

class DataTools():
    def __init__(self, train_df, val_df):
        self.train_df = train_df
        self.val_df = val_df
    
    def get_max_sequence(self):
        # Calcular longitud máxima de secuencia en inglés y francés
        max_seq_length_en_train = self.train_df['english'].apply(lambda x: len(x.split())).max()
        max_seq_length_fr_train = self.train_df['french'].apply(lambda x: len(x.split())).max()
        max_seq_length_en_val = self.val_df['english'].apply(lambda x: len(x.split())).max()
        max_seq_length_fr_val = self.val_df['french'].apply(lambda x: len(x.split())).max()

        maximum = np.argmax([max_seq_length_en_train, max_seq_length_fr_train, max_seq_length_en_val, max_seq_length_fr_val])

        return maximum

class DataPipeline:
    def __init__(self, train_path, val_path):
        self.train_df = pd.read_parquet(train_path)
        self.val_df = pd.read_parquet(val_path)
        self.data_tools = DataTools(train_df=self.train_df, val_df=self.val_df)

    def get_max_sequence(self):
        return self.data_tools.get_max_sequence()

    def prepare_dataset(self, vectorize_input, vectorize_target):
        def _prepare_dataset(eng_batch, fr_batch):
            encoder_inputs = vectorize_input(eng_batch)
            decoder_inputs = vectorize_target(fr_batch)
            labels = decoder_inputs[:, 1:]
            decoder_inputs = decoder_inputs[:, :-1]
            return (encoder_inputs, decoder_inputs), labels
        
        return _prepare_dataset

    def create_datasets(self, BATCH_SIZE, vectorize_input, vectorize_target):
        prepare_func = self.prepare_dataset(vectorize_input, vectorize_target)
        
        train_ds = tf.data.Dataset.from_tensor_slices(
            (self.train_df["english"], self.train_df["french"])
        ).batch(BATCH_SIZE).map(prepare_func, num_parallel_calls=tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices(
            (self.val_df["english"], self.val_df["french"])
        ).batch(BATCH_SIZE).map(prepare_func, num_parallel_calls=tf.data.AUTOTUNE)

        return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)