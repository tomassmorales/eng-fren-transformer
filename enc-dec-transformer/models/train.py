from training.data_pipeline import DataPipeline
from training.training_utils import create_optimizer, loss_function
from transformer import Transformer

# Hiperparámetros del modelo
D_MODEL = 100
NUM_LAYERS = 1
NUM_HEADS = 1
DFF = D_MODEL * 4
RATE = 0.1
BATCH_SIZE = 128
NUM_EPOCHS = 100
PAD_IDX = 0 # Asumiendo 0 como el índice de relleno

# Rutas dentro del contenedor
TRAIN_DATA_PATH = "/workspace/data/split/eng-french-train.parquet"
VAL_DATA_PATH = "/workspace/data/split/eng-french-val.parquet"

def main():
    # 1. Pipeline de datos
    data_pipeline = DataPipeline(train_path=TRAIN_DATA_PATH, val_path=VAL_DATA_PATH)
    #max_seq_length = data_pipeline.get_max_sequence()
    #print(max_seq_length)
    max_seq_length = 59

    # 2. Inicializar el modelo con capas de vectorización
    # Las capas de vectorización se crean dentro de la clase Transformer
    # y se adaptan aquí antes de crear los datasets.
    transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=5, # El tamaño del vocabulario se ajusta después
        target_vocab_size=5,
        max_seq_length=max_seq_length,
        rate=RATE
    )

    # 3. Adaptar las capas de vectorización del modelo
    transformer.pre_layer_encoder.adapt(data_pipeline.train_df["english"].values)
    transformer.pre_layer_decoder.adapt(data_pipeline.train_df["french"].values)
    
    # 4. Obtener vocabularios y tamaños finales
    source_vocab = transformer.pre_layer_encoder.get_vocabulary()
    target_vocab = transformer.pre_layer_decoder.get_vocabulary()
    input_vocab_size = len(source_vocab)
    target_vocab_size = len(target_vocab)
    
    # Actualizar los tamaños de vocabulario en el modelo.
    transformer.input_vocab_size = input_vocab_size
    transformer.target_vocab_size = target_vocab_size

    # 5. Crear datasets de TensorFlow
    train_ds, val_ds = data_pipeline.create_datasets(
        BATCH_SIZE, transformer.pre_layer_encoder, transformer.pre_layer_decoder)

    # 6. Inicializar optimizador y compilar el modelo
    optimizer = create_optimizer(len(data_pipeline.train_df), BATCH_SIZE, NUM_EPOCHS)
    
    transformer.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    
    # 7. Verificación (ejemplo)
    for (encoder_inputs, decoder_inputs), labels in train_ds.take(1):
        print("\nBatch de ejemplo:")
        print(f"Encoder inputs shape: {encoder_inputs.shape}")
        print(f"Decoder inputs shape: {decoder_inputs.shape}")
        print(f"Labels shape: {labels.shape}")

        sample_idx = 0
        print("\nTexto decodificado (inglés):",
              " ".join(source_vocab[idx] for idx in encoder_inputs[sample_idx].numpy() if idx != PAD_IDX))
        print("Labels (francés):",
              " ".join(target_vocab[idx] for idx in labels[sample_idx].numpy() if idx != PAD_IDX))

    # 8. Entrenamiento
    history = transformer.fit(
        train_ds,
        epochs=NUM_EPOCHS,
        validation_data=val_ds
    )

if __name__ == "__main__":
    main()