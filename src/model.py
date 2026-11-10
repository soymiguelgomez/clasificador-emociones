"""Arquitectura de la red neuronal: Embedding + LSTM bidireccional + capas densas."""
import keras
from keras import layers

from src import config

MAX_TOKENS = 30_000   # tamaño del vocabulario (incluye relleno y palabra desconocida)
MAX_LEN = 60          # palabras por texto; cubre el 99,9 % del entrenamiento
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DENSE_UNITS = 64
DROPOUT = 0.3
LEARNING_RATE = 1e-3


def build_model(vocab_size: int = MAX_TOKENS) -> keras.Model:
    """Construye y compila el clasificador de emociones.

    - Embedding: convierte cada índice de palabra en un vector denso aprendido.
      mask_zero=True hace que la LSTM ignore el relleno (índice 0).
    - SpatialDropout1D: apaga dimensiones completas del embedding para reducir
      la dependencia de palabras concretas (regularización).
    - Bidirectional(LSTM): lee la frase de izquierda a derecha y al revés, de modo
      que cada emoción se interpreta con su contexto ("not happy" frente a "happy").
    - Dense + Dropout: combina las características y reduce el sobreajuste.
    - Dense(softmax): devuelve una probabilidad por emoción que suma 1.
    """
    inputs = keras.Input(shape=(MAX_LEN,), dtype="int32", name="token_ids")
    x = layers.Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True, name="embedding")(inputs)
    x = layers.SpatialDropout1D(0.2, name="spatial_dropout")(x)
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS), name="bilstm")(x)
    x = layers.Dropout(DROPOUT, name="dropout_1")(x)
    x = layers.Dense(DENSE_UNITS, activation="relu", name="dense")(x)
    x = layers.Dropout(DROPOUT, name="dropout_2")(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax", name="emotion")(x)

    model = keras.Model(inputs, outputs, name="emotion_bilstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        # Las etiquetas son enteros 0..5, por eso la versión "sparse" de la entropía cruzada.
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
