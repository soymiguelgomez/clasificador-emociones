"""Inferencia de la red entrenada usando solo NumPy (sin TensorFlow).

Reproduce paso a paso la red de src/model.py con los pesos aprendidos por Keras:
  Embedding -> LSTM bidireccional (con máscara de relleno) -> Dense ReLU -> Dense softmax.
Las capas de dropout no se aplican en inferencia, igual que en Keras.

Se usa en producción porque TensorFlow ocupa más de 400 MB de RAM, mientras que esta
versión necesita unos pocos MB más el tamaño de los pesos. La equivalencia numérica
con Keras se comprueba en tests/test_numpy_model.py.
"""
from pathlib import Path

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm_last_state(x, lengths, kernel, recurrent, bias):
    """Recorre la secuencia con una LSTM y devuelve el último estado oculto válido.

    Ecuaciones (orden de compuertas de Keras: i, f, c, o):
      z = x_t·W + h_{t-1}·U + b
      i = σ(z_i), f = σ(z_f), g = tanh(z_c), o = σ(z_o)
      c_t = f·c_{t-1} + i·g ;  h_t = o·tanh(c_t)
    En los pasos de relleno (t >= longitud) el estado no cambia, como hace la máscara.
    """
    batch, steps, _ = x.shape
    units = recurrent.shape[0]
    h = np.zeros((batch, units), dtype=np.float32)
    c = np.zeros((batch, units), dtype=np.float32)
    x_proj = x @ kernel + bias  # proyección de todas las entradas de una vez
    for t in range(int(lengths.max(initial=0))):
        z = x_proj[:, t] + h @ recurrent
        i = _sigmoid(z[:, :units])
        f = _sigmoid(z[:, units:2 * units])
        g = np.tanh(z[:, 2 * units:3 * units])
        o = _sigmoid(z[:, 3 * units:])
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        active = (t < lengths)[:, None]
        c = np.where(active, c_new, c)
        h = np.where(active, h_new, h)
    return h


class NumpyEmotionModel:
    def __init__(self, weights_path: Path):
        w = np.load(weights_path)
        self.embedding = w["embedding"]
        self.forward = (w["fw_kernel"], w["fw_recurrent"], w["fw_bias"])
        self.backward = (w["bw_kernel"], w["bw_recurrent"], w["bw_bias"])
        self.dense = (w["dense_kernel"], w["dense_bias"])
        self.output = (w["out_kernel"], w["out_bias"])

    def predict(self, ids: np.ndarray, verbose=0) -> np.ndarray:
        """Misma interfaz que keras.Model.predict: ids (n, max_len) -> probabilidades (n, 6)."""
        ids = np.asarray(ids)
        # El relleno (índice 0) va siempre al final, así que la longitud real es
        # el número de índices distintos de cero.
        lengths = (ids > 0).sum(axis=1)
        reversed_ids = np.zeros_like(ids)
        for row, n in enumerate(lengths):
            reversed_ids[row, :n] = ids[row, :n][::-1]

        h_fw = _lstm_last_state(self.embedding[ids], lengths, *self.forward)
        h_bw = _lstm_last_state(self.embedding[reversed_ids], lengths, *self.backward)
        x = np.concatenate([h_fw, h_bw], axis=1)
        x = np.maximum(0.0, x @ self.dense[0] + self.dense[1])
        logits = x @ self.output[0] + self.output[1]
        logits -= logits.max(axis=1, keepdims=True)  # estabilidad numérica del softmax
        exp = np.exp(logits)
        return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
