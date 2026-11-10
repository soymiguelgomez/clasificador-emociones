"""Exporta el modelo entrenado para producción.

1. Una copia .keras solo para inferencia (sin el estado del optimizador): el checkpoint
   de entrenamiento guarda también los momentos de Adam, unas 3 veces el tamaño de los
   pesos, que la aplicación no necesita.
2. Los pesos en formato .npz para la inferencia con NumPy (src/numpy_model.py), que
   permite servir el modelo sin instalar TensorFlow.

Uso:  python -m src.export_model
"""
import keras
import numpy as np

from src import config

TRAINING_MODEL = config.MODELS_DIR / "emotion_bilstm.keras"
INFERENCE_MODEL = config.MODELS_DIR / "emotion_bilstm_inference.keras"
NUMPY_WEIGHTS = config.MODELS_DIR / "emotion_bilstm_weights.npz"


def export_inference_model(src=TRAINING_MODEL, dst=INFERENCE_MODEL):
    trained = keras.models.load_model(src)
    # Misma arquitectura reconstruida desde su configuración: queda sin compilar y,
    # por tanto, sin optimizador. (clone_model conserva la compilación en Keras 3.)
    inference = keras.Model.from_config(trained.get_config())
    inference.set_weights(trained.get_weights())
    if getattr(inference, "optimizer", None) is not None:
        raise RuntimeError("El modelo de inferencia no debería tener optimizador")
    inference.save(dst)
    print(f"Modelo de inferencia: {dst.relative_to(config.ROOT_DIR)} "
          f"({dst.stat().st_size / 1e6:.1f} MB)")
    export_numpy_weights(inference)
    return dst


def export_numpy_weights(model, dst=NUMPY_WEIGHTS):
    fw_kernel, fw_rec, fw_bias, bw_kernel, bw_rec, bw_bias = model.get_layer("bilstm").get_weights()
    dense_kernel, dense_bias = model.get_layer("dense").get_weights()
    out_kernel, out_bias = model.get_layer("emotion").get_weights()
    np.savez_compressed(
        dst,
        embedding=model.get_layer("embedding").get_weights()[0],
        fw_kernel=fw_kernel, fw_recurrent=fw_rec, fw_bias=fw_bias,
        bw_kernel=bw_kernel, bw_recurrent=bw_rec, bw_bias=bw_bias,
        dense_kernel=dense_kernel, dense_bias=dense_bias,
        out_kernel=out_kernel, out_bias=out_bias,
    )
    print(f"Pesos para NumPy: {dst.relative_to(config.ROOT_DIR)} "
          f"({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


if __name__ == "__main__":
    export_inference_model()
