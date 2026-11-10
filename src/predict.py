"""Predicción con el modelo ya entrenado (no se reentrena nada aquí).

Uso desde consola:  python -m src.predict "i feel so happy and grateful today"

Motor de inferencia (variable de entorno MODEL_BACKEND):
  keras  carga el .keras con TensorFlow (por defecto si TensorFlow está instalado)
  numpy  usa los mismos pesos con src/numpy_model.py, sin TensorFlow (producción)
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

from src import config
from src.text_cleaning import clean_text
from src.tokenizer import WordTokenizer

KERAS_MODEL = config.MODELS_DIR / "emotion_bilstm_inference.keras"
NUMPY_WEIGHTS = config.MODELS_DIR / "emotion_bilstm_weights.npz"
DEFAULT_TOKENIZER = config.MODELS_DIR / "tokenizer.json"


def default_backend() -> str:
    backend = os.environ.get("MODEL_BACKEND")
    if backend:
        return backend.lower()
    return "keras" if importlib.util.find_spec("keras") else "numpy"


class EmotionPredictor:
    """Carga una sola vez el modelo y el vocabulario y aplica a cada texto las mismas
    transformaciones del entrenamiento: clean_text -> tokenizer.encode -> modelo."""

    def __init__(self, backend: str | None = None, tokenizer_path: Path = DEFAULT_TOKENIZER):
        self.backend = backend or default_backend()
        if self.backend == "keras":
            model_path = KERAS_MODEL
        elif self.backend == "numpy":
            model_path = NUMPY_WEIGHTS
        else:
            raise ValueError(f"MODEL_BACKEND desconocido: {self.backend!r} (usa keras o numpy)")
        for path in (model_path, tokenizer_path):
            if not Path(path).exists():
                raise FileNotFoundError(f"No se encontró {path}. Ejecuta: python -m src.train")

        if self.backend == "keras":
            import keras  # importación diferida: solo se paga si se usa este motor
            self.model = keras.models.load_model(model_path)
        else:
            from src.numpy_model import NumpyEmotionModel
            self.model = NumpyEmotionModel(model_path)
        self.tokenizer = WordTokenizer.load(tokenizer_path)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        cleaned = [clean_text(t) for t in texts]
        ids = self.tokenizer.encode(cleaned)
        return self.model.predict(ids, verbose=0)

    def predict(self, text: str) -> dict:
        cleaned = clean_text(text)
        tokens = cleaned.split()
        known = sum(1 for t in tokens[: self.tokenizer.max_len] if t in self.tokenizer.index)
        probs = self.predict_proba([text])[0]
        order = np.argsort(probs)[::-1]
        return {
            "label_id": int(order[0]),
            "label_en": config.LABEL_NAMES_EN[order[0]],
            "label_es": config.LABEL_NAMES_ES[order[0]],
            "confidence": float(probs[order[0]]),
            "probabilities": [
                {"label_id": int(i), "label_en": config.LABEL_NAMES_EN[i],
                 "label_es": config.LABEL_NAMES_ES[i], "probability": float(probs[i])}
                for i in order
            ],
            "cleaned_text": cleaned,
            "n_words": len(tokens),
            "n_known_words": known,
            "truncated": len(tokens) > self.tokenizer.max_len,
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Uso: python -m src.predict "texto en inglés"')
    result = EmotionPredictor().predict(" ".join(sys.argv[1:]))
    print(f"Emoción: {result['label_es']} ({result['confidence']:.1%})")
    for p in result["probabilities"]:
        print(f"  {p['label_es']:<10} {p['probability']:.3f}")
