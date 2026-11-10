"""La inferencia en NumPy (producción) debe dar las mismas probabilidades que Keras."""
import importlib.util

import numpy as np
import pytest

from src import config
from src.predict import EmotionPredictor

SENTENCES = [
    "i feel so nervous about my exam tomorrow",
    "i am feeling really grateful for my friends today",
    "i was shocked when i heard the news",
    "i",                                   # una sola palabra
    "zzzz qqqq",                           # solo palabras desconocidas
    " ".join(["i feel sad"] * 40),         # más de 60 palabras (se trunca)
]

HAS_WEIGHTS = (config.MODELS_DIR / "emotion_bilstm_weights.npz").exists()
HAS_KERAS = importlib.util.find_spec("keras") is not None


@pytest.mark.skipif(not HAS_WEIGHTS, reason="Faltan los pesos exportados")
def test_numpy_probabilities_are_valid():
    probs = EmotionPredictor(backend="numpy").predict_proba(SENTENCES)
    assert probs.shape == (len(SENTENCES), config.NUM_CLASSES)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)


@pytest.mark.skipif(not (HAS_WEIGHTS and HAS_KERAS), reason="Requiere TensorFlow y los pesos")
def test_numpy_matches_keras():
    numpy_probs = EmotionPredictor(backend="numpy").predict_proba(SENTENCES)
    keras_probs = EmotionPredictor(backend="keras").predict_proba(SENTENCES)
    assert np.abs(numpy_probs - keras_probs).max() < 1e-4
    assert (numpy_probs.argmax(1) == keras_probs.argmax(1)).all()


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        EmotionPredictor(backend="otro")
