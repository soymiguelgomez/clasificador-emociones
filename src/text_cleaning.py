"""Limpieza de texto compartida por el entrenamiento y la aplicación web.

Usar la misma función en ambos lados garantiza que el modelo reciba en producción
textos transformados exactamente igual que durante el entrenamiento.
"""
import re

# Restos de etiquetas HTML y URLs que quedaron al quitar la puntuación del corpus
# original (p. ej. "a href http www"). Se eliminan solo estos tokens concretos.
_MARKUP = re.compile(
    r"\b(?:a\s+(?:rel\s+nofollow\s+)?href|href|https?|www|src|img|nbsp|quot|nofollow)\b"
)
# "amp" proviene de la entidad HTML &amp; y en el corpus funciona como "and".
_AMP = re.compile(r"\bamp\b")
_NON_ALPHA = re.compile(r"[^a-z'\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normaliza un texto al formato del corpus: minúsculas, solo letras, sin marcado."""
    text = str(text).lower()
    text = text.replace("’", "'")
    text = _NON_ALPHA.sub(" ", text)
    # El corpus no contiene apóstrofos ("i m", "don t"), así que se separan igual.
    text = text.replace("'", " ")
    text = _MARKUP.sub(" ", text)
    text = _AMP.sub("and", text)
    return _SPACES.sub(" ", text).strip()
