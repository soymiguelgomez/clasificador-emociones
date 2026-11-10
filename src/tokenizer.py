"""Tokenizador de palabras en Python puro.

Se usa exactamente el mismo código para entrenar y para predecir en la web, y el
vocabulario se guarda en JSON, de modo que la aplicación no necesita reajustarlo.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

PAD, OOV = "<pad>", "<unk>"


class WordTokenizer:
    def __init__(self, vocab: list[str], max_len: int):
        if vocab[:2] != [PAD, OOV]:
            raise ValueError("El vocabulario debe empezar por <pad> y <unk>")
        self.vocab = vocab
        self.max_len = max_len
        self.index = {w: i for i, w in enumerate(vocab)}

    @classmethod
    def fit(cls, texts, max_tokens: int, max_len: int) -> "WordTokenizer":
        """Construye el vocabulario SOLO con los textos de entrenamiento: las palabras
        más frecuentes (empates en orden alfabético para que sea reproducible)."""
        counts = Counter(tok for text in texts for tok in text.split())
        words = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[: max_tokens - 2]
        return cls([PAD, OOV] + [w for w, _ in words], max_len)

    def encode(self, texts) -> np.ndarray:
        """Convierte textos limpios en una matriz (n, max_len) de índices.
        Se conservan las primeras max_len palabras y se rellena con 0 al final."""
        out = np.zeros((len(texts), self.max_len), dtype=np.int32)
        for row, text in enumerate(texts):
            ids = [self.index.get(tok, 1) for tok in text.split()[: self.max_len]]
            out[row, : len(ids)] = ids
        return out

    def save(self, path: Path):
        Path(path).write_text(
            json.dumps({"max_len": self.max_len, "vocab": self.vocab}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "WordTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["vocab"], data["max_len"])
