"""Línea base clásica (no neuronal): TF-IDF + regresión logística.

Sirve como punto de comparación: si la red neuronal no la supera, su complejidad
no estaría justificada. Se ajusta solo con entrenamiento y se mide en validación y prueba.
Uso:  python -m src.baseline
Salida: reports/baseline_metrics.json
"""
import json
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src import config
from src.train import load_split


def scores(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, average=None, labels=range(config.NUM_CLASSES))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_per_class": {config.LABEL_NAMES_EN[i]: float(v) for i, v in enumerate(per_class)},
    }


def main():
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    start = time.time()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                                 sublinear_tf=True)
    x_train = vectorizer.fit_transform(train["text"])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_train, train["label"])
    result = {
        "model": "TF-IDF (1-2 gramas, 200k rasgos) + LogisticRegression(class_weight=balanced)",
        "fit_minutes": round((time.time() - start) / 60, 1),
        "val": scores(val["label"], clf.predict(vectorizer.transform(val["text"]))),
        "test": scores(test["label"], clf.predict(vectorizer.transform(test["text"]))),
    }
    (config.REPORTS_DIR / "baseline_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
