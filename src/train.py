"""Entrenamiento de la red neuronal BiLSTM.

Requiere haber ejecutado `python -m src.data_prep`.
Uso:  python -m src.train [--epochs 12] [--batch-size 256]
Salida:
  models/emotion_bilstm.keras   mejor modelo según la pérdida de validación
  models/tokenizer.json         vocabulario ajustado con el conjunto de entrenamiento
  reports/training_history.csv  pérdida y exactitud por época
  reports/training_summary.json configuración, pesos por clase y tiempos
"""
import argparse
import json
import time

import keras
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src import config
from src.export_model import export_inference_model
from src.model import MAX_LEN, MAX_TOKENS, build_model
from src.tokenizer import WordTokenizer

MODEL_PATH = config.MODELS_DIR / "emotion_bilstm.keras"
TOKENIZER_PATH = config.MODELS_DIR / "tokenizer.json"


def load_split(name: str) -> pd.DataFrame:
    path = config.PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero: python -m src.data_prep")
    return pd.read_csv(path, keep_default_na=False)


class EpochTimer(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs["epoch_seconds"] = round(time.time() - self._start, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    keras.utils.set_random_seed(config.RANDOM_SEED)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train, val = load_split("train"), load_split("val")

    # El vocabulario se ajusta solo con entrenamiento para no filtrar información.
    tokenizer = WordTokenizer.fit(train["text"], max_tokens=MAX_TOKENS, max_len=MAX_LEN)
    tokenizer.save(TOKENIZER_PATH)
    x_train, x_val = tokenizer.encode(train["text"]), tokenizer.encode(val["text"])
    y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()
    oov_rate = float((x_val == 1).sum() / (x_val > 0).sum())

    # Pesos "balanced": n / (k * n_clase). Una clase rara como sorpresa pesa más en
    # la pérdida, para que el modelo no la ignore por ser minoritaria.
    classes = np.arange(config.NUM_CLASSES)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_model(vocab_size=len(tokenizer.vocab))
    model.summary()

    callbacks = [
        # Guarda solo el mejor modelo visto en validación.
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
        # Detiene el entrenamiento si la validación deja de mejorar (evita sobreajuste)
        # y restaura los pesos de la mejor época.
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                      restore_best_weights=True),
        # Reduce la tasa de aprendizaje cuando la pérdida se estanca.
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1,
                                          min_lr=1e-5),
        EpochTimer(),
        keras.callbacks.CSVLogger(config.REPORTS_DIR / "training_history.csv"),
    ]

    start = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - start

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    summary = {
        "model_path": str(MODEL_PATH.relative_to(config.ROOT_DIR)),
        "params": int(model.count_params()),
        "vocab_size": len(tokenizer.vocab),
        "max_len": MAX_LEN,
        "val_oov_rate": round(oov_rate, 4),
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "epochs_run": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": float(min(history.history["val_loss"])),
        "best_val_accuracy": float(history.history["val_accuracy"][best_epoch - 1]),
        "class_weight": {config.LABEL_NAMES_EN[c]: round(w, 4) for c, w in class_weight.items()},
        "training_minutes": round(elapsed / 60, 1),
        "keras_version": keras.__version__,
    }
    (config.REPORTS_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    export_inference_model(MODEL_PATH)


if __name__ == "__main__":
    main()
