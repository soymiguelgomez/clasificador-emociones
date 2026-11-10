"""Evaluación final del modelo en el conjunto de prueba (nunca usado para entrenar
ni para elegir hiperparámetros) y generación de las gráficas de resultados.

Requiere: python -m src.train  y  python -m src.baseline
Uso:  python -m src.evaluate
Salida:
  reports/metrics.json            métricas reales (las lee también la aplicación web)
  reports/test_errors_sample.csv  errores más confiados, para el análisis cualitativo
  reports/figures/eval_*.png      gráficas de resultados
"""
import json

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src import config
from src.plot_style import (BLUE_RAMP, DECIMAL, INK, INK_2, PERCENT, SERIES_1, SERIES_2,
                            SURFACE, fmt_dec, fmt_int, fmt_pct, plt, save)
from src.predict import EmotionPredictor
from src.train import load_split

LABELS_ES = [n.capitalize() for n in config.LABEL_NAMES_ES]

# Frases escritas a mano para explorar casos difíciles. No forman parte de las
# métricas: solo ilustran el comportamiento del modelo fuera del corpus.
PROBES = [
    ("negación", "i am not happy at all today"),
    ("negación", "i don t feel scared anymore"),
    ("emociones mezcladas", "i feel happy but also a little scared about the future"),
    ("sin emoción explícita", "the meeting is scheduled for monday at ten"),
    ("texto en español", "me siento muy triste hoy"),
]


def expected_calibration_error(confidence, correct, n_bins=15):
    """ECE: diferencia media (ponderada) entre la confianza declarada y la exactitud real."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def compute_metrics(y_true, probs):
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = y_pred == y_true
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(config.NUM_CLASSES), zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted",
                                               zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))
    off = [(int(cm[i, j]), config.LABEL_NAMES_EN[i], config.LABEL_NAMES_EN[j])
           for i in range(config.NUM_CLASSES) for j in range(config.NUM_CLASSES) if i != j]
    return {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
        "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]),
                     "f1": float(weighted[2])},
        "per_class": {
            config.LABEL_NAMES_EN[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                       "f1": float(f1[i]), "support": int(support[i])}
            for i in range(config.NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "top_confusions": [{"true": t, "pred": pr, "count": n}
                           for n, t, pr in sorted(off, reverse=True)[:6]],
        "errors": int((~correct).sum()),
        "calibration": {
            "ece": expected_calibration_error(confidence, correct),
            "mean_confidence_correct": float(confidence[correct].mean()),
            "mean_confidence_wrong": float(confidence[~correct].mean()),
            "share_wrong_with_conf_over_90": float((confidence[~correct] > 0.9).mean()),
        },
    }


def plot_history(history, best_epoch, metric, ylabel, title, filename, percent):
    epochs = history["epoch"] + 1
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for col, label, color in ((metric, "Entrenamiento", SERIES_1),
                              (f"val_{metric}", "Validación", SERIES_2)):
        ax.plot(epochs, history[col], color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        last = history[col].iloc[-1]
        ax.annotate(label, (epochs.iloc[-1], last), xytext=(10, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
    ax.axvline(best_epoch, color=INK_2, linestyle="--", linewidth=1.2)
    best_val = history[f"val_{metric}"].iloc[best_epoch - 1]
    ax.annotate(f"Mejor época ({best_epoch})\nvalidación = "
                + (fmt_pct(best_val, 2) if percent else fmt_dec(best_val, 4)),
                # Pérdida: la etiqueta va encima de la curva; exactitud: debajo.
                (best_epoch, best_val), xytext=(12, -60 if percent else 28),
                textcoords="offset points",
                color=INK, fontsize=9, arrowprops={"arrowstyle": "-", "color": INK_2})
    ax.set_xticks(epochs)
    ax.set_xlim(0.7, epochs.iloc[-1] + 1.1)
    ax.set_xlabel("Época")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PERCENT if percent else DECIMAL)
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.legend(loc="center right")
    note = ("Nota. La pérdida de entrenamiento está ponderada por clase y se calcula con dropout "
            "activo; la de validación no.") if not percent else \
           ("Nota. La exactitud de entrenamiento se promedia durante la época con dropout "
            "activo, por eso puede quedar por debajo de la de validación al inicio.")
    fig.text(0.01, -0.03, note, fontsize=8, color=INK_2, ha="left")
    return save(fig, filename)


def plot_confusion(cm):
    cm = np.array(cm)
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("blues", BLUE_RAMP[1:])
    cmap.set_bad(SURFACE)
    masked = np.ma.masked_where(cm == 0, row_pct)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(masked, cmap=cmap, norm=LogNorm(vmin=1e-4, vmax=1))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                ax.text(j, i, "0", ha="center", va="center", fontsize=8, color=INK_2)
                continue
            color = "#ffffff" if row_pct[i, j] > 0.05 else INK
            pct = "<0,1 %" if row_pct[i, j] < 0.0005 else fmt_pct(row_pct[i, j], 1)
            ax.text(j, i, f"{pct}\n{fmt_int(cm[i, j])}",
                    ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(len(LABELS_ES)), LABELS_ES, rotation=30, ha="right")
    ax.set_yticks(range(len(LABELS_ES)), LABELS_ES)
    ax.set_xlabel("Emoción predicha")
    ax.set_ylabel("Emoción real")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    cbar.set_ticks(ticks, labels=["0,01 %", "0,1 %", "1 %", "10 %", "100 %"])
    cbar.minorticks_off()
    cbar.set_label("% de los textos de la fila (escala logarítmica)")
    cbar.outline.set_visible(False)
    ax.set_title("Matriz de confusión en el conjunto de prueba")
    fig.text(0.01, -0.04, "Nota. Cada celda muestra el % de la emoción real (fila) y el número "
             "de textos.", fontsize=8, color=INK_2)
    return save(fig, "eval_3_matriz_confusion.png")


def plot_f1_comparison(metrics, baseline):
    names = config.LABEL_NAMES_EN
    nn = [metrics["per_class"][n]["f1"] for n in names] + [metrics["macro"]["f1"]]
    base = [baseline["test"]["f1_per_class"][n] for n in names] + [baseline["test"]["macro_f1"]]
    labels = LABELS_ES + ["Promedio (macro)"]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    for yi, a, b in zip(y, base, nn):
        ax.plot([a, b], [yi, yi], color="#c3c2b7", linewidth=2, zorder=1)
    ax.scatter(base, y, s=70, color=SERIES_2, edgecolor=SURFACE, linewidth=2, zorder=2,
               label="Línea base TF-IDF + regresión logística")
    ax.scatter(nn, y, s=70, color=SERIES_1, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Red neuronal BiLSTM")
    for yi, a, b in zip(y, base, nn):
        ax.text(min(a, b) - 0.002, yi, fmt_dec(min(a, b)), ha="right", va="center",
                fontsize=8.5, color=INK_2)
        ax.text(max(a, b) + 0.002, yi, fmt_dec(max(a, b)), ha="left", va="center",
                fontsize=8.5, color=INK_2)
    ax.axhline(len(names) - 0.5, color="#e1e0d9", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    lo = min(base + nn)
    ax.set_xlim(lo - 0.02, 1.006)
    ax.xaxis.set_major_formatter(DECIMAL)
    ax.set_xlabel("F1-score en el conjunto de prueba")
    ax.grid(axis="y", visible=False)
    ax.set_title("F1 por emoción: red neuronal frente a línea base")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    return save(fig, "eval_4_f1_por_clase.png")


def plot_confidence(probs, y_true):
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    bins = np.linspace(0.15, 1.0, 18)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for mask, label, color in ((correct, f"Aciertos (n = {fmt_int(correct.sum())})", SERIES_1),
                               (~correct, f"Errores (n = {fmt_int((~correct).sum())})",
                                SERIES_2)):
        weights = np.full(mask.sum(), 1 / mask.sum())
        ax.hist(confidence[mask], bins=bins, weights=weights, histtype="step", linewidth=2,
                color=color, label=label)
    ax.set_xlabel("Confianza del modelo (probabilidad de la clase predicha)")
    ax.set_ylabel("% de las predicciones del grupo")
    ax.xaxis.set_major_formatter(PERCENT)
    ax.yaxis.set_major_formatter(PERCENT)
    ax.set_title("Confianza de las predicciones: aciertos frente a errores")
    ax.legend(loc="upper left")
    return save(fig, "eval_5_confianza.png")


def main():
    test = load_split("test")
    baseline = json.loads((config.REPORTS_DIR / "baseline_metrics.json").read_text("utf-8"))
    training = json.loads((config.REPORTS_DIR / "training_summary.json").read_text("utf-8"))
    history = pd.read_csv(config.REPORTS_DIR / "training_history.csv")

    predictor = EmotionPredictor()
    probs = predictor.predict_proba(test["text"].tolist())
    y_true = test["label"].to_numpy()
    metrics = compute_metrics(y_true, probs)
    print(classification_report(y_true, probs.argmax(1), target_names=LABELS_ES, digits=4))

    # Errores más "seguros": útiles para entender las limitaciones del modelo.
    y_pred, conf = probs.argmax(1), probs.max(1)
    errors = test.assign(pred=y_pred, confidence=conf)[y_pred != y_true]
    errors = errors.assign(
        true_es=errors["label"].map(lambda i: config.LABEL_NAMES_ES[i]),
        pred_es=errors["pred"].map(lambda i: config.LABEL_NAMES_ES[i]),
    ).sort_values("confidence", ascending=False)
    errors.head(100).to_csv(config.REPORTS_DIR / "test_errors_sample.csv", index=False)

    figures = [
        plot_history(history, training["best_epoch"], "loss", "Pérdida (entropía cruzada)",
                     "Evolución de la pérdida", "eval_1_perdida.png", percent=False),
        plot_history(history, training["best_epoch"], "accuracy", "Exactitud",
                     "Evolución de la exactitud", "eval_2_exactitud.png", percent=True),
        plot_confusion(metrics["confusion_matrix"]),
        plot_f1_comparison(metrics, baseline),
        plot_confidence(probs, y_true),
    ]

    probes = []
    for case, text in PROBES:
        r = predictor.predict(text)
        probes.append({"case": case, "text": text, "pred_es": r["label_es"],
                       "confidence": round(r["confidence"], 4),
                       "known_words": f"{r['n_known_words']}/{r['n_words']}"})

    report = {
        "model": training["model_path"],
        "test": metrics,
        "baseline_test": baseline["test"],
        "training": {k: training[k] for k in ("best_epoch", "epochs_run", "best_val_loss",
                                              "best_val_accuracy", "training_minutes",
                                              "params")},
        "figures": [str(f.relative_to(config.ROOT_DIR)).replace("\\", "/") for f in figures],
        "qualitative_probes": probes,
    }
    (config.REPORTS_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("qualitative_probes",)}, indent=2,
                     ensure_ascii=False))
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro", "weighted", "errors",
                                              "top_confusions", "calibration")}, indent=2))


if __name__ == "__main__":
    main()
