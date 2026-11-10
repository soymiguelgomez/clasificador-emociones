"""Análisis exploratorio: genera las figuras de la fase de comprensión de datos.

Requiere haber ejecutado antes `python -m src.data_prep`.
Uso:  python -m src.eda
Salida: reports/figures/eda_*.png y reports/eda_summary.json

El análisis de longitudes y vocabulario se hace solo con el conjunto de
entrenamiento, para no tomar decisiones de diseño mirando los datos de prueba.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src import config
from src.plot_style import INK, INK_2, SURFACE, THOUSANDS, fmt_int, fmt_pct, plt, save

MAX_LEN = 60  # longitud de secuencia de la red neuronal (src/model.py)


def plot_class_distribution(report):
    """Figura 1: filas por emoción antes y después de la limpieza."""
    raw = [report["class_counts_raw"][n] for n in config.LABEL_NAMES_EN]
    clean = [report["class_counts_clean"][n] for n in config.LABEL_NAMES_EN]
    total_clean = sum(clean)
    y = np.arange(config.NUM_CLASSES)
    h = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y - h / 2, raw, h, color="#86b6ef", edgecolor=SURFACE, linewidth=2,
            label="Original")
    ax.barh(y + h / 2, clean, h, color="#2a78d6", edgecolor=SURFACE, linewidth=2,
            label="Tras limpieza")
    for yi, c in zip(y, clean):
        ax.text(c + 1500, yi + h / 2, f"{fmt_int(c)}  ({fmt_pct(c / total_clean)})",
                va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y, [n.capitalize() for n in config.LABEL_NAMES_ES])
    ax.invert_yaxis()
    ax.set_xlabel("Número de textos")
    ax.set_xlim(0, max(raw) * 1.25)
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.grid(axis="y", visible=False)
    ax.set_title("Distribución de textos por emoción")
    ax.legend(loc="lower right")
    save(fig, "eda_1_distribucion_clases.png")


def plot_text_length(train):
    """Figura 2: longitud de los textos en palabras (conjunto de entrenamiento)."""
    words = train["text"].str.split().str.len()
    p99 = words.quantile(0.99)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(words, bins=np.arange(0, words.max() + 2, 2), color="#2a78d6",
            edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(words.median(), color=INK_2, linewidth=1.5, linestyle="--")
    ax.axvline(MAX_LEN, color="#e34948", linewidth=1.5)
    ymax = ax.get_ylim()[1]
    ax.text(words.median() + 1.5, ymax * 0.92, f"Mediana = {words.median():.0f}",
            color=INK_2, fontsize=9)
    ax.text(MAX_LEN + 1.5, ymax * 0.80,
            f"Longitud máxima del modelo = {MAX_LEN}\n"
            f"cubre el {fmt_pct((words <= MAX_LEN).mean(), 2)} de los textos\n"
            f"(percentil 99 = {p99:.0f})",
            color=INK_2, fontsize=9)
    ax.set_xlim(0, 120)
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.set_xlabel("Palabras por texto")
    ax.set_ylabel("Número de textos")
    ax.grid(axis="x", visible=False)
    ax.set_title("Longitud de los textos (entrenamiento)")
    save(fig, "eda_2_longitud_textos.png")
    return {"coverage_max_len": float((words <= MAX_LEN).mean()), "max_len": MAX_LEN}


def plot_conflict_pairs(report, top=8):
    """Figura 3: parejas de emociones que más se contradicen en el corpus original."""
    pairs = list(report["conflict_label_pairs"].items())[:top]
    es = dict(zip(config.LABEL_NAMES_EN, config.LABEL_NAMES_ES))
    names = [" / ".join(es[p].capitalize() for p in k.split("|")) for k, _ in pairs]
    values = [v for _, v in pairs]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(names, values, color="#2a78d6", edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(values):
        ax.text(v + 80, i, fmt_int(v), va="center", fontsize=9, color=INK_2)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.15)
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.set_xlabel("Textos idénticos con ambas etiquetas")
    ax.grid(axis="y", visible=False)
    ax.set_title("Parejas de emociones con etiquetas contradictorias")
    save(fig, "eda_3_conflictos_etiquetas.png")


def word_class_counts(train):
    """Cuenta en cuántos textos de cada emoción aparece cada palabra (sin stopwords)."""
    stop = set(ENGLISH_STOP_WORDS) | {"feel", "feeling", "im", "ive", "just", "like", "really"}
    total, per_class = Counter(), [Counter() for _ in range(config.NUM_CLASSES)]
    for text, label in zip(train["text"], train["label"]):
        tokens = {t for t in text.split() if t not in stop and len(t) > 2}
        total.update(tokens)
        per_class[label].update(tokens)
    return total, per_class


def distinctive_words(total, per_class, top=10, min_count=200):
    """Palabras más características de cada emoción: porcentaje de los textos que
    contienen la palabra y pertenecen a esa emoción (100 % = exclusiva de la clase).
    Los empates se ordenan por frecuencia."""
    result = {}
    for c in range(config.NUM_CLASSES):
        scored = [(w, k / total[w], k) for w, k in per_class[c].items() if k >= min_count]
        result[config.LABEL_NAMES_EN[c]] = sorted(scored, key=lambda x: (-x[1], -x[2]))[:top]
    return result


def keyword_exclusivity(train, total, per_class, min_count=50, purity=0.99):
    """Mide cuánto del corpus se explica por palabras casi exclusivas de una emoción,
    un indicio de que las etiquetas se generaron a partir de palabras clave."""
    exclusive = {}
    for c in range(config.NUM_CLASSES):
        for w, k in per_class[c].items():
            if total[w] >= min_count and k / total[w] >= purity:
                exclusive[w] = c
    has_kw = train["text"].map(lambda s: any(t in exclusive for t in s.split()))
    return {
        "exclusive_words": len(exclusive),
        "exclusive_words_rule": f"aparece en >= {min_count} textos y >= {purity:.0%} en una emoción",
        "train_texts_with_exclusive_word": float(has_kw.mean()),
    }


def plot_distinctive_words(words):
    """Figura 4: palabras más características de cada emoción (múltiplos pequeños)."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True)
    for c, ax in enumerate(axes.flat):
        items = words[config.LABEL_NAMES_EN[c]]
        shares = [s * 100 for _, s, _ in items]
        ax.barh([w for w, _, _ in items], shares, color=config.CLASS_COLORS[c],
                edgecolor=SURFACE, linewidth=1.5)
        for i, (_, s, k) in enumerate(items):
            # Etiqueta dentro de la barra; si la barra es corta, a su derecha.
            inside = s * 100 > 25
            ax.text(2 if inside else s * 100 + 2, i, f"n = {fmt_int(k)}", va="center",
                    fontsize=7.5,
                    color=("#0b0b0b" if c == 3 else "#ffffff") if inside else INK_2)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
        ax.set_title(config.LABEL_NAMES_ES[c].capitalize(), fontsize=11)
        if c >= 3:
            ax.set_xlabel("% de los textos con la palabra que tienen esta emoción", fontsize=8)
    fig.suptitle("Palabras más características de cada emoción (entrenamiento)",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "eda_4_palabras_por_emocion.png")


def main():
    report_path = config.REPORTS_DIR / "data_report.json"
    if not report_path.exists():
        raise FileNotFoundError("Ejecuta primero: python -m src.data_prep")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv", keep_default_na=False)

    plot_class_distribution(report)
    length_info = plot_text_length(train)
    plot_conflict_pairs(report)
    total, per_class = word_class_counts(train)
    words = distinctive_words(total, per_class)
    plot_distinctive_words(words)

    summary = {
        **length_info,
        "train_vocabulary_size": int(len({t for s in train["text"] for t in s.split()})),
        "keyword_exclusivity": keyword_exclusivity(train, total, per_class),
        "distinctive_words": {k: [[w, round(s, 4), n] for w, s, n in v]
                              for k, v in words.items()},
    }
    (config.REPORTS_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
