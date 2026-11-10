"""Preparación de datos: limpieza, deduplicación, eliminación de etiquetas en conflicto
y división estratificada en entrenamiento, validación y prueba sin fuga de información.

Uso:  python -m src.data_prep
Salida: data/processed/{train,val,test}.csv y reports/data_report.json
"""
import json
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.text_cleaning import clean_text


def load_raw(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Carga el CSV original y conserva solo las columnas útiles."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {path}. Cópialo en data/raw/text.csv "
            "o define la variable de entorno DATA_PATH."
        )
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    # La primera columna ("Unnamed: 0") es solo el índice 0..n-1 guardado por pandas.
    df = df[["text", "label"]].copy()
    df["label"] = df["label"].astype(int)
    bad = ~df["label"].between(0, config.NUM_CLASSES - 1)
    if bad.any():
        raise ValueError(f"{bad.sum()} filas tienen etiquetas fuera de 0..{config.NUM_CLASSES - 1}")
    return df


def class_counts(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts()
    return {config.LABEL_NAMES_EN[i]: int(counts.get(i, 0)) for i in range(config.NUM_CLASSES)}


def clean_dataframe(df: pd.DataFrame):
    """Aplica la limpieza y elimina duplicados y textos con etiquetas contradictorias.

    Devuelve el DataFrame limpio y un diccionario con las estadísticas de cada paso.
    """
    stats = {"raw_rows": len(df), "raw_nulls": int(df.isna().sum().sum()),
             "class_counts_raw": class_counts(df)}

    df = df.dropna(subset=["text"]).copy()
    original = df["text"].astype(str)
    df["text"] = original.map(clean_text)
    stats["rows_modified_by_cleaning"] = int((df["text"] != original).sum())

    empty = df["text"].str.len() == 0
    stats["rows_empty_after_cleaning"] = int(empty.sum())
    df = df[~empty]

    # 1) Duplicados exactos (mismo texto y misma etiqueta): no aportan información
    #    y, si cayeran en conjuntos distintos, inflarían la métrica de prueba.
    dup = df.duplicated(subset=["text", "label"], keep="first")
    stats["exact_duplicates_removed"] = int(dup.sum())
    df = df[~dup]

    # 2) Textos idénticos con etiquetas distintas: no existe una respuesta correcta
    #    única, así que se excluyen para entrenar y evaluar con etiquetas fiables.
    n_labels = df.groupby("text")["label"].nunique()
    conflict_texts = n_labels[n_labels > 1].index
    conflicting = (df[df["text"].isin(conflict_texts)]
                   .groupby("text")["label"].agg(lambda s: tuple(sorted(s))))
    pair_counter = Counter()
    for labels in conflicting:
        for a_idx, a in enumerate(labels):
            for b in labels[a_idx + 1:]:
                pair_counter[(config.LABEL_NAMES_EN[a], config.LABEL_NAMES_EN[b])] += 1
    in_conflict = df["text"].isin(conflicting.index)
    stats["conflicting_texts"] = int(len(conflicting))
    stats["conflicting_rows_removed"] = int(in_conflict.sum())
    stats["conflict_rows_removed_by_class"] = class_counts(df[in_conflict])
    stats["conflict_label_pairs"] = {f"{a}|{b}": n for (a, b), n in pair_counter.most_common()}
    df = df[~in_conflict].reset_index(drop=True)

    stats["clean_rows"] = len(df)
    stats["class_counts_clean"] = class_counts(df)
    return df, stats


def split_dataset(df: pd.DataFrame):
    """División estratificada 80/10/10. Como cada texto ya es único, ninguna frase
    puede aparecer en dos conjuntos a la vez."""
    train_val, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df["label"], random_state=config.RANDOM_SEED
    )
    val_fraction = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_fraction, stratify=train_val["label"],
        random_state=config.RANDOM_SEED,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def check_no_leakage(train, val, test) -> dict:
    sets = {"train": set(train["text"]), "val": set(val["text"]), "test": set(test["text"])}
    overlaps = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Fuga de información detectada: {overlaps}")
    return overlaps


def write_sample(test: pd.DataFrame, per_class: int = 10):
    """Guarda una muestra pequeña del conjunto de prueba (10 frases por emoción) que sí
    se incluye en el repositorio, para probar el proyecto sin descargar el dataset."""
    sample = (test.groupby("label").sample(n=per_class, random_state=config.RANDOM_SEED)
              .assign(emotion=lambda d: d["label"].map(lambda i: config.LABEL_NAMES_EN[i]))
              .reset_index(drop=True))
    path = config.ROOT_DIR / "data" / "sample" / "sample_texts.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)
    return path


def main():
    df = load_raw()
    clean, stats = clean_dataframe(df)
    train, val, test = split_dataset(clean)
    stats["text_overlap_between_splits"] = check_no_leakage(train, val, test)
    stats["split_sizes"] = {"train": len(train), "val": len(val), "test": len(test)}
    stats["class_counts_by_split"] = {
        "train": class_counts(train), "val": class_counts(val), "test": class_counts(test)
    }
    words = clean["text"].str.split().str.len()
    stats["words_per_text"] = {
        "mean": round(float(words.mean()), 2), "median": float(words.median()),
        "p95": float(words.quantile(0.95)), "p99": float(words.quantile(0.99)),
        "max": int(words.max()),
    }

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(config.PROCESSED_DIR / f"{name}.csv", index=False)
    write_sample(test)
    with open(config.REPORTS_DIR / "data_report.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
