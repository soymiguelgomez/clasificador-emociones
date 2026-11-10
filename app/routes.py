"""Rutas de la aplicación: formulario de predicción, resultados, API y salud."""
import json
import logging
import threading
from functools import lru_cache

from flask import (Blueprint, abort, current_app, jsonify, render_template, request,
                   send_from_directory)

from src import config
from src.predict import EmotionPredictor
from src.text_cleaning import clean_text

bp = Blueprint("main", __name__)
log = logging.getLogger(__name__)

# Si menos del 60 % de las palabras están en el vocabulario, probablemente el texto
# no está en inglés o no se parece al corpus: el resultado se marca como poco fiable.
MIN_KNOWN_RATIO = 0.6
LOW_CONFIDENCE = 0.6

EMOTIONS = {
    "sadness": {"color": config.CLASS_COLORS[0], "desc": "Pena, desánimo, soledad o decepción."},
    "joy": {"color": config.CLASS_COLORS[1],
            "desc": "Bienestar, satisfacción, entusiasmo o calma positiva."},
    "love": {"color": config.CLASS_COLORS[2],
             "desc": "Afecto, ternura, cariño o deseo hacia alguien."},
    "anger": {"color": config.CLASS_COLORS[3],
              "desc": "Enfado, frustración, irritación o resentimiento."},
    "fear": {"color": config.CLASS_COLORS[4], "desc": "Nervios, ansiedad, inseguridad o temor."},
    "surprise": {"color": config.CLASS_COLORS[5],
                 "desc": "Asombro o impresión ante algo inesperado."},
}

EXAMPLES = [
    "i feel so nervous about my exam tomorrow",
    "i am feeling really grateful for my friends today",
    "i feel betrayed and furious at how they treated me",
    "i was shocked when i heard the news",
    "i miss her so much and feel empty without her",
    "i feel so loved and cared for by my partner",
]

# Figuras reales generadas por src/eda.py y src/evaluate.py.
FIGURES = [
    ("eval_1_perdida.png", "Evolución de la pérdida",
     "Pérdida de entrenamiento y validación por época. La línea discontinua marca la época "
     "cuyo modelo se guardó."),
    ("eval_2_exactitud.png", "Evolución de la exactitud",
     "Exactitud de entrenamiento y validación por época."),
    ("eval_3_matriz_confusion.png", "Matriz de confusión",
     "Cómo se reparten las predicciones para cada emoción real del conjunto de prueba."),
    ("eval_4_f1_por_clase.png", "F1 por emoción frente a la línea base",
     "F1-score de la red neuronal y de un modelo clásico TF-IDF + regresión logística."),
    ("eval_5_confianza.png", "Confianza en aciertos y errores",
     "Distribución de la probabilidad asignada a la clase predicha."),
    ("eda_1_distribucion_clases.png", "Distribución de clases",
     "Textos por emoción antes y después de la limpieza de datos."),
    ("eda_2_longitud_textos.png", "Longitud de los textos",
     "Palabras por texto en entrenamiento y longitud máxima usada por el modelo."),
    ("eda_3_conflictos_etiquetas.png", "Etiquetas contradictorias",
     "Parejas de emociones asignadas al mismo texto en el corpus original (textos excluidos)."),
    ("eda_4_palabras_por_emocion.png", "Palabras características",
     "Palabras más asociadas a cada emoción en el conjunto de entrenamiento."),
]
ALLOWED_FIGURES = {name for name, _, _ in FIGURES}

_predictor = None
_predictor_lock = threading.Lock()


def get_predictor() -> EmotionPredictor:
    """Carga el modelo una sola vez por proceso (no se reentrena en cada petición)."""
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = EmotionPredictor()
    return _predictor


@lru_cache(maxsize=1)
def load_reports() -> dict:
    """Lee las métricas y estadísticas reales guardadas por los scripts de entrenamiento."""
    out = {}
    for key, name in (("metrics", "metrics.json"), ("data", "data_report.json")):
        path = config.REPORTS_DIR / name
        out[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return out


def pct(value, decimals=1):
    """Filtro de plantilla: 0.9927 -> '99,3 %'."""
    return f"{value * 100:.{decimals}f} %".replace(".", ",")


def prob(value):
    """Filtro para probabilidades: evita mostrar 100,0 % o 0,0 %, que nunca son exactos."""
    if value >= 0.9995:
        return ">99,9 %"
    if value < 0.0005:
        return "<0,1 %"
    return pct(value)


class InputError(ValueError):
    pass


def validate_text(text) -> str:
    if not isinstance(text, str) or not text.strip():
        raise InputError("Escribe una frase para analizar.")
    max_chars = current_app.config["MAX_TEXT_CHARS"]
    if len(text) > max_chars:
        raise InputError(f"El texto supera el máximo de {max_chars} caracteres "
                         f"({len(text)}).")
    if not clean_text(text):
        raise InputError("El texto no contiene palabras: el modelo solo procesa letras.")
    return text.strip()


def analyze(text: str) -> dict:
    """Ejecuta la predicción real y añade la explicación y las advertencias."""
    result = get_predictor().predict(text)
    known_ratio = result["n_known_words"] / max(1, min(result["n_words"], 60))
    top, second = result["probabilities"][0], result["probabilities"][1]

    warnings = []
    if known_ratio < MIN_KNOWN_RATIO:
        warnings.append(
            f"Solo {result['n_known_words']} de {min(result['n_words'], 60)} palabras están en "
            "el vocabulario del modelo. El modelo se entrenó con frases en inglés: si el texto "
            "está en otro idioma, el resultado no es fiable.")
    if top["probability"] < LOW_CONFIDENCE:
        warnings.append(
            f"El modelo duda: {top['label_es']} ({prob(top['probability'])}) y "
            f"{second['label_es']} ({prob(second['probability'])}) están cerca. Puede que la "
            "frase no exprese una emoción clara o mezcle varias.")
    if result["truncated"]:
        warnings.append("El texto tiene más de 60 palabras: solo se analizaron las 60 primeras.")

    conf = top["probability"]
    reliable = known_ratio >= MIN_KNOWN_RATIO
    detail = (f"asigna un {prob(conf)} de probabilidad a «{top['label_es']}» "
              f"({EMOTIONS[top['label_en']]['desc'].lower().rstrip('.')}). La segunda opción "
              f"es «{second['label_es']}» con un {prob(second['probability'])}.")
    if not reliable:
        # Con pocas palabras conocidas, la probabilidad alta no significa nada.
        explanation = (f"El modelo {detail} Como apenas reconoce las palabras del texto, "
                       "esta probabilidad no indica una predicción fiable.")
    else:
        level = "muy seguro" if conf >= 0.9 else "bastante seguro" \
            if conf >= LOW_CONFIDENCE else "poco seguro"
        explanation = f"El modelo está {level}: {detail}"
    for p in result["probabilities"]:
        p["color"] = EMOTIONS[p["label_en"]]["color"]
    result.update(known_ratio=known_ratio, warnings=warnings, explanation=explanation,
                  reliable=reliable, color=EMOTIONS[top["label_en"]]["color"])
    return result


@bp.route("/", methods=["GET", "POST"])
def index():
    text, result, error, status = "", None, None, 200
    if request.method == "POST":
        text = request.form.get("text", "")
        try:
            result = analyze(validate_text(text))
        except InputError as exc:
            error, status = str(exc), 400
    return render_template("index.html", text=text, result=result, error=error,
                           examples=EXAMPLES, emotions=EMOTIONS,
                           max_chars=current_app.config["MAX_TEXT_CHARS"],
                           reports=load_reports()), status


@bp.route("/resultados")
def results():
    reports = load_reports()
    if reports["metrics"] is None:
        abort(404)
    available = [f for f in FIGURES if (config.FIGURES_DIR / f[0]).exists()]
    return render_template("results.html", reports=reports, figures=available,
                           labels_es=dict(zip(config.LABEL_NAMES_EN, config.LABEL_NAMES_ES)),
                           emotions=EMOTIONS)


@bp.route("/figuras/<name>")
def figure(name):
    if name not in ALLOWED_FIGURES:
        abort(404)
    return send_from_directory(config.FIGURES_DIR, name, max_age=3600)


@bp.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Envía un JSON con el campo 'text'."}), 400
    try:
        result = analyze(validate_text(payload.get("text")))
    except InputError as exc:
        return jsonify({"error": str(exc)}), 400
    keys = ("label_en", "label_es", "confidence", "probabilities", "reliable", "warnings",
            "n_words", "n_known_words")
    return jsonify({k: result[k] for k in keys})


@bp.get("/health")
def health():
    """Comprueba que el modelo carga y predice; útil tras el despliegue."""
    try:
        sample = get_predictor().predict("i feel happy")
        return jsonify({"status": "ok", "model_loaded": True,
                        "backend": get_predictor().backend,
                        "sample_prediction": sample["label_en"]})
    except Exception as exc:  # noqa: BLE001 - se informa del fallo al monitor
        log.exception("Fallo en /health")
        return jsonify({"status": "error", "model_loaded": False, "detail": str(exc)}), 503
