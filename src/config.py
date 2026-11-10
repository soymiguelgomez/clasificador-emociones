"""Configuración central del proyecto: rutas, etiquetas e hiperparámetros compartidos."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# La ruta del CSV original puede sobrescribirse con la variable de entorno DATA_PATH.
RAW_DATA_PATH = Path(os.environ.get("DATA_PATH", ROOT_DIR / "data" / "raw" / "text.csv"))
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"

RANDOM_SEED = 42
TEST_SIZE = 0.10
VAL_SIZE = 0.10

# Correspondencia documentada en la ficha del dataset "Emotions" de Kaggle
# (Elgiriyewithana, 2024): https://www.kaggle.com/datasets/nelgiriyewithana/emotions
LABEL_NAMES_EN = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LABEL_NAMES_ES = ["tristeza", "alegría", "amor", "ira", "miedo", "sorpresa"]
NUM_CLASSES = len(LABEL_NAMES_EN)

# Un color fijo por emoción en todas las gráficas (paleta categórica validada).
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
