import pytest

from app import create_app
from src import config

MODEL_READY = (config.MODELS_DIR / "emotion_bilstm_inference.keras").exists()
needs_model = pytest.mark.skipif(not MODEL_READY, reason="Falta el modelo entrenado")


@pytest.fixture(scope="module")
def client():
    app = create_app({"TESTING": True})
    return app.test_client()


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Analizar una frase" in r.get_data(as_text=True)


def test_empty_text_is_rejected(client):
    r = client.post("/", data={"text": "   "})
    assert r.status_code == 400
    assert "Escribe una frase" in r.get_data(as_text=True)


def test_too_long_text_is_rejected(client):
    r = client.post("/", data={"text": "a" * 1001})
    assert r.status_code == 400
    assert "máximo de 1000" in r.get_data(as_text=True)


def test_text_without_letters_is_rejected(client):
    r = client.post("/", data={"text": "123 !!! ???"})
    assert r.status_code == 400


def test_api_requires_json(client):
    assert client.post("/api/predict", data="x").status_code == 400


def test_unknown_page_and_figure_return_404(client):
    assert client.get("/no-existe").status_code == 404
    assert client.get("/figuras/../config.py").status_code == 404
    assert client.get("/api/no-existe").get_json()["error"]


@needs_model
def test_form_prediction(client):
    r = client.post("/", data={"text": "i feel so nervous about my exam tomorrow"})
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "Probabilidad por emoción" in html
    assert "Miedo" in html


@needs_model
def test_api_prediction_probabilities_sum_to_one(client):
    r = client.post("/api/predict", json={"text": "I'm so happy and grateful today!"})
    data = r.get_json()
    assert r.status_code == 200
    assert data["label_en"] in config.LABEL_NAMES_EN
    assert len(data["probabilities"]) == config.NUM_CLASSES
    assert abs(sum(p["probability"] for p in data["probabilities"]) - 1) < 1e-4


@needs_model
def test_non_english_text_is_flagged(client):
    data = client.post("/api/predict", json={"text": "me siento muy triste hoy"}).get_json()
    assert data["reliable"] is False
    assert data["warnings"]


@needs_model
def test_health_and_results(client):
    assert client.get("/health").get_json()["model_loaded"] is True
    r = client.get("/resultados")
    assert r.status_code == 200
    assert "Figura 1." in r.get_data(as_text=True)
    assert client.get("/figuras/eval_3_matriz_confusion.png").status_code == 200
