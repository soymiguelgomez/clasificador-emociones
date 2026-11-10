"""Punto de entrada de la aplicación.

Desarrollo:  python wsgi.py           (http://127.0.0.1:5000)
Producción:  gunicorn wsgi:app
"""
import os

from app import create_app
from app.routes import get_predictor

app = create_app()

# Carga el modelo al arrancar para que la primera petición no espere.
if os.environ.get("PRELOAD_MODEL", "1") == "1":
    get_predictor()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
