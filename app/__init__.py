"""Aplicación web Flask del clasificador de emociones."""
from flask import Flask, jsonify, render_template, request


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=16 * 1024,  # rechaza peticiones de más de 16 KB
        MAX_TEXT_CHARS=1000,
    )
    if test_config:
        app.config.update(test_config)

    from app.routes import bp, pct, prob
    app.register_blueprint(bp)
    app.add_template_filter(pct)
    app.add_template_filter(prob)
    _register_error_handlers(app)
    return app


def _register_error_handlers(app: Flask):
    def handle(code, title, message):
        def handler(error):
            if request.path.startswith("/api/"):
                return jsonify({"error": message}), code
            return render_template("error.html", code=code, title=title,
                                   message=message), code
        return handler

    app.register_error_handler(404, handle(404, "Página no encontrada",
                                           "La dirección solicitada no existe."))
    app.register_error_handler(405, handle(405, "Método no permitido",
                                           "Esta dirección no admite ese tipo de petición."))
    app.register_error_handler(413, handle(413, "Texto demasiado grande",
                                           "La petición supera el tamaño máximo permitido."))
    app.register_error_handler(500, handle(500, "Error interno",
                                           "Ocurrió un error inesperado al procesar la "
                                           "solicitud. Inténtalo de nuevo."))
