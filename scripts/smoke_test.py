"""Prueba de humo de la aplicación desplegada (o local). Solo usa la biblioteca estándar.

Uso:  python scripts/smoke_test.py https://tu-app.onrender.com
      python scripts/smoke_test.py http://127.0.0.1:5000
"""
import json
import sys
import time
import urllib.error
import urllib.request

CASES = [
    ("i feel so nervous about my exam tomorrow", "fear"),
    ("i feel so loved and cared for by my partner", "love"),
    ("i miss her so much and feel empty without her", "sadness"),
]


def request(base, path, payload=None, timeout=90):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), time.time() - start
    except urllib.error.HTTPError as err:
        return err.code, err.read(), time.time() - start


def main(base):
    base = base.rstrip("/")
    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        failures += not ok
        print(f"[{'OK' if ok else 'FALLO'}] {name} {detail}")

    # La primera petición puede tardar si el servicio gratuito estaba dormido.
    status, body, secs = request(base, "/health")
    health = json.loads(body) if status == 200 else {}
    check("/health", status == 200 and health.get("model_loaded"),
          f"({status}, {secs:.1f} s, motor={health.get('backend')})")

    for text, expected in CASES:
        status, body, secs = request(base, "/api/predict", {"text": text})
        data = json.loads(body) if status == 200 else {}
        check(f"predicción «{text[:40]}»", data.get("label_en") == expected,
              f"-> {data.get('label_en')} ({data.get('confidence', 0):.3f}, {secs * 1000:.0f} ms)")

    status, _, _ = request(base, "/api/predict", {"text": ""})
    check("texto vacío devuelve 400", status == 400, f"({status})")
    status, _, _ = request(base, "/resultados")
    check("/resultados", status == 200, f"({status})")
    status, _, _ = request(base, "/figuras/eval_3_matriz_confusion.png")
    check("figura de la matriz de confusión", status == 200, f"({status})")

    print("\nTodo correcto." if failures == 0 else f"\n{failures} comprobación(es) fallaron.")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
