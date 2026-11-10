# Despliegue en producción (Render)

Esta guía publica la aplicación Flask en [Render](https://render.com) con el plan gratuito.
**Estado: desplegada.** Desde el 9 de noviembre de 2026 la aplicación está en
<https://clasificador-emociones-qmng.onrender.com>, creada con el Blueprint `render.yaml` desde
<https://github.com/soymiguelgomez/clasificador-emociones>. La construcción usó Python 3.13.18, y la
prueba de humo contra la URL pública superó las 7 comprobaciones. Los pasos siguientes
sirven para repetir el despliegue en otra cuenta.

## 1. Decisión de arquitectura: NumPy en producción

| | TensorFlow (desarrollo) | NumPy (producción) |
|---|---|---|
| Memoria del proceso medida en local¹ | 431 MB | 62 MB (pico 72 MB) |
| Dependencias instaladas | TensorFlow, Keras, h5py… | Flask, gunicorn, NumPy |
| Tiempo por frase | — | 1,4 ms |
| Exactitud en validación | 0,991438 | 0,991438 |

¹ Memoria física (*working set*) en Windows 11 tras cargar el modelo y atender peticiones.

El plan gratuito de Render ofrece **512 MB de RAM y 0,1 CPU**. Con TensorFlow el proceso
quedaría al límite, así que en producción la red se ejecuta con `src/numpy_model.py`,
que implementa las ecuaciones de la LSTM con **los mismos pesos entrenados**
(`models/emotion_bilstm_weights.npz`). La equivalencia se comprobó en los 37.139 textos de
validación: diferencia máxima de probabilidad 2,7·10⁻⁶, 100 % de predicciones idénticas.
`tests/test_numpy_model.py` lo vuelve a comprobar en cada ejecución de las pruebas.

El motor se elige con la variable `MODEL_BACKEND` (`numpy` o `keras`).

## 2. Archivos de despliegue

| Archivo | Función |
|---|---|
| `requirements.txt` | Dependencias de producción con versiones fijas (Flask 3.1.3, gunicorn 26.2.0, NumPy 2.5.4) |
| `.python-version` | Fija Python 3.13 (sin él Render usaría 3.14 por defecto) |
| `render.yaml` | Blueprint: comandos de construcción e inicio, ruta de salud y variables de entorno |
| `wsgi.py` | Punto de entrada; carga el modelo al arrancar |
| `scripts/smoke_test.py` | Prueba de funcionamiento tras el despliegue |

**Comando de inicio** (definido en `render.yaml`):

```
gunicorn wsgi:app --workers 1 --threads 4 --timeout 120 --preload --bind 0.0.0.0:$PORT
```

- `--workers 1`: un único proceso; cada proceso adicional cargaría otra copia del modelo.
- `--threads 4`: atiende varias peticiones a la vez dentro del mismo proceso.
- `--preload`: carga el modelo antes de aceptar tráfico.
- `$PORT`: el puerto lo asigna Render.

**Variables de entorno** (ninguna es secreta; el proyecto no usa contraseñas ni claves API):

| Variable | Valor | Uso |
|---|---|---|
| `MODEL_BACKEND` | `numpy` | Motor de inferencia sin TensorFlow |
| `PRELOAD_MODEL` | `1` | Carga el modelo al arrancar |
| `OPENBLAS_NUM_THREADS` | `1` | Evita reservar memoria para hilos de BLAS innecesarios |

**Almacenamiento del modelo:** los artefactos de inferencia viajan en el propio repositorio
(`models/tokenizer.json` 0,3 MB, `models/emotion_bilstm_weights.npz` 14,6 MB y
`models/emotion_bilstm_inference.keras` 15,8 MB). Todos están por debajo del límite de 100 MB
por archivo de GitHub, así que no hace falta Git LFS ni almacenamiento externo. El checkpoint
de entrenamiento (45 MB) y los datos quedan fuera mediante `.gitignore`.

## 3. Subir el código a GitHub

Git no está instalado en el equipo donde se desarrolló el proyecto. Instálalo desde
<https://git-scm.com/download/win> o con:

```bash
winget install --id Git.Git -e
```

Crea un repositorio vacío en GitHub (por ejemplo `clasificador-emociones`, sin README) y,
desde la carpeta del proyecto:

```bash
git init
git add .
git commit -m "Clasificador de emociones con BiLSTM y Flask"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/clasificador-emociones.git
git push -u origin main
```

Antes del `commit`, revisa con `git status` que **no** aparezcan `data/raw/`,
`data/processed/`, `.venv/` ni `models/emotion_bilstm.keras`.

## 4. Crear el servicio en Render

1. Inicia sesión en <https://dashboard.render.com> y conecta tu cuenta de GitHub.
2. Pulsa **New → Blueprint** y elige el repositorio. Render detecta `render.yaml`.
3. Revisa que el servicio `clasificador-emociones` aparezca con plan **Free** y pulsa **Apply**.
4. Espera a que termine la construcción. En los registros (*Logs*) deben aparecer
   `pip install -r requirements.txt`, una línea con Python 3.13 y `Booting worker`.

Alternativa sin Blueprint: **New → Web Service**, con *Runtime* Python 3, *Build Command*
`pip install -r requirements.txt`, *Start Command* el de la sección 2, *Health Check Path*
`/health` y las tres variables de entorno anteriores.

## 5. Verificación después del despliegue

1. Abre `https://TU-SERVICIO.onrender.com/health`. Debe responder
   `{"status": "ok", "model_loaded": true, "backend": "numpy", ...}`.
2. Ejecuta la prueba de humo desde tu equipo:

   ```bash
   python scripts/smoke_test.py https://TU-SERVICIO.onrender.com
   ```

   Debe terminar con `Todo correcto.` (7 comprobaciones: salud, tres predicciones,
   validación de texto vacío, página de resultados y una figura).
3. Prueba la interfaz: analiza una frase de ejemplo y abre **Resultados del modelo**.
4. En la pestaña **Metrics** de Render, comprueba que la memoria se mantiene muy por debajo
   de 512 MB.
5. Solo cuando todo lo anterior funcione, añade la URL al README y al informe.

## 6. Limitaciones del plan gratuito y problemas frecuentes

| Situación | Causa | Solución |
|---|---|---|
| La primera visita tarda alrededor de un minuto | El servicio se duerme tras 15 minutos sin tráfico | Es normal en el plan gratuito; abre la URL un minuto antes de la sustentación |
| El servicio se suspende a fin de mes | Se agotaron las 750 horas gratuitas mensuales | Esperar al mes siguiente o cambiar de plan |
| `No se encontró models/...` en los registros | Los modelos no se subieron a GitHub | Comprueba `.gitignore` y que `models/` esté en el repositorio |
| Error de compilación de NumPy o versión 3.14 | No se leyó `.python-version` | Define `PYTHON_VERSION` con una versión completa, por ejemplo `3.13.5` |
| `Out of memory` | Se usó `MODEL_BACKEND=keras` o más de un *worker* | Usa `numpy` y `--workers 1` |
| Respuestas lentas con muchos usuarios | 0,1 CPU compartida | Suficiente para una demostración; para más tráfico, plan de pago |

### Servir con TensorFlow (opcional)

Para ejecutar el modelo `.keras` original en producción hace falta añadir `tensorflow-cpu`
a `requirements.txt`, poner `MODEL_BACKEND=keras` y elegir un plan con más memoria que 512 MB.
Los planes Free y Starter tienen ambos 512 MB, según la documentación de Render.
