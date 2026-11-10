# Clasificador de emociones en texto con una red LSTM bidireccional

Proyecto final de Inteligencia Artificial. Una red neuronal identifica cuál de seis emociones
—**tristeza, alegría, amor, ira, miedo o sorpresa**— predomina en una frase en inglés.
El modelo está integrado en una aplicación web hecha con Flask y preparado para desplegarse en Render.

> Herramienta académica: no sirve para diagnosticar el estado emocional de nadie.

## Problema que resuelve

Las personas describen cómo se sienten en redes sociales, reseñas y mensajes de soporte, y leer
miles de esos textos a mano no es viable. El sistema recibe una frase como
`i feel so nervous about my exam tomorrow` y devuelve:

- la emoción más probable;
- la probabilidad de cada una de las seis emociones;
- una explicación del resultado;
- avisos cuando el texto se aleja de lo que el modelo conoce (otro idioma, baja confianza o texto truncado).

Es una tarea de **Procesamiento de Lenguaje Natural (PLN)**: clasificación de texto multiclase.

## Objetivos

- **General:** desarrollar y evaluar una red neuronal LSTM bidireccional que clasifique frases
  en inglés en seis emociones, e integrarla en una aplicación web lista para producción.
- **Específicos:**
  1. Analizar la calidad y la distribución del conjunto de datos.
  2. Prepararlo sin fuga de información.
  3. Entrenar y regularizar la red.
  4. Evaluarla en un conjunto de prueba independiente frente a una línea base.
  5. Construir la aplicación web.
  6. Preparar el despliegue.

## Tecnologías

| Uso | Herramientas |
|---|---|
| Lenguaje | Python 3.13 |
| Datos | pandas 3.0, NumPy 2.5 |
| Red neuronal | TensorFlow 2.21, Keras 3.15 |
| Línea base y métricas | scikit-learn 1.9 |
| Gráficas | Matplotlib 3.11 |
| Aplicación web | Flask 3.1, gunicorn 26.2 |
| Pruebas | pytest 9.2 |
| Despliegue | Render (plan gratuito) |
| Entorno de desarrollo | Visual Studio Code en Windows 11, sin GPU |

## Requisitos previos

- Python **3.13** (`python --version`).
- Unos 2 GB libres para las dependencias de desarrollo (TensorFlow).
- El archivo de datos `text.csv`, que no se incluye en el repositorio (ver [Datos](#datos)).
- Git, solo si vas a publicar el código en GitHub (`winget install --id Git.Git -e`).

## Estructura del proyecto

```
├── app/                         Aplicación Flask
│   ├── __init__.py              Crea la aplicación y los manejadores de error
│   ├── routes.py                Validación, predicción, explicación, API y /health
│   ├── templates/               base, index, results y error (HTML)
│   └── static/css/styles.css    Estilos adaptables (móvil y escritorio)
├── src/                         Código del modelo
│   ├── config.py                Rutas, etiquetas y semilla
│   ├── text_cleaning.py         Limpieza de texto (la misma en entrenamiento y en la web)
│   ├── data_prep.py             Limpieza, deduplicación y división 80/10/10
│   ├── eda.py                   Análisis exploratorio y figuras
│   ├── tokenizer.py             Vocabulario y codificación a índices
│   ├── model.py                 Arquitectura de la red BiLSTM
│   ├── train.py                 Entrenamiento con parada temprana
│   ├── baseline.py              Línea base TF-IDF + regresión logística
│   ├── evaluate.py              Métricas en prueba y gráficas
│   ├── export_model.py          Exporta los artefactos de inferencia
│   ├── numpy_model.py           La misma red implementada en NumPy (producción)
│   ├── predict.py               Predicción con el modelo guardado
│   └── plot_style.py            Estilo común de las gráficas
├── models/                      tokenizer.json, emotion_bilstm_inference.keras, emotion_bilstm_weights.npz
├── reports/                     Métricas en JSON, historial, registro de entrenamiento y figuras
├── data/sample/                 60 frases de ejemplo (10 por emoción)
├── tests/                       20 pruebas automatizadas
├── scripts/
│   ├── smoke_test.py            Prueba de humo contra una URL
│   └── make_zip.py              Genera el ZIP de entrega
├── docs/
│   ├── DESPLIEGUE.md            Guía de despliegue en Render
│   └── informe/                 Informe APA 7 (.docx y .pdf)
├── wsgi.py                      Punto de entrada (desarrollo y gunicorn)
├── requirements.txt             Dependencias de producción
├── requirements-dev.txt         Dependencias de desarrollo
├── render.yaml                  Configuración de Render
└── .python-version              Versión de Python para Render
```

## Instalación

### 1. Crear el entorno virtual

```bash
python -m venv .venv
```

En Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

### 2. Instalar las dependencias

Para entrenar, evaluar y ejecutar las pruebas:

```bash
pip install -r requirements-dev.txt
```

Para ejecutar solo la aplicación web (sin TensorFlow):

```bash
pip install -r requirements.txt
```

## Datos

Se usa el conjunto **Emotions** de Kaggle (Elgiriyewithana, 2024): 416.809 mensajes de Twitter en inglés.

- **Enlace:** <https://www.kaggle.com/datasets/nelgiriyewithana/emotions>
- **Etiquetas:** 0 = tristeza, 1 = alegría, 2 = amor, 3 = ira, 4 = miedo, 5 = sorpresa.

1. Descarga `text.csv` desde Kaggle. Necesitas una cuenta gratuita.
2. Cópialo en `data/raw/text.csv`. Si prefieres otra ubicación, define la variable de entorno `DATA_PATH`.

El CSV pesa unos 43 MB y está excluido del repositorio con `.gitignore`. Para probar el proyecto
sin descargarlo, `data/sample/sample_texts.csv` trae 60 frases reales del conjunto de prueba.
La aplicación web no necesita el CSV: solo usa los modelos de `models/`.

## Entrenar el modelo

Ejecuta los pasos en orden desde la carpeta raíz:

```bash
python -m src.data_prep
```
```bash
python -m src.eda
```
```bash
python -m src.train
```
```bash
python -m src.baseline
```
```bash
python -m src.evaluate
```

| Paso | Qué hace | Tiempo aproximado (CPU) |
|---|---|---|
| `data_prep` | Limpieza, deduplicación, eliminación de etiquetas contradictorias y división sin fuga | 1–2 min |
| `eda` | Figuras del análisis exploratorio | 1 min |
| `train` | Entrena la BiLSTM, guarda el mejor modelo y exporta los artefactos de inferencia | 20 min |
| `baseline` | Línea base TF-IDF + regresión logística | 6 min |
| `evaluate` | Métricas en el conjunto de prueba y gráficas de resultados | 1 min |

Para predecir desde la consola:

```bash
python -m src.predict "i feel so nervous about my exam tomorrow"
```

## Ejecutar la aplicación en local

```bash
python wsgi.py
```

Abre <http://127.0.0.1:5000>. La aplicación tiene estas rutas:

| Ruta | Función |
|---|---|
| `/` | Formulario, resultado y explicación |
| `/resultados` | Métricas y gráficas reales |
| `POST /api/predict` | API JSON con el formato `{"text": "..."}` |
| `/health` | Comprueba que el modelo está cargado |

El motor de inferencia se elige con `MODEL_BACKEND`:

- `keras`: el modelo `.keras` con TensorFlow. Es el valor por defecto si TensorFlow está instalado.
- `numpy`: los mismos pesos ejecutados con NumPy, sin TensorFlow.

## Pruebas

```bash
python -m pytest
```

Son 20 pruebas sobre la limpieza, el tokenizador, la ausencia de fuga de datos, la aplicación web
(validación, predicción, API, errores y salud) y la equivalencia entre los motores NumPy y Keras.

## Resultados reales

Medidos en el conjunto de prueba: 37.139 frases que no se usaron ni para entrenar ni para elegir hiperparámetros.

| Emoción | Precisión | Exhaustividad | F1 | F1 línea base |
|---|---|---|---|---|
| Tristeza | 0,996 | 0,994 | 0,995 | 0,970 |
| Alegría | 0,997 | 0,996 | 0,996 | 0,972 |
| Amor | 0,998 | 0,999 | 0,998 | 0,942 |
| Ira | 0,981 | 0,989 | 0,985 | 0,961 |
| Miedo | 0,986 | 0,982 | 0,984 | 0,955 |
| Sorpresa | 0,971 | 0,990 | 0,980 | 0,927 |
| **Promedio macro** | **0,988** | **0,991** | **0,990** | **0,954** |

**Resumen:**

- **Exactitud en prueba:** 99,27 % (272 errores en 37.139 frases). La línea base TF-IDF obtiene 96,46 %.
- **Entrenamiento:** la mejor época fue la 3 de 6 (validación: 99,14 %), con 3.947.462 parámetros. Tardó 19,8 minutos en CPU.
- **Datos:** de 416.809 filas se eliminaron 838 duplicados exactos y 44.582 filas con etiquetas contradictorias. Quedaron 371.389 textos únicos.

Todas las cifras están en `reports/metrics.json` y las gráficas en `reports/figures/`.

## Enlace de producción

**Aplicación en producción:** <https://clasificador-emociones-qmng.onrender.com>

- **Código:** <https://github.com/soymiguelgomez/clasificador-emociones>
- **Plataforma:** Render, plan gratuito, con Python 3.13.18 y gunicorn.
- **Verificación:** el 9 de noviembre de 2026 la prueba de humo contra esta URL superó las 7 comprobaciones.
- **Primera visita lenta:** el servicio se duerme tras 15 minutos sin tráfico y puede tardar alrededor de un minuto en responder.

## Despliegue

La guía paso a paso está en [docs/DESPLIEGUE.md](docs/DESPLIEGUE.md). En resumen:

1. Sube el código a GitHub. El `.gitignore` excluye los datos, el checkpoint de entrenamiento y los entornos.
2. En Render, crea el servicio con **New → Blueprint**, que lee `render.yaml`.
3. Verifica el servicio desplegado:

```bash
python scripts/smoke_test.py https://clasificador-emociones-qmng.onrender.com
```

En producción la red se ejecuta con NumPy (`MODEL_BACKEND=numpy`):

| Motor | Memoria del proceso | Predicciones |
|---|---|---|
| TensorFlow | 431 MB | — |
| NumPy | 62 MB | Idénticas a Keras en las 37.139 frases de validación (diferencia máxima de probabilidad: 2,7·10⁻⁶) |

Con NumPy cabe en los 512 MB del plan gratuito de Render.

## Limitaciones conocidas

- **Idioma.** Solo entiende inglés y frases parecidas a las del corpus («i feel…»). Con texto en español, la aplicación avisa de que el resultado no es fiable.
- **Negación.** No la interpreta: «i am not happy» se clasifica como alegría.
- **Una sola emoción.** Elige una emoción aunque la frase mezcle varias o no exprese ninguna.
- **Etiquetas automáticas.** Las etiquetas tienen ruido: algunos «errores» del modelo son, en realidad, errores de etiqueta.
- **Confianza.** Una confianza alta no garantiza un acierto: el 20 % de los errores tuvo más del 90 % de confianza.
- **Plan gratuito de Render.** El servicio se duerme tras 15 minutos sin tráfico y tarda alrededor de un minuto en despertar.

## Entrega (ZIP)

```bash
python scripts/make_zip.py
```

El script crea `dist/proyecto_final_ia.zip` con:

- el código fuente;
- este README;
- los modelos y los reportes;
- el informe en PDF y DOCX.

Deja fuera los entornos virtuales, las cachés, los datos originales, el checkpoint de entrenamiento y cualquier secreto.
Con `--dry-run` muestra la lista de archivos sin crear el ZIP.

## Créditos y referencias

- Datos: Elgiriyewithana, N. (2024). *Emotions* [Conjunto de datos]. Kaggle. <https://www.kaggle.com/datasets/nelgiriyewithana/emotions>
- Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018). CARER: Contextualized affect representations for emotion recognition. *Proceedings of EMNLP 2018*, 3687–3697. <https://doi.org/10.18653/v1/D18-1404>
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. <https://doi.org/10.1162/neco.1997.9.8.1735>
- Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing, 45*(11), 2673–2681. <https://doi.org/10.1109/78.650093>
- Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. *Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining*, 29–39.

La lista completa (24 referencias en APA 7) está en el informe: `docs/informe/`.

**Autor:** Miguel Andres Gomez Gutierrez. Programa de Ingeniería de Sistemas, Corporación Universitaria del Huila – CORHUILA. Asignatura Inteligencia Artificial, semestre 2026-2. Docente: José Miguel Llanos Mosquera.
