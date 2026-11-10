# Tabla de cumplimiento — Proyecto Final de Inteligencia Artificial

**Fuente:** enunciado «PROYECTO FINAL INTELIGENCIA ARTIFICIAL».

| Dato | Valor |
|---|---|
| Institución | CORHUILA, Ingeniería de Sistemas |
| Asignatura | Inteligencia Artificial, semestre 2026-2 |
| Docente | José Miguel Llanos Mosquera |
| Sustentación | Martes 10 de noviembre de 2026, 6:00–7:40 a. m., salón C5-508, sede Prado Alto |

**Leyenda:**
- ✅ Cumplido, con evidencia.
- ⚠️ Preparado; falta una acción del grupo.
- ❌ Pendiente y obligatorio.

## Instrucciones generales

| Instrucción | Estado | Evidencia |
|---|---|---|
| Grupos de máximo 3 estudiantes | ✅ | Trabajo individual: Miguel Andres Gomez Gutierrez (en la portada del informe, en la diapositiva 1 y en el README) |
| Google Colaboratory o Visual Studio Code | ✅ | Desarrollado en Visual Studio Code |
| Framework Django o Flask | ✅ | Flask 3.1 (`app/`); justificado en el informe |
| Python 3 | ✅ | Python 3.13 (`.python-version`) |
| Sustentación de 15 min: problema, código en un servidor de producción y resultados | ✅ | Presentación de 15 diapositivas, guion y 18 preguntas (`docs/sustentacion/`). La aplicación está en producción (requisito 5) |

## Requisitos

| # | Requisito | Estado | Evidencia |
|---|---|---|---|
| 1 | Dataset de Kaggle u otra fuente abierta para NLP o visión | ✅ | «Emotions» de Kaggle (Elgiriyewithana, 2024): 416.809 tuits, 6 emociones. Problema de NLP |
| 2 | Aplicar una de las metodologías vistas en clase | ⚠️ | Se aplicó CRISP-DM (informe, «Metodología» y Tabla 1). **Confirmar que es una de las vistas en clase**; si fue otra (KDD, SEMMA, TDSP…), hay que ajustar esa sección |
| 3 | Predicción con redes neuronales y técnicas de NLP | ✅ | Red BiLSTM (`src/model.py`) con limpieza, tokenización y *embeddings*. Prueba: exactitud 99,27 % y F1 macro 0,990 (`reports/metrics.json`) |
| 4 | Integrar la red en Django o Flask y mostrar al menos 3 gráficas | ✅ | La aplicación hace predicciones reales y la página `/resultados` muestra 9 gráficas reales |
| 5 | Desplegar todo el proyecto en un servidor de producción | ✅ | En Render: <https://clasificador-emociones-qmng.onrender.com>. Código en <https://github.com/soymiguelgomez/clasificador-emociones>. Prueba de humo contra la URL pública: 7/7 (9/11/2026) |
| 6 | Trabajo escrito de 10 páginas o más, en PDF y APA 7 | ✅ | `docs/informe/Informe_Proyecto_Final_IA.pdf`: 41 páginas y 24 referencias verificadas, con citas y referencias comprobadas automáticamente. Incluye todos los apartados pedidos (detalle abajo) |
| 7 | ZIP en Corhuila Digital con código, README.md y PDF, antes de la sustentación | ⚠️ | `python scripts/make_zip.py` genera `dist/proyecto_final_ia.zip`. **Hay que subirlo antes del 10/11/2026 a las 6:00 a. m.** |
| 8 | Sustentación en grupo | ⚠️ | Material listo; repartir las diapositivas entre los integrantes |

### Detalle del requisito 6

| Apartado pedido | Dónde está en el informe |
|---|---|
| Portada | Página 1: título, autor, programa, universidad, asignatura, docente y fecha |
| Tabla de contenido | Página 2 (campo automático de Word) |
| Índice de tablas y figuras | Página 4: 12 tablas y 12 figuras |
| Introducción | Sección «Introducción» |
| Objetivos | Objetivo general y 6 específicos |
| Desarrollo con todos los pasos de construcción de la red neuronal | Marco teórico, metodología, análisis de datos, preparación, diseño de la red, entrenamiento, evaluación, aplicación web y despliegue |
| Resultados obtenidos | Sección «Resultados obtenidos»; Tablas 8 y 9; Figuras 5–9 |
| Conclusiones | Sección «Conclusiones» |
| Referencias | 24 referencias en APA 7 |
| Formato PDF y APA 7 | Times New Roman 12, doble espacio, márgenes de 1″, número de página y tablas y figuras en formato APA |

## Acciones pendientes, por prioridad (antes del martes a las 6:00 a. m.)

1. ~~Desplegar en Render~~ **Hecho:** <https://clasificador-emociones-qmng.onrender.com>, con la URL ya añadida al README, al informe y a las diapositivas.
2. ~~Escribir los integrantes del grupo~~ **Hecho:** Miguel Andres Gomez Gutierrez.
3. **Confirmar la metodología** vista en clase (requisito 2).
4. **Generar el ZIP** con `python scripts/make_zip.py`; no debe mostrar ningún `[AVISO]`. **Subirlo a Corhuila Digital** antes de las 6:00 a. m.
5. **Ensayar** los 15 minutos y abrir la URL de Render un minuto antes, porque el plan gratuito se duerme tras 15 minutos sin tráfico.
