# Sustentación (15 minutos): guion y preguntas

- **Formato:** 15 diapositivas.
- **Guion:** cada diapositiva lleva sus notas de orador; aquí están completas.

## 1. Distribución del tiempo

| # | Diapositiva | Tema | Tiempo | Acumulado |
|---|---|---|---|---|
| 1 | ¿Qué emoción expresa una frase? | Portada | 0:30 | 0:30 |
| 2 | Leer miles de mensajes a mano no es viable | Problema y motivación | 0:45 | 1:15 |
| 3 | Objetivos | Objetivos | 0:30 | 1:45 |
| 4 | 416.809 tuits, seis emociones | Base de datos | 1:15 | 3:00 |
| 5 | 22.288 frases con dos etiquetas | Calidad y preparación | 1:00 | 4:00 |
| 6 | Seis fases, de los datos a la web | Metodología CRISP-DM | 1:00 | 5:00 |
| 7 | Una LSTM que lee la frase en ambos sentidos | Arquitectura | 1:30 | 6:30 |
| 8 | La parada temprana eligió la época 3 | Entrenamiento | 1:00 | 7:30 |
| 9 | Supera a la línea base en las seis emociones | Resultados | 1:30 | 9:00 |
| 10 | Errores entre emociones vecinas | Evaluación | 1:00 | 10:00 |
| 11 | Demostración en vivo | Aplicación web | 2:00 | 12:00 |
| 12 | La misma red: de 431 MB a 62 MB | Despliegue | 1:15 | 13:15 |
| 13 | Reconoce vocabulario, no comprende la frase | Limitaciones | 0:45 | 14:00 |
| 14 | Tres ideas para llevarse | Conclusiones | 0:45 | 14:45 |
| 15 | Gracias. ¿Preguntas? | Cierre | 0:15 | 15:00 |

Si vas justo de tiempo, recorta primero la diapositiva 6 (30 s) y la 10 (30 s). **Nunca recortes la demostración.**

## 2. Guion

**1. Portada (0:00–0:30).** Buenos días. Mi proyecto responde a una pregunta sencilla: ¿qué emoción expresa una frase? Construí una red neuronal que lee frases en inglés y decide si expresan tristeza, alegría, amor, ira, miedo o sorpresa, y la integré en una aplicación web. En los próximos 15 minutos explicaré el problema, los datos, la red y los resultados reales; después haré una demostración y terminaré con el despliegue y las limitaciones.

**2. Problema (0:30–1:15).** Las personas expresan cómo se sienten en reseñas, redes y mensajes de soporte, y nadie puede leer miles de mensajes a mano.
- El sistema recibe una frase en inglés, por ejemplo «i feel so nervous about my exam tomorrow».
- Devuelve la emoción predominante, aquí miedo, junto con la probabilidad de cada una de las seis emociones.

Técnicamente es un problema de PLN: clasificación de texto multiclase. Lo podrían usar equipos de atención al cliente para priorizar mensajes o investigadores para analizar opiniones. No es una herramienta de diagnóstico psicológico.

**3. Objetivos (1:15–1:45).** El objetivo general es desarrollar y evaluar la red e integrarla en una aplicación web lista para producción. Lo desglosé en seis objetivos:
1. analizar la calidad de los datos;
2. prepararlos sin fuga de información;
3. entrenar y regularizar la red;
4. evaluarla contra una línea base;
5. construir la aplicación;
6. preparar el despliegue.

**4. Datos (1:45–3:00).** Los datos son el conjunto «Emotions» de Kaggle: 416.809 mensajes de Twitter en inglés, cada uno con una etiqueta del 0 al 5. Comprobé que mi archivo coincide byte a byte con el publicado. Hay tres rasgos importantes:
- **Desbalance:** alegría y tristeza suman casi dos tercios y sorpresa solo el 3,6 %, una relación de 9,4 a 1.
- **Longitud:** son textos cortos, de 17 palabras de mediana.
- **Dominio:** casi todos son frases del tipo «I feel…».

**5. Calidad de los datos (3:00–4:00).** Encontré 22.288 frases idénticas etiquetadas con emociones distintas.
- Casi siempre se confunden emociones cercanas, sobre todo alegría con amor y miedo con sorpresa. Esto indica un etiquetado automático.
- Para esas frases no existe una respuesta correcta, así que las excluí: 44.582 filas, el 10,7 %. También quité 838 duplicados exactos.
- Quedaron 371.389 textos únicos, que dividí 80/10/10 por estratos.
- Como cada texto es único, ninguna frase puede aparecer en entrenamiento y prueba a la vez; una comprobación automática lo confirma. Así se evita la fuga de información.

**6. Metodología (4:00–5:00).** Seguí CRISP-DM. Destaco cuatro decisiones de preparación:
- **Limpieza:** es la misma función en el entrenamiento y en la web, así el modelo recibe en producción lo mismo que vio al entrenar.
- **Vocabulario:** 30.000 palabras, construido solo con el conjunto de entrenamiento.
- **Longitud:** 60 palabras por frase, que cubren el 99,9 % de los textos sin recortar.
- **Desbalance:** en vez de borrar datos, ponderé la pérdida. Un error en sorpresa cuesta unas 13 veces más que uno en alegría.

**7. Arquitectura (5:00–6:30).** La red tiene cinco piezas:
1. **Entrada:** 60 índices de palabra.
2. **Embedding:** convierte cada palabra en un vector de 128 números que se aprende al entrenar.
3. **LSTM bidireccional:** una LSTM recorre la frase de izquierda a derecha y otra de derecha a izquierda. Sus compuertas deciden qué recordar y qué olvidar, lo que resuelve el desvanecimiento del gradiente.
4. **Capa densa con ReLU:** combina las características.
5. **Softmax:** da las seis probabilidades.

Son 3,9 millones de parámetros, unos 15 MB. Usé Adam con entropía cruzada ponderada, dropout y parada temprana. No elegí BERT porque ocupa cientos de MB, necesita GPU y no cabe en un servidor gratuito de 512 MB.

**8. Entrenamiento (6:30–7:30).** Esta es la curva real de pérdida.
- Desde la primera época la validación supera el 98 %.
- La mejor pérdida de validación llega en la época 3, con un 99,14 % de exactitud.
- Después, la pérdida de entrenamiento sigue bajando y la de validación sube un poco: es el inicio del sobreajuste.
- La parada temprana se detuvo en la época 6 y restauró los pesos de la época 3.
- Todo tardó menos de 20 minutos sin GPU.

**9. Resultados (7:30–9:00).** Los resultados son sobre 37.139 frases que el modelo nunca vio.
- **Exactitud:** 99,27 %, con 272 errores.
- **F1 macro:** 0,990. Es la métrica principal, porque pesa igual cada emoción.
- **Línea base** (TF-IDF con regresión logística): F1 macro de 0,954.
- La red gana en las seis emociones. La mayor mejora está en las clases minoritarias: amor pasa de 0,94 a 0,998 y sorpresa de 0,93 a 0,98.

**10. Matriz de confusión (9:00–10:00).** La diagonal va del 98,2 % en miedo al 99,9 % en amor.
- Los errores se concentran en emociones negativas vecinas: tristeza predicha como ira (43 casos) y miedo como ira (30).
- Muchos parecen errores de etiqueta. Por ejemplo, «i feel nervous anxious» está etiquetada como tristeza y el modelo dice miedo.
- Por eso, el 99,27 % está cerca del techo que permiten las etiquetas automáticas.

**11. Demostración (10:00–12:00).** La secuencia de la demo está en la sección 3.

**12. Despliegue (12:00–13:15).** Elegí Render, que tiene un plan gratuito de 512 MB.
- Con TensorFlow cargado, la aplicación ocupaba 431 MB.
- Para resolverlo, separé entrenamiento e inferencia: exporté los pesos entrenados y reimplementé la LSTM en NumPy. El proceso bajó a 62 MB.
- Las predicciones coinciden en el 100 % de 37.139 frases, con una diferencia máxima de probabilidad de 0,0000027.
- Cada frase tarda 1,4 ms.
- La configuración está en `render.yaml`, con gunicorn y sin contraseñas.
- Está publicada en <https://clasificador-emociones-qmng.onrender.com>, y la prueba de humo superó las 7 comprobaciones contra esa URL pública.
- La abrí antes de empezar porque el plan gratuito se duerme tras 15 minutos.

**13. Limitaciones (13:15–14:00).** El modelo reconoce vocabulario, pero no comprende la frase:
- No entiende la negación: «i am not happy» lo clasifica como alegría.
- Si una frase mezcla emociones, elige solo una.
- Siempre asigna alguna emoción, aunque la frase sea neutra.
- No entiende español, aunque la aplicación lo detecta y avisa.
- El 20 % de los errores tuvo más del 90 % de confianza.

Como mejoras, propongo un Transformer multilingüe, añadir ejemplos con negación, una clase neutra y calibrar la confianza.

**14. Conclusiones (14:00–14:45).** Tres ideas:
1. Una BiLSTM modesta, entrenada en CPU, alcanza un 99,27 % de exactitud y supera a la línea base en las seis emociones.
2. Los datos pesaron tanto como el modelo: detectar las frases contradictorias permitió evaluar con etiquetas fiables y sin fuga.
3. Separar entrenamiento e inferencia hizo viable la producción gratuita sin cambiar ninguna predicción.

**15. Cierre (14:45–15:00).** Muchas gracias. Quedo atento a sus preguntas.

## 3. Preparación de la demostración

**Antes de entrar:**
- [ ] Abrir <https://clasificador-emociones-qmng.onrender.com> **1 minuto antes**, porque el plan gratuito se duerme tras 15 minutos sin uso. Plan B: `python wsgi.py` en local.
- [ ] Tener abiertas dos pestañas: `/` y `/resultados`.
- [ ] Tener las frases copiadas en un bloc de notas.

**Secuencia (2 minutos):**
1. Escribe «i am feeling really grateful for my friends today». Resultado esperado: alegría con 94,5 % y tristeza con 3,2 % como segunda opción. Lee la explicación.
2. Pulsa el ejemplo «i feel so nervous about my exam tomorrow». Resultado esperado: miedo, >99,9 %.
3. Escribe «me siento muy triste hoy». El modelo conoce solo 1 de 5 palabras y aparece el aviso «Resultado poco fiable». Explica que la aplicación es honesta con sus límites.
4. Abre «Resultados del modelo»: las cifras y las gráficas salen de los archivos de evaluación reales.
5. Cierra con esta idea: el modelo se carga una sola vez, nunca se reentrena en una petición, y la entrada pasa por la misma limpieza y tokenización que en el entrenamiento.

**Plan B:** si falla la conexión, usa las capturas de la diapositiva 11.

## 4. Preguntas probables del docente

**¿Por qué una LSTM bidireccional y no BERT u otro Transformer?**
Por adecuación, no por novedad.
- Los textos son cortos (17 palabras de mediana) y hay casi 300.000 ejemplos de entrenamiento. En ese escenario una BiLSTM aprende bien.
- BERT-base tiene unos 110 millones de parámetros (más de 400 MB), necesita GPU para ajustarse en un tiempo razonable y no cabría en un servidor de 512 MB. La red elegida tiene 3,9 millones y se entrenó en 20 minutos en CPU.
- BERT sí manejaría mejor la negación; por eso lo propongo como mejora.

**¿Por qué no una CNN?**
Una CNN de texto detecta n-gramas locales y también sería válida. Elegí la LSTM porque modela la secuencia completa y el orden de las palabras, que importa en frases como «not happy». Además, la línea base TF-IDF ya cubre el enfoque de «bolsa de n-gramas», así que la comparación es informativa.

**¿Cómo evitaste la fuga de información?**
Con cinco medidas:
1. Deduplicar antes de dividir: cada texto es único, así que ninguna frase puede caer en dos conjuntos. Una función comprueba que la intersección es 0 y detiene el proceso si no lo es.
2. Construir el vocabulario solo con el conjunto de entrenamiento.
3. Calcular los pesos por clase solo con entrenamiento.
4. Tomar las decisiones del EDA sobre longitudes y vocabulario solo con entrenamiento.
5. Usar el conjunto de prueba una única vez, al final.

**¿Eliminar el 10,7 % de los datos no sesga el modelo?**
Sí cambia la distribución: amor pierde el 29 % y sorpresa el 34 %. Lo documenté. Pero esos textos tienen dos etiquetas para la misma frase, así que no hay una respuesta correcta que aprender ni evaluar. Mantenerlos habría metido ruido en el entrenamiento y habría hecho imposible medir bien. Aun así quedan 24.364 ejemplos de amor y 9.823 de sorpresa. La alternativa sería una formulación multietiqueta, que propongo como mejora.

**¿Cómo trataste el desbalance? ¿Por qué no sobremuestreo o SMOTE?**
Con pesos en la función de pérdida: n / (k · n_clase). Sorpresa pesa 6,30 y alegría 0,48. No se crean ni se borran datos, así que no hay riesgo de duplicar textos ni de perder información. SMOTE interpola vectores numéricos y no tiene sentido directo sobre secuencias de palabras. El resultado: ninguna clase baja de 0,98 de F1.

**¿Cómo sabes que no hay sobreajuste?**
Por tres razones:
1. La parada temprana vigila la pérdida de validación y conservó la época 3, justo antes de que esa pérdida subiera.
2. La exactitud en prueba (99,27 %) es igual o mejor que en validación (99,14 %), así que generaliza a datos nunca vistos.
3. Hay dropout de 0,2 y 0,3.

**¿Por qué la exactitud de validación es mayor que la de entrenamiento en la época 1?**
Porque la de entrenamiento es un promedio durante toda la época, mientras el modelo todavía aprendía, y se calcula con dropout activo. La de validación se mide al final de la época, sin dropout.

**Un 99 % de exactitud, ¿no es sospechoso?**
Es la pregunta correcta.
- Descarté la fuga con la verificación de cero textos compartidos.
- La línea base lineal ya obtiene un 96,5 %. El vocabulario emocional es muy informativo en este corpus: palabras como *gloomy* o *apprehensive* pertenecen a una emoción en el 95–99 % de los casos.
- Precisamente por eso probé frases fuera del corpus. Ahí el modelo falla con la negación y con el español.

La cifra es real para este dominio, pero no significa que el modelo comprenda las emociones.

**¿Qué hace la capa Embedding y para qué sirve mask_zero?**
Embedding es una tabla de 30.000 × 128 que se aprende: a cada palabra le asigna un vector, y palabras con usos parecidos quedan cerca. `mask_zero=True` marca el índice 0, el relleno, para que la LSTM lo ignore; así, las frases cortas no se contaminan con ceros.

**¿Cómo funciona una LSTM?**
Mantiene un estado de celda c y tres compuertas sigmoides:
- la de **entrada** decide qué información nueva entra;
- la de **olvido** decide qué se borra de la memoria;
- la de **salida** decide qué se expone en el estado oculto h.

La memoria se actualiza de forma aditiva, con c_t = f·c_{t−1} + i·g, y eso permite que el gradiente fluya sin desvanecerse. En la versión bidireccional se concatenan los estados finales de ambos sentidos (64 + 64 = 128).

**¿Por qué F1 macro y no solo exactitud?**
Porque con clases desbalanceadas, un modelo que ignorara sorpresa (3,6 %) aún podría tener una exactitud alta. El F1 macro promedia el F1 de cada emoción con el mismo peso, así que penaliza fallar en las clases pequeñas.

**¿Por qué NumPy en producción? ¿Sigue siendo la red entrenada?**
Sí. Son los mismos pesos entrenados por Keras, exportados a `.npz`, con las mismas ecuaciones de la LSTM implementadas en NumPy.
- Lo verifiqué en 37.139 frases: diferencia máxima de probabilidad de 2,7·10⁻⁶ y 100 % de predicciones iguales.
- Un test automático lo vuelve a comprobar.
- El motivo es la memoria: TensorFlow ocupa 431 MB y NumPy 62 MB, frente a los 512 MB del plan gratuito.

**¿Cómo garantizas que la web aplica las mismas transformaciones que el entrenamiento?**
Se usa el mismo código: la función `clean_text` y el tokenizador guardado en `tokenizer.json`. Hay pruebas para la limpieza y para el tokenizador.

**¿Qué pasa con textos en español o con negaciones?**
Con español, el modelo apenas conoce las palabras. La aplicación calcula la proporción de palabras conocidas y, si es menor del 60 %, marca el resultado como poco fiable. La negación no la maneja; es la limitación principal, y está documentada.

**¿Qué indica la gráfica de confianza o el ECE?**
El ECE (0,0024) mide la diferencia entre la confianza declarada y el acierto real, y es bajo. La confianza media es del 99,7 % en los aciertos y del 68,8 % en los errores, así que una confianza baja es una buena alarma. Pero el 20 % de los errores tuvo más del 90 %, de modo que una confianza alta no garantiza nada.

**¿Por qué Flask y no Django?**
La aplicación tiene cinco rutas y no necesita base de datos, usuarios ni panel de administración. Flask lo resuelve con poco código y es más fácil de mantener; Django aportaría componentes que no se usan.

**¿Por qué CRISP-DM?**
Es el estándar de facto para proyectos de datos: sus seis fases cubren desde el problema hasta el despliegue y es iterativo. Encaja con lo que hice: volví a la preparación tras el análisis exploratorio.

**¿Es reproducible?**
Sí:
- La semilla está fija (42) y las versiones exactas están en los archivos de requisitos.
- Cada fase se ejecuta con un comando (`python -m src.data_prep`, `src.eda`, `src.train`, `src.baseline` y `src.evaluate`) y deja sus resultados en `reports/`.
- Hay 20 pruebas automatizadas.
