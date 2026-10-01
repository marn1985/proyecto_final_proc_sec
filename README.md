# Whisper · Explorador Encoder/Decoder

**Proyecto Final — Procesamiento de Datos Secuenciales**
Maestría en Inteligencia Artificial y Ciencias de Datos · Universidad Autónoma de Occidente

**Integrantes:**
- Mónica Giraldo
- Andrés Hernández
- Armando Betancourt
- Mario Ramírez

---

## 1. Resumen (Abstract)

Este proyecto aplica **Whisper**, una arquitectura Transformer encoder–decoder desarrollada por OpenAI, al reconocimiento automático del habla (transcripción) y a la traducción de voz hacia el inglés. No se entrenó ningún modelo: se usaron los pesos preentrenados publicados por los autores y se implementó el proceso de inferencia paso a paso, separando explícitamente el preprocesamiento del audio, el encoder, la decodificación autoregresiva y la atención cruzada.

Como escenario de aplicación se construyó un chat que permite enviar mensajes de voz y obtenerlos como texto, ya sea transcritos en su idioma original o traducidos al inglés. La solución es una aplicación en Streamlit con dos modos. El **Asistente** es ese chat: el usuario conversa, envía o graba un audio y recibe la transcripción o la traducción. El **Explorador** muestra lo que ocurre dentro del modelo: el espectrograma Mel de entrada, las representaciones del encoder, la probabilidad con que el decoder eligió cada token y un mapa de atención cruzada que indica qué parte del audio consultó el modelo para generar cada palabra.

> **Resultados:** _[completar con el resumen de la sección 6, por ejemplo: "Con el modelo base se obtuvo un WER promedio de X % en audios en español y un factor de tiempo real de Y en CPU"]_.

---

## 2. Introducción

### Artículo base

- **Artículo:** A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey e I. Sutskever, *Robust Speech Recognition via Large-Scale Weak Supervision*, 2022. [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
- **Repositorio original:** [github.com/openai/whisper](https://github.com/openai/whisper)

### Contexto del problema

El reconocimiento del habla consiste en convertir una señal de audio, que es una secuencia continua de muestras, en una secuencia discreta de palabras. Es un problema de secuencia a secuencia con dos dificultades: las dos secuencias tienen longitudes muy distintas (30 segundos de audio son 480 000 muestras, pero pueden ser solo unas 80 palabras) y no existe una correspondencia fija entre un instante del audio y una palabra.

Los sistemas tradicionales se entrenaban con conjuntos de datos pequeños y muy curados, y fallaban al cambiar de micrófono, acento o nivel de ruido. Whisper aborda el problema entrenando con una cantidad masiva de audio de internet y su transcripción, aunque esas etiquetas no sean perfectas.

### Motivación

El problema encaja con el objetivo del curso: es una secuencia compleja de lenguaje, la arquitectura es un Transformer encoder–decoder clásico y la atención cruzada tiene una interpretación visual muy clara, porque muestra cómo el texto se alinea con el audio.

### Escenario de aplicación

Como caso de uso se planteó un **asistente conversacional tipo chat** que recibe mensajes de voz y los convierte en texto. El usuario conversa con el asistente, le indica qué necesita y le envía el audio, ya sea grabándolo con el micrófono o adjuntando un archivo. El asistente ofrece dos servicios:

- **Transcripción:** convierte el audio en texto en el mismo idioma en que se habló. Por ejemplo, un audio en español se convierte en texto en español.
- **Traducción:** convierte el audio en texto traducido al inglés. Por ejemplo, un audio en español se convierte en texto en inglés.

El asistente identifica la tarea a partir de lo que el usuario escribe (por ejemplo, "quiero traducir este audio") o de los botones de la interfaz. Si recibe un audio antes de saber qué hacer con él, pregunta si debe transcribirlo o traducirlo. Una vez elegida la tarea, la mantiene para los audios siguientes hasta que el usuario pida cambiarla.

El escenario muestra en la práctica la innovación multitarea de Whisper: el mismo modelo, con los mismos pesos, resuelve las dos tareas. Lo único que cambia es el token de tarea (`<|transcribe|>` o `<|translate|>`) que se le entrega al decoder al inicio de la secuencia.

### Objetivo

Implementar la inferencia de Whisper con pesos preentrenados, explicar en detalle su arquitectura y su mecanismo de atención (incluida la generación de Q, K y V) y construir una interfaz interactiva que permita cargar audio y observar el comportamiento interno del modelo.

---

## 3. Marco teórico

### 3.1 Arquitectura general

Whisper es un Transformer encoder–decoder en el que la entrada es audio y la salida es texto.

```
Audio (16 kHz)
   │
   ▼
Espectrograma log-Mel (80 × 3000)        ← 30 s, un frame cada 10 ms
   │
   ▼
2 convoluciones 1D + GELU (stride 2)     ← 3000 frames → 1500 posiciones
   │
   ▼
+ Codificación posicional sinusoidal
   │
   ▼
ENCODER: N bloques [Self-Attention → MLP]
   │
   ▼
Representación del audio (1500 × d_model)  ──────────┐
                                                     │ K, V
Tokens especiales + tokens ya generados              │
   │                                                 │
   ▼                                                 │
Embedding + codificación posicional aprendida        │
   │                                                 │
   ▼                                                 │
DECODER: N bloques [Masked Self-Attention → Cross-Attention → MLP]
   │
   ▼
Capa lineal (pesos compartidos con el embedding) → Softmax → siguiente token
```

Cada bloque usa conexiones residuales y normalización por capa aplicada **antes** de cada subcapa (*pre-norm*), lo que estabiliza el entrenamiento de redes profundas.

Los modelos usados en este proyecto tienen estas dimensiones:

| Modelo | Capas (encoder / decoder) | d_model | Cabezas | d_k por cabeza | Parámetros |
|---|---|---|---|---|---|
| tiny | 4 / 4 | 384 | 6 | 64 | 39 M |
| base | 6 / 6 | 512 | 8 | 64 | 74 M |
| small | 12 / 12 | 768 | 12 | 64 | 244 M |

### 3.2 Entradas y salidas

**Entrada del encoder.** El audio se remuestrea a 16 kHz, se recorta o se rellena con silencio hasta 30 segundos y se convierte en un espectrograma log-Mel de 80 bandas, calculado con ventanas de 25 ms cada 10 ms. El resultado es una matriz de 80 × 3000. Dos convoluciones 1D reducen el eje temporal a la mitad, de modo que el encoder trabaja con 1500 posiciones, cada una equivalente a 20 ms de audio.

**Entrada del decoder.** Una secuencia de tokens especiales que le indican al modelo qué hacer, seguida de los tokens ya generados:

```
<|startoftranscript|> <|es|> <|transcribe|> <|notimestamps|>  …tokens generados…
```

**Salida.** En cada paso, el decoder produce un vector de probabilidades sobre las 51 865 subpalabras del vocabulario. Se elige el token más probable, se agrega a la entrada y se repite hasta generar `<|endoftext|>`.

### 3.3 Mecanismo de atención

Todas las capas de atención de Whisper calculan:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- **QKᵀ** compara lo que busca cada posición (Q) con lo que ofrece cada una de las otras (K). El resultado es una tabla de puntajes.
- **÷ √d_k** evita que los puntajes crezcan con la dimensión; sin esta división, el softmax se satura y los gradientes se anulan.
- **Softmax** convierte cada fila en porcentajes que suman 1: los pesos de atención.
- **× V** mezcla la información de las posiciones según esos pesos.

Whisper usa tres tipos de atención:

| Atención | Dónde | Q viene de | K y V vienen de | Máscara |
|---|---|---|---|---|
| Self-attention | Encoder | Audio | Audio | No: el audio está completo |
| Masked self-attention | Decoder | Tokens | Tokens anteriores | Sí: no puede ver tokens futuros |
| Cross-attention | Decoder | Tokens (decoder) | Salida del encoder | No |

### 3.4 Cómo se generan Q, K y V

En cada capa de atención, Q, K y V se obtienen con tres proyecciones lineales aprendidas:

$$Q = X_q W_Q + b_Q, \qquad K = X_{kv} W_K, \qquad V = X_{kv} W_V + b_V$$

En el modelo base, cada matriz W tiene tamaño 512 × 512. Un detalle propio de la implementación de Whisper es que la proyección de K **no tiene bias**.

- En la **self-attention**, X_q y X_kv son la misma secuencia.
- En la **cross-attention**, X_q es el estado del decoder y X_kv es la salida del encoder. K y V se calculan una sola vez a partir del audio, mientras que Q cambia con cada token generado.

Luego, Q, K y V se dividen en h cabezas. En el modelo base hay 8 cabezas de 512 / 8 = 64 dimensiones cada una. Cada cabeza calcula su propia atención, los resultados se concatenan y una proyección lineal de salida los combina.

### 3.5 Innovaciones

1. **Supervisión débil a gran escala.** El modelo se entrenó con 680 000 horas de audio de internet con transcripciones no verificadas manualmente, en lugar de conjuntos pequeños y curados. Esto lo hace robusto a acentos, ruido y distintos micrófonos sin necesidad de reentrenarlo para cada dominio.
2. **Formato multitarea con tokens especiales.** Un solo modelo transcribe, traduce al inglés, detecta el idioma y, opcionalmente, predice marcas de tiempo. La tarea se indica con tokens al inicio de la secuencia del decoder; no hacen falta modelos ni cabezas de salida distintos.
3. **Multilingüe por diseño.** El mismo modelo maneja 99 idiomas con un vocabulario de subpalabras compartido.
4. **Arquitectura estándar, sin ajustes especiales.** Los autores usaron deliberadamente un Transformer encoder–decoder casi sin modificaciones. La mejora proviene de los datos y del formato multitarea, no de cambios en la arquitectura.
5. **Funciona sin ajuste fino (*zero-shot*).** Se evalúa directamente en conjuntos de prueba que no vio durante el entrenamiento.

---

## 4. Metodología

### Proceso de implementación

1. Selección del artículo y verificación de que el código y los pesos preentrenados estuvieran disponibles.
2. Estudio de la arquitectura en el artículo y en el código fuente de `whisper/model.py`.
3. Implementación manual de la inferencia: en lugar de usar únicamente la función `transcribe()`, se separaron el preprocesamiento, el encoder, el bucle de decodificación y la extracción de la atención cruzada.
4. Construcción de la interfaz en Streamlit con los modos Asistente y Explorador.
5. Pruebas con audios propios y medición de métricas.

### Herramientas

| Herramienta | Uso |
|---|---|
| Python 3.11 / 3.12 | Lenguaje principal |
| PyTorch | Ejecución del modelo |
| openai-whisper | Arquitectura, pesos y tokenizador |
| Streamlit | Interfaz interactiva |
| Matplotlib, pandas | Visualizaciones y tablas |
| imageio-ffmpeg | Decodificación de audio sin instalar ffmpeg en el sistema |

### Uso de pesos preentrenados

No se entrenó ni se ajustó el modelo. Los pesos oficiales se descargan automáticamente la primera vez que se llama a `whisper.load_model()` y quedan en la caché local. Todo el trabajo se centra en la inferencia y en la interpretación de lo que ocurre dentro del modelo.

---

## 5. Desarrollo e implementación

### Estructura del repositorio

```
├── app_whisper_streamlit.py   # Aplicación (Asistente + Explorador)
├── requirements.txt           # Dependencias
├── INSTRUCCIONES.md           # Guía detallada de instalación
└── README.md
```

### Pasos para ejecutar

```bash
python -m venv .venv
# Windows:      .\.venv\Scripts\Activate.ps1
# macOS/Linux:  source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app_whisper_streamlit.py
```

La aplicación se abre en `http://localhost:8501`. La guía completa, con la solución de errores frecuentes, está en [`INSTRUCCIONES.md`](INSTRUCCIONES.md).

### Carga de pesos

```python
@st.cache_resource
def cargar_modelo(nombre):
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(nombre, device=dispositivo)
```

`whisper.load_model` descarga el archivo de pesos oficial (unos 140 MB para el modelo base) y construye la red. `st.cache_resource` evita recargarlo cada vez que se interactúa con la interfaz. La aplicación usa la GPU automáticamente si está disponible.

### Preprocesamiento

```python
audio = cargar_audio(ruta)                    # mono, 16 kHz, valores en [-1, 1]
audio_30s = whisper.pad_or_trim(audio)        # exactamente 30 s (480 000 muestras)
mel = whisper.log_mel_spectrogram(audio_30s)  # (80, 3000)
```

### Inferencia paso a paso (modo Explorador)

**1. Detección de idioma.** El decoder recibe solo `<|startoftranscript|>`, y se comparan las probabilidades de los tokens de idioma.

**2. Encoder.** Se ejecuta una sola vez por audio:

```python
audio_features = modelo.embed_audio(mel.unsqueeze(0))   # (1, 1500, 512)
```

**3. Decoder autoregresivo (greedy, temperatura 0).** En cada paso se pasa toda la secuencia actual, se toma la distribución del último token, se elige el más probable y se agrega a la secuencia:

```python
tokens = list(tok.sot_sequence_including_notimestamps)
for _ in range(dims.n_text_ctx // 2):
    logits = modelo.decoder(torch.tensor([tokens]), audio_features)[0, -1]
    siguiente = int(logits.argmax())
    if siguiente == tok.eot:
        break
    tokens.append(siguiente)
```

**4. Extracción de la atención cruzada.** Se desactiva SDPA (la implementación optimizada de PyTorch, que no devuelve la matriz de atención) y se registra un *hook* en cada capa `cross_attn` del decoder para capturar la matriz QKᵀ. Luego se aplica softmax y se promedian las cabezas y las capas superiores. El resultado es una matriz de tokens × posiciones de audio.

### Flujo del chat (modo Asistente)

1. El asistente saluda y ofrece las dos tareas: transcribir o traducir al inglés.
2. El usuario elige la tarea escribiéndola en el chat o con los botones. La intención se detecta por palabras clave: por ejemplo, "traduc", "inglés" o "english" activan la traducción, y "transcrib", "texto" o "dictado" activan la transcripción.
3. El usuario envía el audio grabándolo con el micrófono o adjuntándolo con el clip de la barra de mensajes (formatos wav, mp3, m4a, flac, ogg y webm).
4. La aplicación llama a `modelo.transcribe(audio, task=tarea)`, donde `tarea` es `"transcribe"` o `"translate"`.
5. El asistente responde con el texto, el idioma detectado, la duración del audio y el tiempo de procesamiento, y queda listo para recibir otro audio con la misma tarea.

A diferencia del Explorador, el Asistente usa la función completa `transcribe()` de Whisper, que divide los audios largos en ventanas de 30 s. Por eso puede procesar mensajes de voz de cualquier duración.

### Visualizaciones del Explorador

1. **Entradas:** duración, frecuencia de muestreo, forma del tensor y espectrograma Mel.
2. **Salidas:** texto, idioma detectado con su probabilidad, número de tokens, log-probabilidad media y tiempo de análisis.
3. **Confianza por token:** probabilidad del token elegido y de la segunda mejor opción.
4. **Q, K y V:** de dónde viene cada tensor en cada tipo de atención y sus dimensiones.
5. **Atención cruzada:** mapa de calor de qué segmento del audio consultó el decoder para generar cada token.

---

## 6. Resultados y análisis

> **Instrucciones para el grupo:** esta sección debe completarse con pruebas propias. Las tablas tienen el formato sugerido; reemplacen los campos entre corchetes con sus mediciones.

### 6.1 Capturas de pantalla

| Captura | Descripción |
|---|---|
| `![Asistente](capturas/asistente.png)` | _[Modo Asistente transcribiendo un audio en español]_ |
| `![Espectrograma](capturas/espectrograma.png)` | _[Espectrograma Mel de entrada]_ |
| `![Confianza](capturas/confianza.png)` | _[Confianza por token]_ |
| `![Atención](capturas/atencion_cruzada.png)` | _[Mapa de atención cruzada]_ |

### 6.2 Métricas de desempeño

Se usaron dos métricas:

- **WER (Word Error Rate):** proporción de palabras incorrectas respecto a una transcripción de referencia. Cuanto más bajo, mejor.
- **RTF (Real-Time Factor):** tiempo de procesamiento dividido entre la duración del audio. Un valor menor que 1 significa que el modelo procesa más rápido que el tiempo real.

```python
import jiwer
wer = jiwer.wer(referencia.lower(), transcripcion.lower())
rtf = tiempo_proceso / duracion_audio
```

| Audio | Idioma | Condición | Modelo | WER | RTF |
|---|---|---|---|---|---|
| _[audio_1]_ | es | Silencio | base | _[ ]_ | _[ ]_ |
| _[audio_2]_ | es | Ruido de fondo | base | _[ ]_ | _[ ]_ |
| _[audio_3]_ | en | Silencio | base | _[ ]_ | _[ ]_ |
| _[audio_1]_ | es | Silencio | tiny | _[ ]_ | _[ ]_ |
| _[audio_1]_ | es | Silencio | small | _[ ]_ | _[ ]_ |

Equipo de prueba: _[CPU / GPU, memoria RAM]_.

### 6.3 Análisis

Puntos sugeridos para analizar con los resultados obtenidos:

- **Tamaño del modelo frente a precisión y velocidad.** Comparar cuánto mejora el WER al pasar de tiny a base y a small, y cuánto aumenta el tiempo.
- **Robustez al ruido.** Diferencia de WER entre audios limpios y audios con ruido.
- **Confianza por token.** Identificar en qué tokens bajó la probabilidad (por lo general nombres propios o palabras poco comunes) y si esos tokens coinciden con los errores del WER.
- **Atención cruzada.** Verificar si se forma una diagonal: los primeros tokens deberían atender al inicio del audio y los últimos al final. Esto muestra que el modelo aprendió a alinear texto y audio sin que nadie le indicara dónde está cada palabra.
- **Detección de idioma.** Probabilidad asignada al idioma correcto y casos de confusión, por ejemplo entre español y portugués.

---

## 7. Conclusiones

### Aprendizajes

_[Completar con las conclusiones del grupo. Algunas ideas:]_

- La atención cruzada es el puente entre las dos modalidades: Q viene del texto y K y V vienen del audio, y su mapa de pesos muestra de forma directa cómo el modelo alinea ambas secuencias.
- La división en tokens especiales permite que un mismo Transformer resuelva varias tareas solo con cambiar el inicio de la secuencia.
- Implementar la decodificación manualmente, en lugar de usar `transcribe()`, permitió entender que la salida se genera token por token y que el encoder se ejecuta una sola vez.

### Limitaciones

- **Ventana fija de 30 segundos.** El Explorador analiza solo los primeros 30 s del audio; el Asistente sí procesa audios largos dividiéndolos en ventanas.
- **Traducción solo hacia el inglés.**
- **Decodificación greedy.** Sin *beam search* ni ajuste de temperatura, el modelo puede repetir frases o generar texto que no está en el audio (alucinaciones), sobre todo con silencios largos.
- **Sin caché de K y V en el bucle manual.** Cada paso recalcula toda la secuencia, lo que hace la inferencia más lenta que la implementación oficial.
- **Nombres propios y vocabulario técnico.** Son los tokens con menor confianza.

### Posibles mejoras

- Usar las `alignment_heads` del modelo para obtener mapas de atención cruzada más precisos.
- Visualizar también la self-attention del encoder y la máscara causal del decoder.
- Implementar caché de K y V y *beam search*.
- Procesar audios largos en el Explorador con ventanas deslizantes.
- Ajustar el modelo (*fine-tuning*) con vocabulario propio del dominio.

---

## 8. Referencias

[1] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey e I. Sutskever, "Robust speech recognition via large-scale weak supervision," en *Proc. 40th Int. Conf. Machine Learning (ICML)*, Honolulu, HI, EE. UU., 2023, pp. 28492–28518.

[2] OpenAI, "Whisper," repositorio de GitHub, 2022. [En línea]. Disponible: https://github.com/openai/whisper

[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser e I. Polosukhin, "Attention is all you need," en *Advances in Neural Information Processing Systems 30 (NeurIPS)*, Long Beach, CA, EE. UU., 2017, pp. 5998–6008.

[4] Streamlit Inc., "Streamlit documentation," 2026. [En línea]. Disponible: https://docs.streamlit.io

[5] A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," en *Advances in Neural Information Processing Systems 32 (NeurIPS)*, Vancouver, Canadá, 2019, pp. 8024–8035.
