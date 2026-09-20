# Whisper Encoder/Decoder Explorer

Página web (FastAPI) para subir un audio y **validar cómo el encoder y el decoder de
Whisper transcriben** la entrada: mel spectrograma, activaciones del encoder capa a capa
y generación del decoder token a token con su top-5 de probabilidades.

## Requisitos

- Python 3.12 y ffmpeg instalados (obligatorio para Whisper).
- Entorno virtual `env` (ver `AGENTS.md`).

## Instalación

```powershell
python -m venv env
env\Scripts\pip.exe install -r requirements.txt
```

> Nota: `requirements.txt` usa PyTorch **CPU** (`+cpu`), correcto para máquinas sin
> GPU NVIDIA. En la primera ejecución, Whisper descarga el modelo `base` (~145 MB).

## Ejecutar

```powershell
.\run.ps1                 # o: env\Scripts\python.exe -m uvicorn app.main:app --reload
```

Abrir http://127.0.0.1:8000 — documentación de la API en http://127.0.0.1:8000/docs

## Endpoints

| Método | Ruta              | Descripción                                       |
|--------|-------------------|---------------------------------------------------|
| GET    | `/`               | Página web (subida + visualización)               |
| POST   | `/api/transcribe` | multipart `file` + `language` (auto\|código) + `task` |
| GET    | `/api/model`      | Info del modelo cargado                           |

## Estructura

- `app/main.py` — servidor FastAPI (el modelo se carga una vez al importar).
- `app/whisper_service.py` — transcripción oficial + pipeline encoder/decoder.
- `app/templates/`, `app/static/` — interfaz.
- `uploads/` — archivos temporales (se eliminan tras procesar).

## Notas técnicas

- El encoder/decoder se instrumenta con `torch` **forward hooks** (salidas de cada
  `ResidualAttentionBlock`) y un bucle decoder que replica `whisper/decoding.py`
  (`install_kv_cache_hooks()`, `SuppressBlank`, `SuppressTokens`, greedy T=0).
- El pipeline visualiza la ventana de **30 s** (`N_FRAMES = 3000`) aunque la
  transcripción se hace sobre el audio completo.
- El primer request tarda más (carga del modelo). Los análisis se serializan con un
  semáforo para no saturar la CPU.