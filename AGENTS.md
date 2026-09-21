# AGENTS.md

Aplicación web (FastAPI) que sube un audio y **valida el encoder/decoder de Whisper**:
mel spectrograma, activaciones del encoder por capa y generación del decoder token a token.
Python 3.12 · Whisper `base` · PyTorch CPU.

## Entorno de desarrollo

- TODO el trabajo se hace con el entorno virtual `env` (no crees `venv` ni otro nombre).
- Windows: usa siempre `env\Scripts\python.exe` / `env\Scripts\pip.exe`.
- Los requisitos fijos están en `requirements.txt`. No reinstales torch desde el índice
  PyPI normal: este equipo **no tiene GPU NVIDIA** y torch debe venir del índice CPU:
  `env\Scripts\pip.exe install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`
  (ya está instalado así: `torch==2.14.0+cpu`).
- ffmpeg es obligatorio (lo usa `whisper.load_audio` vía PATH). Instalado con winget
  (Gyan.FFmpeg) **no aparece en PATH de shells ya abiertos**; `app/whisper_service.py`
  lo localiza solo al importar. No asumas que `ffmpeg` existe en un shell nuevo sin el respaldo.
- El modelo Whisper se descarga una vez (~145 MB) a `%USERPROFILE%\.cache\whisper`.

## Comandos

```powershell
.\run.ps1                                        # arranca uvicorn en :8000 (con --reload)
env\Scripts\python.exe selftest.py               # verificación offline del pipeline (usa samples\jfk.flac)
env\Scripts\pip.exe freeze > requirements.txt    # úsalo al agregar dependencias
```

- PowerShell en este equipo usa `RemoteSigned` (CurrentUser): `.\run.ps1` funciona. Si en otra
  máquina falla con SecurityError de ExecutionPolicy, alterna a:
  `env\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
  (mismo comando que ejecuta `run.ps1`).

- Prueba rápida de la API (multipart): `requests.post('http://127.0.0.1:8000/api/transcribe', files={'file': open('x.wav','rb')}, data={'language':'auto'})`.
  Cuidado: PowerShell 5.1 **no tiene** `Invoke-RestMethod -Form`; usa curl.exe `-F` o Python.

## Arquitectura

- `app/main.py` — servidor FastAPI. El modelo se carga **una vez al importar**
  (`WhisperAnalyzer` global); el primer request tarda más. Endpoints: `GET /` (página),
  `POST /api/transcribe` (multipart `file`+`language`+`task`), `GET /api/model`.
- `app/whisper_service.py` — lógica pesada. El modelo lo fija la constante `MODEL_NAME`
  (`"base"`); es el único punto para probar otra arquitectura (`tiny`/`small`/`medium`),
  aunque el tiempo de CPU crece mucho con el tamaño.
  - transcripción oficial vía `model.transcribe(..., fp16=False)` (CPU).
  - pipeline encoder: `model.encoder(mel)` capturando cada `ResidualAttentionBlock`
    con **forward hooks** (salida `(1, 1500, 512)`).
  - pipeline decoder: bucle autoregresivo **réplica de `whisper/decoding.py`**
    (`install_kv_cache_hooks()` + `tokens[:, -1:]`, `SuppressBlank`, `SuppressTokens`, greedy T=0).
  - Los archivos subidos se guardan en `uploads/` y se **borran tras procesar**.
- `app/templates/index.html` + `app/static/` — UI sin dependencias externas (canvas).

## Gotchas verificados

- El pipeline encoder/decoder trabaja sobre la **ventana de 30 s** (`N_FRAMES=3000`,
  `mel` rellenada con `whisper.pad_or_trim`); la transcripción sí cubre el audio completo.
  `pad_or_trim` conserva los **primeros** 3000 frames: en audios > 30 s la visualización
  corresponde al primer tramo, no a otro chunk del audio.
- Tensores de forward hooks pueden ser **no contiguos**: usa `.reshape(-1)`, no `.view(-1)`.
- Secuencia inicial del decoder (sin timestamps): `[<|startoftranscript|>, <|lang|>,
  <|transcribe|>, <|notimestamps|>]`. Si `language=None`, `get_tokenizer` usa "en" como
  placeholder: hay que sobreescribir el token de idioma en la posición 1 con
  `model.detect_language(features, tokenizer)`.
- En `main.py` las vistas usan la firma nueva de `Jinja2Templates`:
  `TemplateResponse(request, "index.html", {...})` (`request` como primer argumento
  posicional); no uses la firma antigua `TemplateResponse("index.html", {"request": ...})`.
- Para separar texto de tokens especiales: filtrar `id < tokenizer.eot`, igual que
  `transcribe.py` (`new_segment`). `tokenizer.decode()` ya ignora timestamps.
- Rendimiento: CPU, ~4.5 s para 11 s de audio con `base`. Los análisis se serializan con
  un `Semaphore(1)`: no añadas concurrencia sin pensar en la CPU.

## Pruebas

Sin suite formal. `selftest.py` valida el pipeline offline (transcripción + mel + encoder
+ decoder) y necesita `samples/jfk.flac` (11 s). Ojo: `samples/` está en `.gitignore` y
**no viene en el checkout** (ni siquiera existe el directorio): descárgalo una vez antes
del primer run:

```powershell
curl.exe -L --create-dirs -o samples\jfk.flac https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac
```

El primer run de `selftest.py` también descarga el modelo a `%USERPROFILE%\.cache\whisper`.