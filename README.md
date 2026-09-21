# Whisper Encoder/Decoder Explorer

Página web (FastAPI) para subir un audio y **validar cómo el encoder y el decoder de
Whisper transcriben** la entrada: mel spectrograma, activaciones del encoder capa a capa
y generación del decoder token a token con su top-5 de probabilidades.

Python 3.12 · modelo Whisper `base` · PyTorch CPU.

## Requisitos del equipo (una vez)

- **Python 3.12** y **ffmpeg** (obligatorio para Whisper). Si no están instalados:

  ```powershell
  winget install --id Python.Python.3.12 --source winget
  winget install --id Gyan.FFmpeg --source winget
  ```

  > `--source winget` evita fallos del source `msstore` (error de certificado en
  > algunas máquinas). ffmpeg instalado por winget puede no estar en PATH de shells
  > ya abiertos: la app lo localiza sola (`app/whisper_service.py` busca en las
  > carpetas de WinGet al importar).

- **PowerShell ExecutionPolicy** (solo para usar `.\run.ps1`): si falla con
  *SecurityError*, habilita los scripts locales con:

  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```

## Instalación desde cero (después de clonar)

> ⚠️ `torch==2.14.0+cpu` **no existe en PyPI** (índice normal): hay que instalarlo
> desde el índice CPU de PyTorch **antes** de `requirements.txt`, o el pip fallará
> al resolver. Este equipo no tiene GPU NVIDIA; si lo tuviera, adapta el índice.

```powershell
# 1. Entorno virtual (usa el nombre `env`, está en .gitignore)
python -m venv env

# 2. PyTorch / torchaudio CPU (índice CPU, obligatorio ANTES del paso 3)
env\Scripts\pip.exe install "torch==2.14.0+cpu" "torchaudio==2.11.0+cpu" --index-url https://download.pytorch.org/whl/cpu

# 3. Resto de dependencias (torch ya instalado en el paso 2, pip lo omite)
env\Scripts\pip.exe install -r requirements.txt

# 4. Muestra para el selftest (`samples/` está en .gitignore y no viene con el repo)
curl.exe -L --create-dirs -o samples\jfk.flac https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac
```

> Tip: si tu conexión es lenta y pip aborta con *Read timed out*, agrega
> `--timeout 600 --retries 8` a los `pip install`.

## Verificación

```powershell
env\Scripts\python.exe selftest.py
```

Valida el pipeline offline (transcripción + mel + encoder + decoder) con
`samples/jfk.flac` (11 s). El primer run descarga el modelo `base` (~145 MB) a
`%USERPROFILE%\.cache\whisper`.

## Ejecutar

```powershell
.\run.ps1
```

El servidor queda en **http://127.0.0.1:8000** (API docs en
http://127.0.0.1:8000/docs). Equivalente sin scripts de PowerShell:

```powershell
env\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> El modelo se carga **una vez al arrancar** (el primer request o el primer análisis
> tarda más).

## Endpoints

| Método | Ruta              | Descripción                                       |
|--------|-------------------|---------------------------------------------------|
| GET    | `/`               | Página web (subida + visualización)               |
| POST   | `/api/transcribe` | multipart `file` + `language` (auto\|código) + `task` |
| GET    | `/api/model`      | Info del modelo cargado                           |

## Estructura

- `app/main.py` — servidor FastAPI (el modelo se carga una vez al importar).
- `app/whisper_service.py` — transcripción oficial + pipeline encoder/decoder.
- `app/templates/`, `app/static/` — interfaz (sin dependencias externas).
- `uploads/` — archivos temporales (se eliminan tras procesar).

## Notas técnicas

- El encoder/decoder se instrumenta con `torch` **forward hooks** (salidas de cada
  `ResidualAttentionBlock`) y un bucle decoder que replica `whisper/decoding.py`
  (`install_kv_cache_hooks()`, `SuppressBlank`, `SuppressTokens`, greedy T=0).
- El pipeline visualiza la ventana de **30 s** (`N_FRAMES = 3000`) aunque la
  transcripción se hace sobre el audio completo; en audios > 30 s se muestran los
  **primeros** 30 s.
- Los análisis se serializan con un semáforo para no saturar la CPU.