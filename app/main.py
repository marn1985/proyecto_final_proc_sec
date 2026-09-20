"""Servidor FastAPI: página de subida de audio + API de introspección Whisper.

Endpoints:
- GET  /                 -> página web (upload + visualización)
- POST /api/transcribe   -> multipart: file + language (auto|es|en|...) + task
- GET  /api/model        -> info del modelo cargado

El modelo se carga una sola vez al importar este módulo.
"""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from whisper.tokenizer import LANGUAGES

from . import whisper_service

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac",
    ".wma", ".webm", ".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg",
}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

# El análisis es intensivo en CPU: serializamos para no saturar el hilo.
_ANALYZE_LOCK = threading.Semaphore(1)

app = FastAPI(
    title="Whisper Encoder/Decoder Explorer",
    description="Sube un audio y valida cómo el encoder y el decoder de Whisper lo transcriben.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# carga única del modelo al arrancar (primera vez descarga ~145 MB)
analyzer = whisper_service.WhisperAnalyzer(
    model_name=whisper_service.MODEL_NAME, device=whisper_service._pick_device()
)

COMMON_LANGUAGES = [
    ("auto", "Auto (detectar)"),
    ("es", "Español"),
    ("en", "Inglés"),
    ("fr", "Francés"),
    ("de", "Alemán"),
    ("it", "Italiano"),
    ("pt", "Portugués"),
    ("nl", "Neerlandés"),
    ("ja", "Japonés"),
    ("zh", "Chino"),
    ("ko", "Coreano"),
    ("ru", "Ruso"),
]
LANGUAGE_NAMES = {code: name for code, name in LANGUAGES.items()}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "model_name": analyzer.model_name,
            "device": analyzer.device,
            "load_s": analyzer.load_s,
            "languages": COMMON_LANGUAGES,
            "arch": analyzer.model.dims,
            "all_languages": LANGUAGES,
        },
    )


@app.get("/api/model")
async def model_info():
    dims = analyzer.model.dims
    return {
        "model": analyzer.model_name,
        "device": analyzer.device,
        "load_s": analyzer.load_s,
        "arch": {
            "n_mels": dims.n_mels,
            "n_audio_ctx": dims.n_audio_ctx,
            "n_audio_state": dims.n_audio_state,
            "n_audio_layer": dims.n_audio_layer,
            "n_audio_head": dims.n_audio_head,
            "n_text_ctx": dims.n_text_ctx,
            "n_text_state": dims.n_text_state,
            "n_text_layer": dims.n_text_layer,
            "n_text_head": dims.n_text_head,
            "n_vocab": dims.n_vocab,
        },
    }


@app.post("/api/transcribe")
def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    task: str = Form("transcribe"),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400, f"Extensión no permitida: {ext}. Usa audio/video: {sorted(ALLOWED_EXTENSIONS)}"
        )
    if not ext:
        raise HTTPException(400, "El archivo no tiene extensión.")

    if language != "auto" and language not in LANGUAGES:
        raise HTTPException(400, f"Idioma desconocido: {language}")

    import shutil
    import uuid

    audio_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        with audio_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        size = audio_path.stat().st_size
        if size == 0:
            raise HTTPException(400, "El archivo está vacío.")
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(400, f"Archivo demasiado grande (máx 100 MB).")

        with _ANALYZE_LOCK:
            result = analyzer.analyze(
                str(audio_path),
                language=None if language == "auto" else language,
                task=task,
            )
        result["file"] = {"name": Path(file.filename or "audio").name, "size": size}
        return result
    finally:
        audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)