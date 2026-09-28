# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FastAPI web app that uploads an audio file and visualizes how Whisper's encoder/decoder
process it: mel spectrogram, token-by-token decoder generation with top-5 probabilities,
and the decoder's cross-attention alignment matrix (text↔audio). Python 3.12 · Whisper
`base` · PyTorch CPU.

## Environment

- All work happens in the virtualenv `env` (do not create `venv` or another name).
- Windows: always use `env\Scripts\python.exe` / `env\Scripts\pip.exe`.
- Pinned deps are in `requirements.txt`. Do **not** reinstall torch from the normal PyPI
  index — this machine has no NVIDIA GPU, and torch must come from the CPU index:
  `env\Scripts\pip.exe install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`
  (already installed this way: `torch==2.14.0+cpu`).
- ffmpeg is required (used by `whisper.load_audio` via PATH). When installed via winget
  (Gyan.FFmpeg) it does **not** appear in PATH for already-open shells; `app/whisper_service.py`
  locates it itself on import (`_ensure_ffmpeg()`, searches WinGet package folders). Don't
  assume `ffmpeg` is on PATH in a fresh shell without that fallback.
- The Whisper model downloads once (~145 MB) to `%USERPROFILE%\.cache\whisper`.

## Commands

```powershell
.\run.ps1                                        # start uvicorn on :8000 (with --reload)
env\Scripts\python.exe selftest.py               # offline pipeline check (needs samples\jfk.flac)
env\Scripts\pip.exe freeze > requirements.txt    # run after adding a dependency
```

- Equivalent to `run.ps1` without PowerShell scripts:
  `env\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
- If `.\run.ps1` fails with a PowerShell *SecurityError* (ExecutionPolicy), enable local
  scripts once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- Quick API smoke test (multipart): PowerShell 5.1's `Invoke-RestMethod` has no `-Form`;
  use `curl.exe -F` or Python's `requests.post(..., files=..., data=...)`.
- No formal test suite. `selftest.py` exercises the full pipeline (transcription + mel +
  encoder + decoder) against `samples/jfk.flac` (11 s). `samples/` is gitignored and does
  **not** ship with the checkout — fetch it once before the first run:
  ```powershell
  curl.exe -L --create-dirs -o samples\jfk.flac https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac
  ```

## Architecture

- `app/main.py` — FastAPI server. The model loads **once at import time** into a global
  `WhisperAnalyzer` (so the first request after startup is slow, not every request).
  Endpoints: `GET /` (upload/visualization page), `POST /api/transcribe` (multipart
  `file` + `language` + `task`), `GET /api/model`. Uploaded files are written to `uploads/`
  and deleted after processing (`finally: audio_path.unlink(...)`).
- `app/whisper_service.py` — all the heavy logic. `MODEL_NAME` (`"base"`) is the single
  place to swap architecture size (`tiny`/`small`/`medium`); CPU time grows a lot with size.
  - Official transcription via `model.transcribe(..., fp16=False)` (CPU).
  - Decoder pipeline: an autoregressive loop that is a faithful replica of
    `whisper/decoding.py`'s `DecodingTask._main_loop` — `model.install_kv_cache_hooks()`,
    feeding only `tokens[:, -1:]` after the first step, `SuppressBlank`, `SuppressTokens`,
    greedy decoding (T=0).
  - Cross-attention capture: for each decode step, a forward hook on
    `model.decoder.blocks[-1].cross_attn` (last decoder layer) grabs the raw `qk` scores.
    This requires `whisper.model.MultiHeadAttention.use_sdpa = False` for the duration of
    the loop — PyTorch's fused SDPA path never returns attention weights (`qk` comes back
    `None`), only the older manual path does. The flag is a **global class attribute**;
    it's restored in `_decode_steps`'s `finally` block, and only toggled around the custom
    loop (not around `model.transcribe()`, which stays on the fast SDPA path).
  - Analysis is serialized behind `threading.Semaphore(1)` (`_ANALYZE_LOCK` in `main.py`)
    to avoid saturating the CPU — don't add concurrency here without accounting for that.
- `app/templates/index.html` + `app/static/` — dependency-free UI (canvas-based
  visualizations, plus Web Audio API microphone recording encoded client-side to WAV
  mono 16 kHz).

### Verified gotchas (read before touching the pipeline)

- The encoder/decoder pipeline only analyzes the **first 30 s window** (`N_FRAMES = 3000`,
  via `whisper.pad_or_trim`, which keeps the **first** 3000 frames); the official
  transcription in `result["text"]`/`segments` does cover the full audio. For audio > 30 s,
  the encoder/decoder visualization corresponds to the first chunk only, not an arbitrary one.
- Forward-hook output tensors can be **non-contiguous** — use `.reshape(-1)`, never `.view(-1)`.
- Cross-attention heatmap rows are normalized **per row** (`_bucket_1d`, reused from the
  mel/activation bucketing helper), not globally — each generated token's attention gets
  its own 0..1 scale so the alignment diagonal stays visible regardless of how peaked or
  spread out any single token's attention is.
- Initial decoder sequence (no timestamps): `[<|startoftranscript|>, <|lang|>, <|transcribe|>,
  <|notimestamps|>]`. When `language=None` (auto-detect), `get_tokenizer` is called with
  `"en"` as a placeholder — the language token at position 1 must then be overwritten using
  `model.detect_language(features, tokenizer)`.
- `app/main.py` uses the new `Jinja2Templates` call signature:
  `TemplateResponse(request, "index.html", {...})` with `request` as the first positional
  arg — not the old `TemplateResponse("index.html", {"request": ...})` form.
- To separate transcribed text from special tokens, filter `id < tokenizer.eot` (matches
  `whisper/transcribe.py`'s `new_segment`); `tokenizer.decode()` already skips timestamps.
- Rough CPU perf: ~4.5 s to process 11 s of audio with `base`.

## Git

Origin: `https://github.com/marn1985/proyecto_final_proc_sec.git`, branch `main`. Commit
message convention: `tipo: mensaje` (`docs:`, `feat:`, `fix:`, `refactor:`, `chore:`, `test:`).
See [GIT-CHEATSHEET.md](GIT-CHEATSHEET.md) for the full command reference (this is a
learning-oriented cheatsheet, not project-specific policy).
