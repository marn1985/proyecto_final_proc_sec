"""Servicio de transcripción e introspección con Whisper (openai-whisper).

Expone dos cosas:
1. La transcripción "oficial" (``model.transcribe``) con segmentos y tiempos.
2. Un pipeline de análisis que permite *validar* cómo trabaja la arquitectura:
   - Encoder: mel spectrograma de entrada, para ver cómo se prepara el audio.
   - Decoder: generación token a token (greedy, sin timestamps) replicando el
     bucle de ``whisper.decoding.DecodingTask``, con el top-5 de cada paso y la
     matriz de cross-attention (alineación texto↔audio) de la última capa.

La API se replica del código fuente instalado de whisper (decoding.py / model.py):
- ``model.encoder(mel)`` -> (1, n_audio_ctx=1500, n_audio_state=512)
- ``model.decoder(tokens, audio_features, kv_cache)`` -> logits (1, n_tokens, vocab)
- ``model.install_kv_cache_hooks()`` + ``tokens[:, -1:]`` para generación autoregresiva
"""
from __future__ import annotations

import math
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import whisper
from whisper.tokenizer import get_tokenizer

MODEL_NAME = "base"  # tiny / base / small / medium / large (ver whisper.available_models())
N_ATTN_COLS = 150  # columnas (tiempo, downsampled) de la matriz de cross-attention


def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_ffmpeg() -> None:
    """Whisper necesita el binario `ffmpeg` en PATH.

    Instalado con winget (Gyan.FFmpeg) solo aparece en nuevos inicios de sesión;
    si no está en PATH, lo localizamos explícitamente y lo anteponemos.
    """
    if shutil.which("ffmpeg"):
        return
    base = Path(os.environ.get("LOCALAPPDATA", ""))
    candidates = [base / "Microsoft" / "WinGet" / "Links"]
    pkg = base / "Microsoft" / "WinGet" / "Packages"
    if pkg.exists():
        candidates.extend(pkg.glob("Gyan.FFmpeg_*/ffmpeg-*-full_build/bin"))
    for c in candidates:
        if c.exists() and (c / "ffmpeg.exe").exists():
            os.environ["PATH"] = str(c) + os.pathsep + os.environ.get("PATH", "")
            return


_ensure_ffmpeg()


class WhisperAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = _pick_device() if device is None else device
        t0 = time.time()
        self.model = whisper.load_model(model_name, device=self.device)
        self.load_s = round(time.time() - t0, 1)
        # máximo de tokens a generar: n_text_ctx // 2 (igual que DecodingTask)
        self.sample_len = self.model.dims.n_text_ctx // 2

    # ------------------------------------------------------------------ API
    def analyze(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> dict:
        """Analiza un archivo de audio y devuelve el JSON completo para la UI."""
        t0 = time.time()
        dims = self.model.dims
        audio = whisper.load_audio(audio_path)
        duration = len(audio) / whisper.audio.SAMPLE_RATE
        # el pipeline encoder/decoder se valida sobre la ventana de 30 s
        # (Whisper siempre procesa el audio en chunks de 30 s; N_FRAMES = 3000)
        window_s = min(duration, whisper.audio.CHUNK_LENGTH)

        # transcripción oficial (audio completo, con detección de segmentos)
        result = self.model.transcribe(audio, language=language, task=task, fp16=False)

        # mel de la ventana de análisis, rellenada a 30 s→3000 frames
        mel_full = whisper.log_mel_spectrogram(
            audio, dims.n_mels, padding=whisper.audio.N_SAMPLES
        ).to(self.device)
        mel = whisper.pad_or_trim(mel_full, whisper.audio.N_FRAMES)

        decoder = self._decoder_pipeline(mel, language, task)

        elapsed = time.time() - t0
        detected = result.get("language")
        return {
            "meta": {
                "model": self.model_name,
                "model_full": f"{self.model_name} (whisper {whisper.__version__})",
                "device": self.device,
                "load_s": self.load_s,
                "duration_s": round(duration, 2),
                "window_s": round(window_s, 2),
                "language": detected,
                "task": task,
                "elapsed_s": round(elapsed, 2),
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
            },
            "transcription": {
                "text": result["text"],
                "segments": [
                    {
                        "id": s["id"],
                        "start": round(s["start"], 2),
                        "end": round(s["end"], 2),
                        "text": s["text"],
                        "avg_logprob": round(float(s.get("avg_logprob", 0)), 4),
                        "no_speech_prob": round(float(s.get("no_speech_prob", 0)), 4),
                    }
                    for s in result["segments"]
                ],
            },
            "mel": self._mel_payload(mel, window_s),
            "decoder": decoder,
        }

    # ------------------------------------------------------------- decoder
    def _decoder_pipeline(self, mel: torch.Tensor, language, task: str) -> dict:
        """Genera token a token replicando DecodingTask (greedy, sin timestamps)."""
        tokenizer = get_tokenizer(
            self.model.is_multilingual,
            num_languages=self.model.num_languages,
            language=language or "en",  # "en" actúa de placeholder si se auto-detecta
            task=task,
        )
        with torch.no_grad():
            features = self.model.encoder(mel.unsqueeze(0))  # (1, 1500, 512)

        sot_sequence = list(tokenizer.sot_sequence_including_notimestamps)
        detected = None
        language_probs = None
        if language is None:
            lang_tokens, lang_probs = self.model.detect_language(features, tokenizer)
            sot_sequence[1] = int(lang_tokens[0])  # overwrite el token de idioma
            detected = max(lang_probs[0], key=lang_probs[0].get)
            language_probs = {
                k: round(v, 4)
                for k, v in sorted(lang_probs[0].items(), key=lambda kv: -kv[1])[:5]
            }

        steps = self._decode_steps(
            features, tokenizer, sot_sequence, sample_begin=len(sot_sequence)
        )
        return {
            "initial_tokens": [
                {"id": tid, "token": self._token_label(tid, tokenizer)}
                for tid in sot_sequence
            ],
            "detected_language": detected,
            "language_probs": language_probs,
            "steps": steps["steps"],
            "text": steps["text"],
            "avg_logprob": steps["avg_logprob"],
            "no_speech_prob": steps["no_speech_prob"],
            "cross_attention": steps["cross_attention"],
        }

    def _decode_steps(self, features, tokenizer, sot_sequence, sample_begin: int) -> dict:
        """Bucle autoregresivo con kv-cache, réplica fiel de DecodingTask._main_loop."""
        device = features.device
        tokens = torch.tensor([sot_sequence], device=device)
        cache, hooks = self.model.install_kv_cache_hooks()

        # Cross-attention (última capa del decoder) para la matriz de alineación
        # texto↔audio. La ruta rápida (SDPA) no expone los pesos de atención, solo
        # el resultado ya combinado con softmax+values: hay que desactivarla mientras
        # dura este bucle para que qkv_attention() nos devuelva el `qk` crudo.
        prev_sdpa = whisper.model.MultiHeadAttention.use_sdpa
        whisper.model.MultiHeadAttention.use_sdpa = False
        cross_attn_out: dict = {}
        attn_hook = self.model.decoder.blocks[-1].cross_attn.register_forward_hook(
            lambda _m, _i, out: cross_attn_out.__setitem__("qk", out[1])
        )
        hooks.append(attn_hook)

        suppress = list(tokenizer.non_speech_tokens) + [
            tokenizer.transcribe,
            tokenizer.translate,
            tokenizer.sot,
            tokenizer.sot_prev,
            tokenizer.sot_lm,
        ]
        if tokenizer.no_speech is not None:
            suppress.append(tokenizer.no_speech)
        suppress = sorted(set(suppress))

        steps: list = []
        attn_rows: list = []
        logprob_sum = 0.0
        n = 0
        no_speech_prob = None
        try:
            with torch.no_grad():
                for i in range(self.sample_len):
                    if i == 0:
                        logits_all = self.model.decoder(tokens, features, kv_cache=cache)
                        if tokenizer.no_speech is not None:
                            probs_at_sot = logits_all[0, 0].float().softmax(dim=-1)
                            no_speech_prob = round(
                                float(probs_at_sot[tokenizer.no_speech].detach()), 4
                            )
                        logits = logits_all[:, -1]
                    else:
                        # solo el último token: kv-cache reutiliza el contexto previo
                        logits = self.model.decoder(
                            tokens[:, -1:], features, kv_cache=cache
                        )[:, -1]

                    # fila de cross-attention que produjo el token de este paso: última
                    # posición de la consulta, promediada sobre las 8 cabezas y
                    # downsampled/normalizada 0..1 igual que el resto de heatmaps
                    qk = cross_attn_out["qk"][0, :, -1, :].float()  # (n_head, n_audio_ctx)
                    attn_weights = qk.softmax(dim=-1).mean(dim=0)  # (n_audio_ctx,)
                    attn_rows.append(self._bucket_1d(attn_weights, N_ATTN_COLS))

                    if tokens.shape[-1] == sample_begin:  # SuppressBlank
                        logits[:, tokenizer.encode(" ") + [tokenizer.eot]] = -np.inf
                    logits[:, suppress] = -np.inf  # SuppressTokens

                    probs = F.softmax(logits.float(), dim=-1)[0]
                    topk = torch.topk(probs, 5)
                    token_id = int(logits.argmax(dim=-1).item())

                    steps.append(
                        {
                            "step": i,
                            "token": self._token_label(token_id, tokenizer),
                            "id": token_id,
                            "prob": round(float(probs[token_id]), 4),
                            "top5": [
                                {
                                    "token": self._token_label(int(t), tokenizer),
                                    "id": int(t),
                                    "prob": round(float(p), 4),
                                }
                                for t, p in zip(topk.indices, topk.values)
                            ],
                        }
                    )
                    logprob_sum += float(torch.log(probs[token_id]))
                    n += 1
                    tokens = torch.cat(
                        [tokens, torch.tensor([[token_id]], device=device)], dim=-1
                    )
                    if token_id == tokenizer.eot:
                        break
        finally:
            for hook in hooks:
                hook.remove()
            whisper.model.MultiHeadAttention.use_sdpa = prev_sdpa

        text_ids = [t for t in tokens[0].tolist() if t < tokenizer.eot]
        return {
            "steps": steps,
            "text": tokenizer.decode(text_ids),
            "avg_logprob": round(logprob_sum / max(n, 1), 4),
            "no_speech_prob": no_speech_prob,
            "cross_attention": {"n_cols": N_ATTN_COLS, "rows": attn_rows},
        }

    # ------------------------------------------------------------- helpers
    def _mel_payload(self, mel: torch.Tensor, window_s: float) -> dict:
        """Grid del mel de la ventana visible (≈100 frames/s), normalizado 0..1."""
        frames = int(window_s * 100) + 1
        m = mel[:, :frames].float().cpu()
        n_cols = 220
        k = math.ceil(m.shape[-1] / n_cols)
        if k * n_cols != m.shape[-1]:
            m = F.pad(m, (0, k * n_cols - m.shape[-1]))
        grid = m.view(m.shape[0], n_cols, k).max(dim=2).values
        gmin, gmax = float(grid.min()), float(grid.max())
        if gmax - gmin > 1e-9:
            grid = (grid - gmin) / (gmax - gmin)
        return {"n_mels": m.shape[0], "n_cols": n_cols, "grid": grid.tolist()}

    @staticmethod
    def _bucket_1d(values: torch.Tensor, n_buckets: int) -> list:
        """Media de |activación| por bucket de tiempo, normalizado 0..1."""
        v = values.abs().reshape(-1)
        k = math.ceil(v.numel() / n_buckets)
        if k * n_buckets != v.numel():
            v = F.pad(v, (0, k * n_buckets - v.numel()))
        v = v.view(n_buckets, k).mean(dim=1)
        vmin, vmax = float(v.min()), float(v.max())
        if vmax - vmin > 1e-9:
            v = (v - vmin) / (vmax - vmin)
        return [round(float(x), 4) for x in v]

    @staticmethod
    def _token_label(tid: int, tokenizer) -> str:
        for name, sid in tokenizer.special_tokens.items():
            if sid == tid:
                return name
        if tid >= tokenizer.timestamp_begin:
            return tokenizer.decode_with_timestamps([tid])
        return tokenizer.decode([tid])