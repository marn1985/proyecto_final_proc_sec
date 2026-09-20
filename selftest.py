"""Prueba rápida del pipeline sin HTTP."""
import json
import sys

from app.whisper_service import WhisperAnalyzer

print("Cargando modelo…")
w = WhisperAnalyzer()
print(f"Modelo {w.model_name} en {w.device} (carga: {w.load_s}s)\n")

res = w.analyze("samples/jfk.flac")
print("=== Transcripción oficial ===")
print(res["transcription"]["text"])
print(f"segmentos: {len(res['transcription']['segments'])}")

print("\n=== Mel ===")
print(f"grid {len(res['mel']['grid'])}x{len(res['mel']['grid'][0])}")

print("\n=== Encoder ===")
for l in res["encoder"]["layers"][:3]:
    print(f"  capa {l['layer']}: stats={l['stats']} act[0..2]={l['activation'][:3]}")
print(f"  … total {len(res['encoder']['layers'])} filas (incl. ln_post)")

print("\n=== Decoder ===")
print("secuencia inicial:", [t["token"] for t in res["decoder"]["initial_tokens"]])
print("idioma detectado:", res["decoder"]["detected_language"])
print("log-prob media:", res["decoder"]["avg_logprob"], "| p(no_speech):", res["decoder"]["no_speech_prob"])
for s in res["decoder"]["steps"][:12]:
    top = ", ".join(f"{o['token']}({o['prob']})" for o in s["top5"][:3])
    print(f"  paso {s['step']:>2}: {s['token']!r:>24} p={s['prob']:.3f} | top5: {top}")
print("texto decoder:", res["decoder"]["text"])

print("\n=== Elapsed ===")
print(f"{res['meta']['elapsed_s']}s | idioma oficial: {res['meta']['language']}")