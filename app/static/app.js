/* Whisper Encoder/Decoder Explorer — lógica de la interfaz */
"use strict";

const $ = (sel) => document.querySelector(sel);

/* ---------- utilidades ---------- */
function esc(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const ss = (s % 60).toFixed(2).padStart(5, "0");
  return `${m}:${ss}`;
}

function fmtDur(s) {
  if (!isFinite(s) || s < 0) return "--:--";
  const m = Math.floor(s / 60);
  const ss = String(Math.floor(s % 60)).padStart(2, "0");
  return `${m}:${ss}`;
}

/* paleta tipo turbo/viridis */
const STOPS = [
  [0.03, 0.11, 0.30],
  [0.11, 0.46, 0.71],
  [0.22, 0.77, 0.76],
  [0.80, 0.93, 0.38],
  [0.96, 0.70, 0.18],
  [0.90, 0.30, 0.21],
];
function colorFor(t) {
  const x = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(x));
  const f = x - i;
  const a = STOPS[i], b = STOPS[i + 1];
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function drawHeatmap(canvas, rows, cols, getValue) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const cw = w / cols, ch = h / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      ctx.fillStyle = colorFor(getValue(r, c));
      ctx.fillRect(c * cw, r * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }
}

/* ---------- arquitectura (GET /api/model) ---------- */
async function loadArch() {
  try {
    const r = await fetch("/api/model");
    const data = await r.json();
    const a = data.arch;
    $("#arch-summary").innerHTML =
      `Arquitectura <b>base</b> — encoder: ${a.n_audio_layer} capas · ` +
      `${a.n_audio_head} cabezas · entrada mel (${a.n_mels}, 3000) → features ` +
      `(1500 × ${a.n_audio_state}) · decoder: ${a.n_text_layer} capas · ` +
      `${a.n_text_head} cabezas · vocabulario ${a.n_vocab} tokens · contexto ${a.n_text_ctx} tokens.`;
  } catch (e) { /* silencioso */ }
}

/* ---------- manejo del formulario ---------- */
let selectedFile = null;

let previewUrl = null;

function setFile(f) {
  selectedFile = f;
  $("#selected-file").textContent = `📎 ${f.name} (${(f.size / 1024 / 1024).toFixed(2)} MB)`;
  $("#submit-btn").disabled = false;

  // Reproductor: URL temporal del audio subido o grabado
  const player = $("#player"), meta = $("#audio-meta"), dl = $("#download-link");
  if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
  previewUrl = URL.createObjectURL(f);
  player.src = previewUrl;
  player.load();
  dl.href = previewUrl;
  dl.download = f.name || "audio.wav";
  dl.classList.remove("hidden");
  meta.textContent = "";
  player.onloadedmetadata = () => {
    meta.textContent = `Duración ${fmtDur(player.duration)} · Formato ${f.type || "desconocido"} · ${(f.size / 1024).toFixed(1)} KB`;
  };
  player.onerror = () => {
    meta.textContent = "⚠️ Este navegador no pudo reproducir el audio: codec/forma no soportado o archivo dañado.";
  };
  $("#audio-preview").classList.remove("hidden");
}

async function runAnalysis() {
  if (!selectedFile) return;
  const form = new FormData();
  form.append("file", selectedFile);
  form.append("language", $("#language").value);
  form.append("task", $("#task").value);

  $("#error").classList.add("hidden");
  $("#progress").classList.remove("hidden");
  $("#submit-btn").disabled = true;

  try {
    const res = await fetch("/api/transcribe", { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Error HTTP ${res.status}`);
    renderResults(data);
    $("#results").classList.remove("hidden");
    $("#results").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    const el = $("#error");
    el.textContent = String(err && err.message ? err.message : err);
    el.classList.remove("hidden");
  } finally {
    $("#progress").classList.add("hidden");
    $("#submit-btn").disabled = false;
  }
}

function wireUpload() {
  const dz = $("#dropzone"), fi = $("#file-input"), btn = $("#submit-btn");

  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag");
    if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });
  fi.addEventListener("change", () => { if (fi.files.length) setFile(fi.files[0]); });

  $("#upload-form").addEventListener("submit", (e) => {
    e.preventDefault();
    runAnalysis();
  });
}

/* ---------- grabación con micrófono (Web Audio → WAV, fallback MediaRecorder) ---------- */
function wireRecord() {
  const recBtn = $("#record-btn"), stopBtn = $("#stop-btn");
  const timer = $("#rec-timer"), errEl = $("#rec-err");
  const level = $("#rec-level"), levelFill = $("#rec-level-fill");
  let stream = null, startedAt = 0, timerId = null;
  let mode = null; // "webaudio" | "mediarecorder"
  let mediaRecorder = null, chunks = [];

  if (!navigator.mediaDevices?.getUserMedia) {
    recBtn.disabled = true;
    errEl.textContent = "Tu navegador no soporta grabación de micrófono (getUserMedia).";
    errEl.classList.remove("hidden");
    return;
  }

  function stopTimer() {
    if (timerId) { clearInterval(timerId); timerId = null; }
  }

  function tick() {
    const s = Math.floor((Date.now() - startedAt) / 1000);
    timer.textContent = `REC ${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  function showErr(msg) {
    errEl.textContent = msg;
    errEl.classList.remove("hidden");
  }

  function resetButtons() {
    stopTimer();
    timer.classList.add("hidden");
    recBtn.classList.remove("hidden");
    stopBtn.classList.add("hidden");
    level.classList.add("hidden");
    levelFill.style.width = "0%";
  }

  recBtn.addEventListener("click", async () => {
    errEl.classList.add("hidden");
    try {
      stream = await requestMic($("#mic-select").value);
    } catch (e) {
      showErr(
        "No se pudo acceder al micrófono (" + (e.name || e.message) + "). " +
        "Revisa el permiso del navegador y abre la página desde http://127.0.0.1:8000.");
      return;
    }

    if (!startWebAudio(stream) && !startMediaRecorder(stream)) {
      stream.getTracks().forEach((t) => t.stop());
      showErr("Este navegador no puede grabar audio (sin Web Audio ni MediaRecorder).");
      return;
    }

    startedAt = Date.now();
    timer.textContent = "REC 0:00";
    timer.classList.remove("hidden");
    recBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    timerId = setInterval(tick, 500);
  });

  stopBtn.addEventListener("click", () => {
    if (mode === "webaudio") {
      finishWebAudio();
    } else if (mode === "mediarecorder" && mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
  });

  /* ---- modo principal: Web Audio → WAV PCM (evita el bug de MediaRecorder+opus) ---- */
  let ctx = null, srcNode = null, procNode = null, frames = [];

  function startWebAudio(str) {
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx || !Ctx.prototype.createScriptProcessor) return false;
      ctx = new Ctx();
      srcNode = ctx.createMediaStreamSource(str);
      procNode = ctx.createScriptProcessor(4096, 1, 1);
      frames = [];
      procNode.onaudioprocess = (ev) => {
        const ch = ev.inputBuffer.getChannelData(0);
        if (ch.length) frames.push(new Float32Array(ch));
        let peak = 0;
        for (let i = 0; i < ch.length; i += 32) {
          const v = Math.abs(ch[i]);
          if (v > peak) peak = v;
        }
        levelFill.style.width = Math.round(Math.min(1, peak) * 100) + "%";
      };
      const mute = ctx.createGain();
      mute.gain.value = 0; // evitar eco del micrófono por el monitor
      srcNode.connect(procNode);
      procNode.connect(mute);
      mute.connect(ctx.destination);
      if (ctx.state === "suspended") ctx.resume();
      mode = "webaudio";
      level.classList.remove("hidden");
      return true;
    } catch (e) {
      try { srcNode?.disconnect(); procNode?.disconnect(); ctx?.close(); } catch (_) { /* noop */ }
      return false;
    }
  }

  function finishWebAudio() {
    try { srcNode.disconnect(); procNode.disconnect(); } catch (_) { /* noop */ }
    const rate = ctx ? (ctx.sampleRate || 48000) : 48000;
    try { ctx.close(); } catch (_) { /* noop */ }
    const count = frames.reduce((n, f) => n + f.length, 0);
    const raw = new Float32Array(count);
    let at = 0;
    for (const f of frames) { raw.set(f, at); at += f.length; }
    frames = [];
    stream.getTracks().forEach((t) => t.stop());
    resetButtons();
    if (raw.length < 320) { // menos de ~20 ms a 16 kHz
      showErr("La grabación quedó vacía (muy corta). Intenta grabando al menos 1 segundo.");
      return;
    }
    const mono = resampleLinear(raw, rate, 16000);
    const blob = encodeWav(mono, 16000);
    setFile(new File([blob], `mic_${Date.now()}.wav`, { type: "audio/wav" }));
    populateMicSelect();
    runAnalysis();
  }

  /* ---- modo respaldo: MediaRecorder (solo si Web Audio no está disponible) ---- */
  function startMediaRecorder(str) {
    try {
      if (!window.MediaRecorder) return false;
      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")
          ? "audio/ogg;codecs=opus"
          : "";
      mediaRecorder = new MediaRecorder(str, mime ? { mimeType: mime } : undefined);
      chunks = [];
      mediaRecorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      mediaRecorder.onstop = () => {
        const type = mediaRecorder.mimeType || "audio/webm";
        const ext = type.includes("ogg") ? "ogg" : "webm";
        const blob = new Blob(chunks, { type });
        stream.getTracks().forEach((t) => t.stop());
        resetButtons();
        setFile(new File([blob], `mic_${Date.now()}.${ext}`, { type }));
        populateMicSelect();
        runAnalysis();
      };
      mediaRecorder.onerror = () => showErr("Error durante la grabación de audio.");
      mediaRecorder.start();
      mode = "mediarecorder";
      level.classList.add("hidden");
      return true;
    } catch (e) {
      return false;
    }
  }
}

/* ---------- utilidades de audio (codificación WAV PCM) ---------- */
function resampleLinear(input, inRate, outRate) {
  if (inRate === outRate) return input;
  const outLen = Math.max(1, Math.floor(input.length * outRate / inRate));
  const out = new Float32Array(outLen);
  const ratio = inRate / outRate;
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const f = pos - i0;
    out[i] = input[i0] * (1 - f) + input[i1] * f;
  }
  return out;
}

function encodeWav(samples, sampleRate) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const dv = new DataView(buf);
  const wstr = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
  wstr(0, "RIFF"); dv.setUint32(4, 36 + n * 2, true); wstr(8, "WAVE");
  wstr(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sampleRate, true); dv.setUint32(28, sampleRate * 2, true);
  dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
  wstr(36, "data"); dv.setUint32(40, n * 2, true);
  let off = 44;
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    dv.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    off += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}

/* ---------- captura robusta de micrófono ---------- */
// Lista de configuraciones de captura: cubre bugs conocidos de Chromium/Windows
// (AEC que silencia, formatos mono/estéreo, sample rates).
const MIC_CONFIGS = [
  { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  { channelCount: 1, sampleRate: 16000 },
  { channelCount: 2, sampleRate: 48000 },
];

async function requestMic(deviceId) {
  const combos = [];
  for (const cfg of MIC_CONFIGS) combos.push({ cfg, deviceId });
  if (deviceId) for (const cfg of MIC_CONFIGS) combos.push({ cfg, deviceId: "" });
  let lastErr = null;
  for (const { cfg, deviceId: did } of combos) {
    try {
      const audio = { ...cfg };
      if (did) audio.deviceId = { exact: did };
      return await navigator.mediaDevices.getUserMedia({ audio });
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr || new Error("No se pudo abrir el micrófono");
}

function populateMicSelect() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  navigator.mediaDevices.enumerateDevices()
    .then((devs) => {
      const mics = devs.filter((d) => d.kind === "audioinput");
      const sel = $("#mic-select"), wrap = $("#mic-select-wrap");
      if (!sel || !wrap) return;
      if (mics.length <= 1) { wrap.classList.add("hidden"); return; }
      const cur = sel.value;
      sel.innerHTML =
        '<option value="">Predeterminado del sistema</option>' +
        mics.map((m, i) =>
          `<option value="${esc(m.deviceId)}">${esc(m.label || "Micrófono " + (i + 1))}</option>`).join("");
      if (cur) sel.value = cur;
      wrap.classList.remove("hidden");
    })
    .catch(() => { /* sin dispositivos o sin permiso aun */ });
}

/* ---------- render de resultados ---------- */
function renderResults(d) {
  const meta = d.meta;
  const langs = { es: "Español", en: "Inglés", fr: "Francés", de: "Alemán", it: "Italiano",
                  pt: "Portugués", ja: "Japonés", zh: "Chino", ko: "Coreano", ru: "Ruso" };

  $("#meta").innerHTML = `
    <div class="meta-grid">
      <div><b>Audio</b>${esc(d.file?.name || "–")} (${meta.duration_s} s)</div>
      <div><b>Idioma</b>${esc(langs[meta.language] || meta.language || "auto")}</div>
      <div><b>Modelo / device</b>${esc(meta.model)} · ${esc(meta.device)}</div>
      <div><b>Tiempo de análisis</b>${meta.elapsed_s} s</div>
      <div><b>Ventana del pipeline</b>${meta.window_s} s (máx 30 s)</div>
    </div>`;

  if (d.decoder.language_probs) {
    const top = Object.entries(d.decoder.language_probs);
    $("#language-probs").innerHTML =
      '<div class="lang-probs">' +
      top.map(([code, p]) => `
        <div class="bar">
          <span class="bar-label">${esc(langs[code] || code)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
          <span>${(p * 100).toFixed(1)}%</span>
        </div>`).join("") +
      "</div>";
  } else {
    $("#language-probs").innerHTML = "";
  }

  $("#transcript-text").textContent = d.transcription.text;

  $("#segments").innerHTML = d.transcription.segments.map((s) => `
    <li><span class="time">${fmtTime(s.start)} → ${fmtTime(s.end)}</span><span>${esc(s.text)}</span></li>`).join("")
    || "<li>Sin segmentos.</li>";

  drawMel(d.mel);
  drawEncoder(d.encoder);
  drawDecoder(d.decoder);
}

function drawMel(mel) {
  const canvas = $("#mel-canvas");
  const grid = mel.grid;
  drawHeatmap(canvas, mel.n_mels, mel.n_cols, (r, c) => grid[r][c]);
}

function drawEncoder(enc) {
  const canvas = $("#encoder-canvas");
  const rows = enc.layers.map((l) => l.activation);
  const nRows = rows.length, nCols = rows[0].length;
  drawHeatmap(canvas, nRows, nCols, (r, c) => rows[r][c]);

  $("#encoder-stats").innerHTML = `
    <table>
      <thead><tr><th>Capa</th><th>mean</th><th>std</th><th>max |x|</th></tr></thead>
      <tbody>
        ${enc.layers.map((l) => `
          <tr><td>${l.layer === "out" ? "ln_post (salida)" : `bloque ${l.layer}`}</td>
              <td>${l.stats.mean}</td><td>${l.stats.std}</td><td>${l.stats.max_abs}</td></tr>`).join("")}
      </tbody>
    </table>`;
}

function drawDecoder(dec) {
  // detalles
  const chips = dec.initial_tokens.map((t) =>
    `<span class="chip ctrl">${esc(t.token)}</span>`).join("");
  const langInfo = dec.detected_language
    ? ` · idioma detectado: <b>${esc(dec.detected_language)}</b>`
    : "";
  $("#decoder-details").innerHTML = `
    <div class="decoder-details">
      <div class="chip ctrl">secuencia inicial: ${chips}</div>
    </div>
    <p class="stage-note">
      Tokens generados: <b>${dec.steps.length}</b>${langInfo} ·
      log-prob media: <b>${dec.avg_logprob}</b> ·
      p(no-speech): <b>${dec.no_speech_prob ?? "–"}</b>
    </p>
    <p id="decoder-text" class="transcript" style="font-size:1rem"></p>`;

  $("#decoder-text").textContent = dec.text || "—";

  $("#steps-body").innerHTML = dec.steps.map((s) => {
    const opts = s.top5.map((o) => `
      <div class="opt" style="min-width:90px">
        <span title="id ${o.id}">${esc(o.token)}</span>
        <span style="float:right"><b>${(o.prob * 100).toFixed(1)}%</b></span>
        <div style="height:4px;background:var(--border);border-radius:2px;margin-top:3px">
          <div style="height:4px;width:${(o.prob * 100).toFixed(1)}%;background:var(--accent);border-radius:2px"></div>
        </div>
      </div>`).join("");
    return `<tr>
      <td>${s.step}</td>
      <td><span class="chip ${s.id >= 50257 ? "ctrl" : "gen"}">${esc(s.token)}</span></td>
      <td>${(s.prob * 100).toFixed(1)}%</td>
      <td class="wide"><div class="opts">${opts}</div></td>
    </tr>`;
  }).join("");
}

/* ---------- init ---------- */
loadArch();
wireUpload();
wireRecord();
populateMicSelect();