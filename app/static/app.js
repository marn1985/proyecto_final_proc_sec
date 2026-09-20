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

  function setFile(f) {
    selectedFile = f;
    $("#selected-file").textContent = `📎 ${f.name} (${(f.size / 1024 / 1024).toFixed(2)} MB)`;
    btn.disabled = false;
  }

  $("#upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
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
  });
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