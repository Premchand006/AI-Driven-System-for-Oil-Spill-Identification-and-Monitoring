/* ==================================================================
   AI SpillGuard Pro — client-side inference engine
   Runs the U-Net + ResNet34 model fully in the browser via
   ONNX Runtime Web. No backend; deployable as a Vercel static site.
   ================================================================== */

"use strict";

// ---------- Configuration ----------
const MODEL_URL = "model/spillguard.int8.onnx";
const IMG_SIZE = 256;
const NUM_CLASSES = 4;
const MAX_DISPLAY = 1024; // cap very large uploads for canvas performance

const CLASS_NAMES = ["Background", "Oil Spill", "Look-alike", "Ship/Wake"];
const CLASS_COLORS = [
  [0, 0, 0],       // Background
  [255, 0, 124],   // Oil Spill
  [255, 204, 51],  // Look-alike
  [51, 221, 255],  // Ship/Wake
];
const CHART_COLORS = ["#FF007C", "#FFCC33", "#33DDFF", "#2b2b2b"];

const HISTORY_KEY = "spillguard_history";

// ---------- App state ----------
let session = null;
let modelLoading = null;
let current = null; // { fileName, origData (ImageData), w, h, maskOrig, maskModel, stats }
let pieChart = null;

const $ = (id) => document.getElementById(id);

// ---------- ONNX Runtime setup ----------
if (window.ort) {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  ort.env.wasm.numThreads = 1; // single-threaded (no cross-origin isolation needed)
}

async function loadModel() {
  if (session) return session;
  if (modelLoading) return modelLoading;

  modelLoading = (async () => {
    $("modelProgress").style.display = "block";
    const resp = await fetch(MODEL_URL);
    if (!resp.ok) throw new Error(`Failed to fetch model (${resp.status})`);

    const total = Number(resp.headers.get("Content-Length")) || 0;
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (total) {
        const pct = Math.round((received / total) * 100);
        $("progressBar").style.width = pct + "%";
        $("progressText").textContent = `Downloading model weights… ${pct}% (${(received / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB)`;
      } else {
        $("progressText").textContent = `Downloading model weights… ${(received / 1e6).toFixed(1)} MB`;
      }
    }
    const bytes = new Uint8Array(received);
    let pos = 0;
    for (const c of chunks) { bytes.set(c, pos); pos += c.length; }

    $("progressText").textContent = "Initializing inference session…";
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    $("modelProgress").style.display = "none";
    setDeviceStatus(true);
    return session;
  })();

  return modelLoading;
}

function setDeviceStatus(ready) {
  const el = $("deviceStatus");
  if (ready) {
    el.textContent = "✅ Engine: ONNX Runtime Web (WASM) — ready";
  }
}

// ---------- Image loading ----------
function fileToImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not decode image"));
    img.src = URL.createObjectURL(file);
  });
}

// Draw an Image to a canvas at (possibly capped) native size, return ImageData.
function imageToData(img) {
  let w = img.naturalWidth, h = img.naturalHeight;
  const longSide = Math.max(w, h);
  if (longSide > MAX_DISPLAY) {
    const s = MAX_DISPLAY / longSide;
    w = Math.round(w * s);
    h = Math.round(h * s);
  }
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

// Build the model input tensor: resize RGBA ImageData -> 256x256 -> NCHW float32 [0,1].
function preprocess(imageData) {
  const c = document.createElement("canvas");
  c.width = IMG_SIZE; c.height = IMG_SIZE;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  // Put source onto a temp canvas, then draw scaled into the 256 canvas (bilinear).
  const src = document.createElement("canvas");
  src.width = imageData.width; src.height = imageData.height;
  src.getContext("2d").putImageData(imageData, 0, 0);
  ctx.drawImage(src, 0, 0, IMG_SIZE, IMG_SIZE);
  const resized = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;

  const plane = IMG_SIZE * IMG_SIZE;
  const data = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    data[i] = resized[i * 4] / 255;             // R
    data[plane + i] = resized[i * 4 + 1] / 255; // G
    data[2 * plane + i] = resized[i * 4 + 2] / 255; // B
  }
  return new ort.Tensor("float32", data, [1, 3, IMG_SIZE, IMG_SIZE]);
}

// argmax over channel dim of logits [1,4,256,256] -> Uint8Array(256*256)
function argmaxMask(logits) {
  const plane = IMG_SIZE * IMG_SIZE;
  const out = new Uint8Array(plane);
  for (let i = 0; i < plane; i++) {
    let best = 0, bestVal = logits[i];
    for (let c = 1; c < NUM_CLASSES; c++) {
      const v = logits[c * plane + i];
      if (v > bestVal) { bestVal = v; best = c; }
    }
    out[i] = best;
  }
  return out;
}

// Nearest-neighbour resize of a class mask to (w,h).
function resizeMaskNN(mask, srcW, srcH, dstW, dstH) {
  if (srcW === dstW && srcH === dstH) return mask;
  const out = new Uint8Array(dstW * dstH);
  for (let y = 0; y < dstH; y++) {
    const sy = Math.min(srcH - 1, Math.floor((y * srcH) / dstH));
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(srcW - 1, Math.floor((x * srcW) / dstW));
      out[y * dstW + x] = mask[sy * srcW + sx];
    }
  }
  return out;
}

function computeStatistics(mask) {
  const total = mask.length;
  const counts = new Array(NUM_CLASSES).fill(0);
  for (let i = 0; i < total; i++) counts[mask[i]]++;
  const stats = {};
  for (let c = 0; c < NUM_CLASSES; c++) {
    stats[CLASS_NAMES[c]] = { count: counts[c], percentage: (counts[c] / total) * 100 };
  }
  return stats;
}

// ---------- Rendering ----------
function putMaskRGB(canvas, mask, w, h) {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const col = CLASS_COLORS[mask[i]];
    out[i * 4] = col[0]; out[i * 4 + 1] = col[1]; out[i * 4 + 2] = col[2]; out[i * 4 + 3] = 255;
  }
  drawImageData(canvas, new ImageData(out, w, h));
}

function putOriginal(canvas, imageData) {
  drawImageData(canvas, imageData);
}

function buildOverlay(imageData, mask, w, h, alpha, enabled, drawContours) {
  const src = imageData.data;
  const out = new Uint8ClampedArray(src.length);
  out.set(src);
  const enabledNonBg = enabled.filter((c) => c !== 0);
  const isEnabled = new Array(NUM_CLASSES).fill(false);
  enabledNonBg.forEach((c) => (isEnabled[c] = true));

  // Alpha blend
  for (let i = 0; i < w * h; i++) {
    const cls = mask[i];
    if (cls !== 0 && isEnabled[cls]) {
      const col = CLASS_COLORS[cls];
      const j = i * 4;
      out[j] = alpha * col[0] + (1 - alpha) * src[j];
      out[j + 1] = alpha * col[1] + (1 - alpha) * src[j + 1];
      out[j + 2] = alpha * col[2] + (1 - alpha) * src[j + 2];
    }
  }

  // White contours: pixel on an enabled class whose neighbour differs.
  if (drawContours && enabledNonBg.length) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const cls = mask[i];
        if (cls === 0 || !isEnabled[cls]) continue;
        const up = y > 0 ? mask[i - w] : cls;
        const dn = y < h - 1 ? mask[i + w] : cls;
        const lf = x > 0 ? mask[i - 1] : cls;
        const rt = x < w - 1 ? mask[i + 1] : cls;
        if (up !== cls || dn !== cls || lf !== cls || rt !== cls) {
          const j = i * 4;
          out[j] = 255; out[j + 1] = 255; out[j + 2] = 255;
        }
      }
    }
  }
  return new ImageData(out, w, h);
}

function drawImageData(canvas, imageData) {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext("2d").putImageData(imageData, 0, 0);
}

function canvasToBlob(canvas) {
  return new Promise((res) => canvas.toBlob(res, "image/png"));
}

// ---------- Settings (sidebar) ----------
function getSettings() {
  const enabled = [];
  for (let c = 0; c < NUM_CLASSES; c++) if ($("class-" + c).checked) enabled.push(c);
  return {
    alpha: parseFloat($("alpha").value),
    drawContours: $("drawContours").checked,
    enabled,
    autoSave: $("autoSave").checked,
    thresholds: {
      "Oil Spill": parseFloat($("th-oil").value),
      "Look-alike": parseFloat($("th-look").value),
      "Ship/Wake": parseFloat($("th-ship").value),
    },
  };
}

// ---------- Alerts ----------
function checkAlerts(stats, thresholds) {
  const alerts = [];
  const pct = stats["Oil Spill"].percentage;
  const th = thresholds["Oil Spill"];
  if (pct > th) {
    const severity = pct > th * 2 ? "CRITICAL" : "WARNING";
    alerts.push({
      class: "Oil Spill",
      percentage: pct,
      threshold: th,
      severity,
      priority: 1,
      message: `🛢️ Oil spill detected at ${pct.toFixed(2)}% coverage (Threshold: ${th}%)`,
    });
  }
  return alerts;
}

function renderAlerts(alerts) {
  const el = $("alertArea");
  if (alerts.length) {
    el.innerHTML =
      '<h3 class="section-title">🚨 ALERTS TRIGGERED</h3>' +
      alerts
        .map((a) => {
          const cls = a.severity === "CRITICAL" ? "alert-critical" : "alert-warning";
          const icon = a.severity === "CRITICAL" ? "🚨" : "⚠️";
          return `<div class="alert-card ${cls}"><strong>${icon} ${a.severity}</strong>: ${a.message}</div>`;
        })
        .join("");
  } else {
    el.innerHTML =
      '<div class="alert-card alert-success"><strong>✅ ALL CLEAR</strong>: No critical alerts detected. All parameters within normal range.</div>';
  }
}

// ---------- Main detection flow ----------
async function runDetection(file) {
  try {
    await loadModel();
  } catch (e) {
    showToast("❌ " + e.message, true);
    return;
  }

  let imageData;
  try {
    const img = await fileToImage(file);
    imageData = imageToData(img);
  } catch (e) {
    showToast("❌ Failed to load image: " + e.message, true);
    return;
  }

  showToast("🔄 Running AI detection…");
  let maskModel;
  try {
    const input = preprocess(imageData);
    const inputName = session.inputNames[0];
    const feeds = {}; feeds[inputName] = input;
    const results = await session.run(feeds);
    const logits = results[session.outputNames[0]].data;
    maskModel = argmaxMask(logits);
  } catch (e) {
    showToast("❌ Inference failed: " + e.message, true);
    return;
  }

  const w = imageData.width, h = imageData.height;
  const maskOrig = resizeMaskNN(maskModel, IMG_SIZE, IMG_SIZE, w, h);
  const stats = computeStatistics(maskModel); // stats on model-size mask (matches Streamlit app)

  current = { fileName: file.name, origData: imageData, w, h, maskOrig, maskModel, stats };

  $("emptyState").style.display = "none";
  $("results").style.display = "block";

  renderCurrent();

  // Auto-save
  const s = getSettings();
  if (s.autoSave) {
    saveToHistory(file.name, stats, checkAlerts(stats, s.thresholds));
    showToast("✅ Results saved to history");
  } else {
    showToast("✅ Detection complete");
  }
}

// Re-render everything from `current` using live sidebar settings.
function renderCurrent() {
  if (!current) return;
  const s = getSettings();
  const { origData, maskOrig, maskModel, stats, w, h, fileName } = current;

  putOriginal($("canvasOriginal"), origData);
  $("capOriginal").textContent = `Original Image (${w}×${h})`;

  putMaskRGB($("canvasMask"), maskOrig, w, h);

  const overlay = buildOverlay(origData, maskOrig, w, h, s.alpha, s.enabled, s.drawContours);
  drawImageData($("canvasOverlay"), overlay);
  $("capOverlay").textContent = `Detection Overlay (α=${s.alpha.toFixed(2)})`;

  const alerts = checkAlerts(stats, s.thresholds);
  renderAlerts(alerts);
  renderChart(stats);
  renderMetrics(stats, alerts);
}

function renderMetrics(stats, alerts) {
  const oil = stats["Oil Spill"];
  $("oilPct").textContent = oil.percentage.toFixed(2) + "%";
  $("oilPx").textContent = oil.count.toLocaleString() + " px";

  const box = $("statusBox");
  const txt = $("statusText");
  if (alerts.some((a) => a.class === "Oil Spill")) {
    const critical = alerts.some((a) => a.severity === "CRITICAL");
    const color = critical ? "#cc0000" : "#ff8800";
    box.style.borderLeftColor = color;
    txt.style.color = color;
    txt.textContent = `🚨 OIL SPILL ${critical ? "CRITICAL" : "WARNING"}`;
  } else {
    box.style.borderLeftColor = "green";
    txt.style.color = "green";
    txt.textContent = "✅ SAFE";
  }
}

function renderChart(stats) {
  const labels = ["Oil Spill", "Look-alike", "Ship/Wake", "Background"];
  const values = labels.map((l) => stats[l].percentage);
  const ctx = $("pieChart").getContext("2d");
  if (pieChart) pieChart.destroy();
  pieChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: CHART_COLORS, borderColor: "#fff", borderWidth: 2 }],
    },
    options: {
      cutout: "40%",
      plugins: {
        legend: { position: "bottom" },
        tooltip: {
          callbacks: { label: (c) => `${c.label}: ${c.parsed.toFixed(2)}%` },
        },
      },
    },
  });
}

// ---------- Downloads ----------
async function downloadOverlay() {
  if (!current) return;
  const s = getSettings();
  const { origData, maskOrig, w, h, fileName } = current;
  const overlay = buildOverlay(origData, maskOrig, w, h, s.alpha, s.enabled, s.drawContours);
  const c = document.createElement("canvas");
  drawImageData(c, overlay);
  triggerDownload(await canvasToBlob(c), baseName(fileName) + "_overlay.png");
}

async function downloadMask() {
  if (!current) return;
  const { maskOrig, w, h, fileName } = current;
  const c = document.createElement("canvas");
  putMaskRGB(c, maskOrig, w, h);
  triggerDownload(await canvasToBlob(c), baseName(fileName) + "_mask.png");
}

function downloadReport() {
  if (!current) return;
  const s = getSettings();
  const alerts = checkAlerts(current.stats, s.thresholds);
  const report = {
    timestamp: new Date().toISOString(),
    image: current.fileName,
    statistics: current.stats,
    alerts,
    configuration: {
      model: "U-Net ResNet34 (ONNX int8)",
      engine: "ONNX Runtime Web (WASM)",
      alpha: s.alpha,
      enabled_classes: s.enabled,
    },
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  triggerDownload(blob, baseName(current.fileName) + "_report.json");
}

function baseName(name) { return name.replace(/\.[^/.]+$/, ""); }

function triggerDownload(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------- History (localStorage) ----------
function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch { return []; }
}

function saveToHistory(imageName, stats, alerts) {
  const history = loadHistory();
  const now = new Date();
  history.push({
    timestamp: now.toISOString().slice(0, 19).replace(/[-:T]/g, ""),
    datetime: now.toISOString(),
    image_name: imageName,
    statistics: stats,
    alerts,
    alert_count: alerts.length,
    has_critical: alerts.some((a) => a.severity === "CRITICAL"),
  });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  if ($("tab-history").classList.contains("active")) renderHistory();
}

function clearHistory() {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
  showToast("✅ History cleared");
}

function renderHistory() {
  const history = loadHistory();
  const el = $("historyContent");
  if (!history.length) {
    el.innerHTML =
      '<div class="info-box">📭 No detection history available yet</div>' +
      "<p>Detection history will appear here once you start processing images. Each detection is saved with timestamp, statistics, and alert records.</p>";
    return;
  }

  const rows = history
    .map((r) => {
      const st = r.statistics;
      return `<tr>
        <td>${r.timestamp}</td>
        <td>${escapeHtml(r.image_name)}</td>
        <td>${st["Oil Spill"].percentage.toFixed(2)}</td>
        <td>${st["Look-alike"].percentage.toFixed(2)}</td>
        <td>${st["Ship/Wake"].percentage.toFixed(2)}</td>
        <td>${st["Background"].percentage.toFixed(2)}</td>
        <td>${r.alert_count}</td>
        <td>${r.has_critical ? "Yes" : "No"}</td>
      </tr>`;
    })
    .join("");

  const total = history.length;
  const totalAlerts = history.reduce((a, r) => a + r.alert_count, 0);
  const critical = history.filter((r) => r.has_critical).length;
  const alertRate = total ? (totalAlerts / total) * 100 : 0;
  const oilVals = history.map((r) => r.statistics["Oil Spill"].percentage);
  const avg = oilVals.reduce((a, b) => a + b, 0) / oilVals.length;

  el.innerHTML = `
    <h4>📊 Detection Records</h4>
    <div class="table-wrap">
      <table class="history">
        <thead><tr>
          <th>Timestamp</th><th>Image</th><th>Oil Spill (%)</th><th>Look-alike (%)</th>
          <th>Ship/Wake (%)</th><th>Background (%)</th><th>Alerts</th><th>Has Critical</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <hr />
    <h3 class="section-title">📤 Management &amp; Export</h3>
    <div class="downloads">
      <button class="btn-download" id="histCsv">📥 Export as CSV</button>
      <button class="btn-download" id="histJson">📥 Export as JSON</button>
      <button class="btn btn-danger" id="histClear">🗑️ Clear History</button>
    </div>
    <hr />
    <h3 class="section-title">📈 Historical Analytics</h3>
    <div class="metric-row">
      <div class="metric-card"><div class="metric-label">Total Detections</div><div class="metric-value">${total}</div></div>
      <div class="metric-card"><div class="metric-label">Total Alerts</div><div class="metric-value">${totalAlerts}</div></div>
      <div class="metric-card"><div class="metric-label">Critical Alerts</div><div class="metric-value">${critical}</div></div>
      <div class="metric-card"><div class="metric-label">Alert Rate</div><div class="metric-value">${alertRate.toFixed(1)}%</div></div>
    </div>
    <div class="metric-row" style="margin-top:1rem;">
      <div class="metric-card"><div class="metric-label">Avg Oil Coverage</div><div class="metric-value">${avg.toFixed(2)}%</div></div>
      <div class="metric-card"><div class="metric-label">Max Oil Coverage</div><div class="metric-value">${Math.max(...oilVals).toFixed(2)}%</div></div>
      <div class="metric-card"><div class="metric-label">Min Oil Coverage</div><div class="metric-value">${Math.min(...oilVals).toFixed(2)}%</div></div>
    </div>`;

  $("histCsv").onclick = exportCsv;
  $("histJson").onclick = exportJson;
  $("histClear").onclick = clearHistory;
}

function exportCsv() {
  const history = loadHistory();
  const header = ["Timestamp", "DateTime", "Image", "Oil Spill (%)", "Look-alike (%)", "Ship/Wake (%)", "Background (%)", "Alerts", "Has Critical"];
  const lines = [header.join(",")];
  for (const r of history) {
    const st = r.statistics;
    lines.push([
      r.timestamp, r.datetime, `"${r.image_name}"`,
      st["Oil Spill"].percentage.toFixed(2),
      st["Look-alike"].percentage.toFixed(2),
      st["Ship/Wake"].percentage.toFixed(2),
      st["Background"].percentage.toFixed(2),
      r.alert_count, r.has_critical ? "Yes" : "No",
    ].join(","));
  }
  triggerDownload(new Blob([lines.join("\n")], { type: "text/csv" }), "spill_detection_history.csv");
}

function exportJson() {
  triggerDownload(
    new Blob([JSON.stringify(loadHistory(), null, 2)], { type: "application/json" }),
    "spill_detection_history.json"
  );
}

// ---------- UI wiring ----------
function buildLegend() {
  const el = $("legend");
  el.innerHTML = CLASS_NAMES.map((name, i) => {
    const c = CLASS_COLORS[i];
    return `<div class="legend-item"><div class="legend-color-box" style="background-color:rgb(${c[0]},${c[1]},${c[2]});"></div><span>${name}</span></div>`;
  }).join("");
}

function wireSidebar() {
  const sliders = [
    ["alpha", "alpha-val", (v) => v.toFixed(2)],
    ["th-oil", "th-oil-val", (v) => v.toFixed(1)],
    ["th-look", "th-look-val", (v) => v.toFixed(1)],
    ["th-ship", "th-ship-val", (v) => v.toFixed(1)],
  ];
  sliders.forEach(([id, valId, fmt]) => {
    const input = $(id);
    input.addEventListener("input", () => {
      $(valId).textContent = fmt(parseFloat(input.value));
      renderCurrent();
    });
  });
  ["drawContours", "class-0", "class-1", "class-2", "class-3"].forEach((id) => {
    $(id).addEventListener("change", renderCurrent);
  });
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      const id = "tab-" + tab.dataset.tab;
      $(id).classList.add("active");
      if (tab.dataset.tab === "history") renderHistory();
      if (tab.dataset.tab === "api") renderApi();
    });
  });
}

function wireUploader() {
  const dz = $("dropzone");
  const input = $("fileInput");
  $("browseBtn").addEventListener("click", (e) => { e.stopPropagation(); input.click(); });
  dz.addEventListener("click", () => input.click());
  input.addEventListener("change", () => { if (input.files[0]) runDetection(input.files[0]); });
  ["dragover", "dragenter"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); })
  );
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) runDetection(f);
  });
}

function wireDownloads() {
  $("dlOverlay").addEventListener("click", downloadOverlay);
  $("dlMask").addEventListener("click", downloadMask);
  $("dlReport").addEventListener("click", downloadReport);
}

function wireMobileSidebar() {
  $("sidebarToggle").addEventListener("click", () => $("sidebar").classList.toggle("open"));
}

// ---------- API tab ----------
function renderApi() {
  if ($("apiContent").dataset.done) return;
  $("apiContent").dataset.done = "1";
  $("apiContent").innerHTML = `
    <p>Inference runs entirely in the browser via ONNX Runtime Web — but the same ONNX model powers
    server-side or batch pipelines too. Examples below.</p>

    <div class="api-h">🟨 JavaScript (browser / Node, onnxruntime-web / onnxruntime-node)</div>
    <pre class="code">import * as ort from "onnxruntime-web";

const session = await ort.InferenceSession.create("model/spillguard.int8.onnx");

// input: Float32 NCHW [1,3,256,256], RGB normalized to [0,1]
const input = new ort.Tensor("float32", float32Data, [1, 3, 256, 256]);
const { logits } = await session.run({ input });
// argmax over channel dim -> 4-class mask (0=bg,1=oil,2=lookalike,3=ship)</pre>

    <div class="api-h">🐍 Python (onnxruntime — no PyTorch needed)</div>
    <pre class="code">import onnxruntime as ort, numpy as np, cv2

sess = ort.InferenceSession("spillguard.int8.onnx", providers=["CPUExecutionProvider"])
img = cv2.cvtColor(cv2.imread("satellite.png"), cv2.COLOR_BGR2RGB)
x = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
x = x.transpose(2, 0, 1)[None]                     # NCHW
logits = sess.run(None, {"input": x})[0]           # [1,4,256,256]
mask = logits.argmax(1)[0]                          # class indices
oil_pct = (mask == 1).mean() * 100
print(f"Oil spill coverage: {oil_pct:.2f}%")</pre>

    <div class="api-h">🌐 Optional REST wrapper (FastAPI + onnxruntime)</div>
    <pre class="code">from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort, numpy as np, cv2, io
from PIL import Image

app = FastAPI(title="AI SpillGuard Pro API")
sess = ort.InferenceSession("spillguard.int8.onnx")

@app.post("/api/v1/detect")
async def detect(file: UploadFile = File(...)):
    img = np.array(Image.open(io.BytesIO(await file.read())).convert("RGB"))
    x = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
    logits = sess.run(None, {"input": x.transpose(2,0,1)[None]})[0]
    mask = logits.argmax(1)[0]
    pct = {c: float((mask == i).mean() * 100)
           for i, c in enumerate(["Background","Oil Spill","Look-alike","Ship/Wake"])}
    return {"statistics": pct, "has_oil_spill": pct["Oil Spill"] > 0}</pre>

    <div class="api-h">📄 Response / mask classes</div>
    <pre class="code">0 = Background   (clean ocean)      rgb(0,0,0)
1 = Oil Spill    (primary target)   rgb(255,0,124)
2 = Look-alike   (false positive)   rgb(255,204,51)
3 = Ship/Wake    (vessel activity)  rgb(51,221,255)</pre>

    <div class="api-h">📚 Model</div>
    <pre class="code">Architecture : U-Net + ResNet34 encoder
Input        : 256x256 RGB, normalized [0,1], NCHW
Output       : 4-class segmentation logits [1,4,256,256]
Format       : ONNX (opset 17), dynamic int8 quantized (~24 MB)
Parity       : 100% argmax agreement vs original PyTorch model</pre>`;
}

// ---------- Helpers ----------
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

let toastTimer = null;
function showToast(msg, isError) {
  const t = $("toast");
  t.textContent = msg;
  t.style.background = isError ? "#cc0000" : "#1e7e34";
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 3000);
}

// ---------- Init ----------
document.addEventListener("DOMContentLoaded", () => {
  buildLegend();
  wireSidebar();
  wireTabs();
  wireUploader();
  wireDownloads();
  wireMobileSidebar();
  // Warm the model in the background so the first detection is fast.
  loadModel().catch(() => {
    $("deviceStatus").textContent = "⚠️ Model will load on first detection";
  });
});
