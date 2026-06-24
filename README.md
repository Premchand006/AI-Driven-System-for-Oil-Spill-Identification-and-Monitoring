# 🛢️ AI SpillGuard Pro — Vercel Edition

A **100% static, browser-based** port of the AI SpillGuard Pro Streamlit app. It keeps the
same enterprise UI (dark sidebar, gradient header, 3 tabs, donut chart, alert cards, history)
and the **same trained model** — but runs entirely in the user's browser via
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). No Python server, no GPU,
no cold starts — perfect for Vercel's static hosting.

## Why this architecture?

The original app runs a ~280 MB PyTorch model (U-Net + ResNet34). PyTorch alone exceeds
Vercel's serverless function limit (250 MB unzipped), so a Python backend isn't feasible.
Instead, the model was exported to **ONNX and int8-quantized to ~24 MB** and runs **client-side**
in WebAssembly. Inference happens on the visitor's machine — Vercel only serves static files.

- ✅ Identical detection logic (4-class segmentation, same colors, same alert thresholds)
- ✅ **100% argmax parity** with the original PyTorch model (verified); int8 build agrees to ~99.85%
- ✅ Same UI/UX, charts, history, downloads
- ✅ Zero backend cost, instant scale, runs offline after first load
- ⚠️ First visit downloads the ~24 MB model once (then browser-cached for a year)

## Project structure

```
Vercel_version/
├── index.html              # App shell (sidebar, tabs, layout)
├── styles.css              # Enterprise UI styling (mirrors the Streamlit theme)
├── app.js                  # Inference engine + UI logic (ONNX Runtime Web)
├── model/
│   └── spillguard.int8.onnx  # Quantized model (~24 MB), shipped & served
├── vercel.json             # Static config + long-cache headers for the model
├── tools/
│   └── convert_to_onnx.py  # Regenerate the ONNX model from best_model.pth
└── README.md
```

## Deploy to Vercel

### Option A — Vercel CLI
```bash
npm i -g vercel
cd Vercel_version
vercel            # preview
vercel --prod     # production
```

### Option B — Git import (dashboard)
1. Push this folder to a GitHub repo (or set **Root Directory** = `Vercel_version`).
2. In Vercel: **New Project → Import**.
3. Framework Preset: **Other**. Build Command: *(none)*. Output Directory: *(leave default / root)*.
4. Deploy. Done.

> The model file is 24 MB — well under Vercel's static limits. It's served with a
> 1-year immutable cache header (see `vercel.json`).

## Run locally

Any static server works (don't open `index.html` via `file://` — `fetch()` of the model needs http):

```bash
cd Vercel_version
python -m http.server 8080
# open http://localhost:8080
```

## Regenerating the model

If you retrain and produce a new `best_model.pth`, regenerate the ONNX from the project root:

```bash
venv/Scripts/python.exe Vercel_version/tools/convert_to_onnx.py
```

This re-exports `model/spillguard.int8.onnx` (and a full-precision `spillguard.onnx` you can
optionally ship instead by editing `MODEL_URL` in `app.js`). The script also prints a parity
check against the PyTorch model.

## Notes / differences vs the Streamlit app

| Feature | Streamlit app | Vercel edition |
|---|---|---|
| Inference | PyTorch (server) | ONNX Runtime Web (browser) |
| History storage | `detection_history/` on disk | browser `localStorage` |
| Charts | Plotly | Chart.js (donut) |
| Statistics, colors, alerts | — | identical |
| Auto-save, CSV/JSON export, downloads | — | identical |

History is per-browser (localStorage) since there's no server filesystem. Everything else —
the detection pipeline, class colors, alpha-blended overlay with contours, alert thresholds,
and analytics — matches the original.
