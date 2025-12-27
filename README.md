# 🛢️ AI SpillGuard Pro
### Enterprise-Grade Oil Spill Detection System using Deep Learning & Satellite Imagery

AI SpillGuard Pro is a production-ready web application that leverages **U-Net semantic segmentation with ResNet34 backbone** to detect and classify oil spills in SAR (Synthetic Aperture Radar) satellite imagery. The system provides real-time detection, intelligent alerting, persistent storage, and comprehensive analytics for marine environmental monitoring.



## ✨ Key Features
* 🎯 **Multi-Class Segmentation:** Distinguishes between oil spills, look-alikes, ships/wakes, and background with pixel-level precision.
* 🚨 **Intelligent Alert System:** Configurable threshold-based monitoring with priority-ranked critical alerts.
* 📊 **Real-Time Visualization:** Interactive overlay rendering with adjustable transparency and class toggles.
* 💾 **Persistent Storage:** Automatic saving of detection results, images, and searchable history.
* 📈 **Analytics Dashboard:** Comprehensive statistics, coverage distribution, and historical trend analysis.
* 🔌 **REST API Integration:** FastAPI-based programmatic access with Docker deployment support.
* ⚡ **GPU Acceleration:** Optimized PyTorch inference with CUDA support for high-throughput processing.
* 🎨 **Professional UI:** Streamlit-based interface with responsive design and enterprise styling.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Deep Learning Framework** | PyTorch 2.x |
| **Model Architecture** | U-Net with ResNet34 encoder (segmentation_models_pytorch) |
| **Frontend** | Streamlit 1.x |
| **Backend API** | FastAPI (optional deployment) |
| **Image Processing** | OpenCV, PIL, NumPy |
| **Data Storage** | JSON-based history tracking |
| **Visualization** | Matplotlib, Plotly |
| **Development** | Python 3.8+, Jupyter Notebooks |

---

## 🏗️ Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit Frontend                     │
│  (User Interface + Visualization + Controls)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│OilSpillDetector│ │AlertSystem │ │StorageManager│
│(Inference)   │ │(Monitoring) │ │(Persistence) │
└──────┬──────┘ └─────────────┘ └─────────────┘
       │
       ▼
┌─────────────────────┐
│   U-Net Model       │
│   (best_model.pth)  │
│   ResNet34 Encoder  │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│ Detection Results   │
│ • Segmentation Mask │
│ • Statistics        │
│ • Alerts            │
└─────────────────────┘

```

**Component Interactions:**

* **OilSpillDetector:** Handles model loading, preprocessing, inference, and post-processing.
* **AlertSystem:** Monitors detection statistics against configurable thresholds.
* **StorageManager:** Persists detection history to `history.json`.
* **Streamlit UI:** Orchestrates user interactions across detection, history, and API tabs.

---

## 📦 Installation & Setup

### Prerequisites

* Python 3.8 or higher
* CUDA-compatible GPU (optional, for acceleration)
* 4GB+ RAM recommended

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd <project-directory>

```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt

```

### Step 3: Verify Model Files

Ensure `best_model.pth` exists in the project root:

```bash
ls -lh best_model.pth

```

### Step 4: Prepare Dataset (Optional - Training Only)

If retraining the model:

```bash
python src/preprocess_train.py
python src/preprocess_val.py
python src/preprocess_test.py

```

### Step 5: Launch Application

```bash
streamlit run app.py

```

Access the application at `http://localhost:8501`

---

## 🚀 Usage Examples

### Web Interface (Streamlit)

1. **Upload Image for Detection:** Navigate to "Detection" tab, click "Browse files", and view results.
2. **Configure Alert Thresholds:** Sidebar → Alert Thresholds (Adjust sliders for Oil Spill, Look-alike, etc.).
3. **Export Detection History:** Navigate to "History" tab and click "Export History (CSV)".

### Python API (Programmatic)

```python
from app import OilSpillDetector, AlertSystem
import cv2

# Initialize detector
detector = OilSpillDetector("best_model.pth")
alert_system = AlertSystem()

# Load image (ensure RGB format)
image = cv2.imread("satellite_image.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run detection
results = detector.predict(image_rgb)

# Access results
print(f"Oil Spill Coverage: {results['statistics']['Oil Spill']['percentage']:.2f}%")

# Check alerts
alerts = alert_system.check_alerts(results["statistics"])
for alert in alerts:
    print(f"{alert['severity']}: {alert['message']}")

```

---

## 📡 API Reference

### REST API Endpoints (FastAPI)

**Base URL:** `http://localhost:8000/api/v1`

* **POST `/detect**`: Detect oil spills in uploaded image.
* **GET `/health**`: Health check endpoint.

---

## 🎨 Class Definitions

| Class ID | Class Name | RGB Color | Hex Code |
| --- | --- | --- | --- |
| 0 | Background | (0, 0, 0) | #000000 |
| 1 | Oil Spill | (255, 0, 124) | #FF007C |
| 2 | Look-alike | (255, 204, 51) | #FFCC33 |
| 3 | Ship/Wake | (51, 221, 255) | #33DDFF |

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build Image
docker build -t ai-spillguard-api .

# Run Container
docker run -p 8000:8000 ai-spillguard-api

```

---

## 🤝 Contribution Guidelines

A contribution is considered complete when:

* ✅ Follows PEP 8 style guidelines.
* ✅ Type hints and Docstrings are added.
* ✅ Unit tests pass without regression.
* ✅ GPU/CPU compatibility is maintained.

---

## 📝 License
This project is licensed under the MIT License.

Copyright (c) 2025 Prem Chand

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files.

Built with ❤️ for Environmental Protection and Marine Conservation.

---
