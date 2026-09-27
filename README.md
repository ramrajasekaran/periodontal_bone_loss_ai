# 🦷 Periodontal Bone Loss AI

> **Explainable Deep Learning-Based Severity Assessment of Periodontal Bone Resorption**  
> Automated classification of periodontal bone loss from panoramic dental radiographs using CNN, ResNet-50, and EfficientNet-B0.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![Flask](https://img.shields.io/badge/Flask-API-green?logo=flask)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

This project implements an end-to-end AI pipeline for detecting and classifying the severity of periodontal (gum) bone loss from panoramic X-ray images. It compares multiple deep learning architectures and provides:

- ✅ **Multi-model comparison** — Custom CNN, ResNet-50, EfficientNet-B0
- ✅ **Explainability** — Pure PyTorch Grad-CAM (no scipy/sklearn dependency)
- ✅ **Ablation study** — Image-only vs. Tabular-only vs. Multimodal
- ✅ **Zero data leakage** — True patient-level 70/15/15 stratified split
- ✅ **Web interfaces** — Streamlit demo app + Flask REST API

---

## 🏗️ Project Structure

```
periodontal_bone_loss_ai/
├── backend/
│   ├── app.py                  # Flask REST API server
│   ├── streamlit_app.py        # Streamlit demonstration app
│   ├── train_balanced_fast.py  # Quick training with class balancing
│   ├── run_final_pipeline.py   # End-to-end pipeline runner
│   ├── requirements.txt        # Python dependencies
│   ├── src/
│   │   ├── data_split.py       # 70/15/15 patient-level stratified split
│   │   ├── dataset.py          # PyTorch dataset, CLAHE, augmentations
│   │   ├── models.py           # CNN, ResNet50, EfficientNet-B0, TabularMLP
│   │   ├── train.py            # Training loop (ReduceLROnPlateau, EarlyStopping)
│   │   ├── train_final.py      # Final training with best hyperparameters
│   │   ├── evaluate.py         # Test set evaluation & model comparison
│   │   ├── detailed_evaluate.py# Detailed per-class metrics
│   │   ├── predict.py          # Canonical inference & disagreement diagnosis
│   │   ├── ablation.py         # Ablation study (Exp A, B, C)
│   │   ├── metrics.py          # Pure numpy/torch metrics (replaces sklearn)
│   │   ├── losses.py           # Custom loss functions
│   │   └── explainability.py   # Pure PyTorch Grad-CAM
│   ├── configs/                # class_mapping.json, normalization configs
│   ├── data/                   # Datasets, metadata, CSV splits (not tracked)
│   ├── models/                 # Trained model weights (.pth) (not tracked)
│   └── results/                # Figures, tables, metrics, predictions
├── frontend/
│   ├── templates/index.html    # Web UI template
│   ├── static/css/style.css    # Styling
│   └── static/js/main.js      # Frontend JavaScript
├── run_experiments.py          # Experiment runner script
├── run_full_pipeline.bat       # Windows batch pipeline runner
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended) or CPU

### 1. Clone the repository
```bash
git clone https://github.com/ramrajasekaran/periodontal_bone_loss_ai.git
cd periodontal_bone_loss_ai
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r backend/requirements.txt
```

---

## 🚀 Usage

### Step 1 — Generate Data Split (run first!)
*Creates `configs/class_mapping.json` and 70/15/15 patient-level stratified splits.*
```bash
python backend/src/data_split.py
```

### Step 2 — Train Models
```bash
python backend/src/train.py --model efficientnet --epochs 50 --batch_size 8
python backend/src/train.py --model resnet50     --epochs 50 --batch_size 8
python backend/src/train.py --model cnn          --epochs 50 --batch_size 16
```

### Step 3 — Evaluate on Held-Out Test Set
```bash
# Single model
python backend/src/evaluate.py --model efficientnet

# Compare all trained models
python backend/src/evaluate.py --model all
```

### Step 4 — Run Ablation Study
```bash
python backend/src/ablation.py --mode full
```

### Step 5 — Run Inference
```bash
python backend/src/predict.py --image path/to/image.jpg --model all
```

### Step 6 — Launch Streamlit Web App
```bash
python -m streamlit run backend/streamlit_app.py
```

### Step 7 — Launch Flask API
```bash
python backend/app.py
```
API will be available at `http://localhost:5000`.

---

## 🔬 Key Design Decisions

| Feature | Implementation |
|---------|---------------|
| **Zero Data Leakage** | True patient-level split — same patient never appears in train + test |
| **No sklearn/scipy** | Custom `metrics.py` to avoid Windows WDAC DLL blocking issues |
| **Single Source of Truth** | `configs/class_mapping.json` for consistent label interpretation |
| **Canonical Inference** | `predict.py` unifies CLI, Flask, Streamlit preprocessing |
| **Explainability** | Pure PyTorch Grad-CAM (no external `pytorch_grad_cam` dependency) |

---

## 📊 Results

Trained models and result files are stored in `backend/results/`:
- `tables/model_comparison.csv` — Accuracy, F1, AUC across all models
- `tables/classification_report_*.csv` — Per-class precision/recall/F1
- `tables/ablation_study.csv` — Ablation experiment results
- `predictions/` — Per-sample predictions from each model

---

## 🌐 API Reference (Flask)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | `POST` | Upload image, returns severity classification + Grad-CAM |
| `/health` | `GET` | Health check |
| `/models` | `GET` | List available trained models |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- PyTorch & TorchVision for model architectures
- Streamlit for rapid prototyping of the demo app
- OpenCV for CLAHE image preprocessing
