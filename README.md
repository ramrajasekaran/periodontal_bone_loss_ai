# Explainable Deep Learning-Based Severity Assessment of Periodontal Bone Resorption

This repository contains the complete implementation for the research paper on automated, explainable deep-learning severity assessment of periodontal bone resorption from panoramic dental radiographs.

## Pipeline Improvements & Fixes

This codebase has been thoroughly audited and fixed to ensure rigorous machine learning practices:
1. **True Patient-Level Split:** Guarantees zero data leakage between Train, Validation, and Test sets.
2. **Scipy/Sklearn Removal:** Custom `metrics.py` and `explainability.py` to avoid Windows Application Control (WDAC) DLL blocking issues.
3. **Single Source of Truth:** `configs/class_mapping.json` ensures consistent label interpretation across all scripts.
4. **Canonical Inference:** `predict.py` unifies inference for CLI, Flask, and Streamlit, guaranteeing identical preprocessing.
5. **Multimodal Ablation:** `ablation.py` to compare Image-Only, Tabular-Only, and Multimodal approaches.

## Project Structure

```
periodontal_bone_loss_ai/
├── backend/               
│   ├── streamlit_app.py   # Streamlit demonstration app (with Compare mode)
│   ├── app.py             # Flask API server
│   ├── src/               
│   │   ├── data_split.py  # 70/15/15 patient-level stratified split
│   │   ├── dataset.py     # PyTorch dataset, CLAHE, augmentations
│   │   ├── models.py      # CNN, ResNet50, EfficientNet-B0, TabularMLP
│   │   ├── train.py       # Training loop (ReduceLROnPlateau, EarlyStopping)
│   │   ├── evaluate.py    # Test set evaluation & comparison table
│   │   ├── predict.py     # Canonical inference & disagreement diagnosis
│   │   ├── ablation.py    # Ablation study (Exp A, B, C)
│   │   ├── metrics.py     # Pure numpy/torch metrics (replaces sklearn)
│   │   └── explainability.py # Pure PyTorch Grad-CAM (replaces pytorch_grad_cam)
│   ├── data/              # Datasets, metadata, and generated CSV splits
│   ├── models/            # Saved PyTorch trained weights (.pth)
│   ├── configs/           # Class mapping and normalization configs
│   ├── results/           # Generated figures, tables, and metrics
│   └── requirements.txt   # Python dependencies
└── README.md
```

## Setup & Reproducibility

1. **Install dependencies:**
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Generate Data Split & Configs (70/15/15):**
   *Must be run first to generate `configs/class_mapping.json`*
   ```bash
   python backend/src/data_split.py
   ```

3. **Train Models:**
   ```bash
   python backend/src/train.py --model efficientnet --epochs 50 --batch_size 8
   python backend/src/train.py --model resnet50     --epochs 50 --batch_size 8
   python backend/src/train.py --model cnn          --epochs 50 --batch_size 16
   ```

4. **Evaluate on Held-Out Test Set:**
   ```bash
   # Single model
   python backend/src/evaluate.py --model efficientnet
   
   # Compare all trained models
   python backend/src/evaluate.py --model all
   ```

5. **Run Ablation Study (Image vs Tabular vs Multimodal):**
   ```bash
   # Train the tabular baseline, then evaluate all modes
   python backend/src/ablation.py --mode full
   ```

6. **Run Inference / Diagnosis (CLI):**
   ```bash
   python backend/src/predict.py --image path/to/image.jpg --model all
   ```

7. **Run Web Application (Streamlit):**
   ```bash
   python -m streamlit run backend/streamlit_app.py
   ```
