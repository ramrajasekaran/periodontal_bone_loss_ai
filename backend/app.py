from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import base64
import io

# Resolve paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(PROJECT_DIR, 'frontend')
SRC_DIR = os.path.join(BACKEND_DIR, 'src')
MODELS_DIR = os.path.join(BACKEND_DIR, 'models')

sys.path.insert(0, SRC_DIR)
from models import get_model
from dataset import get_transforms
from explainability import generate_gradcam

app = Flask(__name__, 
            template_folder=os.path.join(FRONTEND_DIR, 'templates'),
            static_folder=None)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, 'static'), filename)

# Pre-load available models on startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_available_models():
    """Return list of models that have trained weights."""
    available = []
    for name in ['efficientnet', 'resnet50', 'cnn']:
        path = os.path.join(MODELS_DIR, f'best_{name}.pth')
        if os.path.exists(path):
            available.append(name)
    if available:
        available.insert(0, 'ensemble')
    return available

def validate_dental_xray(image_path):
    """Validates if the image is a panoramic dental X-ray."""
    img = cv2.imread(image_path)
    if img is None:
        return False, "Could not read the uploaded image."
    
    h, w = img.shape[:2]
    aspect_ratio = w / h
    
    if aspect_ratio < 1.4 or aspect_ratio > 3.5:
        return False, f"Invalid aspect ratio ({aspect_ratio:.2f}). Panoramic dental X-rays are distinctly wide. Please upload a valid panoramic radiograph."
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_sat = np.mean(hsv[:, :, 1])
    if mean_sat > 10.0:
        return False, "The image contains too much color to be an X-ray. Dental radiographs are grayscale. Please upload a valid X-ray."
    
    return True, ""

def numpy_to_base64(arr):
    """Convert a numpy image array to base64 string for embedding in HTML."""
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (arr * 255).astype(np.uint8)
    if len(arr.shape) == 2:  # Grayscale heatmap
        img = Image.fromarray(arr, mode='L')
    else:
        img = Image.fromarray(arr, mode='RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@app.route('/')
def index():
    models = get_available_models()
    return render_template('index.html', models=models)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400
    
    file = request.files['image']
    model_name = request.form.get('model', 'ensemble')
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        file.save(tmp.name)
        temp_path = tmp.name
    
    try:
        # Validate
        is_valid, err_msg = validate_dental_xray(temp_path)
        if not is_valid:
            return jsonify({'error': err_msg}), 400
        
        # Preprocess image
        _, val_transform = get_transforms()
        img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({'error': 'Could not decode image.'}), 400
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(np.ascontiguousarray(img, dtype=np.uint8))
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img_rgb)
        transformed = val_transform(pil_img)
        input_tensor = torch.as_tensor(transformed).unsqueeze(0).to(device)
        
        # Lookup tabular features from metadata CSV
        import pandas as pd
        import re
        meta_csv_path = os.path.join(BACKEND_DIR, 'data', 'meta_data.csv')
        tabular_features = torch.zeros((1, 6), dtype=torch.float32)
        if os.path.exists(meta_csv_path) and file.filename:
            df = pd.read_csv(meta_csv_path)
            clean_name = re.sub(r'\s*\(\d+\)|_\d+$', '', os.path.splitext(file.filename)[0].lower())
            patient_row = df[df['File name'].astype(str).str.lower().str.contains(clean_name, regex=False)]
            if not patient_row.empty:
                row = patient_row.iloc[0]
                age = float(row['Age']) / 100.0
                gender = float(row['Gender']) / 2.0
                missing_teeth = float(row['Number of missing teeth']) / 32.0
                implant = float(row['Implant'])
                residual_root = float(row['Residual root'])
                func_tooth_log = float(row['Functional tooth logarithm']) / 32.0
                tabular_features = torch.tensor([[age, gender, missing_teeth, implant, residual_root, func_tooth_log]], dtype=torch.float32)
        tabular_features = tabular_features.to(device)
        
        # Determine models to evaluate
        if model_name == 'ensemble':
            eval_models = ['efficientnet', 'resnet50', 'cnn']
        else:
            eval_models = [model_name]
        
        all_probs = []
        loaded_models = []
        for name in eval_models:
            path = os.path.join(MODELS_DIR, f'best_{name}.pth')
            if os.path.exists(path):
                model = get_model(name).to(device)
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                model.eval()
                with torch.no_grad():
                    outputs = model(input_tensor, tabular_features)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    all_probs.append(probs)
                loaded_models.append(name)
        
        if not all_probs:
            return jsonify({'error': f'No trained weight checkpoint found.'}), 400
        
        # Soft voting average
        avg_probs = np.mean(all_probs, axis=0)
        pred_class = int(np.argmax(avg_probs))
        
        classes = ["Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)"]
        
        # Grad-CAM visualization model (use requested model or efficientnet for ensemble)
        xai_model = 'efficientnet' if model_name == 'ensemble' or 'efficientnet' in loaded_models else loaded_models[0]
        gradcam_data = {}
        try:
            rgb_img, heatmap, viz = generate_gradcam(xai_model, temp_path, file.filename, model_dir=MODELS_DIR)
            gradcam_data = {
                'original': numpy_to_base64(rgb_img),
                'heatmap': numpy_to_base64(heatmap),
                'overlay': numpy_to_base64(viz),
            }
        except Exception as e:
            gradcam_data = {'error': str(e)}
        
        result = {
            'prediction': classes[pred_class],
            'grade': pred_class + 1,
            'confidence': {
                'Grade 1': round(float(avg_probs[0]) * 100, 2),
                'Grade 2': round(float(avg_probs[1]) * 100, 2),
                'Grade 3': round(float(avg_probs[2]) * 100, 2),
            },
            'model_used': model_name if model_name != 'ensemble' else f"Ensemble ({', '.join(loaded_models)})",
            'gradcam': gradcam_data,
        }
        
        return jsonify(result)
    
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    app.run(host='0.0.0.0', port=5000, debug=True)
