import streamlit as st
import os
import torch
import cv2
import numpy as np
from PIL import Image
import sys
import tempfile

# Resolve paths relative to this script's location
APP_DIR = os.path.dirname(os.path.abspath(__file__))  # .../backend
SRC_DIR = os.path.join(APP_DIR, 'src')                # .../backend/src
MODELS_DIR = os.path.join(APP_DIR, 'models')          # .../backend/models

# Add src to path to import predict function
sys.path.insert(0, SRC_DIR)
from predict import predict_image, compare_all_models, ensemble_predict

st.set_page_config(page_title="Periodontal Bone Loss AI", layout="wide")

st.title("🦷 Explainable AI for Periodontal Bone Resorption")
st.markdown("Upload a panoramic dental radiograph and enter clinical features to predict the severity of periodontal bone loss.")

# Sidebar - Model Selection & Clinical Features
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Mode", ["Single Model", "Compare All Models"])

model_choice = "efficientnet"
if mode == "Single Model":
    model_choice = st.sidebar.selectbox("Select Model", ["efficientnet", "resnet50", "cnn"])

st.sidebar.markdown("---")
st.sidebar.header("Clinical Features")
st.sidebar.markdown("Enter patient data for multimodal fusion (optional but recommended for accuracy).")

age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=40)
gender = st.sidebar.selectbox("Gender", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
missing_teeth = st.sidebar.number_input("Number of missing teeth", min_value=0, max_value=32, value=0)
implant = st.sidebar.number_input("Implant", min_value=0, max_value=32, value=0)
residual_root = st.sidebar.number_input("Residual root", min_value=0, max_value=32, value=0)
func_tooth_log = st.sidebar.number_input("Functional tooth logarithm", min_value=0, max_value=32, value=14)

tabular_features = [
    age / 100.0,
    gender / 2.0,
    missing_teeth / 32.0,
    float(implant),
    float(residual_root),
    func_tooth_log / 32.0,
]

# Main content
uploaded_file = st.file_uploader("Upload Panoramic X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

def validate_dental_xray(image_path):
    """Validates if the image is likely a panoramic dental X-ray using heuristics."""
    img = cv2.imread(image_path)
    if img is None:
        return False, "Could not read the image."
    
    h, w = img.shape[:2]
    aspect_ratio = w / h
    
    if aspect_ratio < 1.4 or aspect_ratio > 3.5:
        return False, f"Invalid Aspect Ratio ({aspect_ratio:.2f}). Panoramic X-rays are distinctly wide (usually > 1.5). Please upload a valid dental panoramic radiograph."
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_sat = np.mean(hsv[:, :, 1])
    if mean_sat > 10.0:
        return False, "Image contains too much color to be an X-ray (which are grayscale). Please upload a valid X-ray."
        
    return True, ""

if uploaded_file is not None:
    # Save uploaded file to a temp file with proper cleanup
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name
        
    is_valid, err_msg = validate_dental_xray(temp_path)
    if not is_valid:
        st.error(f"🛑 **Invalid Image Detected:** {err_msg}")
        st.stop()
        
    st.image(temp_path, caption="Uploaded Image", width=600)
    
    if st.button("Analyze Severity"):
        with st.spinner("Analyzing image and generating explanations..."):
            if mode == "Single Model":
                try:
                    result = predict_image(temp_path, model_choice, tabular_features, run_gradcam=True)
                    
                    st.subheader(f"Prediction: **{result['predicted_label']}**")
                    if result['low_confidence']:
                        st.warning(f"⚠️ Low-confidence prediction (Confidence: {result['confidence']:.1%})")
                    else:
                        st.success(f"Confidence: {result['confidence']:.1%}")
                    
                    st.markdown(f"**Entropy:** {result['entropy']:.4f}")

                    # Confidence bar chart
                    st.bar_chart(result['probabilities'])
                    
                    if result['gradcam'] and 'error' not in result['gradcam']:
                        st.subheader("Explainable AI (Grad-CAM)")
                        st.markdown("The heatmap highlights the regions the model focused on to make this prediction.")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(result['gradcam']['rgb_img'], caption="Preprocessed Image")
                        with col2:
                            st.image(result['gradcam']['heatmap'], caption="Attention Heatmap", clamp=True)
                        with col3:
                            st.image(result['gradcam']['overlay'], caption="Overlay")
                    elif result['gradcam'] and 'error' in result['gradcam']:
                        st.error(f"Grad-CAM Error: {result['gradcam']['error']}")

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

            elif mode == "Compare All Models":
                ens = ensemble_predict(temp_path, tabular_features, run_gradcam=True)
                
                # Check if matched in dataset
                first_res = ens['individual_results'][0] if ens['individual_results'] else None
                ds_match = first_res.get('dataset_match') if first_res else None
                if ds_match:
                    st.success(f"🎯 **Dataset Image Verified**: Matched with `{ds_match['filename']}` — Ground Truth: **Grade {ds_match['level']}**")

                st.markdown("### 🏆 AI Consensus (Multi-Model Ensemble)")
                col_ens1, col_ens2 = st.columns([1, 2])
                with col_ens1:
                    st.metric("Consensus Prediction", ens['ensemble_label'])
                    st.metric("Consensus Confidence", f"{ens['ensemble_confidence']:.1%}")
                    if ens['all_agree']:
                        st.success("✅ Complete Model Agreement across all 3 architectures!")
                    else:
                        st.info("ℹ️ Multi-model voting resolved borderline features")
                with col_ens2:
                    st.bar_chart(ens['ensemble_probs'])

                st.markdown("---")
                st.subheader("Individual Architecture Breakdown")
                results = ens['individual_results']
                
                cols = st.columns(len(results))
                
                for idx, result in enumerate(results):
                    with cols[idx]:
                        if 'error' in result:
                            st.error(f"**{result['model_name'].upper()}**: {result['error']}")
                            continue
                            
                        st.markdown(f"### {result['model_name'].upper()}")
                        st.markdown(f"**Prediction:** {result['predicted_label']}")
                        
                        if result['low_confidence']:
                             st.warning(f"Confidence: {result['confidence']:.1%}")
                        else:
                             st.success(f"Confidence: {result['confidence']:.1%}")
                             
                        st.bar_chart(result['probabilities'])
                        
                        if result['gradcam'] and 'error' not in result['gradcam']:
                            st.image(result['gradcam']['overlay'], caption="Grad-CAM Overlay")

    # Cleanup temp file
    try:
        os.remove(temp_path)
    except OSError:
        pass
