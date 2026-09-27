"""
detailed_evaluate.py — Comprehensive evaluation and prediction script for all models.

Generates:
1. Confusion matrices
2. Classification reports (CSV) with per-class and macro/weighted metrics
3. Individual predictions (CSV) for all test images
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import per_class_prf, accuracy_np, confusion_matrix_np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BACKEND_DIR, 'data')
MODELS_DIR = os.path.join(BACKEND_DIR, 'models')
RESULTS_DIR = os.path.join(BACKEND_DIR, 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, 'predictions')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

sys.path.insert(0, SRC_DIR)
from dataset import PeriodontalDataset, get_transforms
from models import get_model

CLASS_NAMES = ["Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)"]
# The class mapping used during training is Level 1=0, Level 2=1, Level 3=2

def get_predictions(model_name, test_loader, device, filenames, exp_num=1):
    if exp_num > 1:
        model_path = os.path.join(MODELS_DIR, f'best_{model_name}_exp{exp_num}.pth')
    else:
        # Default to old naming for backward compatibility if we didn't rename it
        model_path = os.path.join(MODELS_DIR, f'best_{model_name}_exp1.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, f'best_{model_name}.pth')
    if not os.path.exists(model_path):
        print(f"[SKIP] Model checkpoint not found: {model_path}")
        return None
        
    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_confs = []
    
    with torch.no_grad():
        for images, tabular_features, labels in tqdm(test_loader, desc=f"Evaluating {model_name}", leave=False):
            images = images.to(device)
            tabular_features = tabular_features.to(device)
            
            outputs = model(images, tabular_features)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_confs.extend(np.max(probs, axis=1).tolist())
            
    return np.array(all_preds), np.array(all_labels), np.array(all_confs)

def evaluate_models(exp_num=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_csv = os.path.join(DATA_DIR, 'test.csv')
    
    _, val_transform = get_transforms()
    test_dataset = PeriodontalDataset(test_csv, DATA_DIR, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    filenames = test_dataset.data_frame['File name'].tolist()
    
    models_to_eval = ['cnn', 'resnet50', 'efficientnet']
    
    for model_name in models_to_eval:
        res = get_predictions(model_name, test_loader, device, filenames, exp_num)
        if res is None:
            continue
            
        preds, labels, confs = res
        
        # Calculate metrics
        acc = accuracy_np(labels, preds)
        
        # Per class metrics
        p_class, r_class, f1_class, support = per_class_prf(labels, preds, num_classes=3)
        
        # Macro metrics
        p_macro, r_macro, f1_macro = float(np.mean(p_class)), float(np.mean(r_class)), float(np.mean(f1_class))
        
        # Weighted metrics
        total_s = float(np.sum(support)) if np.sum(support) > 0 else 1.0
        p_weight = float(np.sum(p_class * support) / total_s)
        r_weight = float(np.sum(r_class * support) / total_s)
        f1_weight = float(np.sum(f1_class * support) / total_s)
        
        # 1. Save Classification Report CSV
        report_data = []
        for i in range(3):
            report_data.append({
                'class': CLASS_NAMES[i],
                'precision': p_class[i],
                'recall': r_class[i],
                'f1-score': f1_class[i],
                'support': support[i]
            })
            
        report_data.append({
            'class': 'accuracy',
            'precision': '',
            'recall': '',
            'f1-score': acc,
            'support': len(labels)
        })
        
        report_data.append({
            'class': 'macro avg',
            'precision': p_macro,
            'recall': r_macro,
            'f1-score': f1_macro,
            'support': len(labels)
        })
        
        report_data.append({
            'class': 'weighted avg',
            'precision': p_weight,
            'recall': r_weight,
            'f1-score': f1_weight,
            'support': len(labels)
        })
        
        df_report = pd.DataFrame(report_data)
        report_path = os.path.join(TABLES_DIR, f'classification_report_{model_name}_exp{exp_num}.csv')
        df_report.to_csv(report_path, index=False)
        
        # 2. Save Confusion Matrix Plot
        cm = confusion_matrix_np(labels, preds, num_classes=3)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'Confusion Matrix — {model_name.upper()} (Exp {exp_num})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_path = os.path.join(FIG_DIR, f'cm_{model_name}_exp{exp_num}.png')
        plt.savefig(cm_path, dpi=120)
        plt.close()
        
        # 3. Save Predictions CSV
        preds_data = []
        for i in range(len(filenames)):
            true_l = labels[i]
            pred_l = preds[i]
            preds_data.append({
                'image_filename': filenames[i],
                'true_label': true_l,
                'true_grade': CLASS_NAMES[true_l],
                'predicted_label': pred_l,
                'predicted_grade': CLASS_NAMES[pred_l],
                'confidence': confs[i]
            })
            
        df_preds = pd.DataFrame(preds_data)
        preds_path = os.path.join(PREDICTIONS_DIR, f'predictions_{model_name}_exp{exp_num}.csv')
        df_preds.to_csv(preds_path, index=False)
        
        # Print results to terminal
        print(f"\n============================================================")
        print(f"RESULTS FOR MODEL: {model_name.upper()} (Experiment {exp_num})")
        print(f"============================================================")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro P/R/F1: {p_macro:.4f} / {r_macro:.4f} / {f1_macro:.4f}")
        print(f"Weighted P/R/F1: {p_weight:.4f} / {r_weight:.4f} / {f1_weight:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report saved to:", report_path)
        print("Confusion Matrix plot saved to:", cm_path)
        print("Predictions saved to:", preds_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=int, default=1, choices=[1, 2, 3, 4], help="Experiment number (1-4)")
    args = parser.parse_args()
    evaluate_models(args.experiment)
