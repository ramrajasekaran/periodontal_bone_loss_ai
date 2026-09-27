"""
evaluate.py  —  Fixed evaluation pipeline for held-out test set.

Bugs fixed:
  BUG-2: model was called as model(images) — tabular features not passed
  BUG-8: relative paths replaced with __file__-relative absolute paths
  BUG-10: consistent "Grade 1/2/3" naming everywhere
  BUG-11: tabular features now correctly loaded and passed during evaluation

Usage:
    python backend/src/evaluate.py --model efficientnet
    python backend/src/evaluate.py --model resnet50
    python backend/src/evaluate.py --model cnn
    python backend/src/evaluate.py --model all     # evaluates all three + prints comparison table
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')          # headless backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SRC_DIR)
DATA_DIR    = os.path.join(BACKEND_DIR, 'data')
MODELS_DIR  = os.path.join(BACKEND_DIR, 'models')
CONFIGS_DIR = os.path.join(BACKEND_DIR, 'configs')
RESULTS_DIR = os.path.join(BACKEND_DIR, 'results')

sys.path.insert(0, SRC_DIR)
from dataset import PeriodontalDataset, get_transforms
from models  import get_model
from metrics import (
    confusion_matrix_np, accuracy_np, per_class_prf, macro_prf,
    macro_f1_np, roc_auc_ovr, classification_report_str
)

CLASS_NAMES = ["Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)"]


# ---------------------------------------------------------------------------
# Verify checkpoint against class_mapping.json
# ---------------------------------------------------------------------------

def verify_checkpoint(model_name, model_path, configs_dir):
    """Print a verification banner before loading weights."""
    class_map_path = os.path.join(configs_dir, 'class_mapping.json')
    norm_path      = os.path.join(configs_dir, 'normalization_config.json')

    if not os.path.exists(class_map_path):
        print(f"  [WARNING] class_mapping.json not found — run data_split.py first")
        class_cfg = {}
    else:
        with open(class_map_path) as f:
            class_cfg = json.load(f)

    if not os.path.exists(norm_path):
        print(f"  [WARNING] normalization_config.json not found")
        norm_cfg = {}
    else:
        with open(norm_path) as f:
            norm_cfg = json.load(f)

    print(f"\n{'─'*50}")
    print(f"  CHECKPOINT VERIFICATION")
    print(f"{'─'*50}")
    print(f"  Model          : {model_name}")
    print(f"  Checkpoint     : {model_path}")
    print(f"  Exists         : {'[OK]' if os.path.exists(model_path) else '[MISSING]'}")
    print(f"  Classes        : {class_cfg.get('class_names', CLASS_NAMES)}")
    print(f"  Num classes    : {class_cfg.get('num_classes', 3)}")
    print(f"  Input size     : {norm_cfg.get('img_size', 224)} × {norm_cfg.get('img_size', 224)}")
    print(f"  Normalization  : mean={norm_cfg.get('mean','(see config)')}")
    print(f"{'─'*50}")


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate(model_name, batch_size=8, verbose=True):
    """
    Evaluate model on the held-out TEST set.

    Returns dict of metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_csv   = os.path.join(DATA_DIR, 'test.csv')
    model_path = os.path.join(MODELS_DIR, f'best_{model_name}.pth')

    if not os.path.exists(test_csv):
        sys.exit(f"[ERROR] test.csv not found: {test_csv}\nRun data_split.py first.")
    if not os.path.exists(model_path):
        sys.exit(f"[ERROR] Checkpoint not found: {model_path}\nRun train.py for {model_name} first.")

    verify_checkpoint(model_name, model_path, CONFIGS_DIR)

    _, val_transform = get_transforms()
    test_dataset = PeriodontalDataset(test_csv, DATA_DIR, transform=val_transform)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)

    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, tabular_features, labels in tqdm(test_loader,
                desc=f"Evaluating {model_name}", leave=False):
            images          = images.to(device)
            tabular_features = tabular_features.to(device)   # BUG-2 fix: pass tabular
            outputs  = model(images, tabular_features)
            probs    = torch.softmax(outputs, dim=1).cpu().numpy()
            _, preds = torch.max(outputs, 1)
            all_preds .extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs .extend(probs.tolist())

    all_preds  = np.array(all_preds,  dtype=int)
    all_labels = np.array(all_labels, dtype=int)
    all_probs  = np.array(all_probs,  dtype=float)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    acc              = accuracy_np(all_labels, all_preds)
    macro_p, macro_r, macro_f = macro_prf(all_labels, all_preds, num_classes=3)
    precision, recall, f1, support = per_class_prf(all_labels, all_preds, num_classes=3)
    auc              = roc_auc_ovr(all_labels, all_probs, num_classes=3)
    cm               = confusion_matrix_np(all_labels, all_preds, num_classes=3)

    # Param count
    n_params = sum(p.numel() for p in model.parameters())

    metrics = {
        'model':          model_name,
        'accuracy':       round(float(acc),     4),
        'macro_precision':round(float(macro_p), 4),
        'macro_recall':   round(float(macro_r), 4),
        'macro_f1':       round(float(macro_f), 4),
        'roc_auc_ovr':    round(float(auc),     4),
        'n_params':       n_params,
        'per_class': {
            CLASS_NAMES[i]: {
                'precision': round(float(precision[i]), 4),
                'recall':    round(float(recall[i]),    4),
                'f1':        round(float(f1[i]),        4),
                'support':   int(support[i]),
            }
            for i in range(3)
        },
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  TEST RESULTS — {model_name.upper()}")
        print(f"{'='*60}")
        print(f"  Accuracy        : {acc:.4f}")
        print(f"  Macro Precision : {macro_p:.4f}")
        print(f"  Macro Recall    : {macro_r:.4f}")
        print(f"  Macro F1        : {macro_f:.4f}")
        print(f"  ROC-AUC (OvR)   : {auc:.4f}")
        print(f"  Parameters      : {n_params:,}")
        print(f"\n  Classification Report:")
        print(classification_report_str(all_labels, all_preds, CLASS_NAMES))
        print(f"\n  Confusion Matrix:\n{cm}")

    # -----------------------------------------------------------------------
    # Save confusion matrix plot
    # -----------------------------------------------------------------------
    fig_dir = os.path.join(RESULTS_DIR, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(fig_dir, f'cm_{model_name}.png')
    plt.savefig(cm_path, dpi=120)
    plt.close()
    if verbose:
        print(f"\n  Saved confusion matrix: {cm_path}")

    # -----------------------------------------------------------------------
    # Save per-model metrics JSON
    # -----------------------------------------------------------------------
    tables_dir = os.path.join(RESULTS_DIR, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    json_path = os.path.join(tables_dir, f'metrics_{model_name}.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        print(f"  Saved metrics JSON  : {json_path}")

    return metrics


# ---------------------------------------------------------------------------
# Comparison table (all models)
# ---------------------------------------------------------------------------

def compare_all_models():
    """Evaluate all available models and print comparison table."""
    model_names = [m for m in ['cnn', 'resnet50', 'efficientnet']
                   if os.path.exists(os.path.join(MODELS_DIR, f'best_{m}.pth'))]
    if not model_names:
        print("[ERROR] No trained checkpoints found in models/.")
        return

    all_metrics = []
    for name in model_names:
        m = evaluate(name, verbose=True)
        all_metrics.append(m)

    # -----------------------------------------------------------------------
    # Model comparison table (Markdown)
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("  FINAL MODEL COMPARISON TABLE  (held-out TEST set only)")
    print(f"{'='*80}")
    header = f"{'Model':<16} {'Params':>12} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'Macro-F1':>10} {'ROC-AUC':>9}"
    print(header)
    print('─' * 80)
    for m in all_metrics:
        print(
            f"{m['model']:<16} "
            f"{m['n_params']:>12,} "
            f"{m['accuracy']:>10.4f} "
            f"{m['macro_precision']:>10.4f} "
            f"{m['macro_recall']:>8.4f} "
            f"{m['macro_f1']:>10.4f} "
            f"{m['roc_auc_ovr']:>9.4f}"
        )

    # Per-class table
    print(f"\n--- Per-Class F1 ---")
    print(f"{'Model':<16} {'Grade':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print('─' * 80)
    for m in all_metrics:
        for grade, pc in m['per_class'].items():
            print(
                f"{m['model']:<16} {grade:<25} "
                f"{pc['precision']:>10.4f} {pc['recall']:>8.4f} "
                f"{pc['f1']:>8.4f} {pc['support']:>9}"
            )
        print()

    # Save combined CSV
    rows = [{
        'Model':     m['model'],
        'Params':    m['n_params'],
        'Accuracy':  m['accuracy'],
        'Precision': m['macro_precision'],
        'Recall':    m['macro_recall'],
        'Macro_F1':  m['macro_f1'],
        'ROC_AUC':   m['roc_auc_ovr'],
    } for m in all_metrics]
    tables_dir = os.path.join(RESULTS_DIR, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(tables_dir, 'model_comparison.csv'), index=False)
    print(f"\n  Saved: results/tables/model_comparison.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate periodontal classifier on test set')
    parser.add_argument('--model', type=str, default='all',
                        choices=['cnn', 'resnet50', 'efficientnet', 'all'],
                        help='"all" evaluates all three and prints comparison table')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    if args.model == 'all':
        compare_all_models()
    else:
        evaluate(args.model, batch_size=args.batch_size, verbose=True)
