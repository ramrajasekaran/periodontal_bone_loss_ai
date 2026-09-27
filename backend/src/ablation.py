"""
ablation.py  —  Multimodal ablation study.

Compares three fusion experiments as required for the research paper:

  Experiment A: Image only      (tabular features forced to zero)
  Experiment B: Tabular only    (TabularMLP — no image backbone)
  Experiment C: Image + tabular (full multimodal — default model)

For each experiment, evaluates all three visual architectures
(CNN, ResNet50, EfficientNet) + the TabularMLP baseline.

Usage:
    # Evaluate existing checkpoints (no retraining needed for A and C):
    python backend/src/ablation.py --mode eval

    # Train TabularMLP baseline (needed for Experiment B):
    python backend/src/ablation.py --mode train_tabular

    # Full ablation pipeline:
    python backend/src/ablation.py --mode full
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SRC_DIR)
DATA_DIR    = os.path.join(BACKEND_DIR, 'data')
MODELS_DIR  = os.path.join(BACKEND_DIR, 'models')
CONFIGS_DIR = os.path.join(BACKEND_DIR, 'configs')
RESULTS_DIR = os.path.join(BACKEND_DIR, 'results')

sys.path.insert(0, SRC_DIR)
from dataset  import PeriodontalDataset, get_transforms
from models   import get_model
from metrics  import accuracy_np, macro_prf, macro_f1_np, classification_report_str

SEED        = 42
CLASS_NAMES = ["Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)"]


# ---------------------------------------------------------------------------
# Core evaluation (one model, one mode)
# ---------------------------------------------------------------------------

def _evaluate_one(model_name, mode, batch_size=8):
    """
    Evaluate a single model on the test set.

    mode:
        'image_only'  — pass zeros for tabular features
        'multimodal'  — pass actual tabular features
        'tabular_only'— pass zeros for images (TabularMLP variant)
    """
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_csv  = os.path.join(DATA_DIR, 'test.csv')
    ckpt_name = f'best_{model_name}.pth'
    if mode == 'tabular_only':
        ckpt_name = 'best_tabular_mlp.pth'
        model_name = 'tabular_mlp'

    ckpt_path = os.path.join(MODELS_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        return None

    _, val_transform = get_transforms()
    dataset = PeriodontalDataset(test_csv, DATA_DIR, transform=val_transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, tabular_features, labels in tqdm(
                loader, desc=f"  {model_name}/{mode}", leave=False):
            images          = images.to(device)
            tabular_features = tabular_features.to(device)

            if mode == 'image_only':
                # Force tabular to zeros — model falls back to visual branch only
                tabular_features = torch.zeros_like(tabular_features)
            elif mode == 'tabular_only':
                # TabularMLP ignores images; pass through unchanged
                pass

            outputs = model(images, tabular_features)
            _, preds = torch.max(outputs, 1)
            all_preds .extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    acc              = accuracy_np(all_labels, all_preds)
    macro_p, macro_r, macro_f = macro_prf(all_labels, all_preds, num_classes=3)

    return {
        'model':     model_name if mode != 'tabular_only' else 'tabular_mlp',
        'mode':      mode,
        'accuracy':  round(float(acc),     4),
        'precision': round(float(macro_p), 4),
        'recall':    round(float(macro_r), 4),
        'macro_f1':  round(float(macro_f), 4),
    }


# ---------------------------------------------------------------------------
# Train TabularMLP baseline (Experiment B)
# ---------------------------------------------------------------------------

def train_tabular_baseline(epochs=80, batch_size=32, lr=5e-4, patience=15):
    """Train a tabular-only MLP on the training set (Experiment B baseline)."""
    import torch.nn as nn
    import torch.optim as optim
    from train import compute_class_weights, set_seed

    set_seed(SEED)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    val_csv   = os.path.join(DATA_DIR, 'val.csv')

    print("\n--- Training TabularMLP (Experiment B — tabular only) ---")

    _, val_transform = get_transforms()
    train_dataset = PeriodontalDataset(train_csv, DATA_DIR, transform=val_transform)
    val_dataset   = PeriodontalDataset(val_csv,   DATA_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(train_csv).to(device)
    model     = get_model('tabular_mlp').to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    best_f1 = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for images, tabular_features, labels in train_loader:
            tabular_features = tabular_features.to(device)
            labels           = labels.to(device)
            optimizer.zero_grad()
            # TabularMLP ignores images; pass None-compatible dummy
            outputs = model(images.to(device), tabular_features)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, tabular_features, labels in val_loader:
                tabular_features = tabular_features.to(device)
                outputs = model(images.to(device), tabular_features)
                _, preds = torch.max(outputs, 1)
                all_preds .extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())

        val_f1 = macro_f1_np(all_labels, all_preds, num_classes=3)
        scheduler.step(val_f1)
        print(f"  Epoch {epoch+1:3d}: Macro-F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_tabular_mlp.pth'))
            print(f"    [OK] Saved TabularMLP (F1={val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\n  TabularMLP best Macro-F1: {best_f1:.4f}")


# ---------------------------------------------------------------------------
# Full ablation evaluation
# ---------------------------------------------------------------------------

def run_ablation_eval():
    """
    Evaluate all available models in each mode and print comparison table.
    """
    visual_models = ['cnn', 'resnet50', 'efficientnet']
    results       = []

    print("\n" + "=" * 60)
    print("ABLATION STUDY  —  Test Set Results")
    print("=" * 60)

    for model_name in visual_models:
        ckpt = os.path.join(MODELS_DIR, f'best_{model_name}.pth')
        if not os.path.exists(ckpt):
            print(f"[SKIP] {model_name}: no checkpoint found")
            continue

        for mode in ['image_only', 'multimodal']:
            r = _evaluate_one(model_name, mode)
            if r:
                results.append(r)
                label = 'Exp-A (Image only)' if mode == 'image_only' else 'Exp-C (Multimodal)'
                print(f"  {model_name:<15} {label:<22}: Acc={r['accuracy']:.4f}  Macro-F1={r['macro_f1']:.4f}")

    # Tabular-only (Experiment B)
    tab_ckpt = os.path.join(MODELS_DIR, 'best_tabular_mlp.pth')
    if os.path.exists(tab_ckpt):
        r = _evaluate_one('tabular_mlp', 'tabular_only')
        if r:
            results.append(r)
            print(f"  {'TabularMLP':<15} {'Exp-B (Tabular only)':<22}: Acc={r['accuracy']:.4f}  Macro-F1={r['macro_f1']:.4f}")
    else:
        print("\n  [INFO] Tabular-only checkpoint not found.")
        print("         Run:  python backend/src/ablation.py --mode train_tabular")
        print("         Then: python backend/src/ablation.py --mode eval")

    if not results:
        print("\n  No results — run train.py for each model first.")
        return

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'─'*80}")
    print(f"{'Model':<15} {'Experiment':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'Macro-F1':>10}")
    print(f"{'─'*80}")
    exp_label = {
        'image_only':   'Exp-A: Image only',
        'tabular_only': 'Exp-B: Tabular only',
        'multimodal':   'Exp-C: Multimodal',
    }
    for r in results:
        print(
            f"{r['model']:<15} {exp_label.get(r['mode'], r['mode']):<25} "
            f"{r['accuracy']:>10.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>8.4f} {r['macro_f1']:>10.4f}"
        )

    # Save
    os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)
    ablation_path = os.path.join(RESULTS_DIR, 'tables', 'ablation_study.csv')
    pd.DataFrame(results).to_csv(ablation_path, index=False)
    print(f"\n  Saved: results/tables/ablation_study.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation study: image vs tabular vs multimodal')
    parser.add_argument('--mode', default='eval',
                        choices=['eval', 'train_tabular', 'full'],
                        help=(
                            'eval         — evaluate existing checkpoints\n'
                            'train_tabular— train TabularMLP (Exp B)\n'
                            'full         — train tabular then evaluate all'
                        ))
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    if args.mode in ('train_tabular', 'full'):
        train_tabular_baseline()

    if args.mode in ('eval', 'full'):
        run_ablation_eval()
