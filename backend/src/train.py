"""
train.py  —  Fixed training pipeline for all three model architectures.

Bugs fixed:
  BUG-4: Removed sklearn imports (blocked by WDAC via scipy DLL)
  BUG-5: Removed triple imbalance stacking; uses CrossEntropyLoss(weight) only
  BUG-6: Replaced OneCycleLR (incompatible with early stopping) with ReduceLROnPlateau
  BUG-7: Now saves class_mapping.json, normalization_config.json, training_config.json
  BUG-9: sys.path now uses __file__-relative insert

Usage:
    python backend/src/train.py --model efficientnet --epochs 50 --batch_size 8
    python backend/src/train.py --model resnet50     --epochs 50 --batch_size 8
    python backend/src/train.py --model cnn          --epochs 50 --batch_size 16
"""
import os
import sys
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve paths from __file__ (works from any CWD)
# ---------------------------------------------------------------------------
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SRC_DIR)
DATA_DIR    = os.path.join(BACKEND_DIR, 'data')
MODELS_DIR  = os.path.join(BACKEND_DIR, 'models')
CONFIGS_DIR = os.path.join(BACKEND_DIR, 'configs')
RESULTS_DIR = os.path.join(BACKEND_DIR, 'results')

sys.path.insert(0, SRC_DIR)
from dataset import PeriodontalDataset, get_transforms, get_heavy_transforms
from models  import get_model
from losses  import FocalLoss
from metrics import (
    macro_f1_np, per_class_prf, accuracy_np,
    classification_report_str, confusion_matrix_np
)

SEED = 42
CLASS_NAMES = ["Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Class weights from training CSV  (inverse-frequency, no sklearn)
# ---------------------------------------------------------------------------

def compute_class_weights(train_csv, num_classes=3):
    """
    Inverse-frequency weights from 'Level' column.
    Level {1,2,3} -> class index {0,1,2}.
    Returns tensor of shape (num_classes,).
    """
    import pandas as pd
    df = pd.read_csv(train_csv)
    counts = np.zeros(num_classes, dtype=float)
    for level in range(1, num_classes + 1):
        counts[level - 1] = (df['Level'] == level).sum()
    total   = counts.sum()
    weights = total / (num_classes * counts)      # inverse frequency
    weights = weights / weights.min()             # normalise so minimum weight = 1.0
    print(f"  Class distribution  : Grade1={int(counts[0])}, Grade2={int(counts[1])}, Grade3={int(counts[2])}")
    print(f"  Class weights       : {[round(w, 3) for w in weights.tolist()]}")
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Save training artefacts
# ---------------------------------------------------------------------------

def save_training_artefacts(model_name, model, lr, epochs, batch_size):
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)

    # Load class mapping (created by data_split.py)
    class_map_path = os.path.join(CONFIGS_DIR, 'class_mapping.json')
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(
            f"class_mapping.json not found at {class_map_path}.\n"
            "Run data_split.py first."
        )

    # Load normalization config
    norm_path = os.path.join(CONFIGS_DIR, 'normalization_config.json')
    with open(norm_path) as f:
        norm_cfg = json.load(f)

    # Training config (per-model)
    training_cfg = {
        "model_name":    model_name,
        "seed":          SEED,
        "lr":            lr,
        "epochs":        epochs,
        "batch_size":    batch_size,
        "optimizer":     "AdamW",
        "scheduler":     "ReduceLROnPlateau(mode=max, factor=0.5, patience=4)",
        "loss":          "CrossEntropyLoss(class_weights)",
        "early_stopping_patience": 12,
        "img_size":      norm_cfg["img_size"],
        "normalization": norm_cfg,
        "class_mapping_file": "configs/class_mapping.json",
        "checkpoint":    f"models/best_{model_name}.pth",
    }
    cfg_path = os.path.join(CONFIGS_DIR, f'training_config_{model_name}.json')
    with open(cfg_path, 'w') as f:
        json.dump(training_cfg, f, indent=2)

    return training_cfg


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model_name, epochs=50, batch_size=8, lr=None, patience=12, exp_num=1):
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name.upper()} (Experiment {exp_num})")
    print(f"{'='*60}")
    print(f"Device  : {device}")

    # -----------------------------------------------------------------------
    # Check configs exist (data_split.py must run first)
    # -----------------------------------------------------------------------
    class_map_path = os.path.join(CONFIGS_DIR, 'class_mapping.json')
    if not os.path.exists(class_map_path):
        sys.exit(
            "[ERROR] configs/class_mapping.json not found.\n"
            "Run: python backend/src/data_split.py  first."
        )

    # -----------------------------------------------------------------------
    # Learning rate defaults
    # -----------------------------------------------------------------------
    if lr is None:
        lr = 5e-4 if model_name == 'cnn' else 1e-4

    # -----------------------------------------------------------------------
    # Transforms
    # -----------------------------------------------------------------------
    if exp_num == 3:
        train_transform, val_transform = get_heavy_transforms()
    else:
        train_transform, val_transform = get_transforms()

    # -----------------------------------------------------------------------
    # CSV paths
    # -----------------------------------------------------------------------
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    val_csv   = os.path.join(DATA_DIR, 'val.csv')

    # -----------------------------------------------------------------------
    # Class weights (inverse-frequency)
    # -----------------------------------------------------------------------
    print("\n--- Class Weights (inverse-frequency) ---")
    class_weights = compute_class_weights(train_csv)
    
    # -----------------------------------------------------------------------
    # Datasets & DataLoaders
    # -----------------------------------------------------------------------
    train_dataset = PeriodontalDataset(train_csv, DATA_DIR, transform=train_transform)
    val_dataset   = PeriodontalDataset(val_csv,   DATA_DIR, transform=val_transform)

    if exp_num in [2, 3]:
        # WeightedRandomSampler for balanced batches
        # Assign a weight to each sample in the dataset based on its class
        labels_series = train_dataset.data_frame['Level'].to_numpy(dtype=int) - 1
        sample_weights = [class_weights[l].item() for l in labels_series]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=False)
    else:
        # Standard random shuffle
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    class_weights = class_weights.to(device)

    # -----------------------------------------------------------------------
    # Model, loss, optimizer, scheduler
    # -----------------------------------------------------------------------
    model = get_model(model_name).to(device)
    
    if exp_num == 4:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        loss_desc = "FocalLoss(gamma=2.0, alpha=class_weights)"
    elif exp_num in [2, 3]:
        # If using WeightedRandomSampler, the batches are already balanced, so we don't weight the loss
        criterion = nn.CrossEntropyLoss()
        loss_desc = "CrossEntropyLoss(unweighted due to WeightedRandomSampler)"
    else:
        # Experiment 1 (Baseline)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        loss_desc = "CrossEntropyLoss(class_weights)"
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    # ReduceLROnPlateau is compatible with early stopping (no fixed total steps)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-6
    )

    # -----------------------------------------------------------------------
    # Save training config
    # -----------------------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    training_cfg = save_training_artefacts(model_name, model, lr, epochs, batch_size)
    print(f"\n--- Training Configuration ---")
    print(f"  LR             : {lr}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Max epochs     : {epochs}")
    print(f"  Early stopping : patience={patience}")
    print(f"  Loss           : {loss_desc}")
    print(f"  Scheduler      : ReduceLROnPlateau(max, factor=0.5, patience=4)")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_f1    = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(epochs):
        # -- Train --
        model.train()
        running_loss = 0.0
        for images, tabular_features, labels in tqdm(train_loader,
                desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            images, tabular_features, labels = (
                images.to(device), tabular_features.to(device), labels.to(device)
            )
            optimizer.zero_grad()
            outputs = model(images, tabular_features)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        history['train_loss'].append(epoch_loss)

        # -- Validate --
        model.eval()
        val_loss   = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, tabular_features, labels in tqdm(val_loader,
                    desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                images, tabular_features, labels = (
                    images.to(device), tabular_features.to(device), labels.to(device)
                )
                outputs = model(images, tabular_features)
                loss    = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds .extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        epoch_val_loss = val_loss / len(val_dataset)
        val_acc = accuracy_np(all_labels, all_preds)
        val_f1  = macro_f1_np(all_labels, all_preds, num_classes=3)
        unique_predicted = sorted(set(all_preds))

        history['val_loss'].append(epoch_val_loss)
        history['val_acc'] .append(val_acc)
        history['val_f1']  .append(val_f1)

        cur_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1:3d}/{epochs}  "
            f"TrainLoss={epoch_loss:.4f}  ValLoss={epoch_val_loss:.4f}  "
            f"ValAcc={val_acc:.4f}  ValMacroF1={val_f1:.4f}  "
            f"PredClasses={unique_predicted}  LR={cur_lr:.2e}"
        )

        # Detect mode collapse (model only predicting one class)
        if len(unique_predicted) == 1:
            print(f"  [WARNING] Mode collapse detected: model predicting only class {unique_predicted[0]}")

        scheduler.step(val_f1)

        # -- Early stopping & best-model save --
        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            epochs_no_improve = 0
            os.makedirs(MODELS_DIR, exist_ok=True)
            model_save_name = f'best_{model_name}_exp{exp_num}.pth'
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, model_save_name))
            print(f"  [OK] Saved best model {model_save_name} (Macro-F1={val_f1:.4f})")
            report = classification_report_str(all_labels, all_preds, CLASS_NAMES)
            print(report)
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping triggered at epoch {epoch+1}.")
                break

    # -- Save training history --
    hist_path = os.path.join(RESULTS_DIR, f'history_{model_name}_exp{exp_num}.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n[OK] Training complete.  Best Macro-F1 = {best_val_f1:.4f}")
    print(f"   Best model saved to : models/best_{model_name}_exp{exp_num}.pth")
    print(f"   Training history    : results/history_{model_name}_exp{exp_num}.json")
    print(f"   Training config     : configs/training_config_{model_name}.json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train periodontal bone-loss classifier')
    parser.add_argument('--model', type=str, default='efficientnet',
                        choices=['cnn', 'resnet50', 'efficientnet', 'tabular_mlp'],
                        help='Model architecture to train')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--lr',         type=float, default=None,
                        help='Learning rate (default: 5e-4 for cnn, 1e-4 for pretrained)')
    parser.add_argument('--patience',   type=int,   default=12)
    parser.add_argument('--experiment', type=int,   default=1, choices=[1, 2, 3, 4],
                        help='Experiment number (1-4)')
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        exp_num=args.experiment
    )
