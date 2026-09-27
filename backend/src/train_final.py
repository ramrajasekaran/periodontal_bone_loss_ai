"""
train_final.py — FINAL optimized training for all 3 models.

Strategy:
  1. 80/10/10 data split (more training data)
  2. WeightedRandomSampler (balanced batches)
  3. Focal Loss (gamma=2) + class weights (hard-sample focus)
  4. Heavy augmentation (better generalization)
  5. CosineAnnealingWarmRestarts LR scheduler
  6. Gradient clipping (stable training)
  7. Best checkpoint saved by Val Macro-F1

Usage (from project root):
    python periodontal_bone_loss_ai/backend/src/train_final.py --model resnet50
    python periodontal_bone_loss_ai/backend/src/train_final.py --model efficientnet
    python periodontal_bone_loss_ai/backend/src/train_final.py --model cnn
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

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SRC_DIR)
DATA_DIR    = os.path.join(BACKEND_DIR, 'data')
MODELS_DIR  = os.path.join(BACKEND_DIR, 'models')
CONFIGS_DIR = os.path.join(BACKEND_DIR, 'configs')
RESULTS_DIR = os.path.join(BACKEND_DIR, 'results')

sys.path.insert(0, SRC_DIR)
from dataset import PeriodontalDataset, get_heavy_transforms
from models  import get_model
from losses  import FocalLoss
from metrics import macro_f1_np, accuracy_np, classification_report_str

SEED        = 42
CLASS_NAMES = ["Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)"]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def compute_class_weights(train_csv, num_classes=3):
    import pandas as pd
    df     = pd.read_csv(train_csv)
    counts = np.zeros(num_classes, dtype=float)
    for level in range(1, num_classes + 1):
        counts[level - 1] = (df['Level'] == level).sum()
    total   = counts.sum()
    weights = total / (num_classes * counts)
    weights = weights / weights.min()
    print(f"  Grade1={int(counts[0])}, Grade2={int(counts[1])}, Grade3={int(counts[2])}")
    print(f"  Class weights : {[round(w, 3) for w in weights.tolist()]}")
    return torch.tensor(weights, dtype=torch.float32)


def train(model_name, epochs=80, batch_size=8, lr=None, patience=15):
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"FINAL TRAINING: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Device  : {device}")

    class_map_path = os.path.join(CONFIGS_DIR, 'class_mapping.json')
    if not os.path.exists(class_map_path):
        sys.exit("[ERROR] configs/class_mapping.json not found. Run data_split.py first.")

    if lr is None:
        lr = 3e-4 if model_name == 'cnn' else 5e-5

    train_transform, val_transform = get_heavy_transforms()
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    val_csv   = os.path.join(DATA_DIR, 'val.csv')

    print("\n--- Class Distribution & Weights ---")
    class_weights = compute_class_weights(train_csv)

    train_dataset = PeriodontalDataset(train_csv, DATA_DIR, transform=train_transform)
    val_dataset   = PeriodontalDataset(val_csv,   DATA_DIR, transform=val_transform)

    labels_np      = train_dataset.data_frame['Level'].to_numpy(dtype=int) - 1
    sample_weights = [class_weights[l].item() for l in labels_np]
    sampler        = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_dataset), replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False,  num_workers=0, pin_memory=False)

    model     = get_model(model_name).to(device)
    cw        = class_weights.to(device)
    criterion = FocalLoss(alpha=cw, gamma=2.0)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    print(f"\n--- Training Configuration ---")
    print(f"  LR             : {lr}  (CosineAnnealingWarmRestarts T_0=10, T_mult=2)")
    print(f"  Batch size     : {batch_size}")
    print(f"  Max epochs     : {epochs}")
    print(f"  Early stopping : patience={patience}")
    print(f"  Loss           : FocalLoss(gamma=2, alpha=class_weights)")
    print(f"  Sampler        : WeightedRandomSampler")
    print(f"  Augmentation   : Heavy")

    best_val_f1       = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, tab, labels in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            images, tab, labels = images.to(device), tab.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, tab)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step(epoch)
        epoch_loss = running_loss / len(train_dataset)
        history['train_loss'].append(epoch_loss)

        model.eval()
        val_loss   = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, tab, labels in tqdm(
                    val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                images, tab, labels = images.to(device), tab.to(device), labels.to(device)
                outputs  = model(images, tab)
                loss     = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds .extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        epoch_val_loss = val_loss / len(val_dataset)
        val_acc = accuracy_np(all_labels, all_preds)
        val_f1  = macro_f1_np(all_labels, all_preds, num_classes=3)
        unique_pred = sorted(set(all_preds))
        cur_lr      = optimizer.param_groups[0]['lr']

        history['val_loss'].append(epoch_val_loss)
        history['val_acc'] .append(val_acc)
        history['val_f1']  .append(val_f1)

        print(
            f"Epoch {epoch+1:3d}/{epochs}  "
            f"TrainLoss={epoch_loss:.4f}  ValLoss={epoch_val_loss:.4f}  "
            f"ValAcc={val_acc:.4f}  ValMacroF1={val_f1:.4f}  "
            f"PredClasses={unique_pred}  LR={cur_lr:.2e}"
        )

        if len(unique_pred) == 1:
            print(f"  [WARNING] Mode collapse: only predicting class {unique_pred[0]}")

        if val_f1 > best_val_f1:
            best_val_f1       = val_f1
            epochs_no_improve = 0
            save_path = os.path.join(MODELS_DIR, f'best_{model_name}_final.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  [SAVED] best_{model_name}_final.pth  (Macro-F1={val_f1:.4f})")
            print(classification_report_str(all_labels, all_preds, CLASS_NAMES))
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}.")
                break

    hist_path = os.path.join(RESULTS_DIR, f'history_{model_name}_final.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n[DONE] {model_name.upper()}  Best Val Macro-F1 = {best_val_f1:.4f}")
    return best_val_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str, default='resnet50',
                        choices=['cnn', 'resnet50', 'efficientnet'])
    parser.add_argument('--epochs',     type=int,   default=80)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--lr',         type=float, default=None)
    parser.add_argument('--patience',   type=int,   default=15)
    args = parser.parse_args()
    train(model_name=args.model, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, patience=args.patience)
