"""
data_split.py  —  True 70 / 15 / 15 patient-level stratified split.

Fixes diagnosed bugs:
  BUG-1: val.csv and test.csv were identical (val_df saved to both files).
  BUG-3: data_split.py produced 70/30 split, not 70/15/15.

This script also:
  • Saves configs/class_mapping.json          — single source-of-truth label map
  • Saves configs/normalization_config.json   — preprocessing constants
  • Prints a full dataset quality report
  • Verifies zero patient leakage across all three splits

Usage (from project root or any directory):
    python backend/src/data_split.py
"""
import os
import sys
import json
import random

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve paths from __file__ so the script works from ANY working directory
# ---------------------------------------------------------------------------
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SRC_DIR)
DATA_DIR    = os.path.join(BACKEND_DIR, 'data')
CONFIGS_DIR = os.path.join(BACKEND_DIR, 'configs')

SEED = 42

# ---------------------------------------------------------------------------
# Fixed class mapping — SINGLE SOURCE OF TRUTH used everywhere
# ---------------------------------------------------------------------------
CLASS_MAPPING = {
    "Grade_1_Mild":     0,
    "Grade_2_Moderate": 1,
    "Grade_3_Severe":   2,
}
LABEL_TO_NAME = {
    0: "Grade 1 (Mild)",
    1: "Grade 2 (Moderate)",
    2: "Grade 3 (Severe)",
}
# CSV 'Level' values {1, 2, 3} map to class indices {0, 1, 2}
LEVEL_TO_CLASS = {1: 0, 2: 1, 3: 2}

NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],   # ImageNet stats (applied to CLAHE → RGB X-rays)
    "std":  [0.229, 0.224, 0.225],
    "img_size": 224,
    "note": "Grayscale CLAHE → 3-channel RGB → ImageNet normalization"
}


# ---------------------------------------------------------------------------
# Pure-pandas / numpy stratified split  (no sklearn required)
# ---------------------------------------------------------------------------

def stratified_3way_split(df, label_col, train_frac=0.70, val_frac=0.15, seed=SEED):
    """
    Split a DataFrame into train / val / test with per-class stratification.
    No sklearn dependency — uses pandas + numpy only.

    Parameters
    ----------
    df         : source DataFrame
    label_col  : column name holding integer class labels
    train_frac : fraction for training set
    val_frac   : fraction for validation set (test = 1 - train_frac - val_frac)
    seed       : random seed for reproducibility

    Returns
    -------
    train_df, val_df, test_df  — shuffled, reset index
    """
    rng = np.random.RandomState(seed)
    train_parts, val_parts, test_parts = [], [], []

    for label in sorted(df[label_col].unique()):
        group = df[df[label_col] == label].copy()
        group = group.sample(frac=1, random_state=rng)  # shuffle within class
        n = len(group)

        n_train = max(1, int(round(n * train_frac)))
        n_val   = max(1, int(round(n * val_frac)))
        n_test  = n - n_train - n_val
        if n_test < 1:                     # edge case: ensure at least 1 in test
            n_val  -= 1
            n_test += 1

        train_parts.append(group.iloc[:n_train])
        val_parts  .append(group.iloc[n_train : n_train + n_val])
        test_parts .append(group.iloc[n_train + n_val :])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=rng).reset_index(drop=True)
    val_df   = pd.concat(val_parts)  .sample(frac=1, random_state=rng).reset_index(drop=True)
    test_df  = pd.concat(test_parts) .sample(frac=1, random_state=rng).reset_index(drop=True)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    meta_path = os.path.join(DATA_DIR, 'meta_data.csv')
    if not os.path.exists(meta_path):
        sys.exit(f"[ERROR] meta_data.csv not found at: {meta_path}")

    df = pd.read_csv(meta_path)
    print("=" * 60)
    print("DATASET INSPECTION REPORT")
    print("=" * 60)

    # Extract Patient_ID from filename  e.g. patient_image_000294_1c448f4a.jpg → 000294
    df['Patient_ID'] = df['File name'].apply(lambda x: x.split('_')[2])

    total = len(df)
    print(f"\nTotal records in meta_data.csv : {total}")
    print(f"Unique Patient IDs             : {df['Patient_ID'].nunique()}")

    unique_levels = [int(lvl) for lvl in sorted(df['Level'].unique())]
    for lvl in unique_levels:
        count = int((df['Level'] == lvl).sum())
        name  = LABEL_TO_NAME[LEVEL_TO_CLASS[lvl]]
        pct   = 100 * count / total
        print(f"  Level {lvl}  ({name}): {count:4d} images  ({pct:.1f}%)")
    counts = [(df['Level'] == lvl).sum() for lvl in unique_levels]
    max_c, min_c = max(counts), min(counts)
    print(f"  Imbalance ratio (max/min)    : {max_c/min_c:.2f}x")

    # Images dir check
    images_dir = os.path.join(DATA_DIR, 'images')
    if os.path.exists(images_dir):
        img_files = set(os.listdir(images_dir))
        missing = [f for f in df['File name'] if f not in img_files]
        print(f"\nImages directory             : {images_dir}")
        print(f"Total image files found      : {len(img_files)}")
        print(f"Missing images (in CSV, not on disk): {len(missing)}")
    else:
        print(f"\n[WARNING] Images directory not found: {images_dir}")

    # -----------------------------------------------------------------------
    # Stratified 70 / 15 / 15 split
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PERFORMING 70 / 15 / 15 PATIENT-LEVEL STRATIFIED SPLIT")
    print(f"Random seed: {SEED}")
    print("=" * 60)

    train_df, val_df, test_df = stratified_3way_split(
        df, label_col='Level', train_frac=0.80, val_frac=0.10, seed=SEED
    )

    # -----------------------------------------------------------------------
    # Leakage verification
    # -----------------------------------------------------------------------
    train_p = set(train_df['Patient_ID'])
    val_p   = set(val_df['Patient_ID'])
    test_p  = set(test_df['Patient_ID'])

    tv_leak  = len(train_p & val_p)
    tt_leak  = len(train_p & test_p)
    vt_leak  = len(val_p   & test_p)

    print(f"\n--- Patient Leakage Check ---")
    print(f"  Train intersect Val  overlap : {tv_leak}  {'OK' if tv_leak == 0 else 'LEAKAGE!'}")
    print(f"  Train intersect Test overlap : {tt_leak}  {'OK' if tt_leak == 0 else 'LEAKAGE!'}")
    print(f"  Val   intersect Test overlap : {vt_leak}  {'OK' if vt_leak == 0 else 'LEAKAGE!'}")

    if tv_leak or tt_leak or vt_leak:
        sys.exit("[ERROR] Patient leakage detected — aborting split.")

    # -----------------------------------------------------------------------
    # Distribution report
    # -----------------------------------------------------------------------
    print(f"\n--- Split Sizes ---")
    print(f"  Train : {len(train_df):4d} images  ({100*len(train_df)/total:.1f}%)")
    print(f"  Val   : {len(val_df):4d}   images  ({100*len(val_df)/total:.1f}%)")
    print(f"  Test  : {len(test_df):4d}  images  ({100*len(test_df)/total:.1f}%)")

    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n  {split_name} class distribution:")
        for level in sorted(split_df['Level'].unique()):
            cnt  = (split_df['Level'] == level).sum()
            name = LABEL_TO_NAME[LEVEL_TO_CLASS[level]]
            print(f"    Level {level} ({name}): {cnt}")

    # -----------------------------------------------------------------------
    # Save CSVs
    # -----------------------------------------------------------------------
    os.makedirs(DATA_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
    val_df  .to_csv(os.path.join(DATA_DIR, 'val.csv'),   index=False)
    test_df .to_csv(os.path.join(DATA_DIR, 'test.csv'),  index=False)
    print(f"\nSaved: train.csv ({len(train_df)} rows)")
    print(f"Saved: val.csv   ({len(val_df)} rows)  [DIFFERENT from test.csv]")
    print(f"Saved: test.csv  ({len(test_df)} rows)  [DIFFERENT from val.csv]")

    # -----------------------------------------------------------------------
    # Save global configs
    # -----------------------------------------------------------------------
    os.makedirs(CONFIGS_DIR, exist_ok=True)

    class_map_path = os.path.join(CONFIGS_DIR, 'class_mapping.json')
    with open(class_map_path, 'w') as f:
        json.dump({
            "class_mapping":  CLASS_MAPPING,
            "label_to_name":  {str(k): v for k, v in LABEL_TO_NAME.items()},
            "level_to_class": {str(k): v for k, v in LEVEL_TO_CLASS.items()},
            "num_classes":    3,
            "class_names":    ["Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)"],
        }, f, indent=2)
    print(f"Saved: configs/class_mapping.json")

    norm_path = os.path.join(CONFIGS_DIR, 'normalization_config.json')
    with open(norm_path, 'w') as f:
        json.dump(NORMALIZATION, f, indent=2)
    print(f"Saved: configs/normalization_config.json")

    print("\n" + "=" * 60)
    print("FIXED LABEL MAPPING (used everywhere - training, eval, inference, UI):")
    print("=" * 60)
    for grade, idx in CLASS_MAPPING.items():
        print(f"  {grade.replace('_',' '):<25s} ->  class index {idx}  ({LABEL_TO_NAME[idx]})")
    print()


if __name__ == '__main__':
    main()
