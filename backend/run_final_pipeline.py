"""
run_final_pipeline.py — Master pipeline: re-split data then train all 3 models.

Usage (from project root):
    python periodontal_bone_loss_ai/backend/run_final_pipeline.py
"""
import os
import sys
import subprocess

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
SRC_DIR     = os.path.join(THIS_DIR, 'src')
PYTHON      = sys.executable

def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"STEP: {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Step failed: {desc}")
        sys.exit(result.returncode)
    print(f"[OK] {desc} completed.")

def main():
    print("\n" + "="*60)
    print("PERIODONTAL BONE-LOSS AI — FINAL PIPELINE")
    print("Goal: >= 85% accuracy, all models agree on same image")
    print("="*60)

    # Step 1: Re-split data to 80/10/10
    run(
        f'"{PYTHON}" "{os.path.join(SRC_DIR, "data_split.py")}"',
        "Data re-split (80% train / 10% val / 10% test)"
    )

    # Step 2: Train ResNet50 (best baseline model)
    run(
        f'"{PYTHON}" "{os.path.join(SRC_DIR, "train_final.py")}" --model resnet50 --epochs 80 --batch_size 8 --patience 15',
        "Training ResNet50 (final)"
    )

    # Step 3: Train EfficientNet-B0
    run(
        f'"{PYTHON}" "{os.path.join(SRC_DIR, "train_final.py")}" --model efficientnet --epochs 80 --batch_size 8 --patience 15',
        "Training EfficientNet-B0 (final)"
    )

    # Step 4: Train Custom CNN
    run(
        f'"{PYTHON}" "{os.path.join(SRC_DIR, "train_final.py")}" --model cnn --epochs 80 --batch_size 16 --patience 15',
        "Training Custom CNN (final)"
    )

    print("\n" + "="*60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("Checkpoints saved as:")
    print("  models/best_resnet50_final.pth")
    print("  models/best_efficientnet_final.pth")
    print("  models/best_cnn_final.pth")
    print("\nNow run evaluation:")
    print(f'  python "{os.path.join(SRC_DIR, "evaluate.py")}"')
    print("="*60)

if __name__ == '__main__':
    main()
