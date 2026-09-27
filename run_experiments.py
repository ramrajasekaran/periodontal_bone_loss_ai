import os
import shutil
import subprocess
import pandas as pd

def copy_baseline(model_name="resnet50"):
    print("============================================================")
    print("EXPERIMENT 1: BASELINE (Copying existing artifacts)")
    print("============================================================")
    
    base_dir = "backend"
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    
    # 1. Copy model checkpoint
    src_model = os.path.join(models_dir, f"best_{model_name}.pth")
    dst_model = os.path.join(models_dir, f"best_{model_name}_exp1.pth")
    if os.path.exists(src_model):
        shutil.copy(src_model, dst_model)
        print(f"Copied {src_model} to {dst_model}")
        
    # 2. Copy history
    src_hist = os.path.join(results_dir, f"history_{model_name}.json")
    dst_hist = os.path.join(results_dir, f"history_{model_name}_exp1.json")
    if os.path.exists(src_hist):
        shutil.copy(src_hist, dst_hist)
        print(f"Copied {src_hist} to {dst_hist}")

def run_experiment(exp_num, model_name="resnet50"):
    print("\n============================================================")
    print(f"TRAINING EXPERIMENT {exp_num}: {model_name.upper()}")
    print("============================================================")
    
    # Run training
    cmd_train = ["python", "backend/src/train.py", "--model", model_name, "--experiment", str(exp_num), "--epochs", "50", "--batch_size", "8"]
    subprocess.run(cmd_train, check=True)

def evaluate_all():
    print("\n============================================================")
    print("EVALUATING ALL EXPERIMENTS")
    print("============================================================")
    for exp_num in range(1, 5):
        cmd_eval = ["python", "backend/src/detailed_evaluate.py", "--experiment", str(exp_num)]
        subprocess.run(cmd_eval, check=True)

def compare_results():
    print("\n============================================================")
    print("COMPARISON OF RESNET50 EXPERIMENTS")
    print("============================================================")
    
    results = []
    
    for exp_num in range(1, 5):
        report_path = f"backend/results/tables/classification_report_resnet50_exp{exp_num}.csv"
        if not os.path.exists(report_path):
            continue
            
        df = pd.read_csv(report_path)
        
        g1 = df[df['class'] == 'Grade 1 (Mild)'].iloc[0]
        g2 = df[df['class'] == 'Grade 2 (Moderate)'].iloc[0]
        g3 = df[df['class'] == 'Grade 3 (Severe)'].iloc[0]
        acc = df[df['class'] == 'accuracy'].iloc[0]['f1-score']
        macro = df[df['class'] == 'macro avg'].iloc[0]
        
        results.append({
            'Experiment': f"Exp {exp_num}",
            'G1 Recall': g1['recall'],
            'G2 Recall': g2['recall'],
            'G3 Recall': g3['recall'],
            'Macro F1': macro['f1-score'],
            'Accuracy': acc
        })
        
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    comparison_path = "backend/results/tables/experiment_comparison_resnet50.csv"
    df_results.to_csv(comparison_path, index=False)
    print(f"\nSaved comparison to {comparison_path}")

if __name__ == "__main__":
    copy_baseline("resnet50")
    run_experiment(2, "resnet50")
    run_experiment(3, "resnet50")
    run_experiment(4, "resnet50")
    evaluate_all()
    compare_results()
