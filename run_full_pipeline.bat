@echo off
echo ============================================================
echo STARTING FULL PIPELINE RUN
echo ============================================================

echo.
echo [1/6] Training CNN...
python backend/src/train.py --model cnn --epochs 50 --batch_size 16

echo.
echo [2/6] Training ResNet50...
python backend/src/train.py --model resnet50 --epochs 50 --batch_size 8

echo.
echo [3/6] Training EfficientNet-B0...
python backend/src/train.py --model efficientnet --epochs 50 --batch_size 8

echo.
echo [4/6] Evaluating all models on TEST set...
python backend/src/evaluate.py --model all

echo.
echo [5/6] Training TabularMLP (for Ablation Experiment B)...
python backend/src/ablation.py --mode train_tabular

echo.
echo [6/6] Running full Ablation Study...
python backend/src/ablation.py --mode eval

echo.
echo ============================================================
echo PIPELINE COMPLETE! Results saved to backend/results/
echo ============================================================
pause
