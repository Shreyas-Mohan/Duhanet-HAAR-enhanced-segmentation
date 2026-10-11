#!/bin/bash
# ==============================================================================
# DUHA-Net: 100-Epoch AWS EC2 GPU Training, Evaluation, and Auto-Shutdown Script
# ==============================================================================
set -e

# --- 1. Detect and Activate Python / PyTorch Environment ---
if [ -f "/opt/pytorch/bin/activate" ]; then
    echo "⚡ Activating AWS DLAMI PyTorch virtual environment (/opt/pytorch)..."
    source /opt/pytorch/bin/activate
    PYTHON_CMD="/opt/pytorch/bin/python"
    PIP_CMD="/opt/pytorch/bin/pip"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "⚡ Using active virtual environment: $VIRTUAL_ENV"
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="training_100ep.log"
echo "=================================================================" | tee -a "$LOG_FILE"
echo "🚀 DUHA-NET 100-EPOCH AWS GPU PIPELINE" | tee -a "$LOG_FILE"
echo "Start Time: $(date)" | tee -a "$LOG_FILE"
echo "Working Dir: $SCRIPT_DIR" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

# --- 2. Hardware Verification ---
echo "" | tee -a "$LOG_FILE"
echo "=== SYSTEM & ACCELERATOR CHECK ===" | tee -a "$LOG_FILE"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi | tee -a "$LOG_FILE"
else
    echo "⚠️ nvidia-smi not found. Training will fallback to available compute device." | tee -a "$LOG_FILE"
fi

# --- 3. Dependency Verification ---
echo "" | tee -a "$LOG_FILE"
echo "=== VERIFYING PYTHON DEPENDENCIES ===" | tee -a "$LOG_FILE"
$PIP_CMD install -q timm albumentations prettytable scikit-learn opencv-python matplotlib

# Verify PyTorch CUDA recognition
$PYTHON_CMD -c "import torch; print(f'PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device Count: {torch.cuda.device_count()}')" | tee -a "$LOG_FILE"

# --- 4. Parse Arguments ---
# Usage:
#   bash run_duhanet_100ep.sh                      (Runs from scratch, epoch 0 to 100)
#   bash run_duhanet_100ep.sh --resume ./output/DUHANet_potsdam_best.pth  (Resumes from checkpoint)
EXTRA_ARGS=""
if [ "$1" == "--resume" ]; then
    CKPT_PATH="${2:-./output/DUHANet_potsdam_best.pth}"
    echo "🔄 Configured to resume training from: $CKPT_PATH" | tee -a "$LOG_FILE"
    EXTRA_ARGS="--resume $CKPT_PATH"
fi

# Ensure output directory exists
mkdir -p output test_results

# --- 5. Step 1: 100-Epoch Training ---
echo "" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"
echo ">>> STEP 1/3: STARTING DUHA-NET 100-EPOCH TRAINING LOOP" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

$PYTHON_CMD -u train.py \
    --data_name potsdam \
    --epoch 100 \
    --samples_per_image 25 \
    --batchsize 4 \
    --lr 1e-4 \
    --decay_epoch 45 \
    --save_path ./output/ \
    $EXTRA_ARGS 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Training phase finished at: $(date)" | tee -a "$LOG_FILE"

# --- 6. Step 2: Evaluation on Test Hold-Out Split ---
echo "" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"
echo ">>> STEP 2/3: BENCHMARKING BEST CHECKPOINT ON 8 TEST TILES" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

$PYTHON_CMD -u test.py \
    --model_path ./output/DUHANet_potsdam_best.pth \
    --data_name potsdam \
    --samples_per_image 25 2>&1 | tee -a "$LOG_FILE"

# --- 7. Step 3: Visualization & Qualitative Predictions ---
echo "" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"
echo ">>> STEP 3/3: GENERATING METRICS PLOT & QUALITATIVE SAMPLES" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

$PYTHON_CMD visualization/plot_metrics.py "$LOG_FILE" 2>&1 | tee -a "$LOG_FILE" || true
$PYTHON_CMD save_predictions.py \
    --model_path ./output/DUHANet_potsdam_best.pth \
    --num_samples 10 2>&1 | tee -a "$LOG_FILE" || true

echo "" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"
echo "🎉 ALL TASKS COMPLETED AT: $(date)" | tee -a "$LOG_FILE"
echo "Weights saved to: ./output/DUHANet_potsdam_best.pth" | tee -a "$LOG_FILE"
echo "Logs saved to   : ./$LOG_FILE" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

# --- 8. Cost Protection: Auto-Shutdown ---
# Uncomment the following line if you wish to automatically power down the instance
# to eliminate idle billing when training completes overnight:
# echo "Shutting down EC2 instance to prevent idle charges..." | tee -a "$LOG_FILE"
# sudo shutdown -h now
