#!/bin/bash
# ==============================================================================
# DUHA-Net: EC2 Environment Setup Script (Ubuntu 22.04 / AWS DLAMI)
# ==============================================================================
set -e

echo "=== Setting up System & Python Environment for DUHA-Net ==="

# Update system packages and install OpenCV runtime dependencies
sudo apt-get update -y
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 tmux htop rsync unzip

# Detect python/pip
if [ -f "/opt/pytorch/bin/activate" ]; then
    echo "Found AWS DLAMI environment (/opt/pytorch). Activating..."
    source /opt/pytorch/bin/activate
    PIP_CMD="/opt/pytorch/bin/pip"
else
    PIP_CMD="pip3"
fi

echo "Installing required Python packages..."
$PIP_CMD install --upgrade pip
$PIP_CMD install timm==0.6.12 albumentations prettytable scikit-learn opencv-python matplotlib

echo "=== Testing PyTorch & CUDA ==="
python3 -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count())"

echo "=== Environment Setup Complete! ==="
