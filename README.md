# DUHA-Net: Haar-Enhanced Dual-Branch Network for Aerial Image Segmentation

This repository contains an optimized and heavily refactored PyTorch implementation of **DUHA-Net**, designed for semantic segmentation of high-resolution remote sensing imagery.

The architecture leverages a dual-branch structure (**ConvNeXt** for local features + **Swin Transformer** for global dependencies), enhanced by a **Haar Enhancement Module (HEM)** and **Semantic-Guided Decoupling Aggregation (S-GDA)**. It utilizes an **Uncertainty-aware Dual-branch Training (UDT)** strategy to dynamically supervise consistency between branches.

## 🚀 Key Optimizations & Refactoring

The original codebase was refactored to prioritize hardware efficiency, memory safety, and training stability:

- **Hardware-Agnostic Execution:** Removed hardcoded `.cuda()` calls. The pipeline now seamlessly scales from local CPU execution to multi-GPU/Colab accelerator environments.
- **Dynamic Patch-Based Sampling:** Replaced full-image loading, which causes OOM errors on 6000×6000 images, with dynamic **256 × 256** random patch cropping.
- **Virtual Epoch Scaling:** Introduced a `samples_per_image` parameter to control spatial exposure per epoch, multiplying dataset diversity without increasing memory overhead.
- **Vectorized Fusion Layers:** Eliminated sequential loops in `sgda.py` in favor of vectorized tensor broadcasting, fixing `.view()` contiguity crashes via safe `.reshape()` operations.

## 📂 Project Structure

```text
HAAR-ENHANCED-NETWORK-MAIN/
│
├── data/                   # [Ignored by Git] ISPRS Potsdam Dataset (train/test splits)
├── model/                  # Core architecture (DUHA-Net, Swin, ConvNeXt, S-GDA, HEM, UDT)
├── output/                 # [Ignored by Git] Saved model weights (.pth) and logs
├── utils/                  # Dataloaders and evaluation metrics
│
├── plot_metrics.py         # Script to parse training logs into matplotlib accuracy/loss graphs
├── requirements.txt        # Python dependencies
├── test.py                 # Evaluation script with multi-patch testing
├── train.py                # Hardware-agnostic training loop with UDT warmup
├── verify_pipeline.py      # Quick sanity-check script for data pairing
└── README.md
```

## 🛠️ Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Core dependencies:** `torch`, `torchvision`, `timm==0.6.12`, `albumentations`, `opencv-python`, `scikit-learn`

## 📊 Dataset Preparation (.gitignore)

This project relies on the **ISPRS Potsdam** dataset. Because aerial imagery is massive, the `data/` and `output/` directories should be added to your `.gitignore` file to prevent pushing gigabytes of data to GitHub.

**Recommended `.gitignore` additions:**

```gitignore
# Datasets and Outputs
data/
output/
Potsdam/
test_results/

# Logs and visual graphs
*.txt
*.png
```

Place your 3-channel RGB imagery (`2_Ortho_RGB`) and ground truth labels (`5_Labels_all`) into the following structure:

- `data/potsdam/train/images/`
- `data/potsdam/train/labels/`
- `data/potsdam/test/images/`
- `data/potsdam/test/labels/`

## 💻 Usage

### Training the Model

To train the model using the optimized multi-patch sampling technique (e.g., extracting 25 random **256 × 256** crops from each high-resolution tile per epoch):

```bash
python train.py \
    --data_name potsdam \
    --epoch 20 \
    --samples_per_image 25 \
    --batchsize 4
```

The script automatically tracks validation IoU at the end of every epoch and saves the best-performing weights to:

```text
./output/DUHANet_potsdam_best.pth
```

### Evaluating the Model

To ensure a robust and fair evaluation, the test script evaluates across multiple patches per hold-out tile to eliminate sampling variance.

```bash
python test.py \
    --model_path ./output/DUHANet_potsdam_best.pth \
    --data_name potsdam \
    --samples_per_image 25
```

### Visualizing Results

To generate loss decay and IoU progression graphs for reports, pipe your terminal output to a text file (e.g., `training_log.txt`) and run:

```bash
python plot_metrics.py
```

## 📈 Baseline Results

On a localized, CPU-bound 20-epoch training run (sampling 12,000 unique patches total), the refactored architecture achieved a **Test mIoU of 0.6402** on the Potsdam hold-out set, confirming the mathematical integrity and rapid convergence capabilities of the S-GDA and Haar Enhancement modules.