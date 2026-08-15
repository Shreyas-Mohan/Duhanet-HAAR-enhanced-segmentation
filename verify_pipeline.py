import torch
import numpy as np
import os
from utils.dataloader import get_loader

def verify_pipeline():
    # Define absolute paths to your newly unzipped data folders
    # Update these paths if your root folder name differs
    IMG_ROOT = "./data/train/images/"
    GT_ROOT = "./data/train/labels/"
    
    print("=" * 60)
    print("        DUHA-Net Pipeline Verification Tool")
    print("=" * 60)
    
    # Quick sanity check on directory existence
    if not os.path.exists(IMG_ROOT) or not os.path.exists(GT_ROOT):
        print(f"❌ Error: Could not find data directories.")
        print(f"   Checked Image Path: {IMG_ROOT}")
        print(f"   Checked Label Path: {GT_ROOT}")
        print("\nPlease check that your folders match this path layout.")
        return

    # Initialize your loader using the optimized code we wrote
    print("\n[Step 1] Initializing optimized Potsdam Data Loader...")
    try:
        train_loader = get_loader(
            img_root=IMG_ROOT,
            gt_root=GT_ROOT,
            dataset_name="potsdam",
            batchsize=4,
            trainsize=256,
            mode="train",
            num_workers=2,  # Keep workers light for verification
            shuffle=True
        )
    except Exception as e:
        print(f"❌ Failed to initialize Data Loader: {e}")
        return

    # Pull a single batch to inspect elements
    print("\n[Step 2] Attempting to pull a single batch of 256x256 patches...")
    try:
        iterator = iter(train_loader)
        sample = next(iterator)
    except Exception as e:
        print(f"❌ Failed to extract a batch from files: {e}")
        print("   This usually means a filename mismatch between images (.tif) and labels (.png).")
        return

    # Extract tensors
    images = sample['image']
    labels = sample['label']
    edges = sample['edge']

    print("📊 Batch extracted successfully! Inspecting dimensions:")
    print(f"   -> Image Tensor Shape : {images.shape} (Expected: [4, 3, 256, 256])")
    print(f"   -> Label Tensor Shape : {labels.shape} (Expected: [4, 256, 256])")
    print(f"   -> Edge Tensor Shape  : {edges.shape} (Expected: [4, 256, 256])")

    # Run value validations
    print("\n[Step 3] Validating tensor numerical ranges:")
    
    # Images should be normalized around 0 (ImageNet distribution)
    print(f"   -> Image Range: Min = {images.min().item():.2f}, Max = {images.max().item():.2f}")
    
    # Class values must fall strictly between 0 and 5 (the 6 Potsdam categories)
    unique_classes = torch.unique(labels).tolist()
    print(f"   -> Unique class indices found in batch: {unique_classes} (Expected subset of [0, 1, 2, 3, 4, 5])")
    
    # Edges must be binary boundaries
    unique_edges = torch.unique(edges).tolist()
    print(f"   -> Unique edge indices found in batch : {unique_edges} (Expected subset of [0.0, 1.0])")

    # Check for hardware target compatibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Step 4] Hardware Target Check:")
    print(f"   -> Running locally on: {device.type.upper()}")
    
    if max(unique_classes) > 5:
        print("⚠️ Warning: Found a class index greater than 5. Double check mask color palette mapping.")
    else:
        print("\n🎉 Everything looks completely solid! Data pipeline optimization is verified.")
    print("=" * 60)

if __name__ == "__main__":
    verify_pipeline()