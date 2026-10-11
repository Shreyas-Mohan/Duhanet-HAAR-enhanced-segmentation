import os
import argparse
from importlib import import_module
import numpy as np
import torch
from PIL import Image
from utils.dataloader import get_loader
from model.duha_net import DUHANet

# 0: Impervious surfaces (White)
# 1: Building (Blue)
# 2: Low vegetation (Cyan)
# 3: Tree (Green)
# 4: Car (Yellow)
# 5: Clutter / Background (Red)
POTSDAM_COLORMAP = np.array([
    [255, 255, 255], # 0: Impervious surfaces
    [0, 0, 255],     # 1: Building
    [0, 255, 255],   # 2: Low vegetation
    [0, 255, 0],     # 3: Tree
    [255, 255, 0],   # 4: Car
    [255, 0, 0]      # 5: Clutter
], dtype=np.uint8)

def colorize_mask(mask_2d):
    """Converts a 2D class-index mask (0-5) into an RGB color image."""
    h, w = mask_2d.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(POTSDAM_COLORMAP)):
        color_mask[mask_2d == class_id] = POTSDAM_COLORMAP[class_id]
    return color_mask

def main():
    parser = argparse.ArgumentParser(description="Save qualitative predictions from DUHA-Net")
    parser.add_argument('--model_path', type=str, default='output/DUHANet_potsdam_best.pth', help='Path to weights')
    parser.add_argument('--data_name', type=str, default='potsdam', help='Dataset name')
    parser.add_argument('--test_img', type=str, default='data/potsdam/test/images', help='Images directory')
    parser.add_argument('--test_gt', type=str, default='data/potsdam/test/labels', help='Labels directory')
    parser.add_argument('--save_dir', type=str, default='data/test_results/qualitative_vis', help='Output save directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of random training patches to visualize')
    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(opt.save_dir, exist_ok=True)

    print(f"[VIS] Loading model weights from: {opt.model_path}")
    model = DUHANet(num_classes=6).to(device)
    
    checkpoint = torch.load(opt.model_path, map_location=device, weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    train_loader = get_loader(
        img_root=opt.test_img,
        gt_root=opt.test_gt,
        dataset_name=opt.data_name,
        batchsize=1,
        trainsize=256,
        mode='train',
        shuffle=True,
        samples_per_image=1
    )

    print(f"[VIS] Running inference to extract visual predictions...")
    sample_count = 0

    with torch.no_grad():
        for i, sample in enumerate(train_loader):
            if sample_count >= opt.num_samples:
                break

            images = sample['image'].to(device)
            gts = sample['label']
            preds = model(images)
            
            # Extract argmax class prediction map
            if isinstance(preds, dict):
                preds = preds['seg']
            elif isinstance(preds, (list, tuple)):
                preds = preds[0]
            
            pred_mask = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy()
            gt_mask = gts.squeeze(0).cpu().numpy()

            # Denormalize image tensor to 0-255 RGB Image
            img_np = images.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255).astype(np.uint8)

            # Colorize ground truth and predictions
            pred_rgb = colorize_mask(pred_mask)
            gt_rgb = colorize_mask(gt_mask)

            # Save individual patch components
            Image.fromarray(img_np).save(os.path.join(opt.save_dir, f"patch_{i}_input.png"))
            Image.fromarray(gt_rgb).save(os.path.join(opt.save_dir, f"patch_{i}_gt.png"))
            Image.fromarray(pred_rgb).save(os.path.join(opt.save_dir, f"patch_{i}_duha_pred.png"))

            sample_count += 1
            print(f"Saved sample {sample_count}/{opt.num_samples} -> patch_{i}_*.png")

    print(f"[SUCCESS] Predictions saved to: {os.path.abspath(opt.save_dir)}")

if __name__ == '__main__':
    main()