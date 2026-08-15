import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import prettytable
from sklearn.metrics import confusion_matrix, accuracy_score

from utils import dataloader
from utils.metrics import Evaluator
from model.duha_net import create_model

# Standard ISPRS 6-class palette
CLASS_NAMES = ['Impervious', 'Building', 'Low Veg', 'Tree', 'Car', 'Background']

def test_duhanet(test_loader, model, device, model_name='DUHANet'):
    model.eval()
    Eva = Evaluator(6)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Testing"):
            X = sample['image'].to(device, non_blocking=True)
            Y = sample['label'].to(device, non_blocking=True)

            output = model(X)
            # Fetch 'seg' key from our refactored output dictionary
            output = output['seg'] if isinstance(output, dict) else output[0]
            
            output = F.interpolate(output, size=Y.shape[1:], mode='bilinear', align_corners=True)
            pred = torch.argmax(output, dim=1)

            pred_np = pred.cpu().numpy()
            target_np = Y.cpu().numpy()

            for b in range(pred_np.shape[0]):
                Eva.add_batch(target_np[b], pred_np[b])
                all_preds.extend(pred_np[b].flatten())
                all_labels.extend(target_np[b].flatten())

    return Eva, np.array(all_labels), np.array(all_preds)

def main():
    parser = argparse.ArgumentParser(description='Test DUHA-Net Model')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_name', type=str, default='potsdam')
    parser.add_argument('--test_img', type=str, default='./data/potsdam/test/images/')
    parser.add_argument('--test_gt', type=str, default='./data/potsdam/test/labels/')
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--save_path', type=str, default='./test_results/')
    parser.add_argument('--samples_per_image', type=int, default=25, help='Number of test patches per tile')
    
    opt = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_loader = dataloader.get_loader(
        img_root=opt.test_img, gt_root=opt.test_gt, dataset_name=opt.data_name,
        batchsize=opt.batchsize, trainsize=256, mode='test', shuffle=False, samples_per_image=opt.samples_per_image
    )

    model = create_model(num_classes=6, feature_dim=opt.feature_dim, img_size=256).to(device)
    
    checkpoint = torch.load(opt.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    
    Eva, all_labels, all_preds = test_duhanet(test_loader, model, device)
    
    # Print results... (rest of the formatting remains same as your original)
    print(f"\nFinal Test mIoU: {Eva.Mean_Intersection_over_Union():.4f}")

if __name__ == '__main__':
    main()