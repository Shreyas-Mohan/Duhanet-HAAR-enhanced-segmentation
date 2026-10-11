import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import random
import torch
import torch.utils.data as data
import numpy as np
import cv2
try:
    cv2.setLogLevel(0)
except Exception:
    pass
from typing import List, Tuple, Dict, Optional, Mapping

def fast_mask_to_label(mask_np: np.ndarray, palette: List[List[int]]) -> np.ndarray:
    """
    Highly optimized color mask to class index mapping using vectorized numpy.
    Converts a HxWx3 RGB palette image directly to HxW class indices.
    """
    H, W, _ = mask_np.shape
    label_matrix = np.zeros((H, W), dtype=np.uint8)
    
    # Flatten the image array and palette to perform a single vectorized match
    flat_mask = mask_np.reshape(-1, 3)
    flat_labels = np.zeros(flat_mask.shape[0], dtype=np.uint8)
    
    for idx, color in enumerate(palette):
        # Create a single boolean mask matching the color
        match = (flat_mask[:, 0] == color[0]) & (flat_mask[:, 1] == color[1]) & (flat_mask[:, 2] == color[2])
        flat_labels[match] = idx
        
    return flat_labels.reshape(H, W)

def extract_edge_cv2(mask_np: np.ndarray) -> np.ndarray:
    """Uses ultra-fast OpenCV Laplacian or Canny filter instead of slow PIL ImageFilter."""
    # Find transitions between different class integers
    grad_x = cv2.Sobel(mask_np.astype(np.float32), cv2.CV_32F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(mask_np.astype(np.float32), cv2.CV_32F, 0, 1, ksize=1)
    edge = (np.abs(grad_x) + np.abs(grad_y) > 0).astype(np.float32)
    return edge

class SemiSupervisedSegmentationDataset(data.Dataset):
    """
    Refactored, high-performance dataset class.
    Handles cropping first before performing heavy matrix transformations.
    """
    def __init__(
        self, 
        img_root: str, 
        gt_root: Optional[str], 
        palette: List[List[int]], 
        trainsize: int, 
        mode: str,
        label_ratio: float = 1.0, 
        use_partial_labels: bool = False,
        samples_per_image:int =1
    ):
        self.trainsize = trainsize
        self.image_root = img_root
        self.gt_root = gt_root
        self.palette = palette
        self.mode = mode

        # Find files safely
        if not os.path.exists(self.image_root):
            raise FileNotFoundError(f"Image directory not found: {self.image_root}")
            
        self.image_files = sorted([
            f for f in os.listdir(self.image_root) 
            if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))
        ])

        self.images = []
        self.gts = []

        if mode == 'unlabeled' or self.gt_root is None:
            self.images = [os.path.join(self.image_root, f) for f in self.image_files]
        else:
            if not os.path.exists(self.gt_root):
                raise FileNotFoundError(f"Label directory not found: {self.gt_root}")
                
            for img_name in self.image_files:
                img_base = os.path.splitext(img_name)[0]
                label_base = img_base[:-4] if img_base.endswith('_RGB') else img_base
                matched_label = f"{label_base}_label.tif"

                if os.path.exists(os.path.join(self.gt_root, matched_label)):
                    self.images.append(os.path.join(self.image_root, img_name))
                    self.gts.append(os.path.join(self.gt_root, matched_label))
                else:
                    print(f"⚠️ Warning: Mismatch detected. Could not find a matching label for image tile: {img_name}")
        self.actual_size=len(self.images)
        if mode in ['train', 'test', 'val']:
                self.samples_per_image = samples_per_image
        else:
                self.samples_per_image = 1
        self.size = self.actual_size*self.samples_per_image
        print(f"[{mode.upper()}] Successfully paired and verified {self.actual_size} image-mask pairs.")
        # Allow train, val, and test modes to scale patches
        if self.actual_size == 0:
            raise RuntimeError("Fatal: Zero valid image-mask pairs successfully aligned. Check file name alignments.")

    def _random_crop_and_flip(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        height, width = img.shape[:2]
        crop_size = self.trainsize

        if height < crop_size or width < crop_size:
            pad_h = max(crop_size - height, 0)
            pad_w = max(crop_size - width, 0)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            if mask is not None:
                mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            height, width = img.shape[:2]

        top = random.randint(0, height - crop_size)
        left = random.randint(0, width - crop_size)

        img_crop = img[top:top + crop_size, left:left + crop_size]
        mask_crop = mask[top:top + crop_size, left:left + crop_size] if mask is not None else None

        if self.mode in ('train', 'unlabeled'):
            if random.random() > 0.5:
                img_crop = np.fliplr(img_crop).copy()
                if mask_crop is not None:
                    mask_crop = np.fliplr(mask_crop).copy()
            if random.random() > 0.5:
                img_crop = np.flipud(img_crop).copy()
                if mask_crop is not None:
                    mask_crop = np.flipud(mask_crop).copy()

        return img_crop, mask_crop

    def __getitem__(self, index: int):
        # Bound guarding for worker processes
        index = index % self.actual_size
        #img_idx=index%self.actual_size
        img_path = self.images[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == 'unlabeled':
            img_crop, _ = self._random_crop_and_flip(image)
            img_tensor = torch.from_numpy(img_crop.transpose((2, 0, 1))).float() / 255.0
            return {'image': img_tensor, 'index': index, 'has_label': False}

        gt_path = self.gts[index]
        mask = cv2.imread(gt_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        img_crop, mask_crop = self._random_crop_and_flip(image, mask)

        assert mask_crop is not None
        label_crop = fast_mask_to_label(mask_crop, self.palette)
        edge_crop = extract_edge_cv2(label_crop)

        img_tensor = torch.from_numpy(img_crop.transpose((2, 0, 1))).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        label_tensor = torch.from_numpy(label_crop).long()
        edge_tensor = torch.from_numpy(edge_crop).float()

        result = {
            'image': img_tensor,
            'label': label_tensor,
            'edge': edge_tensor
        }
        
        return result

    def __len__(self) -> int:
        return self.size

def get_loader(
    img_root: str, 
    gt_root: Optional[str], 
    dataset_name: str, 
    batchsize: int, 
    trainsize: int, 
    mode: str,
    label_ratio: float = 1.0,          # Added back to prevent legacy crashes
    use_partial_labels: bool = False,  # Added back to prevent legacy crashes
    num_workers: int = 4, 
    shuffle: bool = True, 
    pin_memory: bool = True,
    samples_per_image: int = 1
) -> data.DataLoader:
    
    palette = [
        [255, 255, 255],  # Impervious surfaces
        [0, 0, 255],      # Buildings
        [0, 255, 255],    # Low Vegetation
        [0, 255, 0],      # Trees
        [255, 255, 0],    # Cars
        [255, 0, 0],      # Clutter/Background
    ]

    dataset = SemiSupervisedSegmentationDataset(
        img_root=img_root,
        gt_root=gt_root,
        palette=palette,
        trainsize=trainsize,
        mode=mode,
        label_ratio=label_ratio,
        use_partial_labels=use_partial_labels,
        samples_per_image=samples_per_image
    )

    return data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle if mode == 'train' else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(mode == 'train')
    )