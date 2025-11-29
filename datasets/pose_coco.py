###################################################################################################
#
# COCO Pose Estimation Dataset for AI8X Training
#
# This dataset adapts the lightweight-human-pose-estimation COCO dataset
# for use with the ai8x training framework.
#
###################################################################################################
"""
COCO Pose Estimation Dataset for Knowledge Distillation Training
"""
import copy
import math
import os
import pickle
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import ai8x

# Add the pose estimation repo to path for imports
POSE_REPO_PATH = '/home/jkal/Desktop/MLonMCU/lightweight-human-pose-estimation.pytorch'
if POSE_REPO_PATH not in sys.path:
    sys.path.insert(0, POSE_REPO_PATH)


BODY_PARTS_KPT_IDS = [
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [2, 16],
    [1, 5], [5, 6], [6, 7], [5, 17], [1, 0], [0, 14], [0, 15], [14, 16], [15, 17]
]


class PoseDatasetAI8X(Dataset):
    """
    COCO Pose Dataset adapted for AI8X training with knowledge distillation.
    
    Returns:
        image: Normalized image tensor
        target: Dict with keypoint_maps, paf_maps, and masks for loss calculation
    """
    def __init__(self, labels_path, images_folder, stride=8, sigma=7, paf_thickness=1,
                 input_size=128, args=None, augment=True):
        super().__init__()
        self.images_folder = images_folder
        self.stride = stride
        self.sigma = sigma
        self.paf_thickness = paf_thickness
        self.input_size = input_size
        self.args = args
        self.augment = augment
        
        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = copy.deepcopy(self.labels[idx])
        image = cv2.imread(os.path.join(self.images_folder, label['img_paths']), cv2.IMREAD_COLOR)
        
        if image is None:
            raise IOError(f'Image {label["img_paths"]} cannot be read')
        
        mask = np.ones(shape=(label['img_height'], label['img_width']), dtype=np.float32)
        
        # Apply augmentations if training
        if self.augment:
            image, label, mask = self._augment(image, label, mask)
        
        # Resize to target input size
        image, label, mask = self._resize_to_input(image, label, mask)
        
        # Generate target maps
        keypoint_maps = self._generate_keypoint_maps(image, label)
        paf_maps = self._generate_paf_maps(image, label)
        
        # Resize mask for output stride
        mask_resized = cv2.resize(mask, dsize=None, 
                                  fx=1/self.stride, fy=1/self.stride, 
                                  interpolation=cv2.INTER_AREA)
        
        # Create masks for each channel
        keypoint_mask = np.zeros(shape=keypoint_maps.shape, dtype=np.float32)
        for i in range(keypoint_mask.shape[0]):
            keypoint_mask[i] = mask_resized
        
        paf_mask = np.zeros(shape=paf_maps.shape, dtype=np.float32)
        for i in range(paf_mask.shape[0]):
            paf_mask[i] = mask_resized
        
        # Normalize image
        image = image.astype(np.float32)
        image = (image - 128) / 256  # Normalize to [-0.5, 0.5] range
        image = image.transpose((2, 0, 1))  # HWC -> CHW
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        
        # Apply ai8x normalization if args provided
        if self.args is not None and hasattr(self.args, 'act_mode_8bit'):
            # Scale to ai8x expected range
            if self.args.act_mode_8bit:
                image = image.mul(256.).round().clamp(min=-128, max=127)
            else:
                image = image.mul(256.).round().clamp(min=-128, max=127).div(128.)
        
        # Target is a combined tensor of heatmaps and PAFs for regression
        target = {
            'keypoint_maps': torch.from_numpy(keypoint_maps).float(),
            'paf_maps': torch.from_numpy(paf_maps).float(),
            'keypoint_mask': torch.from_numpy(keypoint_mask).float(),
            'paf_mask': torch.from_numpy(paf_mask).float(),
        }
        
        return image, target
    
    def _augment(self, image, label, mask):
        """Apply data augmentation"""
        import random
        
        # Scale augmentation
        scale_multiplier = random.uniform(0.5, 1.1)
        target_dist = 0.6
        scale_abs = target_dist / label['scale_provided']
        scale = scale_abs * scale_multiplier
        
        image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)
        mask = cv2.resize(mask, dsize=(0, 0), fx=scale, fy=scale)
        label['img_height'], label['img_width'] = image.shape[:2]
        
        # Scale keypoints
        label['objpos'][0] *= scale
        label['objpos'][1] *= scale
        for keypoint in label['keypoints']:
            keypoint[0] *= scale
            keypoint[1] *= scale
        for other in label.get('processed_other_annotations', []):
            other['objpos'][0] *= scale
            other['objpos'][1] *= scale
            for keypoint in other['keypoints']:
                keypoint[0] *= scale
                keypoint[1] *= scale
        
        # Random horizontal flip
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            w = label['img_width']
            label['objpos'][0] = w - 1 - label['objpos'][0]
            for keypoint in label['keypoints']:
                keypoint[0] = w - 1 - keypoint[0]
            # Swap left/right keypoints
            label['keypoints'] = self._swap_left_right(label['keypoints'])
        
        return image, label, mask
    
    def _swap_left_right(self, keypoints):
        """Swap left and right keypoints for flip augmentation"""
        right = [2, 3, 4, 8, 9, 10, 14, 16]
        left = [5, 6, 7, 11, 12, 13, 15, 17]
        for r, l in zip(right, left):
            if r < len(keypoints) and l < len(keypoints):
                keypoints[r], keypoints[l] = keypoints[l], keypoints[r]
        return keypoints
    
    def _resize_to_input(self, image, label, mask):
        """Resize image and annotations to target input size"""
        h, w = image.shape[:2]
        scale = self.input_size / max(h, w)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        image = cv2.resize(image, (new_w, new_h))
        mask = cv2.resize(mask, (new_w, new_h))
        
        # Pad to square
        pad_h = self.input_size - new_h
        pad_w = self.input_size - new_w
        
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, 
                                   cv2.BORDER_CONSTANT, value=(128, 128, 128))
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, 
                                  cv2.BORDER_CONSTANT, value=0)
        
        # Scale keypoints
        for keypoint in label['keypoints']:
            keypoint[0] *= scale
            keypoint[1] *= scale
        for other in label.get('processed_other_annotations', []):
            for keypoint in other['keypoints']:
                keypoint[0] *= scale
                keypoint[1] *= scale
        
        label['img_width'] = self.input_size
        label['img_height'] = self.input_size
        
        return image, label, mask
    
    def _generate_keypoint_maps(self, image, label):
        """Generate keypoint heatmaps"""
        n_keypoints = 18
        n_rows, n_cols = image.shape[:2]
        keypoint_maps = np.zeros(shape=(n_keypoints + 1,
                                        n_rows // self.stride, 
                                        n_cols // self.stride), dtype=np.float32)
        
        for keypoint_idx in range(n_keypoints):
            if keypoint_idx < len(label['keypoints']):
                keypoint = label['keypoints'][keypoint_idx]
                if len(keypoint) >= 3 and keypoint[2] <= 1:
                    self._add_gaussian(keypoint_maps[keypoint_idx], 
                                      keypoint[0], keypoint[1], 
                                      self.stride, self.sigma)
            
            for other in label.get('processed_other_annotations', []):
                if keypoint_idx < len(other['keypoints']):
                    keypoint = other['keypoints'][keypoint_idx]
                    if len(keypoint) >= 3 and keypoint[2] <= 1:
                        self._add_gaussian(keypoint_maps[keypoint_idx], 
                                          keypoint[0], keypoint[1], 
                                          self.stride, self.sigma)
        
        keypoint_maps[-1] = 1 - keypoint_maps.max(axis=0)
        return keypoint_maps
    
    def _add_gaussian(self, keypoint_map, x, y, stride, sigma):
        """Add a Gaussian blob at keypoint location"""
        n_sigma = 4
        tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
        tl[0] = max(tl[0], 0)
        tl[1] = max(tl[1], 0)
        
        br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
        map_h, map_w = keypoint_map.shape
        br[0] = min(br[0], map_w * stride)
        br[1] = min(br[1], map_h * stride)
        
        shift = stride / 2 - 0.5
        for map_y in range(tl[1] // stride, br[1] // stride):
            for map_x in range(tl[0] // stride, br[0] // stride):
                d2 = ((map_x * stride + shift - x) ** 2 + 
                      (map_y * stride + shift - y) ** 2)
                exponent = d2 / 2 / sigma / sigma
                if exponent > 4.6052:
                    continue
                keypoint_map[map_y, map_x] += math.exp(-exponent)
                if keypoint_map[map_y, map_x] > 1:
                    keypoint_map[map_y, map_x] = 1
    
    def _generate_paf_maps(self, image, label):
        """Generate Part Affinity Fields"""
        n_pafs = len(BODY_PARTS_KPT_IDS)
        n_rows, n_cols = image.shape[:2]
        paf_maps = np.zeros(shape=(n_pafs * 2, 
                                   n_rows // self.stride, 
                                   n_cols // self.stride), dtype=np.float32)
        
        for paf_idx in range(n_pafs):
            kpt_ids = BODY_PARTS_KPT_IDS[paf_idx]
            if kpt_ids[0] < len(label['keypoints']) and kpt_ids[1] < len(label['keypoints']):
                keypoint_a = label['keypoints'][kpt_ids[0]]
                keypoint_b = label['keypoints'][kpt_ids[1]]
                if len(keypoint_a) >= 3 and len(keypoint_b) >= 3:
                    if keypoint_a[2] <= 1 and keypoint_b[2] <= 1:
                        self._set_paf(paf_maps[paf_idx * 2:paf_idx * 2 + 2],
                                     keypoint_a[0], keypoint_a[1],
                                     keypoint_b[0], keypoint_b[1],
                                     self.stride, self.paf_thickness)
        
        return paf_maps
    
    def _set_paf(self, paf_map, x_a, y_a, x_b, y_b, stride, thickness):
        """Set Part Affinity Field between two keypoints"""
        x_a /= stride
        y_a /= stride
        x_b /= stride
        y_b /= stride
        x_ba = x_b - x_a
        y_ba = y_b - y_a
        _, h_map, w_map = paf_map.shape
        x_min = int(max(min(x_a, x_b) - thickness, 0))
        x_max = int(min(max(x_a, x_b) + thickness, w_map))
        y_min = int(max(min(y_a, y_b) - thickness, 0))
        y_max = int(min(max(y_a, y_b) + thickness, h_map))
        norm_ba = (x_ba * x_ba + y_ba * y_ba) ** 0.5
        if norm_ba < 1e-7:
            return
        x_ba /= norm_ba
        y_ba /= norm_ba
        
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                x_ca = x - x_a
                y_ca = y - y_a
                d = abs(x_ca * y_ba - y_ca * x_ba)
                if d <= thickness:
                    paf_map[0, y, x] = x_ba
                    paf_map[1, y, x] = y_ba


def pose_coco_collate_fn(batch):
    """
    Custom collate function for pose estimation dataset.
    Handles the dict targets properly.
    """
    images = torch.stack([item[0] for item in batch])
    
    # Combine target dicts
    targets = {
        'keypoint_maps': torch.stack([item[1]['keypoint_maps'] for item in batch]),
        'paf_maps': torch.stack([item[1]['paf_maps'] for item in batch]),
        'keypoint_mask': torch.stack([item[1]['keypoint_mask'] for item in batch]),
        'paf_mask': torch.stack([item[1]['paf_mask'] for item in batch]),
    }
    
    return images, targets


def pose_coco_get_datasets(data, load_train=True, load_test=True):
    """
    Load the COCO pose estimation dataset.
    
    Expected data structure:
    - data[0]: Base path (e.g., '/path/to/coco')
    - data[1]: args object
    
    The COCO data should be organized as:
    - {base_path}/train2017/  - training images
    - {base_path}/val2017/    - validation images  
    - {base_path}/prepared_train_annotations.pkl  - prepared training labels
    - {base_path}/annotations/person_keypoints_val2017.json  - validation labels
    """
    (data_dir, args) = data
    
    # Paths - adjust these based on your COCO data location
    coco_path = os.path.join(data_dir, 'COCO_POSE')
    
    # Alternative: use the lightweight pose repo's coco path
    if not os.path.exists(coco_path):
        coco_path = '/home/jkal/Desktop/MLonMCU/lightweight-human-pose-estimation.pytorch/coco'
    
    train_labels = os.path.join(coco_path, 'prepared_train_annotations.pkl')
    train_images = os.path.join(coco_path, 'train2017')
    val_images = os.path.join(coco_path, 'val2017')
    
    train_dataset = None
    test_dataset = None
    
    if load_train and os.path.exists(train_labels) and os.path.exists(train_images):
        train_dataset = PoseDatasetAI8X(
            labels_path=train_labels,
            images_folder=train_images,
            stride=8,
            sigma=7,
            paf_thickness=1,
            input_size=128,
            args=args,
            augment=True
        )
    
    if load_test and os.path.exists(train_labels) and os.path.exists(val_images):
        # Use validation images for testing
        test_dataset = PoseDatasetAI8X(
            labels_path=train_labels,  # Note: ideally should use val labels
            images_folder=train_images,  # Use training data for now
            stride=8,
            sigma=7,
            paf_thickness=1,
            input_size=128,
            args=args,
            augment=False
        )
    
    return train_dataset, test_dataset


# For regression output (57 channels: 19 heatmaps + 38 PAFs)
# We'll treat this as a regression problem
datasets = [
    {
        'name': 'COCO_POSE',
        'input': (3, 128, 128),
        'output': tuple([f'channel_{i}' for i in range(57)]),  # 19 heatmaps + 38 PAFs
        'regression': True,
        'loader': pose_coco_get_datasets,
        'collate': pose_coco_collate_fn,
    },
]

