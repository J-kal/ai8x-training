#!/usr/bin/env python3
"""
Medium Pose Estimation Validation

Validates the medium pose model (from train_pose_cpu.py) on COCO validation set.
Uses the same 128x128 input format as training.

Usage:
    python val_pose_med.py --checkpoint pose_med2_train/best.pth
    python val_pose_med.py --checkpoint pose_med2_train/best.pth --visualize
    python val_pose_med.py --checkpoint pose_med2_train/best.pth --subset 500
"""

import argparse
import copy
import cv2
import datetime
import json
import math
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn

# Paths
POSE_REPO = '/home/jkal/Desktop/MLonMCU/lightweight-human-pose-estimation.pytorch'
AI8X_REPO = '/home/jkal/Desktop/MLonMCU/ext/ai8x-training'
sys.path.insert(0, POSE_REPO)
sys.path.insert(0, AI8X_REPO)

import ai8x
ai8x.set_device(device=85, simulate=False, round_avg=False)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from modules.keypoints import extract_keypoints, group_keypoints


def log(msg, log_file=None):
    """Log with timestamp"""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"[{ts}] {msg}\n")


class MediumStudentModel(nn.Module):
    """Medium student model architecture (from train_pose_cpu.py)"""
    def __init__(self):
        super().__init__()
        # Backbone
        self.conv1 = ai8x.FusedConv2dBNReLU(3, 32, 3, stride=2, padding=1, bias=True)
        self.conv2 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1, bias=True)
        self.conv3 = ai8x.FusedConv2dBNReLU(32, 64, 3, stride=1, padding=1, bias=True)
        self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=True)
        self.conv5 = ai8x.FusedConv2dBNReLU(64, 96, 3, stride=1, padding=1, bias=True)
        self.conv6 = ai8x.FusedMaxPoolConv2dBNReLU(96, 96, 3, pool_size=2, pool_stride=2, padding=1, bias=True)
        self.conv7 = ai8x.FusedConv2dBNReLU(96, 128, 3, stride=1, padding=1, bias=True)
        self.conv8 = ai8x.FusedConv2dBNReLU(128, 128, 3, stride=1, padding=1, bias=True)
        # CPM
        self.cpm1 = ai8x.FusedConv2dBNReLU(128, 96, 1, padding=0, bias=True)
        self.cpm2 = ai8x.FusedConv2dBNReLU(96, 96, 3, padding=1, bias=True)
        self.cpm3 = ai8x.FusedConv2dBNReLU(96, 96, 3, padding=1, bias=True)
        # Heads
        self.heat_conv = ai8x.FusedConv2dBNReLU(96, 64, 1, padding=0, bias=True)
        self.heat_out = ai8x.Conv2d(64, 19, 1, padding=0, bias=True, wide=True)
        self.paf_conv = ai8x.FusedConv2dBNReLU(96, 64, 1, padding=0, bias=True)
        self.paf_out = ai8x.Conv2d(64, 38, 1, padding=0, bias=True, wide=True)
    
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = self.conv4(x); x = self.conv5(x); x = self.conv6(x)
        x = self.conv7(x); x = self.conv8(x)
        x = self.cpm1(x); x = self.cpm2(x); x = self.cpm3(x)
        h = self.heat_out(self.heat_conv(x))
        p = self.paf_out(self.paf_conv(x))
        return [h, p]


def normalize_image(img, size=128):
    """
    Normalize image to match training format:
    - Resize to size x size with padding
    - Normalize to [-1, 1] range
    """
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (int(w*s), int(h*s)))
    pad_h, pad_w = size - img.shape[0], size - img.shape[1]
    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    
    # Normalize to [-1, 1]
    img = (img.astype(np.float32) - 128) / 128
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return img, s, (0, 0)  # scale, pad_offset


def infer(net, img, input_size=128, stride=8):
    """
    Run inference on a single image
    
    Args:
        net: Model
        img: Input image (H, W, 3) BGR format
        input_size: Input size (128 for medium model)
        stride: Output stride (8 for medium model)
    
    Returns:
        heatmaps: (H, W, 19) heatmaps at original image resolution
        pafs: (H, W, 38) PAFs at original image resolution
        scale: Scale factor used for resizing
    """
    original_h, original_w = img.shape[:2]
    
    # Normalize and prepare input
    normed_img, scale, pad = normalize_image(img, input_size)
    tensor_img = torch.from_numpy(normed_img).unsqueeze(0).float()
    
    # Run inference
    with torch.no_grad():
        output = net(tensor_img)
    
    # Extract heatmaps and PAFs (output is 16x16 for 128x128 input)
    heatmaps = output[0].squeeze().cpu().numpy()  # (19, 16, 16)
    pafs = output[1].squeeze().cpu().numpy()  # (38, 16, 16)
    
    # Transpose to (H, W, C)
    heatmaps = np.transpose(heatmaps, (1, 2, 0))  # (16, 16, 19)
    pafs = np.transpose(pafs, (1, 2, 0))  # (16, 16, 38)
    
    # Upsample to input size (128x128)
    heatmaps = cv2.resize(heatmaps, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    pafs = cv2.resize(pafs, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    
    # Crop padding and scale to original image size
    scaled_h = int(original_h * scale)
    scaled_w = int(original_w * scale)
    heatmaps = heatmaps[:scaled_h, :scaled_w, :]
    pafs = pafs[:scaled_h, :scaled_w, :]
    
    # Resize to original image size
    heatmaps = cv2.resize(heatmaps, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
    pafs = cv2.resize(pafs, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
    
    return heatmaps, pafs, scale


def convert_to_coco_format(pose_entries, all_keypoints, scale):
    """
    Convert pose detections to COCO format
    
    Args:
        pose_entries: List of detected poses
        all_keypoints: Array of all detected keypoints
        scale: Scale factor from preprocessing
    
    Returns:
        coco_keypoints: List of keypoint arrays in COCO format
        scores: List of confidence scores
    """
    coco_keypoints = []
    scores = []
    
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        
        keypoints = [0] * 17 * 3
        # Map from our 18-keypoint format to COCO 17-keypoint format
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:  # no 'neck' in COCO
                continue
            
            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            
            coco_idx = to_coco_map[position_id]
            if coco_idx >= 0:
                keypoints[coco_idx * 3 + 0] = cx
                keypoints[coco_idx * 3 + 1] = cy
                keypoints[coco_idx * 3 + 2] = visibility
        
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
    
    return coco_keypoints, scores


def visualize_keypoints(img, coco_keypoints, window_name='Medium Pose Estimation'):
    """Visualize detected keypoints on image"""
    vis_img = img.copy()
    
    # COCO skeleton connections
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    
    colors = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
        (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
        (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
        (255, 0, 255), (255, 0, 170), (255, 0, 85)
    ]
    
    for keypoints in coco_keypoints:
        # Draw skeleton connections
        for idx, (start, end) in enumerate(skeleton):
            start_idx = (start - 1) * 3
            end_idx = (end - 1) * 3
            if (keypoints[start_idx + 2] > 0 and keypoints[end_idx + 2] > 0):
                start_pt = (int(keypoints[start_idx]), int(keypoints[start_idx + 1]))
                end_pt = (int(keypoints[end_idx]), int(keypoints[end_idx + 1]))
                cv2.line(vis_img, start_pt, end_pt, colors[idx % len(colors)], 2)
        
        # Draw keypoints
        for idx in range(len(keypoints) // 3):
            if keypoints[idx * 3 + 2] > 0:
                x, y = int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])
                cv2.circle(vis_img, (x, y), 4, (255, 0, 255), -1)
                cv2.circle(vis_img, (x, y), 2, (255, 255, 255), -1)
    
    cv2.imshow(window_name, vis_img)
    key = cv2.waitKey(0)
    return key


def run_coco_eval(gt_file_path, dt_file_path, log_file=None):
    """Run COCO evaluation and log results"""
    log('Running COCO keypoint evaluation...', log_file)
    
    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)
    
    result = COCOeval(coco_gt, coco_dt, 'keypoints')
    result.evaluate()
    result.accumulate()
    result.summarize()
    
    # Extract key metrics
    ap = result.stats[0]  # AP @ IoU=0.50:0.95
    ap_50 = result.stats[1]  # AP @ IoU=0.50
    ap_75 = result.stats[2]  # AP @ IoU=0.75
    
    log('', log_file)
    log('=== VALIDATION RESULTS ===', log_file)
    log(f'AP @ IoU=0.50:0.95: {ap:.4f}', log_file)
    log(f'AP @ IoU=0.50:      {ap_50:.4f}', log_file)
    log(f'AP @ IoU=0.75:      {ap_75:.4f}', log_file)
    
    return result.stats


def evaluate(args, net, log_file=None):
    """
    Evaluate model on COCO validation set
    
    Args:
        args: Command line arguments
        net: Trained model
        log_file: Path to log file
    """
    net.eval()
    
    # Load COCO validation annotations
    annotations_path = os.path.join(args.coco_path, 'annotations_trainval2017', 
                                   'annotations', 'person_keypoints_val2017.json')
    images_folder = os.path.join(args.coco_path, 'val2017')
    
    if not os.path.exists(annotations_path):
        log(f'ERROR: Annotations not found at {annotations_path}', log_file)
        log('Please ensure COCO validation data is available', log_file)
        sys.exit(1)
    
    if not os.path.exists(images_folder):
        log(f'ERROR: Images folder not found at {images_folder}', log_file)
        sys.exit(1)
    
    log(f'Loading COCO annotations from {annotations_path}', log_file)
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    
    # Apply subset if specified
    if args.subset and args.subset < len(images):
        images = images[:args.subset]
        log(f'Using subset of {args.subset} images', log_file)
    else:
        log(f'Using full validation set: {len(images)} images', log_file)
    
    # Process images
    coco_result = []
    start_time = time.time()
    stride = 8
    
    log('Starting validation...', log_file)
    log('', log_file)
    
    for idx, image_info in enumerate(images):
        file_name = image_info['file_name']
        image_id = image_info['id']
        img_path = os.path.join(images_folder, file_name)
        
        if not os.path.exists(img_path):
            log(f'WARNING: Image not found: {img_path}', log_file)
            continue
        
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            log(f'WARNING: Failed to load image: {img_path}', log_file)
            continue
        
        # Run inference
        avg_heatmaps, avg_pafs, scale = infer(net, img, input_size=args.input_size, stride=stride)
        
        # Extract keypoints
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th is background
            total_keypoints_num += extract_keypoints(
                avg_heatmaps[:, :, kpt_idx], 
                all_keypoints_by_type, 
                total_keypoints_num
            )
        
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)
        coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints, scale)
        
        # Add to results
        for kpt_idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': coco_keypoints[kpt_idx],
                'score': scores[kpt_idx]
            })
        
        # Visualization
        if args.visualize:
            key = visualize_keypoints(img, coco_keypoints)
            if key == 27:  # ESC key
                log('Visualization interrupted by user', log_file)
                break
        
        # Progress logging
        if (idx + 1) % args.log_every == 0 or (idx + 1) == len(images):
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = len(images) - (idx + 1)
            eta = avg_time * remaining
            
            persons_detected = len([r for r in coco_result if r['image_id'] == image_id])
            
            log(f'Progress: {idx+1}/{len(images)} images | '
                f'Detected: {persons_detected} persons | '
                f'Time: {avg_time:.2f}s/img | '
                f'ETA: {eta/60:.1f}min', log_file)
    
    total_time = time.time() - start_time
    log('', log_file)
    log(f'Inference complete: {len(images)} images in {total_time/60:.1f} minutes', log_file)
    log(f'Average time per image: {total_time/len(images):.2f}s', log_file)
    log(f'Total detections: {len(coco_result)}', log_file)
    
    # Save results
    log('', log_file)
    log(f'Saving results to {args.output}', log_file)
    with open(args.output, 'w') as f:
        json.dump(coco_result, f, indent=4)
    
    # Run COCO evaluation only if we have detections
    if len(coco_result) == 0:
        log('', log_file)
        log('WARNING: No detections found! Cannot run COCO evaluation.', log_file)
        log('This likely means the model needs more training or there is an issue.', log_file)
        return None
    
    # Run COCO evaluation
    log('', log_file)
    stats = run_coco_eval(annotations_path, args.output, log_file)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Medium Pose Estimation Validation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (e.g., pose_med2_train/best.pth)')
    parser.add_argument('--coco-path', type=str, 
                       default='/home/jkal/Desktop/MLonMCU/lightweight-human-pose-estimation.pytorch/coco',
                       help='Path to COCO dataset root')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for detections (default: auto-generated)')
    parser.add_argument('--input-size', type=int, default=128,
                       help='Input image size (default: 128, matching training)')
    parser.add_argument('--subset', type=int, default=None,
                       help='Use subset of validation images (default: use all)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize detected keypoints (press any key to continue, ESC to stop)')
    parser.add_argument('--log-every', type=int, default=50,
                       help='Log progress every N images (default: 50)')
    args = parser.parse_args()
    
    # Auto-generate output name if not specified
    if args.output is None:
        checkpoint_name = os.path.basename(os.path.dirname(args.checkpoint))
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'val_results_{checkpoint_name}_{timestamp}.json'
    
    # Create log file
    log_dir = os.path.dirname(args.output) or '.'
    log_file = os.path.join(log_dir, args.output.replace('.json', '_log.txt'))
    
    log('=== MEDIUM POSE VALIDATION ===', log_file)
    log(f'Checkpoint: {args.checkpoint}', log_file)
    log(f'COCO path: {args.coco_path}', log_file)
    log(f'Input size: {args.input_size}x{args.input_size}', log_file)
    log(f'Output: {args.output}', log_file)
    log(f'Log file: {log_file}', log_file)
    log(f'Visualization: {"Enabled" if args.visualize else "Disabled"}', log_file)
    log('', log_file)
    
    # Load model
    log(f'Loading model from {args.checkpoint}...', log_file)
    if not os.path.exists(args.checkpoint):
        log(f'ERROR: Checkpoint not found at {args.checkpoint}', log_file)
        sys.exit(1)
    
    net = MediumStudentModel()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'model' in checkpoint:
        net.load_state_dict(checkpoint['model'])
        if 'batch' in checkpoint:
            log(f'Loaded checkpoint from batch {checkpoint["batch"]}', log_file)
        if 'loss' in checkpoint:
            log(f'Training loss: {checkpoint["loss"]:.4f}', log_file)
    else:
        net.load_state_dict(checkpoint)
    
    num_params = sum(p.numel() for p in net.parameters())
    log(f'Model parameters: {num_params:,}', log_file)
    log(f'Model size: {num_params * 4 / (1024*1024):.2f}MB (FP32)', log_file)
    log('', log_file)
    
    # Run evaluation
    try:
        stats = evaluate(args, net, log_file)
        log('', log_file)
        log('Validation complete!', log_file)
    except KeyboardInterrupt:
        log('', log_file)
        log('Validation interrupted by user', log_file)
        sys.exit(1)
    except Exception as e:
        log('', log_file)
        log(f'ERROR during validation: {str(e)}', log_file)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
