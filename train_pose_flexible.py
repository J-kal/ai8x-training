#!/usr/bin/env python3
"""
Fast Pose Estimation Training (CPU or GPU) with Batch-Level Checkpointing.

Key differences vs train_pose_cpu.py:
- Select device via flag: --device gpu|cpu (default: gpu; falls back to cpu if CUDA unavailable)
- Pass teacher checkpoint, labels file, and image folder directly.
- Same fast logging and batch-level checkpoint cadence.
"""

import argparse
import copy
import datetime
import json
import os
import sys
import time
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Repo paths (used for defaults) - Colab-friendly
POSE_REPO = '/content/lightweight-human-pose-estimation.pytorch'
AI8X_REPO = '/content/ai8x-training'
sys.path.insert(0, POSE_REPO)
sys.path.insert(0, AI8X_REPO)

import ai8x
ai8x.set_device(device=85, simulate=False, round_avg=False)
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


def log(msg, log_file=None):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"[{ts}] {msg}\n")


class TeacherWrapper(nn.Module):
    def __init__(self, teacher_model):
        super().__init__()
        self.teacher = teacher_model
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(368, 368), mode='bilinear', align_corners=False)
        out = self.teacher(x)
        h = nn.functional.interpolate(out[-2], size=(16, 16), mode='bilinear', align_corners=False)
        p = nn.functional.interpolate(out[-1], size=(16, 16), mode='bilinear', align_corners=False)
        return [h, p]


class StudentModel(nn.Module):
    """Compact student model using ai8x layers"""
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


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, labels_path, images_folder, size=128):
        import pickle
        self.images_folder = images_folder
        self.size = size
        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import cv2, numpy as np, math
        label = copy.deepcopy(self.labels[idx])
        img = cv2.imread(os.path.join(self.images_folder, label['img_paths']))
        if img is None:
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            label['keypoints'] = [[0, 0, 2]] * 18

        # Resize
        h, w = img.shape[:2]
        s = self.size / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
        pad_h, pad_w = self.size - img.shape[0], self.size - img.shape[1]
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(128, 128, 128))

        # Scale keypoints
        for kp in label['keypoints']:
            kp[0] *= s; kp[1] *= s

        # Generate heatmaps (19)
        out_s = self.size // 8
        stride = 8
        sigma = 7
        heatmaps = np.zeros((19, out_s, out_s), dtype=np.float32)
        for ki, kp in enumerate(label['keypoints'][:18]):
            if len(kp) >= 3 and kp[2] <= 1:
                x, y = kp[0] / stride, kp[1] / stride
                if 0 <= x < out_s and 0 <= y < out_s:
                    n_sigma = 4
                    tl_x = max(0, int(x - n_sigma * sigma))
                    tl_y = max(0, int(y - n_sigma * sigma))
                    br_x = min(out_s, int(x + n_sigma * sigma))
                    br_y = min(out_s, int(y + n_sigma * sigma))
                    shift = stride / 2 - 0.5
                    for gy in range(tl_y, br_y):
                        for gx in range(tl_x, br_x):
                            d2 = ((gx * stride + shift - kp[0]) ** 2 +
                                  (gy * stride + shift - kp[1]) ** 2)
                            exponent = d2 / 2 / sigma / sigma
                            if exponent <= 4.6052:  # exp(-4.6052) ≈ 0.01
                                heatmaps[ki, gy, gx] = max(heatmaps[ki, gy, gx],
                                                           math.exp(-exponent))
                                if heatmaps[ki, gy, gx] > 1:
                                    heatmaps[ki, gy, gx] = 1
        heatmaps[18] = 1 - heatmaps[:18].max(0)

        # PAFs (38) - Generate proper Part Affinity Fields
        BODY_PARTS_KPT_IDS = [
            [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [2, 16],
            [1, 5], [5, 6], [6, 7], [5, 17], [1, 0], [0, 14], [0, 15], [14, 16], [15, 17]
        ]
        paf_thickness = 1
        pafs = np.zeros((38, out_s, out_s), dtype=np.float32)
        for paf_idx in range(len(BODY_PARTS_KPT_IDS)):
            kpt_ids = BODY_PARTS_KPT_IDS[paf_idx]
            if (kpt_ids[0] < len(label['keypoints']) and
                    kpt_ids[1] < len(label['keypoints'])):
                kp_a = label['keypoints'][kpt_ids[0]]
                kp_b = label['keypoints'][kpt_ids[1]]
                if (len(kp_a) >= 3 and len(kp_b) >= 3 and
                        kp_a[2] <= 1 and kp_b[2] <= 1):
                    x_a, y_a = kp_a[0] / stride, kp_a[1] / stride
                    x_b, y_b = kp_b[0] / stride, kp_b[1] / stride
                    x_ba = x_b - x_a
                    y_ba = y_b - y_a
                    norm_ba = (x_ba * x_ba + y_ba * y_ba) ** 0.5
                    if norm_ba >= 1e-7:
                        x_ba /= norm_ba
                        y_ba /= norm_ba
                        x_min = int(max(min(x_a, x_b) - paf_thickness, 0))
                        x_max = int(min(max(x_a, x_b) + paf_thickness, out_s))
                        y_min = int(max(min(y_a, y_b) - paf_thickness, 0))
                        y_max = int(min(max(y_a, y_b) + paf_thickness, out_s))
                        for y in range(y_min, y_max):
                            for x in range(x_min, x_max):
                                x_ca = x - x_a
                                y_ca = y - y_a
                                d = abs(x_ca * y_ba - y_ca * x_ba)
                                if d <= paf_thickness:
                                    pafs[paf_idx * 2, y, x] = x_ba
                                    pafs[paf_idx * 2 + 1, y, x] = y_ba

        # Normalize image
        img = (img.astype(np.float32) - 128) / 128
        img = img.transpose(2, 0, 1)

        return torch.tensor(img), torch.tensor(heatmaps), torch.tensor(pafs)


def configure_workers(device_choice):
    # Keep CPU-friendly defaults; lighten pinning threads for GPU case.
    num_cores = multiprocessing.cpu_count()
    compute_threads = max(1, num_cores - 4)
    data_workers = min(8, num_cores // 2)
    if device_choice == 'cpu':
        torch.set_num_threads(compute_threads)
        os.environ['OMP_NUM_THREADS'] = str(compute_threads)
        os.environ['MKL_NUM_THREADS'] = str(compute_threads)
        return data_workers
    return min(8, num_cores)  # GPU: prioritize loader over compute threads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help='Training device (default: gpu)')
    parser.add_argument('--subset', type=int, default=5000, help='Dataset size (default: 5000)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--save-every', type=int, default=100, help='Save every N batches')
    parser.add_argument('--total-batches', type=int, default=5000, help='Total batches to train')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (deprecated, use --checkpoint)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to start from. If not provided, will auto-detect in output dir')
    parser.add_argument('--output', type=str, default='pose_flexible_checkpoints')
    parser.add_argument('--teacher', type=str, default=os.path.join(AI8X_REPO, 'models/checkpoint_iter_370000.pth'),
                        help='Path to teacher checkpoint')
    parser.add_argument('--dataset-root', type=str, default=os.path.join(POSE_REPO, 'coco'),
                        help='Root folder containing images/labels (default: pose repo coco)')
    parser.add_argument('--labels', type=str, default=None, help='Path to labels pickle (overrides dataset-root default)')
    parser.add_argument('--images', type=str, default=None, help='Path to images folder (overrides dataset-root default)')
    parser.add_argument('--num-workers', type=int, default=None, help='Dataloader workers (default auto)')
    args = parser.parse_args()

    # Resolve device
    requested = args.device
    if requested == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        if requested == 'gpu' and not torch.cuda.is_available():
            log("CUDA not available, falling back to CPU")
        device = torch.device('cpu')

    # Data workers / threads
    auto_workers = configure_workers('gpu' if device.type == 'cuda' else 'cpu')
    data_workers = args.num_workers if args.num_workers is not None else auto_workers

    os.makedirs(args.output, exist_ok=True)
    log_file = os.path.join(args.output, 'log.txt')

    # Determine checkpoint path: explicit > resume > auto-detect
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.resume:
        checkpoint_path = args.resume
    else:
        latest_path = os.path.join(args.output, 'latest.pth')
        best_path = os.path.join(args.output, 'best.pth')
        if os.path.exists(latest_path):
            checkpoint_path = latest_path
        elif os.path.exists(best_path):
            checkpoint_path = best_path

    log("=== FLEXIBLE POSE TRAINING ===", log_file)
    log(f"Device: {device}", log_file)
    log(f"Subset: {args.subset}, Batch: {args.batch_size}, Save every: {args.save_every}", log_file)
    log(f"Teacher: {args.teacher}", log_file)

    # Dataset paths
    labels_path = args.labels or os.path.join(args.dataset_root, 'prepared_train_annotations.pkl')
    images_folder = args.images or os.path.join(args.dataset_root, 'train2017')

    # Dataset
    dataset = SimpleDataset(labels_path, images_folder)
    dataset = Subset(dataset, list(range(min(args.subset, len(dataset)))))
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=data_workers,
                        pin_memory=(device.type == 'cuda'),
                        prefetch_factor=2 if data_workers > 0 else None,
                        persistent_workers=(data_workers > 0))
    log(f"Dataset: {len(dataset)} samples, {len(loader)} batches per epoch", log_file)

    # Models
    student = StudentModel().to(device)
    log(f"Student params: {sum(p.numel() for p in student.parameters()):,}", log_file)

    teacher_net = PoseEstimationWithMobileNet(num_refinement_stages=1, num_channels=128)
    ckpt = torch.load(args.teacher, map_location='cpu')
    load_state(teacher_net, ckpt)
    teacher = TeacherWrapper(teacher_net).to(device)
    teacher.eval()
    log("Teacher loaded", log_file)

    optimizer = optim.Adam(student.parameters(), lr=args.lr)

    # Load checkpoint if available
    start_batch = 0
    best_loss = float('inf')
    is_resuming = False
    if checkpoint_path and os.path.exists(checkpoint_path):
        is_resuming = True
        log("", log_file)
        log("=== RESUMING TRAINING ===", log_file)
        log(f"Loading checkpoint from {checkpoint_path}...", log_file)
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        student.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_batch = ckpt.get('batch', 0)
        best_loss = ckpt.get('best_loss', float('inf'))
        log(f"Resumed from batch {start_batch}, best loss: {best_loss:.4f}", log_file)
    elif checkpoint_path:
        log(f"Warning: Checkpoint path specified but not found: {checkpoint_path}", log_file)
        log("Starting fresh training...", log_file)

    # Training
    if not is_resuming:
        log(f"Starting training from batch {start_batch}...", log_file)
    else:
        log(f"Continuing training from batch {start_batch}...", log_file)

    student.train()
    batch_num = start_batch
    epoch = 0

    running_loss = 0
    running_student = 0
    running_distill = 0
    batch_times = []
    training_start = time.time()

    while batch_num < args.total_batches:
        epoch += 1
        for imgs, heatmaps, pafs in loader:
            batch_start = time.time()
            batch_num += 1
            if batch_num > args.total_batches:
                break

            imgs = imgs.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            pafs = pafs.to(device, non_blocking=True)

            # Forward
            out = student(imgs)
            with torch.no_grad():
                t_out = teacher(imgs)

            batch_size = imgs.shape[0]
            heatmap_mask = torch.ones_like(heatmaps)
            paf_mask = torch.ones_like(pafs)

            # Student loss: Masked L2 loss
            heatmap_diff = (out[0] - heatmaps) * heatmap_mask
            heatmap_loss = (heatmap_diff * heatmap_diff).sum() / 2 / batch_size

            paf_diff = (out[1] - pafs) * paf_mask
            paf_loss = (paf_diff * paf_diff).sum() / 2 / batch_size

            student_loss = heatmap_loss + paf_loss

            # Distillation loss
            distill_heatmap_diff = (out[0] - t_out[0].detach())
            distill_heatmap_loss = (distill_heatmap_diff * distill_heatmap_diff).sum() / 2 / batch_size

            distill_paf_diff = (out[1] - t_out[1].detach())
            distill_paf_loss = (distill_paf_diff * distill_paf_diff).sum() / 2 / batch_size

            distill_loss = distill_heatmap_loss + distill_paf_loss

            # Combined loss
            loss = 0.3 * student_loss + 0.7 * distill_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            running_loss += loss.item()
            running_student += student_loss.item()
            running_distill += distill_loss.item()

            # Log every 10 batches with full details (or every batch for first 20)
            should_log = batch_num % 10 == 0 or batch_num <= 20
            if should_log:
                if batch_num <= 20:
                    avg_loss = loss.item()
                    avg_student = student_loss.item()
                    avg_distill = distill_loss.item()
                    avg_time = batch_time
                    remaining = args.total_batches - batch_num
                    eta_sec = avg_time * remaining
                    eta_str = f"{eta_sec/3600:.1f}h" if eta_sec > 3600 else f"{eta_sec/60:.1f}m"

                    log(f"Batch {batch_num}/{args.total_batches} | "
                        f"Loss: {avg_loss:.4f} | Student: {avg_student:.4f} | "
                        f"Distill: {avg_distill:.4f} | Time: {avg_time:.1f}s | ETA: {eta_str}", log_file)
                else:
                    avg_loss = running_loss / 10
                    avg_student = running_student / 10
                    avg_distill = running_distill / 10
                    avg_time = sum(batch_times[-10:]) / min(10, len(batch_times))
                    remaining = args.total_batches - batch_num
                    eta_sec = avg_time * remaining
                    eta_str = f"{eta_sec/3600:.1f}h" if eta_sec > 3600 else f"{eta_sec/60:.1f}m"

                    log(f"Batch {batch_num}/{args.total_batches} | "
                        f"Loss: {avg_loss:.4f} | Student: {avg_student:.4f} | "
                        f"Distill: {avg_distill:.4f} | Time: {avg_time:.1f}s | ETA: {eta_str}", log_file)

                    running_loss = 0
                    running_student = 0
                    running_distill = 0

            # Save checkpoint
            if batch_num % args.save_every == 0:
                ckpt = {
                    'batch': batch_num,
                    'model': student.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss.item(),
                    'best_loss': best_loss,
                }
                path = os.path.join(args.output, f'batch_{batch_num:06d}.pth')
                torch.save(ckpt, path)
                torch.save(ckpt, os.path.join(args.output, 'latest.pth'))

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(ckpt, os.path.join(args.output, 'best.pth'))
                    log(f"  *** Saved BEST at batch {batch_num} ***", log_file)
                else:
                    log(f"  Saved checkpoint at batch {batch_num}", log_file)

    log(f"Training complete! Best loss: {best_loss:.4f}", log_file)
    log(f"Models saved to: {args.output}/", log_file)


if __name__ == '__main__':
    main()

