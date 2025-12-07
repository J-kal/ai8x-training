#!/usr/bin/env python3
"""
Tiny Pose Estimation Training - Second-Level Knowledge Distillation

Distills from trained student model (pose_med2_train/best.pth) into an even smaller model.
Target: ~2MB model size (approximately 400K-500K parameters).

Usage:
    python train_pose_tiny.py --subset 15000 --save-every 100 --total-batches 10000
"""

import argparse, copy, datetime, json, os, sys, time, multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# CPU Optimization
NUM_CORES = multiprocessing.cpu_count()
COMPUTE_THREADS = max(1, NUM_CORES - 4)  # Leave 4 for data loading
DATA_WORKERS = min(8, NUM_CORES // 2)
torch.set_num_threads(COMPUTE_THREADS)
os.environ['OMP_NUM_THREADS'] = str(COMPUTE_THREADS)
os.environ['MKL_NUM_THREADS'] = str(COMPUTE_THREADS)

# Paths
POSE_REPO = '/home/jkal/Desktop/MLonMCU/lightweight-human-pose-estimation.pytorch'
AI8X_REPO = '/home/jkal/Desktop/MLonMCU/ext/ai8x-training'
sys.path.insert(0, POSE_REPO)
sys.path.insert(0, AI8X_REPO)

import ai8x
# QAT is enabled automatically by AI8X layers - weights stay FP32 but quantization is simulated
# simulate=False: Real quantization simulation during forward pass (QAT mode)
# round_avg=False: Disables rounding in average pooling for better accuracy
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
    """Wrapper for the first-level student model (now acting as teacher)"""
    def __init__(self, teacher_model):
        super().__init__()
        self.teacher = teacher_model
        for p in self.teacher.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        # Teacher already outputs at 16x16, no interpolation needed
        out = self.teacher(x)
        return out  # [heatmaps, pafs] already at correct size


class TinyStudentModel(nn.Module):
    """
    Ultra-compact student model targeting ~2MB size (~400K-500K parameters)
    
    Architecture reductions:
    - Fewer channels throughout (max 64 vs 128)
    - Simplified CPM (2 layers vs 3)
    - Smaller head channels (32 vs 64)
    - Removed some intermediate layers
    """
    def __init__(self):
        super().__init__()
        # Backbone - reduced channels
        self.conv1 = ai8x.FusedConv2dBNReLU(3, 16, 3, stride=2, padding=1, bias=True)
        self.conv2 = ai8x.FusedConv2dBNReLU(16, 24, 3, stride=1, padding=1, bias=True)
        self.conv3 = ai8x.FusedConv2dBNReLU(24, 32, 3, stride=1, padding=1, bias=True)
        self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(32, 32, 3, pool_size=2, pool_stride=2, padding=1, bias=True)
        self.conv5 = ai8x.FusedConv2dBNReLU(32, 48, 3, stride=1, padding=1, bias=True)
        self.conv6 = ai8x.FusedMaxPoolConv2dBNReLU(48, 48, 3, pool_size=2, pool_stride=2, padding=1, bias=True)
        self.conv7 = ai8x.FusedConv2dBNReLU(48, 64, 3, stride=1, padding=1, bias=True)
        # Simplified CPM (2 layers instead of 3)
        self.cpm1 = ai8x.FusedConv2dBNReLU(64, 48, 1, padding=0, bias=True)
        self.cpm2 = ai8x.FusedConv2dBNReLU(48, 48, 3, padding=1, bias=True)
        # Heads - smaller channels
        self.heat_conv = ai8x.FusedConv2dBNReLU(48, 32, 1, padding=0, bias=True)
        self.heat_out = ai8x.Conv2d(32, 19, 1, padding=0, bias=True, wide=True)
        self.paf_conv = ai8x.FusedConv2dBNReLU(48, 32, 1, padding=0, bias=True)
        self.paf_out = ai8x.Conv2d(32, 38, 1, padding=0, bias=True, wide=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.cpm1(x)
        x = self.cpm2(x)
        h = self.heat_out(self.heat_conv(x))
        p = self.paf_out(self.paf_conv(x))
        return [h, p]


class SimpleDataset(torch.utils.data.Dataset):
    """Same dataset as train_pose_cpu.py"""
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
            label['keypoints'] = [[0,0,2]]*18
        
        # Resize
        h, w = img.shape[:2]
        s = self.size / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)))
        pad_h, pad_w = self.size - img.shape[0], self.size - img.shape[1]
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(128,128,128))
        
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
                x, y = kp[0]/stride, kp[1]/stride
                if 0 <= x < out_s and 0 <= y < out_s:
                    n_sigma = 4
                    tl_x = max(0, int(x - n_sigma * sigma))
                    tl_y = max(0, int(y - n_sigma * sigma))
                    br_x = min(out_s, int(x + n_sigma * sigma))
                    br_y = min(out_s, int(y + n_sigma * sigma))
                    shift = stride / 2 - 0.5
                    for gy in range(tl_y, br_y):
                        for gx in range(tl_x, br_x):
                            d2 = ((gx * stride + shift - kp[0])**2 + 
                                  (gy * stride + shift - kp[1])**2)
                            exponent = d2 / 2 / sigma / sigma
                            if exponent <= 4.6052:
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
                    x_a, y_a = kp_a[0]/stride, kp_a[1]/stride
                    x_b, y_b = kp_b[0]/stride, kp_b[1]/stride
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


def main():
    parser = argparse.ArgumentParser(description='Tiny Pose Estimation Training - Second-Level Distillation')
    parser.add_argument('--subset', type=int, default=50000, help='Dataset size (default: 50000, use larger for better results)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--save-every', type=int, default=300, help='Save every N batches (default: 300)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train (if specified, overrides --total-batches)')
    parser.add_argument('--total-batches', type=int, default=None, help='Total batches to train (ignored if --epochs specified)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--teacher-checkpoint', type=str, 
                       default='/home/jkal/Desktop/MLonMCU/ext/ai8x-training/pose_med2_train/best.pth',
                       help='Path to teacher model checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to checkpoint to start from. If not provided, will check for latest.pth or best.pth')
    parser.add_argument('--output', type=str, default='pose_tiny_train', help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    log_file = os.path.join(args.output, 'log.txt')
    
    # Determine checkpoint path: explicit > auto-detect
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Auto-detect: check for latest.pth or best.pth in output directory
        latest_path = os.path.join(args.output, 'latest.pth')
        best_path = os.path.join(args.output, 'best.pth')
        if os.path.exists(latest_path):
            checkpoint_path = latest_path
        elif os.path.exists(best_path):
            checkpoint_path = best_path
    
    log(f"=== TINY POSE TRAINING - SECOND-LEVEL DISTILLATION ===", log_file)
    log(f"CPU Cores: {NUM_CORES}, Compute threads: {COMPUTE_THREADS}, Data workers: {DATA_WORKERS}", log_file)
    log(f"Subset: {args.subset}, Batch: {args.batch_size}, Save every: {args.save_every}", log_file)
    log(f"Teacher checkpoint: {args.teacher_checkpoint}", log_file)
    
    # Dataset
    data_path = os.path.join(POSE_REPO, 'coco')
    dataset = SimpleDataset(
        os.path.join(data_path, 'prepared_train_annotations.pkl'),
        os.path.join(data_path, 'train2017')
    )
    dataset = Subset(dataset, list(range(min(args.subset, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                        num_workers=DATA_WORKERS, pin_memory=False, 
                        prefetch_factor=2, persistent_workers=True)
    batches_per_epoch = len(loader)
    log(f"Dataset: {len(dataset)} samples, {batches_per_epoch} batches per epoch", log_file)
    
    # Calculate total batches from epochs if specified
    num_epochs = None
    if args.epochs is not None:
        num_epochs = args.epochs
        total_batches = args.epochs * batches_per_epoch
        log(f"Training for {args.epochs} epochs = {total_batches} total batches", log_file)
    elif args.total_batches is not None:
        total_batches = args.total_batches
        epochs_equivalent = total_batches / batches_per_epoch
        num_epochs = int(epochs_equivalent) + (1 if total_batches % batches_per_epoch > 0 else 0)
        log(f"Training for {total_batches} batches = ~{epochs_equivalent:.1f} epochs", log_file)
    else:
        # Default: 10 epochs
        num_epochs = 10
        total_batches = num_epochs * batches_per_epoch
        log(f"Using default: {num_epochs} epochs = {total_batches} total batches", log_file)
    
    # Models
    student = TinyStudentModel()
    num_params = sum(p.numel() for p in student.parameters())
    model_size_mb = num_params * 4 / (1024 * 1024)  # FP32 size
    quantized_size_mb = num_params * 1 / (1024 * 1024)  # 8-bit quantized size
    log(f"Tiny Student params: {num_params:,}", log_file)
    log(f"Model size: {model_size_mb:.2f}MB (FP32), {quantized_size_mb:.2f}MB (8-bit quantized)", log_file)
    log(f"Note: Model uses QAT - weights are FP32 but quantization is simulated during training", log_file)
    log(f"      This prepares the model for 8-bit quantization after training completes", log_file)
    
    # Load teacher (first-level student model)
    log(f"Loading teacher model from {args.teacher_checkpoint}...", log_file)
    if not os.path.exists(args.teacher_checkpoint):
        log(f"ERROR: Teacher checkpoint not found at {args.teacher_checkpoint}", log_file)
        sys.exit(1)
    
    # Define first-level student model architecture (same as train_pose_cpu.py)
    class FirstLevelStudent(nn.Module):
        """First-level student model (now acting as teacher)"""
        def __init__(self):
            super().__init__()
            self.conv1 = ai8x.FusedConv2dBNReLU(3, 32, 3, stride=2, padding=1, bias=True)
            self.conv2 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1, bias=True)
            self.conv3 = ai8x.FusedConv2dBNReLU(32, 64, 3, stride=1, padding=1, bias=True)
            self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=True)
            self.conv5 = ai8x.FusedConv2dBNReLU(64, 96, 3, stride=1, padding=1, bias=True)
            self.conv6 = ai8x.FusedMaxPoolConv2dBNReLU(96, 96, 3, pool_size=2, pool_stride=2, padding=1, bias=True)
            self.conv7 = ai8x.FusedConv2dBNReLU(96, 128, 3, stride=1, padding=1, bias=True)
            self.conv8 = ai8x.FusedConv2dBNReLU(128, 128, 3, stride=1, padding=1, bias=True)
            self.cpm1 = ai8x.FusedConv2dBNReLU(128, 96, 1, padding=0, bias=True)
            self.cpm2 = ai8x.FusedConv2dBNReLU(96, 96, 3, padding=1, bias=True)
            self.cpm3 = ai8x.FusedConv2dBNReLU(96, 96, 3, padding=1, bias=True)
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
    
    teacher_model = FirstLevelStudent()
    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location='cpu')
    teacher_model.load_state_dict(teacher_ckpt['model'])
    teacher = TeacherWrapper(teacher_model)
    teacher.eval()
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    log(f"Teacher loaded (first-level student with {teacher_params:,} params)", log_file)
    
    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    
    # Load checkpoint if available
    start_batch = 0
    best_loss = float('inf')
    is_resuming = False
    if checkpoint_path and os.path.exists(checkpoint_path):
        is_resuming = True
        log(f"", log_file)  # Empty line for separation
        log(f"=== RESUMING TRAINING ===", log_file)
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
        log(f"Starting fresh training...", log_file)
    
    # Training
    if not is_resuming:
        start_epoch = start_batch // batches_per_epoch
        log(f"Starting training from batch {start_batch} (epoch {start_epoch})...", log_file)
    else:
        start_epoch = start_batch // batches_per_epoch
        log(f"Continuing training from batch {start_batch} (epoch {start_epoch})...", log_file)
    student.train()
    batch_num = start_batch
    epoch = start_epoch
    
    # Tracking for detailed logging
    running_loss = 0
    running_student = 0
    running_distill = 0
    batch_times = []
    training_start = time.time()
    epoch_losses = []
    
    while batch_num < total_batches:
        epoch += 1
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_batches = 0
        
        log(f"Epoch {epoch}/{num_epochs if num_epochs else '?'} starting...", log_file)
        
        for imgs, heatmaps, pafs in loader:
            batch_start = time.time()
            batch_num += 1
            epoch_batches += 1
            if batch_num > total_batches:
                break
            
            # Forward
            out = student(imgs)
            with torch.no_grad():
                t_out = teacher(imgs)
            
            # Create masks (all ones for now)
            batch_size = imgs.shape[0]
            heatmap_mask = torch.ones_like(heatmaps)
            paf_mask = torch.ones_like(pafs)
            
            # Student loss: Masked L2 loss (proper normalization)
            # Heatmap loss
            heatmap_diff = (out[0] - heatmaps) * heatmap_mask
            heatmap_loss = (heatmap_diff * heatmap_diff).sum() / 2 / batch_size
            
            # PAF loss
            paf_diff = (out[1] - pafs) * paf_mask
            paf_loss = (paf_diff * paf_diff).sum() / 2 / batch_size
            
            student_loss = heatmap_loss + paf_loss
            
            # Distillation loss: L2 between student and teacher outputs
            # Teacher outputs are already at correct size (16x16)
            distill_heatmap_diff = (out[0] - t_out[0].detach())
            distill_heatmap_loss = (distill_heatmap_diff * distill_heatmap_diff).sum() / 2 / batch_size
            
            distill_paf_diff = (out[1] - t_out[1].detach())
            distill_paf_loss = (distill_paf_diff * distill_paf_diff).sum() / 2 / batch_size
            
            distill_loss = distill_heatmap_loss + distill_paf_loss
            
            # Combined loss - slightly more weight on distillation since teacher is already good
            loss = 0.25 * student_loss + 0.75 * distill_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            running_loss += loss.item()
            running_student += student_loss.item()
            running_distill += distill_loss.item()
            epoch_loss += loss.item()
            
            # Log every 10 batches with full details (or every batch for first 20)
            should_log = batch_num % 10 == 0 or batch_num <= 20
            if should_log:
                # For first 20 batches, log each batch; otherwise average over last 10
                if batch_num <= 20:
                    avg_loss = loss.item()
                    avg_student = student_loss.item()
                    avg_distill = distill_loss.item()
                    avg_time = batch_time
                    remaining = total_batches - batch_num
                    eta_sec = avg_time * remaining
                    eta_str = f"{eta_sec/3600:.1f}h" if eta_sec > 3600 else f"{eta_sec/60:.1f}m"
                    
                    log(f"Batch {batch_num}/{total_batches} | "
                        f"Loss: {avg_loss:.4f} | Student: {avg_student:.4f} | "
                        f"Distill: {avg_distill:.4f} | Time: {avg_time:.1f}s | ETA: {eta_str}", log_file)
                else:
                    avg_loss = running_loss / 10
                    avg_student = running_student / 10
                    avg_distill = running_distill / 10
                    avg_time = sum(batch_times[-10:]) / min(10, len(batch_times))
                    remaining = total_batches - batch_num
                    eta_sec = avg_time * remaining
                    eta_str = f"{eta_sec/3600:.1f}h" if eta_sec > 3600 else f"{eta_sec/60:.1f}m"
                    
                    log(f"Batch {batch_num}/{total_batches} | "
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
            
            # Break if we've reached total batches
            if batch_num >= total_batches:
                break
        
        # Log epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        epoch_losses.append(avg_epoch_loss)
        log(f"Epoch {epoch} complete | Avg Loss: {avg_epoch_loss:.4f} | Time: {epoch_time/60:.1f}m | "
            f"Batches: {epoch_batches}/{batches_per_epoch}", log_file)
        
        if batch_num >= total_batches:
            break
    
    total_time = time.time() - training_start
    log(f"", log_file)
    log(f"Training complete! Best loss: {best_loss:.4f}", log_file)
    log(f"Total training time: {total_time/3600:.2f} hours", log_file)
    log(f"Models saved to: {args.output}/", log_file)
    log(f"Final model size: {quantized_size_mb:.2f}MB (8-bit quantized)", log_file)


if __name__ == '__main__':
    main()

