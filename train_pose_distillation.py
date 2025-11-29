#!/usr/bin/env python3
###################################################################################################
#
# Pose Estimation Knowledge Distillation Training Script
#
# This is a standalone training script for pose estimation that demonstrates:
# 1. Loading the teacher model (PoseEstimationWithMobileNet)
# 2. Training a compact student model using knowledge distillation
# 3. Quantization-aware training (QAT)
# 4. Post-training pruning
#
# Usage:
#   python train_pose_distillation.py --mode kd      # Knowledge distillation
#   python train_pose_distillation.py --mode qat    # QAT fine-tuning
#   python train_pose_distillation.py --mode prune  # Pruning
#
###################################################################################################

import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add paths
sys.path.insert(0, '/home/jkal/Desktop/MLonMCU/lightweight-human-pose-estimation.pytorch')
sys.path.insert(0, '/home/jkal/Desktop/MLonMCU/ext/ai8x-training')

import ai8x
from datasets.pose_coco import PoseDatasetAI8X, pose_coco_collate_fn
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


class TeacherWrapper(nn.Module):
    """Wrapper for the teacher model with input size adaptation"""
    def __init__(self, teacher_model, student_output_size=(16, 16)):
        super().__init__()
        self.teacher = teacher_model
        self.student_output_size = student_output_size
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Resize input to teacher's expected size
        x_upsampled = nn.functional.interpolate(x, size=(368, 368), 
                                                 mode='bilinear', align_corners=False)
        outputs = self.teacher(x_upsampled)
        
        # Get last stage outputs and resize to match student
        heatmaps = outputs[-2]
        pafs = outputs[-1]
        
        heatmaps = nn.functional.interpolate(heatmaps, size=self.student_output_size,
                                             mode='bilinear', align_corners=False)
        pafs = nn.functional.interpolate(pafs, size=self.student_output_size,
                                         mode='bilinear', align_corners=False)
        
        return [heatmaps, pafs]


def create_student_model(model_name='small', device='cuda'):
    """Create the student model"""
    ai8x.set_device(84, False, False)  # MAX78000 settings
    
    if model_name == 'tiny':
        from models import ai85net_pose
        # Reload to get the model
        import importlib
        import models.ai85net_pose as pose_model
        importlib.reload(pose_model)
        model = pose_model.AI85PoseNetTiny(
            num_classes=57,
            num_channels=3,
            dimensions=(128, 128),
            bias=True
        )
    else:  # small
        from models import ai85net_pose
        import importlib
        import models.ai85net_pose as pose_model
        importlib.reload(pose_model)
        model = pose_model.AI85PoseNetSmall(
            num_classes=57,
            num_channels=3,
            dimensions=(128, 128),
            bias=True
        )
    
    return model.to(device)


def create_teacher_model(checkpoint_path, device='cuda'):
    """Create and load the teacher model"""
    teacher = PoseEstimationWithMobileNet(
        num_refinement_stages=1,
        num_channels=128,
        num_heatmaps=19,
        num_pafs=38
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    load_state(teacher, checkpoint)
    
    # Wrap teacher for input adaptation
    teacher_wrapped = TeacherWrapper(teacher, student_output_size=(16, 16))
    teacher_wrapped = teacher_wrapped.to(device)
    teacher_wrapped.eval()
    
    return teacher_wrapped


def pose_loss(pred_heatmaps, pred_pafs, gt_heatmaps, gt_pafs, 
              heatmap_mask, paf_mask, batch_size):
    """Masked L2 loss for pose estimation"""
    heatmap_loss = ((pred_heatmaps - gt_heatmaps) * heatmap_mask).pow(2).sum() / 2 / batch_size
    paf_loss = ((pred_pafs - gt_pafs) * paf_mask).pow(2).sum() / 2 / batch_size
    return heatmap_loss + paf_loss


def train_epoch_kd(student, teacher, dataloader, optimizer, device, 
                   student_weight=0.3, distill_weight=0.7):
    """Train one epoch with knowledge distillation"""
    student.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        gt_heatmaps = targets['keypoint_maps'].to(device)
        gt_pafs = targets['paf_maps'].to(device)
        heatmap_mask = targets['keypoint_mask'].to(device)
        paf_mask = targets['paf_mask'].to(device)
        
        batch_size = images.shape[0]
        
        # Forward pass
        student_output = student(images)
        student_heatmaps = student_output[0]
        student_pafs = student_output[1]
        
        with torch.no_grad():
            teacher_output = teacher(images)
            teacher_heatmaps = teacher_output[0]
            teacher_pafs = teacher_output[1]
        
        # Student loss (with ground truth)
        student_loss = pose_loss(student_heatmaps, student_pafs,
                                 gt_heatmaps, gt_pafs,
                                 heatmap_mask, paf_mask, batch_size)
        
        # Distillation loss (with teacher)
        distill_loss = (nn.functional.mse_loss(student_heatmaps, teacher_heatmaps) +
                       nn.functional.mse_loss(student_pafs, teacher_pafs))
        
        # Combined loss
        loss = student_weight * student_loss + distill_weight * distill_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def train_epoch_standard(model, dataloader, optimizer, device):
    """Standard training epoch (for QAT or pruning fine-tuning)"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        gt_heatmaps = targets['keypoint_maps'].to(device)
        gt_pafs = targets['paf_maps'].to(device)
        heatmap_mask = targets['keypoint_mask'].to(device)
        paf_mask = targets['paf_mask'].to(device)
        
        batch_size = images.shape[0]
        
        # Forward pass
        output = model(images)
        pred_heatmaps = output[0]
        pred_pafs = output[1]
        
        # Loss
        loss = pose_loss(pred_heatmaps, pred_pafs,
                        gt_heatmaps, gt_pafs,
                        heatmap_mask, paf_mask, batch_size)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validation loop"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            gt_heatmaps = targets['keypoint_maps'].to(device)
            gt_pafs = targets['paf_maps'].to(device)
            heatmap_mask = targets['keypoint_mask'].to(device)
            paf_mask = targets['paf_mask'].to(device)
            
            batch_size = images.shape[0]
            
            output = model(images)
            pred_heatmaps = output[0]
            pred_pafs = output[1]
            
            loss = pose_loss(pred_heatmaps, pred_pafs,
                           gt_heatmaps, gt_pafs,
                           heatmap_mask, paf_mask, batch_size)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Pose Estimation Training with KD, QAT, and Pruning')
    parser.add_argument('--mode', type=str, choices=['kd', 'qat', 'prune'], default='kd',
                        help='Training mode: kd (knowledge distillation), qat, or prune')
    parser.add_argument('--model', type=str, choices=['tiny', 'small'], default='small',
                        help='Student model size')
    parser.add_argument('--teacher-checkpoint', type=str,
                        default='/home/jkal/Desktop/MLonMCU/ext/ai8x-training/models/checkpoint_iter_370000.pth',
                        help='Path to teacher model checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to resume training from checkpoint')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--data-path', type=str,
                        default='/home/jkal/Desktop/MLonMCU/lightweight-human-pose-estimation.pytorch/coco',
                        help='Path to COCO data')
    parser.add_argument('--output-dir', type=str, default='pose_checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=== Pose Estimation Training ({args.mode.upper()}) ===")
    print(f"Student model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print()
    
    # Create dataset
    train_labels = os.path.join(args.data_path, 'prepared_train_annotations.pkl')
    train_images = os.path.join(args.data_path, 'train2017')
    
    if not os.path.exists(train_labels):
        print(f"ERROR: Training labels not found at {train_labels}")
        print("Please prepare training labels first using:")
        print("  python scripts/prepare_train_labels.py --labels <path_to_coco_annotations>")
        sys.exit(1)
    
    # Create a mock args object for dataset
    class MockArgs:
        act_mode_8bit = False
    
    dataset = PoseDatasetAI8X(
        labels_path=train_labels,
        images_folder=train_images,
        input_size=128,
        args=MockArgs(),
        augment=True
    )
    
    # Split for train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=pose_coco_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=pose_coco_collate_fn
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()
    
    # Create student model
    student = create_student_model(args.model, args.device)
    print(f"Student model parameters: {sum(p.numel() for p in student.parameters()):,}")
    
    # Load checkpoint if resuming
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        student.load_state_dict(checkpoint['model_state_dict'])
    
    # Mode-specific setup
    if args.mode == 'kd':
        # Knowledge distillation
        print(f"Loading teacher model from {args.teacher_checkpoint}")
        teacher = create_teacher_model(args.teacher_checkpoint, args.device)
        teacher_params = sum(p.numel() for p in teacher.teacher.parameters())
        print(f"Teacher model parameters: {teacher_params:,}")
        print(f"Compression ratio: {teacher_params / sum(p.numel() for p in student.parameters()):.1f}x")
        print()
        
        optimizer = optim.Adam(student.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    milestones=[30, 60, 90], 
                                                    gamma=0.5)
        
        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            train_loss = train_epoch_kd(student, teacher, train_loader, optimizer, 
                                        args.device, student_weight=0.3, distill_weight=0.7)
            val_loss = validate(student, val_loader, args.device)
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(args.output_dir, 'best.pth'))
                print(f"  New best model saved!")
    
    elif args.mode == 'qat':
        # Quantization-aware training
        print("Applying QAT...")
        ai8x.fuse_bn_layers(student)
        ai8x.initiate_qat(student, {'weight_bits': 8})
        
        optimizer = optim.Adam(student.parameters(), lr=args.lr * 0.1)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    milestones=[15, 30], 
                                                    gamma=0.5)
        
        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            train_loss = train_epoch_standard(student, train_loader, optimizer, args.device)
            val_loss = validate(student, val_loader, args.device)
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            
            torch.save(checkpoint, os.path.join(args.output_dir, f'qat_checkpoint_epoch_{epoch+1}.pth'))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(args.output_dir, 'qat_best.pth'))
                print(f"  New best QAT model saved!")
    
    elif args.mode == 'prune':
        # Post-training pruning (simplified - uses magnitude pruning)
        print("Applying pruning...")
        
        # Simple magnitude-based pruning
        import torch.nn.utils.prune as prune
        
        parameters_to_prune = []
        for name, module in student.named_modules():
            if isinstance(module, (nn.Conv2d, ai8x.FusedConv2dBNReLU)):
                if hasattr(module, 'op'):
                    parameters_to_prune.append((module.op, 'weight'))
                elif hasattr(module, 'weight'):
                    parameters_to_prune.append((module, 'weight'))
        
        if parameters_to_prune:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=0.3,  # Start with 30% sparsity
            )
            print(f"Applied 30% global L1 unstructured pruning")
        
        optimizer = optim.Adam(student.parameters(), lr=args.lr * 0.01)
        
        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            train_loss = train_epoch_standard(student, train_loader, optimizer, args.device)
            val_loss = validate(student, val_loader, args.device)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Increase sparsity gradually
            if epoch > 0 and epoch % 10 == 0:
                for module, name in parameters_to_prune:
                    prune.l1_unstructured(module, name=name, amount=0.1)
                print(f"  Increased sparsity by 10%")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(args.output_dir, 'pruned_best.pth'))
                print(f"  New best pruned model saved!")
        
        # Remove pruning reparametrization
        for module, name in parameters_to_prune:
            prune.remove(module, name)
        
        torch.save({
            'model_state_dict': student.state_dict(),
        }, os.path.join(args.output_dir, 'pruned_final.pth'))
        print(f"\nFinal pruned model saved!")
    
    print("\n=== Training Complete ===")
    print(f"Checkpoints saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()

