###################################################################################################
#
# Pose Estimation Loss Functions
#
# Custom loss functions for training pose estimation models with knowledge distillation.
#
###################################################################################################
"""
Pose Estimation Loss Functions for AI8X Training
"""
import torch
from torch import nn


class PoseLoss(nn.Module):
    """
    Combined loss for pose estimation with heatmaps and PAFs.
    Uses masked L2 loss to handle occluded keypoints.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
    
    def forward(self, output, target):
        """
        Args:
            output: List [heatmaps, pafs] or tuple from model
            target: Dict with keypoint_maps, paf_maps, keypoint_mask, paf_mask
        
        Returns:
            Combined loss value
        """
        if isinstance(output, (list, tuple)):
            pred_heatmaps = output[0]
            pred_pafs = output[1]
        else:
            # If single tensor, split it
            pred_heatmaps = output[:, :19]
            pred_pafs = output[:, 19:]
        
        # Get targets
        gt_heatmaps = target['keypoint_maps']
        gt_pafs = target['paf_maps']
        heatmap_mask = target['keypoint_mask']
        paf_mask = target['paf_mask']
        
        # Move to device
        gt_heatmaps = gt_heatmaps.to(self.device)
        gt_pafs = gt_pafs.to(self.device)
        heatmap_mask = heatmap_mask.to(self.device)
        paf_mask = paf_mask.to(self.device)
        
        batch_size = pred_heatmaps.shape[0]
        
        # Heatmap loss
        heatmap_loss = self._l2_loss(pred_heatmaps, gt_heatmaps, heatmap_mask, batch_size)
        
        # PAF loss
        paf_loss = self._l2_loss(pred_pafs, gt_pafs, paf_mask, batch_size)
        
        return heatmap_loss + paf_loss
    
    def _l2_loss(self, pred, target, mask, batch_size):
        """Masked L2 loss"""
        loss = (pred - target) * mask
        loss = (loss * loss) / 2 / batch_size
        return loss.sum()


class PoseDistillationLoss(nn.Module):
    """
    Knowledge distillation loss for pose estimation.
    Combines:
    - Student loss: L2 between student output and ground truth
    - Distillation loss: L2 between student and teacher outputs
    """
    def __init__(self, student_weight=1.0, distill_weight=1.0, device='cuda'):
        super().__init__()
        self.student_weight = student_weight
        self.distill_weight = distill_weight
        self.device = device
        self.pose_loss = PoseLoss(device=device)
    
    def forward(self, student_output, teacher_output, target):
        """
        Args:
            student_output: Student model output [heatmaps, pafs]
            teacher_output: Teacher model output [heatmaps, pafs]
            target: Ground truth dict
        
        Returns:
            Combined loss value
        """
        # Student loss with ground truth
        student_loss = self.pose_loss(student_output, target)
        
        # Distillation loss (L2 between student and teacher)
        if isinstance(student_output, (list, tuple)):
            student_heatmaps = student_output[0]
            student_pafs = student_output[1]
        else:
            student_heatmaps = student_output[:, :19]
            student_pafs = student_output[:, 19:]
        
        if isinstance(teacher_output, (list, tuple)):
            teacher_heatmaps = teacher_output[0]
            teacher_pafs = teacher_output[1]
        else:
            teacher_heatmaps = teacher_output[:, :19]
            teacher_pafs = teacher_output[:, 19:]
        
        # Resize teacher output to match student if needed
        if student_heatmaps.shape != teacher_heatmaps.shape:
            teacher_heatmaps = nn.functional.interpolate(
                teacher_heatmaps, 
                size=student_heatmaps.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            teacher_pafs = nn.functional.interpolate(
                teacher_pafs,
                size=student_pafs.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        batch_size = student_heatmaps.shape[0]
        
        # Distillation loss
        distill_heatmap = nn.functional.mse_loss(student_heatmaps, teacher_heatmaps.detach())
        distill_paf = nn.functional.mse_loss(student_pafs, teacher_pafs.detach())
        distill_loss = distill_heatmap + distill_paf
        
        total_loss = (self.student_weight * student_loss + 
                      self.distill_weight * distill_loss)
        
        return total_loss


class PoseRegressionLoss(nn.Module):
    """
    Simple MSE regression loss for pose estimation.
    Compatible with ai8x training framework's regression mode.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.mse = nn.MSELoss()
    
    def forward(self, output, target):
        """
        Args:
            output: Model output [heatmaps, pafs] or combined tensor
            target: Ground truth dict or tensor
        
        Returns:
            MSE loss value
        """
        # Convert output to single tensor if needed
        if isinstance(output, (list, tuple)):
            output = torch.cat(output, dim=1)
        
        # Convert target to tensor if dict
        if isinstance(target, dict):
            gt_heatmaps = target['keypoint_maps'].to(self.device)
            gt_pafs = target['paf_maps'].to(self.device)
            target_tensor = torch.cat([gt_heatmaps, gt_pafs], dim=1)
        else:
            target_tensor = target.to(self.device)
        
        return self.mse(output, target_tensor)

