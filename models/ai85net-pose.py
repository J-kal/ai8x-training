###################################################################################################
#
# Lightweight Pose Estimation Student Model for MAX78000/MAX78002
# 
# This is a compact pose estimation model designed for knowledge distillation
# from the larger PoseEstimationWithMobileNet teacher model.
#
###################################################################################################
"""
Lightweight Pose Estimation network for AI85/AI87.
Student model for knowledge distillation from MobileNet-based pose estimation.
"""
import torch
from torch import nn

import ai8x


class AI85PoseNetTiny(nn.Module):
    """
    Tiny Pose Estimation Model - Very compact for MAX78000
    Output: 19 keypoint heatmaps + 38 PAF channels = 57 channels total
    Input: 3x128x128 (downscaled from 368x368)
    """
    def __init__(
            self,
            num_classes=57,  # 19 heatmaps + 38 PAFs
            num_channels=3,
            dimensions=(128, 128),
            num_heatmaps=19,
            num_pafs=38,
            bias=False,
            **kwargs
    ):
        super().__init__()
        self.num_heatmaps = num_heatmaps
        self.num_pafs = num_pafs
        
        # Backbone - compact feature extractor
        self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, 16, 3, stride=2, padding=1, 
                                            bias=bias, **kwargs)
        self.conv2 = ai8x.FusedConv2dBNReLU(16, 32, 3, stride=1, padding=1, 
                                            bias=bias, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(32, 48, 3, pool_size=2, pool_stride=2,
                                                   stride=1, padding=1, bias=bias, **kwargs)
        self.conv4 = ai8x.FusedConv2dBNReLU(48, 48, 3, stride=1, padding=1, 
                                            bias=bias, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(48, 64, 3, pool_size=2, pool_stride=2,
                                                   stride=1, padding=1, bias=bias, **kwargs)
        self.conv6 = ai8x.FusedConv2dBNReLU(64, 64, 3, stride=1, padding=1, 
                                            bias=bias, **kwargs)
        
        # CPM-like module (Convolutional Pose Machine)
        self.cpm1 = ai8x.FusedConv2dBNReLU(64, 64, 1, stride=1, padding=0, 
                                           bias=bias, **kwargs)
        self.cpm2 = ai8x.FusedConv2dBNReLU(64, 64, 3, stride=1, padding=1, 
                                           bias=bias, **kwargs)
        
        # Heatmap head
        self.heatmap_conv1 = ai8x.FusedConv2dBNReLU(64, 48, 1, stride=1, padding=0, 
                                                    bias=bias, **kwargs)
        self.heatmap_out = ai8x.Conv2d(48, num_heatmaps, 1, stride=1, padding=0, 
                                       bias=bias, wide=True, **kwargs)
        
        # PAF head
        self.paf_conv1 = ai8x.FusedConv2dBNReLU(64, 64, 1, stride=1, padding=0, 
                                                bias=bias, **kwargs)
        self.paf_out = ai8x.Conv2d(64, num_pafs, 1, stride=1, padding=0, 
                                   bias=bias, wide=True, **kwargs)

    def forward(self, x):
        """Forward prop"""
        # Backbone
        x = self.conv1(x)  # 64x64
        x = self.conv2(x)
        x = self.conv3(x)  # 32x32
        x = self.conv4(x)
        x = self.conv5(x)  # 16x16
        x = self.conv6(x)
        
        # CPM
        x = self.cpm1(x)
        x = self.cpm2(x)
        
        # Heads
        heatmaps = self.heatmap_conv1(x)
        heatmaps = self.heatmap_out(heatmaps)
        
        pafs = self.paf_conv1(x)
        pafs = self.paf_out(pafs)
        
        # Return list like teacher model [heatmaps, pafs]
        return [heatmaps, pafs]


class AI85PoseNetSmall(nn.Module):
    """
    Small Pose Estimation Model - Better accuracy, still fits MAX78000
    Input: 3x128x128
    """
    def __init__(
            self,
            num_classes=57,
            num_channels=3,
            dimensions=(128, 128),
            num_heatmaps=19,
            num_pafs=38,
            bias=False,
            **kwargs
    ):
        super().__init__()
        self.num_heatmaps = num_heatmaps
        self.num_pafs = num_pafs
        
        # Backbone - more capacity
        self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, 32, 3, stride=2, padding=1, 
                                            bias=bias, **kwargs)
        self.conv2 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1, 
                                            bias=bias, **kwargs)
        self.conv3 = ai8x.FusedConv2dBNReLU(32, 64, 3, stride=1, padding=1, 
                                            bias=bias, **kwargs)
        self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, pool_size=2, pool_stride=2,
                                                   stride=1, padding=1, bias=bias, **kwargs)
        self.conv5 = ai8x.FusedConv2dBNReLU(64, 96, 3, stride=1, padding=1, 
                                            bias=bias, **kwargs)
        self.conv6 = ai8x.FusedMaxPoolConv2dBNReLU(96, 96, 3, pool_size=2, pool_stride=2,
                                                   stride=1, padding=1, bias=bias, **kwargs)
        self.conv7 = ai8x.FusedConv2dBNReLU(96, 128, 3, stride=1, padding=1, 
                                            bias=bias, **kwargs)
        self.conv8 = ai8x.FusedConv2dBNReLU(128, 128, 3, stride=1, padding=1, 
                                            bias=bias, **kwargs)
        
        # CPM-like module with more layers
        self.cpm1 = ai8x.FusedConv2dBNReLU(128, 96, 1, stride=1, padding=0, 
                                           bias=bias, **kwargs)
        self.cpm2 = ai8x.FusedConv2dBNReLU(96, 96, 3, stride=1, padding=1, 
                                           bias=bias, **kwargs)
        self.cpm3 = ai8x.FusedConv2dBNReLU(96, 96, 3, stride=1, padding=1, 
                                           bias=bias, **kwargs)
        
        # Heatmap head
        self.heatmap_conv1 = ai8x.FusedConv2dBNReLU(96, 64, 1, stride=1, padding=0, 
                                                    bias=bias, **kwargs)
        self.heatmap_out = ai8x.Conv2d(64, num_heatmaps, 1, stride=1, padding=0, 
                                       bias=bias, wide=True, **kwargs)
        
        # PAF head
        self.paf_conv1 = ai8x.FusedConv2dBNReLU(96, 64, 1, stride=1, padding=0, 
                                                bias=bias, **kwargs)
        self.paf_out = ai8x.Conv2d(64, num_pafs, 1, stride=1, padding=0, 
                                   bias=bias, wide=True, **kwargs)

    def forward(self, x):
        """Forward prop"""
        # Backbone
        x = self.conv1(x)  # 64x64
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # 32x32
        x = self.conv5(x)
        x = self.conv6(x)  # 16x16
        x = self.conv7(x)
        x = self.conv8(x)
        
        # CPM
        x = self.cpm1(x)
        x = self.cpm2(x)
        x = self.cpm3(x)
        
        # Heads
        heatmaps = self.heatmap_conv1(x)
        heatmaps = self.heatmap_out(heatmaps)
        
        pafs = self.paf_conv1(x)
        pafs = self.paf_out(pafs)
        
        return [heatmaps, pafs]


class AI85PoseNetTeacher(nn.Module):
    """
    Teacher model wrapper for the pre-trained PoseEstimationWithMobileNet.
    This wraps the external model to be compatible with ai8x training framework.
    """
    def __init__(
            self,
            num_classes=57,
            num_channels=3,
            dimensions=(128, 128),
            num_heatmaps=19,
            num_pafs=38,
            checkpoint_path=None,
            **kwargs
    ):
        super().__init__()
        self.num_heatmaps = num_heatmaps
        self.num_pafs = num_pafs
        
        # Import the teacher model architecture
        import sys
        sys.path.insert(0, '/home/jkal/Desktop/MLonMCU/lightweight-human-pose-estimation.pytorch')
        from models.with_mobilenet import PoseEstimationWithMobileNet
        from modules.load_state import load_state
        
        self.teacher = PoseEstimationWithMobileNet(num_refinement_stages=1, 
                                                   num_channels=128,
                                                   num_heatmaps=num_heatmaps, 
                                                   num_pafs=num_pafs)
        
        # Load pretrained weights if provided
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            load_state(self.teacher, checkpoint)
        
        # Freeze teacher weights
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Forward prop - adapts to student input size"""
        # Upsample if needed (student uses 128x128, teacher uses 368x368)
        if x.shape[2] != 368 or x.shape[3] != 368:
            x = nn.functional.interpolate(x, size=(368, 368), mode='bilinear', 
                                         align_corners=False)
        
        outputs = self.teacher(x)
        # Return only the last stage outputs [heatmaps, pafs]
        return outputs[-2:]  # Last heatmap and PAF outputs


def ai85posenet_tiny(pretrained=False, **kwargs):
    """Constructs a tiny PoseNet model."""
    assert not pretrained
    return AI85PoseNetTiny(**kwargs)


def ai85posenet_small(pretrained=False, **kwargs):
    """Constructs a small PoseNet model."""
    assert not pretrained
    return AI85PoseNetSmall(**kwargs)


def ai85posenet_teacher(pretrained=False, **kwargs):
    """Constructs the teacher PoseNet model wrapper."""
    assert not pretrained
    return AI85PoseNetTeacher(**kwargs)


models = [
    {
        'name': 'ai85posenet_tiny',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85posenet_small',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85posenet_teacher',
        'min_input': 1,
        'dim': 2,
    },
]

