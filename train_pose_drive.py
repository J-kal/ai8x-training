#!/usr/bin/env python3
"""
Convenience launcher to train pose distillation with Google Drive paths.

Uses the flexible trainer with the provided teacher, labels, and images:
- Teacher checkpoint: /content/drive/MyDrive/MLonMCU/Models/MAX78000/starting/checkpoint_iter_370000.pth
- Labels pickle:     /content/drive/MyDrive/MLonMCU/Datasets/annotations/prepared_train_annotation.pkl
- Images folder:     /content/drive/MyDrive/MLonMCU/Dataset/train2014

Checkpoints are saved to and resumed from:
- /content/drive/MyDrive/MLonMCU/Models/MAX78000/IP_8MB (default best.pth)

Default device is GPU; will fall back to CPU if CUDA is unavailable.
Mixed precision (AMP) is enabled by default on CUDA to fit larger batches.
"""

import sys

# Import the flexible trainer's main
from train_pose_flexible import main as flexible_main


def build_args():
    drive_model_dir = "/content/drive/MyDrive/MLonMCU/Models/MAX78000/IP_8MB"
    drive_best = f"{drive_model_dir}/best.pth"
    return [
        "train_pose_flexible.py",
        "--device", "gpu",
        "--teacher", "/content/drive/MyDrive/MLonMCU/Models/MAX78000/starting/checkpoint_iter_370000.pth",
        "--labels", "/content/drive/MyDrive/MLonMCU/Datasets/annotations/prepared_train_annotation.pkl",
        "--images", "/content/drive/MyDrive/MLonMCU/Datasets/train2014",
        "--output", drive_model_dir,
        "--checkpoint", drive_best,
        # T4-friendly defaults (adjust as needed)
        "--subset", "120000",       # use most of the dataset
        "--batch-size", "64",       # AMP-enabled, fits better on T4 16GB
        "--epochs", "16",           # ~30k steps with 120k subset @ 64 bs
        "--save-every", "500",      # fewer checkpoints for longer runs
        "--lr", "0.001",
        "--num-workers", "4",       # balanced for Colab T4
    ]


def main():
    # If the user provides extra CLI args, append them to override defaults
    sys.argv = build_args()[:1] + build_args()[1:] + sys.argv[1:]
    flexible_main()


if __name__ == "__main__":
    main()


