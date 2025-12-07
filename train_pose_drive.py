#!/usr/bin/env python3
"""
Convenience launcher to train pose distillation with Google Drive paths.

Uses the flexible trainer with the provided teacher, labels, and images:
- Teacher checkpoint: /content/drive/MyDrive/MLonMCU/Models/MAX78000/starting/checkpoint_iter_370000.pth
- Labels pickle:     /content/drive/MyDrive/MLonMCU/Datasets/annotations/prepared_train_annotation.pkl
- Images folder:     /content/drive/MyDrive/MLonMCU/Dataset/train2014

Default device is GPU; will fall back to CPU if CUDA is unavailable.
"""

import sys
from pathlib import Path

# Import the flexible trainer's main
from train_pose_flexible import main as flexible_main


def build_args():
    return [
        "train_pose_flexible.py",
        "--device", "gpu",
        "--teacher", "/content/drive/MyDrive/MLonMCU/Models/MAX78000/starting/checkpoint_iter_370000.pth",
        "--labels", "/content/drive/MyDrive/MLonMCU/Datasets/annotations/prepared_train_annotation.pkl",
        "--images", "/content/drive/MyDrive/MLonMCU/Dataset/train2014",
        "--output", "pose_drive_checkpoints",
        # T4-friendly defaults (adjust as needed)
        "--subset", "120000",       # use most of the dataset
        "--batch-size", "32",       # T4 16GB should handle 128x128 inputs
        "--save-every", "500",      # fewer checkpoints for longer runs
        "--total-batches", "30000", # extended training
        "--lr", "0.001",
        "--num-workers", "4",       # balanced for Colab T4
    ]


def main():
    # If the user provides extra CLI args, append them to override defaults
    sys.argv = build_args()[:1] + build_args()[1:] + sys.argv[1:]
    flexible_main()


if __name__ == "__main__":
    main()

