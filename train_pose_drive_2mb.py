#!/usr/bin/env python3
"""
Convenience launcher to train the 2 MB student with Drive paths.

Uses the 2 MB flexible trainer with the provided teacher, labels, and images:
- Teacher checkpoint: /content/drive/MyDrive/MLonMCU/Models/MAX78000/IP_8MB/best.pth
- Labels pickle:     /content/drive/MyDrive/MLonMCU/Datasets/annotations/prepared_train_annotation.pkl
- Images folder:     /content/drive/MyDrive/MLonMCU/Datasets/train2014

Checkpoints are saved to and resumed from:
- /content/drive/MyDrive/MLonMCU/Models/MAX78000/IP_2MB (default best.pth)

Default device is GPU; will fall back to CPU if CUDA is unavailable.
The 2 MB script runs FP32 by default (AMP off); add --amp to enable on CUDA.
"""

import sys

# Import the 2 MB flexible trainer's main
from train_pose_flexible_2mb import main as flexible_main_2mb


def build_args():
    drive_model_dir = "/content/drive/MyDrive/MLonMCU/Models/MAX78000/IP_2MB"
    drive_best = f"{drive_model_dir}/best.pth"
    return [
        "train_pose_flexible_2mb.py",
        "--device", "gpu",
        "--teacher", "/content/drive/MyDrive/MLonMCU/Models/MAX78000/IP_8MB/best.pth",
        "--labels", "/content/drive/MyDrive/MLonMCU/Datasets/annotations/prepared_train_annotation.pkl",
        "--images", "/content/drive/MyDrive/MLonMCU/Datasets/train2014",
        "--output", drive_model_dir,
        "--checkpoint", drive_best,
        # T4-friendly defaults (adjust as needed)
        "--subset", "120000",       # use most of the dataset
        "--batch-size", "64",       # fits better on T4 16GB; adjust if OOM
        "--epochs", "16",           # ~30k steps with 120k subset @ 64 bs
        "--save-every", "500",      # fewer checkpoints for longer runs
        "--lr", "0.001",
        "--num-workers", "4",       # balanced for Colab T4
        # FP32 by default; add --amp manually if desired
    ]


def main():
    # Allow user CLI args to override defaults
    sys.argv = build_args()[:1] + build_args()[1:] + sys.argv[1:]
    flexible_main_2mb()


if __name__ == "__main__":
    main()
