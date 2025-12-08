#!/usr/bin/env python3
"""
Convenience launcher for the ~4 MB student (backbone/CPM/head widened) with Drive paths.

- Teacher: /content/drive/MyDrive/MLonMCU/Models/MAX78000/IP_8MB/best.pth
- Labels:  /content/drive/MyDrive/MLonMCU/Datasets/annotations/prepared_train_annotation.pkl
- Images:  /content/drive/MyDrive/MLonMCU/Datasets/train2014
- Output:  /content/drive/MyDrive/MLonMCU/Models/MAX78000/IP_4MB

Default device GPU; FP32 by default (add --amp to enable mixed precision).
"""

import sys

from train_pose_flexible_4mb import main as flexible_main_4mb


def build_args():
    drive_model_dir = "/content/drive/MyDrive/MLonMCU/Models/MAX78000/IP_4MB"
    drive_best = f"{drive_model_dir}/best.pth"
    return [
        "train_pose_flexible_4mb.py",
        "--device", "gpu",
        "--teacher", "/content/drive/MyDrive/MLonMCU/Models/MAX78000/IP_8MB/best.pth",
        "--labels", "/content/drive/MyDrive/MLonMCU/Datasets/annotations/prepared_train_annotation.pkl",
        "--images", "/content/drive/MyDrive/MLonMCU/Datasets/train2014",
        "--output", drive_model_dir,
        "--checkpoint", drive_best,
        # T4-friendly defaults; adjust as needed
        "--subset", "120000",
        "--batch-size", "64",
        "--epochs", "16",
        "--save-every", "500",
        "--lr", "0.001",
        "--num-workers", "4",
    ]


def main():
    # Allow user CLI args to override defaults
    sys.argv = build_args()[:1] + build_args()[1:] + sys.argv[1:]
    flexible_main_4mb()


if __name__ == "__main__":
    main()
