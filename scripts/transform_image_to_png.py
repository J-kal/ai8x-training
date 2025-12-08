#!/usr/bin/env python3
"""
Apply the same resize+pad transform used in SimpleDataset and save as JPG.

Usage:
    python scripts/transform_image_to_png.py --image /path/to/img.jpg --size 128 --output out.jpg
"""

import argparse
import os

import cv2
import numpy as np


def resize_and_pad(img, size):
    """Resize so max(h,w)==size, then pad to a square of (size,size)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = size - new_h, size - new_w
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return padded


def main():
    parser = argparse.ArgumentParser(description="Resize+pad an image and save as JPG.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--size", type=int, default=128, help="Target square size (default: 128)")
    parser.add_argument("--output", type=str, default=None, help="Output JPG path (default: alongside input with suffix)")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    transformed = resize_and_pad(img, args.size)

    if args.output is None:
        base, _ = os.path.splitext(os.path.basename(args.image))
        out_dir = os.path.dirname(args.image)
        args.output = os.path.join(out_dir, f"{base}_transformed_{args.size}.jpg")

    # Write JPEG with default quality; can tweak via cv2.IMWRITE_JPEG_QUALITY if needed
    cv2.imwrite(args.output, transformed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Saved transformed image to: {args.output}")


if __name__ == "__main__":
    main()
