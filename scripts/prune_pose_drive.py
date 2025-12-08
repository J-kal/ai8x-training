#!/usr/bin/env python3
"""
Prune the 2 MB student pose model directly from Drive checkpoints.

Defaults:
- Source checkpoint: /content/drive/MyDrive/MLonMCU/Models/MAX78000/IP_2MB/best.pth
- Output dir:       /content/drive/MyDrive/MLonMCU/Models/MAX78000/Pruned
- Target sparsity:  0.5 (50%)
"""

import argparse
import sys
from pathlib import Path
import torch

# Ensure repo root is importable when the script is invoked by absolute path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_pose_flexible_2mb import StudentModel2MB  # type: ignore
from modules.load_state import load_state  # type: ignore
from scripts.prune_with_ai8x import prune_model


DEFAULT_SOURCE = Path("/content/drive/MyDrive/MLonMCU/Models/MAX78000/IP_2MB/best.pth")
DEFAULT_OUT_DIR = Path("/content/drive/MyDrive/MLonMCU/Models/MAX78000/Pruned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune the Drive pose model with ai8x/Distiller.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Path to the source checkpoint to prune")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Directory to write pruned checkpoints")
    parser.add_argument("--target-sparsity", type=float, default=0.5,
                        help="Fraction of weights to prune (0-1)")
    parser.add_argument("--method", type=str, default="agp-structured",
                        choices=("agp-structured", "agp-unstructured", "l1-structured"),
                        help="Pruning strategy")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run pruning on (cpu or cuda)")
    parser.add_argument("--thin", action="store_true",
                        help="Physically remove pruned filters (requires structured method)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Load model
    model = StudentModel2MB().to(device)
    ckpt = torch.load(args.source, map_location=device)
    # Normalize checkpoint format: support bare state_dict, {"state_dict": ...}, {"model": ...}
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt or "model" in ckpt:
            normalized = ckpt
        else:
            normalized = {"state_dict": ckpt}
    else:
        normalized = {"state_dict": ckpt}
    load_state(model, normalized)

    # Prune
    model, masks, sparsity = prune_model(
        model,
        target_sparsity=args.target_sparsity,
        method=args.method,
        arch="simplenet_cifar",   # only used if thin=True
        dataset="cifar10",        # only used if thin=True
        thin=args.thin,
    )

    # Save
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"best_pruned_{int(args.target_sparsity * 100)}.pth"
    out_path = args.out_dir / out_name
    payload = {
        "state_dict": model.state_dict(),
        "masks": {k: v.mask for k, v in masks.items() if v.mask is not None},
        "target_sparsity": args.target_sparsity,
        "method": args.method,
        "source": str(args.source),
    }
    torch.save(payload, out_path)
    print(f"Pruned model saved to: {out_path}")
    print(f"Final sparsity: {sparsity * 100:.2f}%")


if __name__ == "__main__":
    main()

