#!/usr/bin/env python3
"""
Utility helpers to prune a PyTorch model with ai8x / Distiller pruners.

Typical usage (structured, 50% sparsity):
    python scripts/prune_with_ai8x.py \
        --model-factory my_models.pose:build_model \
        --checkpoint runs/pose.pt \
        --method agp-structured \
        --target-sparsity 0.5 \
        --save-to runs/pose_pruned.pt

You can also import :func:`prune_model` directly and pass in an already-
constructed torch.nn.Module instance.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

import distiller  # type: ignore[import]
from distiller import pruning as pruning_mod  # type: ignore[attr-defined]


PruneResult = Tuple[torch.nn.Module, Dict[str, Any], float]


def _load_model_from_factory(factory_path: str, device: torch.device) -> torch.nn.Module:
    """
    Dynamically import a factory function and build the model.

    The expected format is ``module.submodule:create_fn`` where ``create_fn`` returns
    an initialized ``torch.nn.Module``.
    """
    if ":" not in factory_path:
        raise ValueError("model-factory must look like 'package.module:create_fn'")

    module_path, factory_name = factory_path.split(":", 1)
    module = importlib.import_module(module_path)
    factory_fn = getattr(module, factory_name)
    model = factory_fn()
    return model.to(device)


def infer_prunable_weight_names(model: torch.nn.Module, structured: bool) -> List[str]:
    """Pick sensible default weight tensors to prune."""
    modules = dict(model.named_modules())
    weight_names: List[str] = []
    for name, param in model.named_parameters():
        if not name.endswith("weight"):
            continue
        module_name = name.rsplit(".", 1)[0]
        module = modules.get(module_name)
        if module is None:
            continue
        if structured:
            # Structured pruning only works on convolutional kernels (3D/4D).
            if isinstance(module, torch.nn.Conv2d):
                weight_names.append(name)
        else:
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                weight_names.append(name)
    if not weight_names:
        raise ValueError("No prunable weights found. Pass --weights explicitly.")
    return weight_names


def _apply_pruner(
    model: torch.nn.Module,
    pruner,
    zeros_mask_dict: Dict[str, Any],
    weight_names: Iterable[str],
    meta: Optional[dict],
) -> None:
    params = dict(model.named_parameters())
    for name in weight_names:
        pruner.set_param_mask(params[name], name, zeros_mask_dict, meta)
    for name in weight_names:
        zeros_mask_dict[name].apply_mask(params[name])


def _model_sparsity(model: torch.nn.Module) -> float:
    total: float = 0.0
    zero: float = 0.0
    with torch.no_grad():
        for param in model.parameters():
            total += param.numel()
            zero += torch.count_nonzero(param == 0).item()
    return zero / float(total) if total else 0.0


def prune_model(
    model: torch.nn.Module,
    target_sparsity: float = 0.5,
    method: str = "agp-structured",
    *,
    agp_steps: int = 10,
    weight_names: Optional[List[str]] = None,
    arch: Optional[str] = None,
    dataset: Optional[str] = None,
    thin: bool = False,
) -> PruneResult:
    """
    Prune ``model`` in-place and return the pruned model, mask dict, and sparsity.

    Args:
        model: The torch model to prune.
        target_sparsity: Desired fraction of weights to zero (0.0-1.0).
        method: One of ``agp-structured``, ``agp-unstructured``, ``l1-structured``.
        agp_steps: Virtual epoch steps to reach ``target_sparsity`` for AGP methods.
        weight_names: Optional explicit list of parameter names to prune.
        arch/dataset: Optional identifiers used by ``distiller.remove_filters``.
        thin: When ``True`` and using a structured method, attempt filter removal.
    """
    structured = method in ("agp-structured", "l1-structured")
    weight_names = weight_names or infer_prunable_weight_names(model, structured=structured)
    zeros_mask_dict = distiller.create_model_masks_dict(model)  # type: ignore[attr-defined]

    if method == "agp-structured":
        pruner = pruning_mod.L1RankedStructureParameterPruner_AGP(
            name="agp_structured",
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            group_type="Filters",
            weights=weight_names,
        )
        meta = {
            "current_epoch": agp_steps,
            "starting_epoch": 0,
            "ending_epoch": agp_steps,
            "frequency": 1,
            "model": model,
        }
    elif method == "agp-unstructured":
        pruner = distiller.AutomatedGradualPruner(  # type: ignore[attr-defined]
            name="agp_unstructured",
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            weights=weight_names,
        )
        meta = {
            "current_epoch": agp_steps,
            "starting_epoch": 0,
            "ending_epoch": agp_steps,
            "frequency": 1,
            "model": model,
        }
    elif method == "l1-structured":
        pruner = pruning_mod.L1RankedStructureParameterPruner(
            name="l1_structured",
            group_type="Filters",
            desired_sparsity=target_sparsity,
            weights=weight_names,
        )
        meta = {"model": model}
    else:
        raise ValueError(f"Unsupported pruning method: {method}")

    _apply_pruner(model, pruner, zeros_mask_dict, weight_names, meta)

    if thin and structured and arch and dataset:
        # Physically remove pruned filters to shrink channel dimensions.
        distiller.remove_filters(  # type: ignore[attr-defined]
            model, zeros_mask_dict, arch=arch, dataset=dataset, optimizer=None
        )

    return model, zeros_mask_dict, _model_sparsity(model)


def _maybe_load_checkpoint(model: torch.nn.Module, checkpoint: Optional[Path], device: torch.device) -> None:
    if checkpoint is None:
        return
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    state = state or ckpt
    model.load_state_dict(state)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune a model with ai8x (Distiller) pruners.")
    parser.add_argument("--model-factory", type=str, required=True,
                        help="Import path to a factory function, e.g., models.pose:build_model")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Optional checkpoint containing a state_dict to load before pruning")
    parser.add_argument("--method", type=str, default="agp-structured",
                        choices=("agp-structured", "agp-unstructured", "l1-structured"),
                        help="Pruning strategy to apply")
    parser.add_argument("--target-sparsity", type=float, default=0.5,
                        help="Fraction of weights to zero (0.0-1.0)")
    parser.add_argument("--agp-steps", type=int, default=10,
                        help="Virtual steps used to reach target sparsity for AGP")
    parser.add_argument("--weights", type=str, nargs="*", default=None,
                        help="Explicit parameter names to prune (overrides auto-discovery)")
    parser.add_argument("--arch", type=str, default=None,
                        help="Optional model architecture id for filter thinning")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Optional dataset id for filter thinning")
    parser.add_argument("--thin", action="store_true",
                        help="Physically remove filters after structured pruning")
    parser.add_argument("--save-to", type=Path, default=None,
                        help="Path to save the pruned state_dict and metadata")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to place the model on before pruning")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    model = _load_model_from_factory(args.model_factory, device)
    _maybe_load_checkpoint(model, args.checkpoint, device)

    model, masks, sparsity = prune_model(
        model,
        target_sparsity=args.target_sparsity,
        method=args.method,
        agp_steps=args.agp_steps,
        weight_names=args.weights,
        arch=args.arch,
        dataset=args.dataset,
        thin=args.thin,
    )

    print(f"Final model sparsity: {sparsity * 100:.2f}%")
    if args.save_to:
        payload = {
            "state_dict": model.state_dict(),
            "masks": {k: v.mask for k, v in masks.items() if v.mask is not None},
            "target_sparsity": args.target_sparsity,
            "method": args.method,
        }
        torch.save(payload, args.save_to)
        print(f"Saved pruned checkpoint to {args.save_to}")


if __name__ == "__main__":
    main()
