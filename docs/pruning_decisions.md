## Pruning helper overview

- Script: `scripts/prune_with_ai8x.py`
- Goal: let you pass in any torch model and reach a configurable sparsity target (default 50%) using ai8x/Distiller pruners.
- Supports multiple techniques: structured filter pruning (AGP or one-shot L1) and unstructured AGP.

## What gets pruned

- Default target tensors: convolution weights for structured modes; convolution + linear weights for unstructured.
- BatchNorm and bias tensors are left untouched to avoid destabilizing activations.
- If your model stores weights under different names, pass them explicitly with `--weights layer1.conv1.weight ...`.

## Pruning strategies and rationale

- `agp-structured` (default): L1-ranked filter pruning with an Automated Gradual Pruning schedule. Removes low-magnitude filters over virtual steps to preserve accuracy while driving toward the target sparsity.
- `l1-structured`: One-shot L1-ranked filter pruning to immediately hit the target sparsity when you want a fast pass without scheduling.
- `agp-unstructured`: Magnitude-based AGP on individual weights. Useful when channel-level removal is unsafe but you still want density reduction.
- Optional `--thin`: After structured pruning and when `--arch` and `--dataset` are provided, `distiller.remove_filters` is invoked to physically remove pruned filters so channel dimensions shrink.

## CLI examples

- Drive 2 MB student (default paths), 50% structured pruning:
  - `python scripts/prune_pose_drive.py --target-sparsity 0.5`
- Structured, 50% sparsity, keep topology (masking only):
  - `python scripts/prune_with_ai8x.py --model-factory my_models.pose:build_model --checkpoint runs/pose.pt --method agp-structured --target-sparsity 0.5 --save-to runs/pose_pruned.pt`
- Unstructured, 30% sparsity:
  - `python scripts/prune_with_ai8x.py --model-factory my_models.pose:build_model --checkpoint runs/pose.pt --method agp-unstructured --target-sparsity 0.3`
- Structured with thinning (requires a distiller-known `--arch` and `--dataset`):
  - `python scripts/prune_with_ai8x.py --model-factory my_models.pose:build_model --checkpoint runs/pose.pt --method l1-structured --target-sparsity 0.5 --arch simplenet_cifar --dataset cifar10 --thin --save-to runs/pose_pruned_thin.pt`

## When to adjust parameters

- Increase `--agp-steps` if you plan to integrate the pruning into a fine-tuning loop and want a gentler schedule.
- Lower `--target-sparsity` for accuracy-sensitive layers or pass a custom `--weights` list to exclude fragile parts of the network.
- Switch to `agp-unstructured` when structured pruning fails due to channel dependencies in custom blocks.

