# Pose Estimation Training with Knowledge Distillation and QAT

This guide explains how to train a compact pose estimation model for MAX78000/MAX78002 using:
1. **Knowledge Distillation** - Transfer knowledge from the large MobileNet-based teacher to a small student
2. **Quantization-Aware Training (QAT)** - Prepare the model for 8-bit inference
3. **Post-Training Pruning** - Further reduce model size

## Overview

The original `PoseEstimationWithMobileNet` model from the lightweight-human-pose-estimation repository is too large for edge deployment. We use knowledge distillation to train a much smaller model that can run on MAX78000/MAX78002.

### Model Comparison

| Model | Parameters | Size | Target |
|-------|------------|------|--------|
| Teacher (MobileNet) | ~4.5M | ~18MB | Desktop/GPU |
| Student (Small) | ~200K | ~800KB | MAX78000 |
| Student (Tiny) | ~80K | ~320KB | MAX78000 (constrained) |

## Files Created

```
ai8x-training/
├── models/
│   └── ai85net-pose.py              # Student and teacher model definitions
├── datasets/
│   └── pose_coco.py                 # COCO pose dataset loader
├── losses/
│   └── poseloss.py                  # Pose estimation loss functions
├── policies/
│   ├── schedule-pose.yaml           # Basic training schedule
│   ├── schedule-pose-kd.yaml        # KD training schedule
│   ├── qat_policy_pose.yaml         # QAT policy
│   └── schedule-pose-pruning.yaml   # Pruning schedule
├── scripts/
│   ├── train_pose_kd.sh             # KD training script
│   ├── train_pose_qat.sh            # QAT training script
│   └── prune_pose.sh                # Pruning script
└── train_pose_distillation.py       # Standalone training script
```

## Prerequisites

1. **COCO Dataset**: Download and prepare COCO keypoints dataset
   ```bash
   cd /home/jkal/Desktop/MLonMCU/lightweight-human-pose-estimation.pytorch
   
   # Prepare training labels (if not already done)
   python scripts/prepare_train_labels.py \
       --labels coco/annotations_trainval2017/annotations/person_keypoints_train2017.json \
       --output-label coco/prepared_train_annotations.pkl
   ```

2. **Teacher Checkpoint**: The pre-trained checkpoint is at:
   ```
   /home/jkal/Desktop/MLonMCU/ext/ai8x-training/models/checkpoint_iter_370000.pth
   ```

## Training Workflow

### Option 1: Using Shell Scripts (Recommended)

```bash
cd /home/jkal/Desktop/MLonMCU/ext/ai8x-training

# Step 1: Knowledge Distillation Training
./scripts/train_pose_kd.sh

# Step 2: QAT Fine-tuning (after KD training completes)
./scripts/train_pose_qat.sh logs/pose_kd_ai85posenet_small/best.pth.tar

# Step 3: Pruning (after QAT training completes)
./scripts/prune_pose.sh logs/pose_qat_ai85posenet_small_qat/best.pth.tar
```

### Option 2: Using Standalone Python Script

```bash
cd /home/jkal/Desktop/MLonMCU/ext/ai8x-training

# Step 1: Knowledge Distillation
python train_pose_distillation.py --mode kd --model small --epochs 100

# Step 2: QAT (resume from KD checkpoint)
python train_pose_distillation.py --mode qat --model small \
    --resume pose_checkpoints/best.pth --epochs 50

# Step 3: Pruning (resume from QAT checkpoint)
python train_pose_distillation.py --mode prune --model small \
    --resume pose_checkpoints/qat_best.pth --epochs 40
```

### Option 3: Using ai8x train.py Directly

```bash
cd /home/jkal/Desktop/MLonMCU/ext/ai8x-training

# Knowledge Distillation Training
python train.py \
    --arch ai85posenet_small \
    --dataset COCO_POSE \
    --data /home/jkal/Desktop/MLonMCU/lightweight-human-pose-estimation.pytorch/coco \
    --epochs 150 \
    --batch-size 32 \
    --optimizer Adam \
    --lr 0.001 \
    --compress policies/schedule-pose-kd.yaml \
    --qat-policy policies/qat_policy_pose.yaml \
    --kd-teacher ai85posenet_teacher \
    --kd-resume models/checkpoint_iter_370000.pth \
    --kd-temp 4.0 \
    --kd-distill-wt 0.7 \
    --kd-student-wt 0.3 \
    --name pose_kd \
    --device MAX78000 \
    --regression \
    --use-bias
```

## Model Export

After training, export the model for deployment:

```bash
# Export to ONNX
python train.py \
    --arch ai85posenet_small \
    --dataset COCO_POSE \
    --exp-load-weights-from logs/pose_pruned/best.pth.tar \
    --summary onnx_simplified \
    --device MAX78000 \
    -8

# Generate YAML for ai8x-synthesize
python train.py \
    --arch ai85posenet_small \
    --dataset COCO_POSE \
    --exp-load-weights-from logs/pose_pruned/best.pth.tar \
    --yaml-template networks/pose-ai85.yaml \
    --device MAX78000 \
    -8
```

## Training Tips

### Knowledge Distillation
- Start with `distill_weight > student_weight` (e.g., 0.7 vs 0.3)
- Temperature of 4.0 works well for pose estimation
- Train for at least 100 epochs

### QAT
- Use a lower learning rate (0.1x of KD training)
- QAT should start after the model has converged
- Fine-tune for 30-50 epochs

### Pruning
- Use very low learning rate (0.01x of initial)
- Apply gradual pruning to maintain accuracy
- Target 30-50% sparsity for MAX78000

## Hyperparameters

| Parameter | KD Training | QAT | Pruning |
|-----------|-------------|-----|---------|
| Learning Rate | 0.001 | 0.0001 | 0.00005 |
| Epochs | 100-150 | 30-50 | 30-40 |
| Batch Size | 32 | 32 | 32 |
| Optimizer | Adam | Adam | Adam |

## Output Channels

The model outputs 57 channels:
- **19 Keypoint Heatmaps**: Probability maps for each body keypoint
- **38 Part Affinity Fields (PAFs)**: 19 body parts × 2 directions (x, y)

## Troubleshooting

### Dataset Not Found
Ensure the COCO dataset is properly structured:
```
coco/
├── train2017/           # Training images
├── val2017/             # Validation images
└── prepared_train_annotations.pkl  # Prepared labels
```

### Out of Memory
Reduce batch size or use the `ai85posenet_tiny` model.

### Poor Accuracy
- Increase training epochs
- Adjust distillation weight balance
- Try different learning rates

## References

- [Lightweight Human Pose Estimation](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)
- [ai8x-training](https://github.com/MaximIntegratedAI/ai8x-training)
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531)

