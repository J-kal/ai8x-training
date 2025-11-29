#!/usr/bin/env bash
###################################################################################################
#
# Quantization-Aware Training Script for Pose Estimation
#
# This script fine-tunes the pre-trained student model with QAT for MAX78000 deployment.
# Run this AFTER knowledge distillation training is complete.
#
# Usage: ./scripts/train_pose_qat.sh <checkpoint_path>
#
###################################################################################################

# Check for checkpoint argument
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_kd_checkpoint>"
    echo "Example: $0 logs/pose_kd_ai85posenet_small/best.pth.tar"
    exit 1
fi

CHECKPOINT_PATH="$1"
STUDENT_MODEL="ai85posenet_small"
DATASET="COCO_POSE"
COCO_DATA="/home/jkal/Desktop/MLonMCU/lightweight-human-pose-estimation.pytorch/coco"

# QAT Training parameters
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.0001  # Lower LR for QAT fine-tuning
OPTIMIZER="Adam"

# Output
EXPERIMENT_NAME="pose_qat_${STUDENT_MODEL}"

echo "=== Pose Estimation Quantization-Aware Training ==="
echo "Starting checkpoint: ${CHECKPOINT_PATH}"
echo "Student model: ${STUDENT_MODEL}"
echo ""

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT_PATH}"
    exit 1
fi

# Run QAT training
python train.py \
    --arch ${STUDENT_MODEL} \
    --dataset ${DATASET} \
    --data "${COCO_DATA}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --optimizer ${OPTIMIZER} \
    --lr ${LEARNING_RATE} \
    --compress policies/schedule-pose.yaml \
    --qat-policy policies/qat_policy.yaml \
    --exp-load-weights-from "${CHECKPOINT_PATH}" \
    --name ${EXPERIMENT_NAME} \
    --device MAX78000 \
    --regression \
    --use-bias \
    --enable-tensorboard \
    "$@"

echo ""
echo "=== QAT Training Complete ==="
echo "Checkpoints saved to: logs/${EXPERIMENT_NAME}*"
echo ""
echo "Next step: Run pruning with:"
echo "  ./scripts/prune_pose.sh logs/${EXPERIMENT_NAME}_qat/best.pth.tar"

