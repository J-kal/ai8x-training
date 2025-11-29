#!/usr/bin/env bash
###################################################################################################
#
# Post-Training Pruning Script for Pose Estimation
#
# This script applies structured pruning to the QAT-trained model to further reduce size.
# Run this AFTER QAT training is complete.
#
# Usage: ./scripts/prune_pose.sh <checkpoint_path>
#
###################################################################################################

# Check for checkpoint argument
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_qat_checkpoint>"
    echo "Example: $0 logs/pose_qat_ai85posenet_small_qat/best.pth.tar"
    exit 1
fi

CHECKPOINT_PATH="$1"
STUDENT_MODEL="ai85posenet_small"
DATASET="COCO_POSE"
COCO_DATA="/home/jkal/Desktop/MLonMCU/lightweight-human-pose-estimation.pytorch/coco"

# Pruning parameters
EPOCHS=40
BATCH_SIZE=32
LEARNING_RATE=0.00005  # Very low LR for pruning fine-tuning
OPTIMIZER="Adam"

# Output
EXPERIMENT_NAME="pose_pruned_${STUDENT_MODEL}"

echo "=== Pose Estimation Post-Training Pruning ==="
echo "Starting checkpoint: ${CHECKPOINT_PATH}"
echo "Student model: ${STUDENT_MODEL}"
echo "Target sparsity: 50%"
echo ""

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT_PATH}"
    exit 1
fi

# Run pruning
python train.py \
    --arch ${STUDENT_MODEL} \
    --dataset ${DATASET} \
    --data "${COCO_DATA}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --optimizer ${OPTIMIZER} \
    --lr ${LEARNING_RATE} \
    --compress policies/schedule-pose-pruning.yaml \
    --qat-policy none \
    --exp-load-weights-from "${CHECKPOINT_PATH}" \
    --name ${EXPERIMENT_NAME} \
    --device MAX78000 \
    --regression \
    --use-bias \
    --8-bit-mode \
    --enable-tensorboard \
    "$@"

echo ""
echo "=== Pruning Complete ==="
echo "Checkpoints saved to: logs/${EXPERIMENT_NAME}*"
echo ""
echo "Final pruned model ready for deployment!"
echo ""
echo "To export the model, run:"
echo "  python train.py --arch ${STUDENT_MODEL} --dataset ${DATASET} \\"
echo "    --exp-load-weights-from logs/${EXPERIMENT_NAME}/best.pth.tar \\"
echo "    --summary onnx_simplified --device MAX78000 -8"

