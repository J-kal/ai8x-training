#!/usr/bin/env bash
###################################################################################################
#
# Knowledge Distillation Training Script for Pose Estimation
#
# This script trains a compact pose estimation model using knowledge distillation
# from the pre-trained MobileNet-based teacher model.
#
# Usage: ./scripts/train_pose_kd.sh
#
###################################################################################################

# Configuration
TEACHER_CHECKPOINT="/home/jkal/Desktop/MLonMCU/ext/ai8x-training/models/checkpoint_iter_370000.pth"
STUDENT_MODEL="ai85posenet_small"  # or ai85posenet_tiny for even smaller model
TEACHER_MODEL="ai85posenet_teacher"
DATASET="COCO_POSE"
COCO_DATA="/home/jkal/Desktop/MLonMCU/lightweight-human-pose-estimation.pytorch/coco"

# Training parameters
EPOCHS=150
BATCH_SIZE=32
LEARNING_RATE=0.001
OPTIMIZER="Adam"

# Knowledge Distillation parameters
KD_TEMP=4.0
KD_DISTILL_WT=0.7   # Weight for distillation loss
KD_STUDENT_WT=0.3   # Weight for student loss
KD_START_EPOCH=0

# Output
EXPERIMENT_NAME="pose_kd_${STUDENT_MODEL}"

echo "=== Pose Estimation Knowledge Distillation Training ==="
echo "Teacher checkpoint: ${TEACHER_CHECKPOINT}"
echo "Student model: ${STUDENT_MODEL}"
echo "Dataset: ${DATASET}"
echo ""

# Check if teacher checkpoint exists
if [ ! -f "${TEACHER_CHECKPOINT}" ]; then
    echo "ERROR: Teacher checkpoint not found at ${TEACHER_CHECKPOINT}"
    echo "Please ensure the checkpoint file exists."
    exit 1
fi

# Run training with knowledge distillation
python train.py \
    --arch ${STUDENT_MODEL} \
    --dataset ${DATASET} \
    --data "${COCO_DATA}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --optimizer ${OPTIMIZER} \
    --lr ${LEARNING_RATE} \
    --compress policies/schedule-pose-kd.yaml \
    --qat-policy policies/qat_policy_pose.yaml \
    --kd-teacher ${TEACHER_MODEL} \
    --kd-resume "${TEACHER_CHECKPOINT}" \
    --kd-temp ${KD_TEMP} \
    --kd-distill-wt ${KD_DISTILL_WT} \
    --kd-student-wt ${KD_STUDENT_WT} \
    --kd-start-epoch ${KD_START_EPOCH} \
    --name ${EXPERIMENT_NAME} \
    --device MAX78000 \
    --regression \
    --use-bias \
    --enable-tensorboard \
    "$@"

echo ""
echo "=== Training Complete ==="
echo "Checkpoints saved to: logs/${EXPERIMENT_NAME}*"

