#!/bin/bash
###################################################################################################
#
# Setup script for Pose Estimation Training Environment
#
# This script creates a virtual environment and installs all required dependencies.
#
# Usage: 
#   chmod +x setup_pose_training.sh
#   ./setup_pose_training.sh
#
###################################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv_pose"

echo "=============================================="
echo "Pose Estimation Training Environment Setup"
echo "=============================================="
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "ERROR: Python 3 not found. Please install Python 3.10+"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Create virtual environment
echo "Creating virtual environment at: ${VENV_DIR}"
$PYTHON_CMD -m venv "${VENV_DIR}"

# Activate virtual environment
echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for broad compatibility)
echo ""
echo "Installing PyTorch (CPU version)..."
echo "Note: This may take a few minutes..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo ""
echo "Installing other dependencies..."
pip install numpy opencv-python tqdm matplotlib pillow pycocotools tensorboard

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
import cv2
import numpy as np
from tqdm import tqdm

print('=' * 50)
print('Installation Verification')
print('=' * 50)
print(f'PyTorch version: {torch.__version__}')
print(f'OpenCV version: {cv2.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'CPU threads available: {torch.get_num_threads()}')
print('=' * 50)
print('All dependencies installed successfully!')
"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To start training, run:"
echo "  python train_pose_cpu.py --epochs 50 --subset 1000"
echo ""
echo "For full training (may take several hours):"
echo "  python train_pose_cpu.py --epochs 100"
echo ""








