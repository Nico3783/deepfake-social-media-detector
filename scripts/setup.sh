#!/bin/bash
# Setup script for Deepfake Detection System
# Usage: bash scripts/setup.sh

set -e  # Exit on error

echo "=========================================="
echo "Deepfake Detection System - Setup"
echo "=========================================="

# Check Python version
echo "[1/7] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "Python version: $PYTHON_VERSION"

# Check for GPU
echo "[2/7] Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    HAS_GPU=true
else
    echo "No NVIDIA GPU detected. Using CPU (training will be slow)"
    HAS_GPU=false
fi

# Create virtual environment
echo "[3/7] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "[4/7] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "[5/7] Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (with CUDA if GPU available)
echo "[6/7] Installing PyTorch..."
if [ "$HAS_GPU" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install dependencies
echo "[7/7] Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p outputs/{models,reports,plots,logs}
mkdir -p datasets/{raw,processed,splits}
mkdir -p logs

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python3 -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Download datasets: bash scripts/download_datasets.sh"
echo "  3. Preprocess data: bash scripts/preprocess.sh"
echo "  4. Train models: bash scripts/train.sh"
echo ""
