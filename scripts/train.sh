#!/bin/bash
# Training script
# Usage: bash scripts/train.sh [xceptionnet|efficientnet|all]

set -e

echo "=========================================="
echo "Deepfake Detection - Model Training"
echo "=========================================="

MODEL=${1:-all}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run scripts/setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU available: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    DEVICE="cuda"
else
    echo "No GPU detected. Using CPU (training will be slow)"
    DEVICE="cpu"
fi

# Function to train XceptionNet
train_xceptionnet() {
    echo ""
    echo "[XceptionNet] Starting training..."
    echo "Config: configs/xception.yaml"
    echo "Device: $DEVICE"
    echo ""
    
    python3 -m src.training.train \
        --config configs/xception.yaml \
        --device $DEVICE \
        --experiment-name "xceptionnet_$(date +%Y%m%d_%H%M%S)"
    
    echo ""
    echo "[XceptionNet] Training complete!"
}

# Function to train EfficientNet
train_efficientnet() {
    echo ""
    echo "[EfficientNet] Starting training..."
    echo "Config: configs/efficientnet_b0.yaml"
    echo "Device: $DEVICE"
    echo ""
    
    python3 -m src.training.train \
        --config configs/efficientnet_b0.yaml \
        --device $DEVICE \
        --experiment-name "efficientnet_$(date +%Y%m%d_%H%M%S)"
    
    echo ""
    echo "[EfficientNet] Training complete!"
}

# Train based on argument
case $MODEL in
    xceptionnet|xception)
        train_xceptionnet
        ;;
    efficientnet|efficient)
        train_efficientnet
        ;;
    all)
        train_xceptionnet
        train_efficientnet
        ;;
    *)
        echo "Error: Unknown model '$MODEL'"
        echo "Usage: $0 [xceptionnet|efficientnet|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Next step: bash scripts/evaluate.sh"
echo ""
