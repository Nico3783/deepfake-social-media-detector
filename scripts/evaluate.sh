#!/bin/bash
# Evaluation script
# Usage: bash scripts/evaluate.sh [xceptionnet|efficientnet|all]

set -e

echo "=========================================="
echo "Deepfake Detection - Model Evaluation"
echo "=========================================="

MODEL=${1:-all}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run scripts/setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Function to evaluate XceptionNet
evaluate_xceptionnet() {
    echo ""
    echo "[XceptionNet] Evaluating..."
    
    # Check if model exists
    if [ ! -f "outputs/models/xceptionnet_best.pth" ]; then
        echo "Error: Model not found at outputs/models/xceptionnet_best.pth"
        echo "Run training first: bash scripts/train.sh xceptionnet"
        exit 1
    fi
    
    python3 -m src.evaluation.evaluate \
        --model outputs/models/xceptionnet_best.pth \
        --config configs/xception.yaml \
        --output-dir outputs/reports/xceptionnet \
        --visualize
    
    echo ""
    echo "[XceptionNet] Evaluation complete!"
    echo "Reports saved to: outputs/reports/xceptionnet/"
}

# Function to evaluate EfficientNet
evaluate_efficientnet() {
    echo ""
    echo "[EfficientNet] Evaluating..."
    
    # Check if model exists
    if [ ! -f "outputs/models/efficientnet_best.pth" ]; then
        echo "Error: Model not found at outputs/models/efficientnet_best.pth"
        echo "Run training first: bash scripts/train.sh efficientnet"
        exit 1
    fi
    
    python3 -m src.evaluation.evaluate \
        --model outputs/models/efficientnet_best.pth \
        --config configs/efficientnet_b0.yaml \
        --output-dir outputs/reports/efficientnet \
        --visualize
    
    echo ""
    echo "[EfficientNet] Evaluation complete!"
    echo "Reports saved to: outputs/reports/efficientnet/"
}

# Evaluate based on argument
case $MODEL in
    xceptionnet|xception)
        evaluate_xceptionnet
        ;;
    efficientnet|efficient)
        evaluate_efficientnet
        ;;
    all)
        evaluate_xceptionnet
        evaluate_efficientnet
        ;;
    *)
        echo "Error: Unknown model '$MODEL'"
        echo "Usage: $0 [xceptionnet|efficientnet|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
echo ""
echo "Next step: bash scripts/deploy.sh"
echo ""
