#!/bin/bash
# Preprocessing script
# Usage: bash scripts/preprocess.sh [faceforensics|celebdf|all]

set -e

echo "=========================================="
echo "Deepfake Detection - Data Preprocessing"
echo "=========================================="

DATASET=${1:-all}
BASE_DIR="datasets"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run scripts/setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Function to preprocess FaceForensics++
preprocess_faceforensics() {
    echo ""
    echo "[FaceForensics++] Preprocessing..."
    echo "Step 1: Organizing raw data..."
    
    python3 -m src.data.organize \
        --input "$BASE_DIR/raw/faceforensics" \
        --output "$BASE_DIR/processed/faceforensics" \
        --quality c23 \
        --methods Deepfakes Face2Face FaceSwap NeuralTextures
    
    echo "Step 2: Creating train/val/test splits..."
    
    python3 -m src.data.split_data \
        --input "$BASE_DIR/processed/faceforensics" \
        --output "$BASE_DIR/splits/faceforensics" \
        --ratios 0.7 0.15 0.15 \
        --seed 42
}

# Function to preprocess Celeb-DF
preprocess_celebdf() {
    echo ""
    echo "[Celeb-DF] Preprocessing..."
    echo "Step 1: Organizing raw data..."
    
    python3 -m src.data.organize \
        --input "$BASE_DIR/raw/celebdf" \
        --output "$BASE_DIR/processed/celebdf" \
        --type celebdf
    
    echo "Step 2: Creating train/val/test splits..."
    
    python3 -m src.data.split_data \
        --input "$BASE_DIR/processed/celebdf" \
        --output "$BASE_DIR/splits/celebdf" \
        --ratios 0.7 0.15 0.15 \
        --seed 42
}

# Preprocess based on argument
case $DATASET in
    faceforensics|ff++)
        preprocess_faceforensics
        ;;
    celebdf)
        preprocess_celebdf
        ;;
    all)
        preprocess_faceforensics
        preprocess_celebdf
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET'"
        echo "Usage: $0 [faceforensics|celebdf|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Preprocessing complete!"
echo "=========================================="
echo ""
echo "Next step: bash scripts/train.sh"
echo ""
