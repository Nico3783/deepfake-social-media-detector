#!/bin/bash
# Download datasets script
# Usage: bash scripts/download_datasets.sh [faceforensics|celebdf|all]

set -e

echo "=========================================="
echo "Deepfake Detection - Dataset Download"
echo "=========================================="

DATASET=${1:-all}
BASE_DIR="datasets/raw"

# Check if directory exists
mkdir -p "$BASE_DIR"

# Function to download FaceForensics++
download_faceforensics() {
    echo ""
    echo "[FaceForensics++] Downloading dataset..."
    echo "Note: You need to request access at https://github.com/ondyari/FaceForensics"
    echo ""
    echo "Manual download instructions:"
    echo "1. Visit https://github.com/ondyari/FaceForensics"
    echo "2. Follow the download instructions"
    echo "3. Place the data in: $BASE_DIR/faceforensics/"
    echo ""
    echo "Expected structure:"
    echo "  $BASE_DIR/faceforensics/"
    echo "  ├── manipulation/"
    echo "  │   ├── Deepfakes/"
    echo "  │   ├── Face2Face/"
    echo "  │   ├── FaceSwap/"
    echo "  │   └── NeuralTextures/"
    echo "  └── original/"
    echo ""
}

# Function to download Celeb-DF
download_celebdf() {
    echo ""
    echo "[Celeb-DF] Downloading dataset..."
    echo "Note: You need to request access at https://github.com/yunjey/celeb-deepfakeforensics"
    echo ""
    echo "Manual download instructions:"
    echo "1. Visit https://github.com/yunjey/celeb-deepfakeforensics"
    echo "2. Follow the download instructions"
    echo "3. Place the data in: $BASE_DIR/celebdf/"
    echo ""
    echo "Expected structure:"
    echo "  $BASE_DIR/celebdf/"
    echo "  ├── Celeb-real/"
    echo "  ├── Celeb-synthesis/"
    echo "  └── YouTube-real/"
    echo ""
}

# Download based on argument
case $DATASET in
    faceforensics|ff++)
        download_faceforensics
        ;;
    celebdf)
        download_celebdf
        ;;
    all)
        download_faceforensics
        download_celebdf
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET'"
        echo "Usage: $0 [faceforensics|celebdf|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Download instructions provided."
echo "=========================================="
echo ""
echo "After downloading, run:"
echo "  bash scripts/preprocess.sh"
echo ""
