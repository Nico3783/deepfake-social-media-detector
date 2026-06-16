# Deepfake Detection System

A deep learning system for detecting deepfake videos on social media platforms, developed as part of a B.Tech final year project in Cyber Security at the Federal University of Technology Akure (FUTA).

## Research Objectives

1. Develop a CNN-based detection model to identify manipulated video content
2. Implement frame-level classification (real vs. fake) with temporal aggregation for video-level detection
3. Evaluate the system using standard performance metrics (accuracy, precision, recall, F1-score, ROC-AUC)

## Methodology

- **Primary Model**: XceptionNet with depthwise separable convolutions (299×299 input)
- **Secondary Model**: EfficientNet with compound scaling (224×224 input)
- **Transfer Learning**: ImageNet pre-trained weights, fine-tuned on deepfake datasets
- **Datasets**: FaceForensics++ and Celeb-DF

## Project Structure

```
deepfake-social-media-detector/
├── src/                    # Source code
│   ├── config/             # Configuration management
│   ├── data/               # Dataset pipeline
│   ├── models/             # Model architectures
│   ├── preprocessing/      # Frame extraction and face detection
│   ├── training/           # Training pipeline
│   ├── evaluation/         # Evaluation metrics
│   ├── inference/          # Inference pipeline
│   ├── visualization/      # Visualization tools
│   ├── api/                # REST API (FastAPI)
│   └── utils/              # Utility functions
├── configs/                # YAML configuration files
├── data/                   # Datasets (not tracked)
├── outputs/                # Models, logs, results (not tracked)
├── tests/                  # Test suite
└── thesis/                 # Thesis document
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd deepfake-social-media-detector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit paths as needed
nano .env
```

### Dataset Preparation

```bash
# Download FaceForensics++ (requires manual download)
# Place in data/faceforensics++/

# Download Celeb-DF (requires manual download)
# Place in data/celeb-df/

# Organize and create splits
python -m src.data.organize
```

### Training

```bash
# Train XceptionNet
python -m src.training.train --config configs/training.yaml --model configs/xception.yaml

# Train EfficientNet
python -m src.training.train --config configs/training.yaml --model configs/efficientnet.yaml
```

### Inference

```bash
# Single video inference
python -m src.inference.predict --video path/to/video.mp4

# Batch inference
python -m src.inference.predict --input_dir path/to/videos/ --output_dir outputs/results/
```

## Configuration Files

| File | Description |
|------|-------------|
| `configs/dataset.yaml` | Dataset paths, splits, preprocessing |
| `configs/xception.yaml` | XceptionNet architecture settings |
| `configs/efficientnet.yaml` | EfficientNet architecture settings |
| `configs/training.yaml` | Training hyperparameters, augmentation |
| `configs/inference.yaml` | Inference settings, video processing |

## Research Reproducibility

All experiments are configured through YAML files with fixed random seeds. To reproduce results:

1. Set the same seed in `configs/training.yaml`
2. Use the same dataset splits
3. Train with the same hyperparameters
4. All metrics are logged to `outputs/logs/`

## Thesis Information

- **Title**: Detection of Social Media Deepfake Contents Using Deep Learning Algorithm
- **Student**: Olamijulo Israel D (CYS/22/9071)
- **Department**: Cyber Security
- **Institution**: Federal University of Technology Akure (FUTA)
- **Supervisor**: [Supervisor Name]

## License

This project is for academic research purposes only.
