# Detection of Social Media Deepfake Contents Using Deep Learning Algorithm

A deep learning-based system for detecting deepfake videos distributed through social media platforms. Developed as a final-year Cyber Security research project at the Federal University of Technology Akure (FUTA).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Contributions](#research-contributions)
- [Problem Statement](#problem-statement)
- [Research Objectives](#research-objectives)
- [Methodology](#methodology)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
  - [Dataset Preparation](#dataset-preparation)
- [Running Experiments (Google Colab)](#running-experiments-google-colab)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [REST API](#rest-api)
- [Configuration Reference](#configuration-reference)
- [Model Architectures](#model-architectures)
- [Evaluation Metrics](#evaluation-metrics)
- [Testing](#testing)
- [Reproducibility](#reproducibility)
- [Technology Stack](#technology-stack)
- [Academic Information](#academic-information)
- [License](#license)

---

## Project Overview

Deepfakes — AI-generated synthetic media that convincingly manipulate facial content — pose serious threats to digital trust, cybersecurity, and socio-economic stability on social media platforms. This project implements a complete deep learning pipeline that:

1. **Extracts** video frames at fixed intervals
2. **Detects and crops** human faces using MTCNN/RetinaFace
3. **Classifies** each frame as real or fake using CNN models
4. **Aggregates** frame-level predictions into a video-level verdict
5. **Exposes** results through a REST API built with FastAPI

Two convolutional neural network architectures are implemented and compared:

| Model | Parameters | Input Size | Key Technique |
|-------|-----------|------------|---------------|
| **XceptionNet** (Primary) | ~22.9M | 299×299 | Depthwise separable convolutions |
| **EfficientNet-B0** (Secondary) | ~5.3M | 224×224 | Compound scaling (width, depth, resolution) |

Both models use **transfer learning** from ImageNet pre-trained weights, fine-tuned on deepfake datasets.

---

## Research Contributions

This project makes four concrete contributions to the deepfake detection literature:

### C1: Compression Impact Quantification
Quantify detection accuracy degradation across three JPEG compression levels (raw/c0, light/c23, heavy/c40) on FaceForensics++. Measures how social media re-encoding degrades forensic detection traces.

### C2: Cross-Dataset Generalization
Train on FaceForensics++ and evaluate on Celeb-DF (v2) to measure real-world generalization. Models trained on one dataset are tested on the other to quantify domain shift effects.

### C3: Model Architecture Comparison
Rigorous comparison of XceptionNet (~22.9M params, depthwise separable convolutions) vs EfficientNet-B0 (~5.3M params, compound scaling) on the same datasets with identical training protocols.

### C4: Deployment Readiness Assessment
Benchmark inference latency, GPU memory footprint, and model size to evaluate suitability for real-time social media content moderation at scale.

> **Note:** This project does not propose a new detection architecture. The contributions are **evaluation-driven** — providing empirical evidence on real-world challenges (compression, generalization, model selection, deployment trade-offs) that practitioners face when deploying deepfake detection systems.

See `thesis/contribution_statement.md` for the formal contribution statement.

---

## Problem Statement

Social media platforms have become primary channels for multimedia dissemination, but also for spreading manipulated content. Deepfakes on social media have been used for:

- **Impersonation scams** and financial fraud
- **Political misinformation** and public opinion manipulation
- **Reputational damage** against individuals
- **Erosion of digital trust** in authentic media

Traditional digital forensic methods and human observers are no longer sufficient against modern deepfakes, which are visually indistinguishable from real videos. Social media compression (resizing, re-encoding) further degrades forensic traces, making detection harder. There is a clear need for automated, deep learning-based detection systems that perform reliably under realistic social media conditions.

---

## Research Objectives

1. **Design** a deep learning-based detection model using convolutional neural network architectures (XceptionNet, EfficientNet)
2. **Implement** frame-level and video-level classification of deepfake content with temporal aggregation
3. **Evaluate** the system using standard performance metrics: accuracy, precision, recall, F1-score, and ROC-AUC
4. **Demonstrate** the effectiveness of transfer learning for deepfake detection with limited labeled data
5. **Produce** reproducible, well-documented results on standardized datasets (FaceForensics++, Celeb-DF)

---

## Methodology

### Preprocessing Pipeline

```
Video Input → Frame Extraction → Face Detection → Face Cropping → Normalization → Model-Ready Tensors
```

| Step | Tool/Method | Details |
|------|-------------|---------|
| Frame Extraction | OpenCV | 1 FPS sampling, up to 300 frames per video |
| Face Detection | MTCNN / RetinaFace | Confidence threshold 0.9, min face size 40px |
| Face Cropping | OpenCV | 20px margin padding around detected faces |
| Normalization | TorchVision | Resize to 299×299 (Xception) or 224×224 (EfficientNet), ImageNet mean/std |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001, weight_decay=0.0001) |
| Loss Function | Label Smoothing Cross-Entropy (smoothing=0.1) |
| Learning Rate Scheduler | ReduceLROnPlateau (factor=0.1, patience=5) |
| Batch Size | 32 |
| Max Epochs | 50 |
| Early Stopping | patience=10, min_delta=0.001 |
| Gradient Clipping | max_norm=1.0 |
| Mixed Precision | Enabled |
| Random Seed | 42 |

### Video-Level Aggregation

Frame-level predictions are aggregated into a single video prediction using one of three methods:

- **Mean Probability** — average of all frame prediction scores
- **Majority Voting** — most frequent frame prediction
- **Confidence Weighting** — weighted average by prediction confidence

---

## System Architecture

The system is organized into 8 layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO INPUT                              │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: DATA LAYER                                        │
│  Dataset storage, organization, metadata management         │
│  Datasets: FaceForensics++, Celeb-DF                        │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: PREPROCESSING LAYER                               │
│  Frame extraction → Face detection → Cropping → Normalization│
│  Tools: OpenCV, MTCNN, RetinaFace                           │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: FEATURE LEARNING LAYER                            │
│  CNN feature extraction via transfer learning                │
│  Models: XceptionNet (299×299), EfficientNet-B0 (224×224)   │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4: CLASSIFICATION LAYER                              │
│  Binary classification: Real (0) vs Fake (1)                │
│  Output: prediction scores per frame                        │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 5: AGGREGATION LAYER                                 │
│  Frame predictions → Video-level classification             │
│  Methods: mean, median, max, majority voting                │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 6: EVALUATION LAYER                                  │
│  Accuracy, Precision, Recall, F1-Score, ROC-AUC             │
│  Confusion matrices, classification reports                 │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 7: API LAYER (FastAPI)                               │
│  /predict-video  /predict-image  /health  /model-info       │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 8: VISUALIZATION LAYER                               │
│  Training curves, ROC curves, confusion matrices,           │
│  GradCAM explainability, publication-quality figures         │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
deepfake-social-media-detector/
│
├── src/                          # Source code (modular architecture)
│   ├── config/                   # Configuration management
│   │   ├── settings.py           # Central Settings class (Hydra-compatible)
│   │   ├── constants.py          # Project-wide constants
│   │   └── paths.py              # Path resolution utilities
│   │
│   ├── data/                     # Dataset pipeline
│   │   ├── dataset.py            # DeepfakeDataset (PyTorch Dataset)
│   │   ├── splitter.py           # DatasetSplitter (train/val/test splits)
│   │   ├── metadata.py           # MetadataManager (CSV/catalog handling)
│   │   ├── organize.py           # Dataset organization scripts
│   │   ├── split_data.py         # Data splitting utilities
│   │   └── download.py           # Dataset download helpers
│   │
│   ├── preprocessing/            # Frame and face preprocessing
│   │   ├── frame_extractor.py    # Video → frame extraction (OpenCV)
│   │   ├── face_detector.py      # Face detection (MTCNN / RetinaFace)
│   │   ├── face_cropper.py       # Face region cropping with padding
│   │   ├── image_resizer.py      # Resize to model input dimensions
│   │   └── normalizer.py         # ImageNet normalization
│   │
│   ├── models/                   # Model architectures
│   │   ├── xception.py           # XceptionNet (primary, ~22.9M params)
│   │   ├── efficientnet.py       # EfficientNet-B0 (secondary, ~5.3M params)
│   │   ├── classifier_head.py    # Shared classification head
│   │   ├── model_factory.py      # Model creation from config
│   │   ├── xception_model.py     # Xception wrapper
│   │   └── efficientnet_model.py # EfficientNet wrapper
│   │
│   ├── training/                 # Training pipeline
│   │   ├── train.py              # Main training entry point
│   │   ├── trainer.py            # Training loop orchestration
│   │   ├── losses.py             # FocalLoss, LabelSmoothingLoss
│   │   ├── metrics.py            # MetricsTracker (accuracy, F1, etc.)
│   │   ├── callbacks.py          # EarlyStopping, ModelCheckpoint
│   │   └── schedulers.py         # LR scheduler wrappers
│   │
│   ├── evaluation/               # Evaluation and reporting
│   │   ├── evaluate.py           # Evaluation pipeline
│   │   ├── confusion_matrix.py   # Confusion matrix generation
│   │   ├── roc_auc.py            # ROC-AUC computation and plotting
│   │   ├── report_generator.py   # Evaluation report generation
│   │   └── comparison.py         # Model comparison utilities
│   │
│   ├── inference/                # Inference pipeline
│   │   ├── video_classifier.py   # VideoClassifier (end-to-end video prediction)
│   │   ├── frame_analysis.py     # FrameAnalyzer (frame-level analysis)
│   │   ├── predict_image.py      # ImagePredictor (single image inference)
│   │   └── predict_video.py      # Video prediction utilities
│   │
│   ├── visualization/            # Visualization and explainability
│   │   ├── plots.py              # Training curves, ROC, confusion matrices
│   │   ├── dashboards.py         # Streamlit dashboard components
│   │   └── explainability.py     # GradCAM, feature visualization
│   │
│   ├── api/                      # REST API (FastAPI)
│   │   ├── app.py                # FastAPI application factory
│   │   ├── routes.py             # API endpoint definitions
│   │   ├── schemas.py            # Pydantic request/response models
│   │   ├── services.py           # Business logic layer
│   │   └── models.py             # API model loading utilities
│   │
│   └── utils/                    # Shared utilities
│       ├── logger.py             # Structured logging setup
│       ├── seed.py               # Random seed management
│       ├── file_manager.py       # File I/O utilities
│       └── helpers.py            # General helper functions
│
├── configs/                      # YAML configuration files
│   ├── dataset.yaml              # Dataset paths, splits, preprocessing
│   ├── xception.yaml             # XceptionNet architecture settings
│   ├── efficientnet.yaml         # EfficientNet architecture settings
│   ├── training.yaml             # Training hyperparameters, augmentation
│   └── inference.yaml            # Inference settings, video processing
│
├── scripts/                      # Experiment runner scripts
│   └── run_experiments.py        # Run all 4 thesis contributions
│
├── notebooks/                    # Jupyter/Colab notebooks
│   └── experiment_pipeline.ipynb # Full Colab notebook (train + evaluate)
│
├── tests/                        # Test suite
│   ├── conftest.py               # Shared fixtures
│   ├── test_dataset.py           # Data pipeline tests
│   ├── test_training.py          # Training pipeline tests
│   ├── test_inference.py         # Inference pipeline tests
│   ├── test_models.py            # Model architecture tests
│   ├── test_preprocessing.py     # Preprocessing tests
│   └── test_api.py               # API endpoint tests
│
├── data/                         # Datasets (not tracked in git)
│   ├── faceforensics++/          # FaceForensics++ dataset
│   └── celeb-df/                 # Celeb-DF dataset
│
├── outputs/                      # Generated outputs (not tracked in git)
│   ├── checkpoints/              # Model checkpoints (.pth)
│   ├── logs/                     # Training logs (TensorBoard)
│   ├── results/                  # Evaluation results and reports
│   ├── figures/                  # Generated plots and visualizations
│   └── inference/                # Inference output JSONs
│
├── thesis/                       # Thesis document
│   ├── chapters_1_2_3.md         # Chapters 1–3 (proposal, lit review, methodology)
│   └── contribution_statement.md # Formal contribution statement (4 points)
│
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── CLAUDE.md                     # AI assistant project rules
├── ARCHITECTURE.md               # System architecture documentation
├── REQUIREMENTS.md               # Functional/non-functional requirements
├── PROJECT_CONTEXT.md            # Project context and goals
└── IMPLEMENTATION_ROADMAP.md     # Implementation phases and status
```

---

## Getting Started

### Prerequisites

- **Python** 3.11 or higher
- **CUDA-capable GPU** (recommended) — training works on CPU but is significantly slower
- **Git** for version control
- ~10 GB disk space for datasets (FaceForensics++ ~11GB, Celeb-DF ~3GB)
- ~2 GB disk space for model checkpoints

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd deepfake-social-media-detector

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
# For CPU-only systems:
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

# For CUDA-enabled GPUs:
pip install -r requirements.txt
```

### Environment Setup

```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your local paths
nano .env
```

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_ROOT` | `/path/to/deepfake-social-media-detector` | Project root directory |
| `DATASETS_DIR` | `${PROJECT_ROOT}/datasets` | Dataset storage location |
| `OUTPUTS_DIR` | `${PROJECT_ROOT}/outputs` | Training outputs location |
| `DEFAULT_MODEL` | `xception` | Default model for inference |
| `DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `TENSORBOARD_ENABLED` | `true` | Enable TensorBoard logging |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |

### Dataset Preparation

This project uses two publicly available deepfake datasets. Both require **manual download** due to licensing terms.

#### FaceForensics++ (FF++)

1. Request access at [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
2. Download videos (recommended: `c23` quality for realistic social media conditions)
3. Organize into:
   ```
   data/faceforensics++/
   ├── real/                   # Authentic videos
   │   └── c23/
   ├── manipulated/            # Deepfake videos
   │   ├── Deepfakes/c23/
   │   ├── Face2Face/c23/
   │   ├── FaceSwap/c23/
   │   └── NeuralTextures/c23/
   └── metadata.csv            # Video metadata
   ```

#### Celeb-DF (v2)

1. Request access at [https://github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics)
2. Download real and synthesized videos
3. Organize into:
   ```
   data/celeb-df/
   ├── Celeb-real/             # Real celebrity interview videos
   ├── Celeb-synthesis/        # Deepfake celebrity videos
   └── evaluation.csv          # Evaluation metadata
   ```

#### Automatic Splitting

Once datasets are placed, create train/val/test splits:

```bash
python -m src.data.organize
```

Default split ratios (configured in `configs/dataset.yaml`):
- Train: 70%
- Validation: 15%
- Test: 15%

---

## Running Experiments (Google Colab)

The recommended way to run experiments is via the Colab notebook, which handles GPU setup, dataset mounting from Google Drive, and full training + evaluation.

### Quick Start

1. **Upload datasets to Google Drive** under `deepfake_datasets/`:
   ```
   MyDrive/deepfake_datasets/
   ├── faceforensic++/FF++/real/
   ├── faceforensic++/FF++/fake/
   ├── celeb-df/Celeb-real/
   ├── celeb-df/Celeb-synthesis/
   ├── celeb-df/YouTube-real/
   └── celeb-df/list_of_testing_videos.txt
   ```

2. **Open in Colab**: File → Open notebook → GitHub → `nico3783/deepfake-social-media-detector` → `notebooks/experiment_pipeline.ipynb`

3. **Enable GPU**: Runtime → Change runtime type → T4 GPU

4. **Run all cells**: Runtime → Run all

The notebook automatically:
- Clones the repository from GitHub
- Mounts Google Drive for dataset access
- Sets up Google Drive API for uploading results to shared folder
- Extracts faces from videos (FF++ and Celeb-DF)
- Creates video-level train/val/test splits (70/15/15)
- Trains XceptionNet and EfficientNet-B0
- Runs all 4 experiments (compression, cross-dataset, model comparison, deployment)
- Uploads results to your shared Google Drive folder (`deepfake-project-results/`)

**Estimated time:** 5-10 hours on T4 GPU. Checkpoints save to shared folder, so you can resume if Colab disconnects.

---

## Usage

### Training

Train models using the CLI entry points:

```bash
# Train XceptionNet (primary model)
python -m src.training.train \
    --config configs/training.yaml \
    --model configs/xception.yaml

# Train EfficientNet-B0 (secondary model)
python -m src.training.train \
    --config configs/training.yaml \
    --model configs/efficientnet.yaml
```

#### What Happens During Training

1. **Data Loading** — `DeepfakeDataset` loads video frames and labels from the organized splits
2. **Augmentation** — Random horizontal flips, rotation (±15°), brightness/contrast jittering, JPEG quality variation
3. **Transfer Learning** — Pre-trained ImageNet weights are loaded; base layers are frozen, then unfrozen from layer -10
4. **Training Loop** — Forward pass → loss computation (Label Smoothing Cross-Entropy) → backpropagation → gradient clipping (max_norm=1.0)
5. **Validation** — After each epoch, validation loss and metrics are computed
6. **Early Stopping** — Training stops if validation loss doesn't improve by 0.001 for 10 consecutive epochs
7. **Checkpointing** — Best 3 models saved to `outputs/checkpoints/` (by `val_loss`)
8. **Logging** — Training metrics logged to `outputs/logs/` for TensorBoard visualization

#### Monitoring Training

```bash
# Launch TensorBoard
tensorboard --logdir outputs/logs
```

Metrics tracked per epoch:
- Training loss, validation loss
- Training accuracy, validation accuracy
- Learning rate
- Precision, recall, F1-score

### Inference

#### Single Video

```bash
python -m src.inference.predict_video \
    --video path/to/video.mp4 \
    --checkpoint outputs/checkpoints/best_model.pth \
    --model xception \
    --threshold 0.5
```

#### Single Image

```bash
python -m src.inference.predict_image \
    --image path/to/image.jpg \
    --checkpoint outputs/checkpoints/best_model.pth \
    --model xception
```

#### Batch Inference

```bash
python -m src.inference.predict_video \
    --input_dir path/to/videos/ \
    --output_dir outputs/inference/ \
    --checkpoint outputs/checkpoints/best_model.pth \
    --model xception \
    --format json
```

#### Inference Pipeline

```
Video → Frame Extraction (1 FPS) → Face Detection (MTCNN)
    → Face Cropping → Normalization → Model Prediction (per frame)
    → Aggregation (mean/voting/weighting) → Video Verdict
```

Output format (JSON):

```json
{
  "video_path": "test_video.mp4",
  "prediction": "fake",
  "confidence": 0.947,
  "label": 1,
  "frame_results": [
    {"frame_idx": 0, "timestamp": 0.0, "prediction": "fake", "confidence": 0.92},
    {"frame_idx": 1, "timestamp": 1.0, "prediction": "fake", "confidence": 0.96},
    {"frame_idx": 2, "timestamp": 2.0, "prediction": "real", "confidence": 0.71}
  ],
  "aggregation_method": "mean",
  "num_frames_processed": 3
}
```

### REST API

The system exposes a FastAPI-based REST API for real-time inference.

#### Start the Server

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

#### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict-video` | Upload a video for deepfake detection |
| `POST` | `/predict-image` | Upload an image for deepfake detection |
| `GET` | `/health` | Health check |
| `GET` | `/model-info` | Current model metadata |

#### Example: Predict Video

```bash
curl -X POST "http://localhost:8000/predict-video" \
    -F "file=@test_video.mp4"
```

Response:

```json
{
  "prediction": "fake",
  "confidence": 0.947,
  "label": 1,
  "video_path": "test_video.mp4",
  "frame_results": [...],
  "aggregation_method": "mean"
}
```

#### Example: Health Check

```bash
curl "http://localhost:8000/health"
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "model_name": "xception"
}
```

---

## Configuration Reference

All settings are managed through YAML files in `configs/`. The system follows a priority hierarchy:

```
Environment Variables > CLI Arguments > YAML Config > Defaults
```

### `configs/dataset.yaml`

| Setting | Value | Description |
|---------|-------|-------------|
| `split_ratios.train` | 0.7 | Training set proportion |
| `split_ratios.val` | 0.15 | Validation set proportion |
| `split_ratios.test` | 0.15 | Test set proportion |
| `preprocessing.image_size` | 299 | Default image resize dimension |
| `face_detection.method` | `mtcnn` | Face detection algorithm |
| `face_detection.confidence_threshold` | 0.9 | Minimum face detection confidence |
| `loading.num_workers` | 4 | DataLoader worker processes |

### `configs/xception.yaml`

| Setting | Value | Description |
|---------|-------|-------------|
| `architecture.input_size` | 299 | Input image dimension |
| `architecture.num_classes` | 2 | Binary classification |
| `architecture.dropout_rate` | 0.5 | Dropout before classification head |
| `transfer_learning.strategy` | `feature_extraction` | Freeze base, train head |
| `transfer_learning.unfreeze_from` | -10 | Unfreeze last 10 layers |

### `configs/efficientnet.yaml`

| Setting | Value | Description |
|---------|-------|-------------|
| `architecture.variant` | `B0` | EfficientNet variant (B0–B7) |
| `architecture.num_classes` | 2 | Binary classification |
| `architecture.dropout_rate` | 0.2 | Dropout before classification head |
| `scaling.resolution` | 224 | Input image dimension |

### `configs/training.yaml`

| Setting | Value | Description |
|---------|-------|-------------|
| `general.seed` | 42 | Random seed for reproducibility |
| `general.max_epochs` | 50 | Maximum training epochs |
| `general.patience` | 10 | Early stopping patience |
| `optimizer.name` | `adam` | Optimizer algorithm |
| `optimizer.learning_rate` | 0.001 | Initial learning rate |
| `optimizer.weight_decay` | 0.0001 | L2 regularization |
| `scheduler.name` | `ReduceLROnPlateau` | LR scheduler type |
| `scheduler.factor` | 0.1 | LR reduction factor |
| `loss.name` | `CrossEntropyLoss` | Loss function |
| `loss.label_smoothing` | 0.1 | Label smoothing factor |
| `batch_size` | 32 | Training batch size |
| `checkpointing.save_top_k` | 3 | Number of best models to keep |

### `configs/inference.yaml`

| Setting | Value | Description |
|---------|-------|-------------|
| `video.frame_sampling_rate` | 1 | Extract every Nth frame |
| `video.max_frames` | 300 | Maximum frames per video |
| `video_level.aggregation_method` | `mean` | Frame → video aggregation |
| `face_detection.method` | `mtcnn` | Face detection for inference |
| `preprocessing.image_size` | 299 | Must match model input size |

---

## Model Architectures

### XceptionNet (Primary)

```
Input (299×299×3)
    ↓
Entry Flow (3 blocks) — initial feature extraction
    ↓
Middle Flow (8 blocks) — feature refinement with residual connections
    ↓
Exit Flow (2 blocks) — classification
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Linear (2048 → 2)
    ↓
Output: [real_score, fake_score]
```

- **Total parameters:** ~22.9M
- **Key technique:** Depthwise separable convolutions separate spatial and channel feature learning
- **Transfer learning:** ImageNet pre-trained, last 10 layers unfrozen for fine-tuning

### EfficientNet-B0 (Secondary)

```
Input (224×224×3)
    ↓
MBConv1 blocks — inverted residual with SE attention
    ↓
MBConv6 blocks — expanded inverted residual with SE attention
    ↓
Global Average Pooling
    ↓
Dropout (0.2)
    ↓
Linear (1280 → 2)
    ↓
Output: [real_score, fake_score]
```

- **Total parameters:** ~5.3M
- **Key technique:** Compound scaling balances network width, depth, and resolution
- **Transfer learning:** ImageNet pre-trained, last 10 layers unfrozen for fine-tuning

### Model Comparison

| Property | XceptionNet | EfficientNet-B0 |
|----------|-------------|-----------------|
| Parameters | ~22.9M | ~5.3M |
| Input size | 299×299 | 224×224 |
| GFLOPs | ~8.4 | ~0.39 |
| Depth | 71 layers | 62 layers |
| Key innovation | Depthwise separable convolutions | Compound scaling + SE attention |
| Classification head | GAP → Dropout(0.5) → FC(2048, 2) | GAP → Dropout(0.2) → FC(1280, 2) |

---

## Evaluation Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness |
| **Precision** | TP / (TP + FP) | How many predicted fakes are actually fake |
| **Recall** | TP / (TP + FN) | How many actual fakes are caught |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve | Discrimination ability across all thresholds |

**Target performance goals:**
- Accuracy ≥ 85%
- F1-Score ≥ 0.85
- ROC-AUC ≥ 0.90

*These are research targets, not guaranteed outcomes.*

---

## Testing

Run the test suite to verify module correctness:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_dataset.py -v      # Data pipeline
python -m pytest tests/test_training.py -v     # Training pipeline
python -m pytest tests/test_inference.py -v    # Inference pipeline
python -m pytest tests/test_models.py -v       # Model architectures
python -m pytest tests/test_preprocessing.py -v # Preprocessing
python -m pytest tests/test_api.py -v          # API endpoints

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing
```

Test coverage areas:

| Module | Tests |
|--------|-------|
| `test_dataset.py` | Dataset loading, splitting, metadata management |
| `test_training.py` | Loss functions, metrics, callbacks, schedulers |
| `test_inference.py` | Video classifier, frame analyzer, image predictor |
| `test_models.py` | Model instantiation, forward pass, parameter counts |
| `test_preprocessing.py` | Frame extraction, face detection, normalization |
| `test_api.py` | API endpoints, request/response validation |

---

## Reproducibility

This project is designed for full reproducibility:

1. **Fixed random seeds** — All randomness (PyTorch, NumPy, Python random) is seeded with `42`
2. **Deterministic training** — `torch.backends.cudnn.deterministic = True`
3. **Configuration-driven** — All hyperparameters stored in YAML configs, not hardcoded
4. **Versioned dependencies** — `requirements.txt` with minimum version pins
5. **Dataset splits** — Fixed 70/15/15 splits with seed-controlled shuffling
6. **Checkpoint tracking** — Best models saved with epoch number and metrics

To reproduce any experiment:

```bash
# 1. Use the same config files
# 2. Ensure same dataset splits (seed=42 in configs/training.yaml)
# 3. Train with identical hyperparameters
# 4. All metrics are logged to outputs/logs/
```

---

## Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.11+ | Primary language |
| **Deep Learning** | PyTorch 2.1+ | Model training and inference |
| **Computer Vision** | OpenCV 4.8+ | Video processing, frame extraction |
| **Face Detection** | MTCNN, RetinaFace | Facial region detection |
| **Pre-trained Models** | TorchVision | ImageNet weights for transfer learning |
| **API** | FastAPI | REST API for inference |
| **Data Science** | NumPy, Pandas, scikit-learn | Data manipulation and evaluation |
| **Visualization** | Matplotlib, Seaborn | Training curves, ROC, confusion matrices |
| **Explainability** | GradCAM | Model interpretability |
| **Experiment Tracking** | TensorBoard | Metric visualization |
| **Dashboard** | Streamlit | Interactive demo interface |
| **Configuration** | YAML, python-dotenv | Settings management |
| **Code Quality** | Ruff, mypy | Linting and type checking |
| **Testing** | pytest | Unit and integration tests |

---

## Academic Information

| Field | Detail |
|-------|--------|
| **Title** | Detection of Social Media Deepfake Contents Using Deep Learning Algorithm |
| **Author** | Olamijulo Israel D |
| **Matric Number** | CYS/22/9071 |
| **Department** | Cyber Security, School of Computing |
| **Institution** | Federal University of Technology Akure (FUTA) |
| **Degree** | Bachelor of Technology (B.Tech) in Cyber Security |
| **Supervisor** | [Supervisor Name] |
| **Date** | February 2026 |

### Datasets

| Dataset | Source | Purpose |
|---------|--------|---------|
| **FaceForensics++** | Technical University of Munich (TUM) | Primary training and evaluation |
| **Celeb-DF (v2)** | University at Albany, Chinese Academy of Sciences | Cross-dataset validation |

### Key References

- Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," ICCV 2019
- Li et al., "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics," CVPR 2020
- Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," CVPR 2017
- Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs," ICML 2019

---

## License

This project is for **academic research purposes only**. Datasets are subject to their original licenses and usage terms.

---

## Contributing

This is a research project. For questions or contributions:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

All code must follow the project's coding standards (see `CLAUDE.md`):
- Python 3.11+ with type hints
- Modular, testable components
- Comprehensive docstrings
- Fixed random seeds for reproducibility
