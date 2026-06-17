# System Design

## Overview

The deepfake detection system follows a modular, layered architecture designed for clarity, maintainability, and extensibility.

## Architectural Principles

1. **Separation of Concerns:** Each module handles a single responsibility
2. **Modularity:** Components can be independently modified or replaced
3. **Configurability:** All parameters managed through YAML configuration files
4. **Reproducibility:** Fixed random seeds and version-controlled experiments
5. **Scalability:** Architecture supports future extension and cloud deployment

## Layer Architecture

### Layer 1: Data Layer
- **Components:** Dataset storage, metadata management, train/val/test splits
- **Responsibilities:** Organize raw data, maintain labels, provide split information
- **Output:** Structured datasets ready for preprocessing

### Layer 2: Preprocessing Layer
- **Components:** Frame Extractor, Face Detector, Face Cropper, Normalizer
- **Responsibilities:** Transform raw videos into model-ready face images
- **Output:** Normalized face tensors

### Layer 3: Feature Learning Layer
- **Components:** XceptionNet, EfficientNet-B0 (pre-trained + fine-tuned)
- **Responsibilities:** Extract hierarchical visual features from face images
- **Output:** Feature embeddings

### Layer 4: Classification Layer
- **Components:** Binary classification heads
- **Responsibilities:** Map features to real/fake predictions
- **Output:** Probability scores

### Layer 5: Aggregation Layer
- **Components:** Mean, Majority Vote, Confidence Weighted aggregators
- **Responsibilities:** Convert frame predictions to video-level decisions
- **Output:** Video-level classification

### Layer 6: Evaluation Layer
- **Components:** Metrics, Confusion Matrix, ROC-AUC, Report Generator
- **Responsibilities:** Compute performance metrics and generate reports
- **Output:** Evaluation reports (JSON, Markdown, LaTeX)

### Layer 7: API Layer
- **Components:** FastAPI service with REST endpoints
- **Responsibilities:** Expose inference capabilities via HTTP
- **Output:** JSON prediction responses

### Layer 8: Visualization Layer
- **Components:** Training curves, ROC curves, Confusion matrices, GradCAM
- **Responsibilities:** Generate publication-quality figures
- **Output:** PNG/PDF visualizations

## Data Flow

```
Video Input
    ↓
Frame Extraction (1 FPS)
    ↓
Face Detection (RetinaFace)
    ↓
Face Cropping (30% padding)
    ↓
Normalization (Resize + Scale)
    ↓
Model Inference (XceptionNet/EfficientNet)
    ↓
Frame Classification (Real/Fake probabilities)
    ↓
Video Aggregation (Mean/Majority/Confidence)
    ↓
Final Prediction (Real or Fake)
    ↓
Response (JSON / Visualization)
```

## Directory Structure

```
deepfake-social-media-detector/
├── src/                    # Source code
│   ├── config/            # Configuration management
│   ├── data/              # Data pipeline
│   ├── preprocessing/     # Frame extraction, face detection
│   ├── models/            # CNN architectures
│   ├── training/          # Training loop, losses, metrics
│   ├── evaluation/        # Evaluation pipeline
│   ├── inference/         # Video/image classification
│   ├── visualization/     # Plots, explainability
│   ├── api/               # FastAPI service
│   └── utils/             # Shared utilities
├── configs/               # YAML configuration files
├── datasets/              # Dataset storage and metadata
├── outputs/               # Experiment outputs
│   ├── models/           # Saved model checkpoints
│   ├── reports/          # Evaluation reports
│   └── plots/            # Generated visualizations
├── tests/                 # Test suite
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks
├── docs/                  # Documentation
└── thesis/                # Thesis documents
```
