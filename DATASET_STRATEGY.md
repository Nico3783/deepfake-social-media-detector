# DATASET_STRATEGY.md

## Overview

This document outlines the dataset acquisition, preparation, and management strategy for the deepfake detection project.

## Approved Datasets

### FaceForensics++ (Primary)

- **Source:** Technical University of Munich (TUM)
- **URL:** https://github.com/ondyari/FaceForensics
- **License:** Research use only
- **Content:** 1,000 real + 4,000 manipulated videos
- **Manipulation Methods:**
  - Deepfakes (autoencoder-based face swap)
  - Face2Face (facial reenactment)
  - FaceSwap (3D face model fitting)
  - NeuralTextures (neural rendering)
- **Compression Levels:** c0 (raw), c23 (high quality), c40 (low quality)
- **Target Quality:** c23 (realistic social media conditions)

### Celeb-DF v2 (Secondary/Validation)

- **Source:** University at Albany + Chinese Academy of Sciences
- **URL:** https://github.com/yuezunli/celeb-deepfakeforensics
- **License:** Research use only
- **Content:** 590 real + 5,639 synthetic videos
- **Characteristics:**
  - Celebrity YouTube interviews
  - Improved synthesis pipeline
  - Reduced color mismatch and temporal artifacts
  - More challenging for detection models

## Data Acquisition Steps

### FaceForensics++

1. Request access via official GitHub repository
2. Download c23 compression level videos
3. Extract real videos from `original_sequences/youtube/c23`
4. Extract manipulated videos from `manipulated_sequences/{method}/c23`
5. Verify video integrity and file sizes

### Celeb-DF v2

1. Request access via official repository
2. Download real videos from YouTube interviews
3. Download synthetic videos from the provided links
4. Map video IDs to labels

## Data Organization

```
datasets/
├── raw/
│   ├── faceforensics/
│   │   ├── real/
│   │   │   ├── c23/
│   │   │   └── c40/
│   │   ├── deepfakes/
│   │   ├── face2face/
│   │   ├── faceswap/
│   │   └── neuraltextures/
│   └── celebdf/
│       ├── real/
│       └── fake/
├── processed/
│   ├── faces/
│   │   ├── train/
│   │   │   ├── real/
│   │   │   └── fake/
│   │   ├── val/
│   │   │   ├── real/
│   │   │   └── fake/
│   │   └── test/
│   │       ├── real/
│   │       └── fake/
│   └── metadata/
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── metadata/
    ├── labels.csv
    ├── train.csv
    ├── val.csv
    └── test.csv
```

## Dataset Splits

| Split | Percentage | Real Videos | Fake Videos |
|-------|------------|-------------|-------------|
| Training | 70% | 720 | 3,200 |
| Validation | 10% | 80 | 400 |
| Testing | 20% | 200 | 800 |

## Preprocessing Requirements

1. **Frame Extraction:** 1 FPS sampling rate
2. **Face Detection:** RetinaFace or MTCNN
3. **Face Cropping:** 30% padding around detected face
4. **Resize:** 299x299 (XceptionNet), 224x224 (EfficientNet)
5. **Normalization:** ImageNet mean/std or [0, 1] scaling

## Reproducibility

- Random seed: 42
- Deterministic splits saved to CSV files
- Metadata includes video IDs, labels, split assignments
- All preprocessing steps logged and version-controlled
