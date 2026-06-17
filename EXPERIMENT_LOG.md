# EXPERIMENT_LOG.md

## Overview

This document tracks all experiments conducted during the project, including configurations, results, and observations.

## Experiment Format

Each experiment record includes:
- Experiment ID
- Date
- Model architecture
- Dataset and split
- Hyperparameters
- Training duration
- Evaluation metrics
- Observations

---

## Experiment 001: XceptionNet Baseline on FaceForensics++ c23

**Date:** 2026-02-01  
**Status:** Completed

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | XceptionNet |
| Dataset | FaceForensics++ c23 |
| Train Samples | 180,000 frames |
| Val Samples | 20,000 frames |
| Test Samples | 50,000 frames |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Batch Size | 32 |
| Max Epochs | 50 |
| Early Stopping | Patience=10 |
| Label Smoothing | 0.1 |
| Scheduler | ReduceLROnPlateau |
| Seed | 42 |

### Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.9467 |
| Precision | 0.9523 |
| Recall | 0.9389 |
| F1-Score | 0.9456 |
| ROC-AUC | 0.9812 |
| Training Time | 2h 15m |
| Convergence Epoch | 35 |

### Observations
- Stable training with monotonic loss decrease
- Early stopping triggered at epoch 45
- No significant overfitting observed
- Strong performance on all metrics

---

## Experiment 002: EfficientNet-B0 on FaceForensics++ c23

**Date:** 2026-02-03  
**Status:** Completed

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | EfficientNet-B0 |
| Dataset | FaceForensics++ c23 |
| Train Samples | 180,000 frames |
| Val Samples | 20,000 frames |
| Test Samples | 50,000 frames |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Batch Size | 32 |
| Max Epochs | 50 |
| Early Stopping | Patience=10 |
| Label Smoothing | 0.1 |
| Scheduler | ReduceLROnPlateau |
| Seed | 42 |

### Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.9234 |
| Precision | 0.9312 |
| Recall | 0.9156 |
| F1-Score | 0.9234 |
| ROC-AUC | 0.9645 |
| Training Time | 1h 45m |
| Convergence Epoch | 40 |

### Observations
- Slower convergence than XceptionNet
- 4.3x fewer parameters, 1.3x faster training
- Good generalization with small train-val gap
- Competitive performance for resource-constrained scenarios

---

## Experiment 003: Cross-Dataset Validation (Celeb-DF)

**Date:** 2026-02-05  
**Status:** Completed

### Configuration

| Parameter | Value |
|-----------|-------|
| Models | XceptionNet, EfficientNet-B0 |
| Training Dataset | FaceForensics++ c23 |
| Evaluation Dataset | Celeb-DF v2 |
| Test Samples | 5,900 videos |

### Results

| Metric | XceptionNet | EfficientNet-B0 |
|--------|-------------|-----------------|
| Accuracy | 0.8734 | 0.8456 |
| Precision | 0.8812 | 0.8534 |
| Recall | 0.8623 | 0.8312 |
| F1-Score | 0.8717 | 0.8422 |
| ROC-AUC | 0.9345 | 0.9123 |

### Observations
- Performance drop expected due to different manipulation techniques
- XceptionNet generalizes better (7.3% vs 7.8% accuracy drop)
- Both models meet minimum performance thresholds
- Highlights cross-dataset generalization challenge

---

## Experiment 004: Video Aggregation Methods

**Date:** 2026-02-07  
**Status:** Completed

### Configuration

| Parameter | Value |
|-----------|-------|
| Models | XceptionNet, EfficientNet-B0 |
| Aggregation Methods | Mean, Majority Vote, Confidence Weighted |
| Frames per Video | 20 |

### Results

| Method | XceptionNet Acc | EfficientNet-B0 Acc |
|--------|-----------------|---------------------|
| Mean Probability | 0.9534 | 0.9312 |
| Majority Voting | 0.9489 | 0.9256 |
| Confidence Weighted | 0.9567 | 0.9345 |

### Observations
- Confidence-weighted aggregation performs best
- Video-level accuracy exceeds frame-level accuracy
- Mean probability is a strong baseline

---

## Experiment 005: Frame Sampling Rate Impact

**Date:** 2026-02-09  
**Status:** Completed

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | XceptionNet |
| Sampling Rates | 10, 20, 40, 80 frames per video |

### Results

| Frames | Accuracy | Inference Time |
|--------|----------|----------------|
| 10 | 0.9312 | 85 ms |
| 20 | 0.9467 | 162 ms |
| 40 | 0.9523 | 318 ms |
| 80 | 0.9534 | 624 ms |

### Observations
- Diminishing returns beyond 20 frames
- 20 frames provides best accuracy-speed tradeoff
- Recommended for production deployment

---

## Experiment 006: GradCAM Explainability Analysis

**Date:** 2026-02-11  
**Status:** Completed

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | XceptionNet |
| Samples | 100 random test samples |
| Layers Analyzed | Final convolutional layer |

### Findings

1. Models focus on facial boundaries, eye regions, and mouth areas
2. Real videos show natural texture patterns
3. Fake videos exhibit boundary artifacts and blending seams
4. Eye region is most discriminative for manipulation detection
5. Mouth region artifacts prominent in face-swapping techniques
