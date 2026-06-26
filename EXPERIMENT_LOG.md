# EXPERIMENT_LOG.md

## Overview

This document tracks all experiments conducted during the project, including configurations, results, and observations.

**Note:** All results logged here must come from actual experiments. Fabricated results violate research integrity rules.

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

**Date:** [PENDING]  
**Status:** Not Started

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | XceptionNet |
| Dataset | FaceForensics++ c23 |
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
| Accuracy | - |
| Precision | - |
| Recall | - |
| F1-Score | - |
| ROC-AUC | - |
| Training Time | - |
| Convergence Epoch | - |

### Observations
- [To be filled after experiment]

---

## Experiment 002: EfficientNet-B0 on FaceForensics++ c23

**Date:** [PENDING]  
**Status:** Not Started

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | EfficientNet-B0 |
| Dataset | FaceForensics++ c23 |
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
| Accuracy | - |
| Precision | - |
| Recall | - |
| F1-Score | - |
| ROC-AUC | - |
| Training Time | - |
| Convergence Epoch | - |

### Observations
- [To be filled after experiment]

---

## Experiment 003: Cross-Dataset Validation (Celeb-DF)

**Date:** [PENDING]  
**Status:** Not Started

### Configuration

| Parameter | Value |
|-----------|-------|
| Models | XceptionNet, EfficientNet-B0 |
| Training Dataset | FaceForensics++ c23 |
| Evaluation Dataset | Celeb-DF v2 |

### Results

| Metric | XceptionNet | EfficientNet-B0 |
|--------|-------------|-----------------|
| Accuracy | - | - |
| Precision | - | - |
| Recall | - | - |
| F1-Score | - | - |
| ROC-AUC | - | - |

### Observations
- [To be filled after experiment]

---

## Experiment 004: Video Aggregation Methods

**Date:** [PENDING]  
**Status:** Not Started

### Configuration

| Parameter | Value |
|-----------|-------|
| Models | XceptionNet, EfficientNet-B0 |
| Aggregation Methods | Mean, Majority Vote, Confidence Weighted |

### Results

| Method | XceptionNet Acc | EfficientNet-B0 Acc |
|--------|-----------------|---------------------|
| Mean Probability | - | - |
| Majority Voting | - | - |
| Confidence Weighted | - | - |

### Observations
- [To be filled after experiment]

---

## Experiment 005: Frame Sampling Rate Impact

**Date:** [PENDING]  
**Status:** Not Started

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | XceptionNet |
| Sampling Rates | 10, 20, 40, 80 frames per video |

### Results

| Frames | Accuracy | Inference Time |
|--------|----------|----------------|
| 10 | - | - |
| 20 | - | - |
| 40 | - | - |
| 80 | - | - |

### Observations
- [To be filled after experiment]

---

## Experiment 006: GradCAM Explainability Analysis

**Date:** [PENDING]  
**Status:** Not Started

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | XceptionNet |
| Samples | 100 random test samples |
| Layers Analyzed | Final convolutional layer |

### Findings
- [To be filled after experiment]
