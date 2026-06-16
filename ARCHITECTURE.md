# ARCHITECTURE.md

## System Overview

The system detects whether a video contains authentic or manipulated facial content using deep learning.

---

# High-Level Architecture

Video Input

↓

Frame Extraction

↓

Face Detection

↓

Face Cropping

↓

Image Normalization

↓

Feature Extraction

↓

Deep Learning Model

↓

Frame Classification

↓

Video Aggregation

↓

Final Prediction

↓

Reporting Layer

---

# Layer 1 — Data Layer

Responsibilities:

* Dataset storage
* Dataset organization
* Metadata management

Datasets:

* FaceForensics++
* Celeb-DF

Output:

Structured train/validation/test datasets.

---

# Layer 2 — Preprocessing Layer

Modules:

## Frame Extractor

Responsibilities:

* Read videos
* Sample frames

Output:

Frames

---

## Face Detector

Responsibilities:

* Locate faces

Recommended:

* RetinaFace
* MTCNN

Output:

Face coordinates

---

## Face Cropper

Responsibilities:

* Extract face regions

Output:

Face images

---

## Normalization Module

Responsibilities:

* Resize images
* Normalize pixel values

Output:

Model-ready tensors

---

# Layer 3 — Feature Learning Layer

Primary Model:

XceptionNet

Secondary Model:

EfficientNet

Responsibilities:

* Learn spatial features
* Learn forensic artifacts
* Learn manipulation patterns

Output:

Feature embeddings

---

# Layer 4 — Classification Layer

Responsibilities:

* Binary classification

Classes:

0 = Real

1 = Fake

Output:

Prediction scores

---

# Layer 5 — Aggregation Layer

Responsibilities:

Convert frame predictions into video predictions.

Methods:

* Mean probability
* Majority voting
* Confidence weighting

Output:

Video-level classification

---

# Layer 6 — Evaluation Layer

Responsibilities:

* Accuracy calculation
* Precision calculation
* Recall calculation
* F1 calculation
* ROC-AUC calculation

Output:

Performance reports

---

# Layer 7 — API Layer

Framework:

FastAPI

Endpoints:

/predict-video

/predict-image

/health

/model-info

---

# Layer 8 — Visualization Layer

Responsibilities:

* Training curves
* ROC curves
* Confusion matrices
* Prediction summaries

Output:

Publication-quality figures

---

# Storage Architecture

datasets/

↓

processed/

↓

training/

↓

outputs/

↓

reports/

---

# Security Considerations

Validate:

* File uploads
* Input formats
* Dataset integrity

Prevent:

* Invalid inference requests
* Corrupted inputs

---

# Scalability Considerations

Architecture should support:

* Additional datasets
* Additional models
* Cloud deployment
* Batch processing

without redesign.
