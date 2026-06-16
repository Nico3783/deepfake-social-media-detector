# REQUIREMENTS.md

## Functional Requirements

### FR-001

The system shall load video datasets.

---

### FR-002

The system shall extract frames from videos.

---

### FR-003

The system shall detect human faces.

---

### FR-004

The system shall crop detected facial regions.

---

### FR-005

The system shall normalize images.

---

### FR-006

The system shall train deep learning models.

---

### FR-007

The system shall support XceptionNet.

---

### FR-008

The system shall support EfficientNet.

---

### FR-009

The system shall perform frame-level prediction.

---

### FR-010

The system shall perform video-level prediction.

---

### FR-011

The system shall calculate:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

### FR-012

The system shall generate evaluation reports.

---

### FR-013

The system shall save trained models.

---

### FR-014

The system shall load trained models.

---

### FR-015

The system shall expose inference APIs.

---

### FR-016

The system shall support experiment tracking.

---

### FR-017

The system shall generate Chapter 4 assets.

---

### FR-018

The system shall generate visualizations.

---

# Non-Functional Requirements

## NFR-001

Code must be modular.

---

## NFR-002

Code must be documented.

---

## NFR-003

Experiments must be reproducible.

---

## NFR-004

Training must be configurable.

---

## NFR-005

The system must support GPU acceleration.

---

## NFR-006

The system must support future cloud deployment.

---

## NFR-007

The system must support extension with additional models.

---

## NFR-008

The system must maintain academic traceability.

---

# Performance Requirements

Target Accuracy:

≥ 85%

Target F1 Score:

≥ 0.85

Target ROC-AUC:

≥ 0.90

These are target goals, not guaranteed outcomes.

---

# Documentation Requirements

Every module shall contain:

* Purpose
* Inputs
* Outputs
* Dependencies

---

# Testing Requirements

Coverage Areas:

* Data pipeline
* Preprocessing
* Model loading
* Training
* Evaluation
* Inference
* API

---

# Deliverables

The repository shall produce:

1. Trained model
2. Evaluation reports
3. Visualizations
4. API
5. Experiment logs
6. Thesis-ready outputs
7. Final dissertation support materials
