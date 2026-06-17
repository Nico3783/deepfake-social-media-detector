# Research Objectives

## Aim

The aim of this project is to design, implement, and evaluate a deep learning-based system capable of detecting deepfake videos on social media platforms.

## Primary Objectives

### Objective 1: CNN-Based Detection Model Design
Design a deep learning-based detection model using convolutional neural network architectures (XceptionNet and EfficientNet) with transfer learning from ImageNet pre-trained weights.

**Acceptance Criteria:**
- Implement XceptionNet architecture with depthwise separable convolutions
- Implement EfficientNet-B0 architecture with compound scaling
- Use transfer learning from ImageNet pre-trained weights
- Binary classification (real vs. fake) with probability output
- Model parameters and architecture documented in MODEL_SELECTION.md

### Objective 2: Frame-Level and Video-Level Classification
Implement a system capable of both frame-level and video-level classification of deepfake content, with multiple video aggregation strategies.

**Acceptance Criteria:**
- Extract and classify individual video frames
- Aggregate frame predictions to video-level using mean probability, majority voting, and confidence weighting
- Support configurable frame sampling rates
- Achieve video-level accuracy exceeding frame-level accuracy through temporal aggregation

### Objective 3: Comprehensive Evaluation
Evaluate the designed system using standard metrics: accuracy, precision, recall, F1-score, and ROC-AUC.

**Acceptance Criteria:**
- Calculate all five standard metrics on test sets
- Generate confusion matrices for detailed analysis
- Produce ROC curves with AUC computation
- Evaluate on FaceForensics++ (primary) and Celeb-DF (cross-dataset validation)
- Generate publication-quality visualizations
- Meet target thresholds: Accuracy >= 85%, F1 >= 0.85, ROC-AUC >= 0.90

## Secondary Objectives

### Objective 4: Explainability Analysis
Provide visual explanations of model predictions using GradCAM to identify which facial regions contribute most to detection decisions.

### Objective 5: API Deployment
Expose trained models via a FastAPI inference service for real-time deepfake detection.

### Objective 6: Reproducibility
Ensure all experiments are reproducible with fixed random seeds, documented hyperparameters, and version-controlled dataset splits.

## Research Alignment

| Objective | Research Question | Methodology |
|-----------|-------------------|-------------|
| 1 | RQ1: CNN effectiveness | Model implementation + training |
| 2 | RQ1: Classification approach | Frame + video aggregation |
| 3 | RQ1, RQ2, RQ3: Performance evaluation | Multi-metric evaluation |
| 4 | RQ4: Discriminative regions | GradCAM analysis |
| 5 | Practical deployment | FastAPI service |
| 6 | Research rigor | Reproducible pipeline |
