# Research Contribution Statement

**Author:** Olamijulo Israel D  
**Matric Number:** CYS/22/9071  
**Project:** Detection of Social Media Deepfake Contents Using Deep Learning Algorithm

---

## Research Gap

Existing deepfake detection models (XceptionNet, EfficientNet) report high accuracy under controlled laboratory conditions (99%+ on FaceForensics++ c0). However, social media platforms apply aggressive compression, resizing, and re-encoding that degrade video quality. Most published evaluations do not systematically measure how these real-world distortions impact detection performance. This creates a gap between academic benchmarks and practical deployment readiness.

## Contribution Points

This research contributes four concrete findings to the field of social media deepfake forensics:

### Contribution 1: Compression Impact Quantification

**What we measure:** How detection accuracy degrades across three compression levels:
- c0 (raw/uncompressed)
- c23 (light compression — typical of high-quality web video)
- c40 (heavy compression — typical of WhatsApp, Facebook, TikTok)

**Why it matters:** Social media platforms compress videos automatically. A model that achieves 95% on raw footage may drop to 70% after compression. This experiment quantifies the exact degradation curve for both XceptionNet and EfficientNet, providing actionable data for practitioners.

**Expected outcome:** Accuracy degradation curves showing each model's robustness to compression. XceptionNet (designed for spatial feature extraction) is expected to be more sensitive to compression artifacts than EfficientNet (which uses compound scaling across resolution, depth, and width).

### Contribution 2: Cross-Dataset Generalization

**What we measure:** How models trained on FaceForensics++ perform when evaluated on Celeb-DF (and vice versa), without any fine-tuning.

**Why it matters:** FaceForensics++ uses controlled synthesis methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures). Celeb-DF uses a more realistic synthesis pipeline with YouTube celebrity interviews, complex backgrounds, and diverse lighting. A model that only works on its training distribution is not deployable in the wild.

**Expected outcome:** Generalization gap measurements. Models trained on FaceForensics++ are expected to show significant accuracy drops on Celeb-DF, demonstrating the need for diverse training data or domain adaptation techniques.

### Contribution 3: Head-to-Head Model Comparison

**What we measure:** Direct comparison of XceptionNet vs EfficientNet across all conditions (compression levels, datasets, manipulation methods).

**Why it matters:** Both architectures are widely used for deepfake detection, but direct comparisons under identical experimental conditions are rare. This experiment provides a fair comparison using the same preprocessing pipeline, training procedure, and evaluation metrics.

**Expected outcome:** Per-condition performance breakdown showing which architecture is more robust to specific distortion types and which manipulation methods are hardest to detect.

### Contribution 4: Practical Deployment Assessment

**What we measure:** Inference speed (frames per second), model size (parameters, file size), and memory footprint for both architectures.

**Why it matters:** Social media platforms process millions of videos daily. A model with 95% accuracy but 100ms per frame is impractical for real-time screening. This experiment provides the engineering metrics needed to assess deployment feasibility.

**Expected outcome:** Speed-accuracy tradeoff analysis showing which model offers the best balance for social media deployment scenarios.

---

## Experimental Design

| Experiment | Models | Datasets | Conditions | Metrics |
|------------|--------|----------|------------|---------|
| Compression Impact | XceptionNet, EfficientNet | FaceForensics++ | c0, c23, c40 | Accuracy, F1, ROC-AUC |
| Cross-Dataset | XceptionNet, EfficientNet | FF++ → Celeb-DF | 2 directions | Accuracy, F1, ROC-AUC |
| Model Comparison | XceptionNet, EfficientNet | Both | All conditions | All standard metrics |
| Deployment Assessment | XceptionNet, EfficientNet | FaceForensics++ | c0 | FPS, params, memory |

---

## Expected Chapter 4 Structure

1. **Compression Impact Results** — Degradation curves, tables
2. **Cross-Dataset Results** — Generalization gap analysis
3. **Model Comparison Results** — Head-to-head tables and charts
4. **Deployment Assessment** — Speed-accuracy tradeoff analysis
5. **Discussion** — Interpretation of findings, limitations, future work

---

## Alignment with Thesis Objectives

| Thesis Objective | Contribution |
|------------------|--------------|
| 1. Design a deep learning-based detection model | Contributions 1, 2, 3 (evaluation of designed system) |
| 2. Frame-level and video-level classification | Contributions 1, 2, 3 (both models perform frame classification) |
| 3. Evaluate using accuracy, precision, recall, F1, ROC-AUC | All contributions (standard metrics throughout) |

This research does not claim to invent a new architecture. Instead, it contributes rigorous, reproducible evaluation under realistic social media conditions — filling a documented gap in the existing literature.
