# Deepfake Detection Literature Review

## Overview

This document surveys the current state of deepfake detection research, focusing on deep learning-based approaches.

## Detection Approaches

### 1. Spatial Analysis Methods

**Eye Blinking Detection (Li et al., 2018):**
- Early approach detecting inconsistent eye blinking patterns
- Limited to detecting specific manipulation artifacts
- Outperformed by modern deep learning methods

**Face Forensic Analysis (Afchar et al., 2018):**
- Uses MesoNet architecture for facial analysis
- Focuses on mesoscopic features (between pixel and global levels)
- Achieved ~90% accuracy on early datasets

### 2. CNN-Based Detection

**XceptionNet (Rossler et al., 2019):**
- State-of-the-art on FaceForensics++
- Uses depthwise separable convolutions
- 99.53% accuracy on raw (c0) data
- 95.73% accuracy on compressed (c40) data
- Became the benchmark architecture

**EfficientNet (Tan & Le, 2019):**
- Compound scaling of depth, width, resolution
- More parameter-efficient than XceptionNet
- Competitive performance with fewer resources

**Capsule Networks (Nguyen et al., 2019):**
- Captures spatial relationships between features
- Better generalization than standard CNNs
- Higher computational cost

### 3. Temporal Analysis Methods

**RNN/LSTM-Based (Gu et al., 2019):**
- Models temporal inconsistencies across frames
- Multi-attention mechanisms for frame selection
- Improved detection of temporal artifacts

**3D-CNN Methods:**
- Extract spatiotemporal features jointly
- Higher computational cost but better temporal modeling

### 4. Frequency Analysis

**Spectral Analysis (Qian et al., 2020):**
- Analyzes frequency domain artifacts
- Detects spectral inconsistencies from GAN generation
- Robust to spatial compression

### 5. Attention Mechanisms

**Multi-Attention (Gu et al., 2019):**
- Identifies most discriminative facial regions
- Focuses on manipulation artifacts
- Improved accuracy and interpretability

## Key Findings

1. **Depthwise separable convolutions** are highly effective for forensic analysis
2. **Transfer learning** from ImageNet provides strong feature extractors
3. **Compression** significantly impacts detection performance
4. **Cross-dataset generalization** remains a major challenge
5. **Temporal analysis** can improve detection of subtle artifacts

## Research Gaps

1. Limited evaluation under realistic social media conditions
2. Poor cross-dataset generalization
3. Lack of adversarial robustness evaluation
4. Audio-visual fusion approaches underexplored
5. Real-time deployment constraints not addressed
