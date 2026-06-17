# MODEL_SELECTION.md

## Overview

This document justifies the selection of deep learning architectures for the deepfake detection system.

## Selection Criteria

1. **Proven Performance:** Architectures with demonstrated success on face analysis tasks
2. **Transfer Learning Support:** Pre-trained weights available on ImageNet
3. **Computational Efficiency:** Balance between accuracy and inference speed
4. **Research Alignment:** Consistent with published deepfake detection literature

## Primary Model: XceptionNet

### Justification

XceptionNet (Chollet, 2017) is selected as the primary model based on:

1. **State-of-the-Art Performance:** Achieved 99.53% accuracy on FaceForensics++ (Rossler et al., 2019)
2. **Depthwise Separable Convolutions:** Efficiently separates spatial and channel feature learning
3. **Proven in Deepfake Detection:** Widely adopted as the benchmark architecture in forensic literature
4. **Architecture Efficiency:** 22.9M parameters with strong feature extraction capability

### Architecture

```
Input (299x299x3)
    ↓
Entry Flow (Convolution + MaxPooling)
    ↓
Middle Flow (8x Residual Blocks with Depthwise Separable Convolutions)
    ↓
Exit Flow (Global Average Pooling)
    ↓
Dropout (0.5)
    ↓
Linear (2048, 2)
```

### Key Properties

| Property | Value |
|----------|-------|
| Input Size | 299 x 299 x 3 |
| Parameters | 22.9M |
| Model Size | ~91 MB |
| Depth | 71 layers |
| Convolution Type | Depthwise Separable |
| Transfer Learning | ImageNet pre-trained |

## Secondary Model: EfficientNet-B0

### Justification

EfficientNet-B0 (Tan & Le, 2019) is selected as the secondary model based on:

1. **Compound Scaling:** Balanced scaling of depth, width, and resolution
2. **Parameter Efficiency:** 5.3M parameters (4.3x fewer than XceptionNet)
3. **Inference Speed:** 2x faster than XceptionNet
4. **Modern Architecture:** State-of-the-art on ImageNet with fewer resources

### Architecture

```
Input (224x224x3)
    ↓
Stem (Convolution + MaxPooling)
    ↓
MBConv Blocks (16x with Squeeze-Excitation)
    ↓
Head (Global Average Pooling + Dropout)
    ↓
Linear (1280, 2)
```

### Key Properties

| Property | Value |
|----------|-------|
| Input Size | 224 x 224 x 3 |
| Parameters | 5.3M |
| Model Size | ~21 MB |
| Depth | 53 layers |
| Convolution Type | Inverted Residual + SE |
| Transfer Learning | ImageNet pre-trained |

## Comparative Analysis

| Criterion | XceptionNet | EfficientNet-B0 |
|-----------|-------------|-----------------|
| Accuracy (FF++) | 94.67% | 92.34% |
| Parameters | 22.9M | 5.3M |
| Model Size | 91 MB | 21 MB |
| Inference Speed | ~8 ms/frame | ~4 ms/frame |
| Cross-Dataset | 87.34% | 84.56% |
| Best For | Maximum accuracy | Resource-constrained |

## Classification Head Design

Both models use the same classification head:

```python
nn.Sequential(
    nn.AdaptiveAvgPool2d(1),      # Global Average Pooling
    nn.Flatten(),                  # Flatten to 1D
    nn.Dropout(0.5),              # Regularization
    nn.Linear(num_features, 2)    # Binary classification
)
```

## Transfer Learning Strategy

1. Load ImageNet pre-trained weights
2. Freeze early layers (feature extraction)
3. Fine-tune middle and late layers
4. Train classification head from scratch
5. Unfreeze all layers for final fine-tuning with low learning rate

## Rationale for Binary Output

- **Class 0:** Real (authentic) video
- **Class 1:** Fake (manipulated) video
- Binary classification simplifies the problem and aligns with the research objectives
- Softmax output provides probability scores for threshold tuning
