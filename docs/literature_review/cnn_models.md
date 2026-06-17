# CNN Models for Deepfake Detection Review

## Overview

This review surveys CNN architectures used for deepfake detection, comparing their effectiveness.

## Architectures

### 1. VGG-16/19
- **Parameters:** 138M (VGG-16), 200M (VGG-19)
- **Input:** 224 × 224
- **Strengths:** Simple, well-understood
- **Weaknesses:** Very large, slow inference

### 2. ResNet-50
- **Parameters:** 25.6M
- **Input:** 224 × 224
- **Strengths:** Residual connections, stable training
- **Weaknesses:** Limited receptive field

### 3. XceptionNet
- **Parameters:** 22.9M
- **Input:** 299 × 299
- **Strengths:** Depthwise separable convolutions, strong forensic performance
- **Weaknesses:** Large input requirement

### 4. EfficientNet-B0
- **Parameters:** 5.3M
- **Input:** 224 × 224
- **Strengths:** Parameter efficient, compound scaling
- **Weaknesses:** Less spatial detail

### 5. MesoNet
- **Parameters:** 10K
- **Input:** Variable
- **Strengths:** Lightweight, fast
- **Weaknesses:** Lower accuracy on modern datasets

## Comparison Table

| Architecture | Params | Input | Accuracy (FF++) | Speed | Our Choice |
|-------------|--------|-------|----------------|-------|------------|
| VGG-16 | 138M | 224 | ~92% | Slow | No |
| ResNet-50 | 25.6M | 224 | ~94% | Medium | No |
| XceptionNet | 22.9M | 299 | 97.9% | Medium | **Yes** |
| EfficientNet-B0 | 5.3M | 224 | ~96% | Fast | **Yes** |
| MesoNet | 10K | Variable | ~90% | Very Fast | No |

## Key Findings

1. **Depthwise separable convolutions** are most effective for forensic analysis
2. **Transfer learning** from ImageNet is essential for good performance
3. **Model size** doesn't correlate with forensic detection performance
4. **Input resolution** matters - larger inputs capture more artifacts
5. **Residual connections** help preserve manipulation traces

## Our Selection Rationale

**Primary: XceptionNet**
- Best accuracy on FaceForensics++ benchmark
- Depthwise separable convolutions excel at spatial forensics
- Residual connections preserve subtle artifacts
- Well-documented in deepfake literature

**Secondary: EfficientNet-B0**
- 4× fewer parameters than XceptionNet
- Faster inference for deployment
- Competitive accuracy
- Validates architecture diversity

## References

- Rossler, A., et al. (2019). FaceForensics++. ICCV.
- Li, L., et al. (2018). Face X-ray. CVPR.
- Afchar, D., et al. (2018). MesoNet. IEEE.
