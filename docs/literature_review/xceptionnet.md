# XceptionNet Architecture Review

## Overview

XceptionNet (Chollet, 2017) is a CNN architecture that uses depthwise separable convolutions as the core building block.

## Architecture

### Depthwise Separable Convolution

Standard convolution:
- Filters: K filters of size Dk × Dk × M
- Computation: O(Dk × Dk × M × N × H × W)

Depthwise separable convolution:
1. **Depthwise convolution:** One filter per channel
2. **Pointwise convolution:** 1×1 conv to combine channels
- Computation: O(Dk × Dk × M × H × W + M × N × H × W)
- Reduction: ~8-9x fewer parameters

### Xception Architecture

```
Entry Flow:
- 2 conv + ReLU
- Separable conv block (strided) + residual
- Separable conv block (strided) + residual
- Separable conv block (strided) + residual

Middle Flow (8 repetitions):
- ReLU → Separable conv
- ReLU → Separable conv
- ReLU → Separable conv
- + residual

Exit Flow:
- Separable conv (strided) + residual
- Separable conv + residual
- Global Average Pooling
- Fully connected
```

## Key Properties

| Property | Value |
|----------|-------|
| Parameters | 22.9M |
| Input Size | 299 × 299 |
| Depthwise Separable | Yes |
| Residual Connections | Yes |
| Pre-trained | ImageNet |

## Performance on Deepfake Detection

**FaceForensics++ (Rossler et al., 2019):**
- Raw (c0): 99.53% accuracy
- Low quality (c40): 93.78% accuracy
- High quality (c23): 97.91% accuracy

**Why XceptionNet works for forensics:**
1. Depthwise separable convolutions capture fine-grained spatial artifacts
2. Residual connections preserve manipulation traces
3. Large receptive field captures global inconsistencies
4. Pre-trained features transfer well to forensic tasks

## Implementation Notes

- Requires input normalization to [0, 1] range
- Batch normalization after each convolution
- Dropout (0.5) before final FC layer
- Global average pooling reduces spatial dimensions

## References

- Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. CVPR.
- Rossler, A., et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. ICCV.
