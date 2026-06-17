# EfficientNet Architecture Review

## Overview

EfficientNet (Tan & Le, 2019) introduces compound scaling to jointly optimize depth, width, and resolution for neural networks.

## Architecture

### Compound Scaling

EfficientNet uses three scaling dimensions:
1. **Depth (d):** Number of layers
2. **Width (w):** Number of channels per layer
3. **Resolution (r):** Input image resolution

Compound scaling formula:
```
d = α^φ
w = β^φ
r = γ^φ
α · β^2 · γ^2 ≈ 2
```

Where φ controls overall scaling.

### MBConv Blocks

Mobile Inverted Bottleneck (MBConv) blocks:
```
Input
  ↓
1×1 Conv (expand channels)
  ↓
Depthwise Conv (3×3 or 5×5)
  ↓
Squeeze-Excitation (SE) block
  ↓
1×1 Conv (project channels)
  ↓
+ Input (residual connection)
```

### EfficientNet Variants

| Model | Parameters | Input Size | Top-1 Acc |
|-------|-----------|-----------|-----------|
| B0 | 5.3M | 224 × 224 | 77.1% |
| B1 | 7.8M | 240 × 240 | 79.1% |
| B2 | 9.2M | 260 × 260 | 80.1% |
| B3 | 12M | 300 × 300 | 81.6% |
| B4 | 19M | 380 × 380 | 82.9% |
| B5 | 30M | 456 × 456 | 83.6% |
| B6 | 43M | 528 × 528 | 84.0% |
| B7 | 66M | 600 × 600 | 84.3% |

**We use EfficientNet-B0** for the deepfake detector:
- Most parameter-efficient
- Fastest inference
- Still competitive accuracy

## Key Properties

| Property | Value |
|----------|-------|
| Parameters | 5.3M (B0) |
| Input Size | 224 × 224 |
| MBConv Blocks | Yes |
| Squeeze-Excitation | Yes |
| Compound Scaling | Yes |
| Pre-trained | ImageNet |

## Performance on Deepfake Detection

**Advantages over XceptionNet:**
1. 4× fewer parameters (5.3M vs 22.9M)
2. Faster inference
3. Similar accuracy
4. Better memory efficiency

**Trade-offs:**
1. Smaller receptive field
2. Less spatial detail capture
3. May miss fine-grained artifacts

## Implementation Notes

- Input normalization: scale to [0, 1] range
- Swish activation function
- DropConnect regularization
- Stochastic depth for regularization

## References

- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
