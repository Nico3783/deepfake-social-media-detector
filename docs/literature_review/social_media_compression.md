# Social Media Compression Review

## Overview

Social media platforms apply aggressive video compression to reduce bandwidth and storage. This document reviews compression techniques and their impact on deepfake detection.

## Platform Compression Characteristics

### YouTube
- **Codec:** H.264 (AVC), VP9
- **Quality:** Variable bitrate encoding
- **Resolution:** Up to 4K
- **Impact on detection:** Moderate quality degradation

### Facebook
- **Codec:** H.264 (AVC)
- **Quality:** c40-equivalent compression
- **Resolution:** Up to 1080p
- **Impact on detection:** Significant quality loss

### Twitter/X
- **Codec:** H.264 (AVC)
- **Quality:** High compression ratio
- **Resolution:** Up to 1080p
- **Impact on detection:** Heavy compression artifacts

### Instagram
- **Codec:** H.264 (AVC)
- **Quality:** Aggressive compression
- **Resolution:** Up to 1080p
- **Impact on detection:** Most aggressive compression

### TikTok
- **Codec:** H.264 (AVC)
- **Quality:** Medium compression
- **Resolution:** Up to 1080p
- **Impact on detection:** Moderate quality loss

## Compression Levels in FaceForensics++

| Level | Quality | SSIM | Visual Quality | Detection Impact |
|-------|---------|------|----------------|------------------|
| c0 | Raw | 1.00 | Perfect | No impact |
| c23 | High | ~0.92 | Good | Low impact |
| c40 | Low | ~0.82 | Poor | High impact |

## Impact on Detection

### Artifact Obscuration
- Compression removes high-frequency details
- Blends manipulation boundaries
- Reduces detectable inconsistencies

### Additional Artifacts
- JPEG blocking patterns
- Mosquito noise
- Ringing artifacts

### Model Robustness
- Models trained on raw data perform poorly on compressed data
- Training with compression augmentation improves robustness
- c23 is a reasonable compromise between quality and realism

## Mitigation Strategies

1. **Compression-Augmented Training:** Apply random compression during training
2. **Multi-Quality Evaluation:** Test across compression levels
3. **Robust Feature Extraction:** Use features invariant to compression
4. **Domain Adaptation:** Adapt models to social media conditions

## References

- Dang, H., et al. (2020). DeepRhythm: Exposing DeepFakes with Attentional Visual Rhythms. WACV.
- Gu, Z., et al. (2019). Recurrent Recursive Transformer for Face Forgery Detection. arXiv.
