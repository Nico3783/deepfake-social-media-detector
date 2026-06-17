# Inference Pipeline

## Overview

The inference pipeline handles real-time deepfake detection for video and image inputs.

## Pipeline Components

### 1. Input Handling

**Video Input:**
- Accept .mp4, .avi, .mov, .mkv formats
- Extract frames at configurable FPS (default: 1 FPS)
- Support frame count limits for long videos

**Image Input:**
- Accept .jpg, .jpeg, .png, .bmp formats
- Direct face detection and classification

### 2. Preprocessing

```python
# Same pipeline as training:
# 1. Frame extraction (video only)
# 2. Face detection (RetinaFace)
# 3. Face cropping (30% padding)
# 4. Resize (299x299 or 224x224)
# 5. Normalize (ImageNet stats)
# 6. To tensor
```

### 3. Model Inference

```python
# Load trained model
model = load_model(checkpoint_path)
model.eval()

# Batch inference on extracted faces
with torch.no_grad():
    outputs = model(face_tensors)
    probabilities = F.softmax(outputs, dim=1)
```

### 4. Video Aggregation

| Method | Description | Use Case |
|--------|-------------|----------|
| Mean Probability | Average of all frame predictions | Default, robust |
| Majority Voting | Most frequent frame prediction | Noise-resistant |
| Confidence Weighted | Weighted by prediction confidence | High-precision |

### 5. Post-Processing

- Apply threshold (default: 0.5) for binary classification
- Generate confidence scores
- Compute per-frame and video-level predictions
- Optional: GradCAM heatmap generation

### 6. Response Format

**Video Prediction:**
```json
{
  "filename": "video.mp4",
  "prediction": "fake",
  "confidence": 0.94,
  "probability_fake": 0.94,
  "probability_real": 0.06,
  "frames_analyzed": 20,
  "aggregation_method": "mean",
  "processing_time_ms": 162
}
```

**Image Prediction:**
```json
{
  "filename": "image.jpg",
  "prediction": "real",
  "confidence": 0.87,
  "probability_fake": 0.13,
  "probability_real": 0.87,
  "face_detected": true,
  "processing_time_ms": 45
}
```

## Performance Requirements

| Metric | Target |
|--------|--------|
| Video Inference (20 frames) | < 200 ms |
| Image Inference | < 100 ms |
| Throughput | > 10 videos/sec |

## Edge Cases

- **No face detected:** Return error with face_detected=false
- **Multiple faces:** Classify largest face or all faces
- **Blurry/low quality:** Return prediction with lower confidence
- **Very long videos:** Limit frame count or use sampling
