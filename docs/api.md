# API Documentation

## Overview

The Deepfake Detection API provides RESTful endpoints for classifying images and videos as real or fake using deep learning models (XceptionNet or EfficientNet).

**Base URL:** `http://localhost:8000`

**API Version:** v1

---

## Authentication

Currently, the API does not require authentication. For production deployment, implement appropriate authentication mechanisms.

---

## Endpoints

### Health Check

Check service health and model availability.

**Endpoint:** `GET /api/v1/health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "xception",
  "device": "cuda"
}
```

**Status Codes:**
- `200`: Service is healthy

---

### Image Prediction

Classify a single image as real or fake.

**Endpoint:** `POST /api/v1/predict/image`

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file` (required): Image file (JPEG, PNG)
  - `threshold` (optional, default=0.5): Classification threshold [0.0-1.0]

**Response:**
```json
{
  "filename": "test_image.jpg",
  "is_fake": false,
  "confidence": 0.9234,
  "fake_probability": 0.0766,
  "real_probability": 0.9234,
  "face_detected": true,
  "bounding_box": [120, 80, 350, 400]
}
```

**Status Codes:**
- `200`: Prediction successful
- `400`: Invalid file type (not an image)
- `500`: Prediction failed
- `503`: Model not loaded

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/image" \
  -F "file=@test_image.jpg" \
  -F "threshold=0.5"
```

**Example (Python):**
```python
import requests

url = "http://localhost:8000/api/v1/predict/image"
with open("test_image.jpg", "rb") as f:
    response = requests.post(url, files={"file": f}, data={"threshold": 0.5})
print(response.json())
```

---

### Video Prediction

Classify a video as real or fake by analyzing extracted frames.

**Endpoint:** `POST /api/v1/predict/video`

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file` (required): Video file (MP4, AVI, MOV)
  - `threshold` (optional, default=0.5): Classification threshold [0.0-1.0]
  - `aggregation` (optional, default="mean"): Aggregation method
    - `mean`: Average frame probabilities
    - `majority`: Majority vote of frame decisions
    - `confidence_weighted`: Weight by prediction confidence
  - `frame_sample_rate` (optional, default=5): Process every Nth frame [1-30]

**Response:**
```json
{
  "filename": "test_video.mp4",
  "is_fake": true,
  "confidence": 0.8756,
  "fake_probability": 0.8756,
  "real_probability": 0.1244,
  "total_frames": 150,
  "frames_with_face": 142,
  "aggregation_method": "mean",
  "frame_results": [
    {
      "frame_index": 0,
      "fake_probability": 0.9123,
      "is_fake": true
    },
    {
      "frame_index": 5,
      "fake_probability": 0.8876,
      "is_fake": true
    }
  ]
}
```

**Status Codes:**
- `200`: Prediction successful
- `400`: Invalid file type (not a video)
- `500`: Prediction failed
- `503`: Video predictor not loaded

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/video" \
  -F "file=@test_video.mp4" \
  -F "threshold=0.5" \
  -F "aggregation=mean" \
  -F "frame_sample_rate=5"
```

**Example (Python):**
```python
import requests

url = "http://localhost:8000/api/v1/predict/video"
with open("test_video.mp4", "rb") as f:
    response = requests.post(
        url,
        files={"file": f},
        data={
            "threshold": 0.5,
            "aggregation": "mean",
            "frame_sample_rate": 5,
        },
    )
print(response.json())
```

---

## Data Models

### AggregationMethod (enum)

| Value | Description |
|-------|-------------|
| `mean` | Average of frame probabilities |
| `majority` | Majority vote of frame decisions |
| `confidence_weighted` | Weighted average by prediction confidence |

### ImagePredictionResponse

| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | Input filename |
| `is_fake` | boolean | Fake/real classification |
| `confidence` | float | Prediction confidence [0, 1] |
| `fake_probability` | float | Probability of being fake [0, 1] |
| `real_probability` | float | Probability of being real [0, 1] |
| `face_detected` | boolean | Whether a face was detected |
| `bounding_box` | tuple[int, int, int, int] \| null | Face bounding box (x1, y1, x2, y2) |

### VideoPredictionResponse

| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | Input video filename |
| `is_fake` | boolean | Video-level classification |
| `confidence` | float | Video-level confidence [0, 1] |
| `fake_probability` | float | Aggregated fake probability |
| `real_probability` | float | Aggregated real probability |
| `total_frames` | integer | Number of frames extracted |
| `frames_with_face` | integer | Frames where a face was detected |
| `aggregation_method` | string | Aggregation method used |
| `frame_results` | array[FrameResult] | Per-frame prediction details |

### FrameResult

| Field | Type | Description |
|-------|------|-------------|
| `frame_index` | integer | Frame number in the video |
| `fake_probability` | float | Fake probability for this frame |
| `is_fake` | boolean | Frame-level decision |

### HealthResponse

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Service status |
| `model_loaded` | boolean | Whether a model is loaded |
| `model_type` | string \| null | Loaded model architecture |
| `device` | string | Computation device |

---

## Error Handling

All error responses follow the format:

```json
{
  "error": "Error message",
  "detail": "Additional details"
}
```

### Common Errors

| Status Code | Error | Description |
|-------------|-------|-------------|
| 400 | Bad Request | Invalid file type or parameters |
| 500 | Internal Server Error | Prediction failed |
| 503 | Service Unavailable | Model not loaded |

---

## Running the API

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn src.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

---

## Performance Considerations

- **Image Prediction:** ~50-100ms per image (GPU), ~200-500ms (CPU)
- **Video Prediction:** Depends on video length and frame_sample_rate
  - Example: 30s video @ 30fps = 900 frames
  - With frame_sample_rate=5: 180 frames processed
  - Estimated: ~10-30s (GPU), ~30-60s (CPU)

**Optimization Tips:**
- Use `frame_sample_rate` parameter to reduce processed frames
- Use `mean` aggregation for fastest results
- Use `confidence_weighted` for most accurate results
- Deploy on GPU for production workloads

---

## Limitations

1. **File Size:** Default upload limit is 100MB. Adjust in production.
2. **Video Length:** Very long videos (>5min) may cause memory issues.
3. **Face Detection:** Videos without faces return low-confidence predictions.
4. **Model Loading:** API returns 503 until model is loaded at startup.

---

## Research Traceability

This API supports the following research objectives:

- **Objective:** Deployable deepfake detection system
- **Methodology:** RESTful API wrapping the inference pipeline
- **Implementation:** `src/api/routes.py`, `src/api/app.py`

The API enables:
- Real-time deepfake detection for social media content
- Integration with existing forensic workflows
- Scalable deployment for production use