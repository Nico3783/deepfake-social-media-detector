# Deployment Pipeline

## Overview

The deployment pipeline handles model serving, containerization, and cloud deployment.

## Deployment Architecture

### Local Development

```
FastAPI Server (localhost:8000)
    ↓
Model Checkpoint (outputs/models/)
    ↓
Inference Pipeline
    ↓
REST API Endpoints
```

### Production Deployment

```
Load Balancer (Nginx)
    ↓
FastAPI Container (Docker)
    ↓
Model Storage (Persistent Volume)
    ↓
Health Checks + Monitoring
```

## Docker Configuration

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY outputs/models/ outputs/models/

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./outputs/models:/app/outputs/models
    environment:
      - MODEL_PATH=/app/outputs/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## API Endpoints

### POST /predict-video
- **Input:** Video file (multipart/form-data)
- **Output:** Prediction JSON with probability, confidence, processing time
- **Max File Size:** 100MB

### POST /predict-image
- **Input:** Image file (multipart/form-data)
- **Output:** Prediction JSON with probability, confidence, face detection info
- **Max File Size:** 10MB

### GET /health
- **Output:** Server status, model availability

### GET /model-info
- **Output:** Model architecture, parameters, training info

## Cloud Deployment Options

### AWS
- **Compute:** ECS Fargate or EC2
- **Storage:** EFS for model artifacts
- **Load Balancer:** ALB

### Google Cloud
- **Compute:** Cloud Run or GCE
- **Storage:** Persistent Disk or GCS
- **Load Balancer:** Cloud Load Balancing

### Azure
- **Compute:** Container Instances or AKS
- **Storage:** Azure Files
- **Load Balancer:** Azure Load Balancer

## Monitoring

- **Health Checks:** Automatic restart on failure
- **Logging:** Structured JSON logs
- **Metrics:** Request count, latency, error rate
- **Alerting:** Model availability, high latency
