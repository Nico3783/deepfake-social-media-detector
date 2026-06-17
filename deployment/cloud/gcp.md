# GCP Deployment Guide

## Prerequisites

1. Google Cloud SDK installed and configured
2. Docker installed
3. GCP project created

## Deployment Steps

### 1. Enable Required APIs

```bash
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com
```

### 2. Build and Push to Container Registry

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Build
docker build -t gcr.io/$(gcloud config get-value project)/deepfake-detector \
    -f deployment/Dockerfile .

# Push
docker push gcr.io/$(gcloud config get-value project)/deepfake-detector
```

### 3. Deploy to Cloud Run

```bash
gcloud run deploy deepfake-detector \
    --image gcr.io/$(gcloud config get-value project)/deepfake-detector \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --max-instances 10 \
    --min-instances 1 \
    --port 8000 \
    --set-env-vars MODEL_PATH=/app/outputs/models,LOG_LEVEL=info
```

### 4. Alternative: Deploy to GCE

#### Create VM:
```bash
gcloud compute instances create deepfake-detector-vm \
    --zone us-central1-a \
    --machine-type n1-standard-4 \
    --image-family deepfake-vm-image \
    --image-project my-project \
    --boot-disk-size 100GB \
    --tags deepfake-detector
```

#### Install Docker on VM:
```bash
ssh deepfake-detector-vm
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

#### Run Container:
```bash
docker run -d \
    --name deepfake-api \
    -p 8000:8000 \
    -v /data/models:/app/outputs/models \
    --restart unless-stopped \
    gcr.io/$(gcloud config get-value project)/deepfake-detector:latest
```

### 5. Set Up Load Balancer (Optional)

```bash
# Create health check
gcloud compute health-checks create http deepfake-health \
    --port 8000 \
    --request-path /health

# Create backend service
gcloud compute backend-services create deepfake-backend \
    --protocol HTTP \
    --health-checks deepfake-health \
    --global

# Add VM to backend
gcloud compute backend-services add-backend deepfake-backend \
    --instance-group deepfake-detector-ig \
    --instance-group-zone us-central1-a \
    --balancing-mode UTILIZATION \
    --max-utilization 0.8

# Create URL map
gcloud compute url-maps create deepfake-url-map \
    --default-service deepfake-backend

# Create HTTP proxy
gcloud compute target-http-proxies create deepfake-proxy \
    --url-map deepfake-url-map

# Create forwarding rule
gcloud compute forwarding-rules create deepfake-forwarding-rule \
    --global \
    --target-http-proxy deepfake-proxy \
    --ports 80
```

## Cost Estimation

| Resource | Monthly Cost |
|----------|--------------|
| Cloud Run (2 instances) | ~$40 |
| Container Registry | ~$5 |
| Cloud Load Balancing | ~$18 |
| **Total** | **~$63** |

## Monitoring

- Cloud Monitoring dashboards
- Cloud Logging for application logs
- Cloud Trace for request tracing

## Auto Scaling

Cloud Run automatically scales based on incoming requests:
- **Min Instances:** 1 (always running)
- **Max Instances:** 10 (cost control)
- **Concurrency:** 80 requests per instance
