#!/bin/bash
# Deployment script
# Usage: bash scripts/deploy.sh [local|docker|cloud]

set -e

echo "=========================================="
echo "Deepfake Detection - Deployment"
echo "=========================================="

ENV=${1:-local}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run scripts/setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Function to deploy locally
deploy_local() {
    echo ""
    echo "[Local] Starting API server..."
    echo "API will be available at: http://localhost:8000"
    echo "Press Ctrl+C to stop"
    echo ""
    
    uvicorn src.api.app:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --log-level info
}

# Function to deploy with Docker
deploy_docker() {
    echo ""
    echo "[Docker] Building and starting containers..."
    
    cd deployment
    
    # Build image
    echo "Building Docker image..."
    docker-compose build
    
    # Start services
    echo "Starting services..."
    docker-compose up -d
    
    echo ""
    echo "Services started!"
    echo "API: http://localhost:8000"
    echo "Nginx: http://localhost:80"
    echo ""
    echo "View logs: docker-compose logs -f"
    echo "Stop services: docker-compose down"
    
    cd ..
}

# Function to deploy to cloud
deploy_cloud() {
    echo ""
    echo "[Cloud] Deploying to cloud..."
    echo ""
    echo "Please refer to the deployment guides:"
    echo "  - AWS: deployment/cloud/aws.md"
    echo "  - GCP: deployment/cloud/gcp.md"
    echo ""
    echo "Or use the following commands:"
    echo ""
    echo "AWS ECS:"
    echo "  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com"
    echo "  docker build -t deepfake-detector -f deployment/Dockerfile ."
    echo "  docker tag deepfake-detector:latest <account>.dkr.ecr.us-east-1.amazonaws.com/deepfake-detector:latest"
    echo "  docker push <account>.dkr.ecr.us-east-1.amazonaws.com/deepfake-detector:latest"
    echo ""
    echo "GCP Cloud Run:"
    echo "  gcloud run deploy deepfake-detector --image gcr.io/$(gcloud config get-value project)/deepfake-detector --platform managed --region us-central1"
}

# Deploy based on argument
case $ENV in
    local)
        deploy_local
        ;;
    docker)
        deploy_docker
        ;;
    cloud)
        deploy_cloud
        ;;
    *)
        echo "Error: Unknown environment '$ENV'"
        echo "Usage: $0 [local|docker|cloud]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo ""
