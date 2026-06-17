# Project Scope

## In Scope

### Core Detection System
- Video frame extraction at configurable sampling rates
- Face detection and cropping using RetinaFace/MTCNN
- Image normalization and preprocessing
- XceptionNet model implementation and training
- EfficientNet-B0 model implementation and training
- Transfer learning from ImageNet pre-trained weights
- Binary classification (real vs. fake)

### Evaluation
- Frame-level classification metrics
- Video-level aggregation (mean, majority vote, confidence weighted)
- Confusion matrix generation
- ROC curve and AUC computation
- Cross-dataset validation on Celeb-DF
- GradCAM explainability analysis

### Deployment
- FastAPI inference service with REST endpoints
- Model loading and management
- Video and image prediction endpoints
- Health check and model info endpoints
- Docker containerization

### Documentation
- Architecture documentation
- API documentation
- Experiment logging
- Thesis chapter drafts
- Supervisor review checklist

### Datasets
- FaceForensics++ (c23 compression)
- Celeb-DF v2 (cross-dataset validation)

## Out of Scope

### Audio Analysis
- Audio deepfake detection
- Voice cloning detection
- Audio-visual synchronization analysis

### Advanced Manipulation Types
- Full-body manipulation detection
- Text-to-video generation detection
- Voice conversion detection

### Production Deployment
- Real-time video stream processing
- Social media platform integration
- User-facing application UI/ML pipeline monitoring
- Adversarial robustness testing
- Federated learning across platforms

### Advanced Architectures
- Transformer-based detection models
- Graph neural networks
- Vision-language models

### Scale
- Distributed training across multiple GPUs
- Enterprise-grade infrastructure
- Real-time processing at social media scale
