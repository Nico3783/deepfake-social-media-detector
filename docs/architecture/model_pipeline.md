# Model Pipeline

## Overview

The model pipeline handles architecture definition, weight loading, and model management.

## Supported Architectures

### XceptionNet (Primary)

**Architecture:**
```
Input (B, 3, 299, 299)
    ↓
Entry Flow
├── Conv2d(3, 32, 3, stride=2, padding=1)
├── BatchNorm2d(32)
├── ReLU
├── Conv2d(32, 64, 3, padding=1)
├── BatchNorm2d(64)
├── ReLU
├── SeparableConv2d(64, 128)
├── SeparableConv2d(128, 128)
├── MaxPool2d(3, stride=2, padding=1)
├── SeparableConv2d(128, 256)
├── SeparableConv2d(256, 256)
├── MaxPool2d(3, stride=2, padding=1)
    ↓
Middle Flow (8x)
├── ReLU → SeparableConv2d → BatchNorm
├── ReLU → SeparableConv2d → BatchNorm
├── ReLU → SeparableConv2d → BatchNorm
├── Add (residual connection)
    ↓
Exit Flow
├── SeparableConv2d(728, 1024)
├── MaxPool2d(3, stride=2, padding=1)
├── SeparableConv2d(1024, 1536)
├── SeparableConv2d(1536, 2048)
├── Global Average Pooling
    ↓
Classification Head
├── Dropout(0.5)
├── Linear(2048, 2)
```

**Key Features:**
- Depthwise separable convolutions
- Residual connections in middle flow
- 22.9M parameters
- Input: 299x299

### EfficientNet-B0 (Secondary)

**Architecture:**
```
Input (B, 3, 224, 224)
    ↓
Stem
├── Conv2d(3, 32, 3, stride=2, padding=1)
├── BatchNorm2d(32)
├── Swish
    ↓
Stage 1: MBConv1 (32→16, 3x3)
    ↓
Stage 2: MBConv6 (16→24, 3x3) x 2
    ↓
Stage 3: MBConv6 (24→40, 5x5) x 2
    ↓
Stage 4: MBConv6 (40→80, 3x3) x 3
    ↓
Stage 5: MBConv6 (80→112, 5x5) x 3
    ↓
Stage 6: MBConv6 (112→192, 5x5) x 4
    ↓
Stage 7: MBConv6 (192→320, 3x3) x 1
    ↓
Head
├── Conv2d(320, 1280, 1)
├── BatchNorm2d(1280)
├── Swish
├── Global Average Pooling
    ↓
Classification Head
├── Dropout(0.5)
├── Linear(1280, 2)
```

**Key Features:**
- MBConv blocks with Squeeze-Excitation
- Compound scaling (depth, width, resolution)
- 5.3M parameters
- Input: 224x224

## Model Factory

```python
# src/models/model_factory.py

def create_model(model_name, num_classes=2, pretrained=True):
    """
    Create a model instance.
    
    Args:
        model_name: 'xceptionnet' or 'efficientnet'
        num_classes: Number of output classes
        pretrained: Use ImageNet pre-trained weights
    
    Returns:
        Model instance with classification head
    """
```

## Transfer Learning Strategy

1. **Load Pre-trained Weights:** ImageNet initialization
2. **Replace Classification Head:** New linear layer for binary classification
3. **Fine-tune:** Train all layers with low learning rate
4. **Save:** Full model checkpoint with optimizer state

## Model Management

```
outputs/models/
├── xceptionnet_best.pth      # Best XceptionNet checkpoint
├── efficientnet_best.pth     # Best EfficientNet checkpoint
├── model_info.json           # Model metadata
└── training_history.json     # Training metrics
```
