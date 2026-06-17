# Training Pipeline

## Overview

The training pipeline handles model fine-tuning, loss computation, optimization, and checkpoint management.

## Pipeline Components

### 1. Data Loading

```python
# Dataset: DeepfakeFaceDataset
# DataLoader with configurable batch size, shuffle, num_workers
# Pin memory for GPU acceleration
```

- **Train DataLoader:** Shuffle=True, drop_last=True
- **Val DataLoader:** Shuffle=False, drop_last=False
- **Test DataLoader:** Shuffle=False, drop_last=False

### 2. Model Initialization

```python
# Load pre-trained backbone (ImageNet weights)
# Replace classification head with binary classifier
# Move to device (CUDA/CPU)
```

### 3. Loss Function

**Label Smoothing Cross-Entropy Loss:**
```python
LabelSmoothingLoss(smoothing=0.1)
```

- Prevents overconfident predictions
- Improves generalization
- Handles class imbalance through label softening

### 4. Optimizer

**Adam Optimizer:**
```python
Adam(lr=0.001, weight_decay=0.0001)
```

- Adaptive learning rates per parameter
- Weight decay for regularization
- Betas: (0.9, 0.999)

### 5. Learning Rate Scheduler

**ReduceLROnPlateau:**
```python
ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
```

- Monitors validation loss
- Reduces LR by factor when plateau detected
- Minimum LR: 1e-6

### 6. Training Loop

```python
for epoch in range(max_epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            metrics.update(outputs, labels)
    
    # Scheduler step
    scheduler.step(val_loss)
    
    # Early stopping check
    if val_loss < best_loss:
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

### 7. Callbacks

- **Early Stopping:** Monitor val_loss, patience=10
- **Model Checkpointing:** Save best model based on val_loss
- **Metrics Logging:** Record train/val metrics per epoch
- **LR Logging:** Track learning rate changes

### 8. Checkpoint Management

```
outputs/models/
├── best_model.pth        # Best validation loss
├── last_model.pth        # Latest epoch
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_20.pth
└── training_history.json
```

## Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | Adam | config/training.yaml |
| Learning Rate | 0.001 | config/training.yaml |
| Weight Decay | 0.0001 | config/training.yaml |
| Batch Size | 32 | config/training.yaml |
| Max Epochs | 50 | config/training.yaml |
| Early Stopping Patience | 10 | config/training.yaml |
| Label Smoothing | 0.1 | config/training.yaml |
| LR Scheduler Patience | 5 | config/training.yaml |
| LR Scheduler Factor | 0.5 | config/training.yaml |
| Random Seed | 42 | config/constants.yaml |

## Reproducibility

- Fixed random seed (42) for Python, NumPy, PyTorch
- Deterministic CUDA operations where possible
- All hyperparameters logged to experiment JSON
- Model checkpoints saved with optimizer state
