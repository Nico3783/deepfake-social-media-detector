# Assumptions

## Technical Assumptions

### 1. Hardware Availability
**Assumption:** A GPU with at least 8GB VRAM (e.g., NVIDIA RTX 3060 or better) is available for model training.

**Impact:** Training time estimates are based on this hardware specification. CPU-only training is possible but will be significantly slower (10-20x).

### 2. Dataset Access
**Assumption:** Both FaceForensics++ and Celeb-DF datasets can be obtained through official channels for research purposes.

**Impact:** Research depends on dataset availability. Fallback plan: use publicly available subsets or alternative datasets (e.g., DFDC preview).

### 3. Python Environment
**Assumption:** Python 3.11+ with PyTorch 2.0+ and CUDA 11.8+ is available.

**Impact:** Dependencies are documented in requirements.txt. Virtual environment setup is required.

### 4. Storage Capacity
**Assumption:** At least 100GB of free disk space is available for raw datasets, processed frames, and model checkpoints.

**Impact:** FaceForensics++ raw data is ~1TB; c23 subset is ~100GB. Processed frames require ~50GB.

### 5. Internet Connectivity
**Assumption:** Internet access is available for downloading datasets, pre-trained weights, and dependencies.

**Impact:** Initial setup requires connectivity. Training and evaluation can proceed offline.

## Research Assumptions

### 6. Dataset Representativeness
**Assumption:** FaceForensics++ and Celeb-DF are representative of real-world deepfake content encountered on social media.

**Impact:** Detection performance on these datasets may not fully generalize to novel manipulation techniques.

### 7. Transfer Learning Effectiveness
**Assumption:** ImageNet pre-trained features are transferable to forensic analysis tasks.

**Impact:** This assumption is supported by prior work (Rossler et al., 2019) but may not hold for all architectures.

### 8. Compression Impact
**Assumption:** c23 compression level is representative of typical social media video quality.

**Impact:** Actual social media compression varies by platform and may differ from c23.

### 9. Binary Classification Sufficiency
**Assumption:** Binary (real vs. fake) classification is sufficient for the research objectives.

**Impact:** Multi-class classification (by manipulation type) is out of scope but could provide additional insights.

### 10. Frame Independence
**Assumption:** Individual frames can be classified independently for frame-level analysis.

**Impact:** Temporal dependencies between frames are not modeled in the current approach.
