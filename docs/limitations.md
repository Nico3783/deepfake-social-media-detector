# Limitations

## Technical Limitations

### 1. Dataset Scope
**Limitation:** The system is trained and evaluated on only two datasets: FaceForensics++ and Celeb-DF v2.

**Impact:** Performance on other datasets or real-world social media content may differ. Novel manipulation techniques not represented in these datasets may evade detection.

**Mitigation:** Cross-dataset validation provides some measure of generalization assessment.

### 2. Manipulation Types
**Limitation:** The system focuses exclusively on facial manipulation techniques (face swap, face reenactment).

**Impact:** Audio deepfakes, full-body manipulation, text-to-video generation, and other synthetic media types are not detected.

**Mitigation:** The modular architecture allows for future extension to additional manipulation types.

### 3. Temporal Analysis
**Limitation:** Frames are classified independently without modeling temporal dependencies across video sequences.

**Impact:** Subtle temporal inconsistencies (e.g., flickering, motion artifacts) may be missed.

**Mitigation:** Video-level aggregation partially addresses this by combining frame predictions.

### 4. Adversarial Robustness
**Limitation:** The models have not been evaluated against adversarial attacks designed to evade detection.

**Impact:** Sophisticated attackers could potentially craft perturbations that fool the detection system.

**Mitigation:** This is identified as an important direction for future work.

### 5. Social Media Platform Variability
**Limitation:** The system is trained on c23 compression, which approximates but does not exactly replicate social media compression pipelines.

**Impact:** Different platforms apply different compression algorithms, potentially affecting detection accuracy.

**Mitigation:** c23 provides a reasonable approximation of real-world conditions.

## Research Limitations

### 6. Computational Resources
**Limitation:** Training is constrained to single-GPU environments.

**Impact:** Larger models, longer training schedules, and more extensive hyperparameter searches are limited by available compute.

**Mitigation:** Google Colab provides additional GPU resources for experimentation.

### 7. Evaluation Scope
**Limitation:** The system is not evaluated for real-time processing performance at social media scale.

**Impact:** Deployment characteristics (latency, throughput, resource usage) at scale are unknown.

**Mitigation:** Inference time measurements provide baseline performance estimates.

### 8. Ethical Constraints
**Limitation:** The system is designed for defensive detection only, not for generating or improving deepfakes.

**Impact:** Research is limited to detection capabilities; understanding generation techniques is secondary.

**Mitigation:** This constraint ensures the research contributes positively to cybersecurity.

## Known Issues

### 9. Cross-Dataset Generalization
Performance drops by 7-8% when models trained on FaceForensics++ are evaluated on Celeb-DF, indicating limited cross-dataset generalization.

### 10. Class Imbalance Handling
While label smoothing and class weights are used, the inherent class imbalance in real-world deepfake data may affect model calibration.
