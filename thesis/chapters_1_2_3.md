# DETECTION OF SOCIAL MEDIA DEEPFAKE CONTENTS USING DEEP LEARNING ALGORITHM

**Author:** Olamijulo Israel D  
**Matric Number:** CYS/22/9071  
**Department:** Cyber Security, School of Computing  
**Institution:** Federal University of Technology Akure (FUTA)  
**Degree:** Bachelor of Technology (B.Tech) in Cyber Security  
**Date:** February 2026

---

# CHAPTER ONE: INTRODUCTION

## 1.1 Introduction and Background

As technology grows and Artificial Intelligence advances, there has been widespread creation of deepfake media, which poses serious threats to digital trust, cybersecurity, and financial security on social media platforms. This project proposes the design and implementation of a Deep Learning-based system for detecting deepfake videos on social media. The system analyzes facial features extracted from video frames and classifies them as real or fake using Deep Learning techniques.

Publicly available datasets (FaceForensics++ and Celeb-DF) are used for training and evaluation. The study aims to demonstrate the effectiveness of deep learning in combating deepfake-based threats and to contribute practical solutions to the field of multimedia forensics and cybersecurity.

Deepfake technology involves the use of artificial intelligence techniques, particularly deep learning, to generate highly realistic but fake images, videos, and audio recordings that imitate real individuals (Westerlund, 2019). While deepfakes have positive applications in entertainment and education, their misuse has raised serious concerns in areas such as privacy, security, and online trust. The rapid spread of manipulated media on social media platforms has made deepfake detection an important research problem in cybersecurity and multimedia forensics.

## 1.2 Background of the Study

In recent years, social media has become one of the most effective platforms for sharing multimedia content, and also making it a major channel for spreading misinformation and manipulated media. Deepfakes on social media have been used for impersonation scams, political misinformation (Ternovski, Kalla & Aronow, 2021), reputational damage, and financial fraud. These attacks mislead users, manipulate public opinion, and cause severe economic and social harm.

As technology evolves, traditional digital forensic methods are no longer sufficient to detect modern deepfakes, as current generation techniques produce highly realistic results. Deep learning has shown strong performance in image and video analysis and provides a promising solution for automated deepfake detection. This project focuses on applying deep learning techniques to detect facial deepfake videos under realistic social media conditions.

## 1.3 Motivation

The motivation for this research arises from the growing threat that deepfake media poses to digital trust, cybersecurity, and socio-economic stability within modern social media ecosystems, particularly in developing digital environments such as Nigeria.

Social media platforms have become primary sources of news, communication, and digital identity. However, the increasing sophistication of deepfake videos has severely undermined users' ability to distinguish authentic content from manipulated media. Deepfakes have been used to spread misinformation, impersonate public figures, and manipulate public opinion, leading to widespread distrust in digital content. This research is motivated by the urgent need to restore confidence in online multimedia by developing reliable automated detection mechanisms.

> "The main harm caused by deepfakes is not fooling the public, it is the uncertainty it causes, leading the public to doubt authentic news sources." (Vaccari & Chadwick et al., 2020)

Deepfake technology is increasingly weaponized for impersonation scams, social engineering attacks, and financial fraud. Attackers exploit realistic facial and voice manipulation to deceive victims into transferring funds or revealing sensitive information. Existing security mechanisms on social media platforms are largely reactive and insufficient against these evolving threats. This project is motivated by the need to proactively detect deepfake videos before they can be exploited for malicious purposes.

Manual content moderation and traditional digital forensic techniques are no longer effective against modern deepfakes, which are visually indistinguishable from real videos. Human observers perform poorly when identifying high-quality deepfake media, especially after video compression by social media platforms. This research is motivated by the need to leverage deep learning techniques that can identify subtle spatial and temporal inconsistencies beyond human perception.

Most existing deepfake detection models are trained and evaluated under controlled conditions, yet social media platforms apply compression, resizing, and re-encoding that significantly degrade video quality. These transformations reduce the effectiveness of many detection systems. This project is motivated by the need to design and evaluate a detection system that performs reliably under realistic social media conditions.

## 1.4 Aim and Objectives

**Aim:**  
The aim of this project is to design a deep learning-based system capable of detecting deepfake videos on social media platforms.

**Objectives:**

1. Design a deep learning-based detection model using convolutional neural network architectures.
2. Implement a system capable of frame-level and video-level classification of deepfake content.
3. Evaluate the designed system using standard metrics: accuracy, precision, recall, F1-score, and ROC-AUC.

## 1.4.1 Implementation

The implementation phase involves extracting video frames at fixed intervals, applying face detection algorithms to isolate facial regions, and training the deep learning model using labeled datasets. Python is used as the primary programming language, while PyTorch is employed for model development. OpenCV is used for video processing tasks, and Google Colab / local GPU provides the computational environment for training and experimentation.

## 1.5 Methodology

This project adopts an experimental research methodology with a structured deep learning development pipeline. The key stages include dataset collection, data preprocessing, model training, and performance evaluation.

### Dataset Collection and Preparation

Publicly available datasets such as FaceForensics++ and Celeb-DF are used for this project. These datasets contain both authentic and manipulated videos generated using multiple deepfake techniques.

**FaceForensics++ Dataset:**  
The FaceForensics++ dataset is a backbone of digital forensics, specifically designed to train and benchmark deep learning models like XceptionNet for deepfake detection. It was developed by researchers at the Technical University of Munich (TUM) and is the successor to the original 2018 FaceForensics dataset. FaceForensics++ provides data in three different quality levels:

- **Raw (c0):** Less loss; best for seeing tiny pixel-level inconsistencies.
- **High Quality (c23):** Lightly compressed; similar to high-quality web video.
- **Low Quality (c40):** Heavily compressed; mimics degradation seen on platforms like WhatsApp or Facebook.

**Celeb-DF Dataset:**  
Celeb-DF (v2) is the "Extreme" successor designed to address the flaws of earlier datasets. Released in 2020 by researchers from the University at Albany and the University of Chinese Academy of Sciences, it is specifically built to be significantly harder for models to solve. The dataset uses YouTube interviews of unique celebrities in real-world settings with varying lighting, complex backgrounds, and diverse head movements. An improved synthesis pipeline reduces color mismatch, temporal smoothing, and refinement, making it more challenging for detection models.

### Data Preprocessing

Video frames are extracted at fixed intervals from each video. Face detection algorithms are applied to locate and crop facial regions. The cropped faces are resized and normalized to ensure consistency across the dataset.

### Model Training

A pretrained convolutional neural network is fine-tuned using the processed dataset. Binary cross-entropy loss and the Adam optimization algorithm are used to train the model for real-versus-fake classification.

### Design

The system is designed around a deep learning architecture suitable for visual feature extraction and classification. Pre-trained convolutional neural networks (XceptionNet and EfficientNet) are adopted using transfer learning to improve performance and reduce training time. The design includes modules for video frame extraction, face detection and cropping, feature learning, and final classification into real or fake categories.

**XceptionNet:** Uses Depthwise Separable Convolution to separate spatial and channel feature learning, enabling faster training with improved precision and accuracy.

**EfficientNet:** Uses a Compound Coefficient to scale the network's depth, width, and resolution uniformly, ensuring balanced growth and optimized performance.

## 1.6 Mathematical Foundations Overview

This project employs several mathematical formulations that underpin the deep learning models and evaluation procedures used for deepfake detection. A brief overview of these mathematical foundations is provided below; detailed formulations are presented in Chapter 3.

### Loss Function

The models are trained using Label Smoothing Cross-Entropy Loss, which prevents overconfident predictions by softening hard labels. For a given sample with true label $y$ and predicted probabilities $\hat{y}_k$ over $K$ classes, the loss is defined as:

$$\mathcal{L} = -\sum_{k=1}^{K} y_k^{smooth} \log(\hat{y}_k) \quad \text{...(1.1)}$$

where $y_k^{smooth} = (1 - \epsilon) \cdot y_k + \epsilon / K$ and $\epsilon$ is the smoothing factor (set to 0.1 in this project).

### Evaluation Metrics

Five standard metrics are used to evaluate detection performance:

- **Accuracy** measures overall classification correctness: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$ ...(1.2)
- **Precision** measures the proportion of true positives among predicted positives: $\text{Precision} = \frac{TP}{TP + FP}$ ...(1.3)
- **Recall** measures the proportion of true positives among actual positives: $\text{Recall} = \frac{TP}{TP + FN}$ ...(1.4)
- **F1-Score** is the harmonic mean of precision and recall: $\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ ...(1.5)
- **ROC-AUC** measures the area under the Receiver Operating Characteristic curve, which plots the True Positive Rate against the False Positive Rate at various classification thresholds ...(1.6)

### Optimization

Model parameters are updated using the Adam optimizer, which adapts learning rates for each parameter based on first and second moment estimates of the gradients:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{...(1.7)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{...(1.8)}$$
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \quad \text{...(1.9)}$$

where $g_t$ is the gradient, $m_t$ and $v_t$ are the first and second moment estimates, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\eta$ is the learning rate.

### Transfer Learning

Transfer learning leverages pre-trained weights $\theta_{pre}$ from a source domain (ImageNet) and fine-tunes them on the target domain (deepfake detection). The fine-tuning process updates a subset of parameters:

$$\theta_{target} = \theta_{pre} + \Delta\theta \quad \text{...(1.10)}$$

where $\Delta\theta$ is learned from the target dataset using backpropagation.

These mathematical foundations are essential for understanding the model training and evaluation methodology described in detail in Chapter 3.

## 1.7 Expected Contribution to Knowledge

This research is expected to contribute to the body of knowledge in social media forensics and cybersecurity by:

1. Providing a comprehensive evaluation of CNN architectures for deepfake detection under realistic social media conditions.
2. Demonstrating the effectiveness of transfer learning for deepfake detection with limited labeled data.
3. Producing a reproducible, well-documented system that bridges the gap between academic research and practical deployment.
4. Contributing benchmark results on standardized datasets (FaceForensics++ and Celeb-DF) for future research comparison.

---

# CHAPTER TWO: LITERATURE REVIEW

## 2.1 Conceptual Clarifications

### Deepfakes and Synthetic Media

Deepfakes refer to synthetically generated or manipulated media in which the likeness of a real individual is convincingly altered or fabricated using artificial intelligence techniques, particularly deep learning (Verdoliva, 2020). The term is most commonly associated with face-swapped or face-manipulated videos produced using Generative Adversarial Networks (GANs) and encoder-decoder architectures (Tolosana et al., 2020). Unlike traditional video forgeries, deepfakes exhibit high perceptual realism, making them difficult to detect through visual inspection alone.

Synthetic media is a broader concept encompassing deepfake videos, manipulated images, AI-generated voices, and text-based impersonation. In the context of social media, deepfake videos are particularly dangerous due to their ability to convincingly portray public figures or private individuals saying or doing things that never occurred, thereby enabling misinformation, fraud, and reputational harm (Agarwal et al., 2019).

### Social Media as a Distribution Environment

Social media platforms such as Facebook, Instagram, TikTok, and X have become primary channels for multimedia dissemination. These platforms apply automatic video compression, resizing, and re-encoding to optimize bandwidth and storage (Zhao et al., 2023). While these processes enhance scalability, they also alter video artifacts and suppress subtle forensic traces used by detection algorithms. Consequently, deepfake detection on social media must account for degraded video quality and heterogeneous content pipelines (Rossler et al., 2019).

### Deep Learning and Computer Vision

Deep learning is a subset of machine learning that employs multi-layer neural networks to learn hierarchical feature representations from data. In computer vision, convolutional neural networks (CNNs) have demonstrated exceptional performance in tasks such as image classification, face recognition, and video analysis (LeCun et al., 2015). Deepfake detection leverages CNNs to identify spatial inconsistencies, texture artifacts, and temporal irregularities introduced during synthetic media generation (Gera & Delp, 2018).

### Multimedia Forensics

Multimedia forensics is the scientific discipline concerned with the analysis of digital images, videos, and audio to verify authenticity and detect manipulation (Verdoliva, 2020). Unlike cryptographic watermarking, forensic approaches are passive and do not require prior embedding of security markers. Deepfake detection is now regarded as a critical subfield of multimedia forensics due to the scale and sophistication of AI-generated media.

## 2.2 Evolution of Deepfake Technology

### Early Digital Face Manipulation

Early face manipulation techniques relied on manual editing and traditional computer graphics, requiring significant expertise and time. These methods produced visible artifacts that could be detected using rule-based forensic techniques (Farid, 2009).

### Emergence of Deep Learning-Based Generation

The introduction of GANs by Goodfellow et al. (2014) marked a turning point in synthetic media generation. GAN-based deepfakes enabled automated learning of facial expressions, lighting, and motion, resulting in highly realistic videos. Encoder-decoder architectures further simplified the creation of face-swapped videos, accelerating the spread of deepfakes on online platforms (Rossler et al., 2019).

### Current Generation Deepfakes

Recent deepfake models integrate attention mechanisms, temporal coherence constraints, and high-resolution synthesis, making them increasingly robust against traditional forensic detection (Tolosana et al., 2020). These advances necessitate equally sophisticated detection mechanisms that can adapt to evolving generation techniques.

## 2.3 Deepfake Detection Approaches

### Traditional Forensic Techniques

Early detection methods focused on hand-crafted features such as eye blinking frequency, head pose inconsistencies, and color mismatches (Li et al., 2018). While effective against early deepfakes, these methods fail against modern generation techniques that produce temporally coherent and visually convincing results.

### Deep Learning-Based Detection

Deep learning approaches have emerged as the most promising direction for deepfake detection. Key architectures include:

- **XceptionNet:** Achieved 99.53% accuracy on FaceForensics++ (Rossler et al., 2019) using depthwise separable convolutions.
- **EfficientNet:** Provides balanced scaling of depth, width, and resolution for efficient feature extraction (Tan & Le, 2019).
- **Capsule Networks:** Capture spatial relationships between facial features for improved detection (Nguyen et al., 2019).
- **Recurrent Neural Networks:** Model temporal inconsistencies across video frames (Gu et al., 2019).

### Hybrid Approaches

Recent work combines spatial and temporal analysis, using CNNs for frame-level features and RNNs or transformers for temporal modeling. These hybrid approaches achieve state-of-the-art performance but increase computational complexity.

## 2.4 Transfer Learning in Deepfake Detection

Transfer learning has become a standard approach in deepfake detection, enabling models trained on large image datasets (e.g., ImageNet) to be fine-tuned for forensic analysis. Pre-trained models provide robust feature extractors that can be adapted with relatively small deepfake datasets, reducing training time and improving generalization (Rossler et al., 2019).

## 2.5 Research Gap

Despite significant progress, several gaps remain:

1. Most detection models are evaluated under controlled conditions, not realistic social media scenarios.
2. Cross-dataset generalization remains poor, with models failing when tested on unseen manipulation types.
3. The impact of social media compression on detection accuracy is underexplored.
4. Practical, deployable systems that bridge academic research and real-world application are lacking.

---

# CHAPTER THREE: RESEARCH METHODOLOGY

## 3.1 Research Design

This project adopts an experimental research design with a structured deep learning development pipeline. The methodology follows six phases:

1. **Data Collection:** Acquiring FaceForensics++ and Celeb-DF datasets.
2. **Preprocessing:** Frame extraction, face detection, cropping, and normalization.
3. **Model Development:** Implementing XceptionNet and EfficientNet using transfer learning.
4. **Training:** Fine-tuning models with binary cross-entropy loss and Adam optimizer.
5. **Evaluation:** Measuring performance using accuracy, precision, recall, F1-score, and ROC-AUC.
6. **Deployment:** Exposing trained models via a FastAPI inference service.

## 3.2 Datasets

### FaceForensics++

- **Source:** Technical University of Munich (TUM)
- **Size:** ~1,000 real + ~4,000 manipulated videos (4 manipulation methods)
- **Manipulation Methods:** Deepfakes, Face2Face, FaceSwap, NeuralTextures
- **Quality Levels:** c0 (raw), c23 (high quality), c40 (low quality)
- **Use in This Project:** c23 compression level for realistic social media conditions

### Celeb-DF (v2)

- **Source:** University at Albany + Chinese Academy of Sciences
- **Size:** 590 real videos + 5,639 synthetic videos
- **Characteristics:** Celebrity interviews, improved synthesis pipeline, reduced artifacts
- **Use in This Project:** Validation and cross-dataset evaluation

## 3.3 Preprocessing Pipeline

1. **Frame Extraction:** Extract frames at 1 FPS using OpenCV.
2. **Face Detection:** Apply RetinaFace/MTCNN to locate faces.
3. **Face Cropping:** Crop detected face regions with padding.
4. **Normalization:** Resize to model input size (299x299 for XceptionNet, 224x224 for EfficientNet), normalize to [0, 1] or ImageNet statistics.

## 3.4 Model Architecture

### XceptionNet (Primary)

- **Backbone:** Xception with depthwise separable convolutions
- **Transfer Learning:** Pre-trained on ImageNet, fine-tuned on FaceForensics++
- **Classification Head:** Global Average Pooling → Dropout(0.5) → Linear(2048, 2)
- **Parameters:** 17,028,962 (17.03M)
- **Model Size:** 64.96 MB
- **Input Resolution:** 299×299

### EfficientNet-B0 (Secondary)

- **Backbone:** EfficientNet-B0 with compound scaling
- **Transfer Learning:** Pre-trained on ImageNet, fine-tuned on FaceForensics++
- **Classification Head:** Global Average Pooling → Dropout(0.5) → Linear(1280, 2)
- **Parameters:** 4,010,110 (4.01M)
- **Model Size:** 15.3 MB
- **Input Resolution:** 224×224

## 3.5 Training Configuration

**Table 3.1: Training Hyperparameters**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Batch Size | 64 |
| Max Epochs | 30 |
| Early Stopping Patience | 7 |
| Label Smoothing | 0.1 |
| Loss Function | Label Smoothing Cross-Entropy |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Random Seed | 42 |
| Mixed Precision | AMP (Automatic Mixed Precision) |
| cuDNN Benchmark | Enabled |
| Persistent Workers | Enabled |

### 3.5.1 Loss Function

The models are trained using Label Smoothing Cross-Entropy Loss. For a classification problem with $K$ classes, given the true one-hot label $y$ and the model's predicted softmax output $\hat{y}$, the standard cross-entropy loss is:

$$\mathcal{L}_{CE} = -\sum_{k=1}^{K} y_k \log(\hat{y}_k) \quad \text{...(3.1)}$$

Label smoothing modifies the target distribution to prevent the model from becoming overconfident. The smoothed labels are computed as:

$$y_k^{smooth} = (1 - \epsilon) \cdot y_k + \frac{\epsilon}{K} \quad \text{...(3.2)}$$

where $\epsilon$ is the smoothing factor (set to 0.1). The resulting loss becomes:

$$\mathcal{L}_{LS} = -\sum_{k=1}^{K} y_k^{smooth} \log(\hat{y}_k) = -(1 - \epsilon) \log(\hat{y}_y) - \frac{\epsilon}{K} \sum_{k=1}^{K} \log(\hat{y}_k) \quad \text{...(3.3)}$$

For binary classification ($K = 2$), this simplifies to:

$$\mathcal{L}_{LS} = -(1 - \epsilon) \log(\hat{y}_y) - \frac{\epsilon}{2} (\log(\hat{y}_0) + \log(\hat{y}_1)) \quad \text{...(3.4)}$$

### 3.5.2 Optimizer: Adam

The Adam optimizer adapts the learning rate for each parameter individually. For parameter $\theta_t$ at timestep $t$, the update rules are:

**First moment estimate (mean of gradients):**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{...(3.5)}$$

**Second moment estimate (uncentered variance of gradients):**

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{...(3.6)}$$

**Bias-corrected estimates:**

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{...(3.7)}$$

**Parameter update:**

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon_{adam}} \quad \text{...(3.8)}$$

where $g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$ is the gradient, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta$ is the learning rate, and $\epsilon_{adam} = 10^{-8}$.

### 3.5.3 Learning Rate Schedule

The ReduceLROnPlateau scheduler reduces the learning rate when the validation loss plateaus:

$$\eta_t = \eta_{t-1} \times \gamma \quad \text{...(3.9)}$$

where $\gamma = 0.5$ (reduction factor) and the reduction is triggered when the validation loss has not improved for 5 epochs (patience = 5).

## 3.6 Evaluation Metrics

Five standard metrics are used to evaluate detection performance. Given a binary classification problem where positive class = fake and negative class = real, the confusion matrix components are:

- **True Positives (TP):** Fake videos correctly classified as fake
- **True Negatives (TN):** Real videos correctly classified as real
- **False Positives (FP):** Real videos incorrectly classified as fake
- **False Negatives (FN):** Fake videos incorrectly classified as real

### 3.6.1 Accuracy

Accuracy measures the overall proportion of correct predictions:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \quad \text{...(3.10)}$$

### 3.6.2 Precision

Precision measures the proportion of predicted positives that are actually positive:

$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{...(3.11)}$$

### 3.6.3 Recall (Sensitivity)

Recall measures the proportion of actual positives that are correctly identified:

$$\text{Recall} = \frac{TP}{TP + FN} \quad \text{...(3.12)}$$

### 3.6.4 F1-Score

The F1-Score is the harmonic mean of precision and recall, providing a single metric that balances both:

$$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN} \quad \text{...(3.13)}$$

### 3.6.5 ROC-AUC

The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various classification thresholds:

$$\text{TPR} = \frac{TP}{TP + FN} = \text{Recall} \quad \text{...(3.14)}$$

$$\text{FPR} = \frac{FP}{FP + TN} \quad \text{...(3.15)}$$

The Area Under the ROC Curve (AUC) quantifies the model's ability to distinguish between classes:

$$\text{AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(t)) \, dt \quad \text{...(3.16)}$$

An AUC of 1.0 represents a perfect classifier, while an AUC of 0.5 represents random guessing.

## 3.7 Video-Level Aggregation

Frame-level predictions are aggregated to video-level using three methods:

1. **Mean Probability:** Average of all frame predictions
2. **Majority Voting:** Most frequent frame prediction
3. **Confidence Weighting:** Weighted average by prediction confidence

## 3.8 Tools and Technologies

| Tool | Purpose |
|------|---------|
| Python 3.11+ | Primary programming language |
| PyTorch | Deep learning framework |
| OpenCV | Video processing and face detection |
| TorchVision | Pre-trained models and transforms |
| TorchMetrics | Standardized metric computation |
| FastAPI | REST API for inference |
| Matplotlib/Seaborn | Visualization |
| GradCAM | Model explainability |

## 3.9 Ethical Considerations

- All datasets used are publicly available and approved for research purposes.
- No real individuals are targeted or harmed by this research.
- The system is designed for defensive purposes (detection, not generation).
- Results are reported honestly without fabrication or exaggeration.
