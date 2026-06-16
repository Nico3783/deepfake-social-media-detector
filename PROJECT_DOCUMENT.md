 
A PROJECT REPORT ON
DETECTION OF SOCIAL MEDIA DEEPFAKES CONTENTS USING DEEP LEARNING ALGORITHM.
BY
OLAMIJULO ISRAEL D
CYS/22/9071

SUBMITTED TO
THE DEPARTMENT OF CYBER SECURITY,
SCHOOL OF COMPUTING
FEDERAL UNIVERSITY OF TECHNOLOGY AKURE,
ONDO STATE, NIGERIA.

IN PARTIAL FULFILMENT OF THE REQUIREMENT FOR THE AWARD OF
BACHELOR OF TECHNOLOGY (B. TECH) DEGREE IN CYBER SECURITY









FEBRUARY, 2026

CHAPTER ONE
1.1 INTRODUCTION AND BACKGROUND
As technology grows, and Artificial Intelligence advances, there's been a widespread creation of deepfake media, which poses serious threats to digital trust, cybersecurity, and financial security on social media platforms.`
This project proposes the design and implementation of a Deep Learning based system for detecting deepfake videos on social media. The system will analyze facial features extracted from video frames online, and classify them as real or fake using Deep Learning techniques.
Publicly available datasets will be used for training and evaluation.
The study aims to demonstrate the effectiveness of deep learning in combating deepfake based threats and to contribute practical solutions to the field of multimedia forensics and cybersecurity.
Deepfake technology involves the use of artificial intelligence techniques, particularly deep learning, to generate highly realistic but fake images, videos, and audio recordings that imitate real individuals that maybe misleading (Westerlund, 2019.). While deepfakes have positive applications in entertainment and education, their misuse has raised serious concerns in areas such as privacy, security, and online trust. The rapid spread of manipulated media on social media platforms has made deepfake detection an important research problem in cybersecurity and multimedia forensics.

1.2 BACKGROUND OF THE STUDY
In recent years, Social media has become one of the effective platforms for sharing multimedia contents, and also making it a major channel for spreading misinformation and manipulated media. 
Deepfakes on social media have been used for impersonation scams, political misinformation (J. Ternovski, J. Kalla, and P. M. Aronow, 2021.), reputational damage, and financial fraud. These attacks mislead users, manipulate public opinion, and cause severe economic and social harm.
As technology evolves, traditional digital forensic methods are no longer sufficient to detect modern deepfakes, as current generation techniques produce highly realistic results. Deep learning has shown strong performance in image and video analysis and provides a promising solution for automated deepfake detection. This project focuses on applying deep learning techniques to detect facial deepfake videos under realistic social media conditions.


1.3 MOTIVATION
The motivation for this research arises from the growing threat that deepfake media poses to digital trust, cybersecurity, and socio-economic stability within modern social media ecosystems, particularly in developing digital environments such as Nigeria.
Social media platforms have become primary sources of news, communication, and digital identity. However, the increasing sophistication of deepfake videos has severely undermined users' ability to distinguish authentic content from manipulated media. Deepfakes have been used to spread misinformation, impersonate public figures, and manipulate public opinion, leading to widespread distrust in digital content. This research is motivated by the urgent need to restore confidence in online multimedia by developing reliable automated detection mechanisms.
The main harm caused by�deepfakes�is not fooling the�public, it is the uncertainty it causes, leading the�public�to doubt authentic news sources. (Vaccari & Chadwick et al, 2020)

Deepfake technology is increasingly weaponized for impersonation scams, social engineering attacks, and financial fraud. Attackers exploit realistic facial and voice manipulation to deceive victims into transferring funds or revealing sensitive information. Existing security mechanisms on social media platforms are largely reactive and insufficient against these evolving threats. This project is motivated by the need to proactively detect deepfake videos before they can be exploited for malicious purposes.
Manual content moderation and traditional digital forensic techniques are no longer effective against modern deepfakes, which are visually indistinguishable from real videos. As highlighted in recent studies, human observers perform poorly when identifying high-quality deepfake media, especially after video compression by social media platforms. This research is motivated by the need to leverage deep learning techniques that can identify subtle spatial and temporal inconsistencies beyond human perception.
Most existing deepfake detection models are trained and evaluated under controlled conditions, yet social media platforms apply compression, resizing, and re-encoding that significantly degrade video quality. These transformations reduce the effectiveness of many detection systems. This project is motivated by the need to design and evaluate a detection system that performs reliably under realistic social media conditions.
Despite growing global research on deepfake detection, there remains a need for practical, well-evaluated systems that bridge the gap between academic research and real-world deployment. This study is motivated by the desire to contribute a technically sound and reproducible deep learning-based approach to the field of multimedia forensics and cybersecurity.




1.4 AIM AND OBJECTIVES
Aim
The aim of this project is to design, a deep learning-based system capable of detecting deepfake videos on social media platforms.
Objectives
	i.  Design a deep learning-based detection model using convolutional neural network architectures.
	ii. Implement a system capable of frame-level and video-level classification of deepfake content.
	iii. Evaluate the designed system using standard metrics like accuracy, precision, recall, F1-score, and ROC-AUC

1.4.1 Implementation
The implementation phase involves extracting video frames at fixed intervals, applying face detection algorithms to isolate facial regions, and training the deep learning model using labeled datasets. Python is used as the primary programming language, while PyTorch or TensorFlow is employed for model development. OpenCV is used for video processing tasks, and Google Colab provides the computational environment for training and experimentation.











1.5 METHODOLOGY
This project will adopt an experimental research methodology with a structured deep learning development pipeline. The key stages include dataset collection, data preprocessing, model training, and performance evaluation.
Dataset Collection and Preparation
Publicly available datasets such as FaceForensics++ and Celeb-DF will be used for this project. These datasets contain both authentic and manipulated videos generated using multiple deepfake techniques. 
FaceForensics++ Dataset
The FaceForensics++  dataset is a backbone of digital forensics, specifically designed to train and benchmark deep learning models like XceptionNet for deepfake detection.
It was developed by researchers at the Technical University of Munich (TUM) and is the successor to the original 2018 FaceForensics dataset. 
FaceForensics++ is considered a large-scale dataset, specifically designed to provide enough data for deep neural networks to generalize. Because the raw dataset is massive (terabytes), most researchers download the c23 (compressed) version to save bandwidth, which is what led to XceptionNet becoming the benchmark leader-it proved exceptionally good at finding patterns even in compressed data.
What makes FF++ unique is that it provides the data in three different qualities. This is crucial because real-world deepfakes are often compressed when uploaded to social media, which hides the digital "artifacts" detectors look for.
    - Raw (c0): less loss; best for seeing tiny pixel-level inconsistencies.
    - High Quality (c23): Lightly compressed; similar to a high-quality web video.
    - Low Quality (c40): Heavily compressed; mimics the degradation seen on platforms like WhatsApp or Facebook.

Celeb-DF Dataset
(specifically v2) is the "Extreme" successor designed to address the flaws of earlier datasets. Released in 2020 by researchers from the University at Albany and the University of Chinese Academy of Sciences, it is specifically built to be significantly harder for models like XceptionNet to solve. This dataset takes a more realistic approach by using youtube interviews of unique celebrities, real-world settings with varying lighting, complex backgrounds, and diverse head movements.
Celeb-DF is massive and highly focused on "DeepFake" (face-swapping) rather than general reenactment. 
The creators of this dataset noticed that FF++ dataset was easy to detect with an accuracy of 99% XceptionNet model. This led to making celeb-df database more difficult to detect using an improved synthesis pipeline that reduces color mismatch, temporal smoothing and refinement.
Data Preprocessing
Video frames will be extracted at fixed intervals from each video. Face detection algorithms are applied to locate and crop facial regions. The cropped faces are resized and normalized to ensure consistency across the dataset.
Model Training
A pretrained convolutional neural network will be fine-tuned using the processed dataset. Binary cross-entropy loss and the Adam optimization algorithm are used to train the model for real-versus-fake classification.

1.5.1 Design
The system will be designed around a deep learning architecture suitable for visual feature extraction and classification. A pre-trained convolutional neural network such as XceptionNet or EfficientNet will be adopted using transfer learning to improve performance and reduce training time. The design will include modules for video frame extraction, face detection and cropping, feature learning, and final classification into real or fake categories.
XceptionNet : gotten from the traditional system network (InceptionNet)  which makes use of the standard convolution algorithm that tries to learn the spatial features (shapes/edges) and the channel features (color/intensity) of the dataset together all at once. This InceptionNet approach has proven to be a little complex. This complexity brought about the XceptionNet (Extreme Inception) approach. 
In a standard convolution, we have an input of size (H x W x C) (Height, Width, Channels) and a filter of size (K x K). If we want "N" output channels, the cost is:

    i. Cstd = (H x W) - (K x K) - C - N

In the approach, the XceptionNet makes use of the Depthwise Separable Convolution which helps to simplify the spatial and channel features of the dataset by assuming it can be mapped out completely independently to ensure faster training, precision, and accuracy.
Xception breaks the math into two distinct steps to turn that multiplication into addition.
    ii. Step A: Depthwise Convolution
One K x K filter per input channel:
Cdepthwise = (H x W) - (K- K) -C
    iii. Step B: Pointwise Convolution
A 1 x 1 filter to combine those channels into "N" outputs:
Cpointwise = (H - W) -1 - 1 - C - N
    iv. The Total Xception Cost (Cxcp):
Cxcp = (H - W -C) - (K2 + N)

EfficientNet : Unlike XceptionNet that focuses on the way layers calculate data (using depthwise separable convolutions), EfficientNet focuses on the scale of the entire network such as the depth (ResNet-50 to Resnet-101), width, and resolution.
The creators of EfficientNet discovered that these three are interdependent. If you increase the resolution of your image, you need more layers to find larger patterns and more channels to find finer details.
EfficientNet uses a single "Compound Coefficient"  to scale all three dimensions uniformly. This ensures that the model grows in a balanced, mathematically optimized way. 
While Xception optimizes the internal operation, EfficientNet optimizes the external dimensions of the network. It uses a Compound Coefficient  to scale the network's Depth, Width, and Resolution using these formulas:
    - Depth: d = ?
    - Width: w = ?
    - Resolution: r = ?
EfficientNet creators found the optimal balance by solving for:
? - ?2 -  ?2=2
?  1, ?  1, ?1


1.6 EXPECTED CONTRIBUTION TO KNOWLEDGE
This research is expected to contribute to the body of knowledge in social-media forensics and cybersecurity.
The rapid increase of AI-generated synthetic media has introduced a profound crisis of trust in digital environments. As platforms like X, Instagram, and TikTok become primary sources of news and social interaction, the ability to manipulate video and audio with surgical precision poses a threat to democratic processes, personal reputations, and the collective perception of reality. In this landscape, the most significant contribution of deep learning detection is the restoration of the proof of reality. Deepfakes create a liar's dividend, where genuine evidence can be dismissed as fake and fakes can be accepted as truth. By integrating detection layers into the upload pipeline, social media platforms can provide real-time authenticity labels. This creates a transparent ecosystem where users are empowered with contextual awareness, reducing the psychological distress caused by constant scepticism of digital content.

















CHAPTER TWO
LITERATURE REVIEW
2.1 Conceptual Clarifications
Deepfakes and Synthetic Media
Deepfakes refer to synthetically generated or manipulated media in which the likeness of a real individual is convincingly altered or fabricated using artificial intelligence techniques, particularly deep learning (Verdoliva, 2020). The term is most commonly associated with face-swapped or face-manipulated videos produced using Generative Adversarial Networks (GANs) and encoder-decoder architectures (Tolosana et al., 2020). Unlike traditional video forgeries, deepfakes exhibit high perceptual realism, making them difficult to detect through visual inspection alone.
Synthetic media is a broader concept encompassing deepfake videos, manipulated images, AI-generated voices, and text-based impersonation. In the context of social media, deepfake videos are particularly dangerous due to their ability to convincingly portray public figures or private individuals saying or doing things that never occurred, thereby enabling misinformation, fraud, and reputational harm (Agarwal et al., 2019).

Social Media as a Distribution Environment
Social media platforms such as Facebook, Instagram, TikTok, and X have become primary channels for multimedia dissemination. These platforms apply automatic video compression, resizing, and re-encoding to optimize bandwidth and storage (Zhao et al., 2023). While these processes enhance scalability, they also alter video artifacts and suppress subtle forensic traces used by detection algorithms. Consequently, deepfake detection on social media must account for degraded video quality and heterogeneous content pipelines (Rossler et al., 2019).

Deep Learning and Computer Vision
Deep learning is a subset of machine learning that employs multi-layer neural networks to learn hierarchical feature representations from data. In computer vision, convolutional neural networks (CNNs) have demonstrated exceptional performance in tasks such as image classification, face recognition, and video analysis (LeCun et al., 2015). Deepfake detection leverages CNNs to identify spatial inconsistencies, texture artifacts, and temporal irregularities introduced during synthetic media generation (G�era & Delp, 2018).


Multimedia Forensics
Multimedia forensics is the scientific discipline concerned with the analysis of digital images, videos, and audio to verify authenticity and detect manipulation (Verdoliva, 2020). Unlike cryptographic watermarking, forensic approaches are passive and do not require prior embedding of security markers. Deepfake detection is now regarded as a critical subfield of multimedia forensics due to the scale and sophistication of AI-generated media.

2.2 Evolution of Deepfake Technology
Early Digital Face Manipulation
Early face manipulation techniques relied on manual editing and traditional computer graphics, requiring significant expertise and time. These methods produced visible artifacts that could be detected using rule-based forensic techniques (Farid, 2009).

Emergence of Deep Learning-Based Generation
The introduction of GANs by Goodfellow et al. (2014) marked a turning point in synthetic media generation. GAN-based deepfakes enabled automated learning of facial expressions, lighting, and motion, resulting in highly realistic videos. Encoder-decoder architectures further simplified the creation of face-swapped videos, accelerating the spread of deepfakes on online platforms (Rossler et al., 2019).

Current Generation Deepfakes
Recent deepfake models integrate attention mechanisms, temporal coherence constraints, and high-resolution synthesis, making them increasingly robust against traditional forensic detection (Tolosana et al., 2020). These advances necessitate equally sophisticated detection mechanisms that can adapt to evolving generation techniques.

2.3 Deepfake Detection Approaches
Traditional Forensic Techniques
Early detection methods focused on hand-crafted features such as eye blinking frequency, head pose inconsistencies, and color mismatches (Li et al.)