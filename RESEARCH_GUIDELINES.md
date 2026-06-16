# RESEARCH_GUIDELINES.md

## Purpose

This document defines the academic, scientific, and research standards that must be followed throughout the development of this project.

The repository is not merely a software project.

It is an academic cybersecurity research project intended to support a Bachelor's Degree dissertation.

All implementation decisions must be academically defensible.

---

# Primary Research Document

The authoritative academic source is:

thesis/chapters_1_2_3.md

This document contains:

* Problem statement
* Motivation
* Research objectives
* Scope
* Methodology
* Literature review
* Dataset selection
* Model justification

Before implementing any feature, verify alignment with the research document.

---

# Research Domain

Primary Domain:

Cyber Security

Subdomains:

* Deepfake Detection
* Multimedia Forensics
* Artificial Intelligence
* Deep Learning
* Computer Vision
* Social Media Security

All work must remain within these domains.

---

# Research Questions

The repository should help answer:

1. Can deep learning effectively detect social media deepfake videos?

2. How effective are CNN-based architectures for deepfake detection?

3. How does social media compression affect detection performance?

4. Which model architecture performs better under realistic conditions?

5. Can the system contribute to practical multimedia forensic applications?

---

# Approved Datasets

Primary Datasets:

## FaceForensics++

Purpose:

* Benchmark dataset
* Controlled evaluation
* Comparison with literature

## Celeb-DF

Purpose:

* Realistic deepfake evaluation
* Improved robustness testing
* Social media-like conditions

Any additional dataset must be documented and justified.

---

# Approved Models

Primary Models:

## XceptionNet

Justification:

* Strong literature support
* Proven performance on FaceForensics++
* Widely accepted benchmark

## EfficientNet

Justification:

* Efficient scaling
* Lower computational cost
* Competitive accuracy

Alternative models require written justification.

---

# Experimental Methodology

All experiments must follow:

1. Dataset Collection

2. Dataset Preparation

3. Face Extraction

4. Face Detection

5. Face Cropping

6. Data Normalization

7. Model Training

8. Validation

9. Evaluation

10. Reporting

---

# Evaluation Metrics

Mandatory Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Optional Metrics:

* Confusion Matrix
* Specificity
* Sensitivity

No metric may be fabricated.

---

# Reproducibility Requirements

Every experiment must record:

* Dataset version
* Model version
* Hyperparameters
* Epochs
* Learning rate
* Random seed
* Training date
* Hardware information

Results must be reproducible.

---

# Citation Rules

Whenever external research influences implementation:

Document:

* Author
* Publication year
* DOI or source

Store references in:

thesis/references.bib

---

# Academic Integrity

Never:

* Invent citations
* Invent references
* Invent results
* Invent performance metrics
* Invent dataset statistics

All reported values must originate from actual experiments.

---

# Chapter 4 Support Requirements

The repository must automatically generate:

* Tables
* Graphs
* Training curves
* Validation curves
* Confusion matrices
* Performance comparisons

for inclusion in Chapter 4.

---

# Chapter 5 Support Requirements

The repository must provide evidence for:

* Findings
* Conclusions
* Recommendations
* Future work

through experimentally validated results.

---

# Final Research Goal

Develop a technically sound, reproducible, and academically defensible deepfake detection system suitable for cybersecurity and multimedia forensic applications.
