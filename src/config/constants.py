"""
Constants used throughout the deepfake detection system.

Purpose: Define fixed constants that do not change between experiments.
Responsibilities: Centralize magic numbers, class labels, supported models, dataset names.
Dependencies: None

Research Traceability:
    Research Objective: Standardized experiment configuration
    Methodology: Constants for reproducible research
    Implementation: src/config/constants.py
"""

from __future__ import annotations


class Constants:
    """Application-wide constants.

    All values are immutable and should not be modified at runtime.
    """

    # Class labels
    REAL_LABEL: int = 0
    FAKE_LABEL: int = 1
    CLASS_NAMES: list[str] = ["real", "fake"]
    NUM_CLASSES: int = 2

    # Supported models
    SUPPORTED_MODELS: list[str] = ["xception", "efficientnet", "efficientnet_b0", "efficientnet_b4"]
    DEFAULT_MODEL: str = "xception"

    # Supported datasets
    SUPPORTED_DATASETS: list[str] = ["faceforensics", "celebdf"]
    FACEFORENSICS_MANIPULATION_METHODS: list[str] = [
        "deepfakes",
        "face2face",
        "faceswap",
        "neuraltextures",
    ]

    # Image preprocessing
    XCEPTION_INPUT_SIZE: int = 299
    EFFICIENTNET_INPUT_SIZE: int = 224
    IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

    # Frame extraction
    DEFAULT_FRAME_SAMPLE_RATE: int = 5  # Extract 1 frame every N frames
    MIN_FRAMES_PER_VIDEO: int = 10
    MAX_FRAMES_PER_VIDEO: int = 100

    # Face detection
    FACE_DETECTION_METHODS: list[str] = ["retinaface", "mtcnn", "opencv"]
    DEFAULT_FACE_CONFIDENCE: float = 0.9
    DEFAULT_FACE_MARGIN: float = 0.2

    # Training defaults
    DEFAULT_SEED: int = 42
    DEFAULT_EPOCHS: int = 50
    DEFAULT_BATCH_SIZE: int = 32
    DEFAULT_LEARNING_RATE: float = 1e-4
    DEFAULT_WEIGHT_DECAY: float = 1e-5

    # Video aggregation methods
    AGGREGATION_METHODS: list[str] = ["mean", "median", "majority_vote", "confidence_weighted"]

    # File extensions
    VIDEO_EXTENSIONS: list[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    IMAGE_EXTENSIONS: list[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    # Experiment tracking
    METRIC_NAMES: list[str] = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]

    # API defaults
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_UPLOAD_SIZE_MB: int = 100
