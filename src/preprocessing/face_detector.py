"""
Face detection using MTCNN.

Purpose: Detect faces in video frames for deepfake analysis.
Responsibilities: MTCNN-based face detection with confidence scoring.
Dependencies: facenet-pytorch, torch, PIL, numpy

Research Traceability:
    Research Objective: Face region extraction for targeted analysis
    Methodology: MTCNN for robust face detection
    Implementation: src/preprocessing/face_detector.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FaceDetector:
    """Detect faces in images using MTCNN.

    MTCNN (Multi-task Cascaded Convolutional Networks) provides:
    - Face detection
    - Facial landmark detection
    - Face alignment
    """

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        min_face_size: int = 40,
        device: str | None = None,
    ) -> None:
        """Initialize face detector.

        Args:
            confidence_threshold: Minimum confidence for detection.
            min_face_size: Minimum face size in pixels.
            device: Device for inference ('cuda', 'cpu', or None for auto).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size

        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
            post_process=True,
        )

        logger.info(f"Face detector initialized on {self.device}")

    def detect_faces(self, image: Image.Image | np.ndarray) -> list[dict]:
        """Detect faces in an image.

        Args:
            image: Input image (PIL Image or numpy array).

        Returns:
            List of face dictionaries with:
            - bbox: Bounding box [x1, y1, x2, y2]
            - confidence: Detection confidence
            - landmarks: Facial landmarks (optional)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Detect faces
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)

        faces = []
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob >= self.confidence_threshold:
                    face_info = {
                        "bbox": box.tolist(),
                        "confidence": float(prob),
                    }
                    if landmarks is not None:
                        face_info["landmarks"] = landmarks[i].tolist()
                    faces.append(face_info)

        logger.debug(f"Detected {len(faces)} faces with confidence >= {self.confidence_threshold}")
        return faces

    def detect_faces_batch(
        self,
        images: list[Image.Image | np.ndarray],
    ) -> list[list[dict]]:
        """Detect faces in a batch of images.

        Args:
            images: List of input images.

        Returns:
            List of face detection results for each image.
        """
        results = []
        for image in images:
            faces = self.detect_faces(image)
            results.append(faces)
        return results

    def get_largest_face(self, faces: list[dict]) -> dict | None:
        """Get the largest face from detected faces.

        Args:
            faces: List of detected faces.

        Returns:
            Largest face dictionary, or None if no faces.
        """
        if not faces:
            return None

        def face_area(face: dict) -> float:
            bbox = face["bbox"]
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        return max(faces, key=face_area)

    def detect_from_path(self, image_path: Path) -> list[dict]:
        """Detect faces from an image file path.

        Args:
            image_path: Path to the image file.

        Returns:
            List of detected faces.
        """
        image = Image.open(image_path).convert("RGB")
        return self.detect_faces(image)
