"""
Face cropping and alignment.

Purpose: Crop detected faces from images for deepfake analysis.
Responsibilities: Face cropping, alignment, and padding.
Dependencies: PIL, numpy, opencv-python

Research Traceability:
    Research Objective: Face region extraction for targeted analysis
    Methodology: Bounding box-based cropping with margin
    Implementation: src/preprocessing/face_cropper.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FaceCropper:
    """Crop and align faces from images.

    Supports:
    - Bounding box-based cropping
    - Configurable margin
    - Face alignment using landmarks
    """

    def __init__(
        self,
        target_size: int = 299,
        margin: int = 20,
    ) -> None:
        """Initialize face cropper.

        Args:
            target_size: Target size for cropped face (default: 299 for XceptionNet).
            margin: Margin around face bounding box (pixels).
        """
        self.target_size = target_size
        self.margin = margin

    def crop_face(
        self,
        image: Image.Image | np.ndarray,
        bbox: list[float],
        landmarks: list[list[float]] | None = None,
    ) -> Image.Image:
        """Crop a face from an image using bounding box.

        Args:
            image: Input image.
            bbox: Bounding box [x1, y1, x2, y2].
            landmarks: Optional facial landmarks for alignment.

        Returns:
            Cropped face as PIL Image.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        width, height = image.size
        x1, y1, x2, y2 = bbox

        # Add margin
        x1 = max(0, int(x1 - self.margin))
        y1 = max(0, int(y1 - self.margin))
        x2 = min(width, int(x2 + self.margin))
        y2 = min(height, int(y2 + self.margin))

        # Crop and resize
        face = image.crop((x1, y1, x2, y2))
        face = face.resize(
            (self.target_size, self.target_size),
            Image.Resampling.LANCZOS,
        )

        return face

    def crop_face_from_detection(
        self,
        image: Image.Image | np.ndarray,
        detection: dict,
    ) -> Image.Image:
        """Crop face from a detection dictionary.

        Args:
            image: Input image.
            detection: Face detection with 'bbox' key.

        Returns:
            Cropped face as PIL Image.
        """
        return self.crop_face(
            image,
            bbox=detection["bbox"],
            landmarks=detection.get("landmarks"),
        )

    def crop_largest_face(
        self,
        image: Image.Image | np.ndarray,
        detections: list[dict],
    ) -> Image.Image | None:
        """Crop the largest detected face.

        Args:
            image: Input image.
            detections: List of face detections.

        Returns:
            Cropped face, or None if no detections.
        """
        if not detections:
            return None

        # Find largest face
        def face_area(det: dict) -> float:
            bbox = det["bbox"]
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        largest = max(detections, key=face_area)
        return self.crop_face_from_detection(image, largest)

    def crop_faces_batch(
        self,
        images: list[Image.Image | np.ndarray],
        detections_list: list[list[dict]],
    ) -> list[list[Image.Image]]:
        """Crop faces from a batch of images.

        Args:
            images: List of input images.
            detections_list: List of detection results for each image.

        Returns:
            List of cropped faces for each image.
        """
        results = []
        for image, detections in zip(images, detections_list):
            faces = []
            for detection in detections:
                face = self.crop_face_from_detection(image, detection)
                faces.append(face)
            results.append(faces)
        return results

    def crop(
        self,
        image: Image.Image | np.ndarray,
        detection: dict | list | tuple,
    ) -> Image.Image:
        """Crop face from an image.

        Accepts either a detection dictionary with 'bbox' key or a
        bounding box tuple/list (x1, y1, x2, y2).

        Args:
            image: Input image.
            detection: Face detection dict with 'bbox' key, or bbox tuple/list.

        Returns:
            Cropped face as numpy array (uint8).
        """
        if isinstance(detection, dict):
            face = self.crop_face_from_detection(image, detection)
        else:
            face = self.crop_face(image, bbox=list(detection))
        return np.array(face)

    def align_face(
        self,
        image: Image.Image | np.ndarray,
        landmarks: list[list[float]],
    ) -> Image.Image:
        """Align face using facial landmarks.

        Args:
            image: Input image.
            landmarks: Facial landmarks (5 points: left eye, right eye, nose, left mouth, right mouth).

        Returns:
            Aligned face image.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Convert to numpy for alignment
        img_array = np.array(image)

        # Get eye positions
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        # Calculate angle between eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Calculate center between eyes
        center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2,
        )

        # Rotate image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(
            img_array,
            M,
            (img_array.shape[1], img_array.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        return Image.fromarray(aligned)
