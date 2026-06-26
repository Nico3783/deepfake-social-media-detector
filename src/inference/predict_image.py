"""
Image prediction pipeline.

Purpose: Single image deepfake prediction.
Responsibilities: Image preprocessing, face detection, model inference.
Dependencies: torch, PIL, src.preprocessing, src.models

Research Traceability:
    Research Objective: Frame-level deepfake detection
    Methodology: Face detection → preprocessing → CNN classification
    Implementation: src/inference/predict_image.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.face_cropper import FaceCropper
from src.preprocessing.image_resizer import ImageResizer
from src.preprocessing.normalizer import ImageNormalizer
from src.utils.logger import setup_logger
from src.utils.helpers import get_device

logger = setup_logger(__name__)


@dataclass
class ImagePrediction:
    """Single image prediction result."""

    image_path: str
    is_fake: bool
    confidence: float
    fake_probability: float
    real_probability: float
    face_detected: bool
    bounding_box: tuple[int, int, int, int] | None = None


class ImagePredictor:
    """Classify a single image as real or fake.

    Pipeline:
    1. Load image
    2. Detect face
    3. Crop and preprocess face
    4. Classify with model
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        threshold: float = 0.5,
        target_size: int = 299,
    ) -> None:
        """Initialize image predictor.

        Args:
            model: Trained frame-level classifier.
            device: Device for inference.
            threshold: Classification threshold.
            target_size: Target image size for model input.
        """
        self.device = device or get_device()
        self.threshold = threshold
        self.target_size = target_size

        # Initialize pipeline components
        self.face_detector = FaceDetector()
        self.face_cropper = FaceCropper()
        self.image_resizer = ImageResizer(target_size=target_size)
        self.normalizer = ImageNormalizer()

        # Move model to device and set to eval mode
        self.model = model.to(self.device)
        self.model.eval()

        logger.info(
            f"ImagePredictor initialized: "
            f"threshold={threshold}, target_size={target_size}"
        )

    @torch.no_grad()
    def predict(
        self,
        image_path: str | Path,
    ) -> ImagePrediction:
        """Predict whether an image is real or fake.

        Args:
            image_path: Path to the image file.

        Returns:
            ImagePrediction with classification result.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Detect face
        faces = self.face_detector.detect(image_np)
        face_detected = len(faces) > 0
        bounding_box = None

        if face_detected:
            # Crop face
            face = self.face_cropper.crop(image_np, faces[0])
            bounding_box = tuple(faces[0])  # type: ignore
        else:
            # Use full image if no face detected
            logger.warning(f"No face detected in {image_path}, using full image")
            face = image_np

        # Preprocess
        resized = self.image_resizer.resize(face)
        normalized = self.normalizer.normalize(resized)

        # Add batch dimension and convert to tensor (HWC -> CHW)
        if isinstance(normalized, np.ndarray):
            # Transpose from HWC to CHW, then add batch dim
            tensor = torch.from_numpy(normalized).permute(2, 0, 1)
            input_tensor = tensor.unsqueeze(0).to(self.device)
        else:
            input_tensor = normalized.unsqueeze(0).to(self.device)

        # Classify
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)

        fake_prob = probs[0, 1].item()
        real_prob = probs[0, 0].item()
        is_fake = fake_prob >= self.threshold
        confidence = fake_prob if is_fake else real_prob

        prediction = ImagePrediction(
            image_path=str(image_path),
            is_fake=is_fake,
            confidence=confidence,
            fake_probability=fake_prob,
            real_probability=real_prob,
            face_detected=face_detected,
            bounding_box=bounding_box,
        )

        logger.info(
            f"Image {image_path}: {'FAKE' if is_fake else 'REAL'} "
            f"(confidence={confidence:.4f})"
        )

        return prediction

    def to_dict(self, prediction: ImagePrediction) -> dict[str, Any]:
        """Convert prediction to dictionary.

        Args:
            prediction: ImagePrediction to convert.

        Returns:
            Dictionary representation.
        """
        return {
            "image_path": prediction.image_path,
            "is_fake": prediction.is_fake,
            "classification": "FAKE" if prediction.is_fake else "REAL",
            "confidence": prediction.confidence,
            "fake_probability": prediction.fake_probability,
            "real_probability": prediction.real_probability,
            "face_detected": prediction.face_detected,
            "bounding_box": list(prediction.bounding_box) if prediction.bounding_box else None,
        }


def main() -> None:
    """CLI entry point for single image prediction."""
    import argparse

    parser = argparse.ArgumentParser(description="Predict if an image is real or fake")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Config file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()

    from src.config.settings import Settings
    from src.models.model_factory import create_model

    settings = Settings.from_yaml(args.config)
    device = torch.device(settings.get_device())
    model = create_model(settings.config.model, device)

    predictor = ImagePredictor(model, device, threshold=args.threshold)
    predictor.model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False).get("model_state_dict", torch.load(args.checkpoint, map_location=device, weights_only=False)))

    prediction = predictor.predict(args.image)
    result = predictor.to_dict(prediction)
    print(json.dumps(result, indent=2))
