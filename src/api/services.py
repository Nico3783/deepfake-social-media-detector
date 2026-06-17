"""
API business logic services.

Purpose: Encapsulate prediction and model management logic.
Responsibilities: Model loading, prediction orchestration, result formatting.
Dependencies: src.inference, src.models, src.config

Research Traceability:
    Research Objective: Deployable deepfake detection API
    Methodology: Service layer separating HTTP from inference logic
    Implementation: src/api/services.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.config.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PredictionService:
    """Service for managing model inference.

    Handles model loading, prediction, and result caching.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize prediction service.

        Args:
            settings: Project settings. Uses defaults if None.
        """
        self.settings = settings or Settings()
        self._image_predictor: Any = None
        self._video_predictor: Any = None
        self._model_type: str | None = None
        self._device: str = "cpu"
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._loaded

    @property
    def model_type(self) -> str | None:
        """Get loaded model type."""
        return self._model_type

    @property
    def device(self) -> str:
        """Get computation device."""
        return self._device

    def load_models(
        self,
        model_path: str | Path,
        model_type: str = "xception",
        device: str | None = None,
    ) -> None:
        """Load models for inference.

        Args:
            model_path: Path to model checkpoint.
            model_type: Model architecture name.
            device: Target device.
        """
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = device
        self._model_type = model_type

        try:
            from src.inference.predict_image import ImagePredictor
            from src.inference.video_classifier import VideoClassifier

            self._image_predictor = ImagePredictor(
                model_path=str(model_path),
                model_type=model_type,
                device=device,
            )

            self._video_predictor = VideoClassifier(
                image_predictor=self._image_predictor,
                device=device,
            )

            self._loaded = True
            logger.info(
                "Models loaded: type=%s, device=%s, path=%s",
                model_type, device, model_path,
            )

        except Exception as e:
            logger.error("Failed to load models: %s", e)
            self._loaded = False
            raise

    def predict_image(
        self,
        image_path: str,
        threshold: float = 0.5,
    ) -> Any:
        """Run image prediction.

        Args:
            image_path: Path to input image.
            threshold: Classification threshold.

        Returns:
            Prediction result.

        Raises:
            RuntimeError: If models not loaded.
        """
        if not self._loaded or self._image_predictor is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        t0 = time.perf_counter()
        result = self._image_predictor.predict(image_path, threshold=threshold)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Image prediction: path=%s, fake=%s, conf=%.4f, time=%.1fms",
            image_path, result.is_fake, result.confidence, elapsed_ms,
        )

        return result

    def predict_video(
        self,
        video_path: str,
        threshold: float = 0.5,
        aggregation_method: str = "mean",
        frame_sample_rate: int = 5,
    ) -> Any:
        """Run video prediction.

        Args:
            video_path: Path to input video.
            threshold: Classification threshold.
            aggregation_method: Frame-to-video aggregation method.
            frame_sample_rate: Process every Nth frame.

        Returns:
            Prediction result.

        Raises:
            RuntimeError: If models not loaded.
        """
        if not self._loaded or self._video_predictor is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        t0 = time.perf_counter()
        result = self._video_predictor.predict(
            video_path,
            threshold=threshold,
            aggregation_method=aggregation_method,
            frame_sample_rate=frame_sample_rate,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Video prediction: path=%s, fake=%s, conf=%.4f, frames=%d, time=%.1fms",
            video_path, result.is_fake, result.confidence,
            result.total_frames, elapsed_ms,
        )

        return result

    def get_model_info(self) -> dict:
        """Get information about loaded model.

        Returns:
            Dictionary with model information.
        """
        if not self._loaded:
            return {"status": "no_model_loaded"}

        return {
            "model_type": self._model_type,
            "device": self._device,
            "is_loaded": True,
        }

    def unload_models(self) -> None:
        """Unload models to free memory."""
        self._image_predictor = None
        self._video_predictor = None
        self._loaded = False
        self._model_type = None
        logger.info("Models unloaded")
