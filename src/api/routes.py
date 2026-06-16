"""
API route definitions for the deepfake detection service.

Purpose: Define HTTP endpoints for image and video prediction.
Responsibilities: Request handling, file upload processing, inference dispatch.
Dependencies: fastapi, src.inference, src.api.models

Research Traceability:
    Research Objective: Deployable deepfake detection API
    Methodology: RESTful endpoints wrapping the inference pipeline
    Implementation: src/api/routes.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.models import (
    AggregationMethod,
    ErrorResponse,
    FrameResult,
    HealthResponse,
    ImagePredictRequest,
    ImagePredictionResponse,
    VideoPredictRequest,
    VideoPredictionResponse,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["prediction"])

# ---------------------------------------------------------------------------
# Module-level state set by app startup
# ---------------------------------------------------------------------------
_predictor: Any = None  # ImagePredictor | VideoPredictor loaded at startup
_video_predictor: Any = None
_model_type: str | None = None
_device: str = "cpu"


def set_predictors(
    image_predictor: Any,
    video_predictor: Any,
    model_type: str,
    device: str,
) -> None:
    """Set module-level predictors (called by app startup).

    Args:
        image_predictor: Initialized ImagePredictor.
        video_predictor: Initialized VideoPredictor.
        model_type: Model architecture name.
        device: Computation device string.
    """
    global _predictor, _video_predictor, _model_type, _device
    _predictor = image_predictor
    _video_predictor = video_predictor
    _model_type = model_type
    _device = device


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """Check service health and model availability."""
    return HealthResponse(
        status="healthy",
        model_loaded=_predictor is not None,
        model_type=_model_type,
        device=_device,
    )


# ---------------------------------------------------------------------------
# Image prediction
# ---------------------------------------------------------------------------

@router.post(
    "/predict/image",
    response_model=ImagePredictionResponse,
    summary="Classify a single image",
    responses={500: {"model": ErrorResponse}},
)
async def predict_image(
    file: UploadFile = File(..., description="Image file to classify"),
    threshold: float = 0.5,
) -> ImagePredictionResponse:
    """Classify a single image as real or fake.

    Accepts JPEG/PNG image files and returns a binary classification
    with confidence scores.
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Run prediction
        t0 = time.perf_counter()
        result = _predictor.predict(str(tmp_path), threshold=threshold)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Image prediction: %s → fake=%s conf=%.4f (%.3fs)",
            file.filename, result.is_fake, result.confidence, elapsed,
        )

        # Clean up temp file
        tmp_path.unlink(missing_ok=True)

        return ImagePredictionResponse(
            filename=file.filename or "unknown",
            is_fake=result.is_fake,
            confidence=result.confidence,
            fake_probability=result.fake_probability,
            real_probability=result.real_probability,
            face_detected=result.face_detected,
            bounding_box=result.bounding_box,
        )
    except Exception as e:
        logger.error("Image prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Video prediction
# ---------------------------------------------------------------------------

@router.post(
    "/predict/video",
    response_model=VideoPredictionResponse,
    summary="Classify a video",
    responses={500: {"model": ErrorResponse}},
)
async def predict_video(
    file: UploadFile = File(..., description="Video file to classify"),
    threshold: float = 0.5,
    aggregation: AggregationMethod = AggregationMethod.MEAN,
    frame_sample_rate: int = 5,
) -> VideoPredictionResponse:
    """Classify a video as real or fake.

    Extracts frames, detects faces, runs per-frame classification,
    and aggregates predictions to produce a video-level decision.
    """
    if _video_predictor is None:
        raise HTTPException(status_code=503, detail="Video predictor not loaded")

    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    try:
        suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        t0 = time.perf_counter()
        result = _video_predictor.predict(
            str(tmp_path),
            threshold=threshold,
            aggregation_method=aggregation.value,
            frame_sample_rate=frame_sample_rate,
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            "Video prediction: %s → fake=%s conf=%.4f (%.3fs)",
            file.filename, result.is_fake, result.confidence, elapsed,
        )

        tmp_path.unlink(missing_ok=True)

        frame_results = [
            FrameResult(
                frame_index=fr.frame_index,
                fake_probability=fr.fake_probability,
                is_fake=fr.is_fake,
            )
            for fr in (result.frame_results or [])
        ]

        return VideoPredictionResponse(
            filename=file.filename or "unknown",
            is_fake=result.is_fake,
            confidence=result.confidence,
            fake_probability=result.fake_probability,
            real_probability=result.real_probability,
            total_frames=result.total_frames,
            frames_with_face=result.frames_with_face,
            aggregation_method=aggregation.value,
            frame_results=frame_results,
        )
    except Exception as e:
        logger.error("Video prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
