"""
Pydantic models for API request/response schemas.

Purpose: Define type-safe request/response contracts for the FastAPI inference service.
Responsibilities: Input validation, output serialization, OpenAPI documentation.
Dependencies: pydantic

Research Traceability:
    Research Objective: Deployable deepfake detection API
    Methodology: RESTful API with structured request/response models
    Implementation: src/api/models.py
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AggregationMethod(str, Enum):
    """Aggregation methods for frame-to-video classification."""

    MEAN = "mean"
    MAJORITY = "majority"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


class ModelType(str, Enum):
    """Supported model architectures."""

    XCEPTION = "xception"
    EFFICIENTNET = "efficientnet"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ImagePredictRequest(BaseModel):
    """Single image prediction request."""

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold for fake/real decision",
    )


class VideoPredictRequest(BaseModel):
    """Video prediction request."""

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold for fake/real decision",
    )
    aggregation: AggregationMethod = Field(
        default=AggregationMethod.MEAN,
        description="Frame-to-video aggregation method",
    )
    frame_sample_rate: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Process every Nth frame from the video",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ImagePredictionResponse(BaseModel):
    """Single image prediction result."""

    filename: str = Field(description="Input filename")
    is_fake: bool = Field(description="Whether the image is classified as fake")
    confidence: float = Field(description="Prediction confidence [0, 1]")
    fake_probability: float = Field(description="Probability of being fake [0, 1]")
    real_probability: float = Field(description="Probability of being real [0, 1]")
    face_detected: bool = Field(description="Whether a face was detected")
    bounding_box: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Face bounding box (x1, y1, x2, y2) if detected",
    )


class FrameResult(BaseModel):
    """Per-frame prediction result."""

    frame_index: int = Field(description="Frame number in the video")
    fake_probability: float = Field(description="Fake probability for this frame")
    is_fake: bool = Field(description="Frame-level fake/real decision")


class VideoPredictionResponse(BaseModel):
    """Video prediction result."""

    filename: str = Field(description="Input video filename")
    is_fake: bool = Field(description="Video-level fake/real decision")
    confidence: float = Field(description="Video-level confidence [0, 1]")
    fake_probability: float = Field(description="Aggregated fake probability")
    real_probability: float = Field(description="Aggregated real probability")
    total_frames: int = Field(description="Number of frames extracted")
    frames_with_face: int = Field(description="Frames where a face was detected")
    aggregation_method: str = Field(description="Aggregation method used")
    frame_results: list[FrameResult] = Field(
        default_factory=list,
        description="Per-frame prediction details",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether a model is loaded")
    model_type: str | None = Field(default=None, description="Loaded model architecture")
    device: str = Field(description="Computation device")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(description="Error message")
    detail: str | None = Field(default=None, description="Additional error details")
