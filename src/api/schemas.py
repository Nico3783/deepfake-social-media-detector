"""
API request/response validation schemas.

Purpose: Additional Pydantic schemas for API validation and documentation.
Responsibilities: Extended validation rules, batch request models, enum definitions.
Dependencies: pydantic

Research Traceability:
    Research Objective: Deployable deepfake detection API
    Methodology: Structured request/response validation
    Implementation: src/api/schemas.py
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BatchPredictRequest(BaseModel):
    """Batch prediction request for multiple files."""

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold",
    )
    aggregation: str = Field(
        default="mean",
        description="Aggregation method for video predictions",
    )


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str = Field(description="Model architecture name")
    num_classes: int = Field(description="Number of output classes")
    input_size: int = Field(description="Expected input image size")
    total_parameters: int = Field(description="Total model parameters")
    trainable_parameters: int = Field(description="Trainable parameters")
    device: str = Field(description="Computation device")
    pretrained: bool = Field(description="Whether pretrained weights are used")


class PredictionMetadata(BaseModel):
    """Metadata for a single prediction."""

    filename: str = Field(description="Input filename")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    model_version: str = Field(default="1.0", description="Model version")
    confidence_threshold: float = Field(description="Threshold used for decision")


class DatasetInfo(BaseModel):
    """Dataset information response."""

    name: str = Field(description="Dataset name")
    total_samples: int = Field(description="Total number of samples")
    real_samples: int = Field(description="Number of real samples")
    fake_samples: int = Field(description="Number of fake samples")
    manipulation_methods: list[str] = Field(
        default_factory=list,
        description="Manipulation methods in the dataset",
    )


class TrainingStatus(BaseModel):
    """Training status response."""

    status: str = Field(description="Training status")
    current_epoch: int = Field(description="Current training epoch")
    total_epochs: int = Field(description="Total epochs")
    current_loss: float | None = Field(default=None, description="Current loss")
    current_accuracy: float | None = Field(default=None, description="Current accuracy")
    best_metric: float | None = Field(default=None, description="Best metric value")
    elapsed_time: str | None = Field(default=None, description="Elapsed training time")


class ErrorResponseDetail(BaseModel):
    """Detailed error response."""

    error_code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error message")
    details: dict | None = Field(default=None, description="Additional error details")
    suggestion: str | None = Field(
        default=None,
        description="Suggestion for resolving the error",
    )


class VideoFrameInfo(BaseModel):
    """Information about extracted video frames."""

    frame_index: int = Field(description="Frame index in video")
    timestamp_ms: float = Field(description="Frame timestamp in milliseconds")
    face_detected: bool = Field(description="Whether face was detected")
    face_bbox: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Face bounding box",
    )


class AggregationResult(BaseModel):
    """Result of frame-level aggregation."""

    method: str = Field(description="Aggregation method used")
    video_prediction: bool = Field(description="Video-level prediction")
    video_confidence: float = Field(description="Video-level confidence")
    num_frames: int = Field(description="Number of frames used")
    fake_frame_ratio: float = Field(description="Ratio of frames classified as fake")
