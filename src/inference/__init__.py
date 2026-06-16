"""Inference pipeline for real-time deepfake detection."""

from src.inference.video_classifier import VideoClassifier
from src.inference.predict_image import ImagePredictor
from src.inference.predict_video import VideoPredictor

__all__ = ["VideoClassifier", "ImagePredictor", "VideoPredictor"]
