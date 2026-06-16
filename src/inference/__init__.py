"""Inference pipeline for real-time deepfake detection.

Provides video-level and image-level deepfake classification with
frame-level interpretability analysis.
"""

from src.inference.video_classifier import VideoClassifier
from src.inference.predict_image import ImagePredictor
from src.inference.predict_video import VideoPredictor
from src.inference.frame_analysis import FrameAnalyzer

__all__ = ["VideoClassifier", "ImagePredictor", "VideoPredictor", "FrameAnalyzer"]
