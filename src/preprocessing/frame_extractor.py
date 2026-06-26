"""
Video frame extraction.

Purpose: Extract frames from videos for deepfake detection.
Responsibilities: OpenCV-based frame extraction with configurable sampling.
Dependencies: opencv-python, numpy, pathlib

Research Traceability:
    Research Objective: Frame-level analysis for temporal deepfake detection
    Methodology: Uniform frame sampling from video files
    Implementation: src/preprocessing/frame_extractor.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FrameExtractor:
    """Extract frames from video files.

    Supports:
    - Uniform frame sampling
    - Fixed number of frames per video
    - Configurable frame intervals
    """

    def __init__(
        self,
        sampling_rate: int = 1,
        max_frames: int | None = None,
    ) -> None:
        """Initialize frame extractor.

        Args:
            sampling_rate: Extract every Nth frame (default: 1 = every frame).
            max_frames: Maximum number of frames to extract per video.
        """
        self.sampling_rate = sampling_rate
        self.max_frames = max_frames

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        prefix: str = "frame",
    ) -> list[Path]:
        """Extract frames from a video file.

        Args:
            video_path: Path to the video file.
            output_dir: Directory to save extracted frames.
            prefix: Prefix for frame filenames.

        Returns:
            List of paths to extracted frame images.

        Raises:
            FileNotFoundError: If video file does not exist.
            RuntimeError: If video cannot be opened.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(
            f"Video: {video_path.name} | FPS: {fps:.2f} | "
            f"Total frames: {total_frames} | Duration: {duration:.2f}s"
        )

        frame_paths = []
        frame_idx = 0
        extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.sampling_rate == 0:
                frame_filename = f"{prefix}_{frame_idx:06d}.jpg"
                frame_path = output_dir / frame_filename

                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(frame_path)
                extracted += 1

                if self.max_frames and extracted >= self.max_frames:
                    logger.info(f"Reached max frames limit: {self.max_frames}")
                    break

            frame_idx += 1

        cap.release()

        logger.info(f"Extracted {extracted} frames to {output_dir}")
        return frame_paths

    def extract(
        self,
        video_path: Path,
        sampling_rate: int | None = None,
        sample_rate: int | None = None,
    ) -> list[np.ndarray]:
        """Extract frames as numpy arrays (in-memory, no disk I/O).

        Args:
            video_path: Path to the video file.
            sampling_rate: Override instance sampling_rate for this extraction.
            sample_rate: Alias for sampling_rate.

        Returns:
            List of frames as numpy arrays (RGB, HWC, uint8).
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        effective_rate = sample_rate if sample_rate is not None else sampling_rate
        rate = effective_rate if effective_rate is not None else self.sampling_rate

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

                if self.max_frames and len(frames) >= self.max_frames:
                    break

            frame_idx += 1

        cap.release()
        return frames

    def extract_frames_at_timestamps(
        self,
        video_path: Path,
        output_dir: Path,
        timestamps: list[float],
        prefix: str = "frame",
    ) -> list[Path]:
        """Extract frames at specific timestamps.

        Args:
            video_path: Path to the video file.
            output_dir: Directory to save extracted frames.
            timestamps: List of timestamps in seconds.
            prefix: Prefix for frame filenames.

        Returns:
            List of paths to extracted frame images.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_paths = []

        for i, timestamp in enumerate(timestamps):
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            if ret:
                frame_filename = f"{prefix}_t{timestamp:.2f}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(frame_path)

        cap.release()

        logger.info(f"Extracted {len(frame_paths)} frames at specified timestamps")
        return frame_paths

    def get_video_info(self, video_path: Path) -> dict:
        """Get video file information.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary with video metadata.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0

        cap.release()
        return info
