"""
Structured logging configuration for the deepfake detection system.

Purpose: Provide consistent, structured logging across all modules.
Responsibilities: Configure loggers with file and console handlers, format messages.
Dependencies: logging, pathlib

Research Traceability:
    Research Objective: Reproducible experiment tracking
    Methodology: Structured logging for audit trails
    Implementation: src/utils/logger.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    """Create and configure a logger with console and optional file handlers.

    Args:
        name: Logger name, typically the module name.
        log_dir: Directory for log files. If None, only console output.
        level: Logging level (default: INFO).
        console_output: Whether to output logs to console.
        file_output: Whether to output logs to a file.

    Returns:
        Configured logging.Logger instance.

    Example:
        logger = setup_logger("training", log_dir="outputs/logs")
        logger.info("Starting training epoch 1/50")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file_output and log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / f"{name}.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger by name.

    Args:
        name: The logger name.

    Returns:
        The logging.Logger instance if it exists, otherwise a new one.
    """
    return logging.getLogger(name)
