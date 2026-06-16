"""
File management utilities.

Purpose: Provide safe file I/O operations for the deepfake detection system.
Responsibilities: Read/write JSON, CSV, and other structured data.
Dependencies: json, csv, pathlib

Research Traceability:
    Research Objective: Experiment data persistence
    Methodology: Structured file I/O
    Implementation: src/utils/file_manager.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_json(file_path: str | Path) -> dict[str, Any]:
    """Read a JSON file and return its contents.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Parsed JSON data as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict[str, Any], file_path: str | Path, indent: int = 2) -> Path:
    """Write data to a JSON file.

    Args:
        data: Dictionary to serialize.
        file_path: Output file path.
        indent: JSON indentation level.

    Returns:
        Path to the written file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    return file_path


def read_csv(file_path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file and return rows as dictionaries.

    Args:
        file_path: Path to the CSV file.

    Returns:
        List of dictionaries, one per row.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(rows: list[dict[str, Any]], file_path: str | Path) -> Path:
    """Write rows to a CSV file.

    Args:
        rows: List of dictionaries to write.
        file_path: Output file path.

    Returns:
        Path to the written file.

    Raises:
        ValueError: If rows is empty.
    """
    if not rows:
        raise ValueError("Cannot write empty rows to CSV")

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return file_path
