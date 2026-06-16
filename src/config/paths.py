"""
Path configuration for the deepfake detection system.

Purpose: Define and manage all filesystem paths used by the project.
Responsibilities: Centralize path definitions, ensure directory existence.
Dependencies: pathlib

Research Traceability:
    Research Objective: Reproducible experiment structure
    Methodology: Standardized directory layout
    Implementation: src/config/paths.py
"""

from __future__ import annotations

from pathlib import Path


class PathConfig:
    """Manages all filesystem paths for the project.

    All paths are relative to the project root directory.
    Directories are created automatically when accessed.

    Usage:
        paths = PathConfig()
        dataset_dir = paths.datasets_raw / "faceforensics"
    """

    def __init__(self, project_root: str | Path | None = None) -> None:
        """Initialize path configuration.

        Args:
            project_root: Root directory of the project. If None, auto-detected.
        """
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent.parent
        else:
            self.project_root = Path(project_root).resolve()

    @property
    def src(self) -> Path:
        """Source code directory."""
        return self.project_root / "src"

    @property
    def configs(self) -> Path:
        """Configuration files directory."""
        return self.project_root / "configs"

    @property
    def datasets(self) -> Path:
        """Datasets root directory."""
        return self._ensure_dir(self.project_root / "datasets")

    @property
    def datasets_raw(self) -> Path:
        """Raw datasets directory."""
        return self._ensure_dir(self.datasets / "raw")

    @property
    def datasets_raw_faceforensics(self) -> Path:
        """FaceForensics++ raw dataset directory."""
        return self._ensure_dir(self.datasets_raw / "faceforensics")

    @property
    def datasets_raw_celebdf(self) -> Path:
        """Celeb-DF raw dataset directory."""
        return self._ensure_dir(self.datasets_raw / "celebdf")

    @property
    def datasets_interim(self) -> Path:
        """Interim processed data directory."""
        return self._ensure_dir(self.datasets / "interim")

    @property
    def datasets_interim_frames(self) -> Path:
        """Extracted frames directory."""
        return self._ensure_dir(self.datasets_interim / "extracted_frames")

    @property
    def datasets_interim_faces(self) -> Path:
        """Detected faces directory."""
        return self._ensure_dir(self.datasets_interim / "detected_faces")

    @property
    def datasets_interim_normalized(self) -> Path:
        """Normalized faces directory."""
        return self._ensure_dir(self.datasets_interim / "normalized_faces")

    @property
    def datasets_processed(self) -> Path:
        """Final processed datasets directory."""
        return self._ensure_dir(self.datasets / "processed")

    @property
    def datasets_processed_train(self) -> Path:
        """Training split directory."""
        return self._ensure_dir(self.datasets_processed / "train")

    @property
    def datasets_processed_val(self) -> Path:
        """Validation split directory."""
        return self._ensure_dir(self.datasets_processed / "validation")

    @property
    def datasets_processed_test(self) -> Path:
        """Test split directory."""
        return self._ensure_dir(self.datasets_processed / "test")

    @property
    def datasets_metadata(self) -> Path:
        """Dataset metadata directory."""
        return self._ensure_dir(self.datasets / "metadata")

    @property
    def outputs(self) -> Path:
        """Outputs root directory."""
        return self._ensure_dir(self.project_root / "outputs")

    @property
    def outputs_models(self) -> Path:
        """Saved models directory."""
        return self._ensure_dir(self.outputs / "models")

    @property
    def outputs_checkpoints(self) -> Path:
        """Training checkpoints directory."""
        return self._ensure_dir(self.outputs_models / "checkpoints")

    @property
    def outputs_metrics(self) -> Path:
        """Evaluation metrics directory."""
        return self._ensure_dir(self.outputs / "metrics")

    @property
    def outputs_reports(self) -> Path:
        """Generated reports directory."""
        return self._ensure_dir(self.outputs / "reports")

    @property
    def outputs_predictions(self) -> Path:
        """Prediction outputs directory."""
        return self._ensure_dir(self.outputs / "predictions")

    @property
    def outputs_logs(self) -> Path:
        """Training logs directory."""
        return self._ensure_dir(self.outputs / "logs")

    @property
    def outputs_thesis_figures(self) -> Path:
        """Thesis-ready figures directory."""
        return self._ensure_dir(self.outputs_reports / "thesis_figures")

    @property
    def experiments(self) -> Path:
        """Experiments root directory."""
        return self._ensure_dir(self.project_root / "experiments")

    @property
    def reports(self) -> Path:
        """Reports root directory."""
        return self._ensure_dir(self.project_root / "reports")

    @property
    def reports_figures(self) -> Path:
        """Report figures directory."""
        return self._ensure_dir(self.reports / "figures")

    @property
    def reports_tables(self) -> Path:
        """Report tables directory."""
        return self._ensure_dir(self.reports / "tables")

    @property
    def notebooks(self) -> Path:
        """Jupyter notebooks directory."""
        return self.project_root / "notebooks"

    @property
    def scripts(self) -> Path:
        """Shell scripts directory."""
        return self.project_root / "scripts"

    @property
    def tests(self) -> Path:
        """Test directory."""
        return self.project_root / "tests"

    @property
    def thesis(self) -> Path:
        """Thesis documents directory."""
        return self.project_root / "thesis"

    @staticmethod
    def _ensure_dir(path: Path) -> Path:
        """Create directory if it does not exist.

        Args:
            path: Directory path to ensure exists.

        Returns:
            The same path, guaranteed to exist as a directory.
        """
        path.mkdir(parents=True, exist_ok=True)
        return path
