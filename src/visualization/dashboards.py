"""
Interactive dashboard generation for experiment tracking.

Purpose: Create HTML-based dashboards summarizing model performance and experiments.
Responsibilities: HTML report generation, experiment comparison tables, interactive charts.
Dependencies: pathlib, json, html (stdlib)

Research Traceability:
    Research Objective: Experiment tracking and reproducible evaluation
    Methodology: Structured experiment logging with comparable metrics
    Implementation: src/visualization/dashboards.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.logger import setup_logger
from src.utils.paths import PathConfig

logger = setup_logger(__name__)


class ExperimentDashboard:
    """Generate HTML dashboards for experiment tracking and comparison.

    Produces self-contained HTML files with:
    - Experiment summary tables
    - Model comparison metrics
    - Training configuration snapshots
    """

    def __init__(self, output_dir: str | Path | None = None) -> None:
        """Initialize dashboard generator.

        Args:
            output_dir: Directory for dashboard files. Uses PathConfig.OUTPUT_REPORTS if None.
        """
        if output_dir is None:
            paths = PathConfig()
            self.output_dir = paths.output_reports / "dashboards"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_experiment_report(
        self,
        experiment_name: str,
        config: dict[str, Any],
        results: dict[str, Any],
        filename: str | None = None,
    ) -> Path:
        """Generate a single-experiment HTML report.

        Args:
            experiment_name: Human-readable experiment name.
            config: Training configuration dict (hyperparams, model, dataset).
            results: Evaluation results dict (metrics, timing, etc.).
            filename: Output filename (auto-generated if None).

        Returns:
            Path to saved HTML file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_{experiment_name.lower().replace(' ', '_')}_{timestamp}.html"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics = results.get("metrics", {})
        timing = results.get("timing", {})

        config_rows = "\n".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in self._flatten_dict(config).items()
        )
        metrics_rows = "\n".join(
            f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" if isinstance(v, float) else f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in metrics.items()
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experiment: {experiment_name}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2rem; background: #fafafa; color: #333; }}
    h1 {{ color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 0.5rem; }}
    h2 {{ color: #424242; margin-top: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
    th, td {{ border: 1px solid #e0e0e0; padding: 0.75rem 1rem; text-align: left; }}
    th {{ background: #E3F2FD; font-weight: 600; }}
    tr:nth-child(even) {{ background: #f5f5f5; }}
    .metric-value {{ font-weight: bold; color: #1565C0; }}
    .timestamp {{ color: #9e9e9e; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>Experiment: {experiment_name}</h1>
<p class="timestamp">Generated: {timestamp}</p>

<h2>Configuration</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
{config_rows}
</table>

<h2>Evaluation Results</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{metrics_rows}
</table>

<h2>Timing</h2>
<table>
<tr><th>Stage</th><th>Duration (s)</th></tr>
<tr><td>Inference Time</td><td>{timing.get('inference_time', 'N/A')}</td></tr>
<tr><td>Total Time</td><td>{timing.get('total_time', 'N/A')}</td></tr>
</table>
</body>
</html>"""

        path = self.output_dir / filename
        path.write_text(html, encoding="utf-8")
        logger.info("Dashboard saved: %s", path)
        return path

    def generate_comparison_table(
        self,
        experiments: list[dict[str, Any]],
        filename: str = "comparison.html",
    ) -> Path:
        """Generate a multi-experiment comparison HTML table.

        Args:
            experiments: List of dicts with keys: name, config, results.
            filename: Output filename.

        Returns:
            Path to saved HTML file.
        """
        rows_html = ""
        for exp in experiments:
            name = exp.get("name", "Unknown")
            metrics = exp.get("results", {}).get("metrics", {})
            acc = metrics.get("accuracy", 0.0)
            f1 = metrics.get("f1", 0.0)
            auc = metrics.get("auroc", 0.0)
            precision = metrics.get("precision", 0.0)
            recall = metrics.get("recall", 0.0)
            rows_html += (
                f"<tr>"
                f"<td>{name}</td>"
                f"<td>{acc:.4f}</td>"
                f"<td>{precision:.4f}</td>"
                f"<td>{recall:.4f}</td>"
                f"<td>{f1:.4f}</td>"
                f"<td>{auc:.4f}</td>"
                f"</tr>\n"
            )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Comparison</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2rem; background: #fafafa; color: #333; }}
    h1 {{ color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
    th {{ background: #1565C0; color: white; padding: 0.75rem 1rem; text-align: left; }}
    td {{ border: 1px solid #e0e0e0; padding: 0.75rem 1rem; }}
    tr:nth-child(even) {{ background: #f5f5f5; }}
    .best {{ background: #E8F5E9; font-weight: bold; }}
    .timestamp {{ color: #9e9e9e; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>Model Comparison</h1>
<p class="timestamp">Generated: {timestamp}</p>

<table>
<tr>
    <th>Experiment</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUROC</th>
</tr>
{rows_html}
</table>
</body>
</html>"""

        path = self.output_dir / filename
        path.write_text(html, encoding="utf-8")
        logger.info("Comparison dashboard saved: %s", path)
        return path

    @staticmethod
    def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
        """Flatten nested dict for table display.

        Args:
            d: Nested dictionary.
            parent_key: Prefix for keys.
            sep: Separator between levels.

        Returns:
            Flattened dictionary.
        """
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ExperimentDashboard._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
