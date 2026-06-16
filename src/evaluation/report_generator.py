"""
Report generation.

Purpose: Generate comprehensive evaluation reports for thesis Chapter 4.
Responsibilities: Aggregate metrics, format reports, export to multiple formats.
Dependencies: json, pathlib, src.evaluation

Research Traceability:
    Research Objective: Generate reproducible evaluation evidence
    Methodology: Structured reporting of classification metrics
    Implementation: src/evaluation/report_generator.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ReportGenerator:
    """Generate structured evaluation reports.

    Produces:
    - JSON reports for programmatic access
    - Markdown reports for human reading
    - LaTeX-ready tables for thesis inclusion
    """

    def __init__(self, output_dir: str | Path = "outputs/reports") -> None:
        """Initialize report generator.

        Args:
            output_dir: Directory for report output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ReportGenerator initialized: {self.output_dir}")

    def generate_full_report(
        self,
        eval_metrics: dict[str, Any],
        confusion_matrix_result: dict[str, Any] | None = None,
        roc_result: dict[str, Any] | None = None,
        model_name: str = "unknown",
        dataset_name: str = "unknown",
        experiment_config: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Generate a complete evaluation report.

        Args:
            eval_metrics: Metrics from Evaluator.evaluate().
            confusion_matrix_result: Result from ConfusionMatrixAnalyzer.compute().
            roc_result: Result from ROCCurveAnalyzer.compute().
            model_name: Name of the evaluated model.
            dataset_name: Name of the test dataset.
            experiment_config: Training configuration used.

        Returns:
            Dictionary mapping format name to output file path.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{model_name}_{dataset_name}_{timestamp}"

        output_paths: dict[str, str] = {}

        # Build report structure
        report = self._build_report(
            eval_metrics,
            confusion_matrix_result,
            roc_result,
            model_name,
            dataset_name,
            experiment_config,
        )

        # Save JSON report
        json_path = self.output_dir / f"{base_name}.json"
        self._save_json(report, json_path)
        output_paths["json"] = str(json_path)

        # Save Markdown report
        md_path = self.output_dir / f"{base_name}.md"
        self._save_markdown(report, md_path)
        output_paths["markdown"] = str(md_path)

        # Save LaTeX table
        tex_path = self.output_dir / f"{base_name}_table.tex"
        self._save_latex_table(report, tex_path)
        output_paths["latex"] = str(tex_path)

        logger.info(f"Generated reports: {', '.join(output_paths.values())}")
        return output_paths

    def generate_comparison_report(
        self,
        results: dict[str, dict[str, Any]],
        dataset_name: str = "unknown",
    ) -> dict[str, str]:
        """Generate comparison report for multiple models.

        Args:
            results: Dictionary mapping model names to their evaluation metrics.
            dataset_name: Name of the test dataset.

        Returns:
            Dictionary mapping format name to output file path.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"comparison_{dataset_name}_{timestamp}"

        output_paths: dict[str, str] = {}

        # Build comparison structure
        comparison = self._build_comparison(results, dataset_name)

        # Save JSON
        json_path = self.output_dir / f"{base_name}.json"
        self._save_json(comparison, json_path)
        output_paths["json"] = str(json_path)

        # Save Markdown comparison table
        md_path = self.output_dir / f"{base_name}.md"
        self._save_comparison_markdown(comparison, md_path)
        output_paths["markdown"] = str(md_path)

        # Save LaTeX comparison table
        tex_path = self.output_dir / f"{base_name}_table.tex"
        self._save_comparison_latex(comparison, tex_path)
        output_paths["latex"] = str(tex_path)

        logger.info(f"Generated comparison reports: {', '.join(output_paths.values())}")
        return output_paths

    def _build_report(
        self,
        eval_metrics: dict[str, Any],
        confusion_matrix_result: dict[str, Any] | None,
        roc_result: dict[str, Any] | None,
        model_name: str,
        dataset_name: str,
        experiment_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the complete report structure.

        Args:
            eval_metrics: Evaluation metrics.
            confusion_matrix_result: Confusion matrix analysis.
            roc_result: ROC-AUC analysis.
            model_name: Model name.
            dataset_name: Dataset name.
            experiment_config: Experiment configuration.

        Returns:
            Structured report dictionary.
        """
        report: dict[str, Any] = {
            "metadata": {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "timestamp": datetime.now().isoformat(),
                "total_samples": eval_metrics.get("total_samples", 0),
                "evaluation_time": eval_metrics.get("total_time", 0),
                "samples_per_second": eval_metrics.get("samples_per_second", 0),
            },
            "metrics": {
                "accuracy": eval_metrics.get("accuracy", 0),
                "precision": eval_metrics.get("precision", 0),
                "recall": eval_metrics.get("recall", 0),
                "f1": eval_metrics.get("f1", 0),
            },
        }

        # Add ROC-AUC if available
        if roc_result and "auc" in roc_result:
            report["metrics"]["auroc"] = roc_result["auc"]

        # Add per-class metrics
        if "per_class" in eval_metrics:
            report["per_class"] = eval_metrics["per_class"]

        # Add confusion matrix
        if confusion_matrix_result:
            report["confusion_matrix"] = confusion_matrix_result

        # Add ROC details
        if roc_result:
            report["roc_analysis"] = {
                "auc": roc_result.get("auc", 0),
                "optimal_threshold": roc_result.get("optimal_threshold", 0.5),
                "optimal_j_statistic": roc_result.get("optimal_j_statistic", 0),
                "metrics_at_optimal": roc_result.get("metrics_at_optimal", {}),
            }

        # Add configuration
        if experiment_config:
            report["configuration"] = experiment_config

        return report

    def _build_comparison(
        self,
        results: dict[str, dict[str, Any]],
        dataset_name: str,
    ) -> dict[str, Any]:
        """Build comparison report structure.

        Args:
            results: Per-model results.
            dataset_name: Dataset name.

        Returns:
            Comparison report dictionary.
        """
        comparison: dict[str, Any] = {
            "metadata": {
                "dataset_name": dataset_name,
                "num_models": len(results),
                "timestamp": datetime.now().isoformat(),
            },
            "models": {},
            "ranking": {},
        }

        # Collect metrics per model
        metric_keys = ["accuracy", "precision", "recall", "f1", "auroc"]
        for model_name, metrics in results.items():
            comparison["models"][model_name] = {
                key: metrics.get("metrics", {}).get(key, 0)
                for key in metric_keys
            }

        # Rank models by each metric
        for metric_key in metric_keys:
            model_scores = {
                name: data[metric_key]
                for name, data in comparison["models"].items()
            }
            ranked = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            comparison["ranking"][metric_key] = [
                {"model": name, "score": score}
                for name, score in ranked
            ]

        return comparison

    def _save_json(self, data: dict[str, Any], path: Path) -> None:
        """Save data as JSON.

        Args:
            data: Data to save.
            path: Output path.
        """
        def _serialize(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_serialize)

    def _save_markdown(self, report: dict[str, Any], path: Path) -> None:
        """Save report as Markdown.

        Args:
            report: Report dictionary.
            path: Output path.
        """
        meta = report.get("metadata", {})
        metrics = report.get("metrics", {})
        per_class = report.get("per_class", {})

        lines = [
            f"# Evaluation Report: {meta.get('model_name', 'Unknown')}",
            "",
            f"**Dataset:** {meta.get('dataset_name', 'Unknown')}  ",
            f"**Timestamp:** {meta.get('timestamp', 'Unknown')}  ",
            f"**Total Samples:** {meta.get('total_samples', 0)}  ",
            f"**Evaluation Time:** {meta.get('evaluation_time', 0):.2f}s  ",
            "",
            "## Overall Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Accuracy | {metrics.get('accuracy', 0):.4f} |",
            f"| Precision | {metrics.get('precision', 0):.4f} |",
            f"| Recall | {metrics.get('recall', 0):.4f} |",
            f"| F1-Score | {metrics.get('f1', 0):.4f} |",
        ]

        if "auroc" in metrics:
            lines.append(f"| ROC-AUC | {metrics['auroc']:.4f} |")

        lines.append("")

        # Per-class metrics
        if per_class:
            lines.extend([
                "## Per-Class Metrics",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|-------|-----------|--------|-----|---------|",
            ])
            for class_name, class_metrics in per_class.items():
                lines.append(
                    f"| {class_name} | "
                    f"{class_metrics.get('precision', 0):.4f} | "
                    f"{class_metrics.get('recall', 0):.4f} | "
                    f"{class_metrics.get('f1', 0):.4f} | "
                    f"{class_metrics.get('support', 0)} |"
                )
            lines.append("")

        # Confusion matrix
        cm_data = report.get("confusion_matrix", {})
        if "confusion_matrix" in cm_data:
            cm = cm_data["confusion_matrix"]
            class_names = cm_data.get("class_names", ["real", "fake"])

            lines.extend([
                "## Confusion Matrix",
                "",
                "| Predicted → | " + " | ".join(class_names) + " |",
                "|-------------|" + "|".join(["------"] * len(class_names)) + "|",
            ])

            for i, row in enumerate(cm):
                lines.append(
                    f"| **{class_names[i]}** | "
                    + " | ".join(str(int(v)) for v in row)
                    + " |"
                )
            lines.append("")

        # ROC analysis
        roc_data = report.get("roc_analysis", {})
        if roc_data:
            lines.extend([
                "## ROC Analysis",
                "",
                f"- **AUC:** {roc_data.get('auc', 0):.4f}",
                f"- **Optimal Threshold:** {roc_data.get('optimal_threshold', 0.5):.4f}",
                f"- **Youden's J Statistic:** {roc_data.get('optimal_j_statistic', 0):.4f}",
                "",
            ])

            optimal_metrics = roc_data.get("metrics_at_optimal", {})
            if optimal_metrics:
                lines.extend([
                    "### Metrics at Optimal Threshold",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| Accuracy | {optimal_metrics.get('accuracy', 0):.4f} |",
                    f"| Precision | {optimal_metrics.get('precision', 0):.4f} |",
                    f"| Recall | {optimal_metrics.get('recall', 0):.4f} |",
                    f"| F1-Score | {optimal_metrics.get('f1', 0):.4f} |",
                    "",
                ])

        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _save_latex_table(self, report: dict[str, Any], path: Path) -> None:
        """Save LaTeX-formatted table for thesis.

        Args:
            report: Report dictionary.
            path: Output path.
        """
        metrics = report.get("metrics", {})
        meta = report.get("metadata", {})

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Evaluation Results for " + meta.get("model_name", "Model") + r" on " + meta.get("dataset_name", "Dataset") + r"}",
            r"\label{tab:" + f"{meta.get('model_name', 'model').lower()}_results" + r"}",
            r"\begin{tabular}{|l|c|}",
            r"\hline",
            r"\textbf{Metric} & \textbf{Value} \\",
            r"\hline",
            f"Accuracy & {metrics.get('accuracy', 0):.4f} \\\\",
            f"Precision & {metrics.get('precision', 0):.4f} \\\\",
            f"Recall & {metrics.get('recall', 0):.4f} \\\\",
            f"F1-Score & {metrics.get('f1', 0):.4f} \\\\",
        ]

        if "auroc" in metrics:
            lines.append(f"ROC-AUC & {metrics['auroc']:.4f} \\\\")

        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])

        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _save_comparison_markdown(self, comparison: dict[str, Any], path: Path) -> None:
        """Save comparison as Markdown.

        Args:
            comparison: Comparison report dictionary.
            path: Output path.
        """
        meta = comparison.get("metadata", {})
        models = comparison.get("models", {})

        lines = [
            f"# Model Comparison Report",
            "",
            f"**Dataset:** {meta.get('dataset_name', 'Unknown')}  ",
            f"**Models Compared:** {meta.get('num_models', 0)}  ",
            f"**Timestamp:** {meta.get('timestamp', 'Unknown')}  ",
            "",
            "## Performance Comparison",
            "",
            "| Model | Accuracy | Precision | Recall | F1 | AUC |",
            "|-------|----------|-----------|--------|-----|-----|",
        ]

        for model_name, metrics in models.items():
            lines.append(
                f"| {model_name} | "
                f"{metrics.get('accuracy', 0):.4f} | "
                f"{metrics.get('precision', 0):.4f} | "
                f"{metrics.get('recall', 0):.4f} | "
                f"{metrics.get('f1', 0):.4f} | "
                f"{metrics.get('auroc', 0):.4f} |"
            )

        lines.append("")

        # Rankings
        ranking = comparison.get("ranking", {})
        if ranking:
            lines.extend([
                "## Rankings",
                "",
            ])
            for metric, ranked in ranking.items():
                lines.append(f"### By {metric.replace('_', ' ').title()}")
                lines.append("")
                for i, entry in enumerate(ranked, 1):
                    lines.append(f"{i}. **{entry['model']}**: {entry['score']:.4f}")
                lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _save_comparison_latex(self, comparison: dict[str, Any], path: Path) -> None:
        """Save comparison as LaTeX table.

        Args:
            comparison: Comparison report dictionary.
            path: Output path.
        """
        meta = comparison.get("metadata", {})
        models = comparison.get("models", {})

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Model Comparison on " + meta.get("dataset_name", "Dataset") + r"}",
            r"\label{tab:model_comparison}",
            r"\begin{tabular}{|l|c|c|c|c|c|}",
            r"\hline",
            r"\textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{AUC} \\",
            r"\hline",
        ]

        for model_name, metrics in models.items():
            lines.append(
                f"{model_name} & "
                f"{metrics.get('accuracy', 0):.4f} & "
                f"{metrics.get('precision', 0):.4f} & "
                f"{metrics.get('recall', 0):.4f} & "
                f"{metrics.get('f1', 0):.4f} & "
                f"{metrics.get('auroc', 0):.4f} \\\\"
            )

        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])

        with open(path, "w") as f:
            f.write("\n".join(lines))
