"""Model evaluation and performance analysis.

Provides:
- Evaluator: Full evaluation pipeline for trained models
- ConfusionMatrixAnalyzer: Confusion matrix computation and error analysis
- ROCCurveAnalyzer: ROC curve and AUC analysis
- ReportGenerator: Multi-format report generation (JSON, Markdown, LaTeX)
"""

from src.evaluation.evaluate import Evaluator
from src.evaluation.confusion_matrix import ConfusionMatrixAnalyzer
from src.evaluation.roc_auc import ROCCurveAnalyzer
from src.evaluation.report_generator import ReportGenerator

__all__ = [
    "Evaluator",
    "ConfusionMatrixAnalyzer",
    "ROCCurveAnalyzer",
    "ReportGenerator",
]
