"""Evaluation helpers: metrics, confusion matrix, and PR curve artifacts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe for CI/containers; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Compute the standard metrics block used throughout this project."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True
        ),
    }


def save_confusion_matrix_plot(y_true, y_pred, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No Sepsis", "Sepsis"], ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix (held-out fold)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_pr_curve_plot(y_true, y_proba, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name="GB classifier")
    ax.set_title("Precision-Recall Curve (held-out fold)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
