"""Evaluation helpers: metrics, confusion matrix, plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
LABELS = ["negative", "positive"]


def evaluate(pipeline, X_test, y_test, report_dir: Path = REPORT_DIR) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    y_pred = pipeline.predict(X_test)
    try:
        y_score = pipeline.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_score = pipeline.decision_function(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "report": classification_report(
            y_test, y_pred, target_names=LABELS, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    _plot_confusion(np.array(metrics["confusion_matrix"]), report_dir / "confusion_matrix.png")
    _plot_roc(y_test, y_score, metrics["roc_auc"], report_dir / "roc_curve.png")
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nAccuracy: {metrics['accuracy']:.4f}   ROC-AUC: {metrics['roc_auc']:.4f}")
    print(classification_report(y_test, y_pred, target_names=LABELS))
    return metrics


def _plot_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roc(y_test, y_score, auc: float, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_score)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
