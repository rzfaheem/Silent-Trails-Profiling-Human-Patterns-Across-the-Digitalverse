"""
Evaluation Metrics module.

Computes all metrics needed for deepfake detection evaluation:
- AUC-ROC (primary metric)
- Accuracy, Precision, Recall, F1
- Equal Error Rate (EER)
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve
)


def compute_eer(labels, scores):
    """
    Compute Equal Error Rate (EER).

    EER is the point where False Positive Rate == False Negative Rate.
    Lower EER = better model.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    # Find where FPR ≈ FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]

    return eer, eer_threshold


def compute_all_metrics(labels, probabilities, threshold=0.5):
    """
    Compute all evaluation metrics.

    Args:
        labels: (N,) ground truth (0=real, 1=fake)
        probabilities: (N,) model output probabilities [0, 1]
        threshold: decision threshold for binary predictions

    Returns:
        dict with all metrics
    """
    labels = np.array(labels)
    probabilities = np.array(probabilities)
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "auc": roc_auc_score(labels, probabilities),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }

    eer, eer_threshold = compute_eer(labels, probabilities)
    metrics["eer"] = eer
    metrics["eer_threshold"] = eer_threshold

    return metrics


def print_metrics(metrics, prefix=""):
    """Pretty print metrics."""
    print(f"\n{'─' * 40}")
    if prefix:
        print(f"  {prefix}")
        print(f"{'─' * 40}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  EER:       {metrics['eer']:.4f}")
    print(f"{'─' * 40}\n")
