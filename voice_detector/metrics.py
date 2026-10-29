"""Evaluation metrics. Label 1 = AI-generated, 0 = human; higher score = more likely AI."""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return (EER, threshold at which false-accept and false-reject rates are equal)."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2), float(thresholds[i])


def summarize(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels)
    preds = np.asarray(scores) >= threshold
    out = {
        "n": int(len(labels)),
        "n_human": int((labels == 0).sum()),
        "n_ai": int((labels == 1).sum()),
        "accuracy": float((preds == labels).mean()),
        "human_recall": float((~preds[labels == 0]).mean()) if (labels == 0).any() else None,
        "ai_recall": float(preds[labels == 1].mean()) if (labels == 1).any() else None,
    }
    if 0 < labels.sum() < len(labels):
        out["eer"] = equal_error_rate(labels, scores)[0]
        out["auc"] = float(roc_auc_score(labels, scores))
    return out
