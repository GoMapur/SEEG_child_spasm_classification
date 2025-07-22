# metrics.py
"""Metric functions for EEG classification."""
import numpy as np
from sklearn.metrics import recall_score, accuracy_score

def compute_metric(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Compute recall, specificity, and accuracy metrics."""
    specificity, recallTP = recall_score(labels, predictions, average=None)
    accuracy = accuracy_score(labels, predictions)
    return {"recall": recallTP, "specificity": specificity, "accuracy": accuracy} 