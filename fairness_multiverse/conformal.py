"""
This module contains functions to compute conformal prediction sets.
"""

import numpy as np
import pandas as pd


def compute_nc_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    For each calibration example, return 1 - p_true.
    probs: array of shape (n_samples, 2)
    labels: array of shape (n_samples,) with values in {0, 1}
    """
    p_true = probs[np.arange(len(labels)), labels]
    return 1.0 - p_true


def find_threshold(nonconformity: np.ndarray, alpha: float) -> float:
    """
    Return the (1-alpha)-quantile (higher interpolation) of the nc scores.
    """
    return np.quantile(nonconformity, 1 - alpha, method="higher")


def predict_conformal_sets(model, X: pd.DataFrame, q_hat: float) -> list[set[int]]:
    """
    For each row in X, compute the conformal prediction set.
    Returns a list of sets.
    """
    probs = model.predict_proba(X)  # shape (n, 2)
    nonconf_matrix = 1.0 - probs  # shape (n, 2): nc score for label = 0, 1
    # include c whenver nonconf_matrix[i,c] <= q_hat
    return [set(np.where(nc_row <= q_hat)[0]) for nc_row in nonconf_matrix]


def evaluate_sets(pred_sets: list, y_true: pd.Series) -> dict:
    """
    Compute empirical coverage and average set size.
    """
    hits = [y_true.iloc[i] in pred_sets[i] for i in range(len(y_true))]
    coverage = np.mean(hits)
    avg_size = np.mean([len(s) for s in pred_sets])
    return {"coverage": coverage, "avg_size": avg_size}