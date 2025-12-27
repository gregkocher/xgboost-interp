"""
Model performance metrics utilities.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, objective: str) -> Dict[str, float]:
    """
    Compute model performance metrics based on the objective.
    
    Args:
        y_true: Ground truth labels/values
        y_pred: Model predictions (probabilities for classification, values for regression)
        objective: XGBoost objective string (e.g., 'binary:logistic', 'multi:softmax', 'reg:squarederror')
    
    Returns:
        Dictionary of metric names to values
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, precision_score, recall_score,
        f1_score, log_loss, brier_score_loss, mean_squared_error,
        precision_recall_curve
    )
    from sklearn.preprocessing import label_binarize
    
    objective_name = objective.get("name", "") if isinstance(objective, dict) else str(objective)
    
    # Determine model type
    is_binary = "binary:" in objective_name
    is_multiclass = "multi:" in objective_name
    is_regression = "reg:" in objective_name or "squarederror" in objective_name
    
    if is_regression:
        return _compute_regression_metrics(y_true, y_pred)
    elif is_binary:
        return _compute_binary_metrics(y_true, y_pred)
    elif is_multiclass:
        return _compute_multiclass_metrics(y_true, y_pred)
    else:
        raise NotImplementedError(f"Metrics not implemented for objective: {objective_name}")


def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute binary classification metrics."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, precision_score, recall_score,
        f1_score, log_loss, brier_score_loss, precision_recall_curve
    )
    
    # y_pred should be probabilities for the positive class
    y_proba = y_pred if y_pred.max() <= 1.0 else 1.0 / (1.0 + np.exp(-y_pred))
    
    # Find optimal threshold via F1
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred_binary = (y_proba >= best_threshold).astype(int)
    
    # Calibration metrics (binned, n_bins=10)
    avg_bias, mape, smape = _compute_calibration_metrics(y_true, y_proba, n_bins=10)
    
    return {
        "AUC_ROC": roc_auc_score(y_true, y_proba),
        "AUC_PR": average_precision_score(y_true, y_proba),
        "Precision_best": precision_score(y_true, y_pred_binary, zero_division=0),
        "Recall_best": recall_score(y_true, y_pred_binary, zero_division=0),
        "F1_best": f1_score(y_true, y_pred_binary, zero_division=0),
        "Best_threshold": best_threshold,
        "Logloss": log_loss(y_true, y_proba),
        "Brier_score": brier_score_loss(y_true, y_proba),
        "Avg_bias": avg_bias,
        "MAPE_k10": mape,
        "SMAPE_k10": smape,
    }


def _compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute multi-class classification metrics (macro-averaged)."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, precision_score, recall_score,
        f1_score, log_loss
    )
    from sklearn.preprocessing import label_binarize
    
    n_classes = len(np.unique(y_true))
    
    # Handle probability matrix vs class predictions
    if y_pred.ndim == 2:
        y_proba = y_pred
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred.astype(int)
        y_proba = None
    
    metrics = {
        "Precision_macro": precision_score(y_true, y_pred_classes, average='macro', zero_division=0),
        "Recall_macro": recall_score(y_true, y_pred_classes, average='macro', zero_division=0),
        "F1_macro": f1_score(y_true, y_pred_classes, average='macro', zero_division=0),
    }
    
    # AUC metrics require probability matrix
    if y_proba is not None:
        y_true_onehot = label_binarize(y_true, classes=range(n_classes))
        
        try:
            metrics["AUC_ROC_macro"] = roc_auc_score(
                y_true_onehot, y_proba, average='macro', multi_class='ovr'
            )
        except ValueError:
            metrics["AUC_ROC_macro"] = np.nan
        
        try:
            metrics["AUC_PR_macro"] = average_precision_score(
                y_true_onehot, y_proba, average='macro'
            )
        except ValueError:
            metrics["AUC_PR_macro"] = np.nan
        
        try:
            metrics["Logloss"] = log_loss(y_true, y_proba)
        except ValueError:
            metrics["Logloss"] = np.nan
    
    return metrics


def _compute_calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> tuple:
    """
    Compute binned calibration metrics.
    
    Args:
        y_true: Binary labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
    
    Returns:
        (avg_bias, mape, smape) tuple
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges[1:-1])
    
    biases = []
    apes = []
    sapes = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        
        actual = y_true[mask].mean()
        predicted = y_proba[mask].mean()
        
        bias = predicted - actual
        biases.append(bias)
        
        if actual > 0:
            apes.append(abs(bias) / actual)
        
        denom = abs(actual) + abs(predicted)
        if denom > 0:
            sapes.append(2 * abs(bias) / denom)
    
    avg_bias = np.mean(biases) if biases else 0.0
    mape = np.mean(apes) if apes else 0.0
    smape = np.mean(sapes) if sapes else 0.0
    
    return avg_bias, mape, smape

