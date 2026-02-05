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


def _sanitize_filename(name: str) -> str:
    """Convert name to safe filename (remove punctuation, replace whitespace with _)."""
    import re
    safe = re.sub(r'[^\w\s-]', '', name)
    safe = re.sub(r'\s+', '_', safe)
    return safe


def plot_calibration_curve(
    sort_by_values: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    sort_by_name: str,
    save_dir: str,
    n_bins: int = 10
) -> str:
    """
    Plot calibration/reliability curve.
    
    Args:
        sort_by_values: Values used to sort and bin data
        x_values: Values for x-axis (mean per bin, typically y_pred)
        y_values: Values for y-axis (mean per bin, typically y_true)
        sort_by_name: Name of sort-by column (used in filename)
        save_dir: Directory to save plot
        n_bins: Number of bins
    
    Returns:
        Path to saved figure
    """
    import os
    import matplotlib.pyplot as plt
    
    # Convert to numpy arrays if needed
    sort_by_values = np.asarray(sort_by_values)
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    
    # Sort all arrays by sort_by_values
    sort_idx = np.argsort(sort_by_values)
    x_sorted = x_values[sort_idx]
    y_sorted = y_values[sort_idx]
    
    # Split into n_bins equal-sized bins
    bin_size = len(x_sorted) // n_bins
    x_means = []
    y_means = []
    
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(x_sorted)
        if end > start:
            x_means.append(np.mean(x_sorted[start:end]))
            y_means.append(np.mean(y_sorted[start:end]))
    
    x_means = np.array(x_means)
    y_means = np.array(y_means)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # y=x reference line
    min_val = min(x_means.min(), y_means.min())
    max_val = max(x_means.max(), y_means.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='Perfect')
    
    # Color gradient from coolwarm colormap (blue=low, red=high)
    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(x_means) - 1)) for i in range(len(x_means))]
    
    # Plot line segments with gradient colors
    for i in range(len(x_means) - 1):
        ax.plot(x_means[i:i+2], y_means[i:i+2], '-', color=colors[i], linewidth=2)
    
    # Plot markers with gradient colors
    for i, (x, y) in enumerate(zip(x_means, y_means)):
        ax.plot(x, y, 'o', color=colors[i], markersize=8)
    
    # Add invisible line for legend
    ax.plot([], [], 'o-', color='gray', markersize=6, linewidth=2, label='Calibration')
    
    # Add bin labels near each marker
    for i, (x, y) in enumerate(zip(x_means, y_means)):
        bin_num = i + 1
        if bin_num == 1:
            label = "1 (lowest)"
        elif bin_num == len(x_means):
            label = f"{bin_num} (highest)"
        else:
            label = str(bin_num)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=7, color='black')
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Mean Actual (Fraction of Positives)')
    ax.set_title(f'Calibration Curve - Sorted by {sort_by_name} ({n_bins} bins)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    calib_dir = os.path.join(save_dir, 'calibration_curves')
    os.makedirs(calib_dir, exist_ok=True)
    
    safe_name = _sanitize_filename(sort_by_name)
    filepath = os.path.join(calib_dir, f'calibration_curve_{n_bins}bins_sortby_{safe_name}.png')
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

