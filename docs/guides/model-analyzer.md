# ModelAnalyzer Guide

The `ModelAnalyzer` class provides data-dependent analysis of XGBoost models.

## Basic Usage

```python
from xgboost_interp import TreeAnalyzer, ModelAnalyzer

# First create TreeAnalyzer
tree_analyzer = TreeAnalyzer("your_model.json")

# Then create ModelAnalyzer
model_analyzer = ModelAnalyzer(tree_analyzer)

# Load data
model_analyzer.load_data_from_parquets("path/to/data/")
model_analyzer.load_xgb_model()
```

## Partial Dependence Plots

```python
# PDP with ICE curves
model_analyzer.plot_partial_dependence(
    feature_name="feature_name",
    n_curves=1000
)
```

## Accumulated Local Effects (ALE)

```python
# ALE plot (handles correlated features better than PDP)
model_analyzer.plot_ale(
    feature_name="feature_name",
    grid_size=100,
    include_CI=True
)
```

## Marginal Impact Analysis

```python
# Feature-specific prediction changes
model_analyzer.plot_marginal_impact_univariate(
    feature_name="feature_name",
    scale="linear"
)
```

## Prediction Evolution

```python
# How predictions change across trees
model_analyzer.plot_scores_across_trees(
    tree_indices=[10, 25, 50, 75, 100],
    n_records=1000
)
```

## Early Exit Analysis

```python
# Analyze early exit performance metrics
model_analyzer.analyze_early_exit_performance(
    n_records=5000,
    n_detailed_curves=1000
)
```

This generates:
- Inversion rate, MSE, Kendall-Tau, and Spearman metrics
- Scatter plots comparing early exit vs final predictions
- Detailed score evolution across all trees

## Model Performance Metrics

```python
# Evaluate model performance (requires y_true and y_pred)
metrics = model_analyzer.evaluate_model_performance(y_test, y_pred_proba)

# Generate calibration curves
model_analyzer.generate_calibration_curves(y_test, y_pred_proba, X=X_test, n_bins=10)
```

## Multi-class Models

For multi-class classification, specify the target class:

```python
# Analyze class 0
model_analyzer = ModelAnalyzer(tree_analyzer, target_class=0)
```

## API Reference

### Constructor

```python
ModelAnalyzer(tree_analyzer: TreeAnalyzer, target_class: int = 0)
```

- `tree_analyzer`: Initialized TreeAnalyzer instance
- `target_class`: Class index for multi-class models

### Key Methods

| Method | Description |
|--------|-------------|
| `load_data_from_parquets()` | Load data from parquet files |
| `load_xgb_model()` | Load XGBoost model for predictions |
| `plot_partial_dependence()` | PDP with ICE curves |
| `plot_ale()` | Accumulated Local Effects plots |
| `plot_marginal_impact_univariate()` | Feature-specific impact |
| `plot_scores_across_trees()` | Prediction evolution |
| `analyze_early_exit_performance()` | Early exit metrics (inversion rate, MSE, Kendall-Tau, Spearman) |
| `evaluate_model_performance()` | Compute and save model performance metrics |
| `generate_calibration_curves()` | Calibration curves for binary classification |

