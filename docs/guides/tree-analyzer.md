# TreeAnalyzer Guide

The `TreeAnalyzer` class provides tree-level analysis that doesn't require data.

## Basic Usage

```python
from xgboost_interp import TreeAnalyzer

# Load model
analyzer = TreeAnalyzer("your_model.json")

# Print model summary
analyzer.print_model_summary()
```

## Feature Importance

```python
# Combined importance plot (weight, gain, cover)
analyzer.plot_feature_importance_combined(top_n=20)

# Importance distributions
analyzer.plot_feature_importance_distributions(log_scale=True)

# Scatter plot (usage vs gain)
analyzer.plot_feature_importance_scatter()
```

## Tree Structure Analysis

```python
# Tree depth distribution
analyzer.plot_tree_depth_histogram()

# Cumulative gain across trees
analyzer.plot_cumulative_gain()

# Prediction shift across trees
analyzer.plot_cumulative_prediction_shift()
```

## Feature Co-occurrence

```python
# Tree-level co-occurrence
analyzer.plot_tree_level_feature_cooccurrence()

# Path-level co-occurrence
analyzer.plot_path_level_feature_cooccurrence()

# Sequential dependency
analyzer.plot_sequential_feature_dependency()
```

## API Reference

### Constructor

```python
TreeAnalyzer(model_path: str, save_dir: Optional[str] = None)
```

- `model_path`: Path to XGBoost model JSON file
- `save_dir`: Directory for saving plots (default: derived from model name)

### Key Methods

| Method | Description |
|--------|-------------|
| `print_model_summary()` | Display model metadata |
| `plot_feature_importance_combined()` | Normalized importance by weight, gain, cover |
| `plot_feature_importance_distributions()` | Boxplots of importance distributions |
| `plot_tree_depth_histogram()` | Distribution of tree depths |
| `plot_cumulative_gain()` | Cumulative loss reduction |
| `plot_tree_level_feature_cooccurrence()` | Features appearing in same tree |
| `plot_path_level_feature_cooccurrence()` | Features on same decision paths |

