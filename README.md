<p align="center">
  <img src="docs/assets/images/xgboost-interp-logo.png" alt="xgboost-interp logo" width="400">
</p>

<p align="center">
  <a href="https://www.repostatus.org/#active"><img src="https://www.repostatus.org/badges/latest/active.svg" alt="Project Status: Active"></a>
  <a href="https://github.com/gregkocher/xgboost-interp/blob/main/LICENSE"><img src="https://img.shields.io/github/license/gregkocher/xgboost-interp" alt="License"></a>
  <a href="https://github.com/gregkocher/xgboost-interp/actions/workflows/tests.yml"><img src="https://github.com/gregkocher/xgboost-interp/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="#"><img src="https://img.shields.io/badge/DOI-TBD-lightgrey" alt="DOI"></a>
</p>

# XGBoost Interpretability Package

A comprehensive toolkit for interpreting and analyzing XGBoost models. This package provides both data agnostic and data-dependent model analysis, including XGBoost tree topology analysis; feature importance visualizations; Partial Dependence Plots (PDP), Individual Conditional Expectation (ICE) plots, and Accumulated Local Effects (ALE) plots; various SHAP analyses; and interactive tree exploration.

## Features

### Tree-Level Analysis (No Data Required)
- **Feature Importance**: Weight, gain, and cover-based importance metrics
- **Tree Structure**: Depth analysis, cumulative gain tracking
- **Feature Interactions**: Co-occurrence analysis at tree and path levels
- **Visualization**: Heatmaps, distributions, and summary statistics

### Model Analysis with Data
- **Partial Dependence Plots (PDP)**: Individual Conditional Expectation (ICE) curves overlaid with all-samples average (PDP)
- **Accumulated Local Effects (ALE)**: Unbiased feature effect analysis accounting for feature correlations
- **SHAP Analysis**: SHapley Additive exPlanations for model-agnostic feature importance
- **Prediction Analysis**: Score evolution across tree ensembles
- **Marginal Impact**: Feature-specific prediction changes

### Interactive Visualizations
- **Tree Explorer**: Interactive tree structure visualization with Plotly, showing all split features and split thresholds


## Requirements

- Python 3.10+
- matplotlib >= 3.3.0
- networkx >= 2.5.0
- numpy >= 1.19.0
- pandas >= 1.2.0
- plotly >= 5.0.0
- pyALE >= 0.2.0
- pyarrow >= 10.0.0
- scikit-learn >= 0.24.0
- scipy >= 1.6.0
- seaborn >= 0.11.0
- shap >= 0.40.0
- xgboost >= 1.4.0


## Setup

```bash
git clone https://github.com/gregkocher/xgboost-interp.git
cd xgboost-interp
uv sync
source .venv/bin/activate
```


## Quick Start

```bash
python3 xgboost_interp/examples/user_model_complete_analysis.py YOUR_MODEL.json PATH/TO/YOUR/PARQUET/DATA_DIR/
```


## Examples

Example scripts are located in `xgboost_interp/examples/`:

- `california_housing_example.py`: Complete example with California Housing dataset (regression)
- `iris_classification_example.py`: Classification example with Iris dataset
- `synthetic_imbalanced_classification_example.py`: Synthetic data with known ground-truth relationships for validation
- `user_model_complete_analysis.py`: Run ALL analysis functions on your own model
- `basic_analysis.py`: Tree-level analysis without data (requires your model)
- `advanced_analysis.py`: Full model analysis with data and interactions (requires your model)

### Running Examples

```bash
# Run individual examples
python3 xgboost_interp/examples/california_housing_example.py
python3 xgboost_interp/examples/iris_classification_example.py
python3 xgboost_interp/examples/synthetic_imbalanced_classification_example.py
```

The examples are self-contained and include:
- Data loading and preprocessing
- XGBoost model training (100 trees for housing, 50 for iris, 3000 for synthetic)
- Model saving as JSON
- Complete interpretability analysis

### Synthetic Imbalanced Classification Example

The synthetic example is designed for validating interpretability tools against known ground-truth:

- **100,000 samples** with 10% positive rate (imbalanced binary classification)
- **39 features** with known effects: Normal (IID and correlated), Categorical (15-200 cardinality), Binary, Uniform (linear and quadratic), Trigonometric (periodic), and Noise
- **3,000-tree model** for comprehensive early exit analysis
- Feature names encode their properties (e.g., `norm_iid_pos_strong`, `unif_quad_neg`, `noise_cat`)

**Expected validation results:**
- Strong effect features have high importance
- Noise features have near-zero importance
- Quadratic features show U-shaped PDP curves
- Trigonometric features show periodic wave patterns in PDP
- Categorical features show step patterns in PDP

See `examples/synthetic_imbalanced_classification/SYNTHETIC_MODEL_README.md` for full feature documentation.

### Complete Analysis of Your Own Model

The `user_model_complete_analysis.py` script runs **ALL** available analysis and plotting functions:

```bash
# Analyze your own model
python3 xgboost_interp/examples/user_model_complete_analysis.py your_model.json
python3 xgboost_interp/examples/user_model_complete_analysis.py your_model.json data_dir/

# Multi-class: analyze specific class
python3 xgboost_interp/examples/user_model_complete_analysis.py model.json data_dir/ --target-class 0
```

This example demonstrates:
- ✅ All 15 tree-level analysis functions
- ✅ Partial dependence plots for ALL features
- ✅ Marginal impact analysis for ALL features
- ✅ Prediction evolution across trees
- ✅ Interactive tree visualizations
- ✅ Comprehensive summary report


## API Reference

### TreeAnalyzer

The main class for tree-level analysis that doesn't require data.

**Key Methods:**
- `print_model_summary()`: Display model metadata and structure
- `plot_feature_importance_combined()`: Normalized importance by weight, gain, cover
- `plot_feature_importance_distributions()`: Boxplots of importance distributions
- `plot_tree_depth_histogram()`: Distribution of tree depths
- `plot_cumulative_gain()`: Cumulative loss reduction across trees
- `plot_feature_usage_heatmap()`: Feature co-occurrence patterns
- `plot_gain_stats_per_tree()`: Gain distribution across trees
- `compute_tree_level_feature_cooccurrence()`: Compute features appearing in same tree
- `compute_path_level_feature_cooccurrence()`: Compute features on same decision paths
- `compute_sequential_feature_dependency()`: Compute parent→child feature dependencies
- `plot_tree_level_feature_cooccurrence()`: Plot tree-level co-occurrence heatmap
- `plot_path_level_feature_cooccurrence()`: Plot path-level co-occurrence heatmap
- `plot_sequential_feature_dependency()`: Plot sequential feature co-occurrence heatmap

### ModelAnalyzer

Extended analysis requiring actual data examples.

**Key Methods:**
- `load_data_from_parquets()`: Load data from parquet files
- `load_xgb_model()`: Load XGBoost model for predictions
- `plot_partial_dependence()`: PDP with ICE curves
- `plot_ale()`: Accumulated Local Effects plots
- `plot_scores_across_trees()`: Prediction evolution analysis
- `plot_marginal_impact_univariate()`: Feature-specific impact analysis
- `analyze_early_exit_performance()`: Early exit metrics (inversion rate, MSE, Kendall-Tau, Spearman)
- `evaluate_model_performance()`: Compute and save model performance metrics
- `generate_calibration_curves()`: Calibration curves for binary classification


## Visualization Gallery

### Tree Structure Analysis

#### 1. Cumulative Gain
Cumulative loss reduction across the tree ensemble.

![Cumulative Gain](docs/assets/images/cumulative_gain.png)
*California Housing dataset - shows how model improves with each tree*

#### 2. Feature Importance Scatter Plot
Scatter plot showing feature usage vs gain, with bubble size representing average cover.

![Feature Importance Scatter](docs/assets/images/feature_importance_scatter.png)
*California Housing dataset - bubble chart revealing the relationship between feature usage frequency, gain, and cover*

#### 3. Feature Importance Combined
Combined view of feature importance across weight, gain, and cover metrics.

![Feature Importance](docs/assets/images/feature_importance_combined.png)
*California Housing dataset - shows MedInc (median income) as the most important feature*

#### 4. Marginal Impact
Feature-specific prediction changes across all splits in the model. Shows how the model's prediction changes in different ranges of a feature based on the tree structure alone (no data required). The step function displays the marginal prediction change at each threshold, with color intensity indicating the magnitude of impact.

![Marginal Impact](docs/assets/images/marginal_impact_petal_length.png)
*Iris dataset - marginal impact of petal length on class 2 probability. Strong positive impact in the 3-4.5cm range (darker green) indicates higher probability for class 2 (virginica). Negative impact below 3cm (red) suggests lower probability. The step function shows exact prediction changes at each split threshold across all 150 trees.*

#### 5. Partial Dependence Plot (PDP)
Shows how predictions change as a feature varies, with ICE curves for individual samples. Uses hybrid grid (100 uniform + 100 percentile points) for comprehensive coverage of continuous features.

![Partial Dependence Plot](docs/assets/images/PDP_MedInc.png)
*California Housing dataset - MedInc (median income) shows strong positive relationship with house value*

#### Interactive Tree Visualization
Interactive tree structure exploration with hover information for splits and leaf values.

![Tree 1](docs/assets/images/Iris-Tree_1.png)
*Iris dataset - Tree 1 showing decision structure with split conditions and gains*

![Tree 4](docs/assets/images/Iris-Tree_4.png)
*Iris dataset - Tree 4 demonstrating deeper splits and leaf predictions*

#### Feature Usage Heatmap
Heatmap showing which features are used together in trees.

![Feature Usage Heatmap](docs/assets/images/feature_usage_heatmap.png)
*California Housing dataset - reveals feature co-occurrence patterns*

#### Tree-Level Feature Co-occurrence
Symmetric matrix showing how often pairs of features appear in the same tree.

![Tree-Level Co-occurrence](docs/assets/images/feature_cooccurrence_tree_level.png)
*California Housing dataset - darker colors indicate features frequently used together in trees*

#### Path-Level Feature Co-occurrence
Symmetric matrix showing how often pairs of features appear on the same root-to-leaf decision path (log scale).

![Path-Level Co-occurrence](docs/assets/images/feature_cooccurrence_path_level.png)
*California Housing dataset - reveals tighter feature interactions along decision paths*

#### Sequential Feature Co-occurrence
Asymmetric matrix showing conditional probabilities: when a feature (row) splits, what's the probability that another feature (column) is the immediate next split? This reveals directional parent→child feature dependencies in the tree structure.

![Sequential Feature Co-occurrence](docs/assets/images/feature_cooccurrence_sequential.png)

*California Housing dataset - shows which features tend to follow others in decision paths. High values indicate strong sequential dependencies (e.g., after splitting on feature A, the model frequently splits on feature B next)*

#### Accumulated Local Effects (ALE) Plot
Unbiased feature effect visualization that accounts for feature correlations. ALE plots show the marginal effect of a feature on predictions while properly handling correlated features, making them superior to PDPs when features are correlated.

![ALE Plot](docs/assets/images/ALE_HouseAge.png)
*California Housing dataset - ALE plot for HouseAge showing the local effect on house value predictions. The shaded region indicates 95% confidence intervals. The plot reveals a non-linear relationship where house age has varying impacts on value across different age ranges.*

#### SHAP Analysis
SHAP (SHapley Additive exPlanations) provides model-agnostic explanations by computing the contribution of each feature to individual predictions.

**SHAP Summary Beeswarm Plot**: Shows feature importance and effect direction across all samples.

![SHAP Summary](docs/assets/images/summary_beeswarm.png)
*California Housing dataset - each dot represents a sample, colored by feature value (red=high, blue=low). Position on x-axis shows impact on prediction. MedInc (median income) has the strongest effect, with high values consistently pushing predictions higher.*

**SHAP Waterfall Plot**: Explains individual predictions by showing how each feature pushes the prediction from the base value.

![SHAP Waterfall](docs/assets/images/waterfall_sample_2.png)
*California Housing dataset - waterfall plot for sample 2. Starting from the base value (E[f(x)] = 2.07), features like MedInc (+0.47) and Latitude (+0.37) push the prediction higher, while AveOccup (-0.04) slightly reduces it. Final prediction: f(x) = 2.99.*

#### Feature Gain Distribution
Distribution of gain values across all splits for each feature.

![Feature Gain Distribution](docs/assets/images/feature_gain_distribution.png)
*Iris dataset - boxplot showing gain distributions per feature*

#### Tree Depth Distribution
Histogram showing the distribution of tree depths in the ensemble.

![Tree Depth Histogram](docs/assets/images/tree_depth_histogram.png)
*Iris dataset - most trees have depths between 2-5*

#### Gain Statistics Per Tree
Box plots showing gain statistics for each tree in the ensemble.

![Gain Stats Per Tree](docs/assets/images/gain_stats_per_tree.png)
*California Housing dataset - gain distribution across all 100 trees*

#### Prediction Statistics Per Tree
Statistical analysis of leaf predictions across the ensemble.

![Prediction Stats Per Tree](docs/assets/images/prediction_stats_per_tree.png)
*California Housing dataset - mean, median, and standard deviation of predictions per tree*

#### Prediction Evolution Across Trees
Shows how predicted probabilities change as more trees are added to the ensemble.

![Scores Across Trees](docs/assets/images/scores_across_trees.png)
*Iris dataset - class probability evolution showing model convergence across the ensemble*

#### Early Exit Analysis
Scatter plots comparing predictions at different tree stopping points (early exit) against final model predictions. Each subplot shows how well early-stopped predictions correlate with full ensemble predictions, with MSE displayed. Useful for understanding when additional trees stop providing significant improvements.

![Early Exit Scatter](docs/assets/images/early_exit_scatter.png)
*Synthetic classification dataset (3000 trees) - comparing early exit predictions at quantile points (1, 600, 1200, 1800, 2400 trees) vs final predictions. High correlation at later exit points indicates model convergence.*

**Early Exit Performance Metrics:**

| Tree Index | Inversion Rate | MSE | Kendall-Tau | Spearman |
|------------|----------------|-----|-------------|----------|
| 1 | 17.58% | 39.81 | 0.3221 | 0.4613 |
| 600 | 4.07% | 13.50 | 0.8338 | 0.9621 |
| 1200 | 2.81% | 6.38 | 0.8843 | 0.9813 |
| 1800 | 2.00% | 2.31 | 0.9182 | 0.9905 |
| 2400 | 1.26% | 0.50 | 0.9494 | 0.9964 |
| 3000 | 0.00% | 0.00 | 1.0000 | 1.0000 |

*Metrics comparing early exit predictions to final model (3000 trees). Lower inversion rate and MSE, higher Kendall-Tau and Spearman indicate better agreement with final predictions.*


## Testing

Run all tests:
```bash
uv run pytest tests/ -v
```

Run individual tests:
```bash
uv run pytest tests/test_examples.py::test_iris_example -v
uv run pytest tests/test_examples.py::test_california_housing_example -v
uv run pytest tests/test_examples.py::test_synthetic_imbalanced_classification_example -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{xgboost_interp,
  title={XGBoost Interpretability Package},
  author={Greg Kocher},
  year={2025},
  url={https://github.com/gregkocher/xgboost-interp}
}
```

## Changelog

### v0.1.0 (2025-10-05)
- Initial release
- Tree-level analysis functionality
- Model analysis with data support
- Interactive visualizations
- Comprehensive plotting utilities
