# Quick Start Guide

## Ready-to-Run Examples with Real Data

Complete examples that download data, train models, and demonstrate all interpretability features.

## Complete Analysis of Your Own Model

**Already have a trained XGBoost model?** Use the comprehensive analysis script:

```bash
# Run ALL analysis functions on your model (no data needed) - generates ~15 plots
python xgboost_interp/examples/user_model_complete_analysis.py your_model.json

# Complete analysis with data - generates PDPs and marginal impacts for ALL features
python xgboost_interp/examples/user_model_complete_analysis.py your_model.json data_directory/

# Multi-class models: analyze specific class
python xgboost_interp/examples/user_model_complete_analysis.py your_model.json data_dir/ --target-class 0
```

**What it does:**
- Runs all 15 tree-level analysis functions
- Generates partial dependence plots for ALL features
- Creates marginal impact visualizations for ALL features
- Shows prediction evolution across trees
- Generates interactive tree visualizations
- Produces comprehensive summary report

**Requirements:**
- XGBoost model saved as JSON (use `model.save_model("model.json")`)
- (Optional) Data in parquet format for data-dependent analysis

### Option 1: Interactive Runner (Recommended)

```bash
cd xgboost-interp

# Install the package
pip install -e .

# Run the interactive example chooser
python run_examples.py
```

### Option 2: Direct Execution

```bash
# California Housing (Regression) - 100 trees
python xgboost_interp/examples/california_housing_example.py

# Iris Classification - 50 trees  
python xgboost_interp/examples/iris_classification_example.py

# Synthetic Imbalanced Classification - 3000 trees (for validation)
python xgboost_interp/examples/synthetic_imbalanced_classification_example.py
```

## What You'll Get

### California Housing Example
- **Dataset**: 20,640 samples, 8 features (median income, house age, etc.)
- **Model**: XGBoost Regressor (100 trees, depth 6)
- **Output**: 
  - `california_housing_xgb.json` (trained model)
  - `california_housing_xgb/` (15+ visualization plots)
  - `housing_data/` (processed dataset)

### Iris Classification Example  
- **Dataset**: 150 samples, 4 features (petal/sepal measurements)
- **Model**: XGBoost Classifier (50 trees, depth 4)
- **Output**:
  - `iris_xgb.json` (trained model)
  - `iris_xgb/` (10+ visualization plots)
  - `iris_data/` (processed dataset)

### Synthetic Imbalanced Classification Example
- **Dataset**: 100,000 samples, 39 features with known ground-truth effects
- **Model**: XGBoost Classifier (3000 trees, depth 6)
- **Purpose**: Validate interpretability tools against known feature-target relationships
- **Output**:
  - `synthetic_imbalanced_classification_xgb.json` (trained model)
  - `output/` (comprehensive analysis including early exit metrics)
  - `SYNTHETIC_MODEL_README.md` (detailed feature documentation)

## Generated Visualizations

Both examples create comprehensive analysis including:

### Tree Structure Analysis
- Feature importance (weight, gain, cover)
- Tree depth distribution
- Cumulative gain over trees
- Feature usage heatmaps

### Data-Dependent Analysis
- Partial Dependence Plots (PDP) for all features
- Marginal impact analysis
- Interactive tree visualizations (if Plotly available)
- Prediction evolution across trees

### Advanced Plots
- Feature co-occurrence analysis
- Split depth distributions
- Gain/prediction statistics by tree and depth
- Feature split impact analysis

## Using Your Own Models

After running the examples, you can analyze your own models:

```python
from xgboost_interp import TreeAnalyzer, ModelAnalyzer

# Tree-only analysis (no data needed)
tree_analyzer = TreeAnalyzer("your_model.json")
tree_analyzer.print_model_summary()
tree_analyzer.plot_feature_importance_combined()

# Analysis with your data
model_analyzer = ModelAnalyzer(tree_analyzer)
model_analyzer.load_data_from_parquets("your_data_dir/")
model_analyzer.load_xgb_model()
model_analyzer.plot_partial_dependence("your_feature")
```

## Requirements

The examples automatically handle:
- Data downloading (sklearn datasets)
- Model training with sensible hyperparameters
- JSON model saving
- Comprehensive interpretability analysis


```bash
pip install -e .
python run_examples.py
```

