# Quick Start Guide

## 🚀 Ready-to-Run Examples with Real Data

We've created complete examples that download data, train models, and demonstrate all interpretability features!

### Option 1: Interactive Runner (Recommended)

```bash
cd /Users/gkocher/Desktop/xgboost-interp

# Install the package
pip install -e .

# Run the interactive example chooser
python run_examples.py
```

### Option 2: Direct Execution

```bash
# California Housing (Regression) - 100 trees
python xgboost_interp/examples/sklearn_dataset_example.py

# Iris Classification - 50 trees  
python xgboost_interp/examples/iris_classification_example.py
```

## 📊 What You'll Get

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

## 🎯 Generated Visualizations

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

## 🔧 Using Your Own Models

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

## 📋 Requirements

The examples automatically handle:
- ✅ Data downloading (sklearn datasets)
- ✅ Model training with sensible hyperparameters
- ✅ JSON model saving
- ✅ Comprehensive interpretability analysis

Just install and run!

```bash
pip install -e .
python run_examples.py
```

## 🎉 Expected Runtime

- **Iris example**: ~30 seconds (small dataset)
- **Housing example**: ~2-3 minutes (larger dataset)

Both include complete model training + full interpretability analysis!
