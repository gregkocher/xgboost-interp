# Complete Model Analysis Guide

This guide explains how to use the `user_model_complete_analysis.py` script to perform comprehensive analysis on your XGBoost models.

## Overview

The `user_model_complete_analysis.py` script is designed to run **ALL** available analysis and plotting functions in the xgboost-interp package. It's perfect for:

- Quick model analysis without writing any code
- Generating comprehensive reports for stakeholders
- Understanding model behavior across all dimensions
- Comparing different models (run separately for each)

## Quick Start

### 1. Basic Usage (No Data Required)

If you only have the model JSON file:

```bash
python xgboost_interp/examples/user_model_complete_analysis.py your_model.json
```

**Generates ~15 plots including:**
- Feature importance (3 types: weight, gain, cover)
- Tree structure analysis
- Cumulative gain tracking
- Feature co-occurrence matrices
- Tree depth distributions
- Interactive tree visualizations

### 2. Complete Analysis (With Data)

If you have both model and data:

```bash
python xgboost_interp/examples/user_model_complete_analysis.py your_model.json data_directory/
```

**Generates 15+ base plots PLUS:**
- Partial Dependence Plots (PDPs) for **ALL** features
- Marginal impact analysis for **ALL** features
- Prediction evolution across trees
- ALE plots (if pyALE installed)

### 3. Multi-Class Models

For multi-class classification models, analyze each class separately:

```bash
# Analyze class 0
python xgboost_interp/examples/user_model_complete_analysis.py model.json data_dir/ --target-class 0

# Analyze class 1
python xgboost_interp/examples/user_model_complete_analysis.py model.json data_dir/ --target-class 1

# Analyze class 2
python xgboost_interp/examples/user_model_complete_analysis.py model.json data_dir/ --target-class 2
```

## Data Requirements

### Model File
- XGBoost model saved as JSON
- To save your model: `model.save_model("model.json")`
- Works with: XGBRegressor, XGBClassifier (binary and multi-class)

### Data Files (Optional, for complete analysis)
- Data in Parquet format
- Organized in a directory (can have multiple parquet files)
- Must contain all features used by the model
- Feature names must match model's feature names

**Converting your data to Parquet:**
```python
import pandas as pd

# From CSV
df = pd.read_csv("data.csv")
df.to_parquet("data_dir/data.parquet")

# From DataFrame
df.to_parquet("data_dir/data.parquet")

# Multiple files
df_train.to_parquet("data_dir/train.parquet")
df_test.to_parquet("data_dir/test.parquet")
```

## What Gets Generated

### Part 1: Tree-Level Analysis (15 functions, no data required)

1. **Model Summary** - Basic model statistics
2. **Feature Importance Combined** - Normalized importance across weight/gain/cover
3. **Feature Weight** - Split frequency by feature
4. **Feature Gain Distribution** - Boxplot of gain values
5. **Feature Cover Distribution** - Boxplot of cover values
6. **Feature Importance Scatter** - Usage vs Gain bubble chart
7. **Tree Depth Histogram** - Distribution of tree depths
8. **Cumulative Gain** - Loss reduction over trees
9. **Cumulative Prediction Shift** - Prediction magnitude over trees
10. **Tree-Level Co-occurrence** - Features appearing in same trees
11. **Path-Level Co-occurrence** - Features on same decision paths
12. **Feature Usage Heatmap** - Which features split together
13. **Split Depth Distribution** - Depth analysis per feature
14. **Feature Split Impact** - Impact of splits by feature
15. **Tree Statistics** - Gain and prediction stats per tree/depth

**Plus:** Interactive tree visualizations (first 5 trees)

### Part 2: Data-Dependent Analysis (requires data)

16. **Partial Dependence Plots** - One for EACH feature
    - Shows how predictions change with feature values
    - Includes Individual Conditional Expectation (ICE) curves
    - Shows average effect in red

17. **Marginal Impact Plots** - One for EACH feature
    - Shows prediction changes at each split threshold
    - Color-coded by positive/negative impact
    - Useful for understanding split decisions

18. **Scores Across Trees** - Prediction evolution
    - Shows how predictions converge as trees are added
    - Individual curves + mean/median

19. **ALE Plots** (bonus, requires pyALE)
    - Unbiased feature effects
    - Top 5 most important features

### Summary Report

After completion, you get a comprehensive summary showing:
- Total number of plots generated
- Plots categorized by type
- Complete file listing
- Multi-class analysis suggestions (if applicable)

## Example Output

```
================================================================================
XGBoost Model Complete Analysis
================================================================================
Model: california_housing_xgb.json
Data:  housing_data/
================================================================================

🔧 Initializing TreeAnalyzer...
✅ TreeAnalyzer initialized successfully

======================================================================
PART 1: TREE-LEVEL ANALYSIS (No Data Required)
======================================================================

[1/15] Printing model summary...

--- XGBoost Model Summary ---
Model Path               : california_housing_xgb.json
Number of Trees (Total)  : 100
Number of Trees (Outer)  : 100
Max Tree Depth           : 6
Learning Rate            : 0.1
Base Score               : 2.066
Objective                : reg:squarederror
Number of Features       : 8
Feature Preview          : ['MedInc', 'HouseAge', 'AveRooms', ...]
------------------------------

[2/15] Generating combined feature importance plot...
✅ Generated: feature_importance_combined.png

... [continues for all analyses] ...

======================================================================
✅ PART 1 COMPLETE - All tree-level analysis finished!
======================================================================

... [Part 2 continues with data-dependent analysis] ...

================================================================================
📊 ANALYSIS SUMMARY REPORT
================================================================================

📁 Output Directory: california_housing_xgb/
📈 Total Plots Generated: 32

📊 Plot Categories:
  🌳 Tree Structure Plots: 10
  🔧 Feature Analysis Plots: 5
  📉 Partial Dependence Plots: 8
  📊 Marginal Impact Plots: 8
  📋 Other Plots: 1

📄 All Generated Files:
  ✅ cumulative_gain.png
  ✅ feature_importance_combined.png
  ✅ PDP_MedInc.png
  ... [full list] ...

================================================================================
🎉 ANALYSIS COMPLETE!
================================================================================

All visualizations saved to: california_housing_xgb/
```

## Use Cases

### 1. Model Development
Run after training to understand:
- Which features drive predictions
- How deep your trees are growing
- If any features are overly dominant
- How predictions evolve across the ensemble

### 2. Model Debugging
Identify issues like:
- Unexpected feature importance
- Unusual split patterns
- Non-monotonic partial dependence (where expected)
- Features with minimal impact

### 3. Stakeholder Communication
Generate comprehensive visualizations showing:
- Model structure and complexity
- Feature relationships with predictions
- Model behavior transparency
- Decision-making logic

### 4. Model Comparison
Run for multiple models and compare:
- Feature importance differences
- Prediction behavior on same features
- Ensemble complexity (tree depth, count)
- Split thresholds and patterns

### 5. Production Monitoring
Establish baseline visualizations for:
- Comparing retrained models
- Detecting drift in feature importance
- Validating model updates
- Documenting model versions

## Tips and Tricks

### Handling Large Models

For models with many features (100+):
- Tree-level analysis is fast (no data needed)
- PDPs/marginal impacts will take longer (one per feature)
- Consider modifying the script to analyze top N features only

### Multi-Class Analysis Workflow

```bash
# Analyze all classes
for class_id in 0 1 2; do
    python xgboost_interp/examples/user_model_complete_analysis.py \
        model.json data_dir/ --target-class $class_id
done

# This creates separate output directories for each class
# model_xgb/  (default)
```

### Customizing Output

The script saves plots to a directory named after your model:
- `model.json` → plots in `model/`
- To customize, edit the script or use symlinks

### Memory Considerations

For very large datasets:
- The script loads data in batches (configurable)
- Parquet format is memory-efficient
- Consider sampling your data for analysis
- PDP computations are batched automatically

### Comparing Models

```bash
# Analyze multiple models
python xgboost_interp/examples/user_model_complete_analysis.py model_v1.json data/
python xgboost_interp/examples/user_model_complete_analysis.py model_v2.json data/

# Creates: model_v1/ and model_v2/ directories
# Now manually compare plots side-by-side
```

## Troubleshooting

### "Feature not found in data"
- Ensure data contains all model features
- Check feature name spelling/capitalization
- Verify parquet files loaded correctly

### "Failed to load model"
- Verify model saved as JSON, not binary (.model)
- Use `model.save_model("file.json")` not `pickle.dump()`

### "No tree depth data found"
- Model JSON might be corrupted
- Try re-saving the model

### Slow PDP generation
- Normal for many features
- Consider reducing `n_curves` parameter in script
- Sample your dataset to fewer records

### Import errors
- Install missing dependencies: `pip install plotly networkx pyALE`
- Core functionality works without these (some plots skipped)

## Advanced Usage

### Modifying the Script

The script is designed to be readable and modifiable:

```python
# Edit these parameters in the script:

# Number of ICE curves in PDPs
n_curves=1000  # Reduce for faster generation

# Grid resolution for PDPs
grid_points=50  # Reduce for faster generation

# Number of interactive trees
num_trees_to_plot=5  # Increase to visualize more trees

# Tree indices for prediction evolution
tree_indices = [10, 20, 30, ...]  # Customize checkpoints
```

### Programmatic Usage

Instead of the script, use the package directly:

```python
from xgboost_interp import TreeAnalyzer, ModelAnalyzer

# Your custom analysis logic
tree_analyzer = TreeAnalyzer("model.json")
tree_analyzer.plot_feature_importance_combined(top_n=10)
# ... continue with your specific needs ...
```

## Getting Help

- View command-line help: `python user_model_complete_analysis.py --help`
- See examples in `examples/` directory
- Read API documentation in README.md
- Check issues/discussions in GitHub repo

## What's Next?

After running complete analysis:

1. **Review tree-level plots** - Understand model structure
2. **Examine feature importance** - Identify key predictors
3. **Study PDPs** - See how features affect predictions
4. **Analyze marginal impacts** - Understand split decisions
5. **Share with stakeholders** - Use plots in presentations/reports
6. **Iterate on model** - Use insights to improve model design

---

**Note:** This script is designed as a starting point. For production use or specific research needs, consider creating custom analysis scripts using the xgboost-interp package's API.

