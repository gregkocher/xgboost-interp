# Interactive Plots Guide

The `InteractivePlotter` class creates interactive tree visualizations using Plotly.

## Basic Usage

```python
from xgboost_interp import TreeAnalyzer
from xgboost_interp.plotting import InteractivePlotter

# Load model
tree_analyzer = TreeAnalyzer("your_model.json")

# Create interactive plotter
plotter = InteractivePlotter(tree_analyzer.plotter.save_dir)

# Plot trees
plotter.plot_interactive_trees(
    tree_analyzer.trees,
    tree_analyzer.feature_names,
    top_k=5,
    combined=False
)
```

## Options

### Separate Trees

```python
# Each tree in its own plot
plotter.plot_interactive_trees(
    trees, feature_names,
    top_k=10,
    combined=False
)
```

### Combined View

```python
# All trees in one scrollable plot
plotter.plot_interactive_trees(
    trees, feature_names,
    top_k=30,
    combined=True,
    vertical_spacing=12
)
```

## Features

### Node Colors
- **Split nodes**: Gray
- **Positive leaf nodes**: Green (opacity based on magnitude)
- **Negative leaf nodes**: Red (opacity based on magnitude)

### Node Sizes
- Proportional to cover (number of samples passing through)

### Hover Information

**Split nodes:**
- Feature name
- Threshold value
- Gain
- Cover

**Leaf nodes:**
- Δ Score (prediction contribution)
- Cover

## Requirements

Interactive plots require optional dependencies:

```bash
pip install plotly networkx
```

