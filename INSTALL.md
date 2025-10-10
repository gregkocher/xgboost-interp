# Installation Guide

## Quick Installation

### From Source (Development)

```bash
cd /Users/gkocher/Desktop/xgboost-interp
pip install -e .
```

### Testing the Installation

```bash
python test_package.py
```

### Running Examples

```bash
# Basic tree analysis (no data required)
python -m xgboost_interp.examples.basic_analysis

# Advanced analysis with data
python -m xgboost_interp.examples.advanced_analysis
```

## Package Structure

```
xgboost_interp/
├── __init__.py                 # Main package exports
├── core/
│   ├── tree_analyzer.py       # TreeAnalyzer class (was TreeUnderstanding)
│   └── model_analyzer.py      # ModelAnalyzer class (was ModelWithDataAnalyzer)
├── plotting/
│   ├── base_plotter.py        # Common plotting utilities
│   ├── feature_plots.py       # Feature-specific plots
│   ├── tree_plots.py          # Tree structure plots
│   └── interactive_plots.py   # Plotly interactive plots
├── utils/
│   ├── model_utils.py         # Model loading utilities
│   ├── data_utils.py          # Data loading utilities
│   └── math_utils.py          # Mathematical helpers
└── examples/
    ├── basic_analysis.py      # Basic usage examples
    └── advanced_analysis.py   # Comprehensive analysis
```

## Migration from Original Code

### Before (Original)
```python
from xgboost_interp import TreeUnderstanding, ModelWithDataAnalyzer

tree_parser = TreeUnderstanding("model.json")
analyzer = ModelWithDataAnalyzer(tree_parser)
```

### After (Refactored)
```python
from xgboost_interp import TreeAnalyzer, ModelAnalyzer

tree_analyzer = TreeAnalyzer("model.json")
model_analyzer = ModelAnalyzer(tree_analyzer)
```

## Key Improvements

1. **Modular Design**: Separated concerns into focused modules
2. **Clean API**: Intuitive class and method names
3. **Better Documentation**: Comprehensive docstrings and type hints
4. **Reusable Components**: Common plotting utilities extracted
5. **PyPI Ready**: Proper package structure with setup.py
6. **Examples**: Clear usage examples for different scenarios
7. **No Dead Code**: Removed commented-out and unrelated code

## Dependencies

All dependencies are automatically installed with the package:

- numpy, pandas, matplotlib, seaborn (core)
- scikit-learn, scipy, xgboost (ML libraries)
- plotly, networkx (interactive plots)
- pyALE (ALE plots)

## Next Steps

1. Install the package: `pip install -e .`
2. Update model paths in examples
3. Run examples to verify functionality
4. Use in your own analysis scripts
