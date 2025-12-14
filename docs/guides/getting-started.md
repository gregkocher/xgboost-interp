# Getting Started

This guide covers installation and basic usage of the xgboost-interp package.

## Installation

```bash
git clone https://github.com/gregkocher/xgboost-interp.git
cd xgboost-interp
uv sync
source .venv/bin/activate
```

## Quick Start

Analyze your own XGBoost model:

```bash
python xgboost_interp/examples/user_model_complete_analysis.py your_model.json path/to/data/
```

## Next Steps

- [TreeAnalyzer Guide](tree-analyzer.md) - Tree-level analysis without data
- [ModelAnalyzer Guide](model-analyzer.md) - Data-dependent analysis
- [Interactive Plots Guide](interactive-plots.md) - Interactive tree visualization

