"""
XGBoost Interpretability Package

A comprehensive toolkit for interpreting and analyzing XGBoost models.
Provides tree-level analysis, feature importance, partial dependence plots,
and interactive visualizations.
"""

from .core.tree_analyzer import TreeAnalyzer
from .core.model_analyzer import ModelAnalyzer
from .core.model_diff import ModelDiff

__version__ = "0.1.0"
__author__ = "Greg Kocher"

__all__ = [
    "TreeAnalyzer",
    "ModelAnalyzer",
    "ModelDiff",
]
