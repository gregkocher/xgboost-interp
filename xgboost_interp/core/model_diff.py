"""
Model diffing for comparing two XGBoost models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List

from .tree_analyzer import TreeAnalyzer
from ..plotting.base_plotter import BasePlotter


class ModelDiff:
    """
    Compare two XGBoost models and identify structural differences.
    
    Diffs are directional: delta = model_b - model_a
    (positive delta means model_b has higher value)
    """
    
    def __init__(
        self,
        analyzer_a: TreeAnalyzer,
        analyzer_b: TreeAnalyzer,
        label_a: str = "Model A",
        label_b: str = "Model B",
        save_dir: Optional[str] = None,
    ):
        """
        Initialize ModelDiff.
        
        Args:
            analyzer_a: TreeAnalyzer for the baseline/old model
            analyzer_b: TreeAnalyzer for the candidate/new model
            label_a: Display label for model A
            label_b: Display label for model B
            save_dir: Directory to save plots (defaults to "model_diff_output")
        """
        self.analyzer_a = analyzer_a
        self.analyzer_b = analyzer_b
        self.label_a = label_a
        self.label_b = label_b
        
        if save_dir is None:
            save_dir = "model_diff_output"
        self.plotter = BasePlotter(save_dir)
    
    def print_summary(self) -> None:
        """Print side-by-side comparison of model metadata."""
        a = self.analyzer_a
        b = self.analyzer_b
        
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Metric':<25} {self.label_a:<20} {self.label_b:<20}")
        print("-" * 70)
        print(f"{'Number of Trees':<25} {a.num_trees_total:<20} {b.num_trees_total:<20}")
        print(f"{'Max Depth':<25} {a.max_depth:<20} {b.max_depth:<20}")
        print(f"{'Learning Rate':<25} {a.learning_rate:<20} {b.learning_rate:<20}")
        print(f"{'Base Score':<25} {a.base_score:<20} {b.base_score:<20}")
        print(f"{'Objective':<25} {str(a.objective):<20} {str(b.objective):<20}")
        print(f"{'Number of Features':<25} {len(a.feature_names):<20} {len(b.feature_names):<20}")
        print("=" * 70 + "\n")
    
    def find_feature_changes(self) -> Dict[str, List[str]]:
        """
        Identify features that are new, dropped, or shared between models.
        
        Returns:
            Dict with keys:
            - 'new_in_b': Features in model B but not in model A
            - 'dropped_in_b': Features in model A but not in model B
            - 'in_both': Features present in both models
        """
        features_a = set(self.analyzer_a.feature_names)
        features_b = set(self.analyzer_b.feature_names)
        
        return {
            'new_in_b': sorted(features_b - features_a),
            'dropped_in_b': sorted(features_a - features_b),
            'in_both': sorted(features_a & features_b),
        }
    
    def compare_cumulative_gain(self) -> None:
        """Plot overlay of cumulative gain curves from both models."""
        # Compute cumulative gain for model A
        gains_a = []
        for tree in self.analyzer_a.trees:
            total_gain = sum(tree.get("loss_changes", []))
            gains_a.append(total_gain)
        cumulative_a = np.cumsum(gains_a)
        
        # Compute cumulative gain for model B
        gains_b = []
        for tree in self.analyzer_b.trees:
            total_gain = sum(tree.get("loss_changes", []))
            gains_b.append(total_gain)
        cumulative_b = np.cumsum(gains_b)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(range(len(cumulative_a)), cumulative_a, 
                linewidth=2, label=self.label_a, color='blue')
        ax.plot(range(len(cumulative_b)), cumulative_b, 
                linewidth=2, label=self.label_b, color='red')
        
        ax.set_xlabel("Tree Index")
        ax.set_ylabel("Cumulative Loss Change")
        ax.set_title(f"Cumulative Gain Comparison: {self.label_a} vs {self.label_b}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        self.plotter._save_plot('cumulative_gain_comparison.png')
        print("Generated: cumulative_gain_comparison.png")
    
    def plot_importance_scatter(self, metric: str = "gain") -> None:
        """
        Plot scatterplot comparing feature importance between models.
        
        X-axis: Model A importance
        Y-axis: Model B importance
        Points above diagonal = feature increased in model B
        Points below diagonal = feature decreased in model B
        
        Args:
            metric: "gain", "weight", or "cover"
        """
        # Get feature importance from both models
        weight_a, gain_a, cover_a = self.analyzer_a.get_feature_importance()
        weight_b, gain_b, cover_b = self.analyzer_b.get_feature_importance()
        
        # Select the appropriate metric
        if metric == "weight":
            values_a = dict(weight_a)
            values_b = dict(weight_b)
            metric_label = "Weight (Split Frequency)"
        elif metric == "gain":
            values_a = {f: sum(v) for f, v in gain_a.items()}
            values_b = {f: sum(v) for f, v in gain_b.items()}
            metric_label = "Gain (Total Loss Reduction)"
        elif metric == "cover":
            values_a = {f: sum(v) for f, v in cover_a.items()}
            values_b = {f: sum(v) for f, v in cover_b.items()}
            metric_label = "Cover (Total Sum Hessian)"
        else:
            raise ValueError(f"Invalid metric '{metric}'. Must be 'gain', 'weight', or 'cover'")
        
        # Get features that appear in both models
        common_features = set(values_a.keys()) & set(values_b.keys())
        
        # Build data arrays
        features = []
        x_vals = []
        y_vals = []
        
        for feat in common_features:
            val_a = values_a[feat]
            val_b = values_b[feat]
            # Skip features with zero values (can't plot on log scale)
            if val_a > 0 and val_b > 0:
                features.append(feat)
                x_vals.append(val_a)
                y_vals.append(val_b)
        
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Scatter points
        ax.scatter(x_vals, y_vals, alpha=0.7, s=50, c='steelblue', edgecolors='black', linewidth=0.5)
        
        # Add feature labels
        for i, feat in enumerate(features):
            ax.annotate(feat, (x_vals[i], y_vals[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, alpha=0.8)
        
        # y=x diagonal reference line
        min_val = min(x_vals.min(), y_vals.min())
        max_val = max(x_vals.max(), y_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'k--', linewidth=1.5, alpha=0.7, label='y = x (no change)')
        
        # Log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlabel(f"{self.label_a} - {metric_label}", fontsize=11)
        ax.set_ylabel(f"{self.label_b} - {metric_label}", fontsize=11)
        ax.set_title(f"Feature {metric.title()} Comparison\n(above line = increased in {self.label_b})", 
                    fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        
        # Make plot square
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        filename = f'importance_scatter_{metric}.png'
        self.plotter._save_plot(filename)
        print(f"Generated: {filename}")
    
    def plot_all_importance_scatters(self) -> None:
        """Generate all 3 importance scatterplots (gain, weight, cover)."""
        for metric in ["gain", "weight", "cover"]:
            self.plot_importance_scatter(metric)
