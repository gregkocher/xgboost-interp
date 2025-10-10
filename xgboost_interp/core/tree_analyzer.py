"""
Tree-level analysis for XGBoost models.

This module provides functionality to analyze XGBoost models at the tree level,
including feature importance, tree structure analysis, and various visualizations
that don't require actual data examples.
"""

import os
from typing import Optional, List, Dict, Any
from collections import defaultdict

from ..utils.model_utils import ModelLoader
from ..plotting.base_plotter import BasePlotter


class TreeAnalyzer:
    """
    Analyze XGBoost models at the tree structure level.
    
    This class provides methods to analyze XGBoost models based purely on
    their tree structure, without requiring actual data examples.
    """
    
    def __init__(self, json_path: str, save_dir: Optional[str] = None):
        """
        Initialize the TreeAnalyzer.
        
        Args:
            json_path: Path to the XGBoost model JSON file
            save_dir: Directory to save plots (defaults to model name without extension)
        """
        self.json_path = json_path
        self.model_json = ModelLoader.load_model_json(json_path)
        
        # Extract model metadata
        metadata = ModelLoader.extract_model_metadata(self.model_json)
        self.num_trees_total = metadata["num_trees_total"]
        self.num_trees_outer = metadata["num_trees_outer"]
        self.max_depth = metadata["max_depth"]
        self.learning_rate = metadata["learning_rate"]
        self.base_score = metadata["base_score"]
        self.feature_names = metadata["feature_names"]
        self.objective = metadata["objective"]
        
        # Extract trees
        self.trees = ModelLoader.extract_trees(self.model_json)
        
        # Setup plotting
        if save_dir is None:
            save_dir = os.path.splitext(json_path)[0]  # Remove .json extension
        self.plotter = BasePlotter(save_dir)
        
        # Cache for computed features
        self.feature_gains = defaultdict(list)
        self._gains_computed = False
    
    def print_model_summary(self) -> None:
        """Print a summary of the model structure and parameters."""
        print("\n--- XGBoost Model Summary ---")
        print(f"Model Path               : {self.json_path}")
        print(f"Number of Trees (Total)  : {self.num_trees_total}")
        print(f"Number of Trees (Outer)  : {self.num_trees_outer}")
        print(f"Max Tree Depth           : {self.max_depth}")
        print(f"Learning Rate            : {self.learning_rate}")
        print(f"Base Score               : {self.base_score}")
        print(f"Objective                : {self.objective}")
        print(f"Number of Features       : {len(self.feature_names)}")
        print(f"Feature Preview          : {self.feature_names[:10]}")
        print("------------------------------\n")
    
    def _collect_feature_gains(self) -> None:
        """Collect gain values for each feature across all trees."""
        if self._gains_computed:
            return
        
        self.feature_gains.clear()
        
        for tree in self.trees:
            split_indices = tree.get("split_indices", [])
            loss_changes = tree.get("loss_changes", [])
            left_children = tree.get("left_children", [])
            
            for node_id, feature_index in enumerate(split_indices):
                if left_children[node_id] == -1:  # Skip leaf nodes
                    continue
                
                if 0 <= feature_index < len(self.feature_names):
                    feature_name = self.feature_names[feature_index]
                else:
                    feature_name = f"f{feature_index}"
                
                gain = loss_changes[node_id]
                self.feature_gains[feature_name].append(gain)
        
        self._gains_computed = True
    
    def plot_feature_importance_combined(self, top_n: Optional[int] = None) -> None:
        """
        Plot normalized feature importance by weight, gain, and cover in one chart.
        
        Args:
            top_n: Number of top features to show (None for all)
        """
        weight_counts, gain_distributions, cover_distributions = (
            self.plotter._compute_feature_stats(self.trees, self.feature_names)
        )
        
        # Get all features that appear in any metric
        all_features = list(set(weight_counts) | set(gain_distributions) | set(cover_distributions))
        
        # Normalize each metric
        weight_norm = self.plotter._normalize_importance_dict(dict(weight_counts))
        
        gain_totals = {f: sum(gain_distributions[f]) for f in gain_distributions}
        gain_norm = self.plotter._normalize_importance_dict(gain_totals)
        
        cover_totals = {f: sum(cover_distributions[f]) for f in cover_distributions}
        cover_norm = self.plotter._normalize_importance_dict(cover_totals)
        
        # Combine and sort by gain
        combined = [
            (f, weight_norm.get(f, 0), gain_norm.get(f, 0), cover_norm.get(f, 0))
            for f in all_features
        ]
        combined.sort(key=lambda x: -x[2])  # Sort by gain
        
        if top_n:
            combined = combined[:top_n]
        
        if not combined:
            print("⚠️ No feature importance data found")
            return
        
        # Create the plot
        import matplotlib.pyplot as plt
        import numpy as np
        
        features = [x[0] for x in combined]
        weights = [x[1] for x in combined]
        gains = [x[2] for x in combined]
        covers = [x[3] for x in combined]
        
        y = np.arange(len(features))
        bar_width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
        ax.barh(y - bar_width, weights, bar_width, label="Weight", alpha=0.8)
        ax.barh(y, gains, bar_width, label="Gain", alpha=0.8)
        ax.barh(y + bar_width, covers, bar_width, label="Cover", alpha=0.8)
        
        ax.set_yticks(y)
        ax.set_yticklabels(features, fontsize=7)
        ax.set_xlabel("Normalized Importance (0–1)")
        ax.set_title("Feature Importances by Type (Normalized)")
        ax.invert_yaxis()
        ax.legend()
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        self.plotter._save_plot('feature_importance_combined.png')
    
    def plot_feature_importance_distributions(self, log_scale: bool = True, 
                                            top_n: Optional[int] = None) -> None:
        """
        Plot feature importance distributions using different metrics.
        
        Args:
            log_scale: Whether to use log scale for boxplots
            top_n: Number of top features to show (None for all)
        """
        weight_counts, gain_distributions, cover_distributions = (
            self.plotter._compute_feature_stats(self.trees, self.feature_names)
        )
        
        # Plot weight (split frequency) as bar chart
        if weight_counts:
            self.plotter._plot_horizontal_bar(
                dict(weight_counts), 
                "Feature Importance by Weight (Split Frequency)",
                "Split Count",
                'feature_weight.png',
                top_n
            )
        
        # Plot gain distributions as boxplot
        if gain_distributions:
            self.plotter._plot_boxplot(
                gain_distributions,
                "Feature Gain Distribution (Loss Change)",
                "Gain",
                'feature_gain_distribution.png',
                top_n,
                log_scale
            )
        
        # Plot cover distributions as boxplot
        if cover_distributions:
            self.plotter._plot_boxplot(
                cover_distributions,
                "Feature Cover Distribution (Sum Hessian)",
                "Cover",
                'feature_cover_distribution.png',
                top_n,
                log_scale
            )
    
    def plot_tree_depth_histogram(self) -> None:
        """Plot histogram of actual tree depths."""
        import matplotlib.pyplot as plt
        from ..utils.math_utils import MathUtils
        
        depths = []
        for tree in self.trees:
            lefts = tree.get("left_children", [])
            rights = tree.get("right_children", [])
            depth = MathUtils.compute_tree_depth(lefts, rights, 0)
            depths.append(depth)
        
        if not depths:
            print("⚠️ No tree depth data found")
            return
        
        fig, ax = plt.subplots(figsize=(8, 5))
        # Center bins on integer values by shifting bin edges by 0.5
        min_depth, max_depth = min(depths), max(depths)
        bin_edges = [i - 0.5 for i in range(min_depth, max_depth + 2)]
        ax.hist(depths, bins=bin_edges, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Tree Depth")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Tree Depths")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        # Set x-ticks to actual depth values
        ax.set_xticks(range(min_depth, max_depth + 1))
        
        plt.tight_layout()
        self.plotter._save_plot('tree_depth_histogram.png')
    
    def plot_cumulative_gain(self) -> None:
        """Plot cumulative gain (loss_change) across all trees."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        gains = []
        for tree in self.trees:
            total_gain = sum(tree.get("loss_changes", []))
            gains.append(total_gain)
        
        if not gains:
            print("⚠️ No gain data found")
            return
        
        cumulative_gain = np.cumsum(gains)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(cumulative_gain, linewidth=2, color='blue')
        ax.set_xlabel("Tree Index")
        ax.set_ylabel("Cumulative Loss Change")
        ax.set_title("Cumulative Gain Over Trees")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        self.plotter._save_plot('cumulative_gain.png')
    
    def plot_cumulative_prediction_shift(self) -> None:
        """Plot cumulative prediction shift using mean absolute leaf outputs."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        shifts = []
        for tree in self.trees:
            base_weights = tree.get("base_weights", [])
            lefts = tree.get("left_children", [])
            
            # Get leaf values (nodes where left_children == -1)
            leaf_vals = [w for i, w in enumerate(base_weights) if lefts[i] == -1]
            mean_abs_shift = np.mean(np.abs(leaf_vals)) if leaf_vals else 0
            shifts.append(mean_abs_shift)
        
        if not shifts:
            print("⚠️ No prediction shift data found")
            return
        
        cumulative_shift = np.cumsum(shifts)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(cumulative_shift, linewidth=2, color='red')
        ax.set_xlabel("Tree Index")
        ax.set_ylabel("Cumulative Σ |Leaf Output|")
        ax.set_title("Cumulative Prediction Shift Over Trees")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        self.plotter._save_plot('cumulative_prediction_shift.png')
