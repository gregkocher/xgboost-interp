"""
Tree structure plotting functionality.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from .base_plotter import BasePlotter


class TreePlotter(BasePlotter):
    """Specialized plotting for tree structure analysis."""
    
    def plot_prediction_and_gain_stats(self, trees: List[Dict], 
                                     log_scale: bool = False, 
                                     top_k: Optional[int] = None) -> None:
        """
        Plot prediction and gain statistics across trees and depth levels.
        
        Args:
            trees: List of tree dictionaries
            log_scale: Whether to use log scale
            top_k: Number of trees to analyze (None for all)
        """
        if top_k:
            trees = trees[:top_k]
        
        # Helper function to compute statistics
        def compute_stats(data_dict: Dict, sorted_keys: List) -> tuple:
            means = [np.mean(data_dict[k]) for k in sorted_keys]
            medians = [np.median(data_dict[k]) for k in sorted_keys]
            stds = [np.std(data_dict[k]) for k in sorted_keys]
            return means, medians, stds
        
        # Helper function to plot statistics
        def plot_stats(x: List, means: List, medians: List, stds: List,
                      x_label: str, y_label: str, title: str, filename: str) -> None:
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.plot(x, means, label="Mean", linewidth=2, marker='o')
            ax.plot(x, medians, label="Median", linewidth=2, marker='s')
            ax.plot(x, stds, label="Std Dev", linewidth=2, marker='^')
            
            if log_scale:
                ax.set_yscale("log")
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title, fontsize=12)
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax.legend()
            
            plt.tight_layout()
            self._save_plot(filename)
        
        # 1. Leaf predictions per tree
        per_tree_weights = defaultdict(list)
        for idx, tree in enumerate(trees):
            weights = tree.get("base_weights", [])
            lefts = tree.get("left_children", [])
            leaf_vals = [w for i, w in enumerate(weights) if lefts[i] == -1]
            per_tree_weights[idx] = leaf_vals or [0]
        
        tree_ids = sorted(per_tree_weights.keys())
        means, medians, stds = compute_stats(per_tree_weights, tree_ids)
        plot_stats(tree_ids, means, medians, stds,
                  "Tree Index", "Predictions", 
                  "Per-Tree Leaf Predictions (Mean/Median/Std)",
                  "prediction_stats_per_tree.png")
        
        # 2. Leaf predictions per depth level
        level_weights = defaultdict(list)
        for tree in trees:
            weights = tree.get("base_weights", [])
            lefts = tree.get("left_children", [])
            rights = tree.get("right_children", [])
            
            def dfs(node: int, depth: int) -> None:
                if node == -1 or node >= len(weights):
                    return
                level_weights[depth].append(weights[node])
                dfs(lefts[node], depth + 1)
                dfs(rights[node], depth + 1)
            
            dfs(0, 0)
        
        levels = sorted(level_weights.keys())
        means, medians, stds = compute_stats(level_weights, levels)
        plot_stats(levels, means, medians, stds,
                  "Tree Depth Level", "Base Weight",
                  "Leaf Predictions by Tree Depth Level",
                  "prediction_stats_by_depth.png")
        
        # 3. Gain per tree
        per_tree_gains = defaultdict(list)
        for idx, tree in enumerate(trees):
            gains = tree.get("loss_changes", [])
            lefts = tree.get("left_children", [])
            
            # Only include split nodes (not leaves)
            split_gains = [gain for i, gain in enumerate(gains) if lefts[i] != -1]
            per_tree_gains[idx] = split_gains or [0]
        
        tree_ids = sorted(per_tree_gains.keys())
        means, medians, stds = compute_stats(per_tree_gains, tree_ids)
        plot_stats(tree_ids, means, medians, stds,
                  "Tree Index", "Loss Change (Gain)",
                  "Per-Tree Split Gain (Mean/Median/Std)",
                  "gain_stats_per_tree.png")
        
        # 4. Gain per depth level
        level_gains = defaultdict(list)
        for tree in trees:
            gains = tree.get("loss_changes", [])
            lefts = tree.get("left_children", [])
            rights = tree.get("right_children", [])
            
            def dfs(node: int, depth: int) -> None:
                if node == -1 or node >= len(gains):
                    return
                if lefts[node] != -1:  # Only split nodes
                    level_gains[depth].append(gains[node])
                dfs(lefts[node], depth + 1)
                dfs(rights[node], depth + 1)
            
            dfs(0, 0)
        
        levels = sorted(level_gains.keys())
        means, medians, stds = compute_stats(level_gains, levels)
        plot_stats(levels, means, medians, stds,
                  "Tree Depth Level", "Loss Change (Gain)",
                  "Gain by Depth Level Across Trees",
                  "gain_stats_by_depth.png")
    
    def plot_gain_heatmap(self, trees: List[Dict], feature_names: List[str]) -> None:
        """
        Plot per-tree feature gain heatmap.
        
        Args:
            trees: List of tree dictionaries
            feature_names: List of feature names
        """
        feature_to_index = {f: i for i, f in enumerate(feature_names)}
        gain_matrix = np.zeros((len(trees), len(feature_names)))
        
        for tree_idx, tree in enumerate(trees):
            split_indices = tree.get("split_indices", [])
            loss_changes = tree.get("loss_changes", [])
            left_children = tree.get("left_children", [])
            
            for node_id, feature_index in enumerate(split_indices):
                if left_children[node_id] == -1:  # Skip leaf nodes
                    continue
                
                if 0 <= feature_index < len(feature_names):
                    feature_name = feature_names[feature_index]
                    feat_idx = feature_to_index[feature_name]
                    gain_matrix[tree_idx, feat_idx] += loss_changes[node_id]
        
        if not gain_matrix.any():
            print("⚠️ All gains are zero — no valid splits found.")
            return
        
        fig, ax = plt.subplots(figsize=(max(18, len(feature_names) * 0.25), 8))
        
        import seaborn as sns
        sns.heatmap(
            gain_matrix,
            xticklabels=feature_names,
            yticklabels=[f"Tree {i}" for i in range(len(trees))],
            cmap="viridis",
            cbar_kws={'label': 'Gain (loss_changes)'},
            ax=ax
        )
        
        ax.set_title("Per-Tree Feature Gain Heatmap")
        ax.set_xlabel("Features")
        ax.set_ylabel("Trees")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, fontsize=6, ha="right")
        plt.tight_layout()
        self._save_plot('gain_heatmap.png')
