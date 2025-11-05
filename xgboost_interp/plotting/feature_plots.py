"""
Feature-specific plotting functionality.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from itertools import combinations

from .base_plotter import BasePlotter


class FeaturePlotter(BasePlotter):
    """Specialized plotting for feature analysis."""
    
    def plot_feature_cooccurrence_heatmap(self, matrix: np.ndarray, labels: List[str], 
                                        title: str, filename: str, 
                                        log_scale: bool = False) -> None:
        """
        Plot feature co-occurrence as a heatmap.
        
        Args:
            matrix: Co-occurrence matrix
            labels: Feature labels
            title: Plot title
            filename: Output filename
            log_scale: Whether to apply log scaling
        """
        matrix = np.array(matrix)
        if log_scale:
            matrix = np.log1p(matrix)
        
        fig_size = (max(12, len(labels) * 0.25), max(10, len(labels) * 0.25))
        fig, ax = plt.subplots(figsize=fig_size)
        
        heatmap = sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, 
                   cmap="viridis", square=True, ax=ax, cbar_kws={'label': ''})
        
        # Make colorbar text bigger
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        
        ax.set_xticklabels(labels, rotation=45, fontsize=6, ha="right")
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(title, fontsize=16)
        
        plt.tight_layout()
        self._save_plot(filename)
    
    def plot_feature_usage_heatmap(self, trees: List[Dict], feature_names: List[str],
                                 top_k: Optional[int] = None, log_scale: bool = False) -> None:
        """
        Plot heatmap of feature usage across trees.
        
        Args:
            trees: List of tree dictionaries
            feature_names: List of feature names
            top_k: Number of trees to include (None for all)
            log_scale: Whether to apply log scaling
        """
        if top_k:
            trees = trees[:top_k]
        
        num_features = len(feature_names)
        num_trees = len(trees)
        usage_matrix = np.zeros((num_features, num_trees), dtype=int)
        
        for t_idx, tree in enumerate(trees):
            split_indices = tree.get("split_indices", [])
            lefts = tree.get("left_children", [])
            
            for node_idx, feat_idx in enumerate(split_indices):
                if lefts[node_idx] == -1:  # Skip leaf nodes
                    continue
                if 0 <= feat_idx < num_features:
                    usage_matrix[feat_idx][t_idx] += 1
        
        # Sort features by total usage
        total_usage = usage_matrix.sum(axis=1)
        sorted_indices = np.argsort(-total_usage)
        usage_matrix = usage_matrix[sorted_indices]
        feature_labels = [feature_names[i] for i in sorted_indices]
        
        if log_scale:
            usage_matrix = np.log1p(usage_matrix)
        
        fig_width = max(16, num_trees * 0.02)
        fig_height = max(10, num_features * 0.3)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        heatmap = sns.heatmap(usage_matrix, cmap="viridis", xticklabels=100, 
                   yticklabels=feature_labels, ax=ax)
        
        # Make colorbar text bigger
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        
        ax.set_xlabel("Tree Index")
        ax.set_ylabel("Feature (sorted by total usage)")
        ax.set_title("Feature Usage Across Trees", fontsize=16)
        
        plt.tight_layout()
        self._save_plot('feature_usage_heatmap.png')
    
    def plot_split_depth_per_feature(self, trees: List[Dict], feature_names: List[str],
                                   top_n: Optional[int] = None) -> None:
        """
        Plot boxplot of split depths for each feature.
        
        Args:
            trees: List of tree dictionaries
            feature_names: List of feature names
            top_n: Number of top features to show
        """
        feature_depths = defaultdict(list)
        
        for tree in trees:
            split_indices = tree.get("split_indices", [])
            lefts = tree.get("left_children", [])
            rights = tree.get("right_children", [])
            
            def traverse(node: int, depth: int) -> None:
                if node == -1 or node >= len(split_indices):
                    return
                
                # Skip leaf nodes - they don't split on features
                if lefts[node] == -1:
                    return
                
                feat_idx = split_indices[node]
                if 0 <= feat_idx < len(feature_names):
                    feature_name = feature_names[feat_idx]
                    feature_depths[feature_name].append(depth)
                
                traverse(lefts[node], depth + 1)
                traverse(rights[node], depth + 1)
            
            traverse(0, 0)
        
        if not feature_depths:
            print("⚠️ No split depth data found")
            return
        
        # Sort by mean depth
        mean_depths = {f: np.mean(depths) for f, depths in feature_depths.items()}
        sorted_feats = sorted(mean_depths, key=lambda f: mean_depths[f])
        
        if top_n:
            sorted_feats = sorted_feats[:top_n]
        
        distributions = [feature_depths[f] for f in sorted_feats]
        
        fig, ax = plt.subplots(figsize=(max(16, len(sorted_feats) * 0.25), 6))
        ax.boxplot(distributions, vert=True, patch_artist=True, 
                  showfliers=False, showmeans=True)
        
        ax.set_xticks(range(1, len(sorted_feats) + 1))
        ax.set_xticklabels(sorted_feats, rotation=45, fontsize=6, ha="right")
        ax.set_ylabel("Split Depth")
        ax.set_title("Distribution of Split Depth per Feature")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        self._save_plot('split_depth_per_feature.png')
    
    def plot_feature_split_impact(self, trees: List[Dict], feature_names: List[str],
                                log_scale: bool = False, top_n: Optional[int] = None) -> None:
        """
        Plot the impact of splits on predictions for each feature.
        
        Args:
            trees: List of tree dictionaries
            feature_names: List of feature names
            log_scale: Whether to use log scale
            top_n: Number of top features to show
        """
        feature_deltas = defaultdict(list)
        
        for tree in trees:
            split_indices = tree.get("split_indices", [])
            base_weights = tree.get("base_weights", [])
            left_children = tree.get("left_children", [])
            right_children = tree.get("right_children", [])
            
            for i, feat_idx in enumerate(split_indices):
                # Skip leaf nodes
                if left_children[i] == -1 or right_children[i] == -1:
                    continue
                if feat_idx >= len(feature_names):
                    continue
                
                # Calculate difference between left and right children
                left = base_weights[left_children[i]]
                right = base_weights[right_children[i]]
                
                # Absolute difference between children (prediction divergence at split)
                delta = abs(right - left)
                feature_name = feature_names[feat_idx]
                feature_deltas[feature_name].append(delta)
        
        if not feature_deltas:
            print("⚠️ No split impact data found")
            return
        
        self._plot_boxplot(
            feature_deltas,
            "Feature Split Impact on Predictions",
            "Split Δ in Prediction (avg child-parent abs delta)",
            'feature_split_impact.png',
            top_n,
            log_scale
        )
