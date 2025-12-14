"""
Tree-level analysis for XGBoost models.

This module provides functionality to analyze XGBoost models at the tree level,
including feature importance, tree structure analysis, and various visualizations
that don't require actual data examples.
"""

import os
from typing import Optional, List, Dict, Any

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
        
        def compute_tree_depth(left_children, right_children, start_node=0):
            """Compute the depth of a tree given its structure."""
            def _compute_depth_recursive(node):
                if node == -1 or node >= len(left_children):
                    return 0
                left_depth = _compute_depth_recursive(left_children[node])
                right_depth = _compute_depth_recursive(right_children[node])
                return 1 + max(left_depth, right_depth)
            return _compute_depth_recursive(start_node)
        
        depths = []
        for tree in self.trees:
            lefts = tree.get("left_children", [])
            rights = tree.get("right_children", [])
            depth = compute_tree_depth(lefts, rights, 0)
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
        """Plot cumulative mean absolute leaf magnitude across trees."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        mean_abs_magnitudes = []
        for tree in self.trees:
            base_weights = tree.get("base_weights", [])
            lefts = tree.get("left_children", [])
            
            # Get leaf values (nodes where left_children == -1)
            leaf_vals = [w for i, w in enumerate(base_weights) if lefts[i] == -1]
            mean_abs_magnitude = np.mean(np.abs(leaf_vals)) if leaf_vals else 0
            mean_abs_magnitudes.append(mean_abs_magnitude)
        
        if not mean_abs_magnitudes:
            print("⚠️ No leaf magnitude data found")
            return
        
        cumulative_magnitude = np.cumsum(mean_abs_magnitudes)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(cumulative_magnitude, linewidth=2, color='red')
        ax.set_xlabel("Tree Index")
        ax.set_ylabel("Cumulative Σ Mean(|Leaf Values|)")
        ax.set_title("Cumulative Leaf Magnitude Over Trees")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        self.plotter._save_plot('cumulative_prediction_shift.png')
    
    def compute_tree_level_feature_cooccurrence(self) -> tuple:
        """
        Compute symmetric co-occurrence matrix of features appearing in the same tree.
        
        Returns:
            tuple: (co_matrix, feature_names) where co_matrix is a 2D numpy array [F x F]
        """
        import numpy as np
        from itertools import combinations
        
        num_features = len(self.feature_names)
        co_matrix = np.zeros((num_features, num_features), dtype=int)
        
        for tree in self.trees:
            split_indices = tree.get("split_indices", [])
            left_children = tree.get("left_children", [])
            right_children = tree.get("right_children", [])
            
            # Only include actual split nodes (not leaf nodes)
            features_in_tree = set()
            for node_id, feat_idx in enumerate(split_indices):
                # Skip leaf nodes (both children == -1)
                is_leaf = (left_children[node_id] == -1 and right_children[node_id] == -1)
                if is_leaf:
                    continue
                if 0 <= feat_idx < num_features:
                    features_in_tree.add(self.feature_names[feat_idx])
            
            # Count co-occurrences
            for f1, f2 in combinations(features_in_tree, 2):
                i, j = self.feature_names.index(f1), self.feature_names.index(f2)
                co_matrix[i][j] += 1
                co_matrix[j][i] += 1
            for f in features_in_tree:
                i = self.feature_names.index(f)
                co_matrix[i][i] += 1
        
        return co_matrix, self.feature_names
    
    def compute_path_level_feature_cooccurrence(self) -> tuple:
        """
        Compute symmetric co-occurrence matrix of features appearing on the same decision path.
        
        Returns:
            tuple: (co_matrix, feature_names) where co_matrix is a 2D numpy array [F x F]
        """
        import numpy as np
        from itertools import combinations
        
        num_features = len(self.feature_names)
        co_matrix = np.zeros((num_features, num_features), dtype=int)
        
        for tree in self.trees:
            split_indices = tree.get("split_indices", [])
            left_children = tree.get("left_children", [])
            right_children = tree.get("right_children", [])
            
            def dfs(node, path):
                if node == -1:
                    return
                
                # Check if this is a leaf node
                if left_children[node] == -1 and right_children[node] == -1:
                    # Leaf node - process the path (don't add this node's split_index)
                    features_in_path = set(self.feature_names[i] for i in path 
                                          if 0 <= i < num_features)
                    for f1, f2 in combinations(features_in_path, 2):
                        i, j = self.feature_names.index(f1), self.feature_names.index(f2)
                        co_matrix[i][j] += 1
                        co_matrix[j][i] += 1
                    for f in features_in_path:
                        i = self.feature_names.index(f)
                        co_matrix[i][i] += 1
                    return
                
                # Internal node - add to path and recurse
                path.append(split_indices[node])
                dfs(left_children[node], path)
                dfs(right_children[node], path)
                path.pop()
            
            dfs(0, [])
        
        return co_matrix, self.feature_names
    
    def compute_sequential_feature_dependency(self) -> tuple:
        """
        Compute asymmetric matrix showing rate of feature B following feature A.
        
        Conditional probability calculation:
        - P(feature_j follows feature_i) = (# times j is child of i) / (# times i is interior node)
        - Denominator: Only counts interior nodes (nodes with at least one child)
        - Numerator: Counts both left and right children separately
        - If feature B appears in BOTH children of A, it's counted twice
        
        Returns:
            tuple: (dependency_matrix, feature_names) 
                   dependency_matrix is [F x F] with values in [0, 1]
        """
        import numpy as np
        
        num_features = len(self.feature_names)
        
        # Denominator: count of CHILDREN opportunities per feature
        # (count how many non-leaf children each parent feature has)
        parent_children_count = np.zeros(num_features, dtype=int)
        
        # Numerator: count of parent→child pairs
        parent_child_counts = np.zeros((num_features, num_features), dtype=int)
        
        for tree in self.trees:
            split_indices = tree.get("split_indices", [])
            left_children = tree.get("left_children", [])
            right_children = tree.get("right_children", [])
            
            for node_id, parent_feat_idx in enumerate(split_indices):
                # CRITICAL: Skip leaf nodes from denominator
                # In XGBoost, leaf nodes have BOTH children == -1
                # After pruning, interior nodes may have only 1 child (either left or right)
                is_leaf = (left_children[node_id] == -1 and right_children[node_id] == -1)
                if is_leaf:
                    continue
                
                # Validate feature index
                if not (0 <= parent_feat_idx < num_features):
                    continue
                
                # Process both left and right children using a loop to avoid duplication
                for child_id in [left_children[node_id], right_children[node_id]]:
                    if child_id != -1 and child_id < len(split_indices):
                        # Check if this child is NOT a leaf (has at least one child itself)
                        child_is_leaf = (left_children[child_id] == -1 and 
                                        right_children[child_id] == -1)
                        if not child_is_leaf:
                            # Count this as an opportunity in denominator
                            parent_children_count[parent_feat_idx] += 1
                            
                            child_feat_idx = split_indices[child_id]
                            if 0 <= child_feat_idx < num_features:
                                parent_child_counts[parent_feat_idx][child_feat_idx] += 1
        
        # Compute conditional probabilities: P(child | parent)
        dependency_matrix = np.zeros((num_features, num_features), dtype=float)
        for i in range(num_features):
            if parent_children_count[i] > 0:
                dependency_matrix[i, :] = parent_child_counts[i, :] / parent_children_count[i]
        
        return dependency_matrix, self.feature_names
    
    def plot_tree_level_feature_cooccurrence(self) -> None:
        """Plot heatmap of feature co-occurrence at tree level."""
        from ..plotting.feature_plots import FeaturePlotter
        
        matrix, labels = self.compute_tree_level_feature_cooccurrence()
        plotter = FeaturePlotter(self.plotter.save_dir)
        plotter.plot_feature_cooccurrence_heatmap(
            matrix,
            labels,
            title="Same Tree Feature Co-occurrence",
            filename="feature_cooccurrence_tree_level.png",
            log_scale=False
        )
    
    def plot_path_level_feature_cooccurrence(self) -> None:
        """Plot heatmap of feature co-occurrence at path level (log scale)."""
        from ..plotting.feature_plots import FeaturePlotter
        
        matrix, labels = self.compute_path_level_feature_cooccurrence()
        plotter = FeaturePlotter(self.plotter.save_dir)
        plotter.plot_feature_cooccurrence_heatmap(
            matrix,
            labels,
            title="Path-Based Feature Co-occurrence",
            filename="feature_cooccurrence_path_level.png",
            log_scale=True
        )
    
    def plot_sequential_feature_dependency(self, top_n: Optional[int] = None) -> None:
        """
        Plot asymmetric heatmap of sequential feature dependencies.
        
        Shows: P(feature B follows feature A) as a heatmap.
        Rows = parent features, Columns = child features.
        
        Args:
            top_n: Show only top N features by total parent occurrences (None for all)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        dependency_matrix, feature_names = self.compute_sequential_feature_dependency()
        
        # Filter to top N features if specified
        if top_n is not None and top_n < len(feature_names):
            # Rank by how often features appear as parents
            parent_freq = dependency_matrix.sum(axis=1)
            top_indices = np.argsort(parent_freq)[::-1][:top_n]
            
            dependency_matrix = dependency_matrix[np.ix_(top_indices, top_indices)]
            feature_names = [feature_names[i] for i in top_indices]
        
        # Apply log scaling to match other co-occurrence plots
        matrix_display = np.log1p(dependency_matrix)
        
        # Dynamic figure size based on number of features (same as other co-occurrence plots)
        fig_size = (max(12, len(feature_names) * 0.25), max(10, len(feature_names) * 0.25))
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Use YlGnBu colormap (lighter) with grid lines to match other co-occurrence plots
        heatmap = sns.heatmap(
            matrix_display,
            xticklabels=feature_names,
            yticklabels=feature_names,
            cmap='YlGnBu',
            square=True,
            ax=ax,
            cbar_kws={'label': ''},
            linewidths=0.5,
            linecolor='black'
        )
        
        # Make colorbar text bigger (match other plots)
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
        
        # Use same font sizes as other co-occurrence plots
        ax.set_xticklabels(feature_names, rotation=45, fontsize=6, ha='right')
        ax.set_yticklabels(feature_names, fontsize=6)
        ax.set_title('Sequential Feature Co-occurrence', fontsize=20)
        
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.plotter.save_dir, 'feature_cooccurrence_sequential.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance_scatter(self, top_n: Optional[int] = None, 
                                       min_size: int = 50, max_size: int = 1000) -> None:
        """
        Plot feature importance as a scatter plot: usage (weight) vs gain, sized by cover.
        
        Args:
            top_n: Number of top features to show (None for all)
            min_size: Minimum bubble size
            max_size: Maximum bubble size
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        weight_counts, gain_distributions, cover_distributions = (
            self.plotter._compute_feature_stats(self.trees, self.feature_names)
        )
        
        # Compute statistics for each feature
        features = []
        weights = []
        avg_gains = []
        avg_covers = []
        
        for feat in weight_counts:
            if feat in gain_distributions and feat in cover_distributions:
                features.append(feat)
                weights.append(weight_counts[feat])
                avg_gains.append(np.mean(gain_distributions[feat]))
                avg_covers.append(np.mean(cover_distributions[feat]))
        
        if not features:
            print("⚠️ No feature data found")
            return
        
        # Sort by gain and take top_n
        if top_n:
            indices = np.argsort(avg_gains)[::-1][:top_n]
            features = [features[i] for i in indices]
            weights = [weights[i] for i in indices]
            avg_gains = [avg_gains[i] for i in indices]
            avg_covers = [avg_covers[i] for i in indices]
        
        # Normalize bubble sizes
        if avg_covers:
            min_cover = min(avg_covers)
            max_cover = max(avg_covers)
            if max_cover > min_cover:
                sizes = [min_size + (max_size - min_size) * (c - min_cover) / (max_cover - min_cover) 
                        for c in avg_covers]
            else:
                sizes = [min_size] * len(avg_covers)
        else:
            sizes = [min_size] * len(features)
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(weights, avg_gains, s=sizes, alpha=0.7, 
                           c='lightblue', edgecolors='black', linewidth=1)
        
        # Add feature labels (offset slightly above the markers)
        for i, feat in enumerate(features):
            ax.annotate(feat, (weights[i], avg_gains[i]), 
                       xytext=(0, 8), textcoords='offset points',
                       fontsize=8, alpha=0.8, ha='center', va='bottom')
        
        ax.set_xlabel('Feature Usage (Split Count)', fontsize=12)
        ax.set_ylabel('Average Gain (Loss Reduction)', fontsize=12)
        ax.set_title('Feature Importance: Usage vs Gain (bubble size = avg cover)', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        
        # Set log scale on y-axis (gain)
        ax.set_yscale('log')
        
        # Add legend for bubble sizes
        if len(set(sizes)) > 1:
            legend_sizes = [min_size, (min_size + max_size) / 2, max_size]
            legend_labels = [f'{min_cover:.1f}', f'{(min_cover + max_cover)/2:.1f}', f'{max_cover:.1f}']
            legend_handles = [plt.scatter([], [], s=s, c='lightblue', alpha=0.7, edgecolors='black', linewidth=1) 
                            for s in legend_sizes]
            legend = ax.legend(legend_handles, legend_labels, 
                             title='Avg Cover', loc='upper right', framealpha=0.9)
        
        plt.tight_layout()
        self.plotter._save_plot('feature_importance_scatter.png')