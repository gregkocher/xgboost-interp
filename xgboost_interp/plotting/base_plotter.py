"""
Base plotting utilities and common functionality.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from collections import Counter, defaultdict


class BasePlotter:
    """Base class for all plotting functionality with common utilities."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize the base plotter.
        
        Args:
            save_dir: Directory to save plots. If None, uses current directory.
        """
        self.save_dir = save_dir or "."
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _save_plot(self, filename: str) -> None:
        """Save the current plot to the specified directory."""
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _setup_plot(self, figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
        """Setup a standard plot with consistent styling."""
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    
    def _format_feature_plot(self, ax: plt.Axes, features: List[str], 
                           title: str, xlabel: str, ylabel: str = None) -> None:
        """Apply consistent formatting to feature-based plots."""
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10)
        
        # Format y-axis labels for feature names
        if len(features) > 0:
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=8)
            ax.invert_yaxis()
        
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
    
    def _compute_feature_stats(self, trees: List[Dict], feature_names: List[str]) -> Tuple[Counter, defaultdict, defaultdict]:
        """
        Compute feature statistics across all trees.
        
        Returns:
            Tuple of (weight_counts, gain_distributions, cover_distributions)
        """
        weight_counts = Counter()
        gain_distributions = defaultdict(list)
        cover_distributions = defaultdict(list)
        
        for tree in trees:
            split_indices = tree.get("split_indices", [])
            loss_changes = tree.get("loss_changes", [])
            sum_hessians = tree.get("sum_hessian", [])
            left_children = tree.get("left_children", [])
            
            for i, feat_idx in enumerate(split_indices):
                if left_children[i] == -1:  # Skip leaf nodes
                    continue
                
                if 0 <= feat_idx < len(feature_names):
                    feat_name = feature_names[feat_idx]
                    weight_counts[feat_name] += 1
                    gain_distributions[feat_name].append(loss_changes[i])
                    cover_distributions[feat_name].append(sum_hessians[i])
        
        return weight_counts, gain_distributions, cover_distributions
    
    def _compute_feature_stats_by_depth(
        self, trees: List[Dict], feature_names: List[str]
    ) -> Dict[int, Tuple[Counter, defaultdict, defaultdict]]:
        """
        Compute feature statistics grouped by node depth in a single pass.
        
        Returns:
            Dict mapping depth -> (weight_counts, gain_distributions, cover_distributions).
            Only depths that have at least one split are included.
        """
        from collections import deque
        
        stats_by_depth: Dict[int, Tuple[Counter, defaultdict, defaultdict]] = {}
        
        for tree in trees:
            split_indices = tree.get("split_indices", [])
            loss_changes = tree.get("loss_changes", [])
            sum_hessians = tree.get("sum_hessian", [])
            left_children = tree.get("left_children", [])
            right_children = tree.get("right_children", [])
            
            if not left_children:
                continue
            
            # BFS to compute depth of every node
            node_depths = [0] * len(left_children)
            queue = deque([0])  # start at root
            while queue:
                node = queue.popleft()
                left = left_children[node]
                right = right_children[node]
                if left != -1 and left < len(left_children):
                    node_depths[left] = node_depths[node] + 1
                    queue.append(left)
                if right != -1 and right < len(left_children):
                    node_depths[right] = node_depths[node] + 1
                    queue.append(right)
            
            # Accumulate stats per depth for internal nodes
            for i, feat_idx in enumerate(split_indices):
                if left_children[i] == -1:  # Skip leaf nodes
                    continue
                if 0 <= feat_idx < len(feature_names):
                    depth = node_depths[i]
                    if depth not in stats_by_depth:
                        stats_by_depth[depth] = (Counter(), defaultdict(list), defaultdict(list))
                    weight_counts, gain_dists, cover_dists = stats_by_depth[depth]
                    feat_name = feature_names[feat_idx]
                    weight_counts[feat_name] += 1
                    gain_dists[feat_name].append(loss_changes[i])
                    cover_dists[feat_name].append(sum_hessians[i])
        
        return stats_by_depth
    
    def _plot_horizontal_bar(self, data: Dict[str, float], title: str, 
                           xlabel: str, filename: str, top_n: Optional[int] = None,
                           highlight_features: Optional[List[str]] = None) -> None:
        """
        Create a horizontal bar plot with consistent styling.
        
        Args:
            data: Dict mapping feature name -> value
            title: Plot title
            xlabel: X-axis label
            filename: Output filename
            top_n: Number of top features to show (None for all)
            highlight_features: Optional list of feature names to highlight.
                If provided, highlighted features are drawn in red at full
                opacity while all other bars and labels are faded.
                If None or empty, all bars are drawn normally.
        """
        sorted_items = sorted(data.items(), key=lambda x: -x[1])
        if top_n:
            sorted_items = sorted_items[:top_n]
        
        if not sorted_items:
            print(f"No data to plot for {title}")
            return
        
        features, values = zip(*sorted_items)
        features = list(features)
        values = list(values)
        
        do_highlight = bool(highlight_features)
        hl_set = set(highlight_features) if do_highlight else set()
        
        fig, ax = self._setup_plot(figsize=(10, max(6, len(features) * 0.3)))
        
        if do_highlight:
            colors = ['red' if f in hl_set else '#1f77b4' for f in features]
            alphas = [1.0 if f in hl_set else 0.2 for f in features]
            bars = ax.barh(features, values, color=colors)
            for bar, a in zip(bars, alphas):
                bar.set_alpha(a)
        else:
            ax.barh(features, values)
        
        self._format_feature_plot(ax, features, title, xlabel)
        
        # Apply highlight styling to y-tick labels after _format_feature_plot sets them
        if do_highlight:
            for lbl in ax.get_yticklabels():
                if lbl.get_text() in hl_set:
                    lbl.set_fontweight('bold')
                    lbl.set_alpha(1.0)
                    lbl.set_color('darkred')
                else:
                    lbl.set_alpha(0.25)
        
        self._save_plot(filename)
    
    def _plot_boxplot(self, data: Dict[str, List[float]], title: str, 
                     ylabel: str, filename: str, top_n: Optional[int] = None,
                     log_scale: bool = False) -> None:
        """Create a boxplot with consistent styling."""
        # Sort by mean values
        mean_vals = {f: np.mean(v or [0]) for f, v in data.items()}
        sorted_feats = sorted(mean_vals, key=lambda f: -mean_vals[f])
        if top_n:
            sorted_feats = sorted_feats[:top_n]
        
        distributions = [data[f] for f in sorted_feats]
        
        if not any(distributions):
            print(f"No data to plot for {title}")
            return
        
        fig, ax = self._setup_plot(figsize=(max(16, len(sorted_feats) * 0.25), 8))
        ax.boxplot(distributions, vert=True, patch_artist=True, 
                  showfliers=False, showmeans=True)
        
        ax.set_xticks(range(1, len(sorted_feats) + 1))
        ax.set_xticklabels(sorted_feats, rotation=45, fontsize=6, ha="right")
        
        if log_scale:
            ax.set_yscale("log")
            ylabel += " (log scale)"
        
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        self._save_plot(filename)
    
    def _normalize_importance_dict(self, importance_dict: Dict[str, float]) -> Dict[str, float]:
        """Normalize importance values to 0-1 range."""
        if not importance_dict:
            return {}
        
        max_val = max(importance_dict.values())
        if max_val == 0:
            return {k: 0 for k in importance_dict}
        
        return {k: v / max_val for k, v in importance_dict.items()}
