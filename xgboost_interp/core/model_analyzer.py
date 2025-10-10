"""
Model analysis with data for XGBoost interpretability.

This module provides functionality to analyze XGBoost models using actual data,
including partial dependence plots, ALE plots, and prediction analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Union
from scipy.special import expit

from .tree_analyzer import TreeAnalyzer
from ..utils.data_utils import DataLoader


class ModelAnalyzer:
    """
    Analyze XGBoost models using actual data examples.
    
    This class extends tree-level analysis with data-dependent methods
    like partial dependence plots and prediction analysis.
    """
    
    def __init__(self, tree_analyzer: TreeAnalyzer):
        """
        Initialize the ModelAnalyzer.
        
        Args:
            tree_analyzer: TreeAnalyzer instance for the model
        """
        self.tree_analyzer = tree_analyzer
        self.df = None
        self.xgb_model = None
    
    def load_data_from_parquets(self, data_dir_path: str, 
                               cols_to_load: Optional[List[str]] = None,
                               num_files_to_read: int = 1000) -> None:
        """
        Load data from parquet files.
        
        Args:
            data_dir_path: Directory containing parquet files
            cols_to_load: List of columns to load (None for all)
            num_files_to_read: Maximum number of files to read
        """
        self.df = DataLoader.load_parquet_files(
            data_dir_path, cols_to_load, num_files_to_read
        )
    
    def load_xgb_model(self, json_path: Optional[str] = None) -> None:
        """
        Load XGBoost model for predictions.
        
        Args:
            json_path: Path to model JSON (uses tree_analyzer's path if None)
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost is required for model loading")
        
        if json_path is None:
            json_path = self.tree_analyzer.json_path
        
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(json_path)
        print(f"✅ Loaded XGBoost model from {json_path}")
    
    def predict_in_batches(self, X: pd.DataFrame, batch_size: int = 10000,
                          base_margin: Optional[pd.Series] = None) -> np.ndarray:
        """
        Predict probabilities in batches to avoid memory issues.
        
        Args:
            X: Input features
            batch_size: Size of prediction batches
            base_margin: Base margin for predictions
            
        Returns:
            Array of predicted probabilities
        """
        if self.xgb_model is None:
            raise ValueError("XGBoost model not loaded. Call load_xgb_model() first.")
        
        y_pred_prob = []
        
        for i in range(0, len(X), batch_size):
            X_batch = X.iloc[i:i + batch_size]
            
            if base_margin is not None:
                base_margin_batch = base_margin.iloc[i:i + batch_size]
            else:
                base_margin_batch = None
            
            y_pred_batch = self.xgb_model.predict_proba(
                X_batch, base_margin=base_margin_batch
            )
            y_pred_prob.extend(y_pred_batch[:, 1])
        
        return np.array(y_pred_prob)
    
    def plot_partial_dependence(self, feature_name: str, grid_points: int = 20,
                               n_curves: int = 1000) -> None:
        """
        Plot partial dependence for a feature.
        
        Args:
            feature_name: Name of the feature to analyze
            grid_points: Number of grid points for PDP
            n_curves: Number of ICE curves to show
        """
        self._check_data_and_model()
        
        try:
            from sklearn.inspection import partial_dependence
        except ImportError:
            raise ImportError("scikit-learn is required for partial dependence plots")
        
        if feature_name not in self.df.columns:
            raise ValueError(f"Feature '{feature_name}' not found in data")
        
        X_base = self.df[self.tree_analyzer.feature_names].iloc[:n_curves]
        feat_idx = self.tree_analyzer.feature_names.index(feature_name)
        
        print(f"Computing PDP for feature '{feature_name}' (index {feat_idx})")
        
        pd_result = partial_dependence(
            estimator=self.xgb_model,
            X=X_base,
            features=[feat_idx],
            grid_resolution=grid_points,
            kind='both'
        )
        
        averaged = pd_result['average'][0]
        ice_curves = pd_result['individual'][0]
        grid_values = pd_result['grid_values'][0]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # ICE curves (gray, transparent)
        for ice in ice_curves:
            ax.plot(grid_values, ice, color='gray', alpha=0.2, linewidth=1)
        
        # Average PDP (red dotted line)
        ax.plot(grid_values, averaged, color='red', linestyle='--', 
               linewidth=2.0, label="Average PDP")
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Predicted Probability")
        ax.set_title(f"Partial Dependence Plot for '{feature_name}'")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        filename = f'PDP_{feature_name}.png'
        filepath = os.path.join(self.tree_analyzer.plotter.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_ale(self, feature_name: str, grid_size: int = 50,
                include_CI: bool = True, confidence: float = 0.95,
                n_curves: int = 10000) -> None:
        """
        Plot Accumulated Local Effects (ALE) for a feature.
        
        Args:
            feature_name: Name of the feature to analyze
            grid_size: Number of grid points
            include_CI: Whether to include confidence intervals
            confidence: Confidence level for intervals
            n_curves: Number of data points to use
        """
        self._check_data_and_model()
        
        try:
            from pyALE import ale
        except ImportError:
            raise ImportError("PyALE is required for ALE plots")
        
        if feature_name not in self.df.columns:
            raise ValueError(f"Feature '{feature_name}' not found in data")
        
        X_sample = self.df[self.tree_analyzer.feature_names].iloc[:n_curves]
        
        print(f"Computing ALE for feature '{feature_name}' "
              f"(grid_size={grid_size}, CI={include_CI}, n_curves={n_curves})")
        
        ale_eff = ale(
            X=X_sample,
            model=self.xgb_model,
            feature=[feature_name],
            grid_size=grid_size,
            include_CI=include_CI,
            C=confidence,
        )
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 5))
        
        ax.plot(ale_eff['feature_values'], ale_eff['ale_values'], 
               color='blue', linewidth=2, label='ALE')
        
        if include_CI:
            ax.fill_between(
                ale_eff['feature_values'],
                ale_eff['ale_values'] - ale_eff['ale_values_std'],
                ale_eff['ale_values'] + ale_eff['ale_values_std'],
                color='gray', alpha=0.2,
                label=f"{int(confidence * 100)}% CI"
            )
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel("ALE (Accumulated Local Effect)")
        ax.set_title(f"ALE Plot for '{feature_name}'")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        filename = f'ALE_{feature_name}.png'
        filepath = os.path.join(self.tree_analyzer.plotter.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_scores_across_trees(self, tree_indices: List[int], 
                                n_records: int = 1000) -> None:
        """
        Plot prediction scores at different tree stopping points.
        
        Args:
            tree_indices: List of tree indices to evaluate
            n_records: Number of records to analyze
        """
        self._check_data_and_model()
        
        X = self.df[self.tree_analyzer.feature_names].iloc[:n_records]
        
        scores_matrix = []
        for k in tree_indices:
            pred_logit = self.xgb_model.predict(
                X, iteration_range=(0, k), output_margin=True
            )
            pred_prob = expit(pred_logit)
            scores_matrix.append(pred_prob)
        
        scores_matrix = np.array(scores_matrix).T
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Plot individual curves
        for i in range(scores_matrix.shape[0]):
            ax.plot(tree_indices, scores_matrix[i], 
                   color="gray", alpha=0.1, marker='o', markersize=3)
        
        # Plot summary statistics
        avg_scores = np.mean(scores_matrix, axis=0)
        median_scores = np.median(scores_matrix, axis=0)
        
        ax.plot(tree_indices, avg_scores, color="red", linewidth=2, 
               linestyle='--', marker='o', label="Mean")
        ax.plot(tree_indices, median_scores, color="blue", linewidth=2, 
               linestyle='--', marker='o', label="Median")
        
        ax.set_xlabel("Tree Index")
        ax.set_ylabel("Predicted Score")
        ax.set_title("Predicted Score at Early Exits")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        filename = 'scores_across_trees.png'
        filepath = os.path.join(self.tree_analyzer.plotter.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_marginal_impact_univariate(self, feature_name: str, 
                                       scale: str = "linear") -> None:
        """
        Plot marginal impact of a feature on predicted probability.
        
        Args:
            feature_name: Name of the feature to analyze
            scale: Scale for x-axis ("linear" or "log")
        """
        if feature_name not in self.tree_analyzer.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in model")
        
        feat_idx = self.tree_analyzer.feature_names.index(feature_name)
        thresholds, prob_deltas = [], []
        split_info = []
        
        global_split_counter = 0
        
        for tree_idx, tree in enumerate(self.tree_analyzer.trees):
            split_indices = tree.get("split_indices", [])
            split_conditions = tree.get("split_conditions", [])
            lefts = tree.get("left_children", [])
            rights = tree.get("right_children", [])
            weights = tree.get("base_weights", [])
            
            def dfs(node: int, depth: int) -> None:
                nonlocal global_split_counter
                if node == -1 or node >= len(split_indices):
                    return
                
                if split_indices[node] == feat_idx:
                    threshold = split_conditions[node]
                    left_val = weights[lefts[node]] if lefts[node] != -1 else 0
                    right_val = weights[rights[node]] if rights[node] != -1 else 0
                    delta = expit(right_val) - expit(left_val)
                    
                    thresholds.append(threshold)
                    prob_deltas.append(delta)
                    split_info.append((global_split_counter, tree_idx, depth, threshold, delta))
                    global_split_counter += 1
                
                dfs(lefts[node], depth + 1)
                dfs(rights[node], depth + 1)
            
            dfs(0, 0)
        
        if not thresholds:
            print(f"No splits found for feature '{feature_name}'")
            return
        
        # Print split information
        print(f"Splits for feature '{feature_name}':")
        for split_global_idx, tree_idx, depth, threshold, delta in split_info:
            print(f"  Split {split_global_idx}, Tree {tree_idx}, Depth {depth}: "
                  f"{feature_name} < {threshold:.4f} → Δ prediction: {delta:.4f}")
        
        # Sort by threshold
        sorted_indices = sorted(range(len(thresholds)), key=lambda i: thresholds[i])
        thresholds = [thresholds[i] for i in sorted_indices]
        prob_deltas = [prob_deltas[i] for i in sorted_indices]
        
        # Create step plot
        min_val, max_val = min(thresholds), max(thresholds)
        margin = 0.01 * (max_val - min_val) if max_val > min_val else 0.01
        start = min_val - margin
        end = max_val + margin
        
        regions = [start] + thresholds + [end]
        values = [0] + prob_deltas
        
        fig, ax = plt.subplots(figsize=(16, 4))
        
        # Color regions based on positive/negative impact
        for i in range(len(regions) - 1):
            color = 'green' if values[i] > 0 else 'red'
            alpha = min(0.9, 0.1 + abs(values[i]) * 10)
            ax.axvspan(regions[i], regions[i + 1], color=color, alpha=alpha, linewidth=0)
        
        # Step plot
        ax.step(regions[:-1], values, where='post', color='black', 
               label="Marginal Prediction Change", linewidth=1.5)
        ax.axhline(0, color='black', linestyle=':', linewidth=1.0)
        
        ax.set_ylabel("Δ Probability")
        ax.set_xlabel(feature_name)
        ax.set_title(f"Marginal Impact of Feature '{feature_name}' on Predicted Probability")
        ax.legend()
        ax.grid(False)
        
        if scale == "log":
            ax.set_xscale("log")
        
        ax.set_xlim(regions[0], regions[-1])
        
        plt.tight_layout()
        
        # Save plot
        filename = f"marginal_impact_{feature_name}.png"
        filepath = os.path.join(self.tree_analyzer.plotter.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _check_data_and_model(self) -> None:
        """Check that both data and model are loaded."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data_from_parquets() first.")
        if self.xgb_model is None:
            raise ValueError("XGBoost model not loaded. Call load_xgb_model() first.")
