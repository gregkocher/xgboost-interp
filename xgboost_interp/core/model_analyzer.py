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
    
    def __init__(self, tree_analyzer: TreeAnalyzer, target_class: int = 0):
        """
        Initialize the ModelAnalyzer.
        
        Args:
            tree_analyzer: TreeAnalyzer instance for the model
            target_class: Target class index for multi-class models (default: 0)
        """
        self.tree_analyzer = tree_analyzer
        self.df = None
        self.xgb_model = None
        self.target_class = target_class
        self.num_classes = None  # Will be set when model is loaded
    
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
        
        # Determine if it's a classifier or regressor from the objective
        objective = self.tree_analyzer.objective
        if isinstance(objective, dict):
            objective_name = objective.get("name", "")
        else:
            objective_name = str(objective)
        
        # Load appropriate model type
        if "reg:" in objective_name or "squarederror" in objective_name:
            self.xgb_model = xgb.XGBRegressor()
        else:
            self.xgb_model = xgb.XGBClassifier()
        
        self.xgb_model.load_model(json_path)
        
        # Detect number of classes for multi-class models
        if hasattr(self.xgb_model, 'n_classes_'):
            self.num_classes = self.xgb_model.n_classes_
        elif hasattr(self.xgb_model, '_le') and hasattr(self.xgb_model._le, 'classes_'):
            self.num_classes = len(self.xgb_model._le.classes_)
        else:
            # Try to infer from objective
            objective_str = str(self.tree_analyzer.objective)
            if 'multi:' in objective_str:
                # Parse from objective or assume binary
                self.num_classes = 3  # Default assumption for multi-class
            else:
                self.num_classes = 2  # Binary classification
        
        if self.num_classes > 2:
            print(f"✅ Loaded XGBoost multi-class model from {json_path}")
            print(f"   Number of classes: {self.num_classes}")
            print(f"   Analyzing target class: {self.target_class}")
        else:
            print(f"✅ Loaded XGBoost model from {json_path}")
    
    def predict_in_batches(self, X: pd.DataFrame, batch_size: int = 10000,
                          base_margin: Optional[pd.Series] = None) -> np.ndarray:
        """
        Predict values in batches to avoid memory issues.
        
        Args:
            X: Input features
            batch_size: Size of prediction batches
            base_margin: Base margin for predictions
            
        Returns:
            Array of predictions (probabilities for target class in classification, values for regression)
        """
        if self.xgb_model is None:
            raise ValueError("XGBoost model not loaded. Call load_xgb_model() first.")
        
        y_pred = []
        
        for i in range(0, len(X), batch_size):
            X_batch = X.iloc[i:i + batch_size]
            
            if base_margin is not None:
                base_margin_batch = base_margin.iloc[i:i + batch_size]
            else:
                base_margin_batch = None
            
            # Use predict_proba for classification, predict for regression
            if hasattr(self.xgb_model, 'predict_proba') and 'multi:' in str(self.tree_analyzer.objective):
                y_pred_batch = self.xgb_model.predict_proba(
                    X_batch, base_margin=base_margin_batch
                )
                # Extract probability for target class
                if y_pred_batch.shape[1] == 2:
                    # Binary classification - use class 1 probability
                    y_pred.extend(y_pred_batch[:, 1])
                else:
                    # Multi-class - use specified target class
                    y_pred.extend(y_pred_batch[:, self.target_class])
            else:
                y_pred_batch = self.xgb_model.predict(
                    X_batch, base_margin=base_margin_batch
                )
                y_pred.extend(y_pred_batch)
        
        return np.array(y_pred)
    
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
        
        # For multi-class, specify target class
        if self.num_classes and self.num_classes > 2:
            print(f"  Multi-class model: computing PDP for class {self.target_class}")
            pd_result = partial_dependence(
                estimator=self.xgb_model,
                X=X_base,
                features=[feat_idx],
                grid_resolution=grid_points,
                kind='both'
            )
            # sklearn returns shape (1, n_outputs, n_grid) for multi-class
            # We need to extract the specific class
            if len(pd_result['average']) > 1:
                averaged = pd_result['average'][self.target_class]
                ice_curves = pd_result['individual'][self.target_class]
            else:
                averaged = pd_result['average'][0]
                ice_curves = pd_result['individual'][0]
            grid_values = pd_result['grid_values'][0]
        else:
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
        if self.num_classes and self.num_classes > 2:
            ax.set_title(f"Partial Dependence Plot for '{feature_name}' (Class {self.target_class})")
        else:
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
        
        For multi-class models, shows probability evolution for the target class.
        
        Args:
            tree_indices: List of tree indices to evaluate
            n_records: Number of records to analyze
        """
        self._check_data_and_model()
        
        X = self.df[self.tree_analyzer.feature_names].iloc[:n_records]
        
        scores_matrix = []
        for k in tree_indices:
            if self.num_classes and self.num_classes > 2:
                # Multi-class: iteration_range refers to boosting rounds (not total trees)
                # Each round trains num_classes trees
                # So k total trees = k // num_classes rounds
                num_rounds = k // self.num_classes
                if num_rounds == 0:
                    num_rounds = 1
                pred_proba = self.xgb_model.predict_proba(
                    X, iteration_range=(0, num_rounds)
                )
                pred_prob = pred_proba[:, self.target_class]
            else:
                # Binary: use margin and sigmoid
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
        ax.set_ylabel("Predicted Probability")
        if self.num_classes and self.num_classes > 2:
            ax.set_title(f"Class {self.target_class} Probability Evolution Across Trees")
        else:
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
        
        For multi-class models, this analyzes the impact on the target class probability.
        Note: For multi-class, we analyze trees for the target class only.
        
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
        
        # For multi-class models, we need to look at trees for the target class only
        # XGBoost stores trees in round-robin fashion: tree0->class0, tree1->class1, tree2->class2, tree3->class0, etc.
        trees_to_analyze = self.tree_analyzer.trees
        if self.num_classes and self.num_classes > 2:
            # Select only trees for the target class
            trees_to_analyze = [tree for i, tree in enumerate(self.tree_analyzer.trees) 
                               if i % self.num_classes == self.target_class]
            print(f"Analyzing {len(trees_to_analyze)} trees for class {self.target_class} (out of {len(self.tree_analyzer.trees)} total trees)")
        
        for tree_idx, tree in enumerate(trees_to_analyze):
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
                    
                    # For multi-class, weights are raw logits that get softmax-ed
                    # For binary, weights are logits that get sigmoid-ed
                    # Since we're looking at marginal changes, we use the simpler approximation
                    if self.num_classes and self.num_classes > 2:
                        # For multi-class: approximate probability change
                        # This is simplified - true impact depends on other class logits
                        delta = expit(right_val) - expit(left_val)
                    else:
                        # Binary classification
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
        if self.num_classes and self.num_classes > 2:
            ax.set_title(f"Marginal Impact of '{feature_name}' on Class {self.target_class} Probability")
        else:
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
