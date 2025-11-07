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
                               n_curves: int = 1000, categorical_threshold: int = 250) -> None:
        """
        Plot partial dependence for a feature with automatic categorical detection.
        
        Categorical features (<=250 unique values) are plotted as bar plots using
        only actual category values. Continuous features use line plots with 
        interpolated grid points.
        
        Args:
            feature_name: Name of the feature to analyze
            grid_points: Number of grid points for PDP (continuous features only)
            n_curves: Number of ICE curves to show
            categorical_threshold: Max unique values to treat as categorical (default: 250)
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
        
        # Auto-detect if feature is categorical
        unique_values = self.df[feature_name].dropna().unique()
        n_unique = len(unique_values)
        is_categorical = n_unique <= categorical_threshold
        
        if is_categorical:
            # Sort unique values for categorical features
            unique_values = sorted(unique_values)
            print(f"Computing PDP for feature '{feature_name}' (index {feat_idx})")
            print(f"  Detected as CATEGORICAL ({n_unique} unique values)")
            print(f"  Using actual category values only (no interpolation)")
        else:
            print(f"Computing PDP for feature '{feature_name}' (index {feat_idx})")
            print(f"  Detected as CONTINUOUS ({n_unique} unique values)")
            print(f"  Using {grid_points} interpolated grid points")
        
        # Compute partial dependence
        if is_categorical:
            # For categorical: use actual unique values as grid
            # Try using custom_values parameter (sklearn >= 1.7) or grid_values (sklearn 1.0-1.6)
            # Fall back to manual computation if neither works
            try:
                if self.num_classes and self.num_classes > 2:
                    print(f"  Multi-class model: computing PDP for class {self.target_class}")
                    try:
                        # Try custom_values first (sklearn >= 1.7)
                        pd_result = partial_dependence(
                            estimator=self.xgb_model,
                            X=X_base,
                            features=[feat_idx],
                            custom_values=[unique_values],  # sklearn >= 1.7
                            kind='average'
                        )
                    except TypeError:
                        # Fall back to grid_values (sklearn 1.0-1.6)
                        pd_result = partial_dependence(
                            estimator=self.xgb_model,
                            X=X_base,
                            features=[feat_idx],
                            grid_values=[unique_values],  # sklearn 1.0-1.6
                            kind='average'
                        )
                    
                    if len(pd_result['average']) > 1:
                        averaged = pd_result['average'][self.target_class]
                    else:
                        averaged = pd_result['average'][0]
                    grid_values = pd_result['grid_values'][0] if 'grid_values' in pd_result else pd_result['values'][0]
                    ice_curves = None
                else:
                    try:
                        # Try custom_values first (sklearn >= 1.7)
                        pd_result = partial_dependence(
                            estimator=self.xgb_model,
                            X=X_base,
                            features=[feat_idx],
                            custom_values=[unique_values],
                            kind='average'
                        )
                    except TypeError:
                        # Fall back to grid_values (sklearn 1.0-1.6)
                        pd_result = partial_dependence(
                            estimator=self.xgb_model,
                            X=X_base,
                            features=[feat_idx],
                            grid_values=[unique_values],
                            kind='average'
                        )
                    
                    averaged = pd_result['average'][0]
                    grid_values = pd_result['grid_values'][0] if 'grid_values' in pd_result else pd_result['values'][0]
                    ice_curves = None
            except TypeError as e:
                # Fallback for very old sklearn versions - manual computation
                if 'grid_values' in str(e) or 'custom_values' in str(e):
                    print(f"  Note: Using manual PDP computation (sklearn < 1.0 detected)")
                    # Manually compute PDP for each category value
                    averaged = []
                    grid_values = unique_values
                    
                    for cat_value in unique_values:
                        # Create modified dataset with this category value for all samples
                        X_modified = X_base.copy()
                        X_modified.iloc[:, feat_idx] = cat_value
                        
                        # Get predictions
                        if self.num_classes and self.num_classes > 2:
                            preds = self.xgb_model.predict_proba(X_modified)[:, self.target_class]
                        else:
                            preds = self.xgb_model.predict(X_modified)
                        
                        # Average prediction for this category
                        averaged.append(preds.mean())
                    
                    averaged = np.array(averaged)
                    ice_curves = None
                else:
                    raise  # Re-raise if it's a different error
        else:
            # For continuous: use regular grid with ICE curves
            if self.num_classes and self.num_classes > 2:
                print(f"  Multi-class model: computing PDP for class {self.target_class}")
                pd_result = partial_dependence(
                    estimator=self.xgb_model,
                    X=X_base,
                    features=[feat_idx],
                    grid_resolution=grid_points,
                    kind='both'
                )
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
        
        if is_categorical:
            # Bar plot for categorical features
            colors = ['steelblue' if val >= 0 else 'coral' for val in averaged]
            bars = ax.bar(range(len(grid_values)), averaged, color=colors, alpha=0.7, edgecolor='black')
            
            # Set x-tick labels to actual category values
            ax.set_xticks(range(len(grid_values)))
            if n_unique <= 20:
                # Show all labels if few categories
                ax.set_xticklabels([f'{int(v)}' if v == int(v) else f'{v:.2f}' 
                                   for v in grid_values], rotation=45, ha='right')
            else:
                # Show every nth label for many categories
                step = max(1, n_unique // 20)
                labels = [f'{int(grid_values[i])}' if grid_values[i] == int(grid_values[i]) 
                         else f'{grid_values[i]:.2f}' 
                         if i % step == 0 else '' 
                         for i in range(len(grid_values))]
                ax.set_xticklabels(labels, rotation=45, ha='right')
            
            ax.set_ylabel("Average Predicted Probability")
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            
            # Add a horizontal line at y=0 for reference
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            
        else:
            # Line plot for continuous features with ICE curves
            if ice_curves is not None:
                # ICE curves (gray, transparent)
                for i, ice in enumerate(ice_curves):
                    if i == 0:
                        ax.plot(grid_values, ice, color='gray', alpha=0.2, linewidth=1, label="ICE curves")
                    else:
                        ax.plot(grid_values, ice, color='gray', alpha=0.2, linewidth=1)
            
            # Average PDP (red line)
            ax.plot(grid_values, averaged, color='red', linestyle='--', 
                   linewidth=2.0, label="Average PDP", marker='o', markersize=4)
            
            ax.set_ylabel("Predicted Probability")
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()
        
        ax.set_xlabel(feature_name)
        
        # Set title
        if self.num_classes and self.num_classes > 2:
            title = f"Partial Dependence Plot for '{feature_name}' (Class {self.target_class})"
        else:
            title = f"Partial Dependence Plot for '{feature_name}'"
        
        if is_categorical:
            title += f"\n[Categorical: {n_unique} categories]"
        else:
            title += f"\n[Continuous: {n_unique} unique values]"
        
        ax.set_title(title, fontsize=11)
        
        plt.tight_layout()
        
        # Save plot in PDP subdirectory
        pdp_dir = os.path.join(self.tree_analyzer.plotter.save_dir, 'pdp')
        os.makedirs(pdp_dir, exist_ok=True)
        filename = f'PDP_{feature_name}.png'
        filepath = os.path.join(pdp_dir, filename)
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
        
        # Save plot in ALE subdirectory
        ale_dir = os.path.join(self.tree_analyzer.plotter.save_dir, 'ale')
        os.makedirs(ale_dir, exist_ok=True)
        filename = f'ALE_{feature_name}.png'
        filepath = os.path.join(ale_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_scores_across_trees(self, tree_indices: List[int], 
                                n_records: int = 1000) -> None:
        """
        Plot prediction probability evolution at different tree stopping points.
        
        Shows how predicted probabilities change as more trees are added to the ensemble.
        Works for both binary and multi-class classification.
        
        For multi-class models, shows probability evolution for the target class.
        For binary models, shows probability evolution for the positive class.
        
        Args:
            tree_indices: List of tree indices to evaluate
            n_records: Number of records to analyze
        """
        self._check_data_and_model()
        
        X = self.df[self.tree_analyzer.feature_names].iloc[:n_records]
        
        scores_matrix = []
        for k in tree_indices:
            if self.num_classes and self.num_classes > 2:
                # Multi-class: iteration_range refers to boosting rounds
                # Each round trains num_classes trees (one per class)
                # Tree index k corresponds to round ceil(k/num_classes)
                num_rounds = (k + self.num_classes - 1) // self.num_classes  # Ceiling division
                pred_proba = self.xgb_model.predict_proba(
                    X, iteration_range=(0, num_rounds)
                )
                pred_prob = pred_proba[:, self.target_class]
            else:
                # Binary/Regression: iteration_range is actual tree count
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
            ax.set_title(f"Class {self.target_class} Early Exit Score Across Trees")
        else:
            # Binary classification - show probability evolution too
            ax.set_title("Early Exit Score Across Trees (Binary Classification)")
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
                    left_child = lefts[node]
                    right_child = rights[node]
                    
                    # Skip if either child is a leaf (shouldn't happen for valid splits)
                    if left_child == -1 or right_child == -1:
                        return
                    
                    left_val = weights[left_child]
                    right_val = weights[right_child]
                    
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
        
        # Merge very close thresholds to avoid ringing (within 0.1% of range)
        min_val, max_val = min(thresholds), max(thresholds)
        threshold_range = max_val - min_val if max_val > min_val else 1.0
        merge_tolerance = 0.001 * threshold_range
        
        merged_thresholds = []
        merged_deltas = []
        
        if thresholds:
            merged_thresholds.append(thresholds[0])
            merged_deltas.append(prob_deltas[0])
            
            for i in range(1, len(thresholds)):
                if thresholds[i] - merged_thresholds[-1] < merge_tolerance:
                    # Merge: average the threshold and sum the deltas
                    merged_thresholds[-1] = (merged_thresholds[-1] + thresholds[i]) / 2
                    merged_deltas[-1] += prob_deltas[i]
                else:
                    merged_thresholds.append(thresholds[i])
                    merged_deltas.append(prob_deltas[i])
        
        thresholds = merged_thresholds
        prob_deltas = merged_deltas
        
        # Create step plot regions - update min/max after merging
        if thresholds:
            min_val, max_val = min(thresholds), max(thresholds)
            threshold_range = max_val - min_val if max_val > min_val else 1.0
        
        margin = 0.01 * threshold_range if threshold_range > 0 else 0.01
        start = min_val - margin
        end = max_val + margin
        
        # Build step function: each region has a constant delta value
        # Region before first threshold: 0 (no splits yet)
        # Region after threshold i: uses delta at that threshold
        region_boundaries = [start] + thresholds + [end]
        region_values = [0] + prob_deltas  # First region = 0, then each region uses its delta
        
        fig, ax = plt.subplots(figsize=(16, 4))
        
        # Calculate better color scale based on actual value range
        all_values = region_values
        max_abs_value = max(abs(v) for v in all_values) if all_values else 1.0
        
        # Color regions based on impact with better gradation
        for i in range(len(region_boundaries) - 1):
            value = region_values[i]
            
            if value > 0:
                color = 'green'
                # Better scaling: use square root for more gradation
                intensity = np.sqrt(abs(value) / max_abs_value) if max_abs_value > 0 else 0
                alpha = 0.2 + 0.7 * intensity  # Range: 0.2 to 0.9
            elif value < 0:
                color = 'red'
                intensity = np.sqrt(abs(value) / max_abs_value) if max_abs_value > 0 else 0
                alpha = 0.2 + 0.7 * intensity  # Range: 0.2 to 0.9
            else:
                color = 'gray'
                alpha = 0.1
            
            ax.axvspan(region_boundaries[i], region_boundaries[i + 1], color=color, alpha=alpha, linewidth=0)
        
        # Step plot: add final point to complete the last horizontal segment
        # Step plot needs all boundaries and values to show complete steps
        step_x = region_boundaries  # Include the final boundary
        step_y = region_values + [region_values[-1]]  # Extend last value to final boundary
        ax.step(step_x, step_y, where='post', color='black', 
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
        
        ax.set_xlim(region_boundaries[0], region_boundaries[-1])
        
        plt.tight_layout()
        
        # Save plot in marginal_impact subdirectory
        marginal_dir = os.path.join(self.tree_analyzer.plotter.save_dir, 'marginal_impact')
        os.makedirs(marginal_dir, exist_ok=True)
        filename = f"marginal_impact_{feature_name}.png"
        filepath = os.path.join(marginal_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _check_data_and_model(self) -> None:
        """Check that both data and model are loaded."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data_from_parquets() first.")
        if self.xgb_model is None:
            raise ValueError("XGBoost model not loaded. Call load_xgb_model() first.")
