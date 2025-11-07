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
        self.correct_base_score = None  # Extracted from JSON to fix sklearn loading issue
        self.base_score_adjustment = None  # Logit adjustment to apply to predictions
    
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
        
        NOTE: XGBoost's sklearn API doesn't preserve base_score when loading from JSON.
        We extract the correct base_score and compute an adjustment for accurate predictions.
        
        Args:
            json_path: Path to model JSON (uses tree_analyzer's path if None)
        """
        try:
            import xgboost as xgb
            import json
        except ImportError:
            raise ImportError("XGBoost is required for model loading")
        
        if json_path is None:
            json_path = self.tree_analyzer.json_path
        
        # Extract correct base_score from JSON
        try:
            with open(json_path, 'r') as f:
                model_json = json.load(f)
                self.correct_base_score = float(
                    model_json.get('learner', {}).get('learner_model_param', {}).get('base_score', 0.5)
                )
        except Exception:
            self.correct_base_score = 0.5
        
        # Determine model type from objective
        objective = self.tree_analyzer.objective
        objective_name = objective.get("name", "") if isinstance(objective, dict) else str(objective)
        is_regression = "reg:" in objective_name or "squarederror" in objective_name
        
        # Load model
        self.xgb_model = xgb.XGBRegressor() if is_regression else xgb.XGBClassifier()
        self.xgb_model.load_model(json_path)
        
        # Initialize base_score adjustment (computed lazily when first needed)
        self.base_score_adjustment = None
        self._base_score_computed = False
        
        if not is_regression:
            print(f"📊 Correct base_score from JSON: {self.correct_base_score:.6f} "
                  f"→ {expit(self.correct_base_score):.6f} prob ({expit(self.correct_base_score)*100:.4f}%)")
        
        # Detect number of classes
        self.num_classes = self._detect_num_classes()
        
        # Print summary
        if self.num_classes > 2:
            print(f"✅ Loaded multi-class model: {self.num_classes} classes (analyzing class {self.target_class})")
        else:
            print(f"✅ Loaded XGBoost model from {os.path.basename(json_path)}")
    
    def _detect_num_classes(self) -> int:
        """Detect number of classes for classification models."""
        if hasattr(self.xgb_model, 'n_classes_'):
            return self.xgb_model.n_classes_
        elif hasattr(self.xgb_model, '_le') and hasattr(self.xgb_model._le, 'classes_'):
            return len(self.xgb_model._le.classes_)
        elif 'multi:' in str(self.tree_analyzer.objective):
            return 3  # Default for multi-class
        return 2  # Binary classification
    
    def _compute_base_score_adjustment(self, X_sample: pd.DataFrame) -> None:
        """
        Compute base_score adjustment by testing prediction with 0 trees.
        Called lazily when first needing corrected predictions.
        
        Args:
            X_sample: Sample data to test predictions
        """
        if self.base_score_adjustment is not None or self._base_score_computed:
            return
        
        try:
            import xgboost as xgb
            booster = self.xgb_model.get_booster()
            dtest = xgb.DMatrix(X_sample, feature_names=self.tree_analyzer.feature_names)
            
            # Predict with 0 trees to detect actual loaded base_score
            pred_0 = booster.predict(dtest, iteration_range=(0, 0), output_margin=True)
            loaded_base_score = pred_0[0]
            
            # Calculate adjustment
            self.base_score_adjustment = self.correct_base_score - loaded_base_score
            self._base_score_computed = True
            
            print(f"⚙️  Base_score adjustment: {self.base_score_adjustment:.6f} logit units "
                  f"(loaded: {loaded_base_score:.4f}, correct: {self.correct_base_score:.4f})")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not compute base_score adjustment: {e}")
            self.base_score_adjustment = 0.0
            self._base_score_computed = True
    
    def _get_corrected_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get predictions with base_score correction applied.
        
        Args:
            X: Input features
            
        Returns:
            Array of corrected probabilities
        """
        if not self._base_score_computed:
            self._compute_base_score_adjustment(X.head(min(100, len(X))))
        
        import xgboost as xgb
        booster = self.xgb_model.get_booster()
        dtest = xgb.DMatrix(X, feature_names=self.tree_analyzer.feature_names)
        
        # Get logits for target class
        if self.num_classes and self.num_classes > 2:
            logits = booster.predict(dtest, output_margin=True)
            logits_target = logits[:, self.target_class]
        else:
            logits_target = booster.predict(dtest, output_margin=True)
        
        # Apply adjustment and convert to probabilities
        corrected_logits = logits_target + self.base_score_adjustment
        return expit(corrected_logits)
    
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
        
        Categorical features (<=250 unique values) are plotted as bar plots.
        Continuous features use line plots with interpolated grid points.
        
        Args:
            feature_name: Name of the feature to analyze
            grid_points: Number of grid points for continuous features
            n_curves: Number of data points to use for PDP computation
            categorical_threshold: Max unique values to treat as categorical (default: 250)
        """
        self._check_data_and_model()
        
        try:
            from sklearn.inspection import partial_dependence
            from scipy.special import logit as scipy_logit
        except ImportError:
            raise ImportError("scikit-learn and scipy are required for partial dependence plots")
        
        if feature_name not in self.df.columns:
            raise ValueError(f"Feature '{feature_name}' not found in data")
        
        X_base = self.df[self.tree_analyzer.feature_names].iloc[:n_curves]
        feat_idx = self.tree_analyzer.feature_names.index(feature_name)
        
        # Auto-detect categorical features
        unique_values = sorted(self.df[feature_name].dropna().unique())
        n_unique = len(unique_values)
        is_categorical = n_unique <= categorical_threshold
        
        # Print detection info
        feature_type = "CATEGORICAL" if is_categorical else "CONTINUOUS"
        grid_info = "actual category values" if is_categorical else f"{grid_points} grid points"
        print(f"Computing PDP for '{feature_name}' (index {feat_idx})")
        print(f"  Type: {feature_type} ({n_unique} unique values), using {grid_info}")
        
        # Compute PDP using sklearn (fast), then apply base_score correction
        if not self._base_score_computed:
            self._compute_base_score_adjustment(X_base.head(min(100, len(X_base))))
        
        averaged, ice_curves, grid_values = self._compute_pdp(
            X_base, feat_idx, unique_values, is_categorical, grid_points
        )
        
        # Apply base_score correction (convert prob -> logit -> adjust -> prob)
        averaged = np.clip(averaged, 1e-7, 1 - 1e-7)
        averaged = expit(scipy_logit(averaged) + self.base_score_adjustment)
        
        if ice_curves is not None:
            ice_curves = np.clip(ice_curves, 1e-7, 1 - 1e-7)
            ice_curves = expit(scipy_logit(ice_curves) + self.base_score_adjustment)
        
        # Create and save plot
        self._plot_and_save_pdp(
            feature_name, grid_values, averaged, ice_curves, is_categorical, n_unique
        )
    
    def _compute_pdp(self, X_base, feat_idx, unique_values, is_categorical, grid_points):
        """Compute PDP using sklearn or fallback to manual computation."""
        from sklearn.inspection import partial_dependence
        
        if is_categorical:
            # Try sklearn with different API versions
            try:
                try:
                    pd_result = partial_dependence(
                        self.xgb_model, X_base, features=[feat_idx],
                        custom_values=[unique_values], kind='average'
                    )
                except TypeError:
                    pd_result = partial_dependence(
                        self.xgb_model, X_base, features=[feat_idx],
                        grid_values=[unique_values], kind='average'
                    )
                averaged = pd_result['average'][0]
                grid_values = pd_result.get('grid_values', pd_result.get('values'))[0]
                return averaged, None, grid_values
            except (TypeError, AttributeError):
                # Fallback to manual computation for old sklearn
                averaged = [
                    self._get_corrected_predictions(
                        X_base.copy().assign(**{X_base.columns[feat_idx]: val})
                    ).mean()
                    for val in unique_values
                ]
                return np.array(averaged), None, unique_values
        else:
            # Continuous features: use sklearn's optimized PDP
            pd_result = partial_dependence(
                self.xgb_model, X_base, features=[feat_idx],
                grid_resolution=grid_points, kind='both'
            )
            return pd_result['average'][0], pd_result['individual'][0], pd_result['grid_values'][0]
    
    def _plot_and_save_pdp(self, feature_name, grid_values, averaged, ice_curves, 
                           is_categorical, n_unique):
        """Create and save PDP plot."""
        fig, ax = plt.subplots(figsize=(14, 5))
        
        if is_categorical:
            # Bar plot for categorical features
            colors = ['steelblue' if val >= 0 else 'coral' for val in averaged]
            ax.bar(range(len(grid_values)), averaged, color=colors, alpha=0.7, edgecolor='black')
            
            # Format x-axis labels
            ax.set_xticks(range(len(grid_values)))
            if n_unique <= 20:
                labels = [f'{int(v)}' if v == int(v) else f'{v:.2f}' for v in grid_values]
                ax.set_xticklabels(labels, rotation=45, ha='right')
            else:
                step = max(1, n_unique // 20)
                labels = [
                    (f'{int(grid_values[i])}' if grid_values[i] == int(grid_values[i]) 
                     else f'{grid_values[i]:.2f}') if i % step == 0 else ''
                    for i in range(len(grid_values))
                ]
                ax.set_xticklabels(labels, rotation=45, ha='right')
            
            ax.set_ylabel("Average Predicted Probability")
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        else:
            # Line plot with ICE curves for continuous features
            if ice_curves is not None:
                for i, ice in enumerate(ice_curves):
                    label = "ICE curves" if i == 0 else None
                    ax.plot(grid_values, ice, color='gray', alpha=0.2, linewidth=1, label=label)
            
            ax.plot(grid_values, averaged, color='red', linestyle='--', 
                   linewidth=2.0, label="Average PDP", marker='o', markersize=4)
            ax.set_ylabel("Predicted Probability")
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()
        
        ax.set_xlabel(feature_name)
        
        # Title
        title_suffix = f" (Class {self.target_class})" if self.num_classes > 2 else ""
        type_info = f"Categorical: {n_unique} categories" if is_categorical else f"Continuous: {n_unique} unique values"
        ax.set_title(f"Partial Dependence Plot for '{feature_name}'{title_suffix}\n[{type_info}]", 
                    fontsize=11)
        
        plt.tight_layout()
        
        # Save
        pdp_dir = os.path.join(self.tree_analyzer.plotter.save_dir, 'pdp')
        os.makedirs(pdp_dir, exist_ok=True)
        filepath = os.path.join(pdp_dir, f'PDP_{feature_name}.png')
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
        
        Shows how predicted probabilities change as more trees are added.
        For multi-class models, shows probability for the target class.
        
        Args:
            tree_indices: List of tree indices to evaluate
            n_records: Number of records to analyze
        """
        self._check_data_and_model()
        
        X = self.df[self.tree_analyzer.feature_names].iloc[:n_records]
        
        if not self._base_score_computed:
            self._compute_base_score_adjustment(X.head(min(100, len(X))))
        
        import xgboost as xgb
        booster = self.xgb_model.get_booster()
        dtest = xgb.DMatrix(X, feature_names=self.tree_analyzer.feature_names)
        
        # Compute predictions at each tree index
        scores_matrix = []
        for k in tree_indices:
            if self.num_classes and self.num_classes > 2:
                num_rounds = (k + self.num_classes - 1) // self.num_classes
                logits = booster.predict(dtest, iteration_range=(0, num_rounds), output_margin=True)
                logits_target = logits[:, self.target_class]
            else:
                logits_target = booster.predict(dtest, iteration_range=(0, k), output_margin=True)
            
            # Apply base_score correction
            corrected_probs = expit(logits_target + self.base_score_adjustment)
            scores_matrix.append(corrected_probs)
        
        scores_matrix = np.array(scores_matrix).T
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Individual curves
        for i in range(scores_matrix.shape[0]):
            ax.plot(tree_indices, scores_matrix[i], 
                   color="gray", alpha=0.1, marker='o', markersize=3)
        
        # Summary statistics
        ax.plot(tree_indices, np.mean(scores_matrix, axis=0), 
               color="red", linewidth=2, linestyle='--', marker='o', label="Mean")
        ax.plot(tree_indices, np.median(scores_matrix, axis=0), 
               color="blue", linewidth=2, linestyle='--', marker='o', label="Median")
        
        ax.set_xlabel("Tree Index")
        ax.set_ylabel("Predicted Probability")
        
        title = f"Class {self.target_class} " if self.num_classes > 2 else ""
        ax.set_title(f"{title}Early Exit Score Across Trees")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.tree_analyzer.plotter.save_dir, 'scores_across_trees.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_marginal_impact_univariate(self, feature_name: str, 
                                       scale: str = "linear") -> None:
        """
        Plot marginal impact of a feature on predicted probability.
        
        For multi-class models, analyzes impact on the target class only.
        
        Args:
            feature_name: Name of the feature to analyze
            scale: Scale for x-axis ("linear" or "log")
        """
        if feature_name not in self.tree_analyzer.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in model")
        
        feat_idx = self.tree_analyzer.feature_names.index(feature_name)
        thresholds, prob_deltas, split_info = [], [], []
        global_split_counter = 0
        
        # For multi-class, analyze only trees for the target class
        trees_to_analyze = self.tree_analyzer.trees
        if self.num_classes and self.num_classes > 2:
            trees_to_analyze = [tree for i, tree in enumerate(self.tree_analyzer.trees) 
                               if i % self.num_classes == self.target_class]
            print(f"Analyzing {len(trees_to_analyze)} trees for class {self.target_class}")
        
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
                    left_child, right_child = lefts[node], rights[node]
                    
                    if left_child == -1 or right_child == -1:
                        return
                    
                    # Compute probability delta
                    delta = expit(weights[right_child]) - expit(weights[left_child])
                    
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
        
        # Print split summary
        print(f"Found {len(thresholds)} splits for feature '{feature_name}':")
        for split_global_idx, tree_idx, depth, threshold, delta in split_info[:5]:  # Show first 5
            print(f"  Split {split_global_idx}, Tree {tree_idx}, Depth {depth}: "
                  f"{feature_name} < {threshold:.4f} → Δ = {delta:.4f}")
        if len(split_info) > 5:
            print(f"  ... and {len(split_info) - 5} more")
        
        # Sort and merge nearby thresholds
        sorted_idx = sorted(range(len(thresholds)), key=lambda i: thresholds[i])
        thresholds = [thresholds[i] for i in sorted_idx]
        prob_deltas = [prob_deltas[i] for i in sorted_idx]
        
        # Merge thresholds within 0.1% of range
        threshold_range = max(thresholds) - min(thresholds) if len(thresholds) > 1 else 1.0
        merge_tolerance = 0.001 * threshold_range
        
        merged_thresholds, merged_deltas = [thresholds[0]], [prob_deltas[0]]
        for i in range(1, len(thresholds)):
            if thresholds[i] - merged_thresholds[-1] < merge_tolerance:
                # Merge: average threshold, sum deltas
                merged_thresholds[-1] = (merged_thresholds[-1] + thresholds[i]) / 2
                merged_deltas[-1] += prob_deltas[i]
            else:
                merged_thresholds.append(thresholds[i])
                merged_deltas.append(prob_deltas[i])
        
        # Build step plot
        min_val, max_val = min(merged_thresholds), max(merged_thresholds)
        margin = 0.01 * (max_val - min_val if max_val > min_val else 1.0)
        region_boundaries = [min_val - margin] + merged_thresholds + [max_val + margin]
        region_values = [0] + merged_deltas
        
        fig, ax = plt.subplots(figsize=(16, 4))
        
        # Color regions by impact
        max_abs_value = max(abs(v) for v in region_values)
        for i in range(len(region_boundaries) - 1):
            value = region_values[i]
            if value > 0:
                color, intensity = 'green', np.sqrt(abs(value) / max_abs_value)
            elif value < 0:
                color, intensity = 'red', np.sqrt(abs(value) / max_abs_value)
            else:
                color, intensity = 'gray', 0.1
            
            alpha = 0.2 + 0.7 * intensity if value != 0 else 0.1
            ax.axvspan(region_boundaries[i], region_boundaries[i + 1], color=color, 
                      alpha=alpha, linewidth=0)
        
        # Step line
        step_x = region_boundaries
        step_y = region_values + [region_values[-1]]
        ax.step(step_x, step_y, where='post', color='black', 
               label="Marginal Prediction Change", linewidth=1.5)
        ax.axhline(0, color='black', linestyle=':', linewidth=1.0)
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Δ Probability")
        
        title = (f"Marginal Impact of '{feature_name}' on Class {self.target_class} Probability" 
                if self.num_classes > 2 
                else f"Marginal Impact of '{feature_name}' on Predicted Probability")
        ax.set_title(title)
        ax.legend()
        ax.grid(False)
        
        if scale == "log":
            ax.set_xscale("log")
        ax.set_xlim(region_boundaries[0], region_boundaries[-1])
        
        plt.tight_layout()
        
        # Save
        marginal_dir = os.path.join(self.tree_analyzer.plotter.save_dir, 'marginal_impact')
        os.makedirs(marginal_dir, exist_ok=True)
        filepath = os.path.join(marginal_dir, f"marginal_impact_{feature_name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _check_data_and_model(self) -> None:
        """Check that both data and model are loaded."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data_from_parquets() first.")
        if self.xgb_model is None:
            raise ValueError("XGBoost model not loaded. Call load_xgb_model() first.")
