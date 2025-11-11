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
                               n_curves: int = 1000, categorical_threshold: int = 250,
                               mode: str = "raw") -> None:
        """
        Plot partial dependence for a feature with automatic categorical detection.
        
        Categorical features (<=250 unique values) are plotted as bar plots.
        Continuous features use line plots with interpolated grid points.
        
        Args:
            feature_name: Name of the feature to analyze
            grid_points: Number of grid points for continuous features
            n_curves: Number of data points to use for PDP computation
            categorical_threshold: Max unique values to treat as categorical (default: 250)
            mode: "raw" (default), "probability", or "logit" - Y-axis scale
                - "raw": Probability without base_score correction (original behavior)
                - "probability": Probability with base_score correction
                - "logit": Logit with base_score correction
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
        
        # Compute PDP using sklearn (fast)
        averaged, ice_curves, grid_values = self._compute_pdp(
            X_base, feat_idx, unique_values, is_categorical, grid_points
        )
        
        # Apply transformations based on mode
        if mode == "raw":
            # Raw mode: use sklearn's probabilities directly (no base_score correction)
            pass  # averaged and ice_curves are already probabilities from sklearn
        
        elif mode == "probability" or mode == "logit":
            # For corrected modes: apply base_score correction
            if not self._base_score_computed:
                self._compute_base_score_adjustment(X_base.head(min(100, len(X_base))))
            
            # Convert prob -> logit -> adjust
            averaged = np.clip(averaged, 1e-7, 1 - 1e-7)
            averaged_logits = scipy_logit(averaged) + self.base_score_adjustment
            
            if ice_curves is not None:
                ice_curves = np.clip(ice_curves, 1e-7, 1 - 1e-7)
                ice_logits = scipy_logit(ice_curves) + self.base_score_adjustment
            else:
                ice_logits = None
            
            # Convert to final scale
            if mode == "probability":
                averaged = expit(averaged_logits)
                ice_curves = expit(ice_logits) if ice_logits is not None else None
            else:  # mode == "logit"
                averaged = averaged_logits
                ice_curves = ice_logits
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'raw', 'probability', or 'logit'")
        
        # Create and save plot
        self._plot_and_save_pdp(
            feature_name, grid_values, averaged, ice_curves, is_categorical, n_unique, mode
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
                           is_categorical, n_unique, mode):
        """Create and save PDP plot."""
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Set ylabel based on mode
        if mode == "logit":
            ylabel = "Predicted Logit"
        elif mode == "raw":
            ylabel = "Model Score (no base_score)"
        else:  # probability
            ylabel = "Predicted Probability"
        
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
            
            ax.set_ylabel(f"Average {ylabel}")
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
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()
        
        # Hide Y-axis tick labels in raw mode (values aren't actual probabilities)
        if mode == "raw":
            ax.set_yticklabels([])
        
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
    
    def _get_predictions_at_tree_index(self, X: pd.DataFrame, tree_index: int) -> np.ndarray:
        """
        Get raw logit predictions at a specific tree index.
        
        Args:
            X: Input features
            tree_index: Tree index to stop at
            
        Returns:
            Array of raw logit predictions
        """
        import xgboost as xgb
        booster = self.xgb_model.get_booster()
        dtest = xgb.DMatrix(X, feature_names=self.tree_analyzer.feature_names)
        
        if self.num_classes and self.num_classes > 2:
            num_rounds = (tree_index + self.num_classes - 1) // self.num_classes
            logits = booster.predict(dtest, iteration_range=(0, num_rounds), output_margin=True)
            logits_target = logits[:, self.target_class]
        else:
            logits_target = booster.predict(dtest, iteration_range=(0, tree_index), output_margin=True)
        
        return logits_target
    
    def plot_scores_across_trees(self, tree_indices: List[int], 
                                n_records: int = 1000, mode: str = "raw") -> None:
        """
        Plot prediction evolution at different tree stopping points.
        
        Shows how predictions change as more trees are added.
        For multi-class models, shows predictions for the target class.
        
        Args:
            tree_indices: List of tree indices to evaluate
            n_records: Number of records to analyze
            mode: "raw" (default), "probability", or "logit" - Y-axis scale
                - "raw": Probability without base_score correction (original behavior)
                - "probability": Probability with base_score correction
                - "logit": Logit with base_score correction
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
            
            # Apply transformations based on mode
            if mode == "raw":
                # Raw mode: convert to probability without base_score correction
                scores = expit(logits_target)
            
            elif mode == "probability" or mode == "logit":
                # For corrected modes: apply base_score correction
                if not self._base_score_computed:
                    self._compute_base_score_adjustment(X.head(min(100, len(X))))
                
                corrected_logits = logits_target + self.base_score_adjustment
                
                if mode == "probability":
                    scores = expit(corrected_logits)
                else:  # mode == "logit"
                    scores = corrected_logits
            else:
                raise ValueError(f"Invalid mode '{mode}'. Must be 'raw', 'probability', or 'logit'")
            
            scores_matrix.append(scores)
        
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
        
        # Set ylabel based on mode
        if mode == "logit":
            ylabel = "Predicted Logit"
        elif mode == "raw":
            ylabel = "Model Score (no base_score)"
        else:  # probability
            ylabel = "Predicted Probability"
        ax.set_ylabel(ylabel)
        
        title = f"Class {self.target_class} " if self.num_classes > 2 else ""
        mode_label = "Raw" if mode == "raw" else mode.title()
        ax.set_title(f"{title}Early Exit Score Across Trees [{mode_label} Scale]")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.tree_analyzer.plotter.save_dir, 'scores_across_trees.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _compute_inversion_rate(self, scores_early: np.ndarray, scores_final: np.ndarray) -> float:
        """
        Compute the inversion rate between early exit and final scores.
        
        Args:
            scores_early: Scores at early exit point
            scores_final: Scores at final tree
            
        Returns:
            Inversion rate as a fraction (0 to 1)
        """
        n = len(scores_early)
        if n < 2:
            return 0.0
        
        inversions = 0
        total_pairs = 0
        
        # Count inversions for all pairs (i, j) where i < j
        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                # Inversion: early has i >= j, but final has i < j
                if scores_early[i] >= scores_early[j] and scores_final[i] < scores_final[j]:
                    inversions += 1
        
        return inversions / total_pairs if total_pairs > 0 else 0.0
    
    def analyze_early_exit_performance(self, 
                                      early_exit_points: List[int] = None,
                                      n_records: int = 1000,
                                      n_detailed_curves: int = 100) -> None:
        """
        Analyze early exit performance metrics and generate visualizations.
        
        Computes multiple metrics comparing early exit predictions to final predictions:
        - Inversion Rate: Fraction of pairwise rankings that flip
        - MSE: Mean squared error between early and final logit scores
        - Kendall-Tau: Rank correlation coefficient (robust to outliers)
        - Spearman: Rank correlation coefficient (monotonic relationships)
        
        Also creates scatter plots and detailed score evolution visualizations.
        
        Args:
            early_exit_points: List of tree indices for early exit (default: [100, 500, 1000, 2000, 3000, 4000])
            n_records: Number of data points for analysis (default: 1000)
            n_detailed_curves: Number of curves for detailed evolution plot (default: 100)
        """
        self._check_data_and_model()
        
        if early_exit_points is None:
            early_exit_points = [100, 500, 1000, 2000, 3000, 4000]
        
        # Get total number of trees
        total_trees = len(self.tree_analyzer.trees)
        if self.num_classes and self.num_classes > 2:
            total_trees = total_trees // self.num_classes
        
        # Filter exit points that are within the model's tree count
        early_exit_points = [ep for ep in early_exit_points if ep < total_trees]
        
        if not early_exit_points:
            print(f"Warning: No valid early exit points (model has {total_trees} trees)")
            return
        
        print('='*70)
        print('Early Exit Performance Analysis')
        print('='*70)
        
        # Sample data
        X = self.df[self.tree_analyzer.feature_names].iloc[:n_records]
        
        # Get final predictions (all trees)
        print(f'\nComputing predictions at {len(early_exit_points)} early exit points...')
        final_scores = self._get_predictions_at_tree_index(X, total_trees)
        
        # Compute metrics for each early exit point
        results = []
        all_early_scores = {}
        
        from scipy.stats import kendalltau, spearmanr
        
        for exit_point in early_exit_points:
            early_scores = self._get_predictions_at_tree_index(X, exit_point)
            all_early_scores[exit_point] = early_scores
            
            # Compute inversion rate
            inversion_rate = self._compute_inversion_rate(early_scores, final_scores)
            
            # Compute MSE
            mse = np.mean((early_scores - final_scores) ** 2)
            
            # Compute rank correlation metrics
            kendall_tau, _ = kendalltau(early_scores, final_scores)
            spearman_rho, _ = spearmanr(early_scores, final_scores)
            
            results.append({
                'Tree Index': exit_point,
                'Inversion Rate': f'{inversion_rate * 100:.2f}%',
                'MSE': f'{mse:.6f}',
                'Kendall-Tau': f'{kendall_tau:.4f}',
                'Spearman': f'{spearman_rho:.4f}'
            })
        
        # Print table
        print('\n' + '='*70)
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        print('='*70)
        
        # Save table to file
        table_path = os.path.join(self.tree_analyzer.plotter.save_dir, 'early_exit_analysis.txt')
        with open(table_path, 'w') as f:
            f.write('='*70 + '\n')
            f.write('Early Exit Performance Analysis\n')
            f.write('='*70 + '\n\n')
            f.write(df_results.to_string(index=False))
            f.write('\n' + '='*70 + '\n')
        print(f'\n✅ Saved table to: {table_path}')
        
        # Create scatter plots
        self._plot_early_exit_scatter(all_early_scores, final_scores, early_exit_points)
        
        # Create detailed evolution plot
        self._plot_detailed_evolution(X[:n_detailed_curves], total_trees)
    
    def _plot_early_exit_scatter(self, all_early_scores: dict, final_scores: np.ndarray, 
                                 early_exit_points: List[int]) -> None:
        """Create scatter plots comparing early exit vs final scores."""
        n_plots = len(early_exit_points)
        
        # Use subplots if we have multiple exit points
        if n_plots <= 3:
            fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
            if n_plots == 1:
                axes = [axes]
        else:
            rows = (n_plots + 2) // 3
            fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
            axes = axes.flatten()
        
        colors = plt.cm.viridis(np.linspace(0, 0.9, n_plots))
        
        for idx, exit_point in enumerate(early_exit_points):
            ax = axes[idx]
            early_scores = all_early_scores[exit_point]
            
            # Scatter plot
            ax.scatter(early_scores, final_scores, alpha=0.15, s=20, color=colors[idx])
            
            # Diagonal reference line
            min_val = min(early_scores.min(), final_scores.min())
            max_val = max(early_scores.max(), final_scores.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, 
                   alpha=0.7, label='Perfect Agreement')
            
            # Compute MSE for title
            mse = np.mean((early_scores - final_scores) ** 2)
            
            ax.set_xlabel(f'Score at Tree {exit_point} (logit)')
            ax.set_ylabel('Final Score (logit)')
            ax.set_title(f'Early Exit at Tree {exit_point}\nMSE: {mse:.6f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.tree_analyzer.plotter.save_dir, 'early_exit_scatter.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'✅ Saved scatter plots to: {filepath}')
    
    def _plot_detailed_evolution(self, X: pd.DataFrame, total_trees: int) -> None:
        """Create detailed score evolution plot across all trees."""
        n_curves = len(X)
        print(f'\nComputing detailed evolution for {n_curves} examples across {total_trees} trees...')
        print('   This may take a moment...')
        
        import xgboost as xgb
        booster = self.xgb_model.get_booster()
        dtest = xgb.DMatrix(X, feature_names=self.tree_analyzer.feature_names)
        
        # Compute predictions at every tree index
        # Sample trees densely (every 10th tree for efficiency if > 1000 trees)
        if total_trees > 1000:
            tree_indices = list(range(0, total_trees, 10)) + [total_trees]
        else:
            tree_indices = list(range(0, total_trees + 1))
        
        scores_matrix = []
        for tree_idx in tree_indices:
            if self.num_classes and self.num_classes > 2:
                num_rounds = max(1, (tree_idx + self.num_classes - 1) // self.num_classes)
                logits = booster.predict(dtest, iteration_range=(0, num_rounds), output_margin=True)
                logits_target = logits[:, self.target_class]
            else:
                if tree_idx == 0:
                    # For 0 trees, get base score
                    logits_target = booster.predict(dtest, iteration_range=(0, 0), output_margin=True)
                else:
                    logits_target = booster.predict(dtest, iteration_range=(0, tree_idx), output_margin=True)
            
            scores_matrix.append(logits_target)
        
        scores_matrix = np.array(scores_matrix).T
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Individual curves (low alpha)
        for i in range(scores_matrix.shape[0]):
            ax.plot(tree_indices, scores_matrix[i], color='gray', alpha=0.1, linewidth=0.8)
        
        # Summary statistics (bold)
        mean_scores = np.mean(scores_matrix, axis=0)
        median_scores = np.median(scores_matrix, axis=0)
        
        ax.plot(tree_indices, mean_scores, color='red', linewidth=2.5, 
               linestyle='-', label=f'Mean (n={n_curves})')
        ax.plot(tree_indices, median_scores, color='blue', linewidth=2.5, 
               linestyle='-', label=f'Median (n={n_curves})')
        
        ax.set_xlabel('Tree Index')
        ax.set_ylabel('Predicted Logit')
        ax.set_title(f'Detailed Score Evolution Across All Trees\n({n_curves} examples)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.tree_analyzer.plotter.save_dir, 'early_exit_detailed_evolution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'✅ Saved detailed evolution plot to: {filepath}')
    
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
                    
                    # Skip if threshold is None - this is the root cause!
                    if threshold is None:
                        return
                    
                    left_child, right_child = lefts[node], rights[node]
                    
                    if left_child == -1 or right_child == -1:
                        return
                    
                    # Check that weights exist and are valid
                    if left_child >= len(weights) or right_child >= len(weights):
                        return
                    
                    left_weight = weights[left_child]
                    right_weight = weights[right_child]
                    
                    # Skip if weights are None
                    if left_weight is None or right_weight is None:
                        return
                    
                    # Compute probability delta
                    delta = expit(right_weight) - expit(left_weight)
                    
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
        for split_global_idx, tree_idx, depth, threshold, delta in split_info[:5]:
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
        max_abs_value = max(max_abs_value, 1e-10)  # Avoid division by zero
        
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
        print(f"  ✅ Generated: marginal_impact/{feature_name}.png")
    
    def _check_data_and_model(self) -> None:
        """Check that both data and model are loaded."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data_from_parquets() first.")
        if self.xgb_model is None:
            raise ValueError("XGBoost model not loaded. Call load_xgb_model() first.")
