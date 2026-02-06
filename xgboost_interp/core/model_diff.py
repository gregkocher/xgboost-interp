"""
Model diffing for comparing two XGBoost models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Any
from scipy.stats import pearsonr, spearmanr, kendalltau, gaussian_kde

from .tree_analyzer import TreeAnalyzer
from .model_analyzer import ModelAnalyzer
from ..plotting.base_plotter import BasePlotter


class ModelDiff:
    """
    Compare two XGBoost models and identify structural differences.
    
    Diffs are directional: delta = model_b - model_a
    (positive delta means model_b has higher value)
    """
    
    def __init__(
        self,
        analyzer_a: TreeAnalyzer,
        analyzer_b: TreeAnalyzer,
        label_a: str = "Model A",
        label_b: str = "Model B",
        save_dir: Optional[str] = None,
    ):
        """
        Initialize ModelDiff.
        
        Args:
            analyzer_a: TreeAnalyzer for the baseline/old model
            analyzer_b: TreeAnalyzer for the candidate/new model
            label_a: Display label for model A
            label_b: Display label for model B
            save_dir: Directory to save plots (defaults to "model_diff_output")
        """
        self.analyzer_a = analyzer_a
        self.analyzer_b = analyzer_b
        self.label_a = label_a
        self.label_b = label_b
        
        if save_dir is None:
            save_dir = "model_diff_output"
        self.plotter = BasePlotter(save_dir)
    
    def print_summary(self) -> None:
        """Print side-by-side comparison of model metadata."""
        a = self.analyzer_a
        b = self.analyzer_b
        
        def fmt(val):
            """Format value for display, handling None."""
            return str(val) if val is not None else "N/A"
        
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Metric':<25} {self.label_a:<20} {self.label_b:<20}")
        print("-" * 70)
        print(f"{'Number of Trees':<25} {fmt(a.num_trees_total):<20} {fmt(b.num_trees_total):<20}")
        print(f"{'Max Depth':<25} {fmt(a.max_depth):<20} {fmt(b.max_depth):<20}")
        print(f"{'Learning Rate':<25} {fmt(a.learning_rate):<20} {fmt(b.learning_rate):<20}")
        print(f"{'Base Score':<25} {fmt(a.base_score):<20} {fmt(b.base_score):<20}")
        print(f"{'Objective':<25} {fmt(a.objective):<20} {fmt(b.objective):<20}")
        print(f"{'Number of Features':<25} {len(a.feature_names):<20} {len(b.feature_names):<20}")
        print("=" * 70 + "\n")
    
    @staticmethod
    def _detect_model_type(analyzer: ModelAnalyzer) -> str:
        """
        Detect model type from the analyzer's objective.
        
        Extends ModelAnalyzer.is_regression (which only distinguishes
        regression vs non-regression) to also detect ranking objectives.
        
        Returns:
            "classification", "regression", or "ranking"
        """
        objective = analyzer.tree_analyzer.objective
        obj_name = (objective.get("name", "")
                    if isinstance(objective, dict) else str(objective))
        
        if "rank:" in obj_name:
            return "ranking"
        elif "reg:" in obj_name or "squarederror" in obj_name:
            return "regression"
        else:
            return "classification"
    
    @staticmethod
    def _get_raw_scores(analyzer: ModelAnalyzer, X) -> np.ndarray:
        """
        Get raw prediction scores (no sigmoid/softmax) for regression/ranking.
        
        Uses the same booster.predict(output_margin=True) pattern that
        ModelAnalyzer._get_predictions_at_tree_index() uses.
        
        Args:
            analyzer: ModelAnalyzer with model loaded
            X: Input features DataFrame
            
        Returns:
            Array of raw scores
        """
        import xgboost as xgb
        booster = analyzer.xgb_model.get_booster()
        dtest = xgb.DMatrix(X, feature_names=analyzer.tree_analyzer.feature_names)
        return booster.predict(dtest, output_margin=True)
    
    def find_feature_changes(self) -> Dict[str, List[str]]:
        """
        Identify features that are new, dropped, or shared between models.
        
        Returns:
            Dict with keys:
            - 'new_in_b': Features in model B but not in model A
            - 'dropped_in_b': Features in model A but not in model B
            - 'in_both': Features present in both models
        """
        features_a = set(self.analyzer_a.feature_names)
        features_b = set(self.analyzer_b.feature_names)
        
        return {
            'new_in_b': sorted(features_b - features_a),
            'dropped_in_b': sorted(features_a - features_b),
            'in_both': sorted(features_a & features_b),
        }
    
    def compare_cumulative_gain(self) -> None:
        """Plot overlay of cumulative gain curves from both models."""
        # Compute cumulative gain for model A
        gains_a = []
        for tree in self.analyzer_a.trees:
            total_gain = sum(tree.get("loss_changes", []))
            gains_a.append(total_gain)
        cumulative_a = np.cumsum(gains_a)
        
        # Compute cumulative gain for model B
        gains_b = []
        for tree in self.analyzer_b.trees:
            total_gain = sum(tree.get("loss_changes", []))
            gains_b.append(total_gain)
        cumulative_b = np.cumsum(gains_b)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(range(len(cumulative_a)), cumulative_a, 
                linewidth=2, label=self.label_a, color='red')
        ax.plot(range(len(cumulative_b)), cumulative_b, 
                linewidth=2, label=self.label_b, color='blue')
        
        ax.set_xlabel("Tree Index")
        ax.set_ylabel("Cumulative Loss Change")
        ax.set_title(f"Cumulative Gain Comparison: {self.label_a} vs {self.label_b}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        self.plotter._save_plot('cumulative_gain_comparison.png')
        print("Generated: cumulative_gain_comparison.png")
    
    def plot_importance_scatter(self, metric: str = "gain") -> None:
        """
        Plot scatterplot comparing feature importance between models.
        
        X-axis: Model A importance
        Y-axis: Model B importance
        Points above diagonal = feature increased in model B
        Points below diagonal = feature decreased in model B
        
        Args:
            metric: "gain", "weight", or "cover"
        """
        # Get feature importance from both models
        weight_a, gain_a, cover_a = self.analyzer_a.get_feature_importance()
        weight_b, gain_b, cover_b = self.analyzer_b.get_feature_importance()
        
        # Select the appropriate metric
        if metric == "weight":
            values_a = dict(weight_a)
            values_b = dict(weight_b)
            metric_label = "Weight (Split Frequency)"
        elif metric == "gain":
            values_a = {f: sum(v) for f, v in gain_a.items()}
            values_b = {f: sum(v) for f, v in gain_b.items()}
            metric_label = "Gain (Total Loss Reduction)"
        elif metric == "cover":
            values_a = {f: sum(v) for f, v in cover_a.items()}
            values_b = {f: sum(v) for f, v in cover_b.items()}
            metric_label = "Cover (Total Sum Hessian)"
        else:
            raise ValueError(f"Invalid metric '{metric}'. Must be 'gain', 'weight', or 'cover'")
        
        # Get features that appear in both models
        common_features = set(values_a.keys()) & set(values_b.keys())
        
        # Build data arrays
        features = []
        x_vals = []
        y_vals = []
        
        for feat in common_features:
            val_a = values_a[feat]
            val_b = values_b[feat]
            # Skip features with zero values (can't plot on log scale)
            if val_a > 0 and val_b > 0:
                features.append(feat)
                x_vals.append(val_a)
                y_vals.append(val_b)
        
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Scatter points
        ax.scatter(x_vals, y_vals, alpha=0.7, s=50, c='steelblue', edgecolors='black', linewidth=0.5)
        
        # Add feature labels
        for i, feat in enumerate(features):
            ax.annotate(feat, (x_vals[i], y_vals[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, alpha=0.8)
        
        # y=x diagonal reference line
        min_val = min(x_vals.min(), y_vals.min())
        max_val = max(x_vals.max(), y_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'k--', linewidth=1.5, alpha=0.7, label='y = x (no change)')
        
        # Log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlabel(f"{self.label_a} - {metric_label}", fontsize=11)
        ax.set_ylabel(f"{self.label_b} - {metric_label}", fontsize=11)
        ax.set_title(f"Feature {metric.title()} Comparison\n(above line = increased in {self.label_b})", 
                    fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        
        # Make plot square
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        filename = f'importance_scatter_{metric}.png'
        self.plotter._save_plot(filename)
        print(f"Generated: {filename}")
    
    def plot_all_importance_scatters(self) -> None:
        """Generate all 3 importance scatterplots (gain, weight, cover)."""
        for metric in ["gain", "weight", "cover"]:
            self.plot_importance_scatter(metric)
    
    def compare_pdp(
        self,
        analyzer_a: ModelAnalyzer,
        analyzer_b: ModelAnalyzer,
        feature_name: str,
        n_curves: int = 1000,
        mode: str = "raw",
    ) -> None:
        """
        Plot partial dependence comparison between two models.
        
        Overlays PDP curves from both models on the same plot.
        
        Args:
            analyzer_a: ModelAnalyzer for model A (must have data and model loaded)
            analyzer_b: ModelAnalyzer for model B (must have data and model loaded)
            feature_name: Name of the feature to analyze
            n_curves: Number of data points to use for PDP computation
            mode: "raw" (default), "probability", or "logit" - Y-axis scale
        """
        # Validate both analyzers have data and model loaded
        analyzer_a._check_data_and_model()
        analyzer_b._check_data_and_model()
        
        # Get feature names for each model
        features_a = analyzer_a.tree_analyzer.feature_names
        features_b = analyzer_b.tree_analyzer.feature_names
        
        if feature_name not in features_a:
            raise ValueError(f"Feature '{feature_name}' not found in model A")
        if feature_name not in features_b:
            raise ValueError(f"Feature '{feature_name}' not found in model B")
        
        # Get feature index for each model (may differ due to different feature sets)
        feat_idx_a = features_a.index(feature_name)
        feat_idx_b = features_b.index(feature_name)
        
        # Get data samples filtered to each model's expected features
        # Use the same row indices for fair comparison
        sample_indices = list(range(min(n_curves, len(analyzer_a.df))))
        X_base_a = analyzer_a.df[features_a].iloc[sample_indices]
        X_base_b = analyzer_b.df[features_b].iloc[sample_indices]
        
        # Detect categorical vs continuous (use analyzer_a's data)
        unique_values = sorted(analyzer_a.df[feature_name].dropna().unique())
        n_unique = len(unique_values)
        is_categorical = n_unique <= 250
        
        print(f"Computing PDP comparison for '{feature_name}'")
        print(f"  Type: {'CATEGORICAL' if is_categorical else 'CONTINUOUS'} ({n_unique} unique values)")
        
        # Compute PDP for each model using its own feature-filtered data
        avg_a, ice_a, grid_values = analyzer_a._compute_pdp(
            X_base_a, feat_idx_a, unique_values, is_categorical
        )
        avg_b, ice_b, _ = analyzer_b._compute_pdp(
            X_base_b, feat_idx_b, unique_values, is_categorical
        )
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if is_categorical:
            # Bar plot for categorical features
            bar_width = 0.35
            x_pos = np.arange(len(grid_values))
            
            ax.bar(x_pos - bar_width/2, avg_a, bar_width, 
                   label=self.label_a, color='red', alpha=0.7, edgecolor='black')
            ax.bar(x_pos + bar_width/2, avg_b, bar_width, 
                   label=self.label_b, color='blue', alpha=0.7, edgecolor='black')
            
            ax.set_xticks(x_pos)
            if n_unique <= 20:
                labels = [f'{int(v)}' if v == int(v) else f'{v:.2f}' for v in grid_values]
                ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.axhline(y=0, color='black', linestyle=':', linewidth=0.5)
        else:
            # Line plot for continuous features
            # Plot individual curves (ICE) colored to match their model
            if ice_a is not None:
                for i, ice in enumerate(ice_a):
                    label = f"{self.label_a} ICE" if i == 0 else None
                    ax.plot(grid_values, ice, color='#FF9999', alpha=0.10, linewidth=0.8, label=label)
            
            if ice_b is not None:
                for i, ice in enumerate(ice_b):
                    label = f"{self.label_b} ICE" if i == 0 else None
                    ax.plot(grid_values, ice, color='#6699FF', alpha=0.10, linewidth=0.8, label=label)
            
            # Plot averages
            ax.plot(grid_values, avg_a, color='red', linestyle='-', 
                   linewidth=2.5, label=f"{self.label_a} (avg)", marker='o', markersize=3)
            ax.plot(grid_values, avg_b, color='blue', linestyle='-', 
                   linewidth=2.5, label=f"{self.label_b} (avg)", marker='o', markersize=3)
        
        # Labels and formatting
        ax.set_xlabel(feature_name, fontsize=11)
        
        if mode == "logit":
            ylabel = "Predicted Logit"
        elif mode == "raw":
            ylabel = "Model Score"
        else:
            ylabel = "Predicted Probability"
        ax.set_ylabel(ylabel, fontsize=11)
        
        ax.set_title(f"PDP Comparison: {feature_name}\n{self.label_a} vs {self.label_b}", fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filename = f'pdp_comparison_{feature_name}.png'
        self.plotter._save_plot(filename)
        print(f"Generated: {filename}")
    
    def compare_all_pdp(
        self,
        analyzer_a: ModelAnalyzer,
        analyzer_b: ModelAnalyzer,
        n_curves: int = 1000,
        mode: str = "raw",
    ) -> None:
        """
        Compare PDP for all features common to both models.
        
        Uses find_feature_changes() to identify common features, then
        calls compare_pdp() for each.
        
        Args:
            analyzer_a: ModelAnalyzer for model A (must have data and model loaded)
            analyzer_b: ModelAnalyzer for model B (must have data and model loaded)
            n_curves: Number of data points to use for PDP computation
            mode: "raw" (default), "probability", or "logit" - Y-axis scale
        """
        common_features = self.find_feature_changes()['in_both']
        print(f"Comparing PDP for {len(common_features)} common features...")
        for feature_name in common_features:
            self.compare_pdp(analyzer_a, analyzer_b, feature_name, n_curves, mode)
    
    def compare_predictions(
        self,
        analyzer_a: ModelAnalyzer,
        analyzer_b: ModelAnalyzer,
        y_true: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compare predictions from two models on the same dataset.
        
        Auto-detects model type (classification, regression, ranking) from the
        objective and produces appropriate metrics and visualizations:
        
        All model types:
        - Scatter plot of Model A score vs Model B score
        - Histogram of score differences (Model B - Model A) with KDE overlay
        - Text summary of correlation statistics (Pearson, Spearman, Kendall-tau)
        
        Classification:
        - (If y_true provided) Classification agreement confusion matrix
        
        Regression:
        - MAE, RMSE, R-squared between predictions
        - (If y_true provided) MAE/RMSE of each model vs ground truth
        - (If y_true provided) Bar chart comparing model errors
        
        Ranking:
        - (If y_true provided) Classification agreement matrix (binary labels)
        - (If y_true provided) NDCG and MRR comparison of each model vs y_true
        
        Args:
            analyzer_a: ModelAnalyzer for model A (must have data and model loaded)
            analyzer_b: ModelAnalyzer for model B (must have data and model loaded)
            y_true: Optional ground truth labels/values for task-specific metrics
            n_samples: Number of samples to use (None = use all available data)
            threshold: Classification threshold for binary decisions (default 0.5)
        
        Returns:
            Dict with correlation stats, difference stats, and task-specific
            metrics.
        """
        analyzer_a._check_data_and_model()
        analyzer_b._check_data_and_model()
        
        # ------------------------------------------------------------------
        # Detect model type
        # ------------------------------------------------------------------
        model_type_a = self._detect_model_type(analyzer_a)
        model_type_b = self._detect_model_type(analyzer_b)
        
        if model_type_a != model_type_b:
            print(f"[WARNING] Model types differ: {self.label_a}={model_type_a}, "
                  f"{self.label_b}={model_type_b}. Using {self.label_a} type.")
        model_type = model_type_a
        
        # Axis labels based on model type
        score_labels = {
            "classification": "predicted probability",
            "regression": "predicted value",
            "ranking": "predicted score",
        }
        score_label = score_labels[model_type]
        
        print(f"Model type detected: {model_type}")
        
        # ------------------------------------------------------------------
        # Get scores
        # ------------------------------------------------------------------
        features_a = analyzer_a.tree_analyzer.feature_names
        features_b = analyzer_b.tree_analyzer.feature_names
        
        n = min(len(analyzer_a.df), len(analyzer_b.df))
        if n_samples is not None:
            n = min(n_samples, n)
        X_a = analyzer_a.df[features_a].iloc[:n]
        X_b = analyzer_b.df[features_b].iloc[:n]
        
        print(f"Computing predictions for {n} samples...")
        
        if model_type == "classification":
            scores_a = analyzer_a._get_corrected_predictions(X_a)
            scores_b = analyzer_b._get_corrected_predictions(X_b)
        else:
            # Regression and ranking: use raw scores (no sigmoid)
            scores_a = self._get_raw_scores(analyzer_a, X_a)
            scores_b = self._get_raw_scores(analyzer_b, X_b)
        
        diffs = scores_b - scores_a
        
        # ------------------------------------------------------------------
        # Compute correlation statistics (universal)
        # ------------------------------------------------------------------
        r_pearson, p_pearson = pearsonr(scores_a, scores_b)
        r_spearman, p_spearman = spearmanr(scores_a, scores_b)
        r_kendall, p_kendall = kendalltau(scores_a, scores_b)
        
        stats = {
            'model_type': model_type,
            'n_samples': n,
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_r': r_spearman,
            'spearman_p': p_spearman,
            'kendall_tau': r_kendall,
            'kendall_p': p_kendall,
            'diff_mean': float(np.mean(diffs)),
            'diff_median': float(np.median(diffs)),
            'diff_std': float(np.std(diffs)),
            'diff_min': float(np.min(diffs)),
            'diff_max': float(np.max(diffs)),
            'mean_score_a': float(np.mean(scores_a)),
            'mean_score_b': float(np.mean(scores_b)),
        }
        
        # ------------------------------------------------------------------
        # Regression-specific: MAE, RMSE, R-squared between predictions
        # ------------------------------------------------------------------
        if model_type == "regression":
            mae_between = float(np.mean(np.abs(diffs)))
            rmse_between = float(np.sqrt(np.mean(diffs ** 2)))
            ss_res = np.sum((scores_b - scores_a) ** 2)
            ss_tot = np.sum((scores_b - np.mean(scores_b)) ** 2)
            r2_between = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
            
            stats['mae_between'] = mae_between
            stats['rmse_between'] = rmse_between
            stats['r2_between'] = r2_between
            
            if y_true is not None:
                y_true_arr = np.asarray(y_true[:n], dtype=float)
                stats['mae_a_vs_true'] = float(np.mean(np.abs(scores_a - y_true_arr)))
                stats['mae_b_vs_true'] = float(np.mean(np.abs(scores_b - y_true_arr)))
                stats['rmse_a_vs_true'] = float(np.sqrt(np.mean((scores_a - y_true_arr) ** 2)))
                stats['rmse_b_vs_true'] = float(np.sqrt(np.mean((scores_b - y_true_arr) ** 2)))
        
        # ------------------------------------------------------------------
        # 1. Scatter plot: Model A score (x) vs Model B score (y)
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if n > 2000:
            # Use density coloring for large sample sizes
            xy = np.vstack([scores_a, scores_b])
            density = gaussian_kde(xy)(xy)
            idx = density.argsort()
            ax.scatter(scores_a[idx], scores_b[idx], c=density[idx],
                       cmap='viridis', s=10, alpha=0.6, edgecolors='none')
        else:
            ax.scatter(scores_a, scores_b, alpha=0.5, s=15,
                       c='steelblue', edgecolors='black', linewidth=0.3)
        
        # y=x diagonal
        lo = min(scores_a.min(), scores_b.min())
        hi = max(scores_a.max(), scores_b.max())
        margin = (hi - lo) * 0.02
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'k--', linewidth=1.5, alpha=0.7, label='y = x (no change)')
        
        ax.set_xlabel(f"{self.label_a} {score_label}", fontsize=11)
        ax.set_ylabel(f"{self.label_b} {score_label}", fontsize=11)
        ax.set_title(
            f"Prediction Scatter: {self.label_a} vs {self.label_b}\n"
            f"Pearson r = {r_pearson:.4f}, Spearman r = {r_spearman:.4f}",
            fontsize=12,
        )
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        self.plotter._save_plot('prediction_scatter.png')
        print("Generated: prediction_scatter.png")
        
        # ------------------------------------------------------------------
        # 2. Histogram of score differences with KDE overlay
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(diffs, bins=80, density=True, alpha=0.6,
                color='steelblue', edgecolor='black', linewidth=0.4,
                label='Histogram')
        
        # KDE overlay
        kde = gaussian_kde(diffs)
        x_kde = np.linspace(diffs.min(), diffs.max(), 300)
        ax.plot(x_kde, kde(x_kde), color='darkred', linewidth=2, label='KDE')
        
        # Vertical line at 0
        ax.axvline(x=0, color='black', linestyle=':', linewidth=1.2, alpha=0.8,
                   label='Zero difference')
        # Vertical line at mean diff
        ax.axvline(x=stats['diff_mean'], color='red', linestyle='--',
                   linewidth=1.2, alpha=0.8,
                   label=f"Mean diff = {stats['diff_mean']:.4f}")
        
        ax.set_xlabel(f"Score Difference ({self.label_b} - {self.label_a})", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(
            f"Distribution of Prediction Differences\n"
            f"Mean = {stats['diff_mean']:.4f}, Std = {stats['diff_std']:.4f}, "
            f"Median = {stats['diff_median']:.4f}",
            fontsize=12,
        )
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        self.plotter._save_plot('prediction_diff_histogram.png')
        print("Generated: prediction_diff_histogram.png")
        
        # ------------------------------------------------------------------
        # 3. Correlation / metrics summary text file
        # ------------------------------------------------------------------
        summary_path = os.path.join(self.plotter.save_dir,
                                    'prediction_correlation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("PREDICTION COMPARISON SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model A: {self.label_a}\n")
            f.write(f"Model B: {self.label_b}\n")
            f.write(f"Model type: {model_type}\n")
            f.write(f"Samples: {n}\n\n")
            
            f.write("CORRELATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Pearson r:    {r_pearson:.6f}  (p = {p_pearson:.2e})\n")
            f.write(f"  Spearman r:   {r_spearman:.6f}  (p = {p_spearman:.2e})\n")
            f.write(f"  Kendall tau:  {r_kendall:.6f}  (p = {p_kendall:.2e})\n\n")
            
            f.write("SCORE DIFFERENCES (Model B - Model A)\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Mean:   {stats['diff_mean']:.6f}\n")
            f.write(f"  Median: {stats['diff_median']:.6f}\n")
            f.write(f"  Std:    {stats['diff_std']:.6f}\n")
            f.write(f"  Min:    {stats['diff_min']:.6f}\n")
            f.write(f"  Max:    {stats['diff_max']:.6f}\n\n")
            
            f.write("MEAN SCORES\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Model A mean: {stats['mean_score_a']:.6f}\n")
            f.write(f"  Model B mean: {stats['mean_score_b']:.6f}\n")
            
            # Regression-specific metrics in text file
            if model_type == "regression":
                f.write(f"\nREGRESSION PREDICTION COMPARISON\n")
                f.write("-" * 40 + "\n")
                f.write(f"  MAE between predictions:  {stats['mae_between']:.6f}\n")
                f.write(f"  RMSE between predictions: {stats['rmse_between']:.6f}\n")
                f.write(f"  R-squared (A predicts B): {stats['r2_between']:.6f}\n")
                
                if y_true is not None:
                    f.write(f"\n  VS GROUND TRUTH\n")
                    f.write(f"  {'Metric':<25} {self.label_a:<15} {self.label_b:<15}\n")
                    f.write(f"  {'MAE':<25} {stats['mae_a_vs_true']:<15.6f} {stats['mae_b_vs_true']:<15.6f}\n")
                    f.write(f"  {'RMSE':<25} {stats['rmse_a_vs_true']:<15.6f} {stats['rmse_b_vs_true']:<15.6f}\n")
        
        print(f"Generated: prediction_correlation_summary.txt")
        
        # ------------------------------------------------------------------
        # 4. Classification agreement matrix
        #    - Classification: always (if y_true provided)
        #    - Ranking: if y_true is binary
        #    - Regression: skipped
        # ------------------------------------------------------------------
        if model_type in ("classification", "ranking") and y_true is not None:
            y_true_arr = np.asarray(y_true[:n])
            
            # For ranking, only show agreement if labels are binary
            unique_labels = np.unique(y_true_arr)
            is_binary_labels = len(unique_labels) <= 2
            
            if model_type == "classification" or is_binary_labels:
                if model_type == "classification":
                    pred_a = (scores_a >= threshold).astype(int)
                    pred_b = (scores_b >= threshold).astype(int)
                else:
                    # For ranking, threshold on the raw scores using median
                    median_score = np.median(np.concatenate([scores_a, scores_b]))
                    pred_a = (scores_a >= median_score).astype(int)
                    pred_b = (scores_b >= median_score).astype(int)
                
                # 2x2 agreement matrix
                both_neg = int(np.sum((pred_a == 0) & (pred_b == 0)))
                both_pos = int(np.sum((pred_a == 1) & (pred_b == 1)))
                a_pos_b_neg = int(np.sum((pred_a == 1) & (pred_b == 0)))
                a_neg_b_pos = int(np.sum((pred_a == 0) & (pred_b == 1)))
                
                agreement_rate = (both_neg + both_pos) / n
                
                stats['agreement_rate'] = agreement_rate
                stats['both_positive'] = both_pos
                stats['both_negative'] = both_neg
                stats['a_pos_b_neg'] = a_pos_b_neg
                stats['a_neg_b_pos'] = a_neg_b_pos
                
                # Build confusion matrix
                matrix = np.array([[both_neg, a_neg_b_pos],
                                   [a_pos_b_neg, both_pos]])
                
                fig, ax = plt.subplots(figsize=(7, 6))
                im = ax.imshow(matrix, cmap='Blues', aspect='auto')
                
                # Labels
                labels_row = [f'{self.label_a}\npred=0', f'{self.label_a}\npred=1']
                labels_col = [f'{self.label_b} pred=0', f'{self.label_b} pred=1']
                ax.set_xticks([0, 1])
                ax.set_xticklabels(labels_col, fontsize=10)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(labels_row, fontsize=10)
                
                # Annotate cells with counts and percentages
                for i in range(2):
                    for j in range(2):
                        count = matrix[i, j]
                        pct = count / n * 100
                        ax.text(j, i, f"{count}\n({pct:.1f}%)",
                                ha='center', va='center', fontsize=13,
                                color='white' if count > n * 0.3 else 'black')
                
                threshold_display = threshold if model_type == "classification" else f"median={median_score:.4f}"
                ax.set_title(
                    f"Classification Agreement (threshold={threshold_display})\n"
                    f"Agreement rate: {agreement_rate:.1%}  |  n={n}",
                    fontsize=12,
                )
                plt.colorbar(im, ax=ax, shrink=0.8)
                
                plt.tight_layout()
                self.plotter._save_plot('prediction_agreement_matrix.png')
                print(f"Generated: prediction_agreement_matrix.png")
                
                # Append agreement info to summary file
                with open(summary_path, 'a') as f:
                    f.write(f"\nCLASSIFICATION AGREEMENT (threshold={threshold_display})\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  Agreement rate: {agreement_rate:.4%}\n")
                    f.write(f"  Both predict 0: {both_neg} ({both_neg/n*100:.1f}%)\n")
                    f.write(f"  Both predict 1: {both_pos} ({both_pos/n*100:.1f}%)\n")
                    f.write(f"  A=1, B=0:       {a_pos_b_neg} ({a_pos_b_neg/n*100:.1f}%)\n")
                    f.write(f"  A=0, B=1:       {a_neg_b_pos} ({a_neg_b_pos/n*100:.1f}%)\n")
        
        # ------------------------------------------------------------------
        # 5. Ranking-specific metrics: NDCG and MRR comparison
        # ------------------------------------------------------------------
        if model_type == "ranking" and y_true is not None:
            y_true_arr = np.asarray(y_true[:n], dtype=float)
            
            try:
                from sklearn.metrics import ndcg_score
                
                # NDCG expects 2D arrays: (n_queries, n_documents)
                # Treat the whole set as a single query
                y_true_2d = y_true_arr.reshape(1, -1)
                scores_a_2d = scores_a.reshape(1, -1)
                scores_b_2d = scores_b.reshape(1, -1)
                
                ndcg_a = ndcg_score(y_true_2d, scores_a_2d)
                ndcg_b = ndcg_score(y_true_2d, scores_b_2d)
                
                # Also compute NDCG at common cutoffs
                ndcg_at_k = {}
                for k in [10, 50, 100]:
                    if n >= k:
                        ndcg_at_k[k] = {
                            'a': ndcg_score(y_true_2d, scores_a_2d, k=k),
                            'b': ndcg_score(y_true_2d, scores_b_2d, k=k),
                        }
                
                stats['ndcg_a'] = float(ndcg_a)
                stats['ndcg_b'] = float(ndcg_b)
                stats['ndcg_at_k'] = ndcg_at_k
                
                # Mean Reciprocal Rank (MRR) for binary labels
                unique_labels = np.unique(y_true_arr)
                if len(unique_labels) == 2:
                    rank_a = np.argsort(-scores_a)
                    rank_b = np.argsort(-scores_b)
                    
                    first_relevant_a = None
                    for rank_pos, idx in enumerate(rank_a, start=1):
                        if y_true_arr[idx] > 0:
                            first_relevant_a = rank_pos
                            break
                    
                    first_relevant_b = None
                    for rank_pos, idx in enumerate(rank_b, start=1):
                        if y_true_arr[idx] > 0:
                            first_relevant_b = rank_pos
                            break
                    
                    mrr_a = 1.0 / first_relevant_a if first_relevant_a else 0.0
                    mrr_b = 1.0 / first_relevant_b if first_relevant_b else 0.0
                    stats['mrr_a'] = float(mrr_a)
                    stats['mrr_b'] = float(mrr_b)
                
                # Append ranking metrics to summary file
                with open(summary_path, 'a') as f:
                    f.write(f"\nRANKING METRICS\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  {'Metric':<25} {self.label_a:<15} {self.label_b:<15}\n")
                    f.write(f"  {'NDCG (full)':<25} {ndcg_a:<15.6f} {ndcg_b:<15.6f}\n")
                    for k, vals in ndcg_at_k.items():
                        f.write(f"  {f'NDCG@{k}':<25} {vals['a']:<15.6f} {vals['b']:<15.6f}\n")
                    if 'mrr_a' in stats:
                        f.write(f"  {'MRR':<25} {stats['mrr_a']:<15.6f} {stats['mrr_b']:<15.6f}\n")
                
                print(f"NDCG: {self.label_a}={ndcg_a:.4f}, {self.label_b}={ndcg_b:.4f}")
                
            except ImportError:
                print("[WARNING] sklearn not available, skipping NDCG computation")
        
        # ------------------------------------------------------------------
        # 6. Regression: error comparison bar chart (if y_true provided)
        # ------------------------------------------------------------------
        if model_type == "regression" and y_true is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # MAE comparison
            mae_vals = [stats['mae_a_vs_true'], stats['mae_b_vs_true']]
            axes[0].bar([self.label_a, self.label_b], mae_vals,
                       color=['steelblue', 'coral'], edgecolor='black', linewidth=0.5)
            axes[0].set_ylabel("Mean Absolute Error")
            axes[0].set_title("MAE vs Ground Truth")
            axes[0].grid(True, linestyle='--', alpha=0.3, axis='y')
            for i, v in enumerate(mae_vals):
                axes[0].text(i, v + max(mae_vals) * 0.02, f"{v:.4f}",
                            ha='center', va='bottom', fontsize=11)
            
            # RMSE comparison
            rmse_vals = [stats['rmse_a_vs_true'], stats['rmse_b_vs_true']]
            axes[1].bar([self.label_a, self.label_b], rmse_vals,
                       color=['steelblue', 'coral'], edgecolor='black', linewidth=0.5)
            axes[1].set_ylabel("Root Mean Squared Error")
            axes[1].set_title("RMSE vs Ground Truth")
            axes[1].grid(True, linestyle='--', alpha=0.3, axis='y')
            for i, v in enumerate(rmse_vals):
                axes[1].text(i, v + max(rmse_vals) * 0.02, f"{v:.4f}",
                            ha='center', va='bottom', fontsize=11)
            
            fig.suptitle(f"Model Accuracy vs Ground Truth (n={n})", fontsize=13)
            plt.tight_layout()
            self.plotter._save_plot('regression_error_comparison.png')
            print("Generated: regression_error_comparison.png")
        
        # ------------------------------------------------------------------
        # Print summary to console
        # ------------------------------------------------------------------
        print(f"\n  Model type: {model_type}")
        print(f"  Pearson r = {r_pearson:.4f}, Spearman r = {r_spearman:.4f}, "
              f"Kendall tau = {r_kendall:.4f}")
        print(f"  Score diff: mean={stats['diff_mean']:.4f}, "
              f"std={stats['diff_std']:.4f}, median={stats['diff_median']:.4f}")
        
        if model_type == "regression":
            print(f"  MAE between predictions: {stats['mae_between']:.4f}, "
                  f"RMSE: {stats['rmse_between']:.4f}, R2: {stats['r2_between']:.4f}")
            if y_true is not None:
                print(f"  MAE vs truth: {self.label_a}={stats['mae_a_vs_true']:.4f}, "
                      f"{self.label_b}={stats['mae_b_vs_true']:.4f}")
        
        if 'agreement_rate' in stats:
            print(f"  Agreement rate: {stats['agreement_rate']:.1%}")
        
        if 'ndcg_a' in stats:
            print(f"  NDCG: {self.label_a}={stats['ndcg_a']:.4f}, "
                  f"{self.label_b}={stats['ndcg_b']:.4f}")
        
        # ------------------------------------------------------------------
        # 7. Q-Q plot of score distributions
        # ------------------------------------------------------------------
        self._plot_score_qq_from_scores(scores_a, scores_b)
        
        return stats
    
    @staticmethod
    def _build_qq_percentiles(n_samples: int) -> np.ndarray:
        """
        Build a hybrid percentile grid with extra tail resolution.
        
        For large samples (>= 1000):
            [0.1, 0.5, 1, 2, 3, ..., 97, 98, 99, 99.5, 99.9]  (103 points)
        For small samples (< 1000):
            [1, 2, 3, ..., 97, 98, 99]  (99 points)
            
        The tail percentiles (0.1, 0.5, 99.5, 99.9) are included only when
        there are enough samples for them to be statistically meaningful.
        
        Args:
            n_samples: Number of data points in the score arrays
            
        Returns:
            Sorted array of percentile values
        """
        # Core: 1% increments from 1 to 99
        core = list(range(1, 100))
        
        # Extra tail resolution for large enough samples
        if n_samples >= 1000:
            tail = [0.1, 0.5, 99.5, 99.9]
        else:
            tail = []
        
        percentiles = sorted(set(core + tail))
        return np.array(percentiles)
    
    def _plot_score_qq_from_scores(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        percentiles: Optional[np.ndarray] = None,
    ) -> None:
        """
        Generate a Q-Q plot comparing score distributions from two models.
        
        Plots the quantiles of scores_a (x-axis) against the quantiles of
        scores_b (y-axis) at each percentile. Points on the y=x diagonal
        indicate identical distributions; deviations reveal where the two
        models diverge.
        
        Args:
            scores_a: Prediction scores from model A
            scores_b: Prediction scores from model B
            percentiles: Optional array of percentile values to evaluate.
                         If None, uses the hybrid grid from _build_qq_percentiles().
        """
        n_samples = min(len(scores_a), len(scores_b))
        
        if percentiles is None:
            percentiles = self._build_qq_percentiles(n_samples)
        
        q_a = np.percentile(scores_a, percentiles)
        q_b = np.percentile(scores_b, percentiles)
        
        # ----- Plot -----
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter colored by percentile
        sc = ax.scatter(
            q_a, q_b,
            c=percentiles,
            cmap='coolwarm',
            vmin=0,
            vmax=100,
            s=30,
            edgecolors='black',
            linewidth=0.3,
            zorder=3,
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_ticks(np.arange(0, 101, 10))
        cbar.set_label("Percentile", fontsize=10)
        
        # y=x reference line
        lo = min(q_a.min(), q_b.min())
        hi = max(q_a.max(), q_b.max())
        margin = (hi - lo) * 0.03
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            'k--', linewidth=1.5, alpha=0.7, label='y = x (identical distributions)',
        )
        
        ax.set_xlabel(f"{self.label_a} score quantiles", fontsize=11)
        ax.set_ylabel(f"{self.label_b} score quantiles", fontsize=11)
        ax.set_title(
            f"Q-Q Plot: {self.label_a} vs {self.label_b}\n"
            f"Score distribution comparison ({len(percentiles)} percentiles, "
            f"n={n_samples})",
            fontsize=12,
        )
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        self.plotter._save_plot('score_qq_plot.png')
        print("Generated: score_qq_plot.png")
    
    def plot_score_qq(
        self,
        analyzer_a: ModelAnalyzer,
        analyzer_b: ModelAnalyzer,
        n_samples: Optional[int] = None,
        percentiles: Optional[np.ndarray] = None,
    ) -> None:
        """
        Generate a Q-Q plot comparing predicted score distributions.
        
        Computes predictions from both models on the same data, then plots
        quantiles of model A scores (x-axis) against quantiles of model B
        scores (y-axis). Points along the y=x diagonal indicate the two
        models produce identically distributed scores; deviations reveal
        systematic differences (e.g. one model scoring higher in the tails).
        
        Uses a hybrid percentile grid with extra tail resolution by default:
        [0.1, 0.5, 1, 2, ..., 98, 99, 99.5, 99.9] for large samples, or
        [1, 2, ..., 98, 99] for small samples (< 1000).
        
        Args:
            analyzer_a: ModelAnalyzer for model A (must have data and model loaded)
            analyzer_b: ModelAnalyzer for model B (must have data and model loaded)
            n_samples: Number of samples to use (None = use all available data)
            percentiles: Optional array of percentile values to evaluate.
                         If None, uses the default hybrid grid.
        """
        analyzer_a._check_data_and_model()
        analyzer_b._check_data_and_model()
        
        model_type = self._detect_model_type(analyzer_a)
        
        features_a = analyzer_a.tree_analyzer.feature_names
        features_b = analyzer_b.tree_analyzer.feature_names
        
        n = min(len(analyzer_a.df), len(analyzer_b.df))
        if n_samples is not None:
            n = min(n_samples, n)
        X_a = analyzer_a.df[features_a].iloc[:n]
        X_b = analyzer_b.df[features_b].iloc[:n]
        
        if model_type == "classification":
            scores_a = analyzer_a._get_corrected_predictions(X_a)
            scores_b = analyzer_b._get_corrected_predictions(X_b)
        else:
            scores_a = self._get_raw_scores(analyzer_a, X_a)
            scores_b = self._get_raw_scores(analyzer_b, X_b)
        
        self._plot_score_qq_from_scores(scores_a, scores_b, percentiles)