"""
Synthetic Imbalanced Classification Example.

This example demonstrates XGBoost interpretability on synthetic data with:
- Known feature-target relationships (for validation)
- Imbalanced labels (~10% positive rate)
- Mixed feature types: Normal (iid & correlated), Categorical, Binary, Uniform, Noise

The synthetic data has interpretable feature names that encode their properties.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, 
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
import os
import sys

# Add the package to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from xgboost_interp import TreeAnalyzer, ModelAnalyzer
from xgboost_interp.plotting import FeaturePlotter, TreePlotter

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

RANDOM_SEED = 10
N_SAMPLES = 100000
TARGET_POSITIVE_RATE = 0.10  # 10% positive rate

# Feature effect strengths (coefficients for log-odds)
EFFECT_STRONG = 0.8
EFFECT_MEDIUM = 0.5
EFFECT_WEAK = 0.2
EFFECT_VERY_STRONG = 1.5  # For quadratic features to make relationship obvious

# Normal iid feature parameters: (mean, std, effect_coefficient)
NORMAL_IID_PARAMS = {
    'norm_iid_pos_strong': (0, 1, EFFECT_STRONG),
    'norm_iid_pos_medium': (0, 1, EFFECT_MEDIUM),
    'norm_iid_pos_weak': (0, 1, EFFECT_WEAK),
    'norm_iid_neg_strong': (0, 1, -EFFECT_STRONG),
    'norm_iid_neg_medium': (0, 1, -EFFECT_MEDIUM),
    'norm_iid_neg_weak': (0, 1, -EFFECT_WEAK),
    'norm_iid_zero_1': (0, 1, 0),  # No effect
    'norm_iid_zero_2': (2, 0.5, 0),  # Different dist, no effect
    'norm_iid_zero_3': (-1, 2, 0),  # Different dist, no effect
}

# Correlated normal features: covariance structure
NORMAL_CORR_EFFECTS = {
    'norm_corr_1_pos': EFFECT_MEDIUM,
    'norm_corr_2_pos': EFFECT_WEAK,
    'norm_corr_3_neg': -EFFECT_MEDIUM,
    'norm_corr_4_neg': -EFFECT_WEAK,
    'norm_corr_5_zero': 0,
    'norm_corr_6_zero': 0,
    'norm_corr_7_zero': 0,
}

# Categorical features: (cardinality, effect_type)
# effect_type: 'strong', 'medium', 'weak', 'mixed', 'none'
CATEGORICAL_PARAMS = {
    'cat_15_strong': (15, 'strong'),
    'cat_30_medium': (30, 'medium'),
    'cat_50_weak': (50, 'weak'),
    'cat_75_mixed': (75, 'mixed'),
    'cat_100_strong': (100, 'strong'),
    'cat_150_medium': (150, 'medium'),
    'cat_180_weak': (180, 'weak'),
    'cat_200_none': (200, 'none'),
}

# Binary features: effect coefficient
BINARY_PARAMS = {
    'bin_pos_strong': EFFECT_STRONG,
    'bin_pos_weak': EFFECT_WEAK,
    'bin_neg_strong': -EFFECT_STRONG,
    'bin_neg_weak': -EFFECT_WEAK,
}

# Uniform features: (min, max, relationship_type, effect)
# relationship_type: 'linear', 'quadratic', 'none'
UNIFORM_PARAMS = {
    'unif_linear_pos': (0, 1, 'linear', EFFECT_MEDIUM),
    'unif_linear_neg': (0, 1, 'linear', -EFFECT_MEDIUM),
    'unif_quad_pos': (0, 1, 'quadratic', EFFECT_VERY_STRONG),  # Strong U-shaped
    'unif_quad_neg': (0, 1, 'quadratic', -EFFECT_VERY_STRONG),  # Strong inverted-U
    'unif_none_1': (0, 10, 'none', 0),
    'unif_none_2': (-5, 5, 'none', 0),
}

# Trigonometric features: (frequency, effect)
# x drawn from Uniform(0, 2π), then sin(freq*x) or cos(freq*x) applied
TRIG_PARAMS = {
    'trig_sin_pos': (3, EFFECT_STRONG),   # sin(3x): 3 cycles in [0, 2π]
    'trig_cos_pos': (3, EFFECT_STRONG),   # cos(3x): 3 cycles in [0, 2π]
}

# Noise features (no signal at all)
NOISE_FEATURES = ['noise_norm', 'noise_unif', 'noise_cat']


# =============================================================================
# DATA GENERATION FUNCTIONS
# =============================================================================

def generate_synthetic_data(
    n_samples: int = N_SAMPLES,
    target_positive_rate: float = TARGET_POSITIVE_RATE,
    random_seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generate synthetic imbalanced classification data with known feature properties.
    
    This function can be called independently to regenerate the same data.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    target_positive_rate : float
        Desired proportion of positive labels (approximate)
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with features and 'target' column
    """
    np.random.seed(random_seed)
    
    print(f"Generating {n_samples:,} samples with ~{target_positive_rate:.0%} positive rate...")
    
    data = {}
    log_odds = np.zeros(n_samples)
    
    # -------------------------------------------------------------------------
    # 1. Normal IID Features
    # -------------------------------------------------------------------------
    print("  Generating Normal IID features...")
    for feat_name, (mean, std, effect) in NORMAL_IID_PARAMS.items():
        values = np.random.normal(mean, std, n_samples)
        data[feat_name] = values
        log_odds += effect * values
    
    # -------------------------------------------------------------------------
    # 2. Correlated Normal Features
    # -------------------------------------------------------------------------
    print("  Generating Correlated Normal features...")
    n_corr = len(NORMAL_CORR_EFFECTS)
    
    # Create correlation matrix with some structure
    corr_matrix = np.eye(n_corr)
    # Add correlations between adjacent features
    for i in range(n_corr - 1):
        corr_matrix[i, i+1] = 0.6
        corr_matrix[i+1, i] = 0.6
    # Add some cross-correlations
    if n_corr > 2:
        corr_matrix[0, 2] = 0.3
        corr_matrix[2, 0] = 0.3
    
    # Generate correlated normal samples
    mean_vec = np.zeros(n_corr)
    corr_samples = np.random.multivariate_normal(mean_vec, corr_matrix, n_samples)
    
    for i, (feat_name, effect) in enumerate(NORMAL_CORR_EFFECTS.items()):
        data[feat_name] = corr_samples[:, i]
        log_odds += effect * corr_samples[:, i]
    
    # -------------------------------------------------------------------------
    # 3. Categorical Features
    # -------------------------------------------------------------------------
    print("  Generating Categorical features...")
    for feat_name, (cardinality, effect_type) in CATEGORICAL_PARAMS.items():
        # Generate random category assignments
        categories = np.random.randint(0, cardinality, n_samples)
        data[feat_name] = categories
        
        # Create category-specific effects
        if effect_type == 'none':
            category_effects = np.zeros(cardinality)
        elif effect_type == 'strong':
            category_effects = np.linspace(-EFFECT_STRONG, EFFECT_STRONG, cardinality)
        elif effect_type == 'medium':
            category_effects = np.linspace(-EFFECT_MEDIUM, EFFECT_MEDIUM, cardinality)
        elif effect_type == 'weak':
            category_effects = np.linspace(-EFFECT_WEAK, EFFECT_WEAK, cardinality)
        elif effect_type == 'mixed':
            # Random effects per category
            np.random.seed(random_seed + hash(feat_name) % 1000)
            category_effects = np.random.normal(0, EFFECT_MEDIUM, cardinality)
        
        # Apply category effects
        log_odds += category_effects[categories]
    
    # -------------------------------------------------------------------------
    # 4. Binary Features
    # -------------------------------------------------------------------------
    print("  Generating Binary features...")
    for feat_name, effect in BINARY_PARAMS.items():
        values = np.random.binomial(1, 0.5, n_samples)
        data[feat_name] = values
        log_odds += effect * values
    
    # -------------------------------------------------------------------------
    # 5. Uniform Features
    # -------------------------------------------------------------------------
    print("  Generating Uniform features...")
    for feat_name, (min_val, max_val, rel_type, effect) in UNIFORM_PARAMS.items():
        values = np.random.uniform(min_val, max_val, n_samples)
        data[feat_name] = values
        
        if rel_type == 'linear':
            # Normalize to [-1, 1] range for effect
            normalized = 2 * (values - min_val) / (max_val - min_val) - 1
            log_odds += effect * normalized
        elif rel_type == 'quadratic':
            # Quadratic relationship (centered)
            normalized = 2 * (values - min_val) / (max_val - min_val) - 1
            log_odds += effect * (normalized ** 2 - 0.33)  # Centered
        # 'none' type: no effect
    
    # -------------------------------------------------------------------------
    # 6. Trigonometric Features (periodic relationships)
    # -------------------------------------------------------------------------
    print("  Generating Trigonometric features...")
    for feat_name, (frequency, effect) in TRIG_PARAMS.items():
        # x drawn from Uniform(0, 2π)
        x_values = np.random.uniform(0, 2 * np.pi, n_samples)
        data[feat_name + '_x'] = x_values  # Store the raw x value
        
        # Apply sin or cos based on feature name
        if 'sin' in feat_name:
            trig_values = np.sin(frequency * x_values)
        else:  # cos
            trig_values = np.cos(frequency * x_values)
        
        # Add to log-odds (trig_values already in [-1, 1])
        log_odds += effect * trig_values
    
    # -------------------------------------------------------------------------
    # 7. Noise Features (no signal)
    # -------------------------------------------------------------------------
    print("  Generating Noise features...")
    data['noise_norm'] = np.random.normal(0, 1, n_samples)
    data['noise_unif'] = np.random.uniform(0, 100, n_samples)
    data['noise_cat'] = np.random.randint(0, 50, n_samples)
    
    # -------------------------------------------------------------------------
    # 8. Generate Target
    # -------------------------------------------------------------------------
    print("  Generating target labels...")
    
    # Adjust intercept to achieve desired positive rate
    # We need to find intercept such that mean(sigmoid(log_odds + intercept)) ≈ target_rate
    # Use binary search
    def get_positive_rate(intercept):
        probs = 1 / (1 + np.exp(-(log_odds + intercept)))
        return probs.mean()
    
    low, high = -10, 10
    for _ in range(50):  # Binary search iterations
        mid = (low + high) / 2
        rate = get_positive_rate(mid)
        if rate < target_positive_rate:
            low = mid
        else:
            high = mid
    
    intercept = mid
    probabilities = 1 / (1 + np.exp(-(log_odds + intercept)))
    target = np.random.binomial(1, probabilities)
    
    data['target'] = target
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Print summary
    actual_rate = target.mean()
    print(f"\n  Generated data summary:")
    print(f"    Total features: {len(df.columns) - 1}")
    print(f"    Positive rate: {actual_rate:.2%} ({target.sum():,} positives)")
    print(f"    Negative rate: {1-actual_rate:.2%} ({(1-target).sum():,} negatives)")
    
    return df


def get_feature_names() -> list:
    """Get list of all feature names (excluding target)."""
    features = []
    features.extend(NORMAL_IID_PARAMS.keys())
    features.extend(NORMAL_CORR_EFFECTS.keys())
    features.extend(CATEGORICAL_PARAMS.keys())
    features.extend(BINARY_PARAMS.keys())
    features.extend(UNIFORM_PARAMS.keys())
    # Trig features: the raw x values (used as model input)
    features.extend([f"{name}_x" for name in TRIG_PARAMS.keys()])
    features.extend(NOISE_FEATURES)
    return features


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_feature_pdf(
    df: pd.DataFrame,
    feature_name: str,
    target_col: str = 'target',
    save_dir: str = "examples/synthetic_imbalanced_classification/output/feature_pdfs",
    n_bins: int = 100
) -> None:
    """
    Plot feature distribution with positive rate coloring and KDE overlay.
    
    Creates a visualization showing:
    1. Histogram (density-normalized) with bars colored by empirical positive rate
    2. KDE overlay as proper PDF (area = 1)
    3. Scatter of individual points at y=0 colored by label
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature and target columns
    feature_name : str
        Name of the feature column to plot
    target_col : str
        Name of the target column (default: 'target')
    save_dir : str
        Directory to save the plot
    n_bins : int
        Number of histogram bins (default: 100)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    feature_values = df[feature_name].values
    target_values = df[target_col].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # -------------------------------------------------------------------------
    # 1. Compute histogram with positive rate per bin (density-normalized)
    # -------------------------------------------------------------------------
    bin_edges = np.linspace(feature_values.min(), feature_values.max(), n_bins + 1)
    bin_indices = np.digitize(feature_values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Handle edge cases
    
    # Compute counts and positive rates per bin
    bin_counts = np.zeros(n_bins)
    bin_positive_rates = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = bin_indices == i
        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            bin_positive_rates[i] = target_values[mask].mean()
    
    # Convert counts to density (so histogram area = 1)
    bin_width = bin_edges[1] - bin_edges[0]
    n_total = len(feature_values)
    bin_density = bin_counts / (n_total * bin_width)
    
    # Create colormap: light gray (0) to bright red (1)
    colors_light_gray = np.array([0.85, 0.85, 0.85, 1.0])  # Light gray
    colors_bright_red = np.array([1.0, 0.0, 0.0, 1.0])     # Bright red
    
    # Interpolate colors based on positive rate
    bar_colors = []
    for rate in bin_positive_rates:
        color = (1 - rate) * colors_light_gray + rate * colors_bright_red
        bar_colors.append(color)
    
    # Plot histogram bars (density-normalized)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ax.bar(bin_centers, bin_density, width=bin_width, color=bar_colors, 
           edgecolor='white', linewidth=0.3, zorder=2)
    
    # -------------------------------------------------------------------------
    # 2. Overlay KDE as proper PDF (area = 1, no scaling)
    # -------------------------------------------------------------------------
    kde = stats.gaussian_kde(feature_values)
    x_kde = np.linspace(feature_values.min(), feature_values.max(), 500)
    kde_values = kde(x_kde)
    
    ax.plot(x_kde, kde_values, color='black', linewidth=2, label='KDE', zorder=3)
    
    # -------------------------------------------------------------------------
    # 3. Scatter individual points at y=0
    # -------------------------------------------------------------------------
    # Add small jitter to y for visibility (scaled to density range)
    max_density = max(bin_density.max(), kde_values.max()) if bin_density.max() > 0 else 1
    y_jitter = np.random.uniform(-0.02, 0.02, len(feature_values)) * max_density
    
    # Colors: light gray for label=0, bright red for label=1
    scatter_colors = np.where(
        target_values == 0,
        'lightgray',
        'red'
    )
    
    ax.scatter(
        feature_values, y_jitter, 
        c=scatter_colors, 
        alpha=0.15, 
        s=3, 
        zorder=1,
        rasterized=True  # Optimize for large number of points
    )
    
    # -------------------------------------------------------------------------
    # 4. Formatting
    # -------------------------------------------------------------------------
    ax.set_xlabel(feature_name, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Distribution of {feature_name}\n(bar color = positive rate: gray=0%, red=100%)', 
                 fontsize=14)
    
    # Add colorbar for positive rate
    sm = plt.cm.ScalarMappable(
        cmap=mcolors.LinearSegmentedColormap.from_list('pos_rate', ['lightgray', 'red']),
        norm=plt.Normalize(0, 1)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Positive Rate', fontsize=10)
    
    # Add legend for scatter
    ax.scatter([], [], c='lightgray', alpha=0.5, s=20, label='Label = 0')
    ax.scatter([], [], c='red', alpha=0.5, s=20, label='Label = 1')
    ax.legend(loc='upper right', fontsize=9)
    
    # Set y-axis to start at a small negative value to show scatter points
    ax.set_ylim(bottom=-0.05 * max_density)
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(save_dir, f'{feature_name}_pdf.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_feature_pdfs(
    df: pd.DataFrame,
    feature_names: list,
    target_col: str = 'target',
    save_dir: str = "examples/synthetic_imbalanced_classification/output/feature_pdfs",
    n_bins: int = 100
) -> None:
    """
    Generate PDF plots for all features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and target
    feature_names : list
        List of feature column names
    target_col : str
        Name of the target column
    save_dir : str
        Directory to save plots
    n_bins : int
        Number of histogram bins
    """
    print(f"\nGenerating feature PDF plots for {len(feature_names)} features...")
    os.makedirs(save_dir, exist_ok=True)
    
    for i, feature in enumerate(feature_names):
        plot_feature_pdf(df, feature, target_col, save_dir, n_bins)
        if (i + 1) % 10 == 0 or (i + 1) == len(feature_names):
            print(f"  [{i+1}/{len(feature_names)}] Feature PDF plots generated")
    
    print(f"  All feature PDF plots saved to: {save_dir}")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_xgboost_model(
    df: pd.DataFrame,
    feature_names: list,
    model_path: str = "examples/synthetic_imbalanced_classification/synthetic_imbalanced_classification_xgb.json"
):
    """Train XGBoost binary classification model."""
    print("\n" + "="*60)
    print("TRAINING XGBOOST MODEL")
    print("="*60)
    
    X = df[feature_names]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} samples ({y_train.mean():.2%} positive)")
    print(f"Test set: {len(X_test):,} samples ({y_test.mean():.2%} positive)")
    
    # Train model (no scale_pos_weight since we're not downsampling)
    model = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        objective='binary:logistic',
        eval_metric='auc'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    print(f"\nModel Performance:")
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"  Average Precision: {average_precision_score(y_test, y_pred_proba):.4f}")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"\nModel saved: {model_path}")
    
    return model, X_train, X_test, y_train, y_test


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def run_full_analysis(
    model_path: str,
    data_df: pd.DataFrame,
    feature_names: list,
    y_test=None,
    y_pred_proba=None,
    X_test=None,
    output_dir: str = "examples/synthetic_imbalanced_classification/output"
):
    """Run complete interpretability analysis on the trained model."""
    print("\n" + "="*60)
    print("RUNNING FULL INTERPRETABILITY ANALYSIS")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # Tree-level Analysis (no data needed)
    # -------------------------------------------------------------------------
    print("\n--- Tree-Level Analysis ---")
    
    tree_analyzer = TreeAnalyzer(model_path, save_dir=output_dir)
    tree_analyzer.print_model_summary()
    
    # Feature importance plots
    print("\nGenerating feature importance plots...")
    tree_analyzer.plot_feature_importance_combined()
    tree_analyzer.plot_feature_importance_distributions(log_scale=False)
    tree_analyzer.plot_feature_importance_scatter()
    print("  Feature importance plots saved")
    
    # Tree structure plots
    print("Generating tree structure plots...")
    tree_analyzer.plot_tree_depth_histogram()
    tree_analyzer.plot_cumulative_gain()
    tree_analyzer.plot_cumulative_prediction_shift()
    print("  Tree structure plots saved")
    
    # Feature co-occurrence analysis
    print("Generating feature co-occurrence heatmaps...")
    tree_analyzer.plot_tree_level_feature_cooccurrence()
    tree_analyzer.plot_path_level_feature_cooccurrence()
    tree_analyzer.plot_sequential_feature_dependency()
    print("  Feature co-occurrence heatmaps saved")
    
    # Advanced feature and tree statistics
    print("Generating advanced feature and tree statistics...")
    feature_plotter = FeaturePlotter(output_dir)
    tree_plotter = TreePlotter(output_dir)
    
    feature_plotter.plot_feature_usage_heatmap(
        tree_analyzer.trees, tree_analyzer.feature_names, log_scale=True
    )
    feature_plotter.plot_split_depth_per_feature(
        tree_analyzer.trees, tree_analyzer.feature_names
    )
    feature_plotter.plot_feature_split_impact(
        tree_analyzer.trees, tree_analyzer.feature_names, log_scale=False
    )
    tree_plotter.plot_prediction_and_gain_stats(
        tree_analyzer.trees, log_scale=False
    )
    tree_plotter.plot_gain_heatmap(
        tree_analyzer.trees, tree_analyzer.feature_names
    )
    print("  Advanced statistics plots saved")
    
    # Interactive tree visualization
    print("Generating interactive tree visualizations...")
    try:
        from xgboost_interp.plotting import InteractivePlotter
        interactive_plotter = InteractivePlotter(output_dir)
        interactive_plotter.plot_interactive_trees(
            tree_analyzer.trees, tree_analyzer.feature_names,
            top_k=5, combined=False
        )
        print("  Interactive tree plots generated")
    except Exception as e:
        print(f"  Could not generate interactive plots: {e}")
    
    # -------------------------------------------------------------------------
    # Data-dependent Analysis
    # -------------------------------------------------------------------------
    print("\n--- Data-Dependent Analysis ---")
    
    # Save data for analysis
    data_dir = "examples/synthetic_imbalanced_classification/synthetic_imbalanced_classification_data"
    os.makedirs(data_dir, exist_ok=True)
    data_df.to_parquet(f"{data_dir}/synthetic_data.parquet", index=False)
    
    # Initialize model analyzer
    model_analyzer = ModelAnalyzer(tree_analyzer, target_class=1)
    model_analyzer.load_data_from_parquets(data_dir, num_files_to_read=1)
    model_analyzer.load_xgb_model(model_path)
    
    # Model performance metrics
    if y_test is not None and y_pred_proba is not None:
        print("\nComputing model performance metrics...")
        metrics = model_analyzer.evaluate_model_performance(y_test, y_pred_proba)
        print("Model Performance Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {round(v, 6)}")
        print(f"  Saved to: {output_dir}/model_performance_metrics.txt")
        
        print("\nGenerating calibration curves...")
        model_analyzer.generate_calibration_curves(y_test, y_pred_proba, X=X_test, n_bins=10)
    
    # Subset of features for SHAP dependence plots (full analysis is too slow per-feature)
    important_features = [
        'norm_iid_pos_strong', 'norm_iid_neg_strong',
        'bin_pos_strong', 'bin_neg_strong',
        'norm_corr_1_pos', 'norm_corr_3_neg',
        'unif_linear_pos', 'unif_quad_pos',
        'trig_sin_pos_x', 'trig_cos_pos_x',
        'cat_15_strong', 'cat_75_mixed',
        'noise_norm', 'noise_unif',
    ]
    
    # Partial Dependence Plots (all features)
    print(f"\nGenerating Partial Dependence Plots for all {len(feature_names)} features...")
    for i, feature in enumerate(feature_names):
        try:
            model_analyzer.plot_partial_dependence(
                feature_name=feature,
                n_curves=200
            )
            print(f"  [{i+1}/{len(feature_names)}] {feature}")
        except Exception as e:
            print(f"  [{i+1}/{len(feature_names)}] {feature}: {e}")
    
    # Marginal Impact Analysis (all features)
    print(f"\nGenerating Marginal Impact Analysis for all {len(feature_names)} features...")
    for i, feature in enumerate(feature_names):
        try:
            model_analyzer.plot_marginal_impact_univariate(feature, scale="linear")
            print(f"  [{i+1}/{len(feature_names)}] {feature}")
        except Exception as e:
            print(f"  [{i+1}/{len(feature_names)}] {feature}: {e}")
    
    # Prediction Evolution
    print("\nGenerating prediction evolution plot...")
    try:
        model_analyzer.plot_scores_across_trees(n_records=1000)
        print("  Scores across trees plot saved")
    except Exception as e:
        print(f"  Could not generate scores across trees: {e}")
    
    # Early exit performance analysis
    print("\nGenerating early exit performance analysis...")
    try:
        model_analyzer.analyze_early_exit_performance(n_records=5000, n_detailed_curves=1000)
    except Exception as e:
        print(f"  Could not generate early exit analysis: {e}")
    
    # ALE Plots (all features)
    print(f"\nGenerating ALE Plots for all {len(feature_names)} features...")
    try:
        from PyALE import ale
        for i, feature in enumerate(feature_names):
            print(f"  [{i+1}/{len(feature_names)}] Computing ALE for '{feature}'...")
            model_analyzer.plot_ale(
                feature_name=feature,
                grid_size=50,
                include_CI=True,
                n_curves=min(5000, len(model_analyzer.df))
            )
        print("  ALE plots saved")
    except ImportError:
        print("  PyALE not installed - skipping ALE plots")
    except Exception as e:
        print(f"  Failed to generate ALE plots: {e}")
    
    # SHAP Analysis
    print("\nGenerating SHAP Analysis...")
    try:
        import shap
        import matplotlib.pyplot as plt
        
        X_sample = model_analyzer.df[feature_names].sample(
            n=min(2000, len(model_analyzer.df)), 
            random_state=RANDOM_SEED
        )
        
        explainer = shap.TreeExplainer(model_analyzer.xgb_model)
        shap_values = explainer(X_sample)
        
        shap_dir = os.path.join(output_dir, 'SHAP_analysis')
        shap_dep_dir = os.path.join(shap_dir, 'SHAP_dependence_plots')
        os.makedirs(shap_dir, exist_ok=True)
        os.makedirs(shap_dep_dir, exist_ok=True)
        
        # Summary plots
        plt.figure()
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.savefig(os.path.join(shap_dir, 'summary_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig(os.path.join(shap_dir, 'summary_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Dependence plots for important features
        for feature in important_features:
            if feature in X_sample.columns:
                plt.figure()
                shap.dependence_plot(feature, shap_values.values, X_sample, show=False)
                plt.savefig(os.path.join(shap_dep_dir, f'dependence_{feature}.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        # Waterfall plots
        for idx in range(min(5, len(X_sample))):
            plt.figure()
            shap.waterfall_plot(shap_values[idx], show=False)
            plt.savefig(os.path.join(shap_dir, f'waterfall_sample_{idx}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print("  SHAP analysis saved")
    except ImportError:
        print("  shap not installed - skipping SHAP analysis")
    except Exception as e:
        print(f"  Failed to generate SHAP analysis: {e}")
    
    print(f"\nAnalysis complete! All plots saved to: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function for synthetic imbalanced classification example."""
    print("="*65)
    print("XGBoost Interpretability - Synthetic Imbalanced Classification")
    print("="*65)
    
    # 1. Generate synthetic data
    df = generate_synthetic_data()
    feature_names = get_feature_names()
    
    print(f"\nFeature groups:")
    print(f"  - Normal IID: {len(NORMAL_IID_PARAMS)} features")
    print(f"  - Normal Correlated: {len(NORMAL_CORR_EFFECTS)} features")
    print(f"  - Categorical: {len(CATEGORICAL_PARAMS)} features")
    print(f"  - Binary: {len(BINARY_PARAMS)} features")
    print(f"  - Uniform: {len(UNIFORM_PARAMS)} features")
    print(f"  - Trigonometric: {len(TRIG_PARAMS)} features")
    print(f"  - Noise: {len(NOISE_FEATURES)} features")
    print(f"  - Total: {len(feature_names)} features")
    
    # 2. Train XGBoost model
    model_path = "examples/synthetic_imbalanced_classification/synthetic_imbalanced_classification_xgb.json"
    model, X_train, X_test, y_train, y_test = train_xgboost_model(
        df, feature_names, model_path
    )
    
    # Get predictions for metrics
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 3. Run full analysis
    run_full_analysis(model_path, df, feature_names, y_test, y_pred_proba, X_test)
    
    # 4. Generate feature PDF plots for all features
    plot_all_feature_pdfs(df, feature_names)
    
    print("\n" + "="*65)
    print("Synthetic Imbalanced Classification Example Complete!")
    print("="*65)
    print("\nKey insights:")
    print("  1. Features with strong effects should rank highest in importance")
    print("  2. Noise features should have minimal/zero importance")
    print("  3. Correlated features may share importance due to substitutability")
    print("  4. Categorical features show step patterns in PDP/ALE plots")
    print("  5. Quadratic uniform features show U-shaped or inverted-U relationships")
    print("  6. Trigonometric features show periodic wave patterns in PDP/ALE plots")


if __name__ == "__main__":
    main()

