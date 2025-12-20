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
import os
import sys

# Add the package to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost_interp import TreeAnalyzer, ModelAnalyzer

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

RANDOM_SEED = 10
N_SAMPLES = 50000
TARGET_POSITIVE_RATE = 0.10  # 10% positive rate

# Feature effect strengths (coefficients for log-odds)
EFFECT_STRONG = 0.8
EFFECT_MEDIUM = 0.5
EFFECT_WEAK = 0.2

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
    'unif_quad_pos': (0, 1, 'quadratic', EFFECT_MEDIUM),
    'unif_quad_neg': (0, 1, 'quadratic', -EFFECT_MEDIUM),
    'unif_none_1': (0, 10, 'none', 0),
    'unif_none_2': (-5, 5, 'none', 0),
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
    # 6. Noise Features (no signal)
    # -------------------------------------------------------------------------
    print("  Generating Noise features...")
    data['noise_norm'] = np.random.normal(0, 1, n_samples)
    data['noise_unif'] = np.random.uniform(0, 100, n_samples)
    data['noise_cat'] = np.random.randint(0, 50, n_samples)
    
    # -------------------------------------------------------------------------
    # 7. Generate Target
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
    features.extend(NOISE_FEATURES)
    return features


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
        n_estimators=100,
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
    print(f"\n✅ Model saved: {model_path}")
    
    return model, X_train, X_test, y_train, y_test


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def run_full_analysis(
    model_path: str,
    data_df: pd.DataFrame,
    feature_names: list,
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
    tree_analyzer.plot_feature_importance_combined(top_n=20)
    tree_analyzer.plot_feature_importance_distributions(log_scale=False, top_n=20)
    tree_analyzer.plot_feature_importance_scatter(top_n=20)
    print("  ✅ Feature importance plots saved")
    
    # Tree structure plots
    print("Generating tree structure plots...")
    tree_analyzer.plot_tree_depth_histogram()
    tree_analyzer.plot_cumulative_gain()
    print("  ✅ Tree structure plots saved")
    
    # Interactive tree visualization
    print("Generating interactive tree visualizations...")
    try:
        from xgboost_interp.plotting import InteractivePlotter
        interactive_plotter = InteractivePlotter(output_dir)
        interactive_plotter.plot_interactive_trees(
            tree_analyzer.trees, tree_analyzer.feature_names,
            top_k=5, combined=False
        )
        print("  ✅ Interactive tree plots generated")
    except Exception as e:
        print(f"  ⚠️ Could not generate interactive plots: {e}")
    
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
    
    # Select subset of features for detailed analysis
    # (analyzing all 37 features would take too long)
    important_features = [
        # Strong effects (should be most important)
        'norm_iid_pos_strong', 'norm_iid_neg_strong',
        'bin_pos_strong', 'bin_neg_strong',
        # Medium effects
        'norm_corr_1_pos', 'norm_corr_3_neg',
        'unif_linear_pos', 'unif_quad_pos',
        # Categorical
        'cat_15_strong', 'cat_75_mixed',
        # Noise (should have minimal importance)
        'noise_norm', 'noise_unif',
    ]
    
    # Partial Dependence Plots
    print("\nGenerating Partial Dependence Plots...")
    for i, feature in enumerate(important_features):
        try:
            model_analyzer.plot_partial_dependence(
                feature_name=feature,
                n_curves=200
            )
            print(f"  [{i+1}/{len(important_features)}] ✅ {feature}")
        except Exception as e:
            print(f"  [{i+1}/{len(important_features)}] ⚠️ {feature}: {e}")
    
    # Marginal Impact Analysis
    print("\nGenerating Marginal Impact Analysis...")
    for i, feature in enumerate(important_features):
        try:
            model_analyzer.plot_marginal_impact_univariate(feature, scale="linear")
            print(f"  [{i+1}/{len(important_features)}] ✅ {feature}")
        except Exception as e:
            print(f"  [{i+1}/{len(important_features)}] ⚠️ {feature}: {e}")
    
    # Prediction Evolution
    print("\nGenerating prediction evolution plot...")
    try:
        model_analyzer.plot_scores_across_trees(n_records=500)
        print("  ✅ Scores across trees plot saved")
    except Exception as e:
        print(f"  ⚠️ Could not generate scores across trees: {e}")
    
    # Early exit performance analysis
    print("\nGenerating early exit performance analysis...")
    try:
        model_analyzer.analyze_early_exit_performance(n_records=1000)
    except Exception as e:
        print(f"  ⚠️ Could not generate early exit analysis: {e}")
    
    # ALE Plots
    print("\nGenerating ALE Plots...")
    try:
        from PyALE import ale
        for i, feature in enumerate(important_features):
            print(f"  [{i+1}/{len(important_features)}] Computing ALE for '{feature}'...")
            model_analyzer.plot_ale(
                feature_name=feature,
                grid_size=50,
                include_CI=True,
                n_curves=min(5000, len(model_analyzer.df))
            )
        print("  ✅ ALE plots saved")
    except ImportError:
        print("  ⚠️ PyALE not installed - skipping ALE plots")
    except Exception as e:
        print(f"  ⚠️ Failed to generate ALE plots: {e}")
    
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
        
        print("  ✅ SHAP analysis saved")
    except ImportError:
        print("  ⚠️ shap not installed - skipping SHAP analysis")
    except Exception as e:
        print(f"  ⚠️ Failed to generate SHAP analysis: {e}")
    
    print(f"\n✅ Analysis complete! All plots saved to: {output_dir}")


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
    print(f"  - Noise: {len(NOISE_FEATURES)} features")
    print(f"  - Total: {len(feature_names)} features")
    
    # 2. Train XGBoost model
    model_path = "examples/synthetic_imbalanced_classification/synthetic_imbalanced_classification_xgb.json"
    model, X_train, X_test, y_train, y_test = train_xgboost_model(
        df, feature_names, model_path
    )
    
    # 3. Run full analysis
    run_full_analysis(model_path, df, feature_names)
    
    print("\n" + "="*65)
    print("🎉 Synthetic Imbalanced Classification Example Complete!")
    print("="*65)
    print("\nKey insights:")
    print("  1. Features with strong effects should rank highest in importance")
    print("  2. Noise features should have minimal/zero importance")
    print("  3. Correlated features may share importance due to substitutability")
    print("  4. Categorical features show step patterns in PDP/ALE plots")
    print("  5. Quadratic uniform features show curved relationships")


if __name__ == "__main__":
    main()

