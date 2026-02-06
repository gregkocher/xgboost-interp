"""
Model Diffing Example.

This example demonstrates the ModelDiff class for comparing two XGBoost models
trained on overlapping but distinct feature subsets.

The two models are trained on synthetic data with intentionally different feature sets:
- Model A: 31 features (drops 1 strong + 2 weak + 3 noise/zero-signal features)
- Model B: 33 features (drops 2 weak + 2 noise/zero-signal features)

This allows us to observe:
1. How dropping a high-signal feature affects cumulative gain
2. How feature importance redistributes when key features are missing
3. How PDP curves compare for shared features
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import sys

# Add the package to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from xgboost_interp import TreeAnalyzer, ModelAnalyzer
from xgboost_interp.core import ModelDiff

# Import data generation from synthetic example
from xgboost_interp.examples.synthetic_imbalanced_classification_example import (
    generate_synthetic_data,
    get_feature_names,
    RANDOM_SEED,
)

# =============================================================================
# FEATURE SUBSETS FOR EACH MODEL
# =============================================================================

# All 37 features from the synthetic data
ALL_FEATURES = get_feature_names()

# Model A drops 6 features (1 strong, 2 weak, 3 noise/zero)
MODEL_A_DROPS = [
    'bin_pos_strong',      # HIGH-SIGNAL (effect=0.8)
    'norm_iid_pos_weak',   # WEAK-SIGNAL (effect=0.2)
    'cat_50_weak',         # WEAK-SIGNAL categorical
    'cat_200_none',        # no signal
    'noise_cat',           # noise
    'unif_none_2',         # no signal
]

# Model B drops 4 features (2 weak, 2 noise/zero)
MODEL_B_DROPS = [
    'bin_neg_weak',        # WEAK-SIGNAL (effect=-0.2)
    'norm_corr_4_neg',     # WEAK-SIGNAL (effect=-0.2)
    'noise_unif',          # noise
    'norm_iid_zero_2',     # no signal
]

# Compute feature lists for each model
FEATURES_MODEL_A = [f for f in ALL_FEATURES if f not in MODEL_A_DROPS]
FEATURES_MODEL_B = [f for f in ALL_FEATURES if f not in MODEL_B_DROPS]

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = "examples/model_diff/output"
MODEL_A_PATH = "examples/model_diff/model_a.json"
MODEL_B_PATH = "examples/model_diff/model_b.json"
DATA_DIR = "examples/model_diff/data"

# Use fewer trees for faster training in this demo
N_ESTIMATORS = 500


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_model(
    df: pd.DataFrame,
    feature_names: list,
    model_path: str,
    model_label: str,
) -> xgb.XGBClassifier:
    """Train an XGBoost model on the specified features."""
    print(f"\n{'='*60}")
    print(f"TRAINING {model_label}")
    print(f"{'='*60}")
    print(f"Features: {len(feature_names)}")
    
    X = df[feature_names]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
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
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC: {auc:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"Model saved: {model_path}")
    
    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the model diffing example."""
    print("="*65)
    print("XGBoost Interpretability - Model Diffing Example")
    print("="*65)
    
    # -------------------------------------------------------------------------
    # 1. Generate synthetic data
    # -------------------------------------------------------------------------
    print("\n[1/5] Generating synthetic data...")
    df = generate_synthetic_data()
    
    # Save data for ModelAnalyzer
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(f"{DATA_DIR}/synthetic_data.parquet", index=False)
    
    # -------------------------------------------------------------------------
    # 2. Print feature split summary
    # -------------------------------------------------------------------------
    print("\n[2/5] Feature split summary:")
    print(f"  Total features available: {len(ALL_FEATURES)}")
    print(f"\n  Model A ({len(FEATURES_MODEL_A)} features):")
    print(f"    Drops: {MODEL_A_DROPS}")
    print(f"\n  Model B ({len(FEATURES_MODEL_B)} features):")
    print(f"    Drops: {MODEL_B_DROPS}")
    
    # Features unique to each model
    only_in_a = set(FEATURES_MODEL_A) - set(FEATURES_MODEL_B)
    only_in_b = set(FEATURES_MODEL_B) - set(FEATURES_MODEL_A)
    in_both = set(FEATURES_MODEL_A) & set(FEATURES_MODEL_B)
    
    print(f"\n  Features only in Model A: {sorted(only_in_a)}")
    print(f"  Features only in Model B: {sorted(only_in_b)}")
    print(f"  Features in both: {len(in_both)}")
    
    # -------------------------------------------------------------------------
    # 3. Train both models
    # -------------------------------------------------------------------------
    print("\n[3/5] Training models...")
    model_a = train_model(df, FEATURES_MODEL_A, MODEL_A_PATH, "MODEL A (baseline)")
    model_b = train_model(df, FEATURES_MODEL_B, MODEL_B_PATH, "MODEL B (candidate)")
    
    # -------------------------------------------------------------------------
    # 4. Initialize ModelDiff
    # -------------------------------------------------------------------------
    print("\n[4/5] Initializing ModelDiff...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tree_analyzer_a = TreeAnalyzer(MODEL_A_PATH, save_dir=OUTPUT_DIR)
    tree_analyzer_b = TreeAnalyzer(MODEL_B_PATH, save_dir=OUTPUT_DIR)
    
    model_diff = ModelDiff(
        analyzer_a=tree_analyzer_a,
        analyzer_b=tree_analyzer_b,
        label_a="Model A (31 features)",
        label_b="Model B (33 features)",
        save_dir=OUTPUT_DIR,
    )
    
    # -------------------------------------------------------------------------
    # 5. Run all ModelDiff methods
    # -------------------------------------------------------------------------
    print("\n[5/5] Running all ModelDiff comparisons...")
    
    # 5a. Print summary
    print("\n--- Model Summary ---")
    model_diff.print_summary()
    
    # 5b. Find feature changes
    print("\n--- Feature Changes ---")
    feature_changes = model_diff.find_feature_changes()
    print(f"New in Model B: {feature_changes['new_in_b']}")
    print(f"Dropped in Model B: {feature_changes['dropped_in_b']}")
    print(f"In both models: {len(feature_changes['in_both'])} features")
    
    # 5c. Compare cumulative gain
    print("\n--- Cumulative Gain Comparison ---")
    model_diff.compare_cumulative_gain()
    
    # 5d. Plot all importance scatters (gain, weight, cover)
    print("\n--- Feature Importance Scatters ---")
    model_diff.plot_all_importance_scatters()
    
    # 5e. Compare PDP for select common features
    print("\n--- Partial Dependence Plot Comparisons ---")
    
    # Initialize ModelAnalyzers for PDP comparison
    model_analyzer_a = ModelAnalyzer(tree_analyzer_a, target_class=1)
    model_analyzer_a.load_data_from_parquets(DATA_DIR, num_files_to_read=1)
    model_analyzer_a.load_xgb_model(MODEL_A_PATH)
    
    model_analyzer_b = ModelAnalyzer(tree_analyzer_b, target_class=1)
    model_analyzer_b.load_data_from_parquets(DATA_DIR, num_files_to_read=1)
    model_analyzer_b.load_xgb_model(MODEL_B_PATH)
    
    # Select a subset of common features for PDP comparison
    # (comparing all 29 would be slow)
    pdp_features = [
        # Strong signals (in both)
        'norm_iid_pos_strong',
        'norm_iid_neg_strong',
        'bin_neg_strong',
        # Medium signals
        'norm_corr_1_pos',
        'unif_linear_pos',
        # Categorical
        'cat_15_strong',
        # Noise (should be flat)
        'noise_norm',
    ]
    
    for feature in pdp_features:
        if feature in feature_changes['in_both']:
            print(f"  Comparing PDP for '{feature}'...")
            try:
                model_diff.compare_pdp(
                    analyzer_a=model_analyzer_a,
                    analyzer_b=model_analyzer_b,
                    feature_name=feature,
                    n_curves=1000,
                    mode="raw",
                )
            except Exception as e:
                print(f"    Warning: Could not compare PDP for {feature}: {e}")
        else:
            print(f"  Skipping '{feature}' (not in both models)")
    
    # 5f. Compare predictions
    print("\n--- Prediction Comparison ---")
    pred_stats = model_diff.compare_predictions(
        analyzer_a=model_analyzer_a,
        analyzer_b=model_analyzer_b,
        y_true=df['target'].values,
        n_samples=None,
    )
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*65)
    print("Model Diffing Example Complete")
    print("="*65)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - cumulative_gain_comparison.png")
    print("  - importance_scatter_gain.png")
    print("  - importance_scatter_weight.png")
    print("  - importance_scatter_cover.png")
    print("  - pdp_comparison_*.png (for each compared feature)")
    print("  - prediction_scatter.png")
    print("  - prediction_diff_histogram.png")
    print("  - prediction_correlation_summary.txt")
    print("  - prediction_agreement_matrix.png")
    print("  - score_qq_plot.png")
    print("\nKey observations to look for:")
    print("  1. Model B should have higher cumulative gain (has bin_pos_strong)")
    print("  2. In importance scatters, features above diagonal increased in Model B")
    print("  3. Strong-signal features should have similar PDP shapes in both models")
    print("  4. Noise features should show flat PDP curves in both models")
    print("  5. Prediction scatter shows how correlated the two models are")
    print("  6. Histogram shows the distribution of score differences")
    print("  7. Q-Q plot shows how score distributions compare across quantiles")


if __name__ == "__main__":
    main()
