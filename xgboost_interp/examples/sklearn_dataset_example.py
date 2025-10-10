"""
Complete example using California Housing dataset from sklearn.

This example demonstrates:
1. Loading a real dataset
2. Training an XGBoost model
3. Saving the model as JSON
4. Using our interpretability package to analyze it
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys

# Add the package to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost_interp import TreeAnalyzer, ModelAnalyzer


def load_and_prepare_data():
    """Load and prepare the California housing dataset."""
    print("Loading California Housing dataset...")
    
    # Load the dataset
    housing = fetch_california_housing()
    
    # Create DataFrame for easier handling
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['target'] = housing.target
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(housing.feature_names)}")
    print(f"Target: Median house value in hundreds of thousands of dollars")
    print("\nDataset info:")
    print(df.describe())
    
    return df, housing.feature_names, housing.target


def train_xgboost_model(df, feature_names, target, model_path="california_housing_xgb.json"):
    """Train an XGBoost regression model."""
    print("\nTraining XGBoost model...")
    
    # Prepare features and target
    X = df[feature_names]
    y = target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Create and train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,           # 100 trees as requested
        max_depth=6,                # Reasonable depth for interpretability
        learning_rate=0.1,          # Standard learning rate
        subsample=0.8,              # Some randomness for robustness
        colsample_bytree=0.8,       # Feature sampling
        random_state=42,            # Reproducibility
        objective='reg:squarederror' # Regression objective
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nModel Performance:")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Save model as JSON
    model.save_model(model_path)
    print(f"\n✅ Model saved as: {model_path}")
    
    return model, X_train, X_test, y_train, y_test


def analyze_with_interpretability_package(model_path, data_df, feature_names):
    """Use our interpretability package to analyze the trained model."""
    print(f"\n{'='*60}")
    print("ANALYZING MODEL WITH INTERPRETABILITY PACKAGE")
    print(f"{'='*60}")
    
    # Initialize tree analyzer
    print("Initializing TreeAnalyzer...")
    tree_analyzer = TreeAnalyzer(model_path)
    tree_analyzer.print_model_summary()
    
    # Tree-level analysis
    print("\nGenerating tree-level analysis plots...")
    
    # Feature importance plots
    tree_analyzer.plot_feature_importance_combined(top_n=None)
    tree_analyzer.plot_feature_importance_distributions(log_scale=True, top_n=None)
    
    # Tree structure analysis
    tree_analyzer.plot_tree_depth_histogram()
    tree_analyzer.plot_cumulative_gain()
    tree_analyzer.plot_cumulative_prediction_shift()
    
    # Advanced plotting
    from xgboost_interp.plotting import FeaturePlotter, TreePlotter
    
    # Feature analysis
    feature_plotter = FeaturePlotter(tree_analyzer.plotter.save_dir)
    feature_plotter.plot_feature_usage_heatmap(
        tree_analyzer.trees, tree_analyzer.feature_names, log_scale=True
    )
    feature_plotter.plot_split_depth_per_feature(
        tree_analyzer.trees, tree_analyzer.feature_names
    )
    feature_plotter.plot_feature_split_impact(
        tree_analyzer.trees, tree_analyzer.feature_names, log_scale=False
    )
    
    # Tree structure plots
    tree_plotter = TreePlotter(tree_analyzer.plotter.save_dir)
    tree_plotter.plot_prediction_and_gain_stats(tree_analyzer.trees, log_scale=False)
    tree_plotter.plot_gain_heatmap(tree_analyzer.trees, tree_analyzer.feature_names)
    
    # Model analysis with data
    print("\nInitializing ModelAnalyzer for data-dependent analysis...")
    model_analyzer = ModelAnalyzer(tree_analyzer)
    
    # Save data as parquet for the model analyzer
    data_dir = "housing_data"
    os.makedirs(data_dir, exist_ok=True)
    data_df.to_parquet(f"{data_dir}/housing_data.parquet", index=False)
    
    # Load data and model
    model_analyzer.load_data_from_parquets(data_dir, num_files_to_read=1)
    model_analyzer.load_xgb_model(model_path)
    
    # Generate partial dependence plots for all features
    print("\nGenerating partial dependence plots...")
    for feature in feature_names:
        try:
            model_analyzer.plot_partial_dependence(
                feature_name=feature,
                grid_points=50,
                n_curves=1000
            )
            print(f"✅ Generated PDP for {feature}")
        except Exception as e:
            print(f"⚠️ Could not generate PDP for {feature}: {e}")
    
    # Generate marginal impact analysis for key features
    print("\nGenerating marginal impact analysis...")
    key_features = feature_names[:4]  # First 4 features
    for feature in key_features:
        try:
            model_analyzer.plot_marginal_impact_univariate(feature, scale="linear")
            print(f"✅ Generated marginal impact for {feature}")
        except Exception as e:
            print(f"⚠️ Could not generate marginal impact for {feature}: {e}")
    
    # Interactive tree visualization (first few trees)
    print("\nGenerating interactive tree visualization...")
    try:
        from xgboost_interp.plotting import InteractivePlotter
        interactive_plotter = InteractivePlotter(tree_analyzer.plotter.save_dir)
        interactive_plotter.plot_interactive_trees(
            tree_analyzer.trees, tree_analyzer.feature_names, 
            top_k=3, combined=False
        )
        print("✅ Generated interactive tree plots")
    except ImportError:
        print("⚠️ Plotly not available for interactive plots")
    except Exception as e:
        print(f"⚠️ Could not generate interactive plots: {e}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"All plots saved to: {tree_analyzer.plotter.save_dir}")
    
    # List generated files
    if os.path.exists(tree_analyzer.plotter.save_dir):
        files = os.listdir(tree_analyzer.plotter.save_dir)
        png_files = [f for f in files if f.endswith('.png')]
        print(f"\nGenerated {len(png_files)} visualization files:")
        for f in sorted(png_files):
            print(f"  - {f}")


def main():
    """Main function to run the complete example."""
    print("XGBoost Interpretability Package - California Housing Example")
    print("=" * 65)
    
    # Load and prepare data
    df, feature_names, target = load_and_prepare_data()
    
    # Train XGBoost model
    model_path = "california_housing_xgb.json"
    model, X_train, X_test, y_train, y_test = train_xgboost_model(
        df, feature_names, target, model_path
    )
    
    # Analyze with our interpretability package
    analyze_with_interpretability_package(model_path, df, feature_names)
    
    print("\n🎉 Example completed successfully!")
    print("\nWhat was demonstrated:")
    print("1. ✅ Loaded California Housing dataset from sklearn")
    print("2. ✅ Trained XGBoost regression model with 100 trees")
    print("3. ✅ Saved model as JSON for interpretability analysis")
    print("4. ✅ Used TreeAnalyzer for structure-based analysis")
    print("5. ✅ Used ModelAnalyzer for data-dependent analysis")
    print("6. ✅ Generated comprehensive visualizations")
    
    print(f"\nFiles created:")
    print(f"  - {model_path} (XGBoost model)")
    print(f"  - housing_data/ (dataset for analysis)")
    print(f"  - california_housing_xgb/ (visualization outputs)")


if __name__ == "__main__":
    main()
