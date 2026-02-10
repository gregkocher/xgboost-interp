"""
Classification example using Iris dataset from sklearn.

This example demonstrates XGBoost interpretability for classification tasks.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

from xgboost_interp import TreeAnalyzer, ModelAnalyzer
from xgboost_interp.utils import AnalysisTracker


def load_and_prepare_iris_data():
    """Load and prepare the Iris dataset."""
    print("Loading Iris dataset...")
    
    # Load the dataset
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = [iris.target_names[i] for i in iris.target]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(iris.feature_names)}")
    print(f"Classes: {list(iris.target_names)}")
    print(f"Class distribution:")
    print(df['species'].value_counts())
    
    return df, iris.feature_names, iris.target


def train_iris_xgboost_model(df, feature_names, target, model_path="examples/iris/iris_xgb.json"):
    """Train an XGBoost classification model on Iris."""
    print("\nTraining XGBoost classifier...")
    
    # Prepare features and target
    X = df[feature_names]
    y = target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Create and train XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=50,            # Fewer trees for small dataset
        max_depth=4,                # Shallow trees for interpretability
        learning_rate=0.1,          
        subsample=0.8,              
        colsample_bytree=0.8,       
        random_state=42,            
        objective='multi:softprob'  # Multi-class classification
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\nModel Performance:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_test, 
                              target_names=['setosa', 'versicolor', 'virginica']))
    
    # Save model as JSON
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"\nModel saved as: {model_path}")
    
    return model, X_train, X_test, y_train, y_test


def analyze_iris_model(model_path, data_df, feature_names, y_test=None, y_pred_proba=None):
    """Analyze the Iris model with our interpretability package."""
    tracker = AnalysisTracker()
    
    print(f"\n{'='*50}")
    print("ANALYZING IRIS MODEL")
    print(f"{'='*50}")
    
    # Initialize tree analyzer
    tree_analyzer = TreeAnalyzer(model_path, save_dir="examples/iris/output")
    tree_analyzer.print_model_summary()
    
    # Tree-level analysis
    print("\nGenerating interpretability plots...")
    
    # Feature importance (most relevant for small dataset)
    tree_analyzer.plot_feature_importance_combined(top_n=None)
    tree_analyzer.plot_feature_importance_distributions(log_scale=False, top_n=None)
    tree_analyzer.plot_feature_importance_scatter(top_n=None)
    
    # Tree structure
    tree_analyzer.plot_tree_depth_histogram()
    tree_analyzer.plot_cumulative_gain()
    
    # Interactive tree visualization (first few trees)
    print("\nGenerating interactive tree visualization...")
    try:
        from xgboost_interp.plotting import InteractivePlotter
        interactive_plotter = InteractivePlotter(tree_analyzer.plotter.save_dir)
        interactive_plotter.plot_interactive_trees(
            tree_analyzer.trees, tree_analyzer.feature_names, 
            top_k=5, combined=False
        )
        print("Generated interactive tree plots (opened in browser)")
        tracker.success("Interactive tree plots")
    except ImportError:
        print("Plotly not available for interactive plots")
        tracker.failure("Interactive tree plots", "Plotly not installed")
    except Exception as e:
        print(f"Could not generate interactive plots: {e}")
        tracker.failure("Interactive tree plots", e)
    
    # Save data for analysis
    data_dir = "examples/iris/iris_data"
    os.makedirs(data_dir, exist_ok=True)
    data_df.to_parquet(f"{data_dir}/iris_data.parquet", index=False)
    
    # Model performance metrics (before per-class analysis)
    if y_test is not None and y_pred_proba is not None:
        print("\nComputing model performance metrics...")
        model_analyzer = ModelAnalyzer(tree_analyzer, target_class=0)
        model_analyzer.load_data_from_parquets(data_dir, num_files_to_read=1)
        model_analyzer.load_xgb_model(model_path)
        metrics = model_analyzer.evaluate_model_performance(y_test, y_pred_proba)
        print("Model Performance Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {round(v, 6)}")
        print(f" Saved to: examples/iris/output/model_performance_metrics.txt")
    
    # Analyze each class separately
    class_names = ['setosa', 'versicolor', 'virginica']
    
    for target_class in range(3):  # 3 classes in Iris dataset
        print(f"\n{'='*50}")
        print(f"ANALYZING CLASS {target_class}: {class_names[target_class].upper()}")
        print(f"{'='*50}")
        
        # Model analysis with data - specify target class
        model_analyzer = ModelAnalyzer(tree_analyzer, target_class=target_class)
        
        # Load data and model
        model_analyzer.load_data_from_parquets(data_dir, num_files_to_read=1)
        model_analyzer.load_xgb_model(model_path)
        
        # Partial dependence plots for all features
        print(f"\nGenerating partial dependence plots for class {target_class} ({class_names[target_class]})...")
        for feature in feature_names:
            try:
                model_analyzer.plot_partial_dependence(
                    feature_name=feature,
                    n_curves=150  # All samples
                )
                print(f"Generated PDP for {feature}")
                tracker.success(f"PDP class {target_class}: {feature}")
            except Exception as e:
                print(f"Could not generate PDP for {feature}: {e}")
                tracker.failure(f"PDP class {target_class}: {feature}", e)
        
        # Marginal impact for all features (small dataset)
        print(f"\nGenerating marginal impact analysis for class {target_class} ({class_names[target_class]})...")
        for feature in feature_names:
            try:
                model_analyzer.plot_marginal_impact_univariate(feature, scale="linear")
                print(f"Generated marginal impact for {feature}")
                tracker.success(f"Marginal impact class {target_class}: {feature}")
            except Exception as e:
                print(f"Could not generate marginal impact for {feature}: {e}")
                tracker.failure(f"Marginal impact class {target_class}: {feature}", e)
        
        # Prediction evolution across trees
        print(f"\nGenerating prediction evolution plot for class {target_class} ({class_names[target_class]})...")
        try:
            model_analyzer.plot_scores_across_trees(n_records=1000)
            print(f"Generated scores across trees plot")
            tracker.success(f"Scores across trees class {target_class}")
        except Exception as e:
            print(f"Could not generate scores across trees: {e}")
            tracker.failure(f"Scores across trees class {target_class}", e)
        
        # Early exit performance analysis
        print(f"\nGenerating early exit performance analysis for class {target_class} ({class_names[target_class]})...")
        try:
            model_analyzer.analyze_early_exit_performance(n_records=5000, n_detailed_curves=1000)
            tracker.success(f"Early exit analysis class {target_class}")
        except Exception as e:
            print(f"Could not generate early exit analysis: {e}")
            tracker.failure(f"Early exit analysis class {target_class}", e)
        
        # ALE Plots
        print(f"\n[BONUS] Generating ALE plots for class {target_class} ({class_names[target_class]})...")
        try:
            from PyALE import ale
            for i, feature in enumerate(feature_names, 1):
                print(f"  [{i}/{len(feature_names)}] Computing ALE for '{feature}'...")
                model_analyzer.plot_ale(
                    feature_name=feature,
                    grid_size=100,
                    include_CI=True,
                    n_curves=min(10000, len(model_analyzer.df))
                )
            print(f"  Generated {len(feature_names)} ALE plots in ALE_analysis/")
            tracker.success(f"ALE plots class {target_class}")
        except ImportError:
            print("  PyALE not installed - skipping ALE plots")
            tracker.failure(f"ALE plots class {target_class}", "PyALE not installed")
        except Exception as e:
            print(f"  Failed to generate ALE plots: {e}")
            tracker.failure(f"ALE plots class {target_class}", e)
        
        # SHAP Analysis
        print(f"\nGenerating SHAP analysis for class {target_class} ({class_names[target_class]})...")
        try:
            import shap
            import matplotlib.pyplot as plt
            
            X_sample = model_analyzer.df[feature_names].sample(n=min(1000, len(model_analyzer.df)), random_state=42)
            
            explainer = shap.TreeExplainer(model_analyzer.xgb_model)
            shap_values = explainer(X_sample)
            
            # For multi-class models, extract SHAP values for the target class
            if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
                # Multi-class: shape is (n_samples, n_features, n_classes)
                class_shap_values = shap_values[:, :, target_class]
            else:
                # Binary or single output
                class_shap_values = shap_values
            
            shap_dir = os.path.join(tree_analyzer.plotter.save_dir, f'SHAP_analysis_class_{target_class}')
            shap_dep_dir = os.path.join(shap_dir, 'SHAP_dependence_plots')
            os.makedirs(shap_dir, exist_ok=True)
            os.makedirs(shap_dep_dir, exist_ok=True)
            
            # Summary plots
            plt.figure()
            shap.summary_plot(class_shap_values, X_sample, plot_type="bar", show=False)
            plt.savefig(os.path.join(shap_dir, 'summary_bar.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure()
            shap.summary_plot(class_shap_values, X_sample, show=False)
            plt.savefig(os.path.join(shap_dir, 'summary_beeswarm.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Dependence plots for all features
            for i, feature in enumerate(feature_names):
                plt.figure()
                shap.dependence_plot(i, class_shap_values.values, X_sample, show=False)
                plt.savefig(os.path.join(shap_dep_dir, f'dependence_{feature}.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            # Waterfall plots
            for idx in range(min(5, len(X_sample))):
                plt.figure()
                shap.waterfall_plot(class_shap_values[idx], show=False)
                plt.savefig(os.path.join(shap_dir, f'waterfall_sample_{idx}.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"  Generated SHAP analysis in SHAP_analysis_class_{target_class}/")
            tracker.success(f"SHAP analysis class {target_class}")
        except ImportError:
            print("  shap not installed - skipping SHAP analysis")
            tracker.failure(f"SHAP analysis class {target_class}", "shap not installed")
        except Exception as e:
            print(f"  Failed to generate SHAP analysis: {e}")
            tracker.failure(f"SHAP analysis class {target_class}", e)
    
    tracker.print_summary()
    print(f"\nAnalysis complete! Plots saved to: {tree_analyzer.plotter.save_dir}")


def compare_feature_importance():
    """Compare our feature importance with XGBoost's built-in importance."""
    print(f"\n{'='*50}")
    print("FEATURE IMPORTANCE COMPARISON")
    print(f"{'='*50}")
    
    # Load the trained model for comparison
    model = xgb.XGBClassifier()
    model.load_model("examples/iris/iris_xgb.json")
    
    print("XGBoost built-in feature importance:")
    importance = model.feature_importances_
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    
    for name, imp in zip(feature_names, importance):
        print(f"  {name}: {imp:.4f}")
    


def main():
    """Main function for Iris classification example."""
    print("XGBoost Interpretability Package - Iris Classification Example")
    print("=" * 65)
    
    # Load and prepare data
    df, feature_names, target = load_and_prepare_iris_data()
    
    # Train XGBoost model
    model_path = "examples/iris/iris_xgb.json"
    model, X_train, X_test, y_train, y_test = train_iris_xgboost_model(
        df, feature_names, target, model_path
    )
    
    # Get predictions for metrics (probability matrix for multi-class)
    y_pred_proba = model.predict_proba(X_test)
    
    # Analyze with our interpretability package
    analyze_iris_model(model_path, df, feature_names, y_test, y_pred_proba)
    
    # Compare with built-in importance
    compare_feature_importance()
    
    print("\nIris classification example completed!")


if __name__ == "__main__":
    main()
