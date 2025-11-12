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
import sys

# Add the package to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost_interp import TreeAnalyzer, ModelAnalyzer


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


def train_iris_xgboost_model(df, feature_names, target, model_path="iris_xgb.json"):
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
    model.save_model(model_path)
    print(f"\n✅ Model saved as: {model_path}")
    
    return model, X_train, X_test, y_train, y_test


def analyze_iris_model(model_path, data_df, feature_names):
    """Analyze the Iris model with our interpretability package."""
    print(f"\n{'='*50}")
    print("ANALYZING IRIS MODEL")
    print(f"{'='*50}")
    
    # Initialize tree analyzer
    tree_analyzer = TreeAnalyzer(model_path)
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
        print("✅ Generated interactive tree plots (opened in browser)")
    except ImportError:
        print("⚠️ Plotly not available for interactive plots")
    except Exception as e:
        print(f"⚠️ Could not generate interactive plots: {e}")
    
    # Save data for analysis
    data_dir = "iris_data"
    os.makedirs(data_dir, exist_ok=True)
    data_df.to_parquet(f"{data_dir}/iris_data.parquet", index=False)
    
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
                print(f"✅ Generated PDP for {feature}")
            except Exception as e:
                print(f"⚠️ Could not generate PDP for {feature}: {e}")
        
        # Marginal impact for all features (small dataset)
        print(f"\nGenerating marginal impact analysis for class {target_class} ({class_names[target_class]})...")
        for feature in feature_names:
            try:
                model_analyzer.plot_marginal_impact_univariate(feature, scale="linear")
                print(f"✅ Generated marginal impact for {feature}")
            except Exception as e:
                print(f"⚠️ Could not generate marginal impact for {feature}: {e}")
        
        # Prediction evolution across trees
        print(f"\nGenerating prediction evolution plot for class {target_class} ({class_names[target_class]})...")
        try:
            # For iris: 150 total trees = 50 rounds x 3 classes
            model_analyzer.plot_scores_across_trees(
                tree_indices=[30, 60, 90, 120, 150],
                n_records=150
            )
            print(f"✅ Generated scores across trees plot")
        except Exception as e:
            print(f"⚠️ Could not generate scores across trees: {e}")
    
    print(f"\nAnalysis complete! Plots saved to: {tree_analyzer.plotter.save_dir}")


def compare_feature_importance():
    """Compare our feature importance with XGBoost's built-in importance."""
    print(f"\n{'='*50}")
    print("FEATURE IMPORTANCE COMPARISON")
    print(f"{'='*50}")
    
    # Load the trained model for comparison
    model = xgb.XGBClassifier()
    model.load_model("iris_xgb.json")
    
    print("XGBoost built-in feature importance:")
    importance = model.feature_importances_
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    
    for name, imp in zip(feature_names, importance):
        print(f"  {name}: {imp:.4f}")
    
    print("\nOur package provides more detailed analysis:")
    print("  - Weight: How often each feature is used for splits")
    print("  - Gain: Total improvement in loss from splits on each feature") 
    print("  - Cover: Total number of samples affected by splits on each feature")
    print("  - Distributions: Variability of importance across trees")
    print("  - Partial Dependence: How predictions change with feature values")
    print("  - Marginal Impact: Feature-specific prediction changes")


def main():
    """Main function for Iris classification example."""
    print("XGBoost Interpretability Package - Iris Classification Example")
    print("=" * 65)
    
    # Load and prepare data
    df, feature_names, target = load_and_prepare_iris_data()
    
    # Train XGBoost model
    model_path = "iris_xgb.json"
    model, X_train, X_test, y_train, y_test = train_iris_xgboost_model(
        df, feature_names, target, model_path
    )
    
    # Analyze with our interpretability package
    analyze_iris_model(model_path, df, feature_names)
    
    # Compare with built-in importance
    compare_feature_importance()
    
    print("\n🎉 Iris classification example completed!")
    print("\nKey insights for Iris dataset:")
    print("1. ✅ Perfect classification accuracy (simple dataset)")
    print("2. ✅ Clear feature importance patterns")
    print("3. ✅ Interpretable decision boundaries")
    print("4. ✅ Demonstrates multi-class classification analysis")


if __name__ == "__main__":
    main()
