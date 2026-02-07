"""
Basic usage example for XGBoost interpretability package.

This example shows how to perform basic tree-level analysis without requiring data.
"""

from xgboost_interp import TreeAnalyzer, ModelAnalyzer
from xgboost_interp.utils import AnalysisTracker


def basic_tree_analysis_example():
    """Example of basic tree-level analysis."""
    
    # Replace with your model path
    model_path = "your_model.json"
    
    # Initialize tree analyzer
    tree_analyzer = TreeAnalyzer(model_path)
    
    # Print model summary
    tree_analyzer.print_model_summary()
    
    # Generate various plots
    print("Generating feature importance plots...")
    tree_analyzer.plot_feature_importance_combined()
    tree_analyzer.plot_feature_importance_distributions(log_scale=True)
    
    print("Generating tree structure plots...")
    tree_analyzer.plot_tree_depth_histogram()
    tree_analyzer.plot_cumulative_gain()
    tree_analyzer.plot_cumulative_prediction_shift()
    
    print("Basic analysis complete! Check the output directory for plots.")


def model_with_data_example():
    """Example of model analysis with data."""
    
    # Replace with your model path
    model_path = "your_model.json"
    
    # Initialize analyzers
    tree_analyzer = TreeAnalyzer(model_path)
    model_analyzer = ModelAnalyzer(tree_analyzer)
    
    # Load data
    print("Loading data...")
    model_analyzer.load_data_from_parquets(
        data_dir_path="sample_data",
        num_files_to_read=2
    )
    
    # Load XGBoost model for predictions
    model_analyzer.load_xgb_model()
    
    # Generate data-dependent plots
    print("Generating partial dependence plots...")
    
    # Example features - replace with your actual feature names
    example_features = [
        "Item_NCalculatedTotalCost",
        "CPA_Prediction_Conversion_Logistic_Regression",
        "QueryItem_BM25F_KSink_Tfc_BNorm_v2"
    ]
    
    for feature in example_features:
        try:
            model_analyzer.plot_partial_dependence(
                feature_name=feature, 
                n_curves=1000
            )
            print(f"Generated PDP for {feature}")
            tracker.success(f"PDP: {feature}")
        except Exception as e:
            print(f"Could not generate PDP for {feature}: {e}")
            tracker.failure(f"PDP: {feature}", e)
    
    # Plot prediction evolution across trees
    print("Analyzing prediction evolution...")
    model_analyzer.plot_scores_across_trees(
        tree_indices=[50, 500, 1000, 2000, 3000], 
        n_records=2000
    )
    
    print("Model analysis with data complete!")


if __name__ == "__main__":
    print("XGBoost Interpretability Package - Basic Examples")
    print("=" * 50)
    
    tracker = AnalysisTracker()
    
    # Run basic tree analysis
    try:
        basic_tree_analysis_example()
        tracker.success("Basic tree analysis")
    except Exception as e:
        print(f"Basic analysis failed: {e}")
        tracker.failure("Basic tree analysis", e)
    
    # Run model analysis with data
    try:
        model_with_data_example()
        tracker.success("Model analysis with data")
    except Exception as e:
        print(f"Model analysis with data failed: {e}")
        tracker.failure("Model analysis with data", e)
    
    tracker.print_summary()