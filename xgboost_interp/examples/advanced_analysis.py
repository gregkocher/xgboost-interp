"""
Advanced usage example that mirrors the original main execution.

This example shows comprehensive analysis similar to the original script.
"""

from xgboost_interp import TreeAnalyzer, ModelAnalyzer
from xgboost_interp.plotting import FeaturePlotter, TreePlotter, InteractivePlotter


def comprehensive_analysis_example():
    """
    Comprehensive analysis example that mirrors the original main execution.
    """
    
    # Model path - replace with your actual model
    model_path = "PTR_lambda1_77f_3000trees_top20_nds25pct_20250314_20250522--80pct copy.json"
    
    print("Initializing analyzers...")
    tree_analyzer = TreeAnalyzer(model_path)
    tree_analyzer.print_model_summary()
    
    # Initialize model analyzer for data-dependent analysis
    model_analyzer = ModelAnalyzer(tree_analyzer)
    
    # Load data if available
    try:
        print("Loading data...")
        model_analyzer.load_data_from_parquets(
            data_dir_path="sample_data",
            num_files_to_read=2
        )
        model_analyzer.load_xgb_model()
        data_available = True
    except Exception as e:
        print(f"⚠️ Could not load data: {e}")
        print("Proceeding with tree-only analysis...")
        data_available = False
    
    # Tree-level analysis (always available)
    print("\n" + "="*50)
    print("TREE-LEVEL ANALYSIS")
    print("="*50)
    
    print("Generating feature importance plots...")
    tree_analyzer.plot_feature_importance_combined(top_n=None)
    tree_analyzer.plot_feature_importance_distributions(log_scale=True, top_n=None)
    
    print("Generating tree structure analysis...")
    tree_analyzer.plot_tree_depth_histogram()
    tree_analyzer.plot_cumulative_gain()
    tree_analyzer.plot_cumulative_prediction_shift()
    
    # Advanced plotting using specialized plotters
    print("Generating advanced tree plots...")
    tree_plotter = TreePlotter(tree_analyzer.plotter.save_dir)
    tree_plotter.plot_prediction_and_gain_stats(
        tree_analyzer.trees, log_scale=False, top_k=None
    )
    tree_plotter.plot_gain_heatmap(tree_analyzer.trees, tree_analyzer.feature_names)
    
    print("Generating feature analysis plots...")
    feature_plotter = FeaturePlotter(tree_analyzer.plotter.save_dir)
    
    # Feature co-occurrence analysis
    from itertools import combinations
    import numpy as np
    
    # Tree-level co-occurrence
    trees = tree_analyzer.trees
    num_features = len(tree_analyzer.feature_names)
    co_matrix = np.zeros((num_features, num_features), dtype=int)
    
    for tree in trees:
        split_indices = tree.get("split_indices", [])
        features_in_tree = set(tree_analyzer.feature_names[i] for i in split_indices 
                              if i < len(tree_analyzer.feature_names))
        for f1, f2 in combinations(features_in_tree, 2):
            i, j = tree_analyzer.feature_names.index(f1), tree_analyzer.feature_names.index(f2)
            co_matrix[i][j] += 1
            co_matrix[j][i] += 1
        for f in features_in_tree:
            i = tree_analyzer.feature_names.index(f)
            co_matrix[i][i] += 1
    
    feature_plotter.plot_feature_cooccurrence_heatmap(
        co_matrix, tree_analyzer.feature_names,
        "Tree-Level Feature Co-occurrence", "feature_cooccurrence_tree_level.png"
    )
    
    # Feature usage and depth analysis
    feature_plotter.plot_feature_usage_heatmap(
        trees, tree_analyzer.feature_names, top_k=None, log_scale=True
    )
    feature_plotter.plot_split_depth_per_feature(
        trees, tree_analyzer.feature_names, top_n=None
    )
    feature_plotter.plot_feature_split_impact(
        trees, tree_analyzer.feature_names, log_scale=False, top_n=None
    )
    
    # Interactive tree visualization (if plotly available)
    try:
        print("Generating interactive tree visualization...")
        interactive_plotter = InteractivePlotter(tree_analyzer.plotter.save_dir)
        interactive_plotter.plot_interactive_trees(
            trees, tree_analyzer.feature_names, top_k=3, combined=False
        )
    except ImportError:
        print("⚠️ Plotly not available for interactive plots")
    
    # Data-dependent analysis (if data is available)
    if data_available:
        print("\n" + "="*50)
        print("MODEL ANALYSIS WITH DATA")
        print("="*50)
        
        # Prediction evolution analysis
        print("Analyzing prediction evolution across trees...")
        model_analyzer.plot_scores_across_trees(
            tree_indices=[50, 500, 1000, 2000, 3000], 
            n_records=2000
        )
        
        # Example features for detailed analysis
        example_features = [
            "Item_NCalculatedTotalCost",
            "CPA_Prediction_Conversion_Logistic_Regression", 
            "QueryItem_Text_Embedding_SimEmbA_SquaredEuclideanDist_v2",
            "Buyer_CPA_UserID_Category_Encoding_v1",
            "Item_FastIMA_SalesOverView_7DayDecay_LogSmooth_Domestic_WebAndMobile_V2",
            "Item_PromotedListingAdRate",
            "QueryItem_Title_BigramMatchCount_Unique_NoQueryExp",
            "ItemSeller_FixedPrice_NQty",
            "QueryItem_BM25F_KSink_Tfc_BNorm_v2"
        ]
        
        # Partial dependence plots for key features
        print("Generating partial dependence plots...")
        for feature in example_features:
            if feature in tree_analyzer.feature_names:
                try:
                    model_analyzer.plot_partial_dependence(
                        feature_name=feature, 
                        grid_points=100, 
                        n_curves=1000
                    )
                    print(f"✅ Generated PDP for {feature}")
                except Exception as e:
                    print(f"⚠️ Could not generate PDP for {feature}: {e}")
        
        # Marginal impact analysis for key features
        print("Generating marginal impact analysis...")
        marginal_features = [
            "Item_NCalculatedTotalCost",
            "CPA_Prediction_Conversion_Logistic_Regression",
            "Item_PromotedListingAdRate",
            "QueryItem_BM25F_KSink_Tfc_BNorm_v2"
        ]
        
        for feature in marginal_features:
            if feature in tree_analyzer.feature_names:
                try:
                    model_analyzer.plot_marginal_impact_univariate(feature, scale="linear")
                    print(f"✅ Generated marginal impact for {feature}")
                except Exception as e:
                    print(f"⚠️ Could not generate marginal impact for {feature}: {e}")
        
        # ALE plots (if pyALE is available)
        try:
            print("Generating ALE plots...")
            for feature in example_features[:3]:  # Just first 3 to avoid too many plots
                if feature in tree_analyzer.feature_names:
                    model_analyzer.plot_ale(
                        feature_name=feature,
                        grid_size=50,
                        include_CI=True,
                        n_curves=5000
                    )
                    print(f"✅ Generated ALE plot for {feature}")
        except ImportError:
            print("⚠️ pyALE not available for ALE plots")
        except Exception as e:
            print(f"⚠️ Could not generate ALE plots: {e}")
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)
    print(f"All plots saved to: {tree_analyzer.plotter.save_dir}")
    
    # Summary of generated files
    import os
    if os.path.exists(tree_analyzer.plotter.save_dir):
        files = os.listdir(tree_analyzer.plotter.save_dir)
        png_files = [f for f in files if f.endswith('.png')]
        print(f"Generated {len(png_files)} visualization files:")
        for f in sorted(png_files):
            print(f"  - {f}")


if __name__ == "__main__":
    comprehensive_analysis_example()
